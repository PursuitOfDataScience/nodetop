"""PBS Pro, OpenPBS and Torque.

PBS separates *accepting* a job from *starting* one -- ``enabled`` and
``started`` are two independent booleans per queue -- so a queue can be
``enabled = True, started = False`` and swallow submissions that will never
run.  That is the same class of trap as a Slurm partition left in ``State=DOWN``
with its nodes still advertised, and the neutral :class:`~nodetop.core.model.Queue`
models both switches for exactly this reason.

**PBS has no dry-run.**  ``qsub`` either submits or refuses; there is no
verify-only mode, so this backend reports ``probe=False`` and nodetop must
present entitlement as *declared, unconfirmed*.  Saying nothing would let a
declared ACL read as a verified one.

JSON output (``-F json``, PBS Pro 18+) is preferred and the classic text format
is parsed as a fallback, because Torque and older PBS have no JSON at all.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from ..core.duration import parse_timestamp
from ..core.hardware import identify_accelerator, name_accelerator
from ..core.model import Identity, Job, JobShape, Limits, Node, Queue, Verdict
from ..runner import Runner, SubprocessRunner, resolve, which
from .base import BackendCapabilities, count

__all__ = ["PbsBackend"]

#: Ceiling for the two whole-cluster enumerations, in place of the 30s
#: default. Measured on a 10,624-node PBS Pro cluster: `pbsnodes -a -F json`
#: returns 14 MB and took 9s on an idle server and 26s on a loaded one -- a 3x
#: spread that puts the default within four seconds of expiring, and a timeout
#: there is not a small delay but the whole node list gone, reported as "query
#: failed: nodes" on a healthy cluster. Only these two queries are raised:
#: everywhere else a slow answer is still evidence, which is what the default
#: is for.
_ENUMERATE_TIMEOUT = 90.0

#: States meaning "a job holds the whole machine", where PBS records the
#: exclusivity in the state and *not* in every resource.
#:
#: Measured on a 10,624-node PBS Pro 2022.1 cluster: 10,194 nodes
#: `job-exclusive`, and not one node anywhere reported `ngpus` under
#: `resources_assigned` -- because whole-node placement means the scheduler
#: never has to account for a GPU individually. `ncpus` happened to be
#: assigned in full, so the CPU figures were right and the accelerator figures
#: were not: nodetop announced **62,886 of 63,744 GPUs free** on a machine
#: whose true free count was 1,722. A 36x overstatement, on the one axis people
#: pick that cluster for, from the tool written to find phantom capacity.
#:
#: Modelled as occupancy rather than as a condition on purpose. These nodes are
#: healthy and working; they are not drained, and calling them unschedulable
#: would report 96% of that cluster as out of service and hide it from `health`
#: -- swapping one wrong answer for a louder one. Full is not broken.
_WHOLLY_ALLOCATED = frozenset({"job-exclusive", "resv-exclusive"})

#: PBS node states that mean "will not take new work".
_STATE_TO_CONDITION = {
    "down": "DOWN",
    "offline": "DRAIN",
    "state-unknown": "UNKNOWN",
    "stale": "UNKNOWN",
    "unresolvable": "DOWN",
    "maintenance": "MAINT",
    "provisioning": "MAINT",
    "wait-provisioning": "MAINT",
    "initializing": "MAINT",
    "sleep": "POWERSAVE",
    "resv-exclusive": "RESERVED",
}


#: Bytes in one PBS *word* -- the unit behind the ``w`` suffix family (``w``,
#: ``kw``, ``mw``, ``gw``, ``tw``, ``pw``) of PBS's documented size syntax.
#:
#: **A 64-bit PBS server is assumed here**, and stated rather than implied
#: because the word size is a property of the machine PBS runs on, not of
#: nodetop: PBS defines a word as the host's word, and a size string carries no
#: way to ask which host printed it. Every platform current PBS Pro and OpenPBS
#: ship for is 64-bit, so 8 is right in practice; against a 32-bit relic it
#: reads 2x high, which is the one direction this constant can be wrong in and
#: is why it is not simply left out.
_PBS_WORD_BYTES = 8


def _mem_to_mb(text: str | None) -> int:
    """PBS memory strings: ``256gb``, ``1024mb``, ``2tb``, ``16gw``, bare bytes.

    A PBS suffix is a size prefix *and* a unit, and both have to be read.  The
    ``b``/``w`` half used to be matched and thrown away, which is not the
    harmless half: a word is :data:`_PBS_WORD_BYTES` bytes, so ``1gw`` is 8 GiB
    and reading it as ``1gb`` answered 1024 MB -- an **8x understatement**, the
    direction that makes an over-limit job look acceptable and a node look
    smaller than it is rather than raising a false alarm.  It is also not
    confined to queue ceilings: the same function supplies `Node.memory_mb`,
    so a ``mem = 4gw`` node was published with a quarter of its memory.

    Sub-megabyte sizes still truncate to 0 -- ``1024w`` is 8 KiB either way --
    and :func:`_pbs_mem_mb` is what turns that 0 back into "unread".
    """
    if not text:
        return 0
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)([bw]?)\s*$", str(text), re.IGNORECASE)
    if not m:
        return 0
    value = float(m.group(1))
    if m.group(3).lower() == "w":
        value *= _PBS_WORD_BYTES
    # `p` is in PBS's documented suffix list and was missing here, so `1pb`
    # fell out as 0 -- a node with no memory, and a ceiling of nothing.
    scale = {
        "": 1 / (1024 * 1024), "k": 1 / 1024, "m": 1,
        "g": 1024, "t": 1024 * 1024, "p": 1024 * 1024 * 1024,
    }
    return int(value * scale.get(m.group(2).lower(), 1))


def _pbs_walltime(text: str | None) -> int | None:
    """PBS walltime is always ``[[HH:]MM:]SS`` -- never a bare minute count."""
    if not text:
        return None
    t = str(text).strip()
    if not t or t.lower() in {"unlimited", "none"}:
        return None
    parts = t.split(":")
    if not all(p.strip().isdigit() for p in parts):
        return None
    nums = [int(p) for p in parts]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0]


#: One ``[kind:entity=value]`` cell of a PBS per-entity limit table. The kind
#: (``u`` user, ``g`` group, ``p`` project, ``o`` overall) is optional only
#: because reading a malformed cell as generic beats discarding the ceiling.
#:
#: The value is captured as *text*, not as digits. Cells carry units --
#: ``max_run_res.mem = [u:PBS_GENERIC=200gb]`` -- and a digits-only capture
#: does not see such a cell at all, so the whole table fell through to the
#: "this is not a table" branch and the ceiling was discarded. Whether the
#: winning cell is a number is the caller's question, asked *after* the right
#: cell has been picked; a digits-only capture also answers it during the pick,
#: which is how ``[u:alice=unlimited],[u:bob=4]`` came to hand bob's 4 to
#: everybody as the one remaining "lone" cell.
_PBS_LIMIT_ENTRY = re.compile(
    r"\[\s*(?:([A-Za-z])\s*:\s*)?([^=\[\]]*?)\s*=\s*([^=\[\]]*?)\s*\]")

#: A limit value that is a number and nothing else.
_PBS_BARE_INT = re.compile(r"^\s*(\d+)\s*$")


def _pbs_int(cell: str) -> int | None:
    """A resolved limit cell as a count, or ``None`` if it is not one."""
    m = _PBS_BARE_INT.match(cell)
    return int(m.group(1)) if m else None


def _pbs_mem_mb(cell: str) -> int | None:
    """A resolved limit cell as MB, or ``None`` if it is not readable as size.

    :func:`_mem_to_mb` answers 0 both for "no value" and for a shape it cannot
    read -- a unit outside PBS's suffix list (``2eb``), a word or a cell that
    is not a size at all -- and 0 is not a memory ceiling anybody published, so
    0 becomes ``None`` here and the caller records the attribute as unread
    rather than as absent.  (``2pb`` and ``1gw`` used to land here too, from
    the missing peta prefix and the discarded ``w`` unit; both are read now.)
    """
    return _mem_to_mb(cell) or None


def _pbs_limit_cell(value: str | None, user: str | None = None) -> str | None:
    """The one cell of ``max_run_res.ngpus = [u:PBS_GENERIC=8]`` that is ours.

    PBS expresses a limit as a per-entity *table*: ``[u:alice=99]`` for a named
    user, ``[u:PBS_GENERIC=4]`` for every user without an entry of their own,
    and ``g:``/``p:``/``o:`` forms for group, project and the queue-wide total.
    Which cell applies depends on who is asking, so they are tried in that
    order -- the caller's own ``u:`` entry, then the generic one, then (only
    where a single entity is named and there is no way to tell whether it is
    ours) that lone value.

    Reading the *tail* of the string is not a shortcut for this.  An
    end-anchored search answers 99 for ``[u:PBS_GENERIC=4],[u:alice=99]``, i.e.
    hands back a *different* user's ceiling as the caller's, and PBS prints the
    cells in whatever order the administrator set them.  Where the table names
    only other users the answer is ``None``: no per-entity limit applies to us,
    which is not the same thing as inheriting a stranger's.

    A bare value (Torque's ``max_user_run = 4``, and every ``resources_max.*``
    on PBS Pro) is not a table and is returned as itself, for the caller to
    read as a count or a size.
    """
    if value is None:
        return None
    text = str(value)
    entries = _PBS_LIMIT_ENTRY.findall(text)
    if entries:
        if user:
            for kind, entity, cell in entries:
                if kind in ("u", "") and entity == user:
                    return cell
        for kind, entity, cell in entries:
            if kind in ("u", "") and entity == "PBS_GENERIC":
                return cell
        for _kind, entity, cell in entries:
            if entity == "PBS_GENERIC":
                return cell
        if len(entries) == 1:
            return entries[0][2]
        return None
    return text.strip()


#: PBS resource names this model has a ceiling axis for.
#:
#: A resource missing from the map is one nothing here ever checks
#: (``ompthreads``, ``place``, ``mpiprocs``), so its limit is neither read nor
#: reported as unread -- naming those would put a caveat on every queue about
#: checks that were never going to run.
_PBS_LIMIT_RESOURCES = {"ngpus": "gpu", "ncpus": "cpu", "nodect": "node", "mem": "mem_mb"}


def _pbs_limit_declared(value: str | None) -> bool:
    """Whether a limit attribute held a ceiling, as opposed to nothing.

    The PBS side of :func:`nodetop.backends.slurm._looks_present`, and the
    boundary that keeps `Limits.unreadable` honest: an attribute PBS never
    printed, and one printed with an empty value or an explicit "no limit"
    sentinel, are the same thing -- no ceiling, nothing to warn about. PBS
    prints an unset attribute empty rather than omitting it (the recorded
    fixture's ``acl_users =`` is that shape), so treating empty as present
    would make every queue on the cluster warn.
    """
    if value is None or not str(value).strip():
        return False
    return str(value).strip().lower() not in {
        "unlimited", "infinite", "none", "n/a", "(null)", "-",
    }


class PbsBackend:
    """Adapter for PBS Pro / OpenPBS / Torque."""

    name = "pbs"
    queue_term = "queue"

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or SubprocessRunner()
        self._nodes: list[Node] | None = None
        self._queue_text_cache: str | None = None
        self._queue_json_cache: str | None = None
        # `load_queues` needs the nodes too, and the two now run together. One
        # lock for all three caches: they are only ever taken briefly, and a
        # second `pbsnodes -a -F json` is 14 MB on the largest cluster tested.
        self._cache_lock = threading.RLock()

    #: Binaries only a real PBS installation ships.  A Slurm site has none of
    #: them, and any of them settles the question on its own.
    _PBS_ONLY = ("pbs_server", "pbsdsh", "pbs_mom", "qmgr", "pbs_rstat")

    #: What another scheduler's PBS shim writes into its own output.
    #:
    #: The third signal, and the only one that does not depend on how the
    #: wrapping scheduler was packaged.  Slurm's `pbsnodes` contrib writes
    #: `slurmstate=<state>` into every node's status line, which no real PBS
    #: emits -- and `pbsnodes` is already the first command this backend runs, so
    #: the evidence arrives with data that was being fetched anyway.
    _WRAPPER_OUTPUT_MARKERS = (("slurmstate=", "slurm"),)

    @classmethod
    def wrapped_by(cls, runner: Runner | None = None) -> str | None:
        """The OTHER scheduler whose compatibility shims are on PATH, if any.

        Slurm ships ``contribs/torque`` -- ``qstat``, ``qsub``, ``qdel``,
        ``pbsnodes`` -- and a great many sites install them by default, so
        ``pbsnodes && qstat`` is satisfied on clusters with no PBS at all.
        This is not a niche case; it is the common case for the backend most
        likely to be misdetected.

        And the failure was not a visibly empty report, which would have been
        harmless.  ``pbsnodes`` IS the shim and really does enumerate every
        node, so ``--backend pbs`` produced *correct denominators* -- 1614
        nodes, 422 GPUs, matching the slurm backend exactly -- with ``0 up`` on
        a cluster with 589 up, and exit 0.

        Returns the wrapping scheduler's name, or ``None`` when the clients
        look like a genuine PBS.

        **Three signals, because the first two are properties of PACKAGING.**
        The marker-binary check and the install-prefix check both hold on a site
        that installs Slurm under a versioned prefix -- and the most common
        packaging defeats both: the distro `slurm-torque` RPM puts the shims in
        `/usr/bin`, so there is no `slurm-<version>` path component, and it still
        ships none of the five PBS-only binaries. Neither rule fires and the
        wrappers read as a real PBS again, which is the ordinary case on any
        RPM/DEB-packaged Slurm cluster rather than an exotic one.

        So the last signal asks the wrapper what it is. See
        :data:`_WRAPPER_OUTPUT_MARKERS`.
        """
        if any(which(b) for b in cls._PBS_ONLY):
            # A real PBS: `qmgr`/`pbs_server`/`pbsdsh` have no Slurm equivalent
            # and no Slurm package installs them.
            return None
        for binary in ("pbsnodes", "qstat", "qsub"):
            path = resolve(binary)
            if not path:
                continue
            for other in ("slurm", "lsf", "sge", "gridengine"):
                # A path COMPONENT, so a queue directory called `/data/slurm`
                # cannot be mistaken for an install prefix.
                if other in path.lower().split("/"):
                    return other  # pragma: no cover - subsumed below
                # `/software/slurm-23.02-el7-x86_64/bin/qstat` -- the version is
                # in the directory name, which is how these are laid out.
                #
                # A DIGIT after the dash, so the component has to look like a
                # versioned install prefix. `startswith(other + "-")` alone also
                # matched `slurm-logs`, `slurm-data` and anything else a site
                # happens to name that way, so a genuine PBS installed under such
                # a path was reported as somebody else's wrapper and not detected
                # at all. Narrow: it takes a client-only PBS (none of
                # `_PBS_ONLY`) under a `slurm-*` directory to reach this, but the
                # rule should say what it means.
                if any(
                    c.lower() == other
                    or (c.lower().startswith(other + "-")
                        and c[len(other) + 1:len(other) + 2].isdigit())
                    for c in path.split("/")
                ):
                    return other
        # Signal three: ask the wrapper. Independent of where it was installed.
        return cls._wrapper_says(runner)

    @classmethod
    def _wrapper_says(cls, runner: Runner | None = None) -> str | None:
        """The scheduler `pbsnodes` names in its own output, if it names one.

        Bounded and failure-tolerant: a real PBS answers without any marker, and
        a `pbsnodes` that cannot run at all tells us nothing either way, so both
        come back ``None`` and the caller keeps the answer the path rules gave.
        """
        run = runner or SubprocessRunner()
        try:
            rc, out, err = run.run_full(["pbsnodes"], timeout=10.0)
        except Exception:  # pragma: no cover - a probe must not break detection
            return None
        text = f"{out}\n{err}".lower()
        for marker, other in cls._WRAPPER_OUTPUT_MARKERS:
            if marker in text:
                return other
        return None

    @classmethod
    def detect(cls) -> bool:
        # qstat alone is ambiguous (SGE ships one too); pbsnodes is specific.
        if not (which("pbsnodes") and which("qstat")):
            return False
        # ...and pbsnodes is not specific either where it is somebody else's
        # wrapper. See `wrapped_by`.
        return cls.wrapped_by() is None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            probe=False,
            probe_command="",
            notes=(
                "no verify-only mode: entitlement is DECLARED, from queue ACLs only",
            ),
        )

    # -- nodes --------------------------------------------------------------
    def parse_nodes_json(self, text: str) -> list[Node]:
        data = json.loads(text)
        out: list[Node] = []
        for name, n in (data.get("nodes") or {}).items():
            avail = n.get("resources_available", {}) or {}
            used = n.get("resources_assigned", {}) or {}
            states = [s.strip().lower() for s in str(n.get("state", "")).split(",") if s.strip()]
            conditions = {_STATE_TO_CONDITION[s] for s in states if s in _STATE_TO_CONDITION}
            # PBS reports accelerator model in a custom resource on most sites;
            # the common names are tried in order.
            model = next(
                (
                    str(avail[k])
                    for k in ("gpu_model", "gputype", "accelerator_model", "gpu")
                    if avail.get(k) and not str(avail[k]).isdigit()
                ),
                "",
            )
            labels = [f"{k}={v}" for k, v in avail.items() if isinstance(v, str)]
            # See _WHOLLY_ALLOCATED: exclusive placement is recorded in the
            # state, and the unassigned resources behind it are not free.
            whole = bool(_WHOLLY_ALLOCATED.intersection(states))
            cpus_total = count(avail.get("ncpus"))
            memory_mb = _mem_to_mb(avail.get("mem"))
            gpus_total = count(avail.get("ngpus"))
            out.append(
                Node(
                    name=name,
                    state_raw=str(n.get("state", "")),
                    conditions=frozenset(conditions),
                    cpus_total=cpus_total,
                    cpus_alloc=cpus_total if whole else count(used.get("ncpus")),
                    memory_mb=memory_mb,
                    memory_alloc_mb=(
                        memory_mb if whole else _mem_to_mb(used.get("mem"))),
                    gpus_total=gpus_total,
                    gpus_alloc=gpus_total if whole else count(used.get("ngpus")),
                    accelerator=identify_accelerator(None, model or labels),
                    accelerator_label=name_accelerator(None, model or labels) or "",
                    labels=tuple(labels),
                    queues=tuple(
                        q.strip()
                        for q in str(avail.get("Qlist") or n.get("queue") or "").split(",")
                        if q.strip()
                    ),
                    reason=str(n.get("comment") or ""),
                    unreachable="state-unknown" in states or "stale" in states,
                )
            )
        return out

    def parse_nodes_text(self, text: str) -> list[Node]:
        """Parse classic ``pbsnodes -a`` output (Torque, older PBS)."""
        out: list[Node] = []
        name: str | None = None
        fields: dict[str, str] = {}

        def flush() -> None:
            if name is None:
                return
            states = [
                s.strip().lower() for s in fields.get("state", "").split(",") if s.strip()
            ]
            conditions = {_STATE_TO_CONDITION[s] for s in states if s in _STATE_TO_CONDITION}
            model = fields.get("resources_available.gpu_model") or fields.get(
                "resources_available.gputype", ""
            )
            labels = [f"{k.split('.', 1)[-1]}={v}" for k, v in fields.items() if "." in k]
            # Same rule as the JSON path, and it has to be in both: Torque and
            # PBS before 18 speak only this one. See _WHOLLY_ALLOCATED.
            whole = bool(_WHOLLY_ALLOCATED.intersection(states))
            cpus_total = count(
                fields.get("resources_available.ncpus")
                or fields.get("pcpus")
                or fields.get("np")
            )
            memory_mb = _mem_to_mb(fields.get("resources_available.mem"))
            gpus_total = count(
                fields.get("resources_available.ngpus") or fields.get("gpus")
            )
            out.append(
                Node(
                    name=name,
                    state_raw=fields.get("state", ""),
                    conditions=frozenset(conditions),
                    cpus_total=cpus_total,
                    cpus_alloc=(
                        cpus_total if whole
                        else count(fields.get("resources_assigned.ncpus"))),
                    memory_mb=memory_mb,
                    memory_alloc_mb=(
                        memory_mb if whole
                        else _mem_to_mb(fields.get("resources_assigned.mem"))),
                    gpus_total=gpus_total,
                    gpus_alloc=(
                        gpus_total if whole
                        else count(fields.get("resources_assigned.ngpus"))),
                    accelerator=identify_accelerator(None, model or labels),
                    accelerator_label=name_accelerator(None, model or labels) or "",
                    labels=tuple(labels),
                    queues=tuple(
                        q.strip()
                        for q in (
                            fields.get("resources_available.Qlist")
                            or fields.get("queue", "")
                        ).split(",")
                        if q.strip()
                    ),
                    reason=fields.get("comment", ""),
                    unreachable="state-unknown" in states,
                )
            )

        for raw in _unwrap(text):
            if not raw.strip():
                continue
            if not raw[0].isspace():
                flush()
                name, fields = raw.strip(), {}
                continue
            key, _, value = raw.strip().partition("=")
            if value:
                fields[key.strip()] = value.strip()
        flush()
        return out

    def load_nodes(self) -> list[Node]:  # noqa: D401 - see the cache note below
        # Cached deliberately. load_queues() needs the nodes too, and
        # re-deriving them there would query the control plane a second time --
        # so a single report could mix two different instants, which is exactly
        # what taking one snapshot is supposed to prevent.
        with self._cache_lock:
            if self._nodes is not None:
                return self._nodes
            return self._load_nodes_uncached()

    def _load_nodes_uncached(self) -> list[Node]:
        try:
            nodes = self.parse_nodes_json(
                self.runner.run(
                    ["pbsnodes", "-a", "-F", "json"], timeout=_ENUMERATE_TIMEOUT)
            )
        except Exception:
            # Torque and PBS before 18 have no JSON mode at all.
            nodes = self.parse_nodes_text(
                self.runner.run(["pbsnodes", "-a"], timeout=_ENUMERATE_TIMEOUT))
        self._nodes = nodes
        return nodes

    def _queue_text(self) -> str:
        """``qstat -Qf`` output, fetched once and reused for limits."""
        with self._cache_lock:
            return self._queue_text_locked()

    def _queue_text_locked(self) -> str:
        if self._queue_text_cache is None:
            self._queue_text_cache = self.runner.run(
                ["qstat", "-Qf"], timeout=_ENUMERATE_TIMEOUT)
        return self._queue_text_cache

    def _queue_json(self) -> str | None:
        """``qstat -Qf -F json`` output, fetched once; ``None`` where absent.

        Cached including the failure, so a PBS without JSON mode (Torque, PBS
        before 18) is asked exactly once rather than once per consumer. The
        empty string is the "asked, unavailable" sentinel; `None` from this
        method means the same thing to callers, which all fall back to the text
        form.
        """
        with self._cache_lock:
            if self._queue_json_cache is None:
                try:
                    self._queue_json_cache = self.runner.run(
                        ["qstat", "-Qf", "-F", "json"], timeout=_ENUMERATE_TIMEOUT)
                except Exception:
                    self._queue_json_cache = ""
            return self._queue_json_cache or None

    # -- queues -------------------------------------------------------------
    def parse_queues_json(self, text: str) -> list[Queue]:
        data = json.loads(text)
        out: list[Queue] = []
        for name, q in (data.get("Queue") or {}).items():
            out.append(self._queue_from_fields(name, {k: str(v) for k, v in _flat(q).items()}))
        return out

    def parse_queues_text(self, text: str) -> list[Queue]:
        out: list[Queue] = []
        name: str | None = None
        fields: dict[str, str] = {}
        for raw in _unwrap(text):
            if raw.startswith("Queue:"):
                if name:
                    out.append(self._queue_from_fields(name, fields))
                name, fields = raw.split(":", 1)[1].strip(), {}
                continue
            if name and raw.strip() and "=" in raw:
                key, _, value = raw.strip().partition("=")
                fields[key.strip()] = value.strip()
        if name:
            out.append(self._queue_from_fields(name, fields))
        return out

    def _queue_from_fields(self, name: str, f: dict[str, str]) -> Queue:
        enabled = str(f.get("enabled", "True")).lower().startswith("t")
        started = str(f.get("started", "True")).lower().startswith("t")
        acl_on = str(f.get("acl_user_enable", "False")).lower().startswith("t")
        group_on = str(f.get("acl_group_enable", "False")).lower().startswith("t")
        # A Route queue forwards to its destinations and runs nothing itself.
        # Reporting it as an execution queue with zero nodes both understates
        # the cluster and offers it as a placement target that has no capacity.
        routes = str(f.get("queue_type", "Execution")).strip().lower() == "route"
        destinations = tuple(_csv(f.get("route_destinations"))) if routes else ()
        return Queue(
            forwards_to=destinations,
            name=name,
            # A word, not the two booleans. `enabled=True started=True` is 27
            # characters of which the first 12 fit the state column, so every
            # PBS queue rendered as `enabled=Tru…` -- truncated exactly where
            # the answer was, and the same string whether the queue was open or
            # shut. Seen on a live 2026.1.0 cluster, where six queues showed
            # `enabled=Tru…` and `enabled=Fal…` side by side and neither said
            # anything. The two switches are independent, so all four states get
            # their own word, and PBS's own vocabulary is the one used:
            # `enabled` gates accepting a job, `started` gates running it.
            #
            # Nothing loses information: `enabled` and `started` stay on the
            # Queue as booleans, and they are what every decision reads.
            state_raw=("UP" if enabled and started
                       else "STOPPED" if enabled
                       else "DISABLED" if started
                       else "DOWN"),
            enabled=enabled,
            started=started,
            # An ACL that is switched off is not a restriction; one that is on
            # but empty permits nobody, which the core reads via "none".
            allow_users=(
                tuple(_csv(f.get("acl_users"))) or ("none",) if acl_on else ()
            ),
            allow_groups=(
                tuple(_csv(f.get("acl_groups"))) or ("none",) if group_on else ()
            ),
            max_walltime_seconds=_pbs_walltime(f.get("resources_max.walltime")),
            max_nodes=_int_or_none(f.get("resources_max.nodect")),
            declared_nodes=0,
            is_default=False,
            priority=_int_or_none(f.get("Priority")) or 0,
            limits_name=name,
        )

    def load_queues(self) -> list[Queue]:
        payload = self._queue_json()
        try:
            if payload is None:
                raise ValueError("no JSON mode")
            queues = self.parse_queues_json(payload)
        except Exception:
            queues = self.parse_queues_text(self._queue_text())
        # PBS does not list a queue's nodes; the mapping lives on the nodes'
        # Qlist, so it is reconstructed from the other direction.
        #
        # The rule matters and is easy to get backwards. Qlist *restricts* a
        # node to the named queues; a node that declares none is unrestricted
        # and any execution queue may use it. Requiring an explicit mention
        # instead orphans every unrestricted node -- its capacity becomes
        # invisible to every queue -- and leaves a queue no node happens to
        # name looking genuinely empty, with nothing said about it.
        # Sets, and built ONCE per queue. The membership test used to run
        # against a list, and `set(members)` sat inside the generator's `if`
        # clause -- so the set was rebuilt for every node, making this
        # quadratic in the node count and cubic-ish overall. Measured on a
        # 10,624-node PBS Pro cluster: 51 queues took **151 seconds**, out of a
        # 3m10s run whose scheduler queries accounted for 11s of it. A tool you
        # reach for while a cluster misbehaves cannot spend three minutes
        # arriving. Same output, 0.2s.
        nodes = self.load_nodes()
        unrestricted = {n.name for n in nodes if not n.queues}
        for q in queues:
            if q.routes:
                # Its capacity belongs to the destinations, not to it.
                q.node_names = ()
                q.declared_nodes = 0
                continue
            members = {n.name for n in nodes if q.name in n.queues} | unrestricted
            q.node_names = tuple(n.name for n in nodes if n.name in members)
            q.declared_nodes = len(q.node_names)
        return queues

    # -- limits -------------------------------------------------------------
    def _limit_entity(self) -> str | None:
        """Who we are, for resolving a per-entity limit table -- or ``None``.

        `load_identity` insists on a username, because there an unknown caller
        would read as belonging to no group and that is a false denial.  Here it
        is genuinely optional: ``PBS_GENERIC`` is the right answer for an
        ordinary user anyway, so a host with no passwd entry must not turn every
        queue's declared ceilings into an error.
        """
        try:
            return os.environ.get("USER") or getpass.getuser()
        except Exception:  # no passwd entry and no LOGNAME: getuser() raises
            return None

    def _limits_from_fields(
        self, name: str, fields: dict[str, str], user: str | None = None
    ) -> Limits:
        """One queue's ceilings, from its flat ``key.subkey = value`` map.

        `user` is the entity a per-entity table is resolved against; see
        :func:`_pbs_limit_cell`.
        """
        per_job: dict[str, int] = {}
        per_user: dict[str, int] = {}
        # A field that held something we could not read is not the same as an
        # absent one, and only the first silently disables a check -- the
        # `Limits.unreadable` contract the Slurm adapter established
        # (MaxWall/MaxTRESPerUser/MaxTRESPerJob) and this one left empty. A
        # `max_run_res.*` or `resources_max.*` value in a shape the parser does
        # not understand used to simply disappear, so the queue read as
        # UNLIMITED on that axis and was indistinguishable from one that
        # declares no ceiling at all -- on a scheduler with no dry-run, where
        # the declared ceiling is the only warning there is.
        unreadable: list[str] = []

        def read(key: str, convert: Callable[[str], int | None]) -> int | None:
            """One ceiling, plus a note when it was published but unread."""
            raw = fields.get(key)
            if not _pbs_limit_declared(raw):
                return None  # unset, or an explicit "unlimited": no ceiling
            cell = _pbs_limit_cell(raw, user)
            if cell is None:
                # The table parsed and none of its cells is ours:
                # `[u:alice=99],[u:bob=1]` declares nothing for carol. That is
                # a ceiling correctly not applied, not one we failed to read,
                # and calling it unreadable would warn about a limit that does
                # not exist for the caller.
                return None
            got = convert(cell)
            if got is None:
                unreadable.append(key)
            return got

        # PBS prints attributes in whatever order the administrator set them,
        # so the names are collected in a fixed order rather than that one.
        wall = read("resources_max.walltime", _pbs_walltime)
        for key in sorted(fields):
            if key.startswith("max_run_res."):
                target = per_user
            elif key.startswith("resources_max."):
                target = per_job
            else:
                continue
            mapped = _PBS_LIMIT_RESOURCES.get(key.split(".", 1)[1])
            if mapped is None:
                continue
            got = read(key, _pbs_mem_mb if mapped == "mem_mb" else _pbs_int)
            if got is not None:
                target[mapped] = got
        # Read before the constructor call rather than inside it: `read` is
        # what appends to `unreadable`, so anything read after it is passed
        # would not be in it.
        max_jobs = read("max_run", _pbs_int)
        max_submitted = read("max_queued", _pbs_int)
        return Limits(
            name=name,
            max_walltime_seconds=wall,
            per_job=per_job,
            per_user=per_user,
            # These two are per-entity tables exactly like `max_run_res.*`
            # above -- the recorded fixture carries `max_run =
            # [u:PBS_GENERIC=4]` -- so they resolve through the same cell
            # picker rather than through `_int_or_none`, which anchors at the
            # start of the string and read every one of them as `None`. Slurm
            # fills the same two fields from MaxJobsPerUser /
            # MaxSubmitJobsPerUser, so the per-user resolution matches.
            max_jobs=max_jobs,
            max_submitted=max_submitted,
            source="pbs queue limits",
            unreadable=tuple(unreadable),
        )

    def load_limits(self) -> dict[str, Limits]:
        # Raised, not turned into "no limits". PBS has no dry-run at all, so its
        # declared ceilings are the ONLY warning a caller gets before a job is
        # accepted and then pends forever -- and an empty dict is
        # indistinguishable from a cluster that declares none. Same rule as the
        # Slurm, LSF and Kubernetes backends; Cluster.load records it.
        #
        # The JSON payload is preferred because `load_queues` has usually
        # fetched it already, and the two carry the same fields under the same
        # names -- `_flat` gives the dotted keys the text form prints verbatim.
        # Asking twice used to cost a whole extra query: on a 10,624-node PBS
        # Pro cluster `qstat -Qf -F json` took 23.9s and the plain `qstat -Qf`
        # another 14.9s, 38 seconds for the same 37 KB of queue attributes,
        # fetched at two different instants -- which is also how one report
        # comes to describe two moments.
        user = self._limit_entity()
        payload = self._queue_json()
        if payload is not None:
            try:
                return {
                    name: self._limits_from_fields(
                        name, {k: str(v) for k, v in _flat(q).items()}, user)
                    for name, q in (json.loads(payload).get("Queue") or {}).items()
                }
            except Exception:
                pass  # malformed JSON: fall through to the text form below
        out: dict[str, Limits] = {}
        name: str | None = None
        fields: dict[str, str] = {}

        def flush() -> None:
            if not name:
                return
            out[name] = self._limits_from_fields(name, fields, user)

        for raw in _unwrap(self._queue_text()):
            if raw.startswith("Queue:"):
                flush()
                name, fields = raw.split(":", 1)[1].strip(), {}
                continue
            if name and "=" in raw:
                key, _, value = raw.strip().partition("=")
                fields[key.strip()] = value.strip()
        flush()
        return out

    def load_jobs(self) -> list[Job]:
        """Not implemented for this system.

        The protocol requires the method; an empty list here means "this adapter
        cannot list jobs", which the caller tells apart from "this node has no
        jobs" by asking whether the cluster returned any jobs at all. Reporting
        an idle node because the query does not exist would be phantom capacity
        in a new place.
        """
        return []

    # -- identity -----------------------------------------------------------
    def load_identity(self) -> Identity:
        user = os.environ.get("USER") or getpass.getuser()
        # Committed only if the whole lookup succeeded, and raised otherwise.
        # PBS entitlement is per-queue ACL matched against these groups, and the
        # membership test is tri-state: a half-collected list is read as
        # authoritative and yields a false denial that hides a usable queue,
        # while an empty one reads as "cannot tell" and silently ignores every
        # ACL. See `slurm._unix_groups` for the same reasoning.
        found: set[str] = set()
        import grp
        import pwd

        for g in grp.getgrall():
            if user in g.gr_mem:
                found.add(g.gr_name)
        found.add(grp.getgrgid(pwd.getpwnam(user).pw_gid).gr_name)
        groups = found
        # PBS has no account/association table; entitlement is per-queue ACL,
        # which the Queue objects already carry.
        return Identity(user=user, groups=tuple(sorted(groups)))

    # -- free times ---------------------------------------------------------
    def parse_free_times(self, text: str) -> dict[str, datetime]:
        """Projected end time per node, from ``qstat -f``.

        PBS records no end time for a running job, so it has to be computed:
        ``stime`` (when the job started) plus ``Resource_List.walltime`` (what
        it may consume).  That is an upper bound on when the node frees, which
        is the right direction -- a job can finish early, so promising the node
        sooner would be a promise PBS never made.

        Two shapes have to be read correctly, and they are easy to confuse:

        * ``exec_host = gpu001/0*64+gpu002/0*64``  -- slash-separated
        * ``exec_vnode = (gpu001:ncpus=64)+(gpu002:ncpus=64)``  -- parenthesised

        Only ``exec_host`` is guaranteed present, and matching it with a
        parenthesis-based pattern silently yields nothing at all.
        """
        latest: dict[str, datetime] = {}
        hosts: list[str] = []
        started: datetime | None = None
        walltime: int | None = None

        def flush() -> None:
            if not hosts or started is None or walltime is None:
                return
            end = started + timedelta(seconds=walltime)
            for h in hosts:
                if h not in latest or end > latest[h]:
                    latest[h] = end

        for raw in _unwrap(text):
            line = raw.strip()
            if line.startswith("Job Id:"):
                flush()
                hosts, started, walltime = [], None, None
            elif line.startswith("exec_host"):
                value = line.split("=", 1)[-1]
                # Take the host name from each "host/cpu*count" chunk.
                hosts = [
                    chunk.split("/")[0].split(":")[0].strip("() ")
                    for chunk in value.split("+")
                    if chunk.strip()
                ]
            elif line.startswith("stime"):
                started = parse_timestamp(line.split("=", 1)[-1].strip())
            elif line.startswith("Resource_List.walltime"):
                walltime = _pbs_walltime(line.split("=", 1)[-1].strip())
        flush()
        return latest

    def load_node_free_times(self) -> dict[str, datetime]:
        try:
            text = self.runner.run(["qstat", "-f"])
        except Exception:
            return {}
        return self.parse_free_times(text)

    # -- probe --------------------------------------------------------------
    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        select = f"select={shape.nodes}:ncpus={shape.cpus_per_node}"
        if shape.gpus_per_node:
            select += f":ngpus={shape.gpus_per_node}"
        if shape.memory_gb:
            # A fractional --mem is accepted (`--mem 1.5G` -> memory_gb 1.5), and
            # `int()` on gigabytes threw the remainder away *downwards*: 1.5 became
            # 1G and 0.5 became **0G**, a request for no memory at all. Under-asking
            # is what gets a job killed, so the flags this tool hands over to be
            # pasted must not ask for less than the shape they were computed from.
            # Megabytes are exact and both schedulers take the unit; the Slurm and
            # LSF backends already emit MB for the same reason.
            #
            # Gigabytes are KEPT when the figure is a whole number of them: that is
            # what almost every request is, `mem=64gb` reads better than `mem=65536mb`
            # in a command someone is about to paste, and two existing backend tests
            # rightly asserted that spelling. Megabytes appear only where gigabytes
            # would lose something.
            mb = shape.memory_mb_per_node
            select += f":mem={mb // 1024}gb" if mb % 1024 == 0 else f":mem={mb}mb"
        args = ["-q", queue, "-l", select, "-l", f"walltime={_hhmmss(shape.walltime_seconds)}"]
        if shape.account:
            args += ["-A", shape.account]
        return args

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """PBS offers no verify-only submission, so there is nothing to ask."""
        return None

    def format_nodelist(self, names: Iterable[str]) -> str:
        return "+".join(sorted(names))


def _flat(obj: dict, prefix: str = "") -> dict[str, object]:
    """Flatten nested JSON into ``resources_max.walltime`` style keys."""
    out: dict[str, object] = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, f"{key}."))
        else:
            out[key] = v
    return out


def _unwrap(text: str) -> list[str]:
    """PBS attribute text as logical lines, continuations rejoined.

    **PBS wraps long attribute values at 80 columns**, breaking mid-value and
    indenting the remainder with a tab. Every parser here iterated raw lines and
    required an ``=`` to accept one, so each continuation was silently dropped --
    losing the *tail* of exactly the values long enough to wrap, which are the
    ones that matter:

    * a node's ``resources_available.Qlist``, so a node serving eight queues was
      recorded as serving five. It then goes missing from the queues whose names
      were cut, and its capacity with it.
    * a queue's ``acl_users`` / ``acl_groups``, so a truncated allowlist reads as
      authoritative and produces a **false denial** -- the tool reporting no
      access to a queue that would have taken the job.
    * a job's ``exec_host``, which is the longest field PBS emits and always
      wraps for a multi-node job. It maps running work to nodes, so a truncated
      one attributes free-time estimates to the wrong machines.

    A continuation is an indented line with no ``=``: every real attribute line
    has one, and a record header (``Queue: name``, a node name, ``Job Id:``) is
    never indented, so neither can be mistaken for the other. Rejoined with no
    separator, because that is how PBS broke it.
    """
    out: list[str] = []
    for raw in text.splitlines():
        if out and raw[:1].isspace() and "=" not in raw:
            out[-1] += raw.strip()
        else:
            out.append(raw)
    return out


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in re.split(r"[,\s]+", value) if p.strip()]


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.match(r"^\s*(\d+)", str(value))
    return int(m.group(1)) if m else None


def _hhmmss(seconds: int | None) -> str:
    if seconds is None:
        return "24:00:00"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
