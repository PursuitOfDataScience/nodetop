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
from collections.abc import Iterable
from datetime import datetime, timedelta

from ..core.duration import parse_timestamp
from ..core.hardware import identify_accelerator
from ..core.model import Identity, Job, JobShape, Limits, Node, Queue, Verdict
from ..runner import Runner, SubprocessRunner, which
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


def _mem_to_mb(text: str | None) -> int:
    """PBS memory strings: ``256gb``, ``1024mb``, ``2tb``, bare bytes."""
    if not text:
        return 0
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)([bw]?)\s*$", str(text), re.IGNORECASE)
    if not m:
        return 0
    value = float(m.group(1))
    scale = {"": 1 / (1024 * 1024), "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
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


def _strip_pbs_limit(value: str) -> int | None:
    """``max_run_res.ngpus = [u:PBS_GENERIC=8]`` -> 8.

    PBS expresses a limit as a per-entity table; the generic entry is the one
    that applies to an ordinary user.
    """
    m = re.search(r"=\s*(\d+)\s*\]?\s*$", str(value))
    if m:
        return int(m.group(1))
    m = re.match(r"^\s*(\d+)\s*$", str(value))
    return int(m.group(1)) if m else None


class PbsBackend:
    """Adapter for PBS Pro / OpenPBS / Torque."""

    name = "pbs"
    queue_term = "queue"

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or SubprocessRunner()
        self._nodes: list[Node] | None = None
        self._queue_text_cache: str | None = None
        self._queue_json_cache: str | None = None

    @classmethod
    def detect(cls) -> bool:
        # qstat alone is ambiguous (SGE ships one too); pbsnodes is specific.
        return which("pbsnodes") and which("qstat")

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

    def load_nodes(self) -> list[Node]:
        # Cached deliberately. load_queues() needs the nodes too, and
        # re-deriving them there would query the control plane a second time --
        # so a single report could mix two different instants, which is exactly
        # what taking one snapshot is supposed to prevent.
        if self._nodes is not None:
            return self._nodes
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
            state_raw=f"enabled={enabled} started={started}",
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
    def _limits_from_fields(self, name: str, fields: dict[str, str]) -> Limits:
        """One queue's ceilings, from its flat ``key.subkey = value`` map."""
        per_job: dict[str, int] = {}
        per_user: dict[str, int] = {}
        for key, value in fields.items():
            target = None
            if key.startswith("max_run_res."):
                target = per_user
            elif key.startswith("resources_max."):
                target = per_job
            if target is None:
                continue
            resource = key.split(".", 1)[1]
            got = _strip_pbs_limit(value)
            if got is None:
                continue
            mapped = {"ngpus": "gpu", "ncpus": "cpu", "nodect": "node"}.get(resource)
            if mapped:
                target[mapped] = got
        return Limits(
            name=name,
            max_walltime_seconds=_pbs_walltime(fields.get("resources_max.walltime")),
            per_job=per_job,
            per_user=per_user,
            max_jobs=_int_or_none(fields.get("max_run")),
            max_submitted=_int_or_none(fields.get("max_queued")),
            source="pbs queue limits",
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
        payload = self._queue_json()
        if payload is not None:
            try:
                return {
                    name: self._limits_from_fields(
                        name, {k: str(v) for k, v in _flat(q).items()})
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
            out[name] = self._limits_from_fields(name, fields)

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
            select += f":mem={int(shape.memory_gb)}gb"
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
