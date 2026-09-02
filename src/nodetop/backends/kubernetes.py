"""Kubernetes.

Kubernetes is the strongest case that this tool is not about Slurm: it has
every one of the same lies, in its own vocabulary.

* **Phantom capacity.**  A node can be ``Ready`` with most of its CPU
  unrequested and still take nothing, because it is cordoned
  (``spec.unschedulable``) or carries a taint nothing tolerates.  ``kubectl get
  nodes`` shows ``Ready,SchedulingDisabled`` and every capacity number intact.
* **Declared vs enforced entitlement.**  RBAC says you may create pods; a
  ``ResourceQuota`` or a validating webhook then refuses the specific pod.
* **Admitted then never scheduled.**  A pod within quota whose resource request
  no node can satisfy is accepted and sits ``Pending`` indefinitely -- the
  exact shape of an over-limit Slurm job pending on ``QOSMaxGRESPerJob``.
* **Opaque accelerators.**  ``nvidia.com/gpu: 4`` says nothing about which GPU,
  how much memory it has, or whether it supports the dtype you need.

Kubernetes also has the *best* dry-run of any system here: ``--dry-run=server``
runs real admission, including quota, and ``kubectl auth can-i`` answers the
RBAC question directly.  Both create nothing.

A namespace plus its quota is mapped onto :class:`~nodetop.core.model.Queue`,
which is the closest honest analogue of "a named submission target with its own
policy".
"""

from __future__ import annotations

import getpass
import json
import os
import re
from collections.abc import Iterable
from datetime import datetime

from ..core.hardware import identify_accelerator, name_accelerator
from ..core.model import (
    Identity,
    Job,
    JobShape,
    Limits,
    Node,
    Queue,
    Verdict,
    VerdictCategory,
)
from ..runner import Runner, SubprocessRunner, which
from .base import BackendCapabilities

__all__ = ["KubernetesBackend"]

#: Accelerator resource names, in the order they are looked for.
_GPU_RESOURCES = (
    "nvidia.com/gpu",
    "amd.com/gpu",
    "gpu.intel.com/i915",
    "habana.ai/gaudi",
)

#: Labels that carry the accelerator model.
_MODEL_LABELS = (
    "nvidia.com/gpu.product",
    "gpu.intel.com/product",
    "amd.com/gpu.product",
    "beta.amd.com/gpu.device-id",
    "node.kubernetes.io/instance-type",
    "cloud.google.com/gke-accelerator",
)


def _quantity_to_mb(text: str | None) -> int:
    """Kubernetes quantities: ``256Gi``, ``1000m``, ``2T``, plain bytes."""
    if text is None:
        return 0
    s = str(text).strip()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([EPTGMK]i?|m)?$", s)
    if not m:
        return 0
    value = float(m.group(1))
    suffix = m.group(2) or ""
    binary = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024, "Ti": 1024**2, "Pi": 1024**3,
              "Ei": 1024**4}
    decimal = {"K": 1000 / 1024**2, "M": 1000**2 / 1024**2, "G": 1000**3 / 1024**2,
               "T": 1000**4 / 1024**2, "P": 1000**5 / 1024**2}
    if suffix in binary:
        return int(value * binary[suffix])
    if suffix in decimal:
        return int(value * decimal[suffix])
    if suffix == "m":
        return 0
    return int(value / (1024 * 1024))


def _quantity_to_cpu(text: str | None) -> int:
    """CPU quantities, rounded down to whole cores; ``500m`` -> 0.

    Clamped at zero: a negative core count is meaningless, and letting one
    through puts a negative total in the inventory and in every sum built on
    it.
    """
    if text is None:
        return 0
    s = str(text).strip()
    try:
        value = float(s[:-1]) / 1000 if s.endswith("m") else float(s)
    except ValueError:
        return 0
    return max(0, int(value))


class KubernetesBackend:
    """Adapter for Kubernetes."""

    name = "kubernetes"
    queue_term = "namespace"

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or SubprocessRunner()
        self._nodes: list[Node] | None = None
        self._quota_cache: str | None = None

    @classmethod
    def detect(cls) -> bool:
        if not which("kubectl"):
            return False
        # A kubectl with no reachable cluster is worse than no kubectl: it
        # would claim the backend and then fail every query.
        from ..runner import SubprocessRunner

        return SubprocessRunner().ok(
            ["kubectl", "version", "--request-timeout=3s", "-o", "json"], timeout=8
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            probe=which("kubectl"),
            probe_supported=True,
            probe_command="kubectl auth can-i + --dry-run=server",
            notes=(
                "server-side dry-run runs real admission, so a ResourceQuota breach IS "
                "caught",
                "admitted within quota still pends forever if no node fits the request",
            ),
        )

    # -- nodes --------------------------------------------------------------
    def parse_nodes(self, nodes_json: str, pods_json: str = "") -> list[Node]:
        data = json.loads(nodes_json)

        # Requested resources per node, summed over non-terminal pods.  This is
        # the only correct source: a node's `allocatable` is a capacity, not an
        # occupancy, and reading it as free space overstates availability.
        used_cpu: dict[str, int] = {}
        used_mem: dict[str, int] = {}
        used_gpu: dict[str, int] = {}
        if pods_json:
            for pod in json.loads(pods_json).get("items", []):
                phase = (pod.get("status") or {}).get("phase", "")
                if phase in {"Succeeded", "Failed"}:
                    continue
                node = (pod.get("spec") or {}).get("nodeName")
                if not node:
                    continue
                cpu, mem, gpu = _pod_request(pod)
                used_cpu[node] = used_cpu.get(node, 0) + cpu
                used_mem[node] = used_mem.get(node, 0) + mem
                used_gpu[node] = used_gpu.get(node, 0) + gpu

        out: list[Node] = []
        for item in data.get("items", []):
            meta = item.get("metadata") or {}
            spec = item.get("spec") or {}
            status = item.get("status") or {}
            name = meta.get("name", "")
            if not name:
                continue
            allocatable = status.get("allocatable") or {}
            labels = meta.get("labels") or {}

            conditions: set[str] = set()
            reason = ""
            ready = None
            for cond in status.get("conditions", []):
                ctype, cstatus = cond.get("type"), cond.get("status")
                if ctype == "Ready":
                    ready = cstatus
                    if cstatus != "True":
                        conditions.add("DOWN")
                        reason = cond.get("message") or cond.get("reason") or ""
                elif ctype in {"MemoryPressure", "DiskPressure", "PIDPressure",
                               "NetworkUnavailable"} and cstatus == "True":
                    # Pressure does not stop scheduling outright, but it is
                    # exactly the "running but impaired" signal worth surfacing.
                    reason = reason or f"{ctype} reported by kubelet"
            # Cordoned: Ready, fully capable, and will take nothing.
            if spec.get("unschedulable"):
                conditions.add("DRAIN")
                reason = reason or "cordoned (spec.unschedulable)"

            taints = tuple(
                f"{t.get('key')}={t.get('value', '')}:{t.get('effect')}"
                for t in spec.get("taints", [])
                if t.get("effect") in {"NoSchedule", "NoExecute"}
            )

            gpus = 0
            gpu_resource = ""
            for res in _GPU_RESOURCES:
                if res in allocatable:
                    gpus = int(float(allocatable[res]))
                    gpu_resource = res
                    break

            model_labels = [
                f"{k}={v}" for k, v in labels.items() if k in _MODEL_LABELS
            ]
            out.append(
                Node(
                    name=name,
                    state_raw=(
                        ("Ready" if ready == "True" else "NotReady")
                        + (",SchedulingDisabled" if spec.get("unschedulable") else "")
                    ),
                    conditions=frozenset(conditions),
                    cpus_total=_quantity_to_cpu(allocatable.get("cpu")),
                    cpus_alloc=used_cpu.get(name, 0),
                    memory_mb=_quantity_to_mb(allocatable.get("memory")),
                    memory_alloc_mb=used_mem.get(name, 0),
                    gpus_total=gpus,
                    gpus_alloc=used_gpu.get(name, 0),
                    accelerator=identify_accelerator(gpu_resource, model_labels or None),
                    accelerator_label=name_accelerator(
                        gpu_resource, model_labels or None) or "",
                    labels=tuple(f"{k}={v}" for k, v in sorted(labels.items())),
                    # Every namespace can target every node, so the queue
                    # mapping is filled in by load_queues instead.
                    queues=(),
                    reason=reason,
                    unreachable=ready not in {"True", None},
                    taints=taints,
                )
            )
        return out

    def load_nodes(self) -> list[Node]:
        # Cached deliberately. load_queues() needs the nodes too, and
        # re-deriving them there would query the control plane a second time --
        # so a single report could mix two different instants, which is exactly
        # what taking one snapshot is supposed to prevent.
        if self._nodes is not None:
            return self._nodes
        nodes = self.runner.run(["kubectl", "get", "nodes", "-o", "json"], timeout=60)
        # NOT suppressed, and the comment that used to sit here was wrong in
        # both of its claims. It said the missing query would be "recorded in
        # Cluster.errors" -- it was not, because swallowing it made load_nodes
        # succeed -- and that the caller would not see a node as fully free.
        #
        # Pod requests ARE the occupancy on Kubernetes: allocatable is a
        # capacity, not a free count. With no pods, every node parses as
        # zero-allocated, so a node running 40 of its 48 cores and all 4 of its
        # accelerators was reported as 48/48 and 4/4 free, idle=True. That is
        # phantom capacity -- the precise failure this tool exists to catch --
        # produced silently, and `kubectl get pods --all-namespaces` is
        # routinely forbidden by RBAC for a namespaced user, so it fires in
        # practice rather than in theory.
        #
        # There is no representation for "occupancy unknown" in the model, and
        # inventing one that reads as free is the worse of the two errors. So
        # this fails loudly: Cluster.load records it and the dispatch guard
        # reports no data rather than an idle cluster.
        pods = self.runner.run(
            ["kubectl", "get", "pods", "--all-namespaces",
             "--field-selector=status.phase!=Succeeded", "-o", "json"],
            timeout=120,
        )
        self._nodes = self.parse_nodes(nodes, pods)
        return self._nodes

    # -- queues (namespaces) ------------------------------------------------
    def parse_queues(self, ns_json: str, quota_json: str = "") -> list[Queue]:
        quotas: dict[str, dict] = {}
        if quota_json:
            for item in json.loads(quota_json).get("items", []):
                ns = (item.get("metadata") or {}).get("namespace", "")
                hard = ((item.get("status") or {}).get("hard") or {})
                used = ((item.get("status") or {}).get("used") or {})
                quotas.setdefault(ns, {"hard": {}, "used": {}})
                quotas[ns]["hard"].update(hard)
                quotas[ns]["used"].update(used)

        out: list[Queue] = []
        for item in json.loads(ns_json).get("items", []):
            meta = item.get("metadata") or {}
            name = meta.get("name", "")
            phase = ((item.get("status") or {}).get("phase")) or "Active"
            out.append(
                Queue(
                    name=name,
                    state_raw=phase,
                    # A namespace that is Terminating accepts nothing.
                    enabled=phase == "Active",
                    started=phase == "Active",
                    hidden=name.startswith("kube-"),
                    limits_name=name,
                )
            )
        return out

    def _quota_json(self) -> str:
        """The quota listing, fetched once.

        Both the namespace mapping and the ceilings are read out of it, and
        fetching it twice means a report could describe two different instants.
        """
        # Not suppressed. Swallowing to "" here defeated the caller that most
        # needs the error: `load_limits` then parsed an empty string and raised
        # `JSONDecodeError: Expecting value`, so the failure recorded against
        # "limits" said nothing about the actual cause (typically
        # `Forbidden`). The two callers want different things and each says so
        # for itself -- see `load_queues`, which tolerates the loss because
        # namespaces are its primary data.
        #
        # On failure the cache stays None so a later call retries, rather than
        # latching an empty answer for the life of the process.
        if self._quota_cache is None:
            self._quota_cache = self.runner.run(
                ["kubectl", "get", "resourcequota", "--all-namespaces",
                 "-o", "json"],
                timeout=60,
            )
        return self._quota_cache

    def load_queues(self) -> list[Queue]:
        ns = self.runner.run(["kubectl", "get", "namespaces", "-o", "json"], timeout=60)
        # Quota is supplementary here: RBAC commonly permits listing namespaces
        # while forbidding resourcequota, and losing the ceilings must not cost
        # the whole queue listing. `load_limits` deliberately does NOT tolerate
        # the same failure, because there the ceilings are the answer.
        try:
            quota = self._quota_json()
        except Exception:
            quota = ""
        queues = self.parse_queues(ns, quota)
        # Any namespace can in principle place a pod on any node, so every
        # queue gets the full node list; taints and selectors do the filtering,
        # and they are evaluated per node in the capacity pass.
        all_nodes = tuple(n.name for n in self.load_nodes())
        for q in queues:
            q.node_names = all_nodes
            q.declared_nodes = len(all_nodes)
        return queues

    # -- limits -------------------------------------------------------------
    def parse_limits(self, quota_json: str) -> dict[str, Limits]:
        out: dict[str, Limits] = {}
        for item in json.loads(quota_json).get("items", []):
            meta = item.get("metadata") or {}
            ns = meta.get("namespace", "")
            hard = ((item.get("status") or {}).get("hard") or {})
            per_job: dict[str, int] = {}
            for key, value in hard.items():
                base = key.split("/")[-1] if "/" in key else key
                if base in {"cpu", "limits.cpu", "requests.cpu"}:
                    per_job["cpu"] = _quantity_to_cpu(value)
                elif base in {"memory", "limits.memory", "requests.memory"}:
                    per_job["mem_mb"] = _quantity_to_mb(value)
                elif "gpu" in key:
                    per_job["gpu"] = int(float(value))
            existing = out.get(ns)
            if existing is None:
                out[ns] = Limits(name=ns, per_job=per_job, source="kubernetes ResourceQuota")
            else:
                # Several quotas in one namespace all apply; the tightest wins.
                for k, v in per_job.items():
                    existing.per_job[k] = min(existing.per_job.get(k, v), v)
        return out

    def load_limits(self) -> dict[str, Limits]:
        # Raised, not swallowed into "no quotas". A ResourceQuota is the k8s
        # form of the ceiling that admits a job and then never runs it, and an
        # empty dict here is indistinguishable from a namespace with no quota at
        # all -- so the check is silently disabled exactly when it cannot be
        # made. `Cluster.load` records the failure instead, which is what the
        # Slurm backend has always done for the same query. See `Limits`, which
        # carries an `unreadable` flag for precisely this distinction.
        return self.parse_limits(self._quota_json())

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
        try:
            whoami = self.runner.run(["kubectl", "auth", "whoami", "-o", "json"], timeout=15)
            data = json.loads(whoami)
            info = (data.get("status") or {}).get("userInfo") or {}
            return Identity(
                user=info.get("username") or user,
                groups=tuple(info.get("groups") or ()),
            )
        except Exception:
            # Not swallowed into a group-less Identity. On Kubernetes the groups
            # ARE the entitlement mechanism -- RBAC binds to them -- and the
            # membership check downstream is tri-state: an empty group set reads
            # as "cannot tell", so no GROUP_NOT_ALLOWED is ever emitted and
            # every group restriction is silently ignored. A namespace the
            # caller's groups do not permit would be reported as available.
            #
            # `kubectl auth whoami` is absent before k8s 1.26 and can be refused
            # by RBAC, so this fires in practice rather than only in theory.
            # Raising lets Cluster.load record it and leave `identity` as None,
            # which the entitlement filter reads as "cannot filter" instead of
            # "entitled to nothing". Same rule as the Slurm backend.
            raise

    def load_node_free_times(self) -> dict[str, datetime]:
        # Pods have no deadline by default, so there is no honest per-node
        # free-time estimate to give.
        return {}

    # -- probe --------------------------------------------------------------
    def submit_flags(self, queue: str, shape: JobShape) -> list[str]:
        req = [f"cpu={shape.cpus_per_node}"]
        if shape.memory_gb:
            # Derived from `memory_mb_per_node`, not `int(shape.memory_gb)`.
            # `--mem` takes a fractional size, and truncating gibibytes threw
            # the remainder away *downwards*: 1.5 GiB became `memory=1Gi` and
            # 0.5 GiB became **`memory=0Gi`**, a request for no memory at all.
            #
            # Worse here than in the other adapters, because these flags are
            # not only pasted -- `probe()` feeds them to `kubectl run
            # --dry-run=server`, which is the one dry-run in this package that
            # really does evaluate a ResourceQuota. Asking admission about a
            # figure up to a gibibyte under the shape gets a PASS for a pod
            # that would then be refused, which is the exact "accepted, then
            # never runs" failure this tool exists to catch.
            #
            # `Mi` only where `Gi` would lose something: `memory=64Gi` reads
            # better than `memory=65536Mi` in a command about to be pasted, and
            # both suffixes are ordinary Kubernetes quantities. Same rule as
            # the PBS and Grid Engine adapters.
            mb = shape.memory_mb_per_node
            req.append(
                f"memory={mb // 1024}Gi" if mb % 1024 == 0 else f"memory={mb}Mi"
            )
        if shape.gpus_per_node:
            req.append(f"nvidia.com/gpu={shape.gpus_per_node}")
        return [
            "-n", queue,
            f"--requests={','.join(req)}",
            f"--limits={','.join(req)}",
        ]

    def probe(
        self, queue: str, shape: JobShape, account: str | None = None
    ) -> Verdict | None:
        """Read-only: RBAC check, then server-side dry-run admission.

        ``auth can-i`` answers the permission question and ``--dry-run=server``
        runs real admission including ``ResourceQuota``.  Neither creates
        anything.
        """
        if not self.capabilities().probe:
            # Gated on kubectl existing, same as the Slurm and SGE probes.
            # Without the guard a missing client surfaces as a
            # CONTROL_PLANE_DOWN verdict -- which is in TRANSIENT_CATEGORIES,
            # so the report would invite the caller to retry a condition that
            # no amount of waiting fixes, and blame the cluster for a local
            # problem.
            return None
        can = ["kubectl", "auth", "can-i", "create", "pods", "-n", queue]
        try:
            rc, out, err = self.runner.run_full(can, timeout=20)
        except Exception as exc:
            return Verdict(
                queue=queue, account=account, allowed=False,
                category=VerdictCategory.CONTROL_PLANE_DOWN, reason=str(exc), raw=str(exc),
            )
        if not _can_i_says_yes(out) or rc != 0:
            return Verdict(
                queue=queue, account=account, allowed=False,
                category=VerdictCategory.NOT_ENTITLED,
                reason=_can_i_reason(out, err) or
                f"RBAC: cannot create pods in namespace {queue}",
                raw=f"{out}\n{err}".strip(),
            )

        # Admission dry-run.  --dry-run=server is hard-coded so this path can
        # never create a pod.
        dry = [
            "kubectl", "run", f"nodetop-probe-{os.getpid()}",
            "--image=busybox", "--restart=Never", "--command",
            "--dry-run=server", "-o", "name",
            *self.submit_flags(queue, shape),
            "--", "true",
        ]
        try:
            rc, out, err = self.runner.run_full(dry, timeout=30)
        except Exception as exc:
            return Verdict(
                queue=queue, account=account, allowed=False,
                category=VerdictCategory.CONTROL_PLANE_DOWN, reason=str(exc), raw=str(exc),
            )
        text = f"{out}\n{err}"
        low = text.lower()
        # Trust the exit code.  kubectl returns non-zero when admission
        # refuses, and it writes deprecation and config warnings to stderr --
        # so also requiring the word "error" to be absent turns a successful
        # dry-run into a failure whenever a warning happens to contain it.
        if rc == 0:
            return Verdict(queue=queue, account=account, allowed=True,
                           category=VerdictCategory.OK, raw=text.strip())
        category = VerdictCategory.UNKNOWN
        if "exceeded quota" in low or "forbidden: exceeded" in low:
            category = VerdictCategory.QUOTA_EXCEEDED
        elif "forbidden" in low or "is not allowed" in low:
            category = VerdictCategory.NOT_ENTITLED
        elif "not found" in low and "namespace" in low:
            category = VerdictCategory.UNKNOWN_QUEUE
        elif "insufficient" in low:
            category = VerdictCategory.SHAPE_UNAVAILABLE
        elif "connection refused" in low or "unable to connect" in low or "timeout" in low:
            category = VerdictCategory.CONTROL_PLANE_DOWN
        return Verdict(
            queue=queue, account=account, allowed=False, category=category,
            reason=_first_error(text), raw=text.strip(),
        )

    def format_nodelist(self, names: Iterable[str]) -> str:
        return ",".join(sorted(names))


def _resource_triple(requests: dict) -> tuple[int, int, int]:
    """``(cpu, memory_mb, accelerators)`` from one resource-requests block."""
    gpu = 0
    for res in _GPU_RESOURCES:
        if res in requests:
            gpu = int(float(requests[res]))
            break
    return (
        _quantity_to_cpu(requests.get("cpu")),
        _quantity_to_mb(requests.get("memory")),
        gpu,
    )


def _pod_request(pod: dict) -> tuple[int, int, int]:
    """What the scheduler actually reserves for a pod.

    Kubernetes does **not** simply sum the containers.  The effective request
    per resource is::

        max( sum(regular) + sum(sidecars),
             max(plain init) + sum(sidecars) )

    plus ``spec.overhead``.  Init containers run before the regular ones, so
    the larger of the two phases is what has to fit -- and a sidecar (an init
    container with ``restartPolicy: Always``) runs alongside them, so it adds
    rather than competes.

    Summing only ``containers`` understates usage whenever an init container is
    the biggest thing in the pod, which reports a full node as free.  That is
    the unsafe direction, so it is worth getting right.
    """
    spec = pod.get("spec") or {}
    regular = [0, 0, 0]
    sidecars = [0, 0, 0]
    init_peak = [0, 0, 0]

    for c in spec.get("containers") or []:
        got = _resource_triple((c.get("resources") or {}).get("requests") or {})
        regular = [a + b for a, b in zip(regular, got, strict=True)]

    for c in spec.get("initContainers") or []:
        got = _resource_triple((c.get("resources") or {}).get("requests") or {})
        if c.get("restartPolicy") == "Always":
            sidecars = [a + b for a, b in zip(sidecars, got, strict=True)]
        else:
            init_peak = [max(a, b) for a, b in zip(init_peak, got, strict=True)]

    overhead = _resource_triple(spec.get("overhead") or {})
    return tuple(  # type: ignore[return-value]
        max(reg + side, peak + side) + over
        for reg, side, peak, over in zip(
            regular, sidecars, init_peak, overhead, strict=True
        )
    )


def _can_i_says_yes(out: str) -> bool:
    """Whether ``kubectl auth can-i`` actually granted the permission.

    The verdict is a bare ``yes`` or ``no`` on its own line -- but kubectl also
    prints ``no - <reason>`` when it can explain the denial, and may emit a
    warning line before the verdict.  Testing for a line equal to ``"no"``
    therefore reads an *explained refusal* as permission, which is the exact
    failure this tool exists to prevent.  So the verdict token is located
    positively: only an explicit ``yes`` counts.
    """
    for line in out.splitlines():
        tokens = line.strip().lower().split()
        if not tokens:
            continue
        if tokens[0] in {"yes", "no"}:
            return tokens[0] == "yes"
    return False


def _can_i_reason(out: str, err: str) -> str:
    """The explanation kubectl attached to a denial, if it gave one."""
    for line in (out + "\n" + err).splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("no -"):
            reason = stripped[4:].strip()
            # kubectl's own explanation often already begins "RBAC:"; adding
            # our own prefix unconditionally reads as a stutter.
            return reason if reason.lower().startswith("rbac") else f"RBAC: {reason}"
    return ""


def _first_error(text: str) -> str:
    for line in text.splitlines():
        if "error" in line.lower() or "forbidden" in line.lower():
            return line.strip().removeprefix("error: ").strip()
    return text.strip().splitlines()[-1] if text.strip() else "no output"
