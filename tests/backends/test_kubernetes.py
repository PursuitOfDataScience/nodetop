"""Kubernetes -- the same lies, in a completely different vocabulary."""

from __future__ import annotations

import json

import pytest

from nodetop.core.model import JobShape, VerdictCategory
from nodetop.runner import RecordedRunner


def _nodes(k8s_backend):
    return {n.name: n for n in k8s_backend.load_nodes()}


class TestNodes:
    def test_all_nodes(self, k8s_backend):
        assert len(k8s_backend.load_nodes()) == 6

    def test_allocatable_is_capacity(self, k8s_backend):
        n = _nodes(k8s_backend)["gpu-node-1"]
        assert n.cpus_total == 48
        assert n.gpus_total == 4
        assert n.memory_mb == 340 * 1024

    def test_occupancy_comes_from_pod_requests(self, k8s_backend):
        # allocatable is a capacity, not an occupancy; reading it as free
        # space overstates availability on every busy node.
        n = _nodes(k8s_backend)["gpu-node-1"]
        assert n.cpus_alloc == 12
        assert n.gpus_alloc == 2
        assert n.gpus_free == 2

    def test_terminal_pods_do_not_count_against_capacity(self, k8s_backend):
        # A Succeeded pod on gpu-node-1 requested 2 more GPUs; counting it
        # would report the node as full.
        assert _nodes(k8s_backend)["gpu-node-1"].gpus_free == 2

    def test_millicpu_requests_round_down(self, k8s_backend):
        assert _nodes(k8s_backend)["cpu-node-1"].cpus_alloc == 0

    def test_accelerator_model_from_the_product_label(self, k8s_backend):
        assert _nodes(k8s_backend)["gpu-node-1"].accelerator.model == "A100"
        assert _nodes(k8s_backend)["gpu-node-2"].accelerator.model == "V100"
        assert _nodes(k8s_backend)["gpu-node-3"].accelerator.model == "H100"


class TestPhantomCapacityK8s:
    def test_a_cordoned_node_is_ready_capable_and_unusable(self, k8s_backend):
        # kubectl shows Ready,SchedulingDisabled with every capacity number
        # intact -- the Kubernetes form of an advertised-but-dead resource.
        n = _nodes(k8s_backend)["gpu-node-3"]
        assert n.gpus_total == 8
        assert n.schedulable is False
        assert "cordoned" in n.reason

    def test_notready_node(self, k8s_backend):
        n = _nodes(k8s_backend)["cpu-node-2"]
        assert n.schedulable is False
        assert n.unreachable is True
        assert "Kubelet" in n.reason

    def test_a_taint_is_recorded_not_silently_ignored(self, k8s_backend):
        n = _nodes(k8s_backend)["gpu-node-4"]
        assert n.schedulable is True          # nothing wrong with the node
        assert n.taints == ("dedicated=inference:NoSchedule",)

    def test_an_untolerated_taint_excludes_the_node_from_a_fit(self, k8s_backend):
        from nodetop.core.capacity import hardware_ok

        n = _nodes(k8s_backend)["gpu-node-4"]
        assert hardware_ok(n, JobShape(gpus_per_node=8))[0] is False
        tolerant = JobShape(gpus_per_node=8,
                            tolerates=("dedicated=inference:NoSchedule",))
        assert hardware_ok(n, tolerant)[0] is True


class TestNamespacesAsQueues:
    def _queues(self, k8s_backend):
        return {q.name: q for q in k8s_backend.load_queues()}

    def test_namespaces_become_queues(self, k8s_backend):
        assert set(self._queues(k8s_backend)) == {
            "default", "research", "kube-system", "retiring"
        }

    def test_terminating_namespace_accepts_nothing(self, k8s_backend):
        q = self._queues(k8s_backend)["retiring"]
        assert q.usable is False
        assert "QUEUE_DISABLED" in {b.code for b in q.structural_blockers()}

    def test_system_namespaces_are_hidden_not_removed(self, k8s_backend):
        assert self._queues(k8s_backend)["kube-system"].hidden is True

    def test_every_namespace_can_target_every_node(self, k8s_backend):
        # Taints and selectors do the filtering, per node, in the capacity pass.
        assert len(self._queues(k8s_backend)["research"].node_names) == 6


class TestResourceQuota:
    def test_quota_becomes_a_ceiling(self, k8s_backend):
        limits = k8s_backend.load_limits()["research"]
        assert limits.per_job["cpu"] == 64
        assert limits.per_job["mem_mb"] == 512 * 1024

    def test_the_tightest_of_several_quotas_wins(self, k8s_backend):
        # research has one quota at 4 GPUs and another at 2; only the tighter
        # one can actually be satisfied.
        assert k8s_backend.load_limits()["research"].per_job["gpu"] == 2

    def test_a_breach_is_predicted_from_the_shape(self, k8s_backend):
        limits = k8s_backend.load_limits()["research"]
        got = limits.blockers(JobShape(gpus_per_node=4))
        assert "MAX_GPU_JOB" in {b.code for b in got}


@pytest.fixture(autouse=True)
def _pretend_kubectl_exists(monkeypatch):
    """Satisfy the probe's client-presence gate.

    ``probe()`` returns None when ``kubectl`` is not on PATH -- deliberately,
    so a missing client is not reported as a control-plane outage. These tests
    run on hosts without kubectl and exercise what happens *after* that gate,
    with a RecordedRunner standing in for the binary, so the gate has to be
    satisfied rather than bypassed. ``TestProbeIsGatedOnItsClient`` in
    tests/backends/test_registry.py covers the gate itself.
    """
    monkeypatch.setattr("nodetop.backends.kubernetes.which", lambda _cmd: True)


class TestProbe:
    def _backend(self, responses):
        from nodetop.backends.kubernetes import KubernetesBackend

        return KubernetesBackend(RecordedRunner(responses))

    def test_rbac_refusal_short_circuits(self):
        b = self._backend({"auth can-i": (1, "no\n", "")})
        v = b.probe("research", JobShape())
        assert v.allowed is False
        assert v.category == VerdictCategory.NOT_ENTITLED

    def test_dry_run_is_server_side_and_hard_coded(self):
        runner = RecordedRunner({
            "auth can-i": (0, "yes\n", ""),
            "kubectl run": (0, "pod/probe\n", ""),
        })
        from nodetop.backends.kubernetes import KubernetesBackend

        KubernetesBackend(runner).probe("research", JobShape(gpus_per_node=1))
        run_cmd = " ".join(runner.calls[-1])
        # Server-side dry run creates nothing but does evaluate admission.
        assert "--dry-run=server" in run_cmd

    def test_quota_breach_is_caught_by_admission(self):
        # Unlike every HPC scheduler here, k8s catches this BEFORE submission.
        b = self._backend({
            "auth can-i": (0, "yes\n", ""),
            "kubectl run": (
                1, "",
                'Error from server (Forbidden): pods "p" is forbidden: exceeded quota: '
                "gpu-quota, requested: requests.nvidia.com/gpu=8, used: 0, limited: 2",
            ),
        })
        v = b.probe("research", JobShape(gpus_per_node=8))
        assert v.allowed is False
        assert v.category == VerdictCategory.QUOTA_EXCEEDED

    def test_success(self):
        b = self._backend({
            "auth can-i": (0, "yes\n", ""),
            "kubectl run": (0, "pod/nodetop-probe\n", ""),
        })
        v = b.probe("research", JobShape(gpus_per_node=1))
        assert v.allowed is True
        assert v.category == VerdictCategory.OK

    def test_unreachable_api_server(self):
        b = self._backend({
            "auth can-i": (0, "yes\n", ""),
            "kubectl run": (1, "", "Unable to connect to the server: dial tcp timeout"),
        })
        v = b.probe("research", JobShape())
        assert v.category == VerdictCategory.CONTROL_PLANE_DOWN
        assert v.durable is False


def _memory_quantity(flags):
    """The ``memory=`` quantity out of a ``--requests=`` flag, or None."""
    for flag in flags:
        if flag.startswith("--requests="):
            for part in flag.split("=", 1)[1].split(","):
                key, _, value = part.partition("=")
                if key == "memory":
                    return value
    return None


class TestTheMemoryAskedForIsTheMemoryRequested:
    """A pod request must not be smaller than the shape it was computed from.

    ``--mem`` takes a fractional size, and this adapter built its quantity with
    ``int(shape.memory_gb)``, which threw the remainder away *downwards*::

        asked        emitted (before)   shape wants
        0.5 GiB      memory=0Gi         512 MiB
        1.5 GiB      memory=1Gi         1536 MiB
        3.7 GiB      memory=3Gi         3788 MiB

    The sub-gigabyte row is not a smaller request, it is a meaningless one. And
    these flags are not only pasted: ``probe()`` hands them to ``kubectl run
    --dry-run=server``, the one dry-run here that really evaluates a
    ResourceQuota -- so the quota question was being asked about a figure up to
    a gibibyte under the job.
    """

    #: Sizes a user can actually type. The sub-gigabyte pair is the one that
    #: used to produce a request for nothing.
    SIZES = [0.25, 0.5, 0.9, 1.0, 1.5, 2.0, 3.7, 15.5, 64.0, 500.75]

    @staticmethod
    def _mb(quantity):
        if quantity.endswith("Mi"):
            return int(quantity[:-2])
        if quantity.endswith("Gi"):
            return int(quantity[:-2]) * 1024
        raise AssertionError(f"unexpected memory quantity {quantity!r}")

    @pytest.mark.parametrize("gb", SIZES)
    def test_the_request_is_exact(self, gb):
        from nodetop.backends.kubernetes import KubernetesBackend

        shape = JobShape(cpus_per_task=2, memory_gb=gb)
        got = _memory_quantity(KubernetesBackend().submit_flags("research", shape))
        assert got is not None, f"no memory quantity emitted for {gb} GiB"
        assert self._mb(got) == shape.memory_mb_per_node, (
            f"asks for {got} where the shape is {shape.memory_mb_per_node} MiB"
        )

    def test_a_sub_gigabyte_request_is_not_zero(self):
        from nodetop.backends.kubernetes import KubernetesBackend

        flags = KubernetesBackend().submit_flags(
            "research", JobShape(memory_gb=0.5))
        assert _memory_quantity(flags) == "512Mi"

    def test_the_dry_run_asks_admission_about_the_real_figure(self):
        # The quota verdict is only worth having if it was obtained for the
        # shape that was assessed.
        runner = RecordedRunner({
            "auth can-i": (0, "yes\n", ""),
            "kubectl run": (0, "pod/probe\n", ""),
        })
        from nodetop.backends.kubernetes import KubernetesBackend

        KubernetesBackend(runner).probe("research", JobShape(memory_gb=1.5))
        sent = " ".join(runner.calls[-1])
        assert "memory=1536Mi" in sent, sent
        assert "memory=1Gi" not in sent, sent

    def test_it_agrees_with_the_other_adapters_on_the_number(self):
        # One shape, one figure, whatever each scheduler's spelling of it.
        from nodetop.backends.kubernetes import KubernetesBackend
        from nodetop.backends.slurm import SlurmBackend

        shape = JobShape(cpus_per_task=2, memory_gb=1.5)
        k8s = self._mb(
            _memory_quantity(KubernetesBackend().submit_flags("q", shape)))
        slurm = next(
            int(f.split("=")[1].rstrip("M"))
            for f in SlurmBackend().submit_flags("q", shape)
            if f.startswith("--mem=")
        )
        assert k8s == slurm == 1536


class TestTheZeroAndUnsetMemoryCasesAreUnchanged:
    """Controls: the fix must not invent a request, nor respell an exact one."""

    def test_no_memory_asked_means_no_memory_quantity(self):
        # An absent request must stay absent rather than become `memory=0Gi`.
        from nodetop.backends.kubernetes import KubernetesBackend

        flags = KubernetesBackend().submit_flags(
            "research", JobShape(cpus_per_task=4, memory_gb=0.0))
        assert _memory_quantity(flags) is None
        assert "memory" not in " ".join(flags)

    def test_a_whole_gibibyte_keeps_the_gi_spelling(self):
        # `memory=64Gi` is what someone about to paste this wants to read;
        # megabytes are for the figures gibibytes would lose.
        from nodetop.backends.kubernetes import KubernetesBackend

        flags = KubernetesBackend().submit_flags("research", JobShape(memory_gb=64))
        assert _memory_quantity(flags) == "64Gi"

    def test_requests_and_limits_still_carry_the_same_quantities(self):
        from nodetop.backends.kubernetes import KubernetesBackend

        flags = KubernetesBackend().submit_flags(
            "research", JobShape(cpus_per_task=4, gpus_per_node=2, memory_gb=1.5))
        requests = next(f for f in flags if f.startswith("--requests="))
        limits = next(f for f in flags if f.startswith("--limits="))
        assert requests.split("=", 1)[1] == limits.split("=", 1)[1]
        assert "cpu=4" in requests
        assert "nvidia.com/gpu=2" in requests


class TestIdentity:
    def test_whoami(self, k8s_backend):
        ident = k8s_backend.load_identity()
        assert ident.user == "alice"
        assert "dev" in ident.groups


class TestQuantities:
    def test_memory_units(self):
        from nodetop.backends.kubernetes import _quantity_to_mb

        assert _quantity_to_mb("1Gi") == 1024
        assert _quantity_to_mb("340Gi") == 340 * 1024
        assert _quantity_to_mb("1Ti") == 1024 * 1024
        assert _quantity_to_mb(None) == 0

    def test_cpu_units(self):
        from nodetop.backends.kubernetes import _quantity_to_cpu

        assert _quantity_to_cpu("48") == 48
        assert _quantity_to_cpu("500m") == 0
        assert _quantity_to_cpu("2500m") == 2


class TestAuthCanIParsing:
    """``kubectl auth can-i`` does not always answer with a bare word."""

    def _probe(self, can_out, can_rc=0, dry_rc=0, dry_err=""):
        from nodetop.backends.kubernetes import KubernetesBackend

        return KubernetesBackend(RecordedRunner({
            "auth can-i": (can_rc, can_out, ""),
            "kubectl run": (dry_rc, "pod/p\n", dry_err),
        })).probe("research", JobShape(gpus_per_node=1))

    def test_bare_yes_is_permission(self):
        assert self._probe("yes\n").allowed is True

    def test_bare_no_is_refusal(self):
        assert self._probe("no\n").allowed is False

    def test_an_explained_refusal_is_still_a_refusal(self):
        # kubectl prints "no - <reason>" when it can explain the denial.
        # Testing for a line equal to "no" reads that as permission granted,
        # which is the exact failure this tool exists to prevent.
        v = self._probe("no - RBAC: pods is forbidden\n")
        assert v.allowed is False
        assert v.category == VerdictCategory.NOT_ENTITLED

    def test_the_explanation_is_carried_into_the_reason(self):
        v = self._probe("no - RBAC: pods is forbidden\n")
        assert "forbidden" in v.reason
        assert not v.reason.lower().startswith("rbac: rbac")

    @pytest.mark.parametrize("prefix", [
        "W0822 12:00:00.000 config warning\n",
        "Warning: some deprecation notice\n",
    ])
    def test_a_warning_before_the_verdict_does_not_confuse_it(self, prefix):
        assert self._probe(prefix + "yes\n").allowed is True
        assert self._probe(prefix + "no\n").allowed is False

    def test_empty_output_is_not_permission(self):
        assert self._probe("\n").allowed is False

    def test_a_nonzero_exit_is_a_refusal_whatever_it_printed(self):
        assert self._probe("yes\n", can_rc=1).allowed is False


class TestDryRunExitCode:
    """The dry-run verdict comes from the exit code, not from grepping."""

    def _probe(self, dry_rc, dry_out="pod/p\n", dry_err=""):
        from nodetop.backends.kubernetes import KubernetesBackend

        return KubernetesBackend(RecordedRunner({
            "auth can-i": (0, "yes\n", ""),
            "kubectl run": (dry_rc, dry_out, dry_err),
        })).probe("research", JobShape(gpus_per_node=1))

    def test_a_warning_containing_error_does_not_fail_a_clean_dry_run(self):
        # kubectl writes deprecation and config warnings to stderr. Requiring
        # the word "error" to be absent turns a successful admission into a
        # failure whenever a warning happens to contain it.
        v = self._probe(0, dry_err="W0822 warning: error-reporting webhook is deprecated\n")
        assert v.allowed is True
        assert v.category == VerdictCategory.OK

    def test_a_nonzero_exit_is_still_a_refusal(self):
        assert self._probe(1, dry_err="Error from server: something\n").allowed is False


def _pod(containers=(), init=(), overhead=None):
    def c(cpu=None, mem=None, gpu=None, sidecar=False):
        req = {}
        if cpu:
            req["cpu"] = cpu
        if mem:
            req["memory"] = mem
        if gpu:
            req["nvidia.com/gpu"] = gpu
        out = {"resources": {"requests": req}}
        if sidecar:
            out["restartPolicy"] = "Always"
        return out

    spec = {
        "containers": [c(**k) for k in containers],
        "initContainers": [c(**k) for k in init],
    }
    if overhead:
        spec["overhead"] = overhead
    return {"spec": spec, "status": {"phase": "Running"}}


class TestPodRequest:
    """Kubernetes does not simply sum the containers."""

    def test_containers_are_summed(self):
        from nodetop.backends.kubernetes import _pod_request

        assert _pod_request(_pod([{"cpu": "2", "mem": "4Gi"}])) == (2, 4096, 0)

    def test_an_init_container_larger_than_the_containers_wins(self):
        # Summing only `containers` reports a full node as free, which is the
        # unsafe direction: init containers run first and have to fit.
        from nodetop.backends.kubernetes import _pod_request

        got = _pod_request(
            _pod([{"cpu": "2", "mem": "4Gi"}],
                 [{"cpu": "48", "mem": "200Gi", "gpu": "4"}])
        )
        assert got == (48, 204800, 4)

    def test_the_containers_win_when_they_are_larger(self):
        from nodetop.backends.kubernetes import _pod_request

        assert _pod_request(_pod([{"cpu": "48", "gpu": "4"}], [{"cpu": "2"}])) == (
            48, 0, 4
        )

    def test_a_sidecar_adds_rather_than_competes(self):
        # An init container with restartPolicy: Always runs alongside the
        # regular ones, so its request is additive (k8s 1.28+).
        from nodetop.backends.kubernetes import _pod_request

        assert _pod_request(
            _pod([{"cpu": "4"}], [{"cpu": "2", "sidecar": True}])
        ) == (6, 0, 0)

    def test_a_sidecar_is_added_to_the_init_peak_too(self):
        from nodetop.backends.kubernetes import _pod_request

        got = _pod_request(
            _pod([{"cpu": "4"}], [{"cpu": "2", "sidecar": True}, {"cpu": "10"}])
        )
        assert got == (12, 0, 0)

    def test_pod_overhead_is_included(self):
        from nodetop.backends.kubernetes import _pod_request

        assert _pod_request(
            _pod([{"cpu": "2"}], overhead={"cpu": "1", "memory": "1Gi"})
        ) == (3, 1024, 0)

    def test_an_empty_pod_spec_is_zero_not_a_crash(self):
        from nodetop.backends.kubernetes import _pod_request

        assert _pod_request({"spec": {}}) == (0, 0, 0)
        assert _pod_request({}) == (0, 0, 0)


class TestInitContainerOccupancy:
    def test_a_node_held_by_an_init_container_is_not_reported_free(self, k8s_backend):
        nodes = {n.name: n for n in k8s_backend.load_nodes()}
        n = nodes["gpu-node-4"]
        # The init container asks for all 8 accelerators, so nothing is free
        # even though the regular container asks for none.
        assert n.gpus_alloc == 8
        assert n.gpus_free == 0


class TestOtherVendorResources:
    def test_an_amd_accelerator_is_counted_and_identified(self):
        import json as _json

        from nodetop.backends.kubernetes import KubernetesBackend

        nodes = _json.dumps({"items": [{
            "metadata": {"name": "a1",
                         "labels": {"amd.com/gpu.product": "AMD-Instinct-MI300X"}},
            "spec": {},
            "status": {"allocatable": {"cpu": "96", "amd.com/gpu": "8"},
                       "conditions": [{"type": "Ready", "status": "True"}]}}]})
        n = KubernetesBackend(RecordedRunner({})).parse_nodes(nodes, "")[0]
        assert n.gpus_total == 8
        assert n.accelerator.model == "MI300X"
        assert n.accelerator.vendor == "AMD"


class TestDegenerateInput:
    def test_a_node_with_no_status_survives(self):
        import json as _json

        from nodetop.backends.kubernetes import KubernetesBackend

        nodes = _json.dumps({"items": [{"metadata": {"name": "e1"}, "spec": {},
                                        "status": {}}]})
        n = KubernetesBackend(RecordedRunner({})).parse_nodes(nodes, "")[0]
        assert (n.cpus_total, n.memory_mb, n.gpus_total) == (0, 0, 0)

    def test_a_pod_with_no_containers_key_survives(self):
        import json as _json

        from nodetop.backends.kubernetes import KubernetesBackend

        nodes = _json.dumps({"items": [{"metadata": {"name": "n1"}, "spec": {},
                                        "status": {"allocatable": {"cpu": "8"}}}]})
        pods = _json.dumps({"items": [{"metadata": {"name": "p"},
                                       "spec": {"nodeName": "n1"},
                                       "status": {"phase": "Running"}}]})
        n = KubernetesBackend(RecordedRunner({})).parse_nodes(nodes, pods)[0]
        assert n.cpus_alloc == 0


class TestProbeNeedsItsClient:
    def test_no_kubectl_means_no_verdict(self, monkeypatch):
        # Not a refusal and not an outage -- no answer at all, so the caller
        # falls back to the declared ACL instead of being told to retry.
        from nodetop.backends.kubernetes import KubernetesBackend

        monkeypatch.setattr("nodetop.backends.kubernetes.which", lambda _cmd: False)
        b = KubernetesBackend(RecordedRunner({"auth can-i": (0, "yes\n", "")}))
        assert b.probe("research", JobShape()) is None

    def test_it_does_not_even_run_the_command(self, monkeypatch):
        from nodetop.backends.kubernetes import KubernetesBackend

        monkeypatch.setattr("nodetop.backends.kubernetes.which", lambda _cmd: False)
        runner = RecordedRunner({"auth can-i": (0, "yes\n", "")})
        KubernetesBackend(runner).probe("research", JobShape())
        assert not runner.calls


class TestIdentityFailureIsNotAnEmptyIdentity:
    """A group-less Identity silently disables every RBAC group check.

    On Kubernetes the groups ARE the entitlement mechanism, and the membership
    test downstream is tri-state: an empty group set reads as "cannot tell", so
    no GROUP_NOT_ALLOWED is ever emitted and every restriction is ignored. A
    namespace the caller's groups do not permit was reported as available.

    `kubectl auth whoami` is absent before k8s 1.26 and can itself be refused by
    RBAC, so this fires in practice, not only in theory.
    """

    def test_a_failed_whoami_raises(self):
        from nodetop.backends.kubernetes import KubernetesBackend
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({"kubectl auth whoami -o json": (1, "", "unknown command")})
        with pytest.raises(CommandError):
            KubernetesBackend(runner).load_identity()

    def test_unparseable_output_raises_too(self):
        # A 0 exit with garbage on stdout is the same problem wearing a
        # different hat: json.loads throwing must not become an empty identity.
        from nodetop.backends.kubernetes import KubernetesBackend

        runner = RecordedRunner({"kubectl auth whoami -o json": (0, "not json", "")})
        with pytest.raises(json.JSONDecodeError):
            KubernetesBackend(runner).load_identity()

    def test_a_working_whoami_still_yields_groups(self):
        from nodetop.backends.kubernetes import KubernetesBackend

        payload = ('{"status": {"userInfo": {"username": "alice", '
                   '"groups": ["system:authenticated", "researchers"]}}}')
        runner = RecordedRunner({"kubectl auth whoami -o json": (0, payload, "")})
        ident = KubernetesBackend(runner).load_identity()
        assert ident.user == "alice"
        assert "researchers" in ident.groups


class TestQuotaFailureIsNotAbsenceOfQuota:
    """A ResourceQuota is the k8s form of "admitted, then never runs".

    Returning `{}` when the query fails is indistinguishable from a namespace
    with no quota configured, so the ceiling check is silently disabled at the
    one moment it cannot be performed. `Limits` carries an `unreadable` flag for
    exactly this distinction, and swallowing the error threw it away.
    """

    def test_a_failed_quota_query_raises(self):
        from nodetop.backends.kubernetes import KubernetesBackend
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({
            "kubectl get resourcequota --all-namespaces -o json":
                (1, "", "Error from server (Forbidden)"),
        })
        with pytest.raises(CommandError) as exc:
            KubernetesBackend(runner).load_limits()
        # And with the REAL cause, not a JSONDecodeError from parsing "". The
        # error text is what gets recorded against "limits" and shown to the
        # user, so "Expecting value: line 1 column 1" is not good enough.
        assert "Forbidden" in str(exc.value)

    def test_the_queue_listing_survives_a_forbidden_quota(self):
        # RBAC commonly allows namespaces and forbids resourcequota. Losing the
        # ceilings must not cost the whole listing -- only `load_limits` treats
        # this as fatal, because there the ceilings *are* the answer.
        from nodetop.backends.kubernetes import KubernetesBackend

        runner = RecordedRunner({
            "kubectl get namespaces -o json":
                (0, '{"items": [{"metadata": {"name": "team-a"}}]}', ""),
            "kubectl get resourcequota --all-namespaces -o json":
                (1, "", "Error from server (Forbidden)"),
            "kubectl get nodes -o json": (0, '{"items": []}', ""),
            "kubectl get pods --all-namespaces "
            "--field-selector=status.phase!=Succeeded -o json": (0, '{"items": []}', ""),
        })
        queues = KubernetesBackend(runner).load_queues()
        assert [q.name for q in queues] == ["team-a"]

    def test_a_namespace_with_no_quota_is_still_empty_not_an_error(self, k8s_backend):
        # The distinction being preserved: genuinely no quota is a fact, and
        # must not start raising just because failures now do.
        assert isinstance(k8s_backend.load_limits(), dict)


class TestMissingPodDataIsNotAnIdleCluster:
    """Pod requests ARE the occupancy on Kubernetes.

    `allocatable` is a capacity, not a free count, so with no pod data every
    node parses as zero-allocated. A node running 40 of its 48 cores and all 4
    of its accelerators was reported as 48/48 and 4/4 free, `idle=True` -- the
    phantom capacity this whole tool exists to catch, produced by the tool. The
    suppression that caused it also carried a comment claiming the failure was
    "recorded in Cluster.errors", which it was not: swallowing it made
    `load_nodes` succeed.

    `kubectl get pods --all-namespaces` is routinely forbidden by RBAC for a
    namespaced user, so this was reachable in ordinary use.
    """

    NODES = json.dumps({"items": [{
        "metadata": {"name": "busy"},
        "status": {"allocatable": {"cpu": "48", "memory": "340Gi",
                                   "nvidia.com/gpu": "4"},
                   "conditions": [{"type": "Ready", "status": "True"}]},
        "spec": {},
    }]})

    def test_a_forbidden_pod_list_raises(self):
        from nodetop.backends.kubernetes import KubernetesBackend
        from nodetop.exceptions import CommandError

        runner = RecordedRunner({
            "kubectl get nodes -o json": (0, self.NODES, ""),
            "kubectl get pods --all-namespaces "
            "--field-selector=status.phase!=Succeeded -o json":
                (1, "", "Error from server (Forbidden): pods is forbidden"),
        })
        with pytest.raises(CommandError):
            KubernetesBackend(runner).load_nodes()

    def test_the_cluster_records_it_instead_of_reporting_idle_nodes(self):
        from nodetop.backends.kubernetes import KubernetesBackend
        from nodetop.core.cluster import Cluster

        runner = RecordedRunner({
            "kubectl get nodes -o json": (0, self.NODES, ""),
            "kubectl get pods --all-namespaces "
            "--field-selector=status.phase!=Succeeded -o json":
                (1, "", "Error from server (Forbidden): pods is forbidden"),
        })
        cluster = Cluster.load(KubernetesBackend(runner))
        assert "nodes" in cluster.errors
        # And crucially: no node is presented as free.
        assert cluster.nodes == []

    def test_occupancy_is_still_read_when_pods_are_available(self):
        from nodetop.backends.kubernetes import KubernetesBackend

        pods = json.dumps({"items": [{
            "metadata": {"name": "p", "namespace": "x"},
            "spec": {"nodeName": "busy", "containers": [
                {"resources": {"requests": {"cpu": "40", "nvidia.com/gpu": "4"}}}]},
            "status": {"phase": "Running"},
        }]})
        runner = RecordedRunner({
            "kubectl get nodes -o json": (0, self.NODES, ""),
            "kubectl get pods --all-namespaces "
            "--field-selector=status.phase!=Succeeded -o json": (0, pods, ""),
        })
        node = KubernetesBackend(runner).load_nodes()[0]
        assert node.cpus_free == 8
        assert node.gpus_free == 0
        assert not node.idle
