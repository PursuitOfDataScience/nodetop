"""Backend registry, detection order, and the honesty contract."""

from __future__ import annotations

import pytest

from nodetop import backends
from nodetop.core.model import BackendCapabilities


class TestRegistry:
    def test_every_backend_is_registered(self):
        assert set(backends.names()) == {
            "slurm", "pbs", "lsf", "sge", "kubernetes", "sshpool"
        }

    def test_detection_order_puts_the_universal_fallback_last(self):
        # sshpool always detects, so anything after it would be unreachable.
        assert backends.names()[-1] == "sshpool"

    def test_get_by_name(self):
        assert backends.get("slurm").name == "slurm"

    def test_unknown_name_raises_with_the_options_listed(self):
        with pytest.raises(KeyError, match="unknown backend"):
            backends.get("torque-classic")

    def test_the_fallback_always_detects(self):
        from nodetop.backends.sshpool import SshPoolBackend

        assert SshPoolBackend.detect() is True

    def test_detect_never_fails(self):
        # sshpool guarantees a result, so autodetection cannot come up empty.
        assert backends.detect() is not None


class TestHonestyContract:
    """Every backend must state what it cannot establish."""

    @pytest.mark.parametrize("name", ["slurm", "pbs", "lsf", "sge", "kubernetes", "sshpool"])
    def test_capabilities_are_declared(self, name):
        caps = backends.get(name).capabilities()
        assert isinstance(caps, BackendCapabilities)

    @pytest.mark.parametrize("name,can_probe", [
        # The three systems with a real verify-only mode...
        ("sge", True),
        ("kubernetes", True),
        # ...and the three without.
        ("pbs", False),
        ("lsf", False),
        ("sshpool", False),
    ])
    def test_probe_support_matches_reality(self, name, can_probe):
        caps = backends.get(name).capabilities()
        # SGE's probe is gated on qsub being present, so only assert the
        # negative claims and the positive one that needs no binary.
        if not can_probe:
            assert caps.probe is False

    @pytest.mark.parametrize("name", ["pbs", "lsf", "sshpool"])
    def test_backends_without_a_probe_say_why(self, name):
        # Silence would let a declared ACL read as a verified entitlement.
        caps = backends.get(name).capabilities()
        assert caps.notes, f"{name} must explain that entitlement is unconfirmed"
        assert any(
            "declared" in n.lower() or "no scheduler" in n.lower() for n in caps.notes
        )

    @pytest.mark.parametrize("name", ["pbs", "lsf", "sshpool"])
    def test_backends_without_a_probe_return_none(self, name):
        from nodetop.core.model import JobShape

        assert backends.get(name).probe("q", JobShape()) is None

    @pytest.mark.parametrize("name", ["slurm", "pbs", "lsf", "sge", "kubernetes", "sshpool"])
    def test_queue_term_is_declared(self, name):
        # The vocabulary adapts even though the reasoning does not.
        assert backends.get(name).queue_term

    def test_queue_terms_match_each_system(self):
        terms = {n: backends.get(n).queue_term for n in backends.names()}
        assert terms["slurm"] == "partition"
        assert terms["kubernetes"] == "namespace"
        assert terms["sshpool"] == "pool"


class TestProbeIsGatedOnItsClient:
    """A backend whose dry-run needs a client must return ``None`` without it.

    Trying anyway makes the client's absence look like a control-plane outage:
    the resulting CONTROL_PLANE_DOWN verdict is in ``TRANSIENT_CATEGORIES``, so
    the report invites a retry for a condition no amount of waiting fixes, and
    blames the cluster for a local problem.

    Slurm and SGE both had this guard with the reasoning written down; the
    Kubernetes probe did not, which is why this is enforced across the registry
    rather than per backend.
    """

    @staticmethod
    def _shape():
        from nodetop.core.model import JobShape

        return JobShape(nodes=1, cpus_per_task=1)

    def _backends_with_a_dry_run(self):
        from nodetop import backends as registry

        out = []
        for name in registry.names():
            caps = registry.get(name).capabilities()
            if caps.probe_supported:
                out.append(name)
        return out

    def test_at_least_one_backend_offers_a_dry_run(self):
        # Guards against the parametrisation silently covering nothing.
        assert self._backends_with_a_dry_run()

    def test_probe_returns_none_when_the_client_is_missing(self, monkeypatch):
        from nodetop import backends as registry

        for name in self._backends_with_a_dry_run():
            module = type(registry.get(name)).__module__
            monkeypatch.setattr(f"{module}.which", lambda _cmd: False)
            backend = type(registry.get(name))()
            assert backend.capabilities().probe is False, name
            assert backend.probe("q", self._shape()) is None, (
                f"{name}: probe ran without its client")

    def test_probe_supported_stays_true_without_the_client(self, monkeypatch):
        # The system's capability does not depend on this host -- that split is
        # the whole point, and the reference table reads it.
        from nodetop import backends as registry

        for name in self._backends_with_a_dry_run():
            module = type(registry.get(name)).__module__
            monkeypatch.setattr(f"{module}.which", lambda _cmd: False)
            assert type(registry.get(name))().capabilities().probe_supported, name

    def test_a_backend_with_no_dry_run_declares_neither(self):
        from nodetop import backends as registry

        for name in registry.names():
            caps = registry.get(name).capabilities()
            if not caps.probe_supported:
                assert caps.probe is False, (
                    f"{name}: usable-here is true while the system offers no dry-run")


class TestCapabilityNotesStayOneLine:
    """A note is a fact, not a paragraph.

    Every one of these appears in `nodetop check`'s output under a heading that
    already says what they are, so the explanation of the fact is redundant
    with the heading. They were 100-208 characters -- two to three wrapped
    lines each -- which is what made that footer the last block of prose in any
    command's default output.

    The cap is generous on purpose: this guards against paragraphs, not against
    a clause.
    """

    LIMIT = 100

    def _notes(self):
        from nodetop import backends as registry

        return [(name, note) for name in registry.names()
                for note in registry.get(name).capabilities().notes]

    def test_there_are_notes_to_check(self):
        assert len(self._notes()) >= 8

    def test_every_note_fits_one_line(self):
        long = [(n, len(t), t[:50]) for n, t in self._notes() if len(t) > self.LIMIT]
        assert not long, f"notes over {self.LIMIT} chars: {long}"

    def test_no_note_is_two_sentences(self):
        # A second sentence is almost always the explanation of the first, and
        # the heading above them already supplies that.
        import re

        multi = [(n, t) for n, t in self._notes()
                 if len(re.findall(r"[.!?] +[A-Z]", t)) > 0]
        assert not multi, f"notes with more than one sentence: {multi}"

    def test_none_of_them_shout_the_backend_name_back(self):
        # "PBS has no verify-only submission mode" -- printed under a heading
        # that already names the backend.
        from nodetop import backends as registry

        for name in registry.names():
            for note in registry.get(name).capabilities().notes:
                assert not note.lower().startswith(name.lower()), (name, note)
