"""One condition decides whether a path component names a wrapper.

`PbsBackend.wrapped_by` used to ask twice in a row:

    if other in path.lower().split("/"):
        return other  # pragma: no cover - subsumed below
    if any(c.lower() == other or (c.lower().startswith(other + "-")
           and c[len(other) + 1:len(other) + 2].isdigit())
           for c in path.split("/")):
        return other

Both halves of that marker were wrong. It was not subsumed -- it ran FIRST, so
it was the one that answered, and `test_portability_round1.py`'s
`test_only_a_whole_component_names_a_wrapper` drives it through the
`("slurm/bin", "slurm")` row, so the suite was already executing a line marked
as never executed. And the exclusion hid that the condition duplicated the
`c.lower() == other` disjunct below it: lowering before splitting on "/" and
lowering each component after are the same test.

Collapsed to the one `any(...)`, because two conditions that agree today are
what the marker was papering over. Behaviour is unchanged -- every existing row
still passes -- so the neuter for this is a **source** matter: re-adding the
duplicate reddens the pin here and changes no output, which is exactly the
property that made the marker plausible in the first place.

The docstring above `wrapped_by` names the layout this decides:
`slurm-torque` RPMs put the shims in a plain prefix, "which is the ordinary case
on any RPM/DEB-packaged Slurm cluster rather than an exotic one".
"""

from __future__ import annotations

import pathlib

import pytest

from nodetop.backends import pbs
from nodetop.backends.pbs import PbsBackend

CLIENTS = ("qstat", "qsub", "pbsnodes", "qdel")


@pytest.fixture
def fake_path(monkeypatch):
    """Install a PATH where the given clients resolve under `prefix`."""

    def install(prefix: str, present=CLIENTS, pbs_only=()):
        monkeypatch.setattr(
            pbs, "which",
            lambda b: f"/{prefix}/bin/{b}" if b in pbs_only else None,
        )
        monkeypatch.setattr(
            pbs, "resolve",
            lambda b: f"/{prefix}/bin/{b}" if b in present else None,
        )

    return install


class TestTheComponentCheckAnswers:
    def test_a_plain_install_prefix_is_recognised(self, fake_path) -> None:
        """`/opt/slurm/bin/qstat` -- the layout the docstring calls ordinary,
        and the one the excluded line answered for."""
        fake_path("opt/slurm")
        assert PbsBackend.wrapped_by() == "slurm"

    @pytest.mark.parametrize("scheduler", ["slurm", "lsf", "sge", "gridengine"])
    def test_every_scheduler_name_is_checked_the_same_way(
        self, fake_path, scheduler: str
    ) -> None:
        fake_path(f"opt/{scheduler}")
        assert PbsBackend.wrapped_by() == scheduler

    def test_the_component_match_is_case_insensitive(self, fake_path) -> None:
        """The condition lowercases, which is the half the two spellings shared."""
        fake_path("opt/SLURM")
        assert PbsBackend.wrapped_by() == "slurm"

    def test_only_one_condition_decides_it(self) -> None:
        """The source pin. Two conditions that agree today is the state the
        marker was hiding, so it must not come back.

        Scanned with COMMENTS STRIPPED, because the comment left at the fix site
        quotes the condition it removed -- a pin that reads raw source trips on
        the note explaining the fix, which is the same trap as a docstring that
        quotes a format spec.
        """
        code = "\n".join(
            ln for ln in pathlib.Path(str(pbs.__file__)).read_text().splitlines()
            if not ln.strip().startswith("#")
        )
        assert 'other in path.lower().split("/")' not in code
        assert code.count("c.lower() == other") == 1
        assert "no cover - subsumed below" not in code

    def test_no_coverage_exclusion_sits_on_the_component_check(self) -> None:
        """A line the suite executes must not be marked as unexecuted."""
        source = pathlib.Path(str(pbs.__file__)).read_text().splitlines()
        index = next(i for i, ln in enumerate(source) if "c.lower() == other" in ln)
        window = source[index - 2:index + 3]
        assert not [ln for ln in window if "no cover" in ln], window


class TestControls:
    """Each passes with the duplicate condition restored as well as removed.

    Behaviour is what they cover, and this change was not supposed to move any
    -- verified by running the neuter.
    """

    def test_a_versioned_prefix_is_still_recognised(self, fake_path) -> None:
        """The second disjunct, which the collapse kept."""
        fake_path("software/slurm-23.02-el7-x86_64")
        assert PbsBackend.wrapped_by() == "slurm"

    def test_a_substring_is_still_not_a_component(self, fake_path) -> None:
        fake_path("opt/slurmy-site/pbs")
        assert PbsBackend.wrapped_by() is None

    @pytest.mark.parametrize("prefix", ["data/slurm-logs/pbs", "var/slurm-data/pbs"])
    def test_an_unversioned_dashed_name_is_still_not_a_prefix(
        self, fake_path, prefix: str
    ) -> None:
        """The digit narrowing, which a genuine client-only PBS depends on."""
        fake_path(prefix)
        assert PbsBackend.wrapped_by() is None

    def test_a_real_pbs_still_short_circuits(self, fake_path) -> None:
        """`_PBS_ONLY` returns before any path logic runs."""
        fake_path("opt/slurm", present=CLIENTS, pbs_only=("qmgr",))
        assert PbsBackend.wrapped_by() is None

    def test_no_client_at_all_is_not_a_wrapper(self, fake_path) -> None:
        fake_path("opt/slurm", present=())
        assert PbsBackend.wrapped_by() is None
