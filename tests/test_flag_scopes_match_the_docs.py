"""Which commands take which flag, pinned against what the docs claim.

The audit found four claims that were false against `build_parser()`:

* README and DESIGN.md both said `-p` "is accepted everywhere as an alias for
  `-q`". It is on six commands; `status`, `zoom`, `health`, `backends` and
  `snapshot` take neither, because they do not name a single queue.
* DESIGN.md said `--all` widens "`status`, `queues` and `where`", omitting
  `zoom`, `nodes` and `accelerators`, which also have it.
* README listed `--all`/`--detail`/`--static` unqualified, and `--static`
  exists only on `status`.
* README's vocabulary omitted `pool` (ssh pool) and DESIGN.md's omitted
  `namespace` (Kubernetes) — each doc was missing the other's backend.

A prose claim about a flag's scope has nothing checking it, which is how all
four drifted. The table below is measured from the parser, so a flag gained or
lost reddens this file and the sentence has to be looked at.
"""

from __future__ import annotations

import pathlib

import pytest

from nodetop.cli import build_parser

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: flag -> exactly the commands whose parser accepts it. Measured, not asserted
#: from the docs: the docs are what this file checks.
SCOPES = {
    "-p": ["accelerators", "check", "exclude", "nodes", "queues", "where"],
    "-q": ["accelerators", "check", "exclude", "nodes", "queues", "where"],
    "--all": ["accelerators", "nodes", "queues", "status", "where", "zoom"],
    "--detail": ["queues"],
    "--static": ["status"],
}

COMMANDS = [
    "status", "queues", "zoom", "nodes", "health", "where",
    "check", "exclude", "accelerators", "backends", "snapshot",
]


def _commands_taking(flag: str) -> list[str]:
    parser = build_parser()
    subs = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
    return sorted(
        name
        for name in COMMANDS
        if any(flag in (action.option_strings or []) for action in subs.choices[name]._actions)
    )


class TestTheParserMatchesTheTable:
    @pytest.mark.parametrize("flag", sorted(SCOPES))
    def test_a_flag_is_on_exactly_the_commands_listed(self, flag: str) -> None:
        assert _commands_taking(flag) == SCOPES[flag], flag

    def test_the_table_covers_no_command_that_does_not_exist(self) -> None:
        # Control: keeps the table honest if a command is ever renamed, rather
        # than letting it silently describe nothing.
        for flag, names in SCOPES.items():
            assert set(names) <= set(COMMANDS), (flag, set(names) - set(COMMANDS))

    def test_the_commands_that_name_no_queue_take_neither_alias(self) -> None:
        # The specific claim that was false: "accepted everywhere".
        for name in ("status", "zoom", "health", "backends", "snapshot"):
            assert name not in _commands_taking("-p"), name
            assert name not in _commands_taking("-q"), name


class TestTheDocsDoNotOverclaim:
    @pytest.mark.parametrize("doc", ["README.md", "DESIGN.md"])
    def test_no_doc_says_the_queue_alias_is_everywhere(self, doc: str) -> None:
        text = (ROOT / doc).read_text()
        assert "accepted everywhere as an alias" not in text, doc

    @pytest.mark.parametrize("doc", ["README.md", "DESIGN.md"])
    def test_each_doc_names_every_backend_vocabulary_word(self, doc: str) -> None:
        # Each doc was missing the other's backend: README had no `pool`,
        # DESIGN.md no `namespace`.
        text = (ROOT / doc).read_text()
        for word in ("partition", "queue", "namespace", "pool"):
            assert word in text, (doc, word)

    def test_the_design_all_list_names_every_command_that_has_it(self) -> None:
        text = (ROOT / "DESIGN.md").read_text()
        sentence = next(
            line for line in text.splitlines() if line.startswith("`--all` widens")
        )
        for name in SCOPES["--all"]:
            assert f"`{name}`" in sentence, (name, sentence)


class TestZoomHelpIsBackendNeutral:
    def test_the_help_does_not_hardcode_one_schedulers_word(self) -> None:
        """The parser is built before a backend is detected.

        So the term genuinely cannot be interpolated -- which is why the fix is
        the neutral wording rather than `cluster.queue_term`. What must not
        happen is `zoom --help` saying "partition" on a Kubernetes cluster where
        every other surface says "namespace".
        """
        parser = build_parser()
        subs = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
        action = next(a for a in subs._get_subactions() if a.dest == "zoom")
        assert "queue/partition" in action.help, action.help
