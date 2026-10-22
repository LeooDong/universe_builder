"""Domain operations are CLI verbs the pack declares, not verbs core knows.

`cli.py` once built a `periods` subparser by name, with its flags,
and `cmd_periods` was the only dispatcher: a second domain verb (the history
pack's `media` and `signals` sidecars) had nowhere to go. Now the parser gets
one subparser per `DomainPack.operations` entry, the pack declares the flags
through `OperationSpec.add_arguments`, and core carries no operation name.
"""

from __future__ import annotations

import argparse

import pytest

from universe_builder import cli, journal, registry, store
from universe_builder.cli import build_parser, main
from universe_builder.providers.types import Usage
from universe_builder.registry import DomainPack, OperationSpec, active_pack


def _with_operations(monkeypatch, operations):
    """The active pack, with `operations` swapped in. Frozen dataclass, so a
    replaced copy is activated rather than the field being assigned."""
    base = active_pack()
    swapped = DomainPack(**{**base.__dict__, "operations": operations})
    monkeypatch.setattr(registry, "_active", swapped)
    monkeypatch.setattr(cli, "active_pack", lambda: swapped)
    return swapped


class TestOperationSpecs:
    def test_a_bare_callable_is_promoted_with_no_flags(self):
        def handler(args):
            """One-line help from the docstring."""
            return 0

        pack = DomainPack(**{**active_pack().__dict__, "operations": {"thing": handler}})
        spec = pack.operation_specs()["thing"]
        assert spec.handler is handler
        assert spec.help == "One-line help from the docstring."
        assert spec.add_arguments is None

    def test_something_else_is_refused(self):
        pack = DomainPack(**{**active_pack().__dict__, "operations": {"thing": 42}})
        with pytest.raises(TypeError, match="thing"):
            pack.operation_specs()


class TestParser:
    def test_each_operation_gets_a_subparser_with_its_own_flags(self):
        def add(parser):
            parser.add_argument("--refresh", action="store_true")
            parser.add_argument("--limit", type=int)

        ops = {"media": OperationSpec(handler=lambda a: 0, help="fetch photos", add_arguments=add)}
        args = build_parser("x", ops).parse_args(["media", "--refresh", "--limit", "3"])
        assert (args.command, args.refresh, args.limit) == ("media", True, 3)

    def test_an_operation_that_declares_no_flags_takes_none(self):
        ops = {"signals": OperationSpec(handler=lambda a: 0)}
        parser = build_parser("x", ops)
        assert parser.parse_args(["signals"]).command == "signals"
        with pytest.raises(SystemExit):
            parser.parse_args(["signals", "--refresh"])

    def test_a_name_that_shadows_a_core_verb_is_refused(self):
        with pytest.raises(ValueError, match="'status'"):
            build_parser("x", {"status": OperationSpec(handler=lambda a: 0)})

    def test_core_knows_no_operation_by_name(self):
        """No `periods` unless a pack declares one: the parser built with no
        operations has only the core verbs."""
        with pytest.raises(SystemExit):
            build_parser("x", {}).parse_args(["periods"])
        assert "periods" not in cli.COMMANDS


class TestDispatch:
    def test_main_runs_the_declared_handler_with_its_flags(self, workspace, monkeypatch):
        seen = []

        def add(parser):
            parser.add_argument("--dry-run", action="store_true")

        def handler(args):
            seen.append(args.dry_run)
            return 7

        _with_operations(monkeypatch, {"media": OperationSpec(handler=handler, add_arguments=add)})
        assert main(["media", "--dry-run"]) == 7
        assert seen == [True]

    def test_the_history_pack_still_answers_to_periods(self, workspace):
        """The flags moved from core into the pack; the verb parses as before."""
        args = build_parser("x", active_pack().operation_specs()).parse_args(
            ["periods", "--split", "--model", "gpt-5-nano"])
        assert (args.split, args.apply_labels, args.model) == (True, False, "gpt-5-nano")

    def test_a_pack_without_the_operation_has_no_such_verb(self, workspace, monkeypatch):
        _with_operations(monkeypatch, {})
        with pytest.raises(SystemExit) as exc:
            main(["periods"])
        assert exc.value.code == 2

    def test_cmd_operation_refuses_cleanly_by_name(self, workspace, monkeypatch):
        _with_operations(monkeypatch, {})
        assert cli.cmd_operation("periods", argparse.Namespace()) == 2
        assert cli.cmd_periods(argparse.Namespace()) == 2


class TestWriteBytes:
    def test_bytes_land_atomically_with_parents_made(self, tmp_path):
        target = tmp_path / "_media" / "files" / "person" / "h1.jpg"
        assert store.write_bytes(target, b"\xff\xd8\xff") == target
        assert target.read_bytes() == b"\xff\xd8\xff"
        assert not list(target.parent.glob(".tmp_*")), "no temp debris"

    def test_a_failed_write_leaves_no_debris(self, tmp_path, monkeypatch):
        import os

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        target = tmp_path / "h1.jpg"
        with pytest.raises(OSError):
            store.write_bytes(target, b"x")
        assert not target.exists() and not list(tmp_path.glob(".tmp_*"))


class TestUsageFields:
    def test_measured_usage_passes_all_four_counts(self):
        fields = journal.usage_fields(Usage(10, 20, 30, 40))
        assert fields == {"input_tokens": 10, "output_tokens": 20,
                          "cache_read_tokens": 30, "cache_write_tokens": 40}

    def test_nothing_measured_means_no_keys_not_zeros(self):
        """Zeros would tell `costs` the call was free; absence says unknown."""
        assert journal.usage_fields(Usage()) == {}
        assert journal.usage_fields(None) == {}

    def test_zero_counts_are_dropped_individually(self):
        assert journal.usage_fields(Usage(input_tokens=5)) == {"input_tokens": 5}
