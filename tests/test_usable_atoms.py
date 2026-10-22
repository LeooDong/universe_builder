"""One usable-atom predicate (`store.usable_text`, with
`store.is_placeholder` beside it), and corpus walks that survive a file
vanishing between the listing and the read (`store.iter_payloads`)."""

from __future__ import annotations

import json

from universe_builder import lint, store, topics, work


def _write(path, text, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.write_bytes if binary else path.write_text)(text)
    return path


class TestUsableText:
    def test_the_four_kinds_of_file(self, workspace):
        atom = _write(workspace / "a.json", '{"name": {"common": "A"}}')
        empty = _write(workspace / "e.json", "")
        placeholder = _write(workspace / "p.json", '{"error": "No data found"}')
        binary = _write(workspace / "b.json", b"\xff\xfe\x00", binary=True)
        heard = []
        assert store.usable_text(atom) == '{"name": {"common": "A"}}'
        assert store.usable_text(empty) == ""
        assert store.usable_text(placeholder) is None
        assert store.usable_text(binary, unreadable=lambda p, e: heard.append(p.name)) is None
        assert store.usable_text(workspace / "missing.json") is None
        assert heard == ["b.json"]
        assert [store.is_placeholder(p) for p in (atom, empty, placeholder, binary)] == \
            [False, False, True, False]

    def test_the_three_old_callers_agree(self, workspace):
        real = _write(store.path_for("city", None, "Memphis", "m"), json.dumps({"name": {"common": "M"}}))
        placeholder = _write(store.path_for("city", None, "Thebes", "m"), '{"error": "x"}')
        assert topics.has_real_atom("city", "Memphis") and not topics.has_real_atom("city", "Thebes")
        assert work._usable_text(real) is not None and work._usable_text(placeholder) is None
        assert topics._is_placeholder(placeholder) and not topics._is_placeholder(real)


class TestAVanishingFile:
    def _corpus(self):
        for name in ("Memphis", "Thebes"):
            _write(store.path_for("city", None, name, "m"),
                   json.dumps({"name": {"common": name, "otherNames": []}}))

    def _gone(self, monkeypatch, victim):
        real = store.read

        def read(path):
            if victim in str(path):
                raise FileNotFoundError(path)
            return real(path)
        monkeypatch.setattr(store, "read", read)

    def test_iter_payloads_skips_it(self, workspace, monkeypatch):
        self._corpus()
        self._gone(monkeypatch, "thebes")
        assert [p["name"]["common"] for _, p in store.iter_payloads("city")] == ["Memphis"]

    def test_lint_finishes_without_it(self, workspace, monkeypatch):
        self._corpus()
        self._gone(monkeypatch, "thebes")
        report = lint.run(atom_types=["city"])
        assert not report.errors, report.errors
