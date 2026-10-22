"""The business logic `cli.py` used to hold, at its new homes (since v0.2.0):
`lint.identity_decider`, `sources.run`, `duplicates.triage_scope` /
`verdict_counts` / `save_verdicts`, `collisions.load_seed` and
`pipeline.modern_phase`. The CLI characterisation tests pin what the verbs
print; these pin the functions on their own."""

from __future__ import annotations

import json

import pytest

from universe_builder import collisions, duplicates, lint, pipeline, sources, store


class _Identity:
    """An identity source: Memphis and Men-nefer share a page; Thebes is unknown."""

    def __init__(self):
        self.asked = []

    def resolve_titles(self, names):
        self.asked.append(("titles", list(names)))
        pages = {"Memphis": (1, "Memphis", False), "Men-nefer": (1, "Memphis", True),
                 "Karnak": (2, "Karnak", False)}
        return {n: pages[n] for n in names if n in pages}

    def subject_leads(self, subjects):
        self.asked.append(("leads", list(subjects)))
        return {"Karnak": "Karnak is a temple complex near Luxor."}

    def identity_verdict(self, subject, alias, resolved):
        a, b = resolved.get(subject), resolved.get(alias)
        if a and b:
            return ("same" if a[0] == b[0] else "wrong"), "redirects"
        return None, "unresolved"


def _finding(subject, value, file="f.json"):
    return lint.Finding(check="overbroad-alias", atom_type="city", file=file,
                        subject=subject, detail="", value=value)


class TestIdentityDecider:
    def test_asks_once_says_what_it_does_and_decides(self):
        report = lint.Report(findings=[_finding("Memphis", "Men-nefer"),
                                       _finding("Karnak", "Thebes", file="k.json")])
        identity, said = _Identity(), []
        decide = lint.identity_decider(report, identity, label="src", say=said.append)

        assert said == ["resolving 4 name(s) through src...",
                        "fetching 2 article opening(s) as evidence..."]
        assert identity.asked == [("titles", ["Karnak", "Memphis", "Men-nefer", "Thebes"]),
                                  ("leads", ["Karnak", "Memphis"])]
        assert decide(report.findings[0]) == ("same", "redirects")
        verdict, why = decide(report.findings[1])
        assert verdict is None
        assert why == "unresolved | Karnak opens: Karnak is a temple complex near Luxor."


class TestSourcesRun:
    MODEL = "gpt-5-nano"

    def _atom(self, subject, payload=None):
        path = store.path_for("city", None, subject, self.MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload or {"name": {"common": subject}}), encoding="utf-8")
        return path

    @pytest.fixture
    def resolving(self, monkeypatch):
        monkeypatch.setattr(sources, "resolves", lambda u, timeout=15: (True, "HTTP 200"))

    def test_reports_what_it_examined_and_would_add(self, workspace, resolving):
        self._atom("Memphis")
        self._atom("Thebes")
        seen = []
        report = sources.run(["city"], lambda s, t: [f"https://example.org/{s}"],
                             on_proposal=seen.append)
        assert (report.examined, report.updated, report.urls) == (2, 0, 0)
        assert [p.subject for p in report.proposals] == [p.subject for p in seen]
        assert report.prefetched is None

    def test_write_appends_and_counts(self, workspace, resolving, monkeypatch):
        monkeypatch.setattr(sources, "_validate", lambda atom_type, payload: None)
        path = self._atom("Memphis")
        report = sources.run(["city"], lambda s, t: ["https://example.org/m"], write=True)
        assert (report.examined, report.updated, report.urls) == (1, 1, 1)
        assert json.loads(path.read_text())["name"]["sources"] == ["https://example.org/m"]

    def test_limit_missing_and_prefetch(self, workspace, resolving):
        self._atom("Memphis", {"name": {"common": "Memphis", "sources": ["https://x.org/a"]}})
        self._atom("Thebes")
        self._atom("Abydos")

        def ask(subject, atom_type):
            return []
        ask.prefetch = lambda subjects: len(subjects) - 1
        heard = []
        report = sources.run(["city"], ask, limit=1, missing=True,
                             on_prefetch=lambda m, n: heard.append((m, n)))
        assert report.examined == 1
        assert heard == [(0, 1)] and report.prefetched == (0, 1)


class TestTriageBookkeeping:
    PLAN = duplicates.Plan(undecided=[
        {"atom_type": "person", "name": "a", "files": ["1", "2", "3"], "why": ""},
        {"atom_type": "person", "name": "b", "files": ["4", "5"], "why": ""},
    ])

    def test_scope_counts_components_and_pairs(self):
        assert duplicates.triage_scope(self.PLAN) == (2, 4)
        assert duplicates.triage_scope(self.PLAN, 1) == (1, 3)

    def test_counts_and_saves(self, tmp_path):
        records = [{"verdict": "same"}, {"verdict": "different"}, {"verdict": "same"}]
        assert duplicates.verdict_counts(records) == {"same": 2, "different": 1}
        out = tmp_path / "v.json"
        duplicates.save_verdicts([{"a": "Ré", "verdict": "same"}], out)
        assert out.read_text(encoding="utf-8") == '[\n  {\n    "a": "Ré",\n    "verdict": "same"\n  }\n]'


class TestLoadSeed:
    def test_by_content_id_then_by_topic(self, workspace):
        path = store.path_for("history", None, "Anglo-Saxon History", "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cities": ["York"]}), encoding="utf-8")
        assert collisions.load_seed("Anglo-Saxon History") == ({"cities": ["York"]}, path.name)
        assert collisions.load_seed("Roman History") == (None, None)


def test_modern_phase():
    assert pipeline.modern_phase("history") == "index"
    assert [pipeline.modern_phase(p) for p in ("index", "create", "fix")] == ["index", "create", "fix"]
