"""CLI characterization: what the verbs print, write and return today.

These pin **current** behaviour, including behaviour a planned fix calls
wrong, so that each fix flips its pin on purpose and a pure restructuring
flips none. A test whose name says `today` or whose docstring names a planned
change is a pin of that kind: when the change lands, it is expected to fail,
and the commit that makes it fail says so.

* `lint` (plain, `--review`, `--review --via`, `--apply-review`): stdout lines,
  the review TSV, the follow-up hint, exit codes (v0.2.0 moved the adjudication
  out of `cli.py`; nothing here could change when it did).
* `sources` with a scripted resolver: lines, prefetch, summary, exit codes,
  `--limit`/`--missing`.
* `run --limit` with two atom types across dry-run, sync and batch, and
  `--limit 0` everywhere including `run_batch submit --from-history` — five
  different meanings once, one since v0.1.20.
* batch submit failing on the second atom type, a provider without batch, and
  `run --batch` opening a `_runs/` entry.

Paths in output are replaced by `{F}` / `{D}` so the pins do not depend on
`tmp_path`.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import cli, journal, settings, store
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.domains.history.models.person import Person
from tests.factories import minimal_payload

MODEL = "gpt-5-nano"


def _person(subject, other=(), common=None, sources=None, model=MODEL):
    payload = minimal_payload(Person)
    payload["name"]["common"] = common or subject
    payload["name"]["otherNames"] = list(other)
    if sources is not None:
        payload["name"]["sources"] = list(sources)
    path = store.path_for("person", None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _discovery(subject, sources=None, model=MODEL):
    payload = minimal_payload(Discovery)
    payload["name"]["common"] = subject
    if sources is not None:
        payload["name"]["sources"] = list(sources)
    path = store.path_for("discovery", None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(capsys, argv, **paths):
    rc = cli.main(argv)
    out = capsys.readouterr().out
    for name, path in paths.items():
        out = out.replace(str(path), "{" + name + "}")
    return rc, out.splitlines()


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------

@pytest.fixture
def lint_corpus(workspace, tmp_path, pack_overrides):
    """Three findings: one over-broad alias, one name two atoms share."""
    pack_overrides(aliases_path=tmp_path / "pack_aliases.txt")
    paths = {
        "ii": _person("Ramesses II", ["Ozymandias", "Ramesses III", "Usermaatre"]),
        "iii": _person("Ramesses III", ["Usermaatre"]),
        "akh": _person("Akhenaten", ["Amenhotep IV"]),
    }
    return paths


REVIEW_ROWS = [
    "?\tperson\tperson_h0ad7b449232e51f2_ramesses_ii_gpt-5-nano.json\tRamesses III\tRamesses II\t"
    "agrees: birth.location~0km, death.location~0km, birth.date~0y\t",
    "?\tperson\tperson_h0ad7b449232e51f2_ramesses_ii_gpt-5-nano.json\tUsermaatre\tRamesses II\t"
    "agrees: birth.location~0km, death.location~0km, birth.date~0y\t"
    "containment is the usual trap here; check which subject the name really belongs to",
    "?\tperson\tperson_h4921ac0736907f29_ramesses_iii_gpt-5-nano.json\tUsermaatre\tRamesses III\t"
    "agrees: birth.location~0km, death.location~0km, birth.date~0y\t"
    "containment is the usual trap here; check which subject the name really belongs to",
]


def _rows(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]


class TestLint:
    def test_a_plain_run_prints_findings_by_check_and_exits_1(self, lint_corpus, capsys):
        rc, lines = _run(capsys, ["lint"])
        assert rc == 1
        assert lines == [
            "",
            "overbroad-alias  (1)",
            "  person    Ramesses II                        claims 'Ramesses III', which "
            "ramesses_iii_gpt-5-nano.json calls itself — an alias, or containment mistaken for one?",
            "",
            "shared-alias  (2)",
            "  person    Ramesses II                        claims 'Usermaatre', which no atom is called "
            "and 1 other subject(s) also claim (Ramesses III) — until one side drops it the name "
            "resolves to nothing and invites a false merge",
            "  person    Ramesses III                       claims 'Usermaatre', which no atom is called "
            "and 1 other subject(s) also claim (Ramesses II) — until one side drops it the name "
            "resolves to nothing and invites a false merge",
            "",
            "3 finding(s). Nothing was changed — these are judgement calls, not repairs.",
        ]

    def test_a_clean_corpus_prints_no_findings_and_exits_0(self, workspace, capsys):
        _person("Akhenaten", ["Amenhotep IV"])
        assert _run(capsys, ["lint"]) == (0, ["No findings."])

    def test_an_unknown_check_exits_2(self, lint_corpus, capsys):
        rc, lines = _run(capsys, ["lint", "--check", "nope"])
        assert rc == 2
        assert lines[0].startswith("Unknown check(s) ['nope']. Known: ")

    def test_review_writes_the_tsv_and_the_hint_and_exits_1(self, lint_corpus, tmp_path, capsys):
        review = tmp_path / "r.tsv"
        rc, lines = _run(capsys, ["lint", "--check", "overbroad-alias", "--check", "shared-alias",
                                  "--review", str(review)], F=review)
        assert rc == 1
        assert lines == [
            "Wrote 3 finding(s) to {F}.",
            "Set the verdict column, then: universe_builder lint --apply-review {F} --apply",
        ]
        assert _rows(review) == REVIEW_ROWS
        assert review.read_text(encoding="utf-8").startswith("# Lint findings awaiting a verdict.\n")

    def test_review_via_an_unknown_source_exits_2_before_running(self, lint_corpus, tmp_path, capsys):
        review = tmp_path / "r.tsv"
        assert _run(capsys, ["lint", "--review", str(review), "--via", "bogus"]) == (2, [])
        assert not review.exists()

    def test_review_via_a_source_prefills_its_verdicts_and_abstains_with_evidence(
            self, lint_corpus, tmp_path, capsys, pack_overrides):
        class Identity:
            """Redirects: the two pharaohs are separate articles; Usermaatre is none."""

            pages = {"Ramesses II": (1, "Ramesses II", False), "Ramesses III": (2, "Ramesses III", False)}

            def resolve_titles(self, names):
                return {n: self.pages.get(n, (None, "", False)) for n in names}

            def subject_leads(self, subjects):
                return {"Ramesses II": "Ramesses II, throne name Usermaatre Setepenre, was a pharaoh."}

            def identity_verdict(self, subject, alias, resolved):
                from universe_builder.domains.history.wikipedia import identity_verdict

                return identity_verdict(subject, alias, resolved)

        pack_overrides(identity_resolvers={"fake": Identity()})
        review = tmp_path / "r.tsv"
        rc, lines = _run(capsys, ["lint", "--check", "overbroad-alias", "--check", "shared-alias",
                                  "--review", str(review), "--via", "fake"], F=review)
        assert rc == 1
        assert lines == VIA_LINES
        assert _rows(review) == VIA_ROWS


VIA_LINES = [
    'resolving 3 name(s) through fake...',
    'fetching 2 article opening(s) as evidence...',
    'Wrote 3 finding(s) to {F}.',
    'Set the verdict column, then: universe_builder lint --apply-review {F} --apply',
]
VIA_ROWS = [
    'wrong\tperson\tperson_h0ad7b449232e51f2_ramesses_ii_gpt-5-nano.json\tRamesses III\tRamesses II\tagre'
    "es: birth.location~0km, death.location~0km, birth.date~0y\t'Ramesses II' and 'Ramesses III' are "
    'separate articles',
    '?\tperson\tperson_h0ad7b449232e51f2_ramesses_ii_gpt-5-nano.json\tUsermaatre\tRamesses II\tagrees: bi'
    "rth.location~0km, death.location~0km, birth.date~0y\tno Wikipedia article for 'Usermaatre' | Ram"
    'esses II opens: Ramesses II, throne name Usermaatre Setepenre, was a pharaoh.',
    '?\tperson\tperson_h4921ac0736907f29_ramesses_iii_gpt-5-nano.json\tUsermaatre\tRamesses III\tagrees: '
    "birth.location~0km, death.location~0km, birth.date~0y\tno Wikipedia article for 'Usermaatre'",
]


class TestApplyReview:
    """`lint --apply-review`: dry by default (exit 1), `--apply` writes (exit 0)."""

    @pytest.fixture
    def adjudicated(self, lint_corpus, tmp_path):
        review = tmp_path / "r.tsv"
        rows = [row.split("\t") for row in REVIEW_ROWS]
        rows[0][0], rows[1][0], rows[2][0] = "wrong", "same", "skip"
        review.write_text("# header\n" + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")
        return review

    def test_the_dry_run_shows_both_halves_and_exits_1(self, adjudicated, tmp_path, capsys):
        rc, lines = _run(capsys, ["lint", "--apply-review", str(adjudicated)])
        assert rc == 1
        assert lines == [
            "3 adjudicated row(s): 1 same, 1 wrong, 1 left alone",
            "  alias: Ramesses II = Usermaatre",
            "  remove: person_h0ad7b449232e51f2_ramesses_ii_gpt-5-nano.json: Ramesses III",
            "",
            "Nothing written. Re-run with --apply.",
        ]
        assert not (tmp_path / "pack_aliases.txt").exists()

    def test_apply_writes_the_alias_line_and_edits_the_atom(self, adjudicated, lint_corpus, tmp_path, capsys):
        rc, lines = _run(capsys, ["lint", "--apply-review", str(adjudicated), "--apply"])
        assert rc == 0
        assert lines == [
            "3 adjudicated row(s): 1 same, 1 wrong, 1 left alone",
            "  alias: Ramesses II = Usermaatre",
            "  edited 1 atom(s); wrote 1 alias line(s)",
        ]
        # Into the alias file in force, `settings.ALIASES_LIST` (the workspace's), never
        # the pack's own path behind it. Until v0.1.20 it went to the pack's.
        assert (tmp_path / "aliases.txt").read_text(encoding="utf-8") == (
            "\n\n# Added by `universe_builder lint --apply-review`: adjudicated identity.\n"
            "[*]\nRamesses II = Usermaatre\n")
        assert not (tmp_path / "pack_aliases.txt").exists()
        name = json.loads(lint_corpus["ii"].read_text(encoding="utf-8"))["name"]
        assert name["otherNames"] == ["Ozymandias", "Usermaatre"]

    def test_a_typo_in_a_verdict_refuses_everything_and_exits_2(self, lint_corpus, tmp_path, capsys):
        review = tmp_path / "r.tsv"
        review.write_text("sane\t" + REVIEW_ROWS[0].split("\t", 1)[1] + "\n", encoding="utf-8")
        rc, lines = _run(capsys, ["lint", "--apply-review", str(review)])
        assert rc == 2
        assert lines == [
            "Unrecognised verdict(s) — nothing was applied, because a typo must not silently "
            "discard a decision:",
            "  'sane' on person_h0ad7b449232e51f2_ramesses_ii_gpt-5-nano.json",
            "Use one of: same, wrong, skip, ?",
        ]


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def _resolver(per_atom=1):
    """A free resolver: one good and one bad URL per subject, plus prefetch."""
    def ask(subject, atom_type, hints=()):
        slug = subject.replace(" ", "_")
        return [f"https://good.example/{slug}", f"https://bad.example/{slug}"][:max(per_atom, 2)]

    def prefetch(subjects):
        ask.prefetched = list(subjects)
        return len(subjects) - 1
    ask.prefetch = prefetch
    return ask


@pytest.fixture
def sources_corpus(workspace, monkeypatch, pack_overrides):
    from universe_builder import sources as sources_mod

    monkeypatch.setattr(sources_mod, "resolves",
                        lambda url, timeout=15: (True, "") if "good." in url else (False, "HTTP 404"))
    pack_overrides(url_resolvers={"fake": _resolver})
    return {
        "papyrus": _discovery("Papyrus", sources=["test"]),
        "wheel": _discovery("Wheel", sources=["https://good.example/Wheel"]),
        "ii": _person("Ramesses II", sources=["test"]),
    }


class TestSources:
    def test_the_first_free_resolver_is_the_default(self, sources_corpus, capsys):
        assert _run(capsys, ["sources"]) == (1, SOURCES_DRY)

    def test_a_pack_with_no_free_resolver_has_no_default_and_exits_2(self, sources_corpus, capsys,
                                                                     pack_overrides):
        pack_overrides(url_resolvers={})
        assert _run(capsys, ["sources"]) == (2, [])

    def test_an_unknown_resolver_exits_2(self, sources_corpus, capsys):
        assert _run(capsys, ["sources", "--via", "bogus"]) == (2, [])

    def test_the_dry_run_lists_every_proposal_and_exits_1(self, sources_corpus, capsys):
        rc, lines = _run(capsys, ["sources", "--via", "fake"])
        assert rc == 1
        assert lines == SOURCES_DRY

    def test_apply_writes_the_verified_urls_and_exits_0(self, sources_corpus, capsys):
        rc, lines = _run(capsys, ["sources", "--via", "fake", "--apply"])
        assert rc == 0
        assert lines[-1] == SOURCES_APPLY_SUMMARY
        sources = json.loads(sources_corpus["papyrus"].read_text(encoding="utf-8"))["name"]["sources"]
        assert sources == ["test", "https://good.example/Papyrus"]

    def test_limit_counts_atoms_across_types(self, sources_corpus, capsys):
        """`--limit 1` stops after one atom *in total*, prefetch included —
        unlike `run --limit`, which is per atom type on some paths."""
        rc, lines = _run(capsys, ["sources", "--via", "fake", "--limit", "1"])
        assert rc == 1
        assert lines == SOURCES_LIMIT

    def test_missing_skips_atoms_that_already_cite_something(self, sources_corpus, capsys):
        rc, lines = _run(capsys, ["sources", "--via", "fake", "--missing"])
        assert rc == 1
        assert lines == SOURCES_MISSING


SOURCES_DRY = [
    'prefetched 2/3 titles via fake',
    '',
    ' + Ramesses II                              1 verified, 1 rejected',
    '       https://good.example/Ramesses_II',
    '       -- HTTP 404: https://bad.example/Ramesses_II',
    ' + Papyrus                                  1 verified, 1 rejected',
    '       https://good.example/Papyrus',
    '       -- HTTP 404: https://bad.example/Papyrus',
    '   Wheel                                    0 verified, 2 rejected',
    '       -- already cited: https://good.example/Wheel',
    '       -- HTTP 404: https://bad.example/Wheel',
    '',
    '3 atom(s) examined; nothing written — re-run with --apply.',
]
SOURCES_APPLY_SUMMARY = '3 atom(s) examined; 2 updated with 2 URL(s).'
#: Prefetch counted one subject and the loop examined one, across both types.
SOURCES_LIMIT = [
    'prefetched 0/1 titles via fake',
    '',
    ' + Ramesses II                              1 verified, 1 rejected',
    '       https://good.example/Ramesses_II',
    '       -- HTTP 404: https://bad.example/Ramesses_II',
    '',
    '1 atom(s) examined; nothing written — re-run with --apply.',
]
SOURCES_MISSING = [
    'prefetched 1/2 titles via fake',
    '',
    ' + Ramesses II                              1 verified, 1 rejected',
    '       https://good.example/Ramesses_II',
    '       -- HTTP 404: https://bad.example/Ramesses_II',
    ' + Papyrus                                  1 verified, 1 rejected',
    '       https://good.example/Papyrus',
    '       -- HTTP 404: https://bad.example/Papyrus',
    '',
    '2 atom(s) examined; nothing written — re-run with --apply.',
]


# ---------------------------------------------------------------------------
# run --limit, across the three paths
# ---------------------------------------------------------------------------

@pytest.fixture
def two_types(workspace, tmp_path, monkeypatch, fake_provider):
    """Two atom types with work in every phase: three listed discoveries and
    three listed people (none generated), and two fix sentinels of each."""
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text("Egyptian History\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    history = store.path_for("history", None, "Egyptian History", MODEL)
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(json.dumps({
        "people": ["Narmer", "Djoser", "Khufu"], "discoveries": ["Papyrus", "Wheel", "Faience"],
        "events": [], "cities": [], "artifacts": []}), encoding="utf-8")
    for atom_type, subject in (("discovery", "Calendar"), ("discovery", "Glass"),
                               ("person", "Imhotep"), ("person", "Hatshepsut")):
        sentinel = store.error_path(store.path_for(atom_type, None, subject, MODEL))
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('{"broken": true}', encoding="utf-8")
    # Every atom folder exists, as in a real corpus (a missing one is no work
    # since v0.1.20; see `test_a_missing_atom_folder_is_no_work`).
    for atom_type in cli._atom_types():
        store.dir_for(atom_type).mkdir(parents=True, exist_ok=True)
    fake_provider.script(*([lambda r: json.dumps(minimal_payload(r.schema_model))] * 50))
    yield workspace
    journal.end_run()


def _subjects(fake_provider):
    return sorted((r.metadata["atom_type"], r.metadata["subject"]) for r in fake_provider.requests)


def _dry(capsys, *argv):
    rc, lines = _run(capsys, ["run", *argv, "--dry-run"])
    assert rc == 0
    count = next(int(line.split()[-1]) for line in lines if line.startswith("work items"))
    return count


class TestRunLimit:
    """`--limit N` is per atom type and `--limit 0` is nothing, on every path
    Until v0.1.20 the dry run of fix/verify/expand limited *in total*, and
    it and `run_batch submit` read 0 as "no limit"; those three pins moved
    with that fix."""

    def test_dry_run_create_limits_per_atom_type(self, two_types, capsys):
        assert _dry(capsys, "--phase", "create", "--limit", "1") == 2

    def test_dry_run_fix_limits_per_atom_type(self, two_types, capsys):
        """Moved in v0.1.20: was 1 (in total)."""
        assert _dry(capsys, "--phase", "fix", "--limit", "1") == 2

    def test_dry_run_create_limit_0_is_nothing(self, two_types, capsys):
        assert _dry(capsys, "--phase", "create", "--limit", "0") == 0

    def test_dry_run_fix_limit_0_is_nothing(self, two_types, capsys):
        """Moved in v0.1.20: was 4 (0 read as no limit)."""
        assert _dry(capsys, "--phase", "fix", "--limit", "0") == 0

    def test_a_negative_limit_is_refused(self, two_types, capsys):
        with pytest.raises(SystemExit) as raised:
            cli.main(["run", "--phase", "fix", "--limit", "-1", "--dry-run"])
        assert raised.value.code == 2

    def test_sync_create_limits_per_atom_type(self, two_types, fake_provider):
        assert cli.main(["run", "--phase", "create", "--limit", "1", "--model", MODEL]) == 0
        assert _subjects(fake_provider) == [("discovery", "Papyrus"), ("person", "Narmer")]

    def test_sync_fix_limits_per_atom_type(self, two_types, fake_provider):
        assert cli.main(["run", "--phase", "fix", "--limit", "1", "--model", MODEL]) == 0
        assert _subjects(fake_provider) == [("discovery", "calendar"), ("person", "imhotep")]

    @pytest.mark.parametrize("phase", ["create", "fix"])
    def test_sync_limit_0_is_nothing(self, two_types, fake_provider, phase):
        cli.main(["run", "--phase", phase, "--limit", "0", "--model", MODEL])
        assert fake_provider.requests == []

    def test_batch_create_limits_per_atom_type(self, two_types, fake_provider):
        assert cli.main(["run", "--phase", "create", "--batch", "--limit", "1", "--model", MODEL]) == 0
        assert _subjects(fake_provider) == [("discovery", "Papyrus"), ("person", "Narmer")]

    def test_batch_fix_limits_per_atom_type(self, two_types, fake_provider):
        assert cli.main(["run", "--phase", "fix", "--batch", "--limit", "1", "--model", MODEL]) == 0
        assert _subjects(fake_provider) == [("discovery", "calendar"), ("person", "imhotep")]

    @pytest.mark.parametrize("phase", ["create", "fix"])
    def test_batch_limit_0_submits_nothing_and_exits_1(self, two_types, fake_provider, phase):
        assert cli.main(["run", "--phase", phase, "--batch", "--limit", "0", "--model", MODEL]) == 1
        assert fake_provider.requests == []

    def test_run_batch_submit_from_history_limit_0_is_nothing(self, two_types, fake_provider):
        """Moved in v0.1.20: `if args.limit:` read 0 as falsy and every subject went."""
        from universe_builder import run_batch

        assert run_batch.main(["submit", "--atom", "person", "--model", MODEL,
                               "--from-history", "--limit", "0"]) == 0
        assert fake_provider.requests == []

    def test_run_batch_submit_from_history_limits_per_atom_type(self, two_types, fake_provider):
        from universe_builder import run_batch

        assert run_batch.main(["submit", "--atom", "person", "--model", MODEL,
                               "--from-history", "--limit", "2"]) == 0
        assert len(fake_provider.requests) == 2


@pytest.mark.parametrize("phase", ["fix", "verify", "expand", "append"])
def test_a_missing_atom_folder_is_no_work(workspace, fake_provider, phase):
    """Moved in v0.1.20. Found while writing these pins: `augment_and_save_prompt` called
    `iterdir()` on the atom folder unguarded, so `run --phase fix` (or verify,
    expand, append) over every atom type raised `FileNotFoundError` as soon as
    one type had no folder, while the dry run and batch submit skipped it. A
    missing folder is now "no work" on every path."""
    store.dir_for("discovery").mkdir(parents=True)
    assert cli.main(["run", "--phase", phase, "--model", MODEL]) == 0
    assert fake_provider.requests == []


class TestBatchSubmitFailures:
    def test_a_failure_on_the_second_atom_type_is_reported_beside_what_went(
            self, two_types, fake_provider, monkeypatch, capsys):
        """Moved in v0.1.20: the second group's failure used to raise out of
        `cli.main` after the first had been submitted, with no report. Now
        every group is tried, the failure is listed beside the handle that
        went, and the exit is 1."""
        real = fake_provider.submit_batch
        calls = []

        def second_fails(requests):
            calls.append(requests[0].metadata["atom_type"])
            if len(calls) == 2:
                raise RuntimeError("provider rejected the batch")
            return real(requests)

        monkeypatch.setattr(fake_provider, "submit_batch", second_fails)
        assert cli.main(["run", "--phase", "create", "--batch", "--model", MODEL]) == 1
        assert calls == ["discovery", "person"]
        assert [p.name.split("-", 1)[1] for p in settings.batch_dir().glob("*.json")] == ["discovery.json"]
        out = capsys.readouterr().out
        assert "1 batch(es) submitted:" in out and "-discovery.json" in out
        assert "1 group(s) NOT submitted:" in out and "provider rejected the batch" in out

    def test_a_provider_without_batch_is_reported_not_raised(self, two_types, fake_provider, capsys):
        """Moved in v0.1.20: raised NotImplementedError out of the CLI."""
        fake_provider.without_batch()
        assert cli.main(["run", "--phase", "create", "--batch", "--model", MODEL]) == 1
        out = capsys.readouterr().out
        assert "group(s) NOT submitted" in out and "has no batch job API" in out
        assert fake_provider.requests == []

    def test_run_batch_opens_no_journal_run(self, two_types, fake_provider):
        """Moved in v0.1.20: `run --batch` started a `_runs/` entry with no attempts
        in it (the handle is the record; `run_batch submit` opens none)."""
        assert journal.list_runs() == []
        assert cli.main(["run", "--phase", "create", "--batch", "--model", MODEL]) == 0
        assert journal.list_runs() == []

    def test_run_batch_submit_opens_no_journal_run(self, two_types, fake_provider):
        from universe_builder import run_batch

        assert run_batch.main(["submit", "--atom", "person", "--model", MODEL, "--from-history"]) == 0
        assert journal.list_runs() == []
