"""Prompt experiments (v0.5.0): A on everything, B on a shadow sample.

What these pin, in the order the plan's review raised them:

* experiment output never reaches the corpus, and the corpus writers never
  reach the experiments root (the two guards, and an end-to-end shadow run);
* `PromptSpec("live")` sends exactly the bytes the pipeline sends, and a
  variant's bytes never bleed into another spec's within one process;
* the sample is deterministic, stratified and skips what is already done;
* the price guard is on price, per atom type;
* the verdict arithmetic, against hand-computed values.
"""

import json

import pytest

from universe_builder import cli, experiments, journal, pipeline, prompt_spec, settings, store, topics
from universe_builder import experiment_stats as stats
from universe_builder.domains.history.models.city import City
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.requests import build_request, build_requests, with_create_note
from tests.factories import minimal_payload

MODEL = "gpt-6-luna"
CITIES = ["Memphis", "Thebes", "Abydos", "Edfu", "Aswan", "Philae"]


def _history(cities, name="history_h1111111111111111_egyptian_history_m.json"):
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": [], "discoveries": [], "events": [], "artifacts": [],
                                "cities": cities}), encoding="utf-8")


def _city(name: str, **extra) -> str:
    payload = minimal_payload(City)
    payload["name"]["common"] = name
    payload.update(extra)
    return json.dumps(payload)


class _Every(dict):
    """Batch replies for every custom_id, from one function of the request."""

    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def get(self, key, default=None):
        return self.reply


@pytest.fixture
def variants(tmp_path, pack_overrides):
    root = tmp_path / "variants"

    def make(name, overrides, acceptance=None, files=None):
        folder = root / name
        folder.mkdir(parents=True)
        for filename, text in (files or {}).items():
            (folder / filename).write_text(text, encoding="utf-8")
        body = {"name": name, "hypothesis": "test", "overrides": overrides}
        if acceptance:
            body["acceptance"] = acceptance
        (folder / "variant.yaml").write_text(json.dumps(body), encoding="utf-8")
        return name

    pack_overrides(variants_dir=root)
    return make


# -- separation -------------------------------------------------------------------


class TestSeparation:
    def test_the_experiments_root_is_a_sibling_of_the_data_base(self, workspace):
        root = settings.experiments_dir()
        assert root.parent.name == "experiments"
        assert settings.data_dir() not in root.parents and root not in settings.data_dir().parents

    def test_a_root_inside_the_corpus_is_refused(self, workspace, monkeypatch):
        monkeypatch.setenv("UNIVERSE_BUILDER_EXPERIMENTS_DIR", str(settings.data_dir()))
        with pytest.raises(ValueError, match="overlaps the data directory"):
            settings.experiments_dir()
        monkeypatch.setenv("UNIVERSE_BUILDER_EXPERIMENTS_DIR", str(settings.data_dir() / "inside"))
        with pytest.raises(ValueError, match="overlaps"):
            settings.experiments_dir()

    def test_the_experiment_writer_refuses_the_corpus(self, workspace):
        with pytest.raises(ValueError, match="outside"):
            store.write_experiment(store.dir_for("city") / "x.json", {"a": 1})

    def test_the_corpus_writers_refuse_the_experiments_root(self, workspace):
        target = settings.experiments_dir() / "x" / "city" / "city_h0000000000000000_x_m.json"
        with pytest.raises(ValueError, match="experiments root"):
            store.write_atom(target, City.model_validate(json.loads(_city("X"))))
        with pytest.raises(ValueError, match="experiments root"):
            store.write_error(target, "{}")

    def test_a_wrapped_handle_is_not_a_corpus_handle(self, workspace):
        path = experiments.folder("e") / "_batches" / "B-city.json"
        store.write_experiment(path, {experiments.HANDLE_MARK: 1, "handle": {
            "provider": "fake", "batch_id": "b", "model": MODEL, "submitted_at": "t", "request_count": 1}})
        with pytest.raises(Exception, match="not a batch handle"):
            batch_mod.load(path)


# -- the prompt spec ----------------------------------------------------------------


class TestPromptSpec:
    @pytest.mark.parametrize("atom", ["city", "person", "event", "artifact", "discovery"])
    def test_live_sends_the_pipelines_bytes(self, workspace, monkeypatch, atom):
        monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 2)
        _history(CITIES)
        live = prompt_spec.PromptSpec("live")
        ours = build_request(atom, "Edfu", MODEL, context="egyptian history", prompt=live,
                             user_instructions=live.user_instructions(atom, "Edfu", "egyptian history"))
        theirs = build_requests(atom, [with_create_note(atom, "Edfu", "egyptian history")], MODEL,
                                context="egyptian history")[0]
        assert (ours.system, ours.examples, ours.user, ours.schema_model) == \
               (theirs.system, theirs.examples, theirs.user, theirs.schema_model)

    def test_aa_has_live_s_fingerprint_and_a_variant_does_not(self, workspace, variants):
        variants("terse", {"base_system": "persona.txt"}, files={"persona.txt": "You are terse."})
        atoms = experiments.generating_atoms()
        live = prompt_spec.load("live").fingerprint(atoms)
        assert prompt_spec.load("aa").fingerprint(atoms) == live
        assert prompt_spec.load("terse").fingerprint(atoms) != live

    def test_a_switch_that_is_off_leaves_the_fingerprint_as_it_was(self, workspace):
        """v0.5.6: v0.5.5 hashed `candidates: false` into every fingerprint and broke
        every open experiment's pool. A spec that sends the same bytes must hash the
        same as before the switch existed."""
        import hashlib
        from universe_builder.providers.requests import build_request

        spec = prompt_spec.load("live")
        atoms = experiments.generating_atoms()
        probe = {}
        for atom in sorted(atoms):
            r = build_request(atom, "\x00probe", "probe-model", operation="create", context="\x00context", prompt=spec)
            probe[atom] = {"system": r.system, "user": r.user, "schema": r.schema_model.__name__,
                           "examples": [[e.user, e.assistant_json] for e in r.examples]}
        before = json.dumps({"probe": probe, "neighbours": spec.effective_neighbours(), "notes": True},
                            sort_keys=True, ensure_ascii=False)
        assert spec.fingerprint(atoms) == hashlib.sha256(before.encode("utf-8")).hexdigest()

    def test_no_bytes_bleed_between_specs_in_one_process(self, workspace, variants):
        variants("terse", {"base_system": "persona.txt"}, files={"persona.txt": "You are terse."})
        before = build_request("city", "Memphis", MODEL).system
        terse = build_request("city", "Memphis", MODEL, prompt=prompt_spec.load("terse")).system
        after = build_request("city", "Memphis", MODEL).system
        assert terse.startswith("You are terse.")
        assert before == after and not after.startswith("You are terse.")

    def test_a_variant_example_replaces_the_pack_s(self, workspace, variants):
        variants("ex", {"examples": {"city": ["tiny_town.json"]}},
                 files={"tiny_town.json": _city("Tiny Town")})
        request = build_request("city", "Memphis", MODEL, prompt=prompt_spec.load("ex"))
        assert [e.user.split("'")[1] for e in request.examples] == ["tiny town"]
        assert build_request("city", "Memphis", MODEL).examples != request.examples

    def test_neighbours_and_notes_toggles(self, workspace, monkeypatch, variants):
        monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 2)
        _history(CITIES)
        variants("bare", {"create_neighbours": 0, "create_notes": False})
        bare = prompt_spec.load("bare")
        assert bare.user_instructions("city", "Edfu", "egyptian history") == ""
        assert "Edfu" not in prompt_spec.load("live").user_instructions("city", "Edfu", "egyptian history")
        assert "Abydos" in prompt_spec.load("live").user_instructions("city", "Edfu", "egyptian history")

    def test_candidates_are_off_in_live_and_capped_in_a_variant(self, workspace, variants, pack_overrides):
        long = " ".join(["word"] * 60)
        pack_overrides(create_candidates=lambda atom, subject, context: [
            (f"Title {i}", long, f"https://w/{i}") for i in range(5)])
        variants("cand", {"create_candidates": True})
        assert prompt_spec.load("live").user_instructions("city", "Memphis", "egyptian history") == ""
        text = prompt_spec.load("cand").user_instructions("city", "Memphis", "egyptian history")
        assert text.startswith(prompt_spec.CANDIDATES_INTRO)
        rows = text.split("\n")[1:]
        assert len(rows) == prompt_spec.MAX_CANDIDATES
        assert rows[0] == "1. Title 0: " + " ".join(["word"] * prompt_spec.MAX_SENTENCE_WORDS) + " ... (https://w/0)"
        atoms = experiments.generating_atoms()
        assert prompt_spec.load("cand").fingerprint(atoms) != prompt_spec.load("live").fingerprint(atoms)

    def test_the_live_switch_puts_the_tested_block_in_the_pipeline(self, workspace, monkeypatch, pack_overrides):
        """v0.5.8: a won candidates variant goes live through the same builder."""
        monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 2)
        _history(CITIES)
        pack_overrides(create_candidates=lambda a, s, c: [("Edfu", "Edfu is a city.", "https://w/Edfu")],
                       create_candidates_live=True)
        piped = topics.create_instructions("city", "Edfu", "egyptian history")
        assert prompt_spec.CANDIDATES_INTRO in piped
        assert piped == prompt_spec.load("live").user_instructions("city", "Edfu", "egyptian history")
        assert prompt_spec.load("live").is_live

    def test_the_candidates_version_is_in_the_fingerprint_only_when_on(self, workspace, variants, pack_overrides):
        variants("cand", {"create_candidates": True})
        atoms = experiments.generating_atoms()
        pack_overrides(create_candidates=lambda a, s, c: [], create_candidates_version=1)
        v1, live1 = prompt_spec.load("cand").fingerprint(atoms), prompt_spec.load("live").fingerprint(atoms)
        pack_overrides(create_candidates_version=2)
        assert prompt_spec.load("cand").fingerprint(atoms) != v1
        assert prompt_spec.load("live").fingerprint(atoms) == live1

    def test_no_candidates_no_block_and_no_hook_is_an_error(self, workspace, variants, pack_overrides):
        variants("cand", {"create_candidates": True})
        pack_overrides(create_candidates=lambda atom, subject, context: [])
        assert prompt_spec.load("cand").user_instructions("city", "Memphis", None) == ""
        pack_overrides(create_candidates=None)
        with pytest.raises(prompt_spec.VariantError, match="declares no create_candidates"):
            prompt_spec.load("cand").user_instructions("city", "Memphis", None)

    @pytest.mark.parametrize("body, message", [
        ({"overrides": {"colour": 1}}, "unknown override"),
        ({"overrides": {}, "surprise": 1}, "unknown key"),
        ({"overrides": {"examples": {"no_such_atom": ["a.json"]}}}, "does not have"),
        ({"overrides": {"base_system": "missing.txt"}}, "not a file"),
        ({"overrides": {"examples": {"city": ["missing.json"]}}}, "is in none of"),
    ])
    def test_a_variant_file_is_strict(self, workspace, tmp_path, pack_overrides, body, message):
        folder = tmp_path / "variants" / "v"
        folder.mkdir(parents=True)
        (folder / "variant.yaml").write_text(json.dumps(body), encoding="utf-8")
        pack_overrides(variants_dir=tmp_path / "variants")
        with pytest.raises(prompt_spec.VariantError, match=message):
            prompt_spec.load("v")


# -- sampling and the price guard -----------------------------------------------------------


class TestSampling:
    def _topics(self, names, context="egyptian history"):
        return [topics.Topic(n, context) for n in names]

    def test_deterministic_and_independent_of_order(self):
        many = [f"City {i}" for i in range(100)]
        one = experiments.draw("exp", {"city": self._topics(many)}, 0.1, 3)
        two = experiments.draw("exp", {"city": self._topics(list(reversed(many)))}, 0.1, 3)
        assert [t.subject for t in one["city"]] == [t.subject for t in two["city"]]
        assert len(one["city"]) == 10
        other = experiments.draw("other", {"city": self._topics(many)}, 0.1, 3)
        assert {t.subject for t in other["city"]} != {t.subject for t in one["city"]}

    def test_small_types_get_the_minimum_and_never_more_than_exist(self):
        chosen = experiments.draw("exp", {"city": self._topics([f"C{i}" for i in range(20)]),
                                          "person": self._topics(["P1", "P2"])}, 0.1, 5)
        assert len(chosen["city"]) == 5 and len(chosen["person"]) == 2

    def test_price_guard_is_on_price_per_atom(self, workspace):
        from universe_builder import pipeline

        config = pipeline.load()
        assert experiments.price_refusals({"city": "gpt-6-luna", "person": "gpt-5-nano"}, config) == []
        refused = experiments.price_refusals({"city": "gpt-6-luna", "person": "claude-sonnet-5"}, config)
        assert len(refused) == 1 and refused[0].startswith("person: claude-sonnet-5")

    def test_run_experiment_needs_create_and_batch(self, workspace, caplog):
        _history(CITIES)
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--experiment", "aa",
                         "--model", MODEL, "--dry-run"]) == 2
        assert "create --batch" in caplog.text

    def test_the_shadow_dry_run_prints_b_s_sample(self, workspace, variants, capsys):
        """v0.5.3: it read the dry run's printable plan as work items, and crashed
        on the first real line."""
        _history(CITIES)
        variants("terse", {"base_system": "persona.txt"}, files={"persona.txt": "You are terse."})
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", MODEL,
                         "--experiment", "terse", "--sample-rate", "0.5", "--min-per-type", "2", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "experiment terse: B = terse" in out and "city         3 of 6" in out
        assert not experiments.folder("terse").exists(), "a dry run writes nothing"

    def test_run_experiment_refuses_a_dear_model_before_a_is_submitted(self, workspace, fake_provider, caplog):
        _history(CITIES)
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--experiment", "aa",
                         "--model", "claude-sonnet-5"]) == 2
        assert fake_provider.batches == [], "A must not go when the experiment is refused"


# -- end to end ------------------------------------------------------------------------------


def _corpus_files():
    root = settings.data_dir()
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


class TestShadow:
    def test_a_unchanged_b_beside_it_and_never_in_the_corpus(self, workspace, fake_provider, variants, capsys):
        _history(CITIES)
        variants("terse", {"base_system": "persona.txt"}, files={"persona.txt": "You are terse."})

        def reply(request):
            arm = request.metadata.get("arm", "A")
            return _city(request.metadata["subject"], **({"notableFigures": []} if arm == "B" else {}))

        fake_provider.batch_replies = _Every(reply)
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", MODEL,
                         "--experiment", "terse", "--sample-rate", "0.5", "--min-per-type", "2"]) == 0

        a_batch, b_batch = fake_provider.batches
        assert len(a_batch) == len(CITIES) and len(b_batch) == 3
        assert {r.metadata.get("arm") for r in a_batch} == {None}, "A's requests carry nothing new"
        assert all(not r.system.startswith("You are terse.") for r in a_batch)
        assert all(r.system.startswith("You are terse.") for r in b_batch)
        # A's requests are byte for byte what a run without --experiment builds.
        plain = build_requests("city", [with_create_note("city", c, "egyptian history") for c in CITIES],
                               MODEL, context="egyptian history")
        assert [(r.system, r.user, r.examples) for r in a_batch] == [(r.system, r.user, r.examples) for r in plain]

        # Only A's handle is in the corpus queue.
        assert [batch_mod.load(p).batch_id for p in batch_mod.list_batches()] == ["fake-batch-1"]

        from universe_builder import run_batch
        assert run_batch.main(["collect", "--all"]) == 0
        journal.end_run()
        runs_before = journal.list_runs()
        corpus_before = _corpus_files()

        outcome = experiments.collect("terse")
        assert outcome["written"] == 3 and outcome["from_journal"] == 3

        assert _corpus_files() == corpus_before, "an experiment wrote into the corpus"
        assert journal.list_runs() == runs_before, "an experiment journalled into the corpus journal"
        assert len(list(store.dir_for("city").glob("*.json"))) == len(CITIES)
        rows = experiments.pairs("terse")
        assert [r["state"] for r in rows] == ["complete"] * 3
        assert all(r["A"]["fill"] > r["B"]["fill"] for r in rows)

    def test_a_rerun_does_not_buy_b_twice(self, workspace, fake_provider, variants):
        _history(CITIES)
        meta, a, b = experiments.prepare("aa", mode="replay", variant="aa", rate=0, min_per_type=2)
        chosen = experiments.draw("aa", {"city": [topics.Topic(c, "egyptian history") for c in CITIES]}, 0, 2)
        experiments.submit(meta, {"A": a, "B": b}, chosen, {"city": MODEL}, arms=("B",),
                           max_tokens=1000, population={"city": 6})
        assert experiments.to_submit("aa", "B", chosen) == {}, "in flight counts as done"

    def test_a_changed_prompt_cannot_pool(self, workspace, variants):
        variants("v", {"base_system": "p.txt"}, files={"p.txt": "One."})
        experiments.prepare("v", mode="replay", variant="v", rate=0, min_per_type=2)
        (prompt_spec.variants_dir() / "v" / "p.txt").write_text("Two.", encoding="utf-8")
        with pytest.raises(experiments.ExperimentError, match="prompt changed"):
            experiments.prepare("v", mode="replay", variant="v", rate=0, min_per_type=2)


class TestReplayAndJudge:
    @pytest.fixture
    def collected(self, workspace, fake_provider, variants, pack_overrides):
        _history(CITIES)
        for city in CITIES:
            path = store.path_for("city", None, city, MODEL)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_city(city), encoding="utf-8")
        variants("terse", {"base_system": "persona.txt"}, files={"persona.txt": "You are terse."})
        pack_overrides(experiment_reference=lambda atom, subject, context: f"{subject} is a city in Egypt.",
                       experiment_checks=lambda atom, subject, context, payload: {
                           "no_figures": int(not payload.get("notableFigures"))})

        def reply(request):
            better = request.metadata["arm"] == "B"
            return _city(request.metadata["subject"], notableFigures=["BETTER"] if better else [])

        fake_provider.batch_replies = _Every(reply)
        corpus_before = _corpus_files()
        assert cli.main(["experiment", "replay", "terse", "--atom", "city", "--per-type", "4", "--apply"]) == 0
        assert cli.main(["experiment", "collect", "terse"]) == 0
        assert _corpus_files() == corpus_before
        return fake_provider

    def test_replay_only_samples_the_named_atoms(self, workspace, tmp_path):
        _history(CITIES)
        for city in CITIES:
            path = store.path_for("city", None, city, MODEL)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_city(city), encoding="utf-8")
        keep = {f"city:{store.atom_id('city', c)}" for c in ("Edfu", "Aswan")}
        found = experiments.replay_topics(["city"], only=keep)
        assert sorted(t.subject for t in found["city"]) == ["Aswan", "Edfu"]

    def test_replay_generates_both_arms_outside_the_corpus(self, collected):
        rows = experiments.pairs("terse")
        assert len(rows) == 4 and all(r["state"] == "complete" for r in rows)
        root = experiments.folder("terse") / "results"
        assert len(list((root / "A" / "city").glob("*.json"))) == 4
        assert len(list((root / "B" / "city").glob("*.json"))) == 4

    def test_a_report_only_check_stays_out_of_the_guardrail(self, collected, pack_overrides):
        pack_overrides(experiment_checks=lambda atom, subject, context, payload: {
            "measure": int("BETTER" in json.dumps(payload))}, experiment_report_only_checks=frozenset({"measure"}))
        guard = experiments.report("terse", write=False)["guardrails"]["flags"]
        assert guard["by_check"]["measure"] == {"A": 0, "B": 4} and guard["report_only"] == ["measure"]
        assert guard["mean_b"] == 0 and guard["passes"]

    def test_judge_both_orders_and_the_winner_is_computed_in_code(self, collected, capsys):
        def judge_reply(request):
            one_better = "BETTER" in request.user.split("RECORD 2:")[0].split("RECORD 1:")[1]
            good = {"wrong_subject": False, "contradictions": [], "false_certainty": [], "coverage": 2}
            bad = {"wrong_subject": False, "contradictions": ["x"], "false_certainty": [], "coverage": 2}
            return json.dumps({"record_1": good if one_better else bad,
                               "record_2": bad if one_better else good, "reason": "r"})

        assert cli.main(["experiment", "judge", "terse", "--model", MODEL]) == 0
        assert "Estimate only" in capsys.readouterr().out
        collected.script(*[judge_reply] * 8)
        assert cli.main(["experiment", "judge", "terse", "--model", MODEL, "--apply", "--concurrency", "1"]) == 0
        assert len(experiments.load_judgements("terse")) == 8
        # A second judge sends nothing: every pair is judged in both orders.
        assert experiments.plan_judging("terse", MODEL)["requests"] == []

        result = experiments.report("terse")
        assert result["judged"]["w_b"] == 4 and result["judged"]["w_a"] == 0
        assert result["verdict"] == "CONTINUE", "4 decisive pairs is far short of the target"
        assert result["guardrails"]["flags"]["by_check"] == {"no_figures": {"A": 4, "B": 0}}
        assert (experiments.folder("terse") / "report.md").is_file()


# -- the verdict ------------------------------------------------------------------------------


class TestStats:
    def test_sign_test_and_wilson_by_hand(self):
        assert stats.binom_sf(9, 10) == pytest.approx(11 / 1024)
        assert stats.binom_sf(0, 10) == 1.0
        low, high = stats.wilson(7, 10)
        assert low == pytest.approx(0.3968, abs=1e-4) and high == pytest.approx(0.8922, abs=1e-4)

    def test_holm(self):
        assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_sign_flip_is_seeded_and_sided(self):
        diffs = [1.0] * 12
        p = stats.sign_flip_p(diffs, greater=True, permutations=2000, seed=1)
        assert p < 0.01 and p == stats.sign_flip_p(diffs, greater=True, permutations=2000, seed=1)
        assert stats.sign_flip_p(diffs, greater=False, permutations=2000, seed=1) > 0.9
        assert stats.sign_flip_p([0, 0], permutations=10) == 1.0

    def test_power_matches_the_plan_table(self):
        assert stats.power(120, 0.62) == pytest.approx(0.82, abs=0.01)
        assert stats.power(35, 0.70) == pytest.approx(0.77, abs=0.01)

    def test_both_records_off_subject_is_a_reference_miss(self, workspace):
        """v0.5.1: the first live run's Mendes Stela, read against the city."""
        off = {"wrong_subject": True, "contradictions": [], "false_certainty": [], "coverage": 1}
        on = {"wrong_subject": False, "contradictions": [], "false_certainty": [], "coverage": 2}
        path = experiments.folder("m") / experiments.JUDGEMENTS_FILE
        for cid, first in (("c1", off), ("c2", on)):
            for order in ("AB", "BA"):
                store.append_experiment_line(path, {
                    "atom_type": "artifact", "content_id": cid, "order": order, "judge_model": MODEL,
                    "parsed": {"record_1": first, "record_2": off if first is off else on, "reason": ""}})
        expected = {("artifact", "c1"): experiments.REFERENCE_MISS, ("artifact", "c2"): "tie"}
        assert experiments.pair_outcomes("m", MODEL) == expected

    def test_a_cheap_judge_s_miss_on_a_decisive_pair_counts_against_it(self, workspace):
        off = {"wrong_subject": True, "contradictions": [], "false_certainty": [], "coverage": 1}
        good = {"wrong_subject": False, "contradictions": [], "false_certainty": [], "coverage": 2}
        bad = {"wrong_subject": False, "contradictions": ["x"], "false_certainty": [], "coverage": 2}
        path = experiments.folder("cal") / experiments.JUDGEMENTS_FILE
        for model, one, two in (("strong-model", good, bad), (MODEL, off, off)):
            for order in ("AB", "BA"):
                first, second = (one, two) if order == "AB" else (two, one)
                store.append_experiment_line(path, {
                    "atom_type": "city", "content_id": "c1", "order": order, "judge_model": model,
                    "parsed": {"record_1": first, "record_2": second, "reason": ""}})
        assert experiments.agreement("cal", MODEL, "strong-model") == {
            "against": "strong-model", "pairs": 1, "agree": 0, "rate": 0.0}

    def test_criteria_average_each_arm_over_both_orders(self, workspace):
        """Hand-computed: pair c1, A has 1 then 3 contradictions (mean 2), B 0 and 0;
        A's false certainty 0 and 2 (mean 1), B's 1 and 1."""
        def rec(c, f):
            return {"wrong_subject": False, "contradictions": ["x"] * c, "false_certainty": ["y"] * f, "coverage": 2}
        path = experiments.folder("crit") / experiments.JUDGEMENTS_FILE
        # AB: record_1 is A. BA: record_1 is B.
        for order, first, second in (("AB", rec(1, 0), rec(0, 1)), ("BA", rec(0, 1), rec(3, 2))):
            parsed = {"record_1": first, "record_2": second, "reason": ""}
            store.append_experiment_line(path, {"atom_type": "city", "content_id": "c1", "order": order,
                                                "judge_model": MODEL, "parsed": parsed})
        rows = [{"atom_type": "city", "content_id": "c1", "state": "complete"}]
        out = experiments.criteria("crit", MODEL, rows, experiments.DEFAULT_ACCEPTANCE)
        assert out["pairs"] == 1
        assert (out["contradictions"]["mean_a"], out["contradictions"]["mean_b"]) == (2.0, 0.0)
        assert (out["false_certainty"]["mean_a"], out["false_certainty"]["mean_b"]) == (1.0, 1.0)
        assert out["contradictions"]["mean_diff"] == -2.0 and out["wrong_subject"] == {"A": 0, "B": 0}

    def test_order_winner_is_lexicographic(self):
        clean = {"wrong_subject": False, "contradictions": [], "false_certainty": ["a", "b"], "coverage": 0}
        sloppy = {"wrong_subject": False, "contradictions": ["c"], "false_certainty": [], "coverage": 3}
        wrong = {"wrong_subject": True, "contradictions": [], "false_certainty": [], "coverage": 3}
        assert experiments.order_winner({"record_1": clean, "record_2": sloppy}) == 1
        assert experiments.order_winner({"record_1": wrong, "record_2": sloppy}) == 2
        assert experiments.order_winner({"record_1": clean, "record_2": dict(clean)}) is None


def _guard(**fails):
    base = {k: {"passes": True} for k in ("validity", "fill", "flags", "cost")}
    base["cost"]["ratio"] = 1.0
    for k in fails:
        base[k]["passes"] = False
    return base


def _judged(w_b, w_a, ties=0):
    n, d = w_a + w_b + ties, w_a + w_b
    return {"n": n, "w_a": w_a, "w_b": w_b, "ties": ties, "decisive": d,
            "p_better": stats.binom_sf(w_b, d), "p_worse": stats.binom_sf(w_a, d),
            "score_lower95": stats.wilson(w_b + ties / 2, n, z=stats.Z_ONE_SIDED_95)[0]}


class TestVerdict:
    acc = experiments.DEFAULT_ACCEPTANCE

    def test_a_guardrail_rejects_at_any_look(self):
        assert experiments.verdict(_guard(cost=True), _judged(10, 0), self.acc)[0] == "REJECT"

    def test_harm_rejects_early(self):
        assert experiments.verdict(_guard(), _judged(2, 20), self.acc)[0] == "REJECT"

    def test_short_of_target_continues_even_when_b_leads(self):
        assert experiments.verdict(_guard(), _judged(40, 10), self.acc)[0] == "CONTINUE"

    def test_adopt_only_at_target(self):
        assert experiments.verdict(_guard(), _judged(75, 45), self.acc)[0] == "ADOPT"
        assert experiments.verdict(_guard(), _judged(62, 58), self.acc)[0] == "NO DIFFERENCE"

    def test_adopt_if_cheaper_needs_the_saving_and_non_inferiority(self):
        cheap = _guard()
        cheap["cost"]["ratio"] = 0.8
        # Equal arms clear the 0.45 bound only with many pairs: n = 450 here.
        assert experiments.verdict(cheap, _judged(150, 150, ties=150), self.acc)[0] == "ADOPT IF CHEAPER"
        assert experiments.verdict(cheap, _judged(60, 60, ties=60), self.acc)[0] == "NO DIFFERENCE"
        assert experiments.verdict(cheap, _judged(55, 65, ties=10), self.acc)[0] == "NO DIFFERENCE"

    def test_not_judged_continues(self):
        assert experiments.verdict(_guard(), None, self.acc) == ("CONTINUE", ["not judged yet"])


# -- model and effort as the variable (v0.6.0) ------------------------------------------------


HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"


def _judge_reply(prefer_b=True):
    def reply(request):
        first = request.user.split("RECORD 2:")[0].split("RECORD 1:")[1]
        one_better = ("BETTER" in first) == prefer_b
        good = {"wrong_subject": False, "contradictions": [], "false_certainty": [], "coverage": 2}
        bad = {"wrong_subject": False, "contradictions": ["x"], "false_certainty": [], "coverage": 2}
        return json.dumps({"record_1": good if one_better else bad,
                           "record_2": bad if one_better else good, "reason": "r"})
    return reply


class TestModelAB:
    @pytest.fixture
    def corpus(self, workspace, fake_provider, pack_overrides):
        _history(CITIES)
        for city in CITIES:
            path = store.path_for("city", None, city, MODEL)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_city(city), encoding="utf-8")
        pack_overrides(experiment_reference=lambda atom, subject, context: f"{subject} is a city in Egypt.")

        def reply(request):
            return _city(request.metadata["subject"],
                         notableFigures=["BETTER"] if request.metadata["arm"] == "B" else [])

        fake_provider.batch_replies = _Every(reply)
        return fake_provider

    @pytest.fixture
    def haiku(self, corpus):
        """A on luna (the pipeline's create model), B on Haiku 5.5 at low
        effort, same prompt: the experiment the owner asked for."""
        assert cli.main(["experiment", "replay", "m-haiku", "--atom", "city", "--per-type", "4",
                         "--model-b", HAIKU, "--effort-b", "low", "--apply"]) == 0
        assert cli.main(["experiment", "collect", "m-haiku"]) == 0
        return corpus

    def test_b_runs_on_its_own_model_and_effort_in_its_own_batches(self, haiku):
        by_model = {}
        for chunk in haiku.batches:
            by_model.setdefault(chunk[0].model, set()).update(r.effort for r in chunk)
            assert len({r.model for r in chunk}) == 1, "one model per batch"
        assert by_model == {MODEL: {None}, HAIKU: {"low"}}
        meta = experiments.load_meta("m-haiku")
        assert meta["variant"] == "live", "the prompt is held fixed: B is the live prompt"
        assert meta["arms"]["B"]["model"] == HAIKU and meta["arms"]["B"]["effort"] == "low"
        assert meta["arms"]["A"]["model"] == MODEL and meta["arms"]["A"]["effort"] is None
        assert meta["arms"]["B"]["resolved"]["city"] == {"model": HAIKU, "effort": "low"}
        handles = list((experiments.folder("m-haiku") / "_batches" / "collected").glob("B-*.json"))
        assert handles and all(json.loads(p.read_text())["handle"]["extra"]["effort"] == "low" for p in handles)

    def test_each_reply_is_priced_on_its_own_models_row_at_batch_rates(self, haiku):
        from universe_builder.providers.pricing import estimate_cost
        from universe_builder.providers.types import Usage

        usage = Usage(**vars(haiku.usage))
        for arm, model in (("A", MODEL), ("B", HAIKU)):
            records = [json.loads(p.read_text()) for p in
                       (experiments.folder("m-haiku") / "results" / arm / "city").glob("*.json")]
            assert records and {r["model"] for r in records} == {model}
            assert {r["cost"] for r in records} == {estimate_cost(usage, model, batch=True)}
        assert {r["effort"] for r in [json.loads(p.read_text()) for p in (
            experiments.folder("m-haiku") / "results" / "B" / "city").glob("*.json")]} == {"low"}
        result = experiments.report("m-haiku", write=False)
        assert result["arms"]["B"]["models"] == [HAIKU] and result["arms"]["B"]["efforts"] == ["low"]
        assert result["spend"]["A"] == pytest.approx(4 * estimate_cost(usage, MODEL, batch=True))
        assert result["spend"]["B"] == pytest.approx(4 * estimate_cost(usage, HAIKU, batch=True))
        assert "aa_check" not in result

    def test_another_model_or_effort_for_b_is_a_new_experiment(self, haiku, caplog):
        assert cli.main(["experiment", "replay", "m-haiku", "--atom", "city", "--per-type", "4",
                         "--model-b", HAIKU, "--effort-b", "medium"]) == 2
        assert "arm B's city {'model': 'claude-haiku-5-5', 'effort': 'low'} -> " in caplog.text
        assert cli.main(["experiment", "replay", "m-haiku", "--atom", "city", "--per-type", "4"]) == 2

    def test_the_price_ceiling_holds_b_only_without_allow_dear_b(self, corpus, caplog):
        assert cli.main(["experiment", "replay", "m-sonnet", "--atom", "city", "--per-type", "4",
                         "--model-b", SONNET]) == 2
        assert "B city: claude-sonnet-5-5 outputs at $10" in caplog.text
        assert cli.main(["experiment", "replay", "m-sonnet", "--atom", "city", "--per-type", "4",
                         "--model-b", SONNET, "--allow-dear-b"]) == 0

    def test_a_is_held_to_the_ceiling_whatever_b_is_allowed(self, corpus, tmp_path, caplog):
        base = (settings.PACKAGE_DIR / "pipeline.yaml").read_text(encoding="utf-8")
        dear = tmp_path / "pipeline.yaml"
        dear.write_text(base.replace("  create: balanced", "  create: strong"), encoding="utf-8")
        assert cli.main(["experiment", "replay", "m-dear", "--atom", "city", "--per-type", "4", "--config",
                         str(dear), "--model-b", HAIKU, "--allow-dear-b"]) == 2
        assert "A city: gpt-6.1-sol outputs" in caplog.text

    def test_a_level_the_b_provider_lacks_is_refused(self, corpus, caplog):
        assert cli.main(["experiment", "replay", "m-max", "--atom", "city", "--per-type", "4",
                         "--model-b", MODEL, "--effort-b", "max"]) == 2
        assert "no effort level 'max'" in caplog.text

    def test_live_as_b_needs_a_model_or_effort_difference(self, workspace):
        def resolved(a, b):
            return {"A": {"city": a}, "B": {"city": b}}

        luna = {"model": MODEL, "effort": None}
        with pytest.raises(experiments.ExperimentError, match="cannot be 'live'"):
            experiments.prepare("x", mode="replay", variant="live", rate=0, min_per_type=2, dry_run=True)
        with pytest.raises(experiments.ExperimentError, match="cannot be 'live'"):
            experiments.prepare("x", mode="replay", variant="live", rate=0, min_per_type=2, dry_run=True,
                                resolved=resolved(luna, dict(luna)))
        experiments.prepare("x", mode="replay", variant="live", rate=0, min_per_type=2, dry_run=True,
                            resolved=resolved(luna, {"model": HAIKU, "effort": None}))
        experiments.prepare("x", mode="replay", variant="live", rate=0, min_per_type=2, dry_run=True,
                            resolved=resolved({"model": MODEL, "effort": "low"}, {"model": MODEL, "effort": "high"}))

    @pytest.mark.parametrize("argv", [
        ["--model-b", MODEL],                               # B's model is A's resolved model
        ["--effort-a", "low", "--effort-b", "low"],          # B's effort is A's
        ["--model-b", "balanced"],                           # a tier that resolves to A's model
    ])
    def test_flags_that_resolve_to_a_s_own_arm_are_no_difference(self, corpus, caplog, argv):
        assert cli.main(["experiment", "replay", "same", "--atom", "city", "--per-type", "4", *argv]) == 2
        assert "cannot be 'live'" in caplog.text

    def test_an_effort_b_s_model_ignores_is_recorded_as_none(self, corpus):
        """gpt-4o-mini takes no effort: `--effort-b low` sends none, and the
        experiment records none (B differs by model, so it still runs)."""
        assert cli.main(["experiment", "replay", "mini", "--atom", "city", "--per-type", "2",
                         "--model-b", "gpt-4o-mini", "--effort-b", "low", "--apply"]) == 0
        meta = experiments.load_meta("mini")
        assert meta["arms"]["B"]["model"] == "gpt-4o-mini" and meta["arms"]["B"]["effort"] is None
        b = [c for c in corpus.batches if c[0].model == "gpt-4o-mini"]
        assert b and {r.effort for r in b[0]} == {"low"}, "asked, and dropped by the provider"
        assert cli.main(["experiment", "collect", "mini"]) == 0
        records = [json.loads(p.read_text()) for p in
                   (experiments.folder("mini") / "results" / "B" / "city").glob("*.json")]
        assert records and {r["effort"] for r in records} == {None}
        assert experiments.report("mini", write=False)["arms"]["B"]["efforts"] == ["default"]

    def test_a_dearer_b_is_judged_on_quality_with_the_cost_beside_it(self, corpus):
        """Sonnet costs ~20x luna: a prompt experiment would REJECT on the cost
        guardrail before any judge; a model A/B reports the ratio instead."""
        assert cli.main(["experiment", "replay", "dear", "--atom", "city", "--per-type", "4",
                         "--model-b", SONNET, "--allow-dear-b", "--apply"]) == 0
        assert cli.main(["experiment", "collect", "dear"]) == 0
        result = experiments.report("dear")
        assert result["model_ab"] and result["cost_ratio"] > 1.25
        assert not result["guardrails"]["cost"]["passes"] and not result["guardrails"]["cost"]["gating"]
        assert result["verdict"] == "CONTINUE", "not judged yet: cost does not decide a model A/B"
        md = (experiments.folder("dear") / "report.md").read_text()
        assert f"CONTINUE, {result['cost_ratio']:.1f}x cost" in md and "reported only" in md

    def test_a_cost_cap_the_variant_sets_still_gates_a_model_ab(self, corpus, variants):
        variants("capped", {}, acceptance={"cost_cap": 2.0})
        assert cli.main(["experiment", "replay", "capped", "--variant", "capped", "--atom", "city",
                         "--per-type", "4", "--model-b", SONNET, "--allow-dear-b", "--apply"]) == 0
        assert cli.main(["experiment", "collect", "capped"]) == 0
        result = experiments.report("capped", write=False)
        assert result["guardrails"]["cost"]["gating"] and result["verdict"] == "REJECT"

    def test_a_variant_with_no_overrides_carries_acceptance_for_a_model_ab(self, corpus, variants):
        """`aa-sol`, the live prompt with its own
        acceptance thresholds, B on another model."""
        variants("aa-sol", {}, acceptance={"cost_cap": 30, "target_decisive": 40})
        with pytest.raises(experiments.ExperimentError, match="sends exactly what the live pack sends"):
            experiments.prepare("y", mode="replay", variant="aa-sol", rate=0, min_per_type=2, dry_run=True)
        assert cli.main(["experiment", "replay", "y", "--variant", "aa-sol", "--atom", "city", "--per-type",
                         "4", "--model-b", SONNET, "--allow-dear-b", "--apply"]) == 0
        meta = experiments.load_meta("y")
        assert meta["acceptance"]["cost_cap"] == 30 and meta["arms"]["B"]["model"] == SONNET

    def test_reuse_a_compares_a_second_b_with_the_identical_a(self, haiku):
        before = len(haiku.batches)
        assert cli.main(["experiment", "replay", "m-sonnet", "--atom", "city", "--reuse-a", "m-haiku",
                         "--model-b", SONNET, "--effort-b", "medium", "--allow-dear-b", "--apply"]) == 0
        new = haiku.batches[before:]
        assert [(c[0].model, c[0].effort) for c in new] == [(SONNET, "medium")], "B only; A is not bought"
        assert cli.main(["experiment", "collect", "m-sonnet"]) == 0

        assert experiments.load_sample("m-sonnet") == experiments.load_sample("m-haiku")
        for path in (experiments.folder("m-haiku") / "results" / "A" / "city").glob("*.json"):
            mine = json.loads((experiments.folder("m-sonnet") / "results" / "A" / "city" / path.name).read_text())
            theirs = json.loads(path.read_text())
            assert mine["raw_text"] == theirs["raw_text"] and mine["reused_from"] == "m-haiku"
        rows = experiments.pairs("m-sonnet")
        assert len(rows) == 4 and all(r["state"] == "complete" for r in rows)
        meta = experiments.load_meta("m-sonnet")
        assert meta["arms"]["A"]["reused_from"] == "m-haiku"
        assert meta["arms"]["A"]["resolved"] == experiments.load_meta("m-haiku")["arms"]["A"]["resolved"]
        spend = experiments.report("m-sonnet", write=False)["spend"]
        assert spend["A"] == 0.0 and spend["A_reused"] > 0 and spend["B"] > 0

    def test_reuse_a_is_its_own_arm_config_and_takes_no_a_overrides(self, haiku, caplog):
        assert cli.main(["experiment", "replay", "m-r", "--atom", "city", "--reuse-a", "m-haiku",
                         "--effort-a", "low", "--model-b", HAIKU]) == 2
        assert "--reuse-a takes A" in caplog.text

    def test_two_judges_are_stored_side_by_side_and_reported_as_a_panel(self, haiku, capsys):
        """Luna prefers B everywhere; Opus at low effort agrees on three pairs
        and prefers A on the fourth (by subject), so they agree on 3 of 4."""
        rows = sorted(experiments.pairs("m-haiku"), key=lambda r: r["subject"])
        dissent = rows[0]["subject"]

        def judge(request):
            if request.model == OPUS:
                assert request.effort == "low"
                return _judge_reply(prefer_b=dissent not in request.user)(request)
            return _judge_reply(prefer_b=True)(request)

        haiku.script(*[judge] * 16)
        assert cli.main(["experiment", "judge", "m-haiku", "--model", MODEL, "--apply", "--concurrency", "1"]) == 0
        assert experiments.plan_judging("m-haiku", OPUS, effort="low")["pairs"] == 4
        assert len(experiments.plan_judging("m-haiku", OPUS, effort="low")["requests"]) == 8, \
            "another judge's verdicts do not count as this one's"
        assert cli.main(["experiment", "judge", "m-haiku", "--model", OPUS, "--effort", "low", "--apply",
                         "--concurrency", "1"]) == 0
        lines = experiments.load_judgements("m-haiku")
        assert len(lines) == 16
        assert {experiments.judge_label(j) for j in lines} == {MODEL, "claude-opus-5-5@low"}
        assert experiments.judge_models("m-haiku") == {MODEL: 8, "claude-opus-5-5@low": 8}

        capsys.readouterr()
        assert cli.main(["experiment", "report", "m-haiku", "--panel", f"{MODEL},{OPUS}@low"]) == 0
        out = capsys.readouterr().out
        assert "Panel of 2 judges" in out and "Agreement: 3 of 4" in out
        result = experiments.report("m-haiku", write=False, panel_judges=[MODEL, "claude-opus-5-5@low"])
        panel = result["panel"]
        assert panel["per_judge"][MODEL]["w_b"] == 4
        assert (panel["per_judge"]["claude-opus-5-5@low"]["w_b"], panel["per_judge"]["claude-opus-5-5@low"]["w_a"]) \
            == (3, 1)
        assert (panel["agree"], panel["pairs_judged_by_all"], panel["agreement_rate"]) == (3, 4, 0.75)
        assert (panel["agreed"]["w_b"], panel["agreed"]["w_a"], panel["agreed"]["n"]) == (3, 0, 3)
        assert panel["verdict"] == "CONTINUE"

        from universe_builder.providers.pricing import estimate_cost
        from universe_builder.providers.types import Usage

        usage = Usage(**vars(haiku.usage))
        judges = result["spend"]["judges"]
        assert judges[MODEL] == pytest.approx(8 * estimate_cost(usage, MODEL))
        assert judges["claude-opus-5-5@low"] == pytest.approx(8 * estimate_cost(usage, OPUS))
        assert result["spend"]["judge"] == pytest.approx(sum(judges.values()))

    def test_the_panel_counts_a_pair_only_when_every_judge_agrees(self, workspace):
        """Hand-computed: c1 both B, c2 j1 B / j2 A, c3 both tie, c4 judged by j1 only."""
        good = {"wrong_subject": False, "contradictions": [], "false_certainty": [], "coverage": 2}
        bad = {"wrong_subject": False, "contradictions": ["x"], "false_certainty": [], "coverage": 2}
        path = experiments.folder("pnl") / experiments.JUDGEMENTS_FILE

        def write(judge, cid, winner, judge_effort=None):
            for order in ("AB", "BA"):
                if winner is None:
                    one, two = good, good
                else:
                    first_is_winner = (order == "AB") == (winner == "A")
                    one, two = (good, bad) if first_is_winner else (bad, good)
                line = {"atom_type": "city", "content_id": cid, "order": order, "judge_model": judge,
                        "parsed": {"record_1": one, "record_2": two, "reason": ""}}
                if judge_effort:
                    line["judge_effort"] = judge_effort
                store.append_experiment_line(path, line)

        for cid, w1, w2 in (("c1", "B", "B"), ("c2", "B", "A"), ("c3", None, None)):
            write("j1", cid, w1)
            write("j2", cid, w2, judge_effort="low")
        write("j1", "c4", "A")
        rows = [{"atom_type": "city", "content_id": c, "state": "complete", "weight": 1.0}
                for c in ("c1", "c2", "c3", "c4")]
        out = experiments.panel("pnl", ["j1", "j2@low"], rows, experiments.DEFAULT_ACCEPTANCE)
        assert (out["pairs_judged_by_all"], out["agree"], out["disagree"]) == (3, 2, 1)
        assert (out["decisive_pairs"], out["decisive_agree"]) == (2, 1)
        assert (out["agreed"]["w_b"], out["agreed"]["w_a"], out["agreed"]["ties"]) == (1, 0, 1)
        assert out["per_judge"]["j1"]["w_a"] == 1, "j1's own tally keeps the pair only it judged"

    def test_shadow_takes_b_s_model_and_effort(self, workspace, fake_provider, capsys):
        _history(CITIES)
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", MODEL,
                         "--experiment", "sh", "--model-b", HAIKU, "--effort-b", "low",
                         "--sample-rate", "0.5", "--min-per-type", "2"]) == 0
        b_batches = [c for c in fake_provider.batches if c[0].metadata.get("arm") == "B"]
        assert b_batches and {(c[0].model, c[0].effort) for c in b_batches} == {(HAIKU, "low")}
        a_batches = [c for c in fake_provider.batches if "arm" not in c[0].metadata]
        assert a_batches and {c[0].model for c in a_batches} == {MODEL}
        meta = experiments.load_meta("sh")
        assert meta["variant"] == "live" and meta["arms"]["B"]["model"] == HAIKU


class TestModelABVerdict:
    acc = experiments.DEFAULT_ACCEPTANCE

    def test_a_prompt_experiment_is_unchanged(self):
        assert experiments.verdict(_guard(cost=True), _judged(75, 45), self.acc) == \
            experiments.verdict(_guard(cost=True), _judged(75, 45), self.acc, model_ab=False, cost_gates=True)
        assert experiments.verdict(_guard(cost=True), _judged(75, 45), self.acc)[0] == "REJECT"
        assert experiments.verdict(_guard(), _judged(75, 45), self.acc)[0] == "ADOPT"

    def test_a_model_ab_answers_on_quality(self):
        dear = _guard(cost=True)
        v = experiments.verdict
        assert v(dear, _judged(75, 45), self.acc, model_ab=True, cost_gates=False)[0] == "B BETTER"
        assert v(dear, _judged(2, 20), self.acc, model_ab=True, cost_gates=False)[0] == "A BETTER"
        assert v(dear, _judged(62, 58), self.acc, model_ab=True, cost_gates=False)[0] == "NO DIFFERENCE"
        assert v(dear, _judged(40, 10), self.acc, model_ab=True, cost_gates=False)[0] == "CONTINUE"
        assert v(_guard(validity=True), _judged(75, 45), self.acc, model_ab=True, cost_gates=False)[0] == "REJECT"
        assert v(dear, _judged(75, 45), self.acc, model_ab=True, cost_gates=True)[0] == "REJECT"


class TestModelABFairness:
    @pytest.fixture
    def source(self, workspace, fake_provider, pack_overrides):
        _history(CITIES)
        for city in CITIES:
            path = store.path_for("city", None, city, MODEL)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_city(city), encoding="utf-8")
        pack_overrides(experiment_reference=lambda atom, subject, context: f"{subject} is a city.")
        fake_provider.batch_replies = _Every(lambda r: _city(r.metadata["subject"]))
        assert cli.main(["experiment", "replay", "src", "--atom", "city", "--per-type", "3",
                         "--model-b", HAIKU, "--apply"]) == 0
        assert cli.main(["experiment", "collect", "src"]) == 0
        return fake_provider

    def test_a_reused_a_asked_under_another_budget_is_refused(self, source, tmp_path, caplog):
        base = (settings.PACKAGE_DIR / "pipeline.yaml").read_text(encoding="utf-8")
        other = tmp_path / "pipeline.yaml"
        other.write_text(base.replace("max_tokens: 32768", "max_tokens: 16000"), encoding="utf-8")
        before = len(source.batches)
        assert cli.main(["experiment", "replay", "next", "--atom", "city", "--reuse-a", "src", "--config",
                         str(other), "--model-b", OPUS, "--allow-dear-b", "--apply"]) == 2
        assert "were not asked what today's A would ask" in caplog.text
        assert len(source.batches) == before

    def test_a_reply_with_no_recorded_prompt_is_refused(self, source, caplog):
        path = next((experiments.folder("src") / "results" / "A" / "city").glob("*.json"))
        record = json.loads(path.read_text())
        del record["prompt_sha256"]
        store.write_experiment(path, record)
        assert cli.main(["experiment", "replay", "next", "--atom", "city", "--reuse-a", "src",
                         "--model-b", OPUS, "--allow-dear-b"]) == 2
        assert "no recorded prompt" in caplog.text

    def test_a_source_from_before_v0_6_0_is_refused(self, source, caplog):
        meta = experiments.load_meta("src")
        del meta["arms"]["A"]["resolved"]
        experiments._save_meta(meta)
        assert cli.main(["experiment", "replay", "next", "--atom", "city", "--reuse-a", "src",
                         "--model-b", OPUS, "--allow-dear-b"]) == 2
        assert "predates v0.6.0" in caplog.text

    def test_the_panel_refuses_an_absent_or_repeated_judge(self, source, caplog):
        source.script(*[_judge_reply()] * 6)
        assert cli.main(["experiment", "judge", "src", "--model", MODEL, "--apply", "--concurrency", "1"]) == 0
        assert cli.main(["experiment", "report", "src", "--panel", f"{MODEL},{MODEL}"]) == 2
        assert "names a judge twice" in caplog.text
        assert cli.main(["experiment", "report", "src", "--panel", f"{MODEL},{OPUS}@low"]) == 2
        assert "no judgements on 'src' by claude-opus-5-5@low" in caplog.text

    def test_a_judge_tier_resolves_against_config(self, tmp_path):
        base = (settings.PACKAGE_DIR / "pipeline.yaml").read_text(encoding="utf-8")
        other = tmp_path / "pipeline.yaml"
        other.write_text(base.replace("    strong: gpt-6.1-sol", "    strong: gpt-6-astra", 1), encoding="utf-8")
        assert cli._judge_arg("strong@low") == "gpt-6.1-sol@low"
        assert cli._judge_arg("strong@low", pipeline.load(other)) == "gpt-6-astra@low"

    def test_arm_b_flags_need_an_experiment(self, workspace, caplog):
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--model-b", HAIKU, "--dry-run"]) == 2
        assert "need --experiment" in caplog.text


def test_shadow_a_carries_the_effort_its_batch_sent(workspace, fake_provider):
    _history(CITIES)
    fake_provider.batch_replies = _Every(lambda r: _city(r.metadata["subject"]))
    assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", MODEL,
                     "--effort", "low", "--experiment", "sh-e", "--model-b", HAIKU,
                     "--sample-rate", "0.5", "--min-per-type", "2"]) == 0
    from universe_builder import run_batch

    assert run_batch.main(["collect", "--all"]) == 0
    journal.end_run()
    assert experiments.collect("sh-e")["from_journal"] == 3
    folder = experiments.folder("sh-e") / "results" / "A" / "city"
    records = [json.loads(p.read_text()) for p in folder.glob("*.json")]
    assert records and {r["effort"] for r in records} == {"low"}
    meta = experiments.load_meta("sh-e")
    assert meta["arms"]["A"]["resolved"]["city"] == {"model": MODEL, "effort": "low"}
    assert meta["arms"]["B"]["resolved"]["city"] == {"model": HAIKU, "effort": "low"}
