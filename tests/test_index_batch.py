"""The index phase through batch ("index via batch").

`run --phase index --batch` submits one request per seed subject that has no
index atom yet, on the index phase's tier, one batch per seed; `run_batch
collect` writes each atom through `store` as the synchronous index call would
and journals it at the batch discount. Before this, `--batch` refused the
phase ("no batch shape yet").

The prompt-parity test is the one that matters: the index is the expensive
call, and the batch request must carry what the synchronous one does.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import cli, costs, journal, pipeline, run_batch, settings, store
from universe_builder.domains.history.models.history import History
from universe_builder.providers.pricing import estimate_cost
from universe_builder.providers.types import Usage
from tests.factories import minimal_payload

MODEL = "gpt-5-nano"
SEEDS = ("Egyptian History", "Greek History", "Roman History")


def _history_reply(request=None) -> str:
    return json.dumps(minimal_payload(History))


@pytest.fixture
def seeds(workspace, tmp_path, monkeypatch, fake_provider):
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text("".join(f"{s}\n" for s in SEEDS), encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    # Greek already has its atom: both paths must skip it.
    path = store.path_for("history", None, "Greek History", MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(minimal_payload(History)), encoding="utf-8")
    yield workspace
    journal.end_run()


def _subjects(requests):
    return [r.metadata["subject"] for r in requests]


class TestSubmit:
    def test_one_batch_per_seed_without_an_atom(self, seeds, fake_provider):
        assert cli.main(["run", "--phase", "index", "--batch", "--model", MODEL]) == 0
        assert [_subjects(b) for b in fake_provider.batches] == [["Egyptian History"], ["Roman History"]]
        handles = sorted(settings.batch_dir().glob("*.json"))
        assert len(handles) == 2
        for request in fake_provider.requests:
            assert request.schema_model is History
            assert request.metadata["atom_type"] == "history"
            # Routed as the index phase, so `costs --by phase` can say so.
            assert request.metadata["operation"] == "index"
            assert request.max_tokens == pipeline.load().max_tokens("index")

    def test_without_a_model_it_runs_on_the_index_tier(self, seeds, fake_provider):
        config = pipeline.load()
        frontier = config.resolve_model("index", "history").value
        assert frontier == config.resolve_tier("frontier")
        assert cli.main(["run", "--phase", "index", "--batch"]) == 0
        assert {r.model for r in fake_provider.requests} == {frontier}

    def test_limit_caps_seeds_and_0_is_nothing(self, seeds, fake_provider):
        assert cli.main(["run", "--phase", "index", "--batch", "--limit", "1", "--model", MODEL]) == 0
        assert _subjects(fake_provider.requests) == ["Egyptian History"]
        fake_provider.requests.clear()
        assert cli.main(["run", "--phase", "index", "--batch", "--limit", "0", "--model", MODEL]) == 1
        assert fake_provider.requests == []

    def test_every_seed_with_an_atom_submits_nothing(self, seeds, fake_provider, capsys):
        for subject in ("Egyptian History", "Roman History"):
            store.write_atom(store.path_for("history", None, subject, MODEL),
                             History.model_validate(minimal_payload(History)))
        assert cli.main(["run", "--phase", "index", "--batch", "--model", MODEL]) == 1
        assert fake_provider.requests == []
        assert not settings.batch_dir().exists() or not list(settings.batch_dir().glob("*.json"))

    def test_the_dry_run_lists_the_seeds_and_calls_nothing(self, seeds, fake_provider, capsys):
        assert cli.main(["run", "--phase", "index", "--batch", "--dry-run", "--model", MODEL]) == 0
        out = capsys.readouterr().out
        assert "batch (submit now, collect later)" in out
        assert "work items 2" in out
        assert "Egyptian History" in out and "Roman History" in out and "Greek History" not in out
        assert fake_provider.requests == []

    def test_opens_no_journal_run(self, seeds, fake_provider):
        assert cli.main(["run", "--phase", "index", "--batch", "--model", MODEL]) == 0
        assert journal.list_runs() == []


class TestPromptParity:
    """The batch index request against the synchronous one, for one seed.

    Both are now built by `requests.build_request`, so they differ
    ONLY in the routing (ARCHITECTURE.md, "Batching a phase"):

    1. `custom_id`: sync uses the subject, batch `history-00000-<slug>` (the
       batch routing key).
    2. `metadata`: batch adds `operation: "index"` (routing for collect and
       `costs`); both carry `atom_type` and `subject`.

    Everything else is equal: model, schema, system text (the persona alone —
    no context line, no operation text), the user turn with its
    `f"{prompt_start_text}'{subject}'"` quoting and the create note, the
    examples, `max_tokens`, temperature and caching. Until the prompt-builder
    merge sync quoted the subject as `f"{prompt_start_text} '{subject}'"` (two
    spaces before the quote), which this test normalised.
    """

    def _pair(self, fake_provider):
        from universe_builder import execute_prompt

        notes = settings.CREATE_NOTES
        notes.write_text("history | Egyptian History | The Nile valley, from the Predynastic "
                         "period to the Arab conquest.\n", encoding="utf-8")
        assert cli.main(["run", "--phase", "index", "--batch", "--limit", "1", "--model", MODEL]) == 0
        [batch] = fake_provider.requests
        fake_provider.script(_history_reply)
        assert execute_prompt.run_phase("index", model=MODEL, limit=1) == 0
        sync = fake_provider.requests[1]
        return sync, batch

    def test_the_batch_index_request_is_the_sync_one(self, seeds, fake_provider):
        sync, batch = self._pair(fake_provider)

        assert batch.model == sync.model == MODEL
        assert batch.schema_model is sync.schema_model is History
        assert batch.system == sync.system
        assert batch.user == sync.user
        assert "The Nile valley" in batch.user, "the seed's create note reaches both"
        assert sync.examples == batch.examples
        assert batch.examples, "the one-shot example reaches both"
        assert batch.max_tokens == sync.max_tokens == pipeline.load().max_tokens("index")
        assert (batch.temperature, batch.cache) == (sync.temperature, sync.cache)
        assert {k: v for k, v in batch.metadata.items() if k != "operation"} == sync.metadata
        assert batch.metadata["operation"] == "index"

    def test_the_differences_are_exactly_the_documented_ones(self, seeds, fake_provider):
        """Fails if a difference disappears too, so the docstring stays true."""
        from universe_builder.atoms import atom_mapping

        sync, batch = self._pair(fake_provider)
        start = atom_mapping["history"].prompt_start_text
        assert sync.user.startswith(f"{start}'Egyptian History'")
        assert (sync.custom_id, batch.custom_id) == ("Egyptian History", "history-00000-egyptian_history")
        assert sync.metadata == {"atom_type": "history", "subject": "Egyptian History"}
        assert batch.metadata == {**sync.metadata, "operation": "index"}


class TestCollect:
    def _submit_and_collect(self, fake_provider, reply, model=MODEL):
        assert cli.main(["run", "--phase", "index", "--batch", "--limit", "1", "--model", model]) == 0
        [request] = fake_provider.requests
        fake_provider.batch_replies[request.custom_id] = reply
        return run_batch.main(["collect", "--all"])

    def test_writes_the_atom_where_the_sync_index_would(self, seeds, fake_provider):
        assert self._submit_and_collect(fake_provider, _history_reply) == 0
        path = store.path_for("history", None, "Egyptian History", MODEL)
        assert History.model_validate(store.read_json(path))
        assert not store.error_path(path).exists()
        # Collected, so the handle leaves the sweep, and the seed is now done.
        assert list(settings.batch_dir().glob("*.json")) == []
        fake_provider.requests.clear()
        assert cli.main(["run", "--phase", "index", "--batch", "--limit", "1", "--model", MODEL]) == 0
        assert _subjects(fake_provider.requests) == ["Roman History"]

    def test_a_reply_that_does_not_validate_is_an_error_sentinel(self, seeds, fake_provider):
        assert self._submit_and_collect(fake_provider, '{"people": 3}') == 0
        path = store.path_for("history", None, "Egyptian History", MODEL)
        assert not path.exists()
        assert store.error_path(path).read_text(encoding="utf-8") == '{"people": 3}'
        [entry] = [e for e in costs.iter_attempts()]
        assert entry["validation_status"] == "failed" and entry["work_status"] == journal.NEEDS_FIX

    def test_journals_at_the_batch_discount_and_costs_says_index(self, seeds, fake_provider):
        frontier = pipeline.load().resolve_tier("frontier")
        assert self._submit_and_collect(fake_provider, _history_reply, model=frontier) == 0
        [entry] = list(costs.iter_attempts())
        assert entry["execution_mode"] == "batch"
        assert entry["operation"] == "index"
        assert entry["atom_type"] == "history" and entry["subject"] == "Egyptian History"
        assert entry["requested_model"] == frontier
        usage = Usage(input_tokens=100, output_tokens=50)   # the fake's
        assert entry["estimated_cost"] == pytest.approx(estimate_cost(usage, frontier, batch=True))
        assert entry["estimated_cost"] == pytest.approx(estimate_cost(usage, frontier) / 2)
        assert [r.key for r in costs.build("phase").rows] == ["index (batch)"]
        assert [r.key for r in costs.build("mode").rows] == ["batch"]


def test_run_batch_submit_from_history_still_refuses_the_index_atom(seeds, fake_provider):
    """`run --phase index --batch` is the supported door. `run_batch submit
    --from-history --atom history` reads the create work list, which never
    holds index atoms, and says so as before."""
    with pytest.raises(SystemExit) as raised:
        run_batch.main(["submit", "--atom", "history", "--model", MODEL, "--from-history"])
    assert "Nothing to generate for history" in str(raised.value)
    assert fake_provider.requests == []
