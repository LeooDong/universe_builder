"""Batched expand: fan out one request per period, splice replies per city.

Expand breaks the batch's usual one-request-one-file shape — it is N requests
per file, then a splice — so these pin the two halves: submit emits a request
per unexpanded period carrying the city as payload and the period index in
routing, and collect groups the replies by target city and writes each once.
No network: submit's provider call is faked, and collect's provider is stubbed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from universe_builder import cli, settings, store
from universe_builder.domains.history.models.city import CityExpanded, Population
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.types import BatchHandle, LLMResponse, Usage
from tests.conftest import make_population
from tests.test_operations import valid_city

MODEL = "gpt-5-nano"


def _seed_city(subject, periods):
    payload = json.loads(valid_city())
    payload["cityPeriodsOfHistory"] = list(periods)
    path = store.path_for("city", None, subject, MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestExpandBatchSubmission:
    def _capture(self):
        seen = []

        def fake_submit(atom, items, model, **kw):
            seen.append((atom, list(items), kw))
            return [store.dir_for("city") / "handle.json"]

        return seen, fake_submit

    def test_one_request_per_unexpanded_period(self, workspace):
        _seed_city("Memphis", ["Old Kingdom", "Ptolemaic", "Roman"])
        seen, fake = self._capture()

        with patch("universe_builder.providers.batch.submit", fake):
            assert cli.main(["run", "--phase", "expand", "--atom", "city",
                             "--batch", "--model", MODEL]) == 0

        atom, items, kw = seen[0]
        assert atom == "expand_city"
        assert kw["operation"] == "expand" and kw["schema_model"] == "Population"
        assert len(items) == 3, "one request per period"
        assert [it.routing["period_index"] for it in items] == [0, 1, 2]
        # Every request carries the whole city and points at its file.
        target = str(store.path_for("city", None, "Memphis", MODEL)
                     .relative_to(settings.data_dir()))
        assert {it.target for it in items} == {target}
        assert all(json.loads(it.payload)["cityPeriodsOfHistory"] for it in items)

    def test_already_expanded_periods_are_not_resubmitted(self, workspace):
        """A resumed run asks only for what is still a bare name."""
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [make_population(periodName="Old Kingdom"),
                                           "Ptolemaic"]
        path = store.path_for("city", None, "Memphis", MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        seen, fake = self._capture()

        with patch("universe_builder.providers.batch.submit", fake):
            cli.main(["run", "--phase", "expand", "--atom", "city",
                      "--batch", "--model", MODEL])

        _, items, _ = seen[0]
        assert [it.routing["period_index"] for it in items] == [1], "only the string period"

    def test_a_fully_expanded_corpus_submits_nothing(self, workspace):
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [make_population(periodName="Old Kingdom")]
        path = store.path_for("city", None, "Memphis", MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        seen, fake = self._capture()

        with patch("universe_builder.providers.batch.submit", fake):
            assert cli.main(["run", "--phase", "expand", "--atom", "city",
                             "--batch", "--model", MODEL]) == 1
        assert seen == []


class _FakeProvider:
    def __init__(self, responses):
        self._responses = responses
        self.model = MODEL
        self.supports_batch = True

    def fetch_batch_results(self, handle):
        return list(self._responses)


class TestExpandBatchCollect:
    """Collect groups period-replies by city and splices each city once."""

    def _handle_and_city(self, workspace, periods):
        city = _seed_city("Memphis", periods)
        target = str(city.relative_to(settings.data_dir()))
        routing = {
            f"expand_city-0000{i}-memphis": {
                "atom_type": "expand_city", "subject": f"period {i}",
                "operation": "expand", "target": target,
                "period_index": i, "period_name": p,
            }
            for i, p in enumerate(periods) if isinstance(p, str)
        }
        handle = BatchHandle(provider="openai", batch_id="b1", model=MODEL,
                             submitted_at="t", request_count=len(routing), routing=routing)
        path = batch_mod._write(handle, settings.batch_dir() / "20260101T000000Z-expand_city.json")
        return city, path, routing

    def _response(self, custom_id, name):
        parsed = Population.model_validate(make_population(periodName=name))
        return LLMResponse(parsed.model_dump_json(), parsed, MODEL, Usage(), custom_id)

    def test_every_period_reply_is_spliced_into_one_city(self, workspace):
        city, path, routing = self._handle_and_city(workspace, ["Old Kingdom", "Ptolemaic"])
        ids = list(routing)
        responses = [self._response(ids[0], "Old Kingdom"), self._response(ids[1], "Ptolemaic")]

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            report = batch_mod.collect(path)

        reloaded = CityExpanded.model_validate_json(city.read_text())
        assert reloaded.is_fully_expanded(), "a period reply was not spliced in"
        assert report.written == 2

    def test_a_single_population_never_overwrites_the_city(self, workspace):
        """The bug this shape invites: `parsed` is one Population and the target
        is the city file, so a naive whole-atom write would replace the city
        with a lone period. The city must stay a city."""
        city, path, routing = self._handle_and_city(workspace, ["Old Kingdom", "Ptolemaic"])
        ids = list(routing)
        responses = [self._response(ids[0], "Old Kingdom"), self._response(ids[1], "Ptolemaic")]

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            batch_mod.collect(path)

        data = json.loads(city.read_text())
        assert isinstance(data.get("cityPeriodsOfHistory"), list)
        assert len(data["cityPeriodsOfHistory"]) == 2

    def test_a_failed_period_keeps_its_name_and_writes_no_sentinel(self, workspace):
        city, path, routing = self._handle_and_city(workspace, ["Old Kingdom", "Ptolemaic"])
        ids = list(routing)
        good = self._response(ids[0], "Old Kingdom")
        bad = LLMResponse("garbage", None, MODEL, Usage(), ids[1], "did not validate")

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider([good, bad])):
            batch_mod.collect(path)

        data = json.loads(city.read_text())
        periods = data["cityPeriodsOfHistory"]
        assert isinstance(periods[0], dict) and periods[1] == "Ptolemaic"
        assert not store.error_path(city).exists(), "a failed period must not sentinel the city"

    def _rewrite_periods(self, city, periods):
        """Simulate an edit between submit and collect (a split, a sync run)."""
        data = json.loads(city.read_text())
        data["cityPeriodsOfHistory"] = list(periods)
        city.write_text(json.dumps(data), encoding="utf-8")

    def test_a_shifted_index_is_placed_by_name_not_position(self, workspace):
        """The corruption the gate caught: an insertion before collect shifts
        every later index, so a positional splice would write each period one
        slot off. Placing by name lands them correctly regardless."""
        city, path, routing = self._handle_and_city(workspace, ["Old Kingdom", "Ptolemaic"])
        ids = list(routing)
        responses = [self._response(ids[0], "Old Kingdom"), self._response(ids[1], "Ptolemaic")]
        # A new period was inserted at the front since submit.
        self._rewrite_periods(city, ["New Kingdom", "Old Kingdom", "Ptolemaic"])

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            batch_mod.collect(path)

        periods = json.loads(city.read_text())["cityPeriodsOfHistory"]
        assert periods[0] == "New Kingdom", "the unrelated new period is untouched"
        assert isinstance(periods[1], dict) and periods[1]["periodName"] == "Old Kingdom"
        assert isinstance(periods[2], dict) and periods[2]["periodName"] == "Ptolemaic"

    def test_a_vanished_period_is_skipped_never_misplaced(self, workspace):
        """If the period a reply expanded is gone (renamed, split away) by
        collect, the reply is dropped — the period re-expands next run — rather
        than written onto whatever now sits at its old index."""
        city, path, routing = self._handle_and_city(workspace, ["Old Kingdom", "Ptolemaic"])
        ids = list(routing)
        responses = [self._response(ids[0], "Old Kingdom"), self._response(ids[1], "Ptolemaic")]
        # "Ptolemaic" was replaced by a different period since submit.
        self._rewrite_periods(city, ["Old Kingdom", "Roman"])

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            report = batch_mod.collect(path)

        periods = json.loads(city.read_text())["cityPeriodsOfHistory"]
        assert isinstance(periods[0], dict) and periods[0]["periodName"] == "Old Kingdom"
        assert periods[1] == "Roman", "the Ptolemaic reply must not land on Roman"
        assert report.written == 1, "only the still-present period counts as written"

    def test_recollect_does_not_double_write_an_already_expanded_period(self, workspace):
        """Splicing reads the file fresh and places only still-string indices,
        so collecting the same batch twice is idempotent."""
        city, path, routing = self._handle_and_city(workspace, ["Old Kingdom", "Ptolemaic"])
        ids = list(routing)
        responses = [self._response(ids[0], "Old Kingdom"), self._response(ids[1], "Ptolemaic")]

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            batch_mod.collect(path)
        first = city.read_text()
        # Re-write the handle (collect archived it) and collect again.
        batch_mod._write(routing and BatchHandle(
            provider="openai", batch_id="b1", model=MODEL, submitted_at="t",
            request_count=len(routing), routing=routing), path)
        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            batch_mod.collect(path)

        assert json.loads(city.read_text())["cityPeriodsOfHistory"] == \
            json.loads(first)["cityPeriodsOfHistory"]
