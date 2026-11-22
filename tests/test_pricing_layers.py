"""Pricing rows from a pack or `UNIVERSE_BUILDER_MODEL_PRICING`, laid over the
engine's `model_pricing.json` (v0.9.0): a new model is callable without an
engine release."""

from __future__ import annotations

import json
import os

import pytest

from universe_builder import config
from universe_builder.llm_classes.provider_call import ProviderCall, ProviderCallAugment
from universe_builder.providers import pricing
from universe_builder.providers.errors import UnknownModel
from universe_builder.providers.types import Usage

NEW = "vendor-test-9-sol"
ENGINE = "gpt-6.1-sol"


def _row(**overrides):
    row = {
        "provider": "openai",
        "input": 3.0,
        "cached_input": 0.3,
        "cache_write": None,
        "output": 12.0,
        "batch_discount": 0.5,
        "cache_min_tokens": 1024,
        "context": 400000,
    }
    row.update(overrides)
    return row


def _write(path, models):
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def test_an_unpriced_model_is_unknown_and_not_callable():
    with pytest.raises(UnknownModel, match=pricing.PRICING_ENV):
        pricing.lookup(NEW)
    assert NEW not in pricing.known_models()
    assert NEW not in config.llm_call_mapping
    with pytest.raises(KeyError):
        config.llm_call_mapping[NEW]


def test_the_env_file_adds_a_callable_priced_model(tmp_path, monkeypatch):
    monkeypatch.setenv(pricing.PRICING_ENV, str(_write(tmp_path / "p.json", {NEW: _row()})))

    info = pricing.lookup(NEW)
    assert (info.provider, info.input, info.output) == ("openai", 3.0, 12.0)
    assert pricing.provider_for(NEW) == "openai"
    assert NEW in pricing.known_models()
    assert NEW not in pricing.engine_models()
    assert config.llm_call_mapping[NEW] is ProviderCall
    assert config.llm_call_mapping[(NEW, "augment")] is ProviderCallAugment
    assert NEW in config.llm_call_mapping and (NEW, "augment") in config.llm_call_mapping
    cost = pricing.estimate_cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000), NEW, batch=True)
    assert cost == pytest.approx((3.0 + 12.0) * 0.5)
    # The engine's rows are still there underneath.
    assert pricing.lookup(ENGINE).input == 2.0


def test_a_layered_row_replaces_the_engines(tmp_path, monkeypatch):
    monkeypatch.setenv(pricing.PRICING_ENV, str(_write(tmp_path / "p.json", {ENGINE: _row(input=1.0)})))
    assert pricing.lookup(ENGINE).input == 1.0
    assert pricing._MODELS[ENGINE].input == 2.0


def test_the_pack_file_is_read_and_the_env_file_wins_over_it(tmp_path, monkeypatch, pack_overrides):
    pack_file = _write(tmp_path / "pack.json", {NEW: _row(input=4.0), ENGINE: _row(input=5.0)})
    pack_overrides(model_pricing_path=pack_file)
    assert pricing.lookup(NEW).input == 4.0
    assert pricing.lookup(ENGINE).input == 5.0
    assert config.llm_call_mapping[NEW] is ProviderCall

    monkeypatch.setenv(pricing.PRICING_ENV, str(_write(tmp_path / "env.json", {NEW: _row(input=6.0)})))
    assert pricing.extra_pricing_paths() == [pack_file, tmp_path / "env.json"]
    assert pricing.lookup(NEW).input == 6.0
    assert pricing.lookup(ENGINE).input == 5.0


@pytest.mark.parametrize("content, match", [
    (None, "cannot be read"),
    ("{not json", "cannot be read"),
    ('{"rows": {}}', 'no "models" object'),
    (json.dumps({"models": {NEW: _row(price=1)}}), NEW),
    (json.dumps({"models": {NEW: {"provider": "openai"}}}), NEW),
    (json.dumps({"models": {NEW: "2.0"}}), "a row is an object"),
    (json.dumps({"models": {NEW: _row(input="2.0")}}), "input must be a non-negative number"),
    (json.dumps({"models": {NEW: _row(output=-1)}}), "output must be a non-negative number"),
    (json.dumps({"models": {NEW: _row(cached_input=True)}}), "cached_input must be"),
    ('{"models": {"%s": %s}}' % (NEW, json.dumps(_row()).replace('"input": 3.0', '"input": NaN')),
     "input must be a non-negative number"),
    ('{"models": {"%s": %s}}' % (NEW, json.dumps(_row()).replace('"output": 12.0', '"output": Infinity')),
     "output must be a non-negative number"),
    ('{"models": {"%s": %s}}' % (NEW, json.dumps(_row()).replace('"cache_write": null', '"cache_write": NaN')),
     "cache_write must be"),
    ('{"models": {"%s": %s}}' % (NEW, json.dumps(_row()).replace('"batch_discount": 0.5', '"batch_discount": NaN')),
     "batch_discount must be"),
    (json.dumps({"models": {NEW: _row(cache_write="1")}}), "cache_write must be"),
    (json.dumps({"models": {NEW: _row(batch_discount=0)}}), "batch_discount must be"),
    (json.dumps({"models": {NEW: _row(batch_discount=1.5)}}), "batch_discount must be"),
    (json.dumps({"models": {NEW: _row(context=0)}}), "context must be"),
    (json.dumps({"models": {NEW: _row(cache_min_tokens=1024.5)}}), "cache_min_tokens must be"),
    (json.dumps({"models": {NEW: _row(max_output="128000")}}), "max_output must be"),
    (json.dumps({"models": {NEW: _row(effort="no")}}), "effort must be"),
    (json.dumps({"models": {NEW: _row(effort_levels="high")}}), "effort_levels must be"),
    (json.dumps({"models": {NEW: _row(provider="")}}), "provider must be"),
    (json.dumps({"models": {NEW: _row(provider="opneai")}}), "provider 'opneai' is not one of"),
])
def test_a_declared_file_that_cannot_be_read_stops_the_lookup(tmp_path, monkeypatch, content, match):
    path = tmp_path / "p.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setenv(pricing.PRICING_ENV, str(path))
    with pytest.raises(ValueError, match=match) as err:
        pricing.lookup(ENGINE)
    assert str(path) in str(err.value)


def test_an_edited_file_is_read_again(tmp_path, monkeypatch):
    path = _write(tmp_path / "p.json", {NEW: _row(input=3.0)})
    monkeypatch.setenv(pricing.PRICING_ENV, str(path))
    assert pricing.lookup(NEW).input == 3.0

    before = path.stat().st_mtime_ns
    _write(path, {NEW: _row(input=7.0)})          # same size: only the mtime moves
    os.utime(path, ns=(before + 1_000_000_000, before + 1_000_000_000))
    assert pricing.lookup(NEW).input == 7.0

    _write(path, {})
    with pytest.raises(UnknownModel):
        pricing.lookup(NEW)

    path.unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        pricing.lookup(NEW)


def test_the_engine_table_passes_the_row_checks():
    assert pricing._load(pricing.settings.MODEL_PRICING) == pricing._MODELS
