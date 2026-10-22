"""Tests for the multi-provider layer.

Fully mocked - no API keys and no network. Run from the repository root:
    python -m pytest tests/test_providers.py -v
"""

from __future__ import annotations

import json
import tempfile
import os
import pathlib
import unittest
from unittest.mock import MagicMock, patch

import pytest

from universe_builder import settings
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.domains.history.models.person import Person
from universe_builder.providers.pricing import cache_warning, estimate_cost, lookup
from universe_builder.providers.requests import build_request, build_requests, prefix_tokens
from universe_builder.providers.types import Usage


# ---------------------------------------------------------------------------
# Request assembly and the stable/volatile split
# ---------------------------------------------------------------------------


class TestRequestBuilding(unittest.TestCase):
    def test_subject_goes_in_user_turn_not_system(self):
        request = build_request("person", "Genghis Khan", "claude-sonnet-5")
        self.assertIn("Genghis Khan", request.user)
        self.assertNotIn("Genghis Khan", request.system)

    def test_cache_key_is_identical_across_subjects(self):
        """The whole point: every topic of an atom type shares one cached prefix."""
        a = build_request("person", "Genghis Khan", "claude-sonnet-5")
        b = build_request("person", "Hatshepsut", "claude-sonnet-5")
        self.assertEqual(a.cache_key(), b.cache_key())

    def test_cache_key_differs_by_atom_type(self):
        a = build_request("person", "X", "claude-sonnet-5")
        b = build_request("event", "X", "claude-sonnet-5")
        self.assertNotEqual(a.cache_key(), b.cache_key())

    def test_cache_key_differs_by_operation(self):
        a = build_request("person", "X", "claude-sonnet-5", operation="create")
        b = build_request("person", "X", "claude-sonnet-5", operation="verify")
        self.assertNotEqual(a.cache_key(), b.cache_key())

    def test_payload_goes_in_user_turn(self):
        payload = '{"name": {"common": "test"}}'
        request = build_request(
            "person", "Someone", "claude-sonnet-5", operation="fix", payload=payload
        )
        self.assertIn(payload, request.user)
        self.assertNotIn(payload, request.system)

    def test_operation_instructions_are_in_the_cached_system_text(self):
        request = build_request("person", "X", "claude-sonnet-5", operation="fix")
        self.assertIn("failed schema validation", request.system)

    def test_example_loaded_when_configured(self):
        request = build_request("person", "X", "claude-sonnet-5")
        # `person` is registered few-shot (genghis_khan + ptahhotep); the first
        # example is stable so the cached prefix does not move.
        self.assertEqual(len(request.examples), 2)
        self.assertIn("genghis", request.examples[0].user.lower())

    def test_no_example_when_config_says_null(self):
        """`append_city` is the remaining atom registered without an example."""
        request = build_request("append_city", "X", "claude-sonnet-5")
        self.assertEqual(request.examples, ())

    def test_every_base_atom_has_a_registered_example(self):
        """A missing example shrinks the cached prefix. Every base atom is now
        registered few-shot with two. `city` was the last on one: the corpus
        holds no seed-shape `City` to promote (every stored city is
        post-expand), so `sardis.json` was authored by hand instead."""
        expected = {"person": 2, "event": 2, "city": 2, "discovery": 2, "artifact": 2}
        for atom, n in expected.items():
            with self.subTest(atom=atom):
                self.assertEqual(len(build_request(atom, "X", "claude-sonnet-5").examples), n)

    def test_discovery_resolves_to_its_model(self):
        self.assertIs(build_request("discovery", "X", "claude-sonnet-5").schema_model, Discovery)

    def test_schema_model_resolved_from_config(self):
        self.assertIs(build_request("person", "X", "claude-sonnet-5").schema_model, Person)

    def test_build_requests_assigns_unique_custom_ids(self):
        requests = build_requests("person", ["Alpha", "Beta", "Gamma"], "claude-sonnet-5")
        ids = [r.custom_id for r in requests]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(i.startswith("person-") for i in ids))

    def test_build_requests_share_one_cache_key(self):
        requests = build_requests("person", ["Alpha", "Beta", "Gamma"], "claude-sonnet-5")
        self.assertEqual(len({r.cache_key() for r in requests}), 1)

    def test_example_json_file_accepts_a_list(self):
        """Few-shot: more examples means a bigger cached prefix, which costs
        roughly a tenth as much to reuse as it does to send fresh."""
        from universe_builder.providers.requests import _examples_for

        spec = {
            "prompt_start_text": "Provide a response for ",
            "example_json_file": ["genghis_khan.json", "rosetta_stone.json"],
        }
        examples = _examples_for(spec)
        self.assertEqual(len(examples), 2)
        self.assertIn("genghis khan", examples[0].user)
        self.assertIn("rosetta stone", examples[1].user)

    def test_example_json_file_still_accepts_a_bare_string(self):
        from universe_builder.providers.requests import _examples_for

        spec = {"prompt_start_text": "x ", "example_json_file": "genghis_khan.json"}
        self.assertEqual(len(_examples_for(spec)), 1)

    def test_example_order_is_preserved(self):
        """Reordering examples changes the cached prefix bytes and voids the cache."""
        from universe_builder.providers.requests import _examples_for

        base = {"prompt_start_text": "x "}
        a = _examples_for({**base, "example_json_file": ["genghis_khan.json", "rosetta_stone.json"]})
        b = _examples_for({**base, "example_json_file": ["rosetta_stone.json", "genghis_khan.json"]})
        self.assertNotEqual([e.user for e in a], [e.user for e in b])

    def test_mismatched_example_is_rejected(self):
        """An example that doesn't match its schema teaches the wrong shape."""
        from universe_builder.providers.errors import ProviderError
        from universe_builder.providers.requests import _examples_for

        spec = {"prompt_start_text": "x ", "example_json_file": "rosetta_stone.json"}
        with self.assertRaises(ProviderError):
            _examples_for(spec, Discovery)

    def test_matching_example_is_accepted(self):
        from universe_builder.providers.requests import _examples_for

        spec = {"prompt_start_text": "x ", "example_json_file": "invention_of_toothpaste.json"}
        self.assertEqual(len(_examples_for(spec, Discovery)), 1)

    def test_all_registered_examples_validate(self):
        """Every atom in the active domain pack must have examples that match it.

        `expand_city` mirrors its real callers: the registered class is the
        seed the file holds, and every actual expand request overrides the
        schema to `Population`, the shape its example teaches."""
        from universe_builder import atoms

        for atom in atoms.atom_mapping:
            with self.subTest(atom=atom):
                overrides = {"schema_model": "Population"} if atom == "expand_city" else {}
                build_request(atom, "X", "claude-sonnet-5", **overrides)

    def test_unknown_atom_type_is_rejected(self):
        from universe_builder.providers.errors import ProviderError

        with self.assertRaises(ProviderError):
            build_request("nonsense", "X", "claude-sonnet-5")


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


def _anthropic_provider():
    from universe_builder.providers.anthropic_provider import AnthropicProvider

    with patch("anthropic.Anthropic"):
        return AnthropicProvider()


class TestFewShotCaching(unittest.TestCase):
    def _request(self, filenames):
        from universe_builder.providers.requests import _examples_for
        from universe_builder.providers.types import LLMRequest

        spec = {"prompt_start_text": "x ", "example_json_file": filenames}
        return LLMRequest(
            model="claude-sonnet-5", schema_model=Person, system="s", user="u",
            examples=_examples_for(spec),
        )

    def test_only_the_last_example_carries_the_breakpoint(self):
        """All examples must sit inside one cached region, not one breakpoint
        each - Anthropic allows only four."""
        provider = _anthropic_provider()
        request = self._request(["genghis_khan.json", "rosetta_stone.json", "memphis.json"])
        messages = provider._messages(request)
        assistants = [m for m in messages if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 3)
        marked = [m for m in assistants if "cache_control" in m["content"][0]]
        self.assertEqual(len(marked), 1)
        self.assertIs(marked[0], assistants[-1])

    def test_more_examples_means_a_larger_cached_prefix(self):
        from universe_builder.providers.requests import prefix_tokens

        one = prefix_tokens(self._request(["genghis_khan.json"]))
        three = prefix_tokens(self._request(["genghis_khan.json", "rosetta_stone.json", "memphis.json"]))
        self.assertGreater(three, one)


class TestAnthropicProvider(unittest.TestCase):
    def setUp(self):
        self.provider = _anthropic_provider()
        self.request = build_request("person", "Genghis Khan", "claude-sonnet-5")

    def test_system_block_carries_cache_control(self):
        blocks = self.provider._system_blocks(self.request)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})

    def test_schema_is_inside_the_cached_system_block(self):
        blocks = self.provider._system_blocks(self.request)
        self.assertIn("shortSummary", blocks[0]["text"])

    def test_last_example_carries_a_second_breakpoint(self):
        """Without this the few-shot examples are re-billed on every call. The
        breakpoint goes on the *last* example's assistant turn, so the whole
        example block (one or many) sits inside the cached prefix."""
        messages = self.provider._messages(self.request)
        assistants = [m for m in messages if m["role"] == "assistant"]
        self.assertIn("cache_control", assistants[-1]["content"][0])

    def test_volatile_turn_is_last_and_uncached(self):
        messages = self.provider._messages(self.request)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Genghis Khan", messages[-1]["content"])
        self.assertIsInstance(messages[-1]["content"], str)

    def test_no_cache_control_when_caching_disabled(self):
        request = build_request("person", "X", "claude-sonnet-5", cache=False)
        self.assertNotIn("cache_control", self.provider._system_blocks(request)[0])

    def test_temperature_omitted_by_default(self):
        """Sampling parameters are rejected on newer Opus models."""
        self.assertNotIn("temperature", self.provider._params(self.request))

    def test_temperature_sent_when_explicitly_set(self):
        request = build_request("person", "X", "claude-haiku-4-5-20251001", temperature=0.2)
        self.assertEqual(self.provider._params(request)["temperature"], 0.2)

    def test_batch_params_include_json_schema(self):
        params = self.provider._batch_params(self.request)
        fmt = params["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertFalse(fmt["schema"]["additionalProperties"])

    def test_usage_extraction(self):
        raw = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=4000, cache_creation_input_tokens=0,
        )
        usage = self.provider._usage(raw)
        self.assertEqual(usage.cache_read_tokens, 4000)
        self.assertEqual(usage.output_tokens, 50)

    def test_submit_batch_rejects_oversized_input(self):
        requests = [self.request] * 2
        self.provider.batch_max_requests = 1
        with self.assertRaises(ValueError):
            self.provider.submit_batch(requests)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def _openai_provider():
    from universe_builder.providers.openai_provider import OpenAIProvider

    with patch("openai.OpenAI"):
        return OpenAIProvider()


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.provider = _openai_provider()
        self.request = build_request("person", "Genghis Khan", "gpt-5-nano")

    def test_message_order_is_stable_then_volatile(self):
        messages = self.provider._messages(self.request)
        roles = [m["role"] for m in messages]
        # system, then one user/assistant pair per few-shot example, then the
        # single volatile user turn last. `person` is registered with two examples.
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")
        middle = roles[1:-1]
        self.assertEqual(middle, ["user", "assistant"] * len(self.request.examples))
        self.assertIn("Genghis Khan", messages[-1]["content"])

    def test_strict_schema_sets_additional_properties_false(self):
        schema = self.provider.strict_json_schema(Person)
        self.assertFalse(schema["additionalProperties"])
        for definition in schema.get("$defs", {}).values():
            if "properties" in definition:
                self.assertFalse(definition["additionalProperties"])

    def test_strict_schema_requires_every_property(self):
        schema = self.provider.strict_json_schema(Person)
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_response_format_is_json_schema_strict(self):
        body = self.provider._body(self.request)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["name"], "Person")

    def test_temperature_omitted_by_default(self):
        """B9's forward-facing guard, on the path that is actually live.

        gpt-5 reasoning models answer any explicit temperature with a 400. The
        legacy `OpenAICall` hard-coded 0.2 and so could never call them; this
        layer must send nothing unless a caller asks for it.
        """
        self.assertNotIn("temperature", self.provider._body(self.request))

    def test_no_sampling_parameter_of_any_kind_by_default(self):
        body = self.provider._body(self.request)
        self.assertEqual({"temperature", "top_p", "top_k"} & set(body), set())

    def test_temperature_sent_when_explicitly_set(self):
        request = build_request("person", "X", "gpt-5-nano", temperature=0.2)
        self.assertEqual(self.provider._body(request)["temperature"], 0.2)

    def test_batch_jsonl_line_shape(self):
        requests = build_requests("person", ["A", "B"], "gpt-5-nano")
        lines = self.provider._jsonl(requests).decode().strip().split("\n")
        self.assertEqual(len(lines), 2)
        record = json.loads(lines[0])
        self.assertEqual(record["url"], "/v1/chat/completions")
        self.assertEqual(record["method"], "POST")
        self.assertEqual(record["custom_id"], requests[0].custom_id)

    def test_usage_separates_cached_from_fresh_input(self):
        usage = self.provider._usage(
            {"prompt_tokens": 4200, "completion_tokens": 300,
             "prompt_tokens_details": {"cached_tokens": 4096}}
        )
        self.assertEqual(usage.cache_read_tokens, 4096)
        self.assertEqual(usage.input_tokens, 104)
        self.assertEqual(usage.output_tokens, 300)


class TestOpenRouterProvider(unittest.TestCase):
    def test_openrouter_reports_no_batch_support(self):
        from universe_builder.providers.openai_provider import OpenRouterProvider

        with patch("openai.OpenAI"):
            provider = OpenRouterProvider(api_key="test")
        self.assertFalse(provider.supports_batch)

    def test_slug_models_route_to_openrouter(self):
        from universe_builder.providers.pricing import provider_for

        self.assertEqual(provider_for("anthropic/claude-sonnet-5"), "openrouter")


class TestDeepSeekProvider(unittest.TestCase):
    def _provider(self):
        from universe_builder.providers.openai_provider import DeepSeekProvider

        with patch("openai.OpenAI"):
            return DeepSeekProvider(api_key="test")

    def test_falls_back_to_loose_json_mode(self):
        """DeepSeek has no json_schema, so strict enforcement is unavailable."""
        provider = self._provider()
        request = build_request("person", "X", "deepseek-chat")
        self.assertEqual(provider._response_format(request), {"type": "json_object"})

    def test_schema_still_reaches_the_prompt(self):
        """With no enforcement, the schema in the system text is all we have."""
        provider = self._provider()
        messages = provider._messages(build_request("person", "X", "deepseek-chat"))
        self.assertIn("shortSummary", messages[0]["content"])

    def test_uses_max_tokens_not_max_completion_tokens(self):
        provider = self._provider()
        body = provider._body(build_request("person", "X", "deepseek-chat"))
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_reads_deepseek_specific_cache_fields(self):
        usage = self._provider()._usage(
            {"prompt_tokens": 4200, "completion_tokens": 300,
             "prompt_cache_hit_tokens": 4096, "prompt_cache_miss_tokens": 104}
        )
        self.assertEqual(usage.cache_read_tokens, 4096)
        self.assertEqual(usage.input_tokens, 104)

    def test_no_batch_support(self):
        self.assertFalse(self._provider().supports_batch)


class TestQwenProvider(unittest.TestCase):
    def _provider(self):
        from universe_builder.providers.openai_provider import QwenProvider

        with patch("openai.OpenAI"):
            return QwenProvider(api_key="test")

    def test_keeps_strict_schema_enforcement(self):
        provider = self._provider()
        fmt = provider._response_format(build_request("person", "X", "qwen3.5-flash"))
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["json_schema"]["strict"])

    def test_no_batch_support_until_verified(self):
        self.assertFalse(self._provider().supports_batch)


class TestCredentialIsolation(unittest.TestCase):
    """A compatible provider must never send OPENAI_API_KEY to a third party.

    `openai.OpenAI()` falls back to OPENAI_API_KEY from the environment when no
    key is passed. Combined with a third-party base_url, that transmits the
    user's OpenAI secret to another vendor's endpoint.
    """

    COMPAT = ("DeepSeekProvider", "QwenProvider", "OpenRouterProvider")

    def _cls(self, name):
        from universe_builder.providers import openai_provider

        return getattr(openai_provider, name)

    def _no_vendor_keys(self):
        return patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-USER-OPENAI-SECRET"},
            clear=False,
        )

    def setUp(self):
        self._saved = {
            k: os.environ.pop(k, None)
            for k in ("DEEPSEEK_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY", "OPENROUTER_API_KEY")
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_compat_providers_refuse_to_borrow_the_openai_key(self):
        from universe_builder.providers.errors import ProviderNotConfigured

        with self._no_vendor_keys():
            for name in self.COMPAT:
                with self.subTest(provider=name):
                    with self.assertRaises(ProviderNotConfigured):
                        self._cls(name)()

    def test_the_refusal_names_the_variable_to_set(self):
        from universe_builder.providers.errors import ProviderNotConfigured

        with self._no_vendor_keys():
            with self.assertRaises(ProviderNotConfigured) as ctx:
                self._cls("DeepSeekProvider")()
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_own_key_is_used_and_not_the_openai_one(self):
        with self._no_vendor_keys(), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds-own"}):
            with patch("openai.OpenAI") as mock_client:
                self._cls("DeepSeekProvider")()
        self.assertEqual(mock_client.call_args.kwargs["api_key"], "sk-ds-own")

    def test_first_party_openai_still_uses_the_sdk_env_lookup(self):
        from universe_builder.providers.openai_provider import OpenAIProvider

        self.assertEqual(OpenAIProvider.env_vars, ())
        with self._no_vendor_keys():
            with patch("openai.OpenAI") as mock_client:
                OpenAIProvider()
        self.assertNotIn("api_key", mock_client.call_args.kwargs)


class TestCompatRequestShape(unittest.TestCase):
    def _provider(self, name, **kw):
        from universe_builder.providers import openai_provider

        with patch("openai.OpenAI"):
            return getattr(openai_provider, name)(api_key="test", **kw)

    def test_openrouter_uses_max_tokens_for_non_openai_backends(self):
        provider = self._provider("OpenRouterProvider")
        body = provider._body(build_request("person", "X", "anthropic/claude-sonnet-5"))
        self.assertIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)

    def test_deepseek_prompt_contains_the_literal_lowercase_json_token(self):
        """json_object mode is refused unless the prompt contains 'json'."""
        provider = self._provider("DeepSeekProvider")
        system = provider._messages(build_request("person", "X", "deepseek-chat"))[0]["content"]
        self.assertIn("json", system)

    def test_schema_providers_do_not_get_the_json_nudge(self):
        provider = _openai_provider()
        system = provider._messages(build_request("person", "X", "gpt-5-nano"))[0]["content"]
        self.assertNotIn("single json object and nothing else", system)

    def test_no_batch_providers_raise_batch_not_supported(self):
        from universe_builder.providers.errors import BatchNotSupported

        for name in ("DeepSeekProvider", "QwenProvider", "OpenRouterProvider"):
            provider = self._provider(name)
            with self.subTest(provider=name):
                self.assertFalse(provider.supports_batch)
                with self.assertRaises(BatchNotSupported):
                    provider.submit_batch(build_requests("person", ["A"], "deepseek-chat"))

    def test_openai_batch_still_works(self):
        provider = _openai_provider()
        self.assertTrue(provider.supports_batch)
        payload = provider._jsonl(build_requests("person", ["A", "B"], "gpt-5-nano"))
        self.assertEqual(len(payload.decode().strip().split("\n")), 2)


class TestProviderRouting(unittest.TestCase):
    def test_every_registry_model_routes_to_a_known_provider(self):
        from universe_builder.providers.pricing import known_models, provider_for
        from universe_builder.providers.registry import provider_names

        names = set(provider_names())
        for model in known_models():
            self.assertIn(provider_for(model), names, model)

    def test_openai_still_uses_max_completion_tokens(self):
        provider = _openai_provider()
        body = provider._body(build_request("person", "X", "gpt-5-nano"))
        self.assertIn("max_completion_tokens", body)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing(unittest.TestCase):
    def test_cost_of_one_million_input_tokens(self):
        cost = estimate_cost(Usage(input_tokens=1_000_000), "claude-sonnet-5")
        self.assertAlmostEqual(cost, 2.0, places=4)   # $2/MTok list price

    def test_batch_halves_the_cost(self):
        usage = Usage(input_tokens=1_000_000)
        sync = estimate_cost(usage, "claude-sonnet-5")
        batch = estimate_cost(usage, "claude-sonnet-5", batch=True)
        self.assertAlmostEqual(batch, sync / 2, places=6)

    def test_cache_reads_are_a_tenth_of_input(self):
        fresh = estimate_cost(Usage(input_tokens=1_000_000), "claude-sonnet-5")
        cached = estimate_cost(Usage(cache_read_tokens=1_000_000), "claude-sonnet-5")
        self.assertAlmostEqual(cached, fresh / 10, places=6)

    def test_unknown_model_returns_none_rather_than_raising(self):
        self.assertIsNone(estimate_cost(Usage(input_tokens=1000), "not-a-real-model"))

    def test_cache_warning_fires_below_the_minimum(self):
        self.assertIsNotNone(cache_warning("claude-haiku-4-5-20251001", 1000))

    def test_cache_warning_silent_above_the_minimum(self):
        self.assertIsNone(cache_warning("claude-haiku-4-5-20251001", 5000))

    def test_most_atoms_are_below_the_haiku_cache_minimum(self):
        """Haiku 4.5 needs a 4096-token prefix - the highest of the current
        models - and registering the one-shot examples was not enough to clear
        it for most atoms. Caching is largely inert on Haiku for this pipeline;
        Sonnet 5 (1024) and Opus 5 (512) clear it comfortably.

        `city` left this list when `sardis.json` gave it a second
        example. That is the mechanism working as intended rather than a
        threshold drifting: a second example is what pushes a prefix over the
        Haiku minimum, and `person` cleared it the same way. Adding one to any
        remaining atom here should move it off this list too."""
        model = "claude-haiku-4-5-20251001"
        uncacheable = [
            atom
            for atom in ("person", "event", "city", "discovery", "artifact")
            if cache_warning(model, prefix_tokens(build_request(atom, "X", model)))
        ]
        self.assertEqual(sorted(uncacheable), ["artifact", "discovery", "event"])

    def test_sonnet_caches_every_atom(self):
        model = "claude-sonnet-5"
        for atom in ("person", "event", "city", "discovery", "artifact"):
            with self.subTest(atom=atom):
                self.assertIsNone(cache_warning(model, prefix_tokens(build_request(atom, "X", model))))

    def test_every_registry_entry_is_complete(self):
        from universe_builder.providers.pricing import known_models

        for name in known_models():
            info = lookup(name)
            self.assertGreater(info.input, 0, name)
            self.assertGreater(info.output, 0, name)
            self.assertGreater(info.cache_min_tokens, 0, name)


# ---------------------------------------------------------------------------
# Batch routing
# ---------------------------------------------------------------------------


class TestBatchRouting(unittest.TestCase):
    def test_custom_id_round_trips_to_the_expected_path(self):
        from universe_builder.providers.batch import _output_path
        from universe_builder.providers.types import BatchHandle

        handle = BatchHandle(
            provider="anthropic", batch_id="msgbatch_x", model="claude-sonnet-5",
            submitted_at="2026-09-10T00:00:00Z", request_count=1,
            routing={"person-00007-genghis_khan": {"atom_type": "person", "subject": "Genghis Khan"}},
        )
        path = _output_path(handle, "person-00007-genghis_khan")
        self.assertEqual(path.parent.name, "person")
        self.assertTrue(path.name.endswith("claude-sonnet-5.json"))
        # The id is derived from the subject, not from 00007 — the index is the
        # subject's position in the submitted list, which is stable within one
        # submission and meaningless across two.
        self.assertNotIn("0000000007", path.name)
        self.assertIn("genghis_khan", path.name)

    def test_the_batch_path_and_the_synchronous_path_agree(self):
        """One topic, one file, whichever pass generated it.

        While the batch pass used the submission index as the id, generating a
        topic synchronously and then again in a batch produced two files for the
        same topic, differing only in an id that encoded nothing.
        """
        from universe_builder import store
        from universe_builder.providers.batch import _output_path
        from universe_builder.providers.types import BatchHandle

        handle = BatchHandle(
            provider="anthropic", batch_id="b", model="claude-sonnet-5",
            submitted_at="x", request_count=1,
            routing={"person-00007-genghis_khan": {"atom_type": "person", "subject": "Genghis Khan"}},
        )

        self.assertEqual(
            _output_path(handle, "person-00007-genghis_khan").name,
            store.path_for("person", None, "Genghis Khan", "claude-sonnet-5").name,
        )

    def test_the_index_does_not_affect_the_path(self):
        """Resubmitting a topic in a different position must not rename it."""
        from universe_builder.providers.batch import _output_path
        from universe_builder.providers.types import BatchHandle

        def path_for_index(index: str) -> str:
            custom_id = f"person-{index}-genghis_khan"
            handle = BatchHandle(
                provider="anthropic", batch_id="b", model="claude-sonnet-5",
                submitted_at="x", request_count=1,
                routing={custom_id: {"atom_type": "person", "subject": "Genghis Khan"}},
            )
            return _output_path(handle, custom_id).name

        self.assertEqual(path_for_index("00007"), path_for_index("00042"))

    def test_append_atoms_are_written_to_the_base_folder(self):
        from universe_builder.providers.batch import _output_path
        from universe_builder.providers.types import BatchHandle

        handle = BatchHandle(
            provider="anthropic", batch_id="b", model="claude-sonnet-5",
            submitted_at="x", request_count=1,
            routing={"append_city-00001-memphis": {"atom_type": "append_city", "subject": "Memphis"}},
        )
        self.assertEqual(_output_path(handle, "append_city-00001-memphis").parent.name, "city")

    def test_handle_survives_a_json_round_trip(self):
        from universe_builder.providers.types import BatchHandle

        original = BatchHandle(
            provider="openai", batch_id="batch_abc", model="gpt-5-nano",
            submitted_at="2026-09-10T00:00:00Z", request_count=42,
            routing={"a": {"subject": "A"}}, extra={"schema_model": "Person"},
        )
        restored = BatchHandle.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored.batch_id, original.batch_id)
        self.assertEqual(restored.extra["schema_model"], "Person")
        self.assertEqual(restored.request_count, 42)


# ---------------------------------------------------------------------------
# Pipeline adapter
# ---------------------------------------------------------------------------


def real_fix_system_text(topic: str) -> str:
    """Byte-for-byte the shape `execute_prompt.process_fix_operation` builds.

    Testing the adapter with a clean system string proves nothing - the point is
    that the pipeline hands it a topic-laden one.
    """
    return f"""
            IMPORTANT: Please fix the json provided above.
            Read through all the fields in JSON first, understand the full context,
            break it down into individual questions per json object
            then make it more accurate to the best of your knowledge
            including filling in blank and null fields where you know the answer.
            Context is {topic}
        """


@pytest.mark.usefixtures("workspace")
class TestProviderCallAdapter(unittest.TestCase):
    """The synchronous call builds its request with the batch builder
    whatever system text the call site hands it, the
    request carries the persona, the context line and the operation's
    instructions, and the volatile half goes in the user turn."""

    def _augment_call(self, topic="genghis khan", system=None, **kwargs):
        from universe_builder.llm_classes.provider_call import ProviderCallAugment

        kwargs.setdefault("operation", "fix")
        with patch("universe_builder.llm_classes.provider_call.complete") as mock_complete:
            mock_complete.return_value = MagicMock(
                ok=True, raw_text='{"ok": true}', usage=Usage(), error=None
            )
            return ProviderCallAugment(
                atom_type="person",
                atom_class=Person,
                model_version="claude-sonnet-5",
                prompt_folder="prompts",
                prompt_subject=topic,
                prompt_start_text="Provide a response for ",
                base_system_text=system if system is not None else real_fix_system_text(topic),
                example_json_file=None,
                augment_to_json='{"name": {"common": "Genghis Khan"}}',
                topic=topic,
                **kwargs,
            )

    def test_topic_stays_out_of_the_cached_system_text(self):
        """Even when the call site's own system text carries it."""
        request = self._augment_call().build_request()
        self.assertNotIn("genghis khan", request.system.lower())

    def test_the_system_text_is_the_persona_then_the_operation(self):
        from universe_builder.providers.requests import base_system_text, operation_text

        request = self._augment_call().build_request()
        self.assertEqual(request.system, f"{base_system_text()}\n\n{operation_text('fix')}")

    def test_the_request_is_the_batch_builders(self):
        """Equal to `build_request` for the same work but for the sync
        path's own `custom_id` and metadata."""
        from universe_builder.providers.requests import build_request

        sync = self._augment_call().build_request()
        batch = build_request("person", "genghis khan", "claude-sonnet-5", operation="fix",
                              payload='{"name": {"common": "Genghis Khan"}}', schema_model="Person",
                              max_tokens=sync.max_tokens)
        self.assertEqual((sync.system, sync.user, sync.examples, sync.schema_model),
                         (batch.system, batch.user, batch.examples, batch.schema_model))
        self.assertEqual(sync.custom_id, "genghis khan")
        self.assertEqual(sync.metadata, {"atom_type": "person", "subject": "genghis khan"})

    def test_topic_is_restated_in_the_user_turn(self):
        request = self._augment_call().build_request()
        self.assertIn("genghis khan", request.user.lower())

    def test_payload_goes_in_the_user_turn(self):
        request = self._augment_call().build_request()
        self.assertIn('"common": "Genghis Khan"', request.user)
        self.assertNotIn('"common": "Genghis Khan"', request.system)

    def test_cache_key_is_stable_across_topics_with_real_pipeline_input(self):
        """The regression that matters: one cached prefix for the whole fix pass."""
        keys = {
            self._augment_call(topic=t).build_request().cache_key()
            for t in ("genghis khan", "hatshepsut", "djoser", "ramesses ii")
        }
        self.assertEqual(len(keys), 1)

    def test_create_path_keeps_subject_out_of_system(self):
        from universe_builder.llm_classes.provider_call import ProviderCall

        with patch("universe_builder.llm_classes.provider_call.complete") as mock_complete:
            mock_complete.return_value = MagicMock(
                ok=True, raw_text="{}", usage=Usage(), error=None
            )
            call = ProviderCall(
                atom_type="person", atom_class=Person, model_version="claude-sonnet-5",
                prompt_folder="prompts", prompt_subject="Hatshepsut",
                prompt_start_text="Provide a response for ",
                base_system_text="You are a historian. The context is Ancient Egypt!",
                example_json_file=None, context="Ancient Egypt",
            )
        request = call.build_request()
        self.assertIn("Hatshepsut", request.user)
        self.assertNotIn("Hatshepsut", request.system)
        # The civilisation is constant across a History file, so it stays cached.
        self.assertIn("The context is Ancient Egypt.", request.system)


# ---------------------------------------------------------------------------
# Batch failures must never be dropped silently
# ---------------------------------------------------------------------------


def _handle(provider="openai", model="gpt-5-nano", ids=("person-00000-a", "person-00001-b")):
    from universe_builder.providers.types import BatchHandle

    return BatchHandle(
        provider=provider, batch_id="batch_test", model=model,
        submitted_at="2026-09-10T00:00:00Z", request_count=len(ids),
        routing={i: {"atom_type": "person", "subject": i} for i in ids},
        extra={"schema_model": "Person"},
    )


class TestBatchFailureHandling(unittest.TestCase):
    def test_openai_reads_the_error_file(self):
        """API-level failures go to a separate error file and would otherwise
        never be seen by the caller."""
        provider = _openai_provider()
        handle = _handle()
        provider._client.batches.retrieve.return_value = MagicMock(
            status="completed", output_file_id=None, error_file_id="file_err"
        )
        provider._client.files.content.return_value = MagicMock(
            text=json.dumps({"custom_id": "person-00001-b", "error": {"message": "rate limited"}})
        )
        results = list(provider.fetch_batch_results(handle))
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertIn("rate limited", results[0].error)
        self.assertEqual(results[0].custom_id, "person-00001-b")

    def test_openai_terminal_batch_with_no_files_fails_every_request(self):
        provider = _openai_provider()
        handle = _handle()
        provider._client.batches.retrieve.return_value = MagicMock(
            status="failed", output_file_id=None, error_file_id=None
        )
        results = list(provider.fetch_batch_results(handle))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not r.ok for r in results))
        self.assertEqual({r.custom_id for r in results}, set(handle.routing))

    def test_google_failed_job_fails_every_request(self):
        from universe_builder.providers.google_provider import GoogleProvider

        provider = GoogleProvider.__new__(GoogleProvider)
        provider._client = MagicMock()
        state = MagicMock()
        state.name = "JOB_STATE_FAILED"
        provider._client.batches.get.return_value = MagicMock(state=state)
        handle = _handle(provider="google", model="gemini-2.5-flash-lite")
        handle.extra["order"] = list(handle.routing)
        results = list(provider.fetch_batch_results(handle))
        self.assertEqual(len(results), 2)
        self.assertTrue(all("JOB_STATE_FAILED" in (r.error or "") for r in results))

    def test_expected_custom_ids_prefers_submission_order(self):
        from universe_builder.providers.util import expected_custom_ids

        handle = _handle()
        handle.extra["order"] = ["person-00001-b", "person-00000-a"]
        self.assertEqual(expected_custom_ids(handle), ["person-00001-b", "person-00000-a"])

    def test_expected_custom_ids_falls_back_to_routing(self):
        from universe_builder.providers.util import expected_custom_ids

        self.assertEqual(set(expected_custom_ids(_handle())), set(_handle().routing))


class TestRetryAccounting(unittest.TestCase):
    """A retry costs tokens too. Reporting only the last attempt under-reports spend."""

    def test_validation_retry_reports_both_attempts(self):
        from universe_builder.providers import run
        from universe_builder.providers.types import LLMResponse

        request = build_request("person", "X", "gpt-5-nano")
        bad = LLMResponse("{}", None, request.model, Usage(input_tokens=100, output_tokens=50),
                          request.custom_id, "schema error")
        good = LLMResponse("{}", Person.model_construct(), request.model,
                           Usage(input_tokens=100, output_tokens=60), request.custom_id)
        provider = MagicMock()
        provider.complete.side_effect = [bad, good]
        with patch.object(run, "get_provider", return_value=provider):
            result = run.complete(request, validation_retries=1)
        self.assertTrue(result.ok)
        self.assertEqual(result.usage.input_tokens, 200)
        self.assertEqual(result.usage.output_tokens, 110)

    def test_exhausted_retries_still_report_what_was_spent(self):
        from universe_builder.providers import run
        from universe_builder.providers.types import LLMResponse

        request = build_request("person", "X", "gpt-5-nano")
        bad = LLMResponse("{}", None, request.model, Usage(input_tokens=100, output_tokens=50),
                          request.custom_id, "schema error")
        provider = MagicMock()
        provider.complete.side_effect = [bad, bad]
        with patch.object(run, "get_provider", return_value=provider):
            result = run.complete(request, validation_retries=1)
        self.assertFalse(result.ok)
        self.assertEqual(result.usage.input_tokens, 200)


class TestBatchChunking(unittest.TestCase):
    """Every batch line carries the whole prompt, so the byte cap binds long
    before the request-count cap does."""

    def test_openai_byte_limit_binds_before_the_count_limit(self):
        provider = _openai_provider()
        per = provider.request_size_bytes(build_request("person", "X", "gpt-5-nano"))
        fits = provider.batch_max_bytes // per
        self.assertLess(fits, provider.batch_max_requests)

    def test_chunking_respects_the_byte_limit(self):
        provider = _openai_provider()
        requests = build_requests("person", [f"S{i}" for i in range(20)], "gpt-5-nano")
        provider.batch_max_bytes = provider.request_size_bytes(requests[0]) * 5
        chunks = provider.chunk_for_batch(requests)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            total = sum(provider.request_size_bytes(r) for r in chunk)
            self.assertLessEqual(total, provider.batch_max_bytes)

    def test_chunking_loses_no_requests(self):
        provider = _openai_provider()
        requests = build_requests("person", [f"S{i}" for i in range(20)], "gpt-5-nano")
        provider.batch_max_bytes = provider.request_size_bytes(requests[0]) * 3
        flat = [r for chunk in provider.chunk_for_batch(requests) for r in chunk]
        self.assertEqual([r.custom_id for r in flat], [r.custom_id for r in requests])

    def test_single_oversized_request_is_rejected_loudly(self):
        provider = _openai_provider()
        provider.batch_max_bytes = 10
        with self.assertRaises(ValueError):
            provider.chunk_for_batch(build_requests("person", ["X"], "gpt-5-nano"))

    def test_every_batch_provider_declares_a_byte_limit(self):
        from universe_builder.providers.anthropic_provider import AnthropicProvider
        from universe_builder.providers.google_provider import GoogleProvider
        from universe_builder.providers.openai_provider import OpenAIProvider

        for cls in (AnthropicProvider, OpenAIProvider, GoogleProvider):
            with self.subTest(provider=cls.name):
                self.assertGreater(cls.batch_max_bytes, 0)


class TestHandleNaming(unittest.TestCase):
    def test_same_second_submissions_do_not_overwrite(self):
        """Two submissions of one atom within a second previously collided,
        orphaning a batch that had already been submitted and paid for."""
        import tempfile

        from universe_builder.providers import batch as batch_mod

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "batch_dir", return_value=pathlib.Path(tmp)):
                a = batch_mod._save(_handle(), "20260910T000000Z-person")
                b = batch_mod._save(_handle(), "20260910T000000Z-person")
                c = batch_mod._save(_handle(), "20260910T000000Z-person")
        self.assertEqual(len({a, b, c}), 3)
        self.assertTrue(b.name.endswith("-01.json"))
        self.assertTrue(c.name.endswith("-02.json"))


class TestBatchDirectoryHygiene(unittest.TestCase):
    """A missing-result report must never be mistaken for a batch handle.

    Writing it beside the handles as `<stem>.missing.json` matched the `*.json`
    glob, so one incomplete batch broke every later `list`, `poll --all` and
    `collect --all`.
    """

    def _dir_with(self, tmp, extra_files):
        from universe_builder.providers import batch as batch_mod

        d = pathlib.Path(tmp)
        (d / "20260910T120000Z-person.json").write_text(json.dumps(_handle().to_dict()))
        for name, content in extra_files.items():
            target = d / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(content))
        return d, batch_mod

    def test_missing_report_in_subdir_is_not_listed_as_a_handle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d, batch_mod = self._dir_with(tmp, {"missing/20260910T120000Z-person.json": ["a", "b"]})
            with patch.object(settings, "batch_dir", return_value=d):
                found = batch_mod.list_batches()
            self.assertEqual([p.name for p in found], ["20260910T120000Z-person.json"])

    def test_every_listed_path_loads_as_a_handle(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d, batch_mod = self._dir_with(tmp, {"missing/x.json": ["a"], "stray.json": ["not", "a", "handle"]})
            with patch.object(settings, "batch_dir", return_value=d):
                for path in batch_mod.list_batches():
                    batch_mod.load(path)  # must not raise

    def test_stray_json_is_skipped_not_fatal(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d, batch_mod = self._dir_with(tmp, {"stray.json": ["not", "a", "handle"]})
            with patch.object(settings, "batch_dir", return_value=d):
                found = batch_mod.list_batches()
            self.assertEqual([p.name for p in found], ["20260910T120000Z-person.json"])

    def test_load_rejects_a_non_handle_with_a_clear_error(self):
        import tempfile

        from universe_builder.providers.errors import ProviderError

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "missing.json"
            path.write_text(json.dumps(["person-00001-b"]))
            from universe_builder.providers import batch as batch_mod

            with self.assertRaises(ProviderError):
                batch_mod.load(path)


class TestPollWriteBack(unittest.TestCase):
    def test_poll_updates_the_handle_in_place(self):
        import tempfile

        from universe_builder.providers import batch as batch_mod

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "out-of-tree.json"
            path.write_text(json.dumps(_handle().to_dict()))
            provider = MagicMock()
            provider.poll_batch.return_value = MagicMock(
                done=True, state="completed", succeeded=2, errored=0
            )
            elsewhere = pathlib.Path(tmp) / "_batches"
            with patch.object(batch_mod, "get_provider", return_value=provider), \
                 patch.object(settings, "batch_dir", return_value=elsewhere):
                batch_mod.poll(path)
            self.assertTrue(path.exists())
            self.assertFalse(elsewhere.exists(), "poll relocated the handle")


class TestCollectReconciliation(unittest.TestCase):
    def _collect(self, handle, responses, tmp):
        from universe_builder.providers import batch as batch_mod

        path = pathlib.Path(tmp) / "handle.json"
        path.write_text(json.dumps(handle.to_dict()))
        provider = MagicMock()
        provider.fetch_batch_results.return_value = iter(responses)
        batch_dir = pathlib.Path(tmp) / "_batches"
        with patch.object(batch_mod, "get_provider", return_value=provider), \
             patch.object(settings, "data_dir", return_value=pathlib.Path(tmp)), \
             patch.object(settings, "batch_dir", return_value=batch_dir), \
             patch.object(settings, "missing_dir", return_value=batch_dir / "missing"):
            return batch_mod.collect(path), path

    def test_a_complete_collect_archives_its_handle(self):
        """`collect --all` swept every handle in `_batches/` forever.

        The waste - re-downloading result files on every sweep - is the mild
        half. The damaging half: a batch collected in January stayed eligible, so
        an unrelated `collect --all` in March re-fetched it and rewrote its atoms,
        silently reverting any correction made to them in between.
        """
        import tempfile

        from universe_builder.domains.history.models.person import Person
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import LLMResponse

        handle = _handle(ids=("person-00000-a",))
        atom = Person.model_construct()
        with tempfile.TemporaryDirectory() as tmp:
            report, path = self._collect(
                handle,
                [LLMResponse("{}", atom, handle.model, Usage(), "person-00000-a")],
                tmp,
            )

            self.assertTrue(report.complete)
            self.assertIsNotNone(report.archived, "a complete collect left its handle in place")
            self.assertFalse(path.exists(), "the handle was copied rather than moved")
            self.assertTrue(report.archived.exists())

            with patch.object(settings, "batch_dir", return_value=pathlib.Path(tmp) / "_batches"):
                self.assertNotIn(
                    report.archived, batch_mod.list_batches(),
                    "an archived handle is still in the glob that --all sweeps, so it "
                    "will be collected again and overwrite its own atoms",
                )

    def test_an_incomplete_collect_keeps_its_handle(self):
        """Losing results must not retire the only record of what was submitted."""
        import tempfile

        from universe_builder.providers.types import LLMResponse

        handle = _handle(ids=("person-00000-a", "person-00001-b"))
        with tempfile.TemporaryDirectory() as tmp:
            report, path = self._collect(
                handle,
                [LLMResponse("{}", None, handle.model, Usage(), "person-00000-a", "bad")],
                tmp,
            )

            self.assertFalse(report.complete)
            self.assertIsNone(report.archived)
            self.assertTrue(
                path.exists(),
                "the handle was archived despite lost results, so the batch can no "
                "longer be re-collected once the cause is fixed",
            )

    def test_a_recorded_target_wins_over_a_reconstructed_path(self):
        """An augment result must land on the file its payload came from.

        `store.path_for` rebuilds a *plausible* path from the subject, which is
        the right answer for `create` and the wrong one for `fix`/`verify`: the
        stored file may carry a different model version, or be an `_ERROR`
        sentinel, and neither is recoverable from the subject.
        """
        from universe_builder import settings as settings_mod
        from universe_builder.providers import batch as batch_mod

        handle = _handle(ids=("city-00000-memphis",))
        handle.routing["city-00000-memphis"] = {
            "atom_type": "city", "subject": "Memphis",
            "target": "city/city_h1111111111111111_memphis_gpt-4o-mini.json_ERROR",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings_mod, "data_dir", return_value=pathlib.Path(tmp)):
                out = batch_mod._output_path(handle, "city-00000-memphis")
                want = pathlib.Path(tmp) / handle.routing["city-00000-memphis"]["target"]

            # `.resolve()` on both sides: macOS hands out /var/... tmp paths that
            # are symlinks to /private/var/..., and `_output_path` resolves.
            assert out.resolve() == want.resolve()

    def test_a_target_escaping_the_data_directory_is_refused(self):
        """A handle is a file on disk and could say anything.

        Honouring `../../etc/whatever` would let a tampered or corrupted handle
        write anywhere the process can reach, so the write falls back to the
        content-addressed path instead.
        """
        from universe_builder import settings as settings_mod
        from universe_builder.providers import batch as batch_mod

        handle = _handle(ids=("city-00000-memphis",))
        handle.routing["city-00000-memphis"] = {
            "atom_type": "city", "subject": "Memphis",
            "target": "../../escaped.json",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with patch.object(settings_mod, "data_dir", return_value=root):
                out = batch_mod._output_path(handle, "city-00000-memphis")

        assert "escaped.json" not in str(out), out
        assert out.name.startswith("city_")

    def test_a_failed_repair_keeps_the_payload_it_was_repairing(self):
        """A batched `fix` must not overwrite the sentinel it read its input from.

        `collect` wrote the failure to `error_path(target)`, which for a fix *is*
        the sentinel holding the original failing payload. So a repair that
        failed replaced the only copy of what needed repairing. Worst with an
        API-level failure, which carries no text at all: the payload became
        `// batch error: server overloaded`, and the item stayed queued with
        nothing left to work from. B8, one route further out.
        """
        import tempfile

        from universe_builder import store
        from universe_builder.providers.types import LLMResponse

        original = '{"name": {"common": "Memphis"}, "truncated mid-'
        handle = _handle(ids=("person-00000-a",))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "data_dir", return_value=pathlib.Path(tmp)):
                from universe_builder.providers import batch as batch_mod
                out = batch_mod._output_path(handle, "person-00000-a")
            store.write_error(out, original)
            sentinel = store.error_path(out)

            self._collect(
                handle,
                [LLMResponse("", None, handle.model, Usage(), "person-00000-a",
                             "batch error: server overloaded")],
                tmp,
            )

            assert sentinel.exists()
            assert original in store.read(sentinel), (
                "the repair's own failure replaced the payload it was repairing; "
                "the work item is queued with nothing to repair from"
            )

    def test_a_first_failure_is_still_recorded(self):
        """Preserving must not mean never recording."""
        import tempfile

        from universe_builder import store
        from universe_builder.providers.types import LLMResponse

        handle = _handle(ids=("person-00000-a",))
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._collect(
                handle,
                [LLMResponse('{"bad": true}', None, handle.model, Usage(), "person-00000-a", "nope")],
                tmp,
            )
            with patch.object(settings, "data_dir", return_value=pathlib.Path(tmp)):
                from universe_builder.providers import batch as batch_mod
                out = batch_mod._output_path(handle, "person-00000-a")

            assert report.failed == 1
            assert store.error_path(out).exists()
            assert '{"bad": true}' in store.read(store.error_path(out))

    def test_a_successful_result_retires_the_sentinel_it_supersedes(self):
        """A batch that repairs a failed topic must clear that topic's `_ERROR`.

        The synchronous path does this in `operations._support.persist`; the
        batch path did not, and the consequence is worse than the stale file it
        looks like. The fix pass takes its work from `_ERROR` sentinels and
        `commit_fix` writes the answer to the target path - so a surviving
        sentinel means the next fix pass feeds the *old failed payload* back to
        the model and overwrites the good batch atom with the result.
        """
        import tempfile

        from universe_builder import store
        from universe_builder.domains.history.models.person import Person
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import LLMResponse

        handle = _handle(ids=("person-00000-a",))
        atom = Person.model_construct()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(settings, "data_dir", return_value=pathlib.Path(tmp)):
                out = batch_mod._output_path(handle, "person-00000-a")
            store.write_error(out, '{"broken": true}')
            sentinel = store.error_path(out)
            self.assertTrue(sentinel.exists(), "precondition: the sentinel exists")

            report, _ = self._collect(
                handle,
                [LLMResponse("{}", atom, handle.model, Usage(), "person-00000-a")],
                tmp,
            )

            self.assertEqual(report.written, 1)
            self.assertTrue(out.exists(), "the batch result was written")
            self.assertFalse(
                sentinel.exists(),
                f"{sentinel.name} survived a successful batch result; the next fix "
                "pass will regenerate from the stale failed payload and overwrite "
                "the atom this batch just produced",
            )

    def test_missing_results_are_reported_not_silently_lost(self):
        import tempfile

        from universe_builder.providers.types import LLMResponse

        handle = _handle(ids=("person-00000-a", "person-00001-b", "person-00002-c"))
        with tempfile.TemporaryDirectory() as tmp:
            report, path = self._collect(
                handle,
                [LLMResponse("{}", None, handle.model, Usage(), "person-00000-a", "bad json")],
                tmp,
            )
            self.assertEqual(report.failed, 1)
            self.assertEqual(report.missing, ["person-00001-b", "person-00002-c"])
            self.assertFalse(report.complete)
            report_file = pathlib.Path(tmp) / "_batches" / "missing" / "handle.json"
            self.assertTrue(report_file.exists())
            self.assertEqual(json.loads(report_file.read_text()), report.missing)

    def test_complete_batch_reports_nothing_missing(self):
        import tempfile

        from universe_builder.providers.types import LLMResponse

        handle = _handle(ids=("person-00000-a",))
        person = Person.model_construct()
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._collect(
                handle, [LLMResponse("{}", person, handle.model, Usage(), "person-00000-a")], tmp
            )
            self.assertTrue(report.complete)
            self.assertEqual(report.written, 1)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# B10: strict structured outputs reject allOf and $ref-with-siblings
# ---------------------------------------------------------------------------

class TestStrictSchemaFlattening:
    """Pydantic emits `{"allOf": [{"$ref": ...}], "description": ...}` for any
    field that both references another model and carries a description. OpenAI
    strict mode rejects `allOf` outright, and separately rejects a `$ref` that
    carries sibling keys.

    Confirmed live — every atom in the corpus tripped this, so the
    provider path could not generate a single real atom:

        400 - Invalid schema for response_format 'Discovery':
        In context=('properties', 'ageOfRecord'), 'allOf' is not permitted.
    """

    #: Every model that is ever sent as a response schema. `Population` is not in
    #: `class_mapping` — the expand pass passes it directly as `expander_class` —
    #: so listing only the registry would have missed the one atom the expand
    #: pass depends on.
    ATOMS = ["Discovery", "City", "CityExpanded", "CityAppend",
             "Event", "Person", "Artifact", "History",
             "Polity", "Monument", "Work", "Figure"]

    @staticmethod
    def _offenders(node, path=""):
        out = []
        if isinstance(node, dict):
            if "allOf" in node:
                out.append(f"allOf at {path}")
            if "$ref" in node and len(node) > 1:
                out.append(f"$ref with siblings at {path}")
            for k, v in node.items():
                out += TestStrictSchemaFlattening._offenders(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                out += TestStrictSchemaFlattening._offenders(v, f"{path}[{i}]")
        return out

    @pytest.mark.parametrize("atom", ATOMS)
    def test_no_registered_atom_schema_contains_a_forbidden_construct(self, atom):
        from universe_builder.execute_prompt import class_mapping
        from universe_builder.providers.base import Provider

        schema = Provider.strict_json_schema(class_mapping[atom])
        assert self._offenders(schema) == []

    def test_the_expander_class_is_clean_too(self):
        """`Population` is sent as a response schema but is not in class_mapping."""
        from universe_builder.domains.history.models.city import Population
        from universe_builder.providers.base import Provider

        assert self._offenders(Provider.strict_json_schema(Population)) == []

    def test_every_registered_atom_is_covered(self):
        """A new atom type must be added to ATOMS above."""
        from universe_builder.execute_prompt import class_mapping

        assert set(self.ATOMS) == set(class_mapping)

    def test_the_raw_pydantic_schema_still_has_them(self):
        """If Pydantic stops emitting allOf, this flattening becomes dead code."""
        from universe_builder.domains.history.models.discovery import Discovery

        assert self._offenders(Discovery.model_json_schema()), (
            "Pydantic no longer emits allOf; strict_json_schema's flattening may "
            "be removable — check before deleting it."
        )

    def test_the_field_description_survives_inlining(self):
        """Descriptions are the per-field instructions the model reads."""
        from universe_builder.domains.history.models.discovery import Discovery
        from universe_builder.providers.base import Provider

        raw = Discovery.model_json_schema()["$defs"]["Finding"]["properties"]["ageOfRecord"]
        wanted = raw["description"]

        strict = Provider.strict_json_schema(Discovery)
        node = strict["$defs"]["Finding"]["properties"]["ageOfRecord"]
        assert node.get("description") == wanted
        assert "allOf" not in node

    def test_a_self_referencing_model_does_not_recurse_forever(self):
        """On a cycle the siblings are dropped and the bare $ref kept."""
        from universe_builder.providers.base import Provider

        defs = {"Node": {"type": "object",
                         "properties": {"child": {"$ref": "#/$defs/Node",
                                                  "description": "a child"}}}}
        schema = {"$defs": defs,
                  "properties": {"root": {"$ref": "#/$defs/Node", "description": "the root"}}}
        Provider._flatten_refs(schema, defs)

        assert self._offenders(schema) == []
