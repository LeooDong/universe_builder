"""Bounded concurrency on the create pass.

No network, no API key: the fake `LLMCall` from conftest is used throughout.
The `sequential_id` thread-safety tests went with the counter in v0.3.0:
ids are content-addressed, so there is no shared counter.
"""

from __future__ import annotations

import threading


class TestCreatePassConcurrency:
    def _config(self, ep):
        return ep.OperationConfig(
            atom_name="discovery",
            model_version="claude-sonnet-5",
            base_system_text="test",
        )

    def test_the_default_is_bounded_and_warms_the_prefix(
            self, ep, workspace, monkeypatch, fake_call_factory):
        """The default was raised from 1 to 4 to make the serial expand pass
        usable (it makes many calls per civilisation). It stays *bounded*
        — a modest number under provider rate limits, not unlimited — and the
        warm-first-then-pool discipline (pinned by the sibling tests) is what
        makes >1 safe for prompt caching."""
        from universe_builder.pipeline import load

        configured = int(load().defaults.get("concurrency", 1))
        assert 1 < configured <= 16, f"expected bounded concurrency, got {configured}"

    def test_every_topic_produces_a_result(self, ep, workspace, monkeypatch, fake_call_factory):
        from tests.test_operations import use_fake, valid_discovery

        topics = ["Papyrus", "Obsidian", "Bronze", "Iron"]
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()] * len(topics))

        results = ep.call_and_save_prompt(self._config(ep), topics, concurrency=3)

        assert len(results) == len(topics)
        assert all(r.ok for r in results)

    def test_the_first_topic_is_run_alone_to_warm_the_cache(self, ep, workspace, monkeypatch,
                                                            fake_call_factory):
        """Firing a cold batch in parallel means every call writes the cache and
        none reads it — slower and dearer than running serially."""
        from tests.test_operations import use_fake, valid_discovery

        topics = ["A", "B", "C", "D"]
        Fake = use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()] * len(topics))

        in_flight = []
        concurrent_with_first = []
        lock = threading.Lock()
        original = Fake.call_llm_api

        def traced(self):
            with lock:
                in_flight.append(self.prompt_subject)
                if len(in_flight) == 1:
                    concurrent_with_first.append(len(in_flight))
            return original(self)

        monkeypatch.setattr(Fake, "call_llm_api", traced)
        ep.call_and_save_prompt(self._config(ep), topics, concurrency=3)

        # The first call completed before any other started.
        assert in_flight[0].startswith("A")

    def test_concurrency_of_one_still_works(self, ep, workspace, monkeypatch, fake_call_factory):
        from tests.test_operations import use_fake, valid_discovery

        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()] * 2)
        results = ep.call_and_save_prompt(self._config(ep), ["A", "B"], concurrency=1)

        assert len(results) == 2

    def test_a_single_topic_never_starts_a_pool(self, ep, workspace, monkeypatch,
                                                fake_call_factory):
        from tests.test_operations import use_fake, valid_discovery

        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])
        results = ep.call_and_save_prompt(self._config(ep), ["A"], concurrency=8)

        assert len(results) == 1
