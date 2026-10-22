"""What the three providers now share: one
`BatchNotSupported`, one schema-in-system text, one stored-result validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from universe_builder.providers import batch as batch_mod
from universe_builder.providers.base import Provider
from universe_builder.providers.errors import BatchNotSupported, ProviderError
from universe_builder.providers.types import LLMRequest, Usage


class _Thing(BaseModel):
    name: str


def test_batch_not_supported_is_still_a_not_implemented_error():
    assert issubclass(BatchNotSupported, NotImplementedError)
    assert issubclass(BatchNotSupported, ProviderError)


def test_submit_raises_it_for_a_provider_without_batch(workspace, fake_provider):
    fake_provider.without_batch()
    with pytest.raises(BatchNotSupported, match="has no batch job API"):
        batch_mod.submit("discovery", ["Papyrus"], "gpt-5-nano")


def test_the_system_text_is_byte_for_byte_what_each_provider_built():
    request = LLMRequest(model="m", schema_model=_Thing, system="Be terse.", user="u")
    expected = ("Be terse.\n\nReturn JSON matching this schema exactly:\n"
                + Provider._dumps(Provider.json_schema(_Thing)))

    class _Bare(Provider):
        name = "bare"

        def complete(self, request):
            raise AssertionError

    assert _Bare().system_with_schema(request) == expected


class TestValidateStored:
    def test_a_good_line(self):
        r = Provider._validate_stored('{"name": "x"}', _Thing, "m", Usage(), "c1")
        assert r.parsed == _Thing(name="x") and r.error is None and r.custom_id == "c1"

    def test_a_bad_line_without_a_stop_signal_reads_as_the_validation_error(self):
        r = Provider._validate_stored('{"name": ', _Thing, "m", Usage(output_tokens=5), "c1")
        assert r.parsed is None and "Invalid JSON" in r.error

    def test_a_truncated_line_is_named_when_the_signal_is_given(self):
        r = Provider._validate_stored('{"name": ', _Thing, "m", Usage(output_tokens=5), "c1", "length")
        assert r.error.startswith("Response truncated: m stopped at its output limit")
