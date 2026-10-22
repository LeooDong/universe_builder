"""Filename compose/decompose round-trip.

`fileName` owns the convention `{atom}_{id}_{slug}_{model}.json`, where the id
is either a legacy 10-digit counter or a `h`-prefixed content hash (step 3b).
Both forms have to compose and decompose: legacy files carry the first and
every new write carries the second, and no name may be readable as both.
"""

from __future__ import annotations

import pytest

from universe_builder.store import AtomName as fileName  # the alias module went in v0.3.0


class TestCompose:
    def test_basic_shape(self):
        name = fileName("city", 1, "Memphis", "claude-sonnet-5").getReadableFileName()
        assert name == "city_0000000001_memphis_claude-sonnet-5.json"

    def test_atom_type_is_lowercased(self):
        assert fileName("City", 1, "x", "m").getReadableFileName().startswith("city_")

    def test_id_is_padded_to_ten(self):
        assert fileName("city", 42, "x", "m").padded_id == "0000000042"

    def test_large_id_is_not_truncated(self):
        assert fileName("city", 12345678901, "x", "m").padded_id == "12345678901"

    def test_spaces_become_underscores(self):
        assert fileName("city", 1, "New York City", "m").readable_filename == "new_york_city"

    @pytest.mark.parametrize(
        "subject,expected",
        [
            ("St. Petersburg", "st_petersburg"),
            ("Constantinople (Byzantium)", "constantinople_byzantium"),
            ("Xi'an", "xian"),
            ("Coeur d'Alene", "coeur_dalene"),
            ("50% of Rome", "50_of_rome"),
        ],
    )
    def test_punctuation_is_stripped(self, subject, expected):
        assert fileName("city", 1, subject, "m").readable_filename == expected

    @pytest.mark.parametrize("subject", ["Köln", "Москва", "北京", "Reykjavík"])
    def test_non_ascii_is_stripped_entirely(self, subject):
        """Documenting current behaviour, not endorsing it: non-ASCII subjects
        can slugify to the empty string, which makes the filename ambiguous and
        `decompose_filename` unable to parse it. A hash id (step 3b) removes the
        collision risk; a transliteration step would be the real fix."""
        slug = fileName("city", 1, subject, "m").readable_filename
        assert all(c.isascii() for c in slug)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "atom,subject,model",
        [
            ("city", "Memphis", "claude-sonnet-5"),
            ("person", "Genghis Khan", "gpt-4o-mini"),
            ("discovery", "Invention of Toothpaste", "claude-opus-5"),
            ("artifact", "Rosetta Stone", "claude-haiku-4-5-20251001"),
            ("event", "Battle of Actium", "llama3"),
        ],
    )
    def test_decompose_recovers_the_parts(self, atom, subject, model):
        composed = fileName(atom, 7, subject, model).getReadableFileName()
        parsed = fileName.decompose_filename(composed)

        assert parsed.atom_type == atom
        assert parsed.model_version == model
        assert parsed.readable_filename == fileName(atom, 7, subject, model).readable_filename

    def test_recomposing_a_parsed_name_is_stable(self):
        original = "city_0000000012_memphis_claude-sonnet-5.json"
        parsed = fileName.decompose_filename(original)
        assert parsed.getReadableFileName() == original

    def test_get_topic_reverses_the_slug(self):
        parsed = fileName.decompose_filename("city_0000000001_new_york_city_gpt-4o-mini.json")
        assert parsed.getTopic() == "new york city"

    @pytest.mark.parametrize(
        "bad",
        [
            "city_001_memphis_model.json",          # id too short
            "city_0000000001_memphis_model.txt",    # wrong extension
            "not_a_filename",
            "",
            "city_0000000001_memphis.json",         # no model segment
        ],
    )
    def test_malformed_names_raise(self, bad):
        with pytest.raises(ValueError):
            fileName.decompose_filename(bad)

    def test_error_suffix_must_be_stripped_before_parsing(self):
        """The `_ERROR` sentinel is a filename suffix, so it has to come off
        before `decompose_filename` sees it — which is exactly what
        `augment_and_save_prompt` does. The run journal is meant to retire the
        convention."""
        name = "city_0000000001_memphis_claude-sonnet-5.json"
        with pytest.raises(ValueError):
            fileName.decompose_filename(name + "_ERROR")
        assert fileName.decompose_filename(name.replace("_ERROR", "")).atom_type == "city"


class TestGreedyRegexHazards:
    """`decompose_filename` is greedy on both sides of the id. These document
    where it mis-splits, which step 3b preserved deliberately rather than
    changing the parse under a corpus of stored filenames."""

    def test_underscored_atom_prefix_is_absorbed_correctly(self):
        parsed = fileName.decompose_filename("append_city_0000000001_memphis_gpt-4o-mini.json")
        assert parsed.atom_type == "append_city"

    def test_model_version_containing_an_underscore_is_mis_split(self):
        """A model id with an underscore steals a segment from the subject.
        No current model id has one, so this is a latent hazard, not a live bug."""
        parsed = fileName.decompose_filename("city_0000000001_memphis_my_model.json")
        assert parsed.model_version == "model"
        assert parsed.readable_filename == "memphis_my"


class TestContentAddressedForm:
    """The step 3b shape: `{atom}_h{16 hex}_{slug}_{model}.json`.

    `fileName` is the codec, so it composes whichever id it is given and only
    derives one when given `None`. `store.path_for` is where the policy lives.
    """

    def test_a_missing_id_is_derived_from_the_subject(self):
        from universe_builder import store

        name = fileName("city", None, "Memphis", "claude-sonnet-5").getReadableFileName()

        assert name == f"city_{store.atom_id('city', 'Memphis')}_memphis_claude-sonnet-5.json"

    def test_a_content_id_is_composed_verbatim(self):
        assert fileName("city", "h0123456789abcdef", "x", "m").padded_id == "h0123456789abcdef"

    def test_a_content_id_is_never_zero_padded(self):
        """The legacy branch pads; padding a hash would corrupt it."""
        assert "0000" not in fileName("city", "h0123456789abcdef", "x", "m").getReadableFileName()

    def test_the_two_forms_are_distinguishable_without_guessing(self):
        legacy = fileName.decompose_filename("city_0000000001_memphis_m.json")
        content = fileName.decompose_filename("city_h0123456789abcdef_memphis_m.json")

        assert not legacy.is_content_addressed
        assert content.is_content_addressed

    def test_a_content_addressed_name_decomposes(self):
        parsed = fileName.decompose_filename("city_h0123456789abcdef_new_york_city_gpt-4o-mini.json")

        assert parsed.atom_type == "city"
        assert parsed.padded_id == "h0123456789abcdef"
        assert parsed.readable_filename == "new_york_city"
        assert parsed.model_version == "gpt-4o-mini"

    def test_recomposing_a_parsed_content_addressed_name_is_stable(self):
        original = "append_city_h0123456789abcdef_memphis_claude-sonnet-5.json"

        assert fileName.decompose_filename(original).getReadableFileName() == original

    @pytest.mark.parametrize(
        "bad",
        [
            "city_h0123456789abcde_memphis_m.json",     # 15 hex, one short
            "city_h0123456789abcdef0_memphis_m.json",   # 17 hex, one long
            "city_hZZZZZZZZZZZZZZZZ_memphis_m.json",    # not hex
            "city_H0123456789ABCDEF_memphis_m.json",    # ids are lower case
        ],
    )
    def test_near_miss_ids_are_not_accepted_as_ids(self, bad):
        """A sloppy pattern here would silently reclassify a subject as an id."""
        with pytest.raises(ValueError):
            fileName.decompose_filename(bad)

    def test_a_subject_that_slugs_to_nothing_still_parses(self):
        """`Москва` strips to the empty string under the `[a-zA-Z0-9_]` rule. The
        empty slug is left as-is on purpose: it is constant, so the content id
        alone carries identity and `decompose_filename` still parses the name. A
        non-constant fallback was tried and reverted — it gave two subjects that
        share a content id two different filenames (see slugify)."""
        from universe_builder import store

        name = fileName("city", None, "Москва", "m").getReadableFileName()
        parsed = fileName.decompose_filename(name)

        assert parsed.readable_filename == ""
        assert parsed.atom_type == "city"
        assert parsed.model_version == "m"
        assert parsed.padded_id == store.atom_id("city", "Москва")

    def test_two_subjects_that_slug_alike_get_different_names(self):
        """Both slug to `stpetersburg`; before content ids only the counter kept
        them apart, and the counter was not stable."""
        one = fileName("city", None, "St. Petersburg", "m").getReadableFileName()
        two = fileName("city", None, "St Petersburg", "m").getReadableFileName()

        assert one != two
