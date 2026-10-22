"""Golden identity values: literal ids, normalisations and slugs, frozen.

Every other id test in the suite is relational — "the same
topic gives the same id", "two types give two ids" — and would pass unchanged if
the hash, the separator, the digest size or the normalisation moved, because
both sides of each assertion would move together. An id that moves orphans every
stored atom: the next create regenerates the whole corpus beside itself.

So these are *literals*, computed once with v0.1.19's code and
written in by hand. They are never recomputed by the code under test. If one of
these fails, the change re-points real files; it is not a test to update.

`tests/fixtures/corpus_ids_sample.json` adds 40 (type, topic) -> id rows, each
the id and slug of a real atom on the day it was frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from universe_builder import journal, settings, store, topics

FIXTURES = Path(__file__).parent / "fixtures"

#: (subject, normalise_topic, slugify, atom_id city, atom_id person, atom_id history)
GOLDEN = [
    ("Memphis", "memphis", "memphis",
     "hc5afac8b4020cd95", "ha8f6a79f055bbee4", "h79bd958ef156269e"),
    # Whitespace collapses for the id; the slug turns the leading space into `_`
    # and drops the newline — two spellings, one id, two filenames.
    (" Memphis\n", "memphis", "_memphis",
     "hc5afac8b4020cd95", "ha8f6a79f055bbee4", "h79bd958ef156269e"),
    ("memphis", "memphis", "memphis",
     "hc5afac8b4020cd95", "ha8f6a79f055bbee4", "h79bd958ef156269e"),
    ("Thebes (Luxor)", "thebes (luxor)", "thebes_luxor",
     "h8fdc46cd4a0f6fb6", "h0fbc2eb3387de089", "hb7e99fc7cabba577"),
    # casefold folds ß to ss for the id; the slug drops it.
    ("Straße", "strasse", "strae",
     "hadc0b4573227e884", "h0fb0bf8cb02a7222", "h4e18f0559f3062e9"),
    ("STRASSE", "strasse", "strasse",
     "hadc0b4573227e884", "h0fb0bf8cb02a7222", "h4e18f0559f3062e9"),
    ("Kōfuku-ji", "kōfuku-ji", "kfukuji",
     "h45a1ad8c3f295f6f", "h34604e935a3ce67b", "ha02974742ee1d1da"),
    # CJK and Cyrillic slug to nothing; the id still separates them.
    ("東大寺", "東大寺", "",
     "h3645c71b6ddd917d", "h7689f65aea5c5284", "h68c780d71dcbb9f3"),
    ("Москва", "москва", "",
     "hbfd4b270b2849dad", "ha83b22279f26505a", "ha1a1982c8115f03c"),
    ("Châlons-en-Champagne", "châlons-en-champagne", "chlonsenchampagne",
     "hd088afe7246beec6", "h463e30fc495ad20c", "h005d36c6d86cd274"),
    # Punctuation is significant to the id, not to the slug.
    ("St. Petersburg", "st. petersburg", "st_petersburg",
     "h7b1b2631ed6ca988", "hb1dd346bde97a7b5", "hf7125f32612db2da"),
    ("St Petersburg", "st petersburg", "st_petersburg",
     "h93ac97b07b34644a", "ha7a660bf7dba45dc", "hf968165ec360f41c"),
    # NFKC: fullwidth letters become ASCII for the id; the slug (no NFKC) drops them.
    ("Ｆｕｌｌｗｉｄｔｈ", "fullwidth", "",
     "hf3eb1bd6babe6de4", "h3ba34039e7edd6d1", "h7550a818fa4f036f"),
    # Combining and composed accents: one id, two slugs.
    ("Café", "café", "cafe",
     "h06b7383e2296f183", "hd304d2c1555682c6", "h42a8e4bac650b6bd"),
    ("Café", "café", "caf",
     "h06b7383e2296f183", "hd304d2c1555682c6", "h42a8e4bac650b6bd"),
    ("Boutros Boutros-Ghali", "boutros boutros-ghali", "boutros_boutrosghali",
     "hb8a6297c32431168", "h23949e5192405ba8", "h64b157c03b12e688"),
    ("Anglo-Saxon History", "anglo-saxon history", "anglosaxon_history",
     "heee0b64da9474e63", "hb3c92ff425a20ec6", "h0c2fa6a2a99640bb"),
    ("Narmer (Menes)", "narmer (menes)", "narmer_menes",
     "h039dd363072bebff", "ha890f9cfdeb53982", "hc0eec252f0e2b6c1"),
    ("Ḫattuša", "ḫattuša", "attua",
     "hb2a566f4e71b9179", "hc40bac5211c4f746", "hafb2267abad017f7"),
    ("Ægean Sea", "ægean sea", "gean_sea",
     "ha0b8bd83559f7669", "h6d71dd884f940fe8", "h9cc2760316fe8538"),
    # casefold of a dotted capital I keeps a combining dot above (U+0307).
    ("İstanbul", "i̇stanbul", "istanbul",
     "hcee73ddba575e2f2", "h04b8a5da0dcd0270", "h948d75c4b77d3446"),
    ("ΣΟΦΙΑ", "σοφια", "",
     "h42fad97d297efd07", "h92d4cedaa71a1d1a", "h4b33bf8f70c9f90b"),
    # NFKC expands the fi ligature for the id; the slug drops it.
    ("ﬁre temple", "fire temple", "re_temple",
     "h7288f00766492c86", "h3d1000d803b4193b", "h0f3b7ae7fe7b7457"),
    ("a_b", "a_b", "a_b",
     "hf7e3f5dce13f088f", "h5492c6605dc7adfd", "h3a807dc7e8f735d9"),
    ("", "", "",
     "h67b93d52c84c2305", "h6bd0092debf6cc3f", "hef7dbe24895e20c7"),
    ("Hieroglyphic writing", "hieroglyphic writing", "hieroglyphic_writing",
     "hbacd59b9f8bb9a4b", "h84ca7463a0cec68c", "h03bf6fb844e40ddb"),
    ("Abydos (Hellespont)", "abydos (hellespont)", "abydos_hellespont",
     "hdd41bf907d4e868b", "h143943c0b93280ad", "hb11ca475d57f3b9c"),
]

_IDS = [repr(row[0]) for row in GOLDEN]


@pytest.mark.parametrize("subject, normalised, slug, city, person, history", GOLDEN, ids=_IDS)
class TestGoldenIdentity:
    def test_normalise_topic(self, subject, normalised, slug, city, person, history):
        assert store.normalise_topic(subject) == normalised

    def test_slugify(self, subject, normalised, slug, city, person, history):
        assert store.slugify(subject) == slug

    def test_atom_id(self, subject, normalised, slug, city, person, history):
        assert store.atom_id("city", subject) == city
        assert store.atom_id("person", subject) == person
        assert store.atom_id("history", subject) == history

    def test_an_append_variant_shares_the_base_id(self, subject, normalised, slug, city, person, history):
        assert store.atom_id("append_city", subject) == city
        # The atom type is lowered before hashing.
        assert store.atom_id("City", subject) == city

    def test_work_item_id(self, subject, normalised, slug, city, person, history):
        assert journal.work_item_id("city", subject) == f"city:{city}"
        # The work item keeps the literal type, though the id is the base type's.
        assert journal.work_item_id("append_city", subject) == f"append_city:{city}"


class TestGoldenConstants:
    def test_the_hash_id_shape(self):
        assert store.HASH_ID_PREFIX == "h"
        assert store.HASH_ID_BYTES == 8
        assert store.HASH_ID_LENGTH == 16
        assert store.ID_PAD_WIDTH == 10

    def test_the_nul_separator_keeps_type_and_topic_apart(self):
        assert store.atom_id("city_a", "b") == "h966aeffc81da6096"
        assert store.atom_id("city", "a_b") == "hf7e3f5dce13f088f"

    def test_folder_name(self):
        assert store.folder_name("city") == "city"
        assert store.folder_name("append_city") == "city"
        assert store.folder_name("expand_city") == "expand_city"

    def test_filenames(self, workspace):
        path = store.path_for("city", None, "Thebes (Luxor)", "gpt-5-nano")
        assert path.relative_to(workspace).as_posix() == "city/city_h8fdc46cd4a0f6fb6_thebes_luxor_gpt-5-nano.json"
        append = store.path_for("append_city", None, "Thebes (Luxor)", "gpt-5-nano")
        assert append.relative_to(workspace).as_posix() == \
            "city/append_city_h8fdc46cd4a0f6fb6_thebes_luxor_gpt-5-nano.json"
        cjk = store.path_for("city", None, "東大寺", "m")
        assert cjk.name == "city_h3645c71b6ddd917d__m.json"
        parsed = store.AtomName.decompose_filename(cjk.name)
        assert (parsed.atom_type, parsed.padded_id, parsed.readable_filename, parsed.model_version) == \
            ("city", "h3645c71b6ddd917d", "", "m")

    def test_a_legacy_name_parses_to_its_fields(self):
        parsed = store.AtomName.decompose_filename(
            "city_0000000001_alexandria_context_egypt_cities_gpt-3.5-turbo.json")
        assert (parsed.atom_type, parsed.padded_id, parsed.readable_filename, parsed.model_version) == \
            ("city", "0000000001", "alexandria_context_egypt_cities", "gpt-3.5-turbo")
        assert parsed.getTopic() == "alexandria context egypt cities"


class TestGoldenContextOf:
    """`topics._context_of`: the civilisation a History filename names."""

    @pytest.fixture
    def seed(self, tmp_path, monkeypatch):
        path = tmp_path / "HISTORY_list.txt"
        path.write_text("Egyptian History\nAnglo-Saxon History\nMesoamérican History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", path)

    def test_a_listed_content_id_gives_the_seed_line_back(self, seed):
        assert topics._context_of("history_h0c2fa6a2a99640bb_anglosaxon_history_m.json") == "anglo-saxon history"
        assert topics._context_of("history_h119392faa5c450a5_egyptian_history_m.json") == "egyptian history"

    def test_an_unlisted_content_id_falls_back_to_the_slug(self, seed):
        assert topics._context_of("history_h0000000000000000_anglosaxon_history_m.json") == "anglosaxon history"

    def test_a_legacy_name_drops_its_context_tail(self, seed):
        assert topics._context_of("history_0000000001_egyptian_history_context_history_m.json") == \
            "egyptian history"


def _sample():
    return json.loads((FIXTURES / "corpus_ids_sample.json").read_text(encoding="utf-8"))["rows"]


@pytest.mark.parametrize("row", _sample(), ids=lambda r: f"{r['atom_type']}:{r['id']}")
def test_a_real_corpus_topic_still_hashes_to_its_file(row):
    """Each row was a real atom file when frozen: its id and slug."""
    assert store.atom_id(row["atom_type"], row["topic"]) == row["id"]
    assert store.slugify(row["topic"]) == row["slug"]


def test_the_corpus_sample_is_the_size_it_was_frozen_at():
    rows = _sample()
    assert len(rows) == 40
    assert len({(r["atom_type"], r["id"]) for r in rows}) == 40
