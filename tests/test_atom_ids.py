"""Content-addressed atom ids.

`store.atom_id` replaces a class counter that restarted at 1 every process.
These are the behaviours that made the replacement worth the churn — stability
across processes, uniqueness for subjects the slug cannot represent. The
migration plan that renamed the corpus onto them (`store.rename_plan`) and its
tests were deleted in v0.3.0. No network, no API key.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from universe_builder import store

MODEL = "claude-sonnet-5"


def id_in_a_separate_process(atom_type: str, subject: str) -> str:
    """Compute an id in a fresh interpreter.

    Calling `atom_id` twice in one process proves nothing about the failure being
    fixed: the counter was stable within a process too. The claim is that two
    *runs* agree, so this spawns one.
    """
    code = (
        "import sys; from universe_builder import store; "
        "sys.stdout.write(store.atom_id(sys.argv[1], sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, atom_type, subject],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


class TestStability:
    def test_the_same_topic_yields_the_same_id_in_a_new_process(self):
        assert id_in_a_separate_process("city", "Memphis") == store.atom_id("city", "Memphis")

    def test_two_separate_processes_agree_with_each_other(self):
        """Nothing in the id may depend on process state, PYTHONHASHSEED or clock."""
        first = id_in_a_separate_process("person", "Genghis Khan")
        second = id_in_a_separate_process("person", "Genghis Khan")

        assert first == second

    def test_the_id_does_not_depend_on_hash_randomisation(self):
        """`hash()` would have been the obvious wrong choice: it is salted per
        process for str, so ids would have differed on every run."""
        code = "from universe_builder import store; print(store.atom_id('city', 'Memphis'))"
        seeded = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
        )

        assert seeded.stdout.strip() == store.atom_id("city", "Memphis")

    def test_the_id_does_not_depend_on_the_model(self):
        """Regenerating a topic with a better model must replace the file, not
        add a second one under a new id."""
        one = store.path_for("city", 1, "Memphis", "claude-sonnet-5").name.split("_")[1]
        two = store.path_for("city", 1, "Memphis", "gpt-4o-mini").name.split("_")[1]

        assert one == two


class TestDistinctness:
    def test_different_topics_get_different_ids(self):
        assert store.atom_id("city", "Memphis") != store.atom_id("city", "Thebes")

    def test_different_atom_types_get_different_ids(self):
        assert store.atom_id("city", "Alexandria") != store.atom_id("person", "Alexandria")

    def test_an_append_variant_shares_its_base_types_id(self):
        """Not an accident: it is what merge_helper pairs on. The id follows
        `store.folder_name`, so an append shares its base type's id exactly as it
        shares its base type's directory."""
        assert store.atom_id("append_city", "Memphis") == store.atom_id("city", "Memphis")
        assert store.atom_id("append_person", "Genghis Khan") == store.atom_id("person", "Genghis Khan")

    def test_the_type_and_the_topic_cannot_bleed_into_each_other(self):
        """Without a separator that cannot occur in either half, these collide."""
        assert store.atom_id("city", "a_b") != store.atom_id("city_a", "b")

    def test_a_subject_that_slugs_to_nothing_still_gets_a_unique_id(self):
        """`Москва` and `北京` both strip to the empty string under the
        `[a-zA-Z0-9_]` slug rule, so before content ids only the process-local
        counter kept their filenames apart."""
        moscow = store.atom_id("city", "Москва")
        beijing = store.atom_id("city", "北京")

        assert moscow != beijing
        assert moscow == id_in_a_separate_process("city", "Москва")

    def test_ten_thousand_topics_produce_ten_thousand_ids(self):
        """A cheap floor under the collision maths; 64 bits should not collide
        at this scale and a truncation bug would show up here immediately."""
        ids = {store.atom_id("city", f"topic {n}") for n in range(10_000)}

        assert len(ids) == 10_000


class TestNormalisation:
    @pytest.mark.parametrize("variant", ["Memphis", "memphis", "MEMPHIS", "  Memphis  ", "Memphis\n"])
    def test_case_and_surrounding_whitespace_do_not_change_the_id(self, variant):
        """A trailing newline off a topic list is the likeliest accidental
        variant there is, and case never reached the filename anyway."""
        assert store.atom_id("city", variant) == store.atom_id("city", "Memphis")

    def test_internal_whitespace_runs_collapse(self):
        assert store.atom_id("city", "New  York\tCity") == store.atom_id("city", "New York City")

    def test_equivalent_unicode_spellings_agree(self):
        """NFC and NFD `Reykjavík` render identically; two files would not."""
        composed = "Reykjavík"
        decomposed = "Reykjavík"

        assert composed != decomposed
        assert store.atom_id("city", composed) == store.atom_id("city", decomposed)

    def test_punctuation_is_significant(self):
        """The slug drops it, so these share a readable part — but they are
        different prompts, and folding them would let one overwrite the other."""
        assert store.atom_id("city", "St. Petersburg") != store.atom_id("city", "St Petersburg")

    def test_the_atom_type_is_case_insensitive(self):
        assert store.atom_id("City", "Memphis") == store.atom_id("city", "Memphis")


class TestIdShape:
    def test_a_content_id_is_recognised_and_a_counter_id_is_not(self):
        assert store.is_hash_id(store.atom_id("city", "Memphis"))
        assert not store.is_hash_id("0000000001")

    def test_the_id_is_filename_safe(self):
        token = store.atom_id("city", "Москва")

        assert token == store.slugify(token)
