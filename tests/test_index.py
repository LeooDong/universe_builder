"""The derived cross-reference index: never store what you can derive.

Membership (atom <-> histories) comes from the History atoms; name links come
from `LINK_FIELDS`; both resolve through `topics.resolve_subject` — the same
grouping/alias machinery as the work list, so a name means the same atom here
as everywhere else. Everything runs in `workspace`; nothing calls a model.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import index, journal, registry, settings, store, topics

MODEL = "gpt-5-nano"


def _history(name, **lists):
    payload = {"people": [], "discoveries": [], "events": [], "cities": [], "artifacts": []}
    payload.update(lists)
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _atom(atom_type, subject, payload=None):
    path = store.path_for(atom_type, None, subject, MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"name": {"common": subject}}), encoding="utf-8")
    return path


class TestResolveSubject:
    def test_a_direct_name_resolves_to_the_stored_id(self, workspace):
        _atom("city", "Memphis")

        assert topics.resolve_subject("city", "Memphis") == store.atom_id("city", "Memphis")

    def test_resolution_is_alias_aware(self, workspace, tmp_path, monkeypatch):
        """`Akhetaten` must point at the stored `Amarna (Akhetaten)` atom —
        the same answer the work list gives, because it is the same machinery."""
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Amarna (Akhetaten) = Amarna = Akhetaten\n", encoding="utf-8")
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", aliases)
        _atom("city", "Amarna (Akhetaten)")

        assert (topics.resolve_subject("city", "Akhetaten")
                == store.atom_id("city", "Amarna (Akhetaten)"))

    def test_nothing_on_disk_is_none(self, workspace):
        assert topics.resolve_subject("city", "Atlantis") is None

    def test_blank_is_none(self, workspace):
        assert topics.resolve_subject("city", "   ") is None

    def test_a_placeholder_does_not_resolve(self, workspace):
        """A recorded failure is not an atom; pointing at it would launder a
        gap into a link."""
        _atom("city", "Thonis", payload={"error": "No data found"})

        assert topics.resolve_subject("city", "Thonis") is None

    def test_a_parenthetical_listing_finds_the_atom_stored_under_its_head(self, workspace):
        """The first real index run's finding: one History says
        `Abydos (Abdju)`, another said `Abydos`, and the atom sits under the
        bare spelling. Resolution runs in the company of every listed
        spelling, like the work list, so the pair joins."""
        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Abydos (Abdju)"])
        _history("history_h2222222222222222_egyptian_history_m.json",
                 cities=["Abydos"])
        _atom("city", "Abydos")

        assert (topics.resolve_subject("city", "Abydos (Abdju)", context="egyptian history")
                == store.atom_id("city", "Abydos"))

    def test_a_bare_listing_finds_the_atom_stored_under_its_parenthetical(self, workspace):
        """The mirror case: bare `Narmer` listed, atom under `Narmer (Menes)`."""
        _history("history_h1111111111111111_egyptian_history_m.json",
                 people=["Narmer", "Narmer (Menes)"])
        _atom("person", "Narmer (Menes)")

        assert (topics.resolve_subject("person", "Narmer", context="egyptian history")
                == store.atom_id("person", "Narmer (Menes)"))

    def test_a_context_free_name_two_groups_claim_stays_unresolved(self, workspace,
                                                                   tmp_path, monkeypatch):
        """A reference field saying just `Thebes` names two cities once Greek
        history exists; guessing would link the wrong one silently."""
        aliases = tmp_path / "aliases.txt"
        aliases.write_text(
            "[Egyptian History]\nThebes (Luxor) = Thebes\n"
            "[Greek History]\nThebes (Boeotia) = Thebes\n", encoding="utf-8")
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", aliases)
        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes"])
        _history("history_h2222222222222222_greek_history_m.json",
                 cities=["Thebes"])
        _atom("city", "Thebes (Luxor)")
        _atom("city", "Thebes (Boeotia)")

        assert topics.resolve_subject("city", "Thebes") is None
        assert (topics.resolve_subject("city", "Thebes", context="greek history")
                == store.atom_id("city", "Thebes (Boeotia)"))


class TestResolutionIsDeterministic:
    """The review-gate case: resolution took the first hit while iterating a
    frozenset of keys, so the answer depended on the process's string-hash
    seed — probed over eight PYTHONHASHSEED values, one query resolved to an
    atom under two and to None under six. Keys are now ordered: exact
    spelling, then head, then parenthetical alias."""

    def _company(self, workspace):
        """Two groups whose keys overlap on `foo`: the ambiguous middle."""
        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Foo (Bar)", "Baz (Foo)"])
        _atom("city", "Foo (Bar)")
        _atom("city", "Baz (Foo)")

    def test_the_head_outranks_the_ambiguous_inner(self, workspace):
        """`Bar (Foo)`: head `bar` names one group; inner `foo` names two.
        The stated order resolves via the head; hash-order sometimes hit the
        ambiguous inner first and returned None."""
        self._company(workspace)

        assert (topics.resolve_subject("city", "Bar (Foo)")
                == store.atom_id("city", "Foo (Bar)"))

    def test_the_answer_survives_a_different_hash_seed(self, workspace):
        """The same corpus, resolved in fresh processes with adversarial hash
        seeds, must give the in-process answer."""
        import os
        import subprocess
        import sys

        self._company(workspace)
        expected = topics.resolve_subject("city", "Bar (Foo)")
        assert expected is not None, "precondition: the query resolves"

        script = ("from universe_builder import topics; "
                  "print(topics.resolve_subject('city', 'Bar (Foo)'))")
        for seed in ("0", "1", "42"):
            # Inherit the base UNIVERSE_BUILDER_DATA_DIR the workspace fixture set; data_dir()
            # appends the domain, so passing `workspace` (already base/history)
            # here would double-nest to base/history/history.
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run([sys.executable, "-c", script],
                                 capture_output=True, text=True, env=env)
            assert out.stdout.strip() == expected, (
                f"PYTHONHASHSEED={seed}: {out.stdout.strip()!r} != {expected!r} "
                f"(stderr: {out.stderr[-200:]})"
            )


class TestMemberships:
    def test_each_history_attributes_its_own_subjects(self, workspace):
        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis", "Thonis"])
        _history("history_h2222222222222222_greek_history_m.json",
                 cities=["Athens"])
        _atom("city", "Memphis")
        _atom("city", "Athens")

        report = index.build()

        egyptian = report["memberships"]["egyptian history"]["city"]
        greek = report["memberships"]["greek history"]["city"]
        assert egyptian["Memphis"] == f"city:{store.atom_id('city', 'Memphis')}"
        assert egyptian["Thonis"] is None, "listed but never generated"
        assert greek["Athens"] == f"city:{store.atom_id('city', 'Athens')}"

    def test_the_index_atom_is_never_a_member(self, workspace):
        """`build()` must never key a membership on the index atom itself — it is
        the *source* of the lists, not a member of them. `build()` iterates the
        pack's base atom specs, which do not include the index atom, so this holds
        by construction. Under history the index plural is also None, so this pins
        the invariant without isolating the mechanism;
        `test_domain_medications.py::TestIndexExcludesTheIndexAtom` is the case
        where iterating base specs (rather than `atom_mapping`) is what saves it."""
        from universe_builder.registry import active_pack

        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis"])
        _atom("city", "Memphis")

        report = index.build()

        index_name = active_pack().index.name
        for context, per_type in report["memberships"].items():
            assert index_name not in per_type, (
                f"index atom {index_name!r} appeared as a member under {context!r}"
            )

    def test_a_shared_subject_appears_under_both_histories(self, workspace):
        """The point of deriving membership: one atom, two incoming edges,
        nothing stored in the atom itself."""
        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Alexandria"])
        _history("history_h2222222222222222_greek_history_m.json",
                 cities=["Alexandria"])
        _atom("city", "Alexandria")

        report = index.build()

        target = f"city:{store.atom_id('city', 'Alexandria')}"
        assert report["memberships"]["egyptian history"]["city"]["Alexandria"] == target
        assert report["memberships"]["greek history"]["city"]["Alexandria"] == target


class TestMembershipResolver:
    """v0.4.1: `DomainPack.membership_resolver`, the pack's reading of the
    memberships; every reader sees the resolved view, and `build()` records
    what changed as `folds`."""

    @staticmethod
    def _fold_greek_into_egyptian(raw):
        out = {k: {t: dict(s) for t, s in v.items()} for k, v in raw.items() if k != "greek history"}
        moved = {}
        for atom_type, subjects in raw.get("greek history", {}).items():
            target = out.setdefault("egyptian history", {}).setdefault(atom_type, {})
            for subject, atom in subjects.items():
                target.setdefault(subject, atom)
                if atom:
                    moved.setdefault("egyptian history", []).append(atom)
        return out, ({"greek history": moved} if moved else {})

    def test_build_and_memberships_read_the_resolved_view(self, workspace, pack_overrides):
        pack_overrides(membership_resolver=self._fold_greek_into_egyptian)
        _history("history_h1111111111111111_egyptian_history_m.json", cities=["Memphis"])
        _history("history_h2222222222222222_greek_history_m.json", cities=["Athens"])
        _atom("city", "Memphis")
        _atom("city", "Athens")
        athens = f"city:{store.atom_id('city', 'Athens')}"

        report = index.build()

        assert set(report["memberships"]) == {"egyptian history"}
        assert report["memberships"]["egyptian history"]["city"]["Athens"] == athens
        assert report["folds"] == {"greek history": {"egyptian history": [athens]}}
        assert index.memberships() == report["memberships"]
        assert set(index.memberships(resolved=False)) == {"egyptian history", "greek history"}

    def test_without_a_resolver_nothing_changes(self, workspace):
        _history("history_h1111111111111111_egyptian_history_m.json", cities=["Memphis"])
        _atom("city", "Memphis")

        report = index.build()

        assert "folds" not in report
        assert index.memberships() == index.memberships(resolved=False) == report["memberships"]


class TestReferences:
    def test_a_name_field_resolves_to_its_atom(self, workspace):
        person = _atom("person", "Imhotep")
        city = _atom("city", "Memphis", payload={"name": {"common": "Memphis"},
                                                 "notableFigures": ["Imhotep"]})

        report = index.build()

        assert report["references"] == [{
            "from": f"city:{store.AtomName.decompose_filename(city.name).padded_id}",
            "field": "notableFigures",
            "name": "Imhotep",
            "at": "notableFigures[0]",
            "to": f"person:{store.AtomName.decompose_filename(person.name).padded_id}",
        }]

    def test_an_unresolvable_name_is_reported_not_dropped(self, workspace):
        _atom("city", "Memphis", payload={"name": {"common": "Memphis"},
                                          "notableFigures": ["Some Unknown Vizier"]})

        report = index.build()

        assert report["references"] == []
        assert [e["name"] for e in report["unresolved"]] == ["Some Unknown Vizier"]
        assert report["unresolved"][0]["target_type"] == "person"

    def test_scalar_fields_link_too(self, workspace):
        _atom("person", "Champollion")
        _atom("discovery", "Hieroglyphic writing",
              payload={"name": {"common": "Hieroglyphs"},
                       "discoverorOrInventor": "Champollion"})

        report = index.build()

        assert [e["to"] for e in report["references"]] == [
            f"person:{store.atom_id('person', 'Champollion')}"]

    def test_nulls_and_blanks_are_skipped(self, workspace):
        """`List[Optional[str]]` legitimately holds nulls — `Reason.null_array`
        produces `[None]` on purpose — so traversal must skip them.

        Note the nesting: `parents` lives under `family`. This test used to put
        it at the top level, where the link declaration also (wrongly) put it,
        so it passed against a payload the traversal never reached and proved
        nothing.
        """
        _atom("person", "Imhotep",
              payload={"name": {"common": "Imhotep"},
                       "family": {"parents": [None, "", "  "]}})

        report = index.build()

        assert report["references"] == []
        assert report["unresolved"] == []

    def test_a_nested_relation_resolves(self, workspace):
        """The bug this traversal exists for: `family.parents` is declared on
        `Person.family`, and a flat `payload.get("parents")` found nothing.
        Zero edges came out of thousands of stored mentions, and because a missing
        field reads exactly like an empty list, nothing reported it."""
        kanefer = _atom("person", "Kanefer")
        _atom("person", "Imhotep",
              payload={"name": {"common": "Imhotep"},
                       "family": {"parents": ["Kanefer"], "children": [],
                                  "spouses": [], "otherRelatives": []}})

        report = index.build()

        assert [(e["field"], e["at"], e["to"]) for e in report["references"]] == [
            ("family.parents", "family.parents[0]",
             f"person:{store.AtomName.decompose_filename(kanefer.name).padded_id}")]

    def test_a_doubly_nested_location_resolves(self, workspace):
        """`location.city` is the densest relation in the corpus and was never
        declared: a person's birthplace is a link to a city atom."""
        memphis = _atom("city", "Memphis")
        _atom("person", "Imhotep",
              payload={"name": {"common": "Imhotep"},
                       "birth": {"location": {"city": "Memphis"}}})

        report = index.build()

        assert [(e["field"], e["to"]) for e in report["references"]] == [
            ("birth.location.city",
             f"city:{store.AtomName.decompose_filename(memphis.name).padded_id}")]

    def test_a_missing_or_scalar_branch_yields_nothing_rather_than_raising(self, workspace):
        """Half-migrated and partially-expanded atoms are normal on disk. One
        odd shape must not cost the run every edge behind it."""
        _atom("person", "Imhotep",
              payload={"name": {"common": "Imhotep"},
                       "birth": {"location": None},
                       "death": "not an object"})

        report = index.build()

        assert report["references"] == []
        assert report["unresolved"] == []


class TestProvenanceIsDerivedNotStored:
    """Stale `certainty` closed by derivation: a period's write history comes
    from the journal, so `certainty` never has to carry edit-history it goes
    stale pretending to hold."""

    def test_write_history_is_ordered_oldest_first(self, workspace):
        journal.start_run(config={"test": True})
        path = _atom("city", "Memphis")
        item = f"city:{store.atom_id('city', 'Memphis')}"
        journal.current().record_attempt(
            work_item_id=item, raw_payload=None,
            raw_payload_unavailable_reason="created",
            operation="create", atom_type="city", subject="Memphis",
            requested_model="gpt-5-nano", validation_status="ok",
            target_pointer=str(path))
        journal.current().record_attempt(
            work_item_id=item, raw_payload=None,
            raw_payload_unavailable_reason="government set by majority",
            operation="adjudicate", atom_type="city", subject="Memphis",
            requested_model=None, validation_status="ok", target_pointer=str(path))

        prov = index.build()["provenance"][item]

        assert [w["operation"] for w in prov] == ["create", "adjudicate"]
        assert prov[0]["timestamp"] <= prov[1]["timestamp"]

    def test_a_failed_attempt_is_not_provenance(self, workspace):
        journal.start_run(config={"test": True})
        item = f"city:{store.atom_id('city', 'Memphis')}"
        journal.current().record_attempt(
            work_item_id=item, raw_payload='{"bad": true}',
            operation="fix", atom_type="city", subject="Memphis",
            validation_status="failed", validation_errors=["nope"])

        assert index.build()["provenance"].get(item) is None

    def test_the_provenance_section_is_written_and_invisible_to_iter_atoms(self, workspace):
        journal.start_run(config={"test": True})
        _atom("city", "Memphis")
        index.write(index.build())

        assert "provenance" in json.loads(
            (settings.index_dir() / "links.json").read_text(encoding="utf-8"))
        assert "links.json" not in [p.name for p in store.iter_atoms()]


class TestTheIndexIsMachineryNotAtoms:
    def test_two_builds_are_identical(self, workspace):
        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis", "Thebes"])
        _atom("city", "Memphis", payload={"name": {"common": "Memphis"},
                                          "notableFigures": ["Imhotep"]})
        _atom("person", "Imhotep")

        assert index.build() == index.build()

    def test_write_lands_in_the_underscore_dir(self, workspace):
        path = index.write(index.build())

        assert path == settings.index_dir() / "links.json"
        assert path.parent.name == "_index"
        assert json.loads(path.read_text(encoding="utf-8"))["references"] == []

    def test_the_index_is_invisible_to_iter_atoms(self, workspace):
        _atom("city", "Memphis")
        index.write(index.build())

        seen = [p.name for p in store.iter_atoms()]
        assert "links.json" not in seen

    def test_the_cli_command_reports_and_writes(self, workspace, capsys):
        from universe_builder import cli

        _history("history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis", "Thonis"])
        _atom("city", "Memphis")

        assert cli.main(["index"]) == 0
        out = capsys.readouterr().out

        assert "memberships  1/2" in out, out
        assert (settings.index_dir() / "links.json").exists()


class TestWirePathTraversal:
    """`values_at` is the whole of the traversal, so it is tested directly.

    A one-segment path is a plain field, which is what every declaration
    written before paths existed relies on.
    """

    def test_a_bare_field_is_a_one_segment_path(self):
        assert list(index.values_at({"originalCreator": "Imhotep"},
                                    "originalCreator")) == [
            ("Imhotep", "originalCreator")]

    def test_a_list_leaf_yields_each_entry_with_its_position(self):
        assert list(index.values_at({"notableFigures": ["A", "B"]},
                                    "notableFigures")) == [
            ("A", "notableFigures[0]"), ("B", "notableFigures[1]")]

    def test_a_list_of_objects_is_walked_with_indices(self):
        """`cityPeriodsOfHistory[].tradingPartners` has to say *which* period
        made the assertion, not merely that the city did."""
        payload = {"cityPeriodsOfHistory": [
            {"tradingPartners": ["Byblos"]},
            {"tradingPartners": ["Punt", "Nubia"]},
        ]}

        assert list(index.values_at(payload, "cityPeriodsOfHistory[].tradingPartners")) == [
            ("Byblos", "cityPeriodsOfHistory[0].tradingPartners[0]"),
            ("Punt", "cityPeriodsOfHistory[1].tradingPartners[0]"),
            ("Nubia", "cityPeriodsOfHistory[1].tradingPartners[1]"),
        ]

    def test_an_unexpanded_string_entry_is_skipped_not_fatal(self):
        """A city half-way through expansion holds bare period names beside
        expanded objects. Both shapes live on disk at once."""
        payload = {"cityPeriodsOfHistory": ["Old Kingdom",
                                            {"tradingPartners": ["Byblos"]}]}

        assert list(index.values_at(payload, "cityPeriodsOfHistory[].tradingPartners")) == [
            ("Byblos", "cityPeriodsOfHistory[1].tradingPartners[0]")]

    def test_missing_null_and_wrong_shaped_branches_yield_nothing(self):
        for payload in ({}, {"birth": None}, {"birth": "scalar"},
                        {"birth": {"location": None}}, {"birth": []}):
            assert list(index.values_at(payload, "birth.location.city")) == [], payload


class TestLinkPathsAreValidatedAtRegistration:
    """The `person` relations were wrong for as long as they existed and
    nothing said so, because `payload.get("parents")` on an atom that keeps
    them under `family` returns exactly what an empty list returns. The only
    moment that mistake is still cheap is registration."""

    def _pack(self, link_fields):
        from universe_builder.domains.history import build

        base = build()
        return registry.DomainPack(
            name="probe", index=base.index, atoms=base.atoms,
            link_fields=link_fields)

    def test_the_shipped_history_pack_validates(self):
        from universe_builder.domains.history import build

        registry.validate_link_paths(build())  # must not raise

    def test_a_nested_field_declared_flat_is_rejected(self):
        with pytest.raises(ValueError, match="no wire field 'parents'"):
            registry.validate_link_paths(self._pack({"person": {"parents": "person"}}))

    def test_a_typo_names_the_segment_and_the_shapes_searched(self):
        with pytest.raises(ValueError, match="no wire field 'famliy'"):
            registry.validate_link_paths(
                self._pack({"person": {"famliy.parents": "person"}}))

    def test_a_leaf_that_cannot_hold_a_name_is_rejected(self):
        with pytest.raises(ValueError, match="cannot hold a name"):
            registry.validate_link_paths(
                self._pack({"person": {"birth.location.latitude": "city"}}))

    def test_an_unknown_target_type_is_rejected(self):
        with pytest.raises(ValueError, match="unknown atom type 'planet'"):
            registry.validate_link_paths(
                self._pack({"person": {"family.parents": "planet"}}))

    def test_a_wire_alias_resolves(self):
        """Paths are wire names. `Artifact.finderOrDiscoverer` is stored under
        the misspelled alias, and that is what the index reads."""
        registry.validate_link_paths(
            self._pack({"artifact": {"finderOrDiscoveror": "person"}}))

    def test_an_expanded_only_path_validates_against_the_stage_model(self):
        """`tradingPartners` exists only on `Population`, which a city reaches
        through `CityExpanded`. Validating against `City` alone would reject a
        path that is correct on disk."""
        registry.validate_link_paths(
            self._pack({"city": {"cityPeriodsOfHistory[].tradingPartners": "city"}}))


class TestStoredAtomsDeclareTheirOwnNames:
    """The corpus knew things the resolver could not see.

    An atom records `name.common` and `name.otherNames`, and nothing read
    them: resolution ran on History-listed spellings, the alias file and
    filename topics only. So `Foundation of Memphis` could carry `Founding of
    Memphis` in its own `otherNames` — naming its rival outright — and the
    index still reported the name unresolved.
    """

    def test_a_name_only_in_other_names_resolves(self, workspace):
        _atom("person", "Ramesses II",
              payload={"name": {"common": "Ramesses II",
                                "otherNames": ["Ramses the Great"]}})

        assert (topics.resolve_subject("person", "Ramses the Great")
                == store.atom_id("person", "Ramesses II"))

    def test_the_id_comes_from_the_file_not_from_hashing_the_declared_name(self, workspace):
        """The Memphis case. The atom is stored under one spelling and declares
        another; hashing the declared name would mint the id the atom *would*
        have had under that spelling, which for a duplicate is a different
        atom's id."""
        _atom("event", "Foundation of Memphis",
              payload={"name": {"common": "Foundation of Memphis",
                                "otherNames": ["Founding of Memphis"]}})

        resolved = topics.resolve_subject("event", "Founding of Memphis")

        assert resolved == store.atom_id("event", "Foundation of Memphis")
        assert resolved != store.atom_id("event", "Founding of Memphis")

    def test_a_claim_never_overrides_an_earlier_answer(self, workspace):
        """Payload claims are a last resort, and the ordering is the safety
        property: the new index can turn a None into an id, never an id into a
        different id. Here an atom sits under `Thonis` while another atom
        declares `Thonis` among its other names — the atom stored under the
        queried spelling still wins."""
        _atom("city", "Thonis")
        _atom("city", "Heracleion",
              payload={"name": {"common": "Heracleion", "otherNames": ["Thonis"]}})

        assert (topics.resolve_subject("city", "Thonis")
                == store.atom_id("city", "Thonis"))

    def test_a_name_two_atoms_declare_stays_unresolved(self, workspace):
        """The standing rule, applied to the new index: a name two atoms claim
        is a duplicate to report, not a coin to flip."""
        _atom("person", "Ptolemy VIII Euergetes II",
              payload={"name": {"common": "Ptolemy VIII Euergetes II",
                                "otherNames": ["Ptolemy VIII"]}})
        _atom("person", "Ptolemy VIII Physcon",
              payload={"name": {"common": "Ptolemy VIII Physcon",
                                "otherNames": ["Ptolemy VIII"]}})

        assert topics.resolve_subject("person", "Ptolemy VIII") is None

    def test_a_placeholder_declares_nothing(self, workspace):
        _atom("city", "Thonis", payload={"error": "No data found"})

        assert topics.stored_name_claims("city") == {}

    def test_a_stored_parenthetical_does_not_claim_its_inner_name(self, workspace):
        """Keys are exact folded spellings on the claiming side. Splitting a
        stored parenthetical would have `Ancient Salamis (Cyprus)` claim the
        bare key `cyprus`, which is a country, not that city."""
        _atom("city", "Ancient Salamis (Cyprus)")

        claims = topics.stored_name_claims("city")

        assert "ancient salamis cyprus" in claims
        assert "cyprus" not in claims


class TestDuplicateAtomsAreReported:
    """Two atoms for one subject pass every invariant the project has.
    `test_corpus.py` counts files sharing an *id*, and these do not: the ids
    were hashed from the two spellings two models happened to use."""

    def test_two_atoms_declaring_one_name_are_reported(self, workspace):
        older = _atom("event", "Founding of Memphis")
        newer = _atom("event", "Foundation of Memphis",
                      payload={"name": {"common": "Foundation of Memphis",
                                        "otherNames": ["Founding of Memphis"]}})

        found = topics.duplicate_atoms("event")

        assert found == [{
            "name": "founding of memphis",
            "atom_ids": sorted([
                store.AtomName.decompose_filename(older.name).padded_id,
                store.AtomName.decompose_filename(newer.name).padded_id,
            ]),
        }]

    def test_distinct_subjects_are_not_duplicates(self, workspace):
        _atom("city", "Memphis")
        _atom("city", "Thebes")

        assert topics.duplicate_atoms("city") == []

    def test_the_report_reaches_the_index(self, workspace):
        _atom("event", "Founding of Memphis")
        _atom("event", "Foundation of Memphis",
              payload={"name": {"common": "Foundation of Memphis",
                                "otherNames": ["Founding of Memphis"]}})

        assert "event" in index.build()["duplicates"]


class TestTheListMarkerIsValidated:
    """`[]` marks a list of objects, and getting it wrong is silent.

    Both a missing marker and a spurious one validate cleanly and then yield
    *zero* edges at read time — the exact failure mode the validator exists to
    make impossible. Declaring `cityPeriodsOfHistory.tradingPartners` without
    `[]` passed, because the validator looked through the list to `Population`
    and found the field; traversal then hit a list where it wanted a dict and
    returned nothing.
    """

    def _pack(self, link_fields):
        from universe_builder.domains.history import build

        base = build()
        return registry.DomainPack(name="probe", index=base.index,
                                   atoms=base.atoms, link_fields=link_fields)

    def test_a_missing_marker_on_a_list_is_rejected(self):
        with pytest.raises(ValueError, match=r"needs '\[\]'"):
            registry.validate_link_paths(
                self._pack({"city": {"cityPeriodsOfHistory.tradingPartners": "city"}}))

    def test_a_marker_on_a_field_that_is_not_a_list_is_rejected(self):
        with pytest.raises(ValueError, match="is not a list"):
            registry.validate_link_paths(
                self._pack({"person": {"birth[].location.city": "city"}}))

    def test_the_correct_marker_still_validates(self):
        registry.validate_link_paths(
            self._pack({"city": {"cityPeriodsOfHistory[].tradingPartners": "city"}}))

    def test_a_leaf_list_needs_no_marker(self):
        """`notableFigures` is a list of strings, not of objects; traversal
        yields each entry without a marker and always has."""
        registry.validate_link_paths(
            self._pack({"city": {"notableFigures": "person"}}))
