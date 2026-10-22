"""`store.read_snapshot`: inside it `atoms_for` answers from one folder listing per atom type, with
the same answers in the same order as the glob; outside it nothing changes. And the lookup caches
`index.build` leans on leave every answer as it was."""
import json

from universe_builder import index, store, topics


def _atom(atom_type, subject, model="gpt-4o-mini"):
    path = store.path_for(atom_type, None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": {"common": subject}}), encoding="utf-8")
    return path


def _corpus():
    _atom("city", "Memphis")
    _atom("city", "Memphis", model="gpt-5-nano")          # two models, one topic
    _atom("city", "Thebes (Luxor)")
    _atom("person", "Memphis")                            # the same subject in another type
    store.write_error(store.path_for("city", None, "Giza", "gpt-4o-mini"), "broken")   # a sentinel only
    folder = store.dir_for("city")
    (folder / f"city_{store.atom_id('city', 'Abydos')}_abydos").mkdir()               # a directory, not a file
    (folder / f"city_{store.atom_id('city', 'Saqqara')}").write_text("{}")            # no underscore after the id
    (folder / ".tmp_city_half.part").write_text("{")                                  # a write in progress


SUBJECTS = ["Memphis", "Thebes (Luxor)", "Thebes", "Giza", "Abydos", "Saqqara", "Nowhere", ""]


def test_snapshot_answers_as_the_glob_does(workspace):
    _corpus()
    plain = {(t, s): store.atoms_for(t, s) for t in ("city", "person", "event") for s in SUBJECTS}
    with store.read_snapshot():
        snap = {(t, s): store.atoms_for(t, s) for t in ("city", "person", "event") for s in SUBJECTS}
    assert snap == plain
    assert len(plain[("city", "Memphis")]) == 2 and plain[("city", "Giza")] == []


def test_a_write_inside_is_not_seen_until_the_block_ends(workspace):
    _corpus()
    with store.read_snapshot():
        assert store.atoms_for("city", "Memphis")          # the listing is read here
        _atom("city", "Abu Simbel")
        assert store.atoms_for("city", "Abu Simbel") == []  # documented: a pure-read scope
    assert store.atoms_for("city", "Abu Simbel")


def test_nested_blocks_share_the_outer_listing_and_restore_none(workspace):
    _corpus()
    with store.read_snapshot():
        store.atoms_for("city", "Memphis")
        outer = store._snapshot.get()
        with store.read_snapshot():
            assert store._snapshot.get() is outer
        assert store._snapshot.get() is outer
    assert store._snapshot.get() is None


def test_the_snapshot_is_dropped_when_the_block_raises(workspace):
    _corpus()
    try:
        with store.read_snapshot():
            store.atoms_for("city", "Memphis")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store._snapshot.get() is None
    _atom("city", "Abu Simbel")
    assert store.atoms_for("city", "Abu Simbel")


def test_cached_name_folding_is_the_same_function(workspace):
    for name in ["Châlons", "d’état", "Kατάνη", "Thebes (Luxor)", '"Abydos (Hellespont)"', "", "  Kō  "]:
        assert topics.name_key(name) == topics.name_key.__wrapped__(name)
        assert topics.alias_keys(name) == topics.alias_keys.__wrapped__(name)
        assert topics.alias_keys(name) is topics.alias_keys(name)    # cached: one object, and frozen


def test_index_resolves_alike_with_and_without_the_caches(workspace, monkeypatch):
    _corpus()
    resolve = index._resolver(topics.load_aliases())
    asked = [("city", s) for s in SUBJECTS if s] * 2       # each twice: the second from the cache
    cached = [resolve(t, s) for t, s in asked]
    fresh = [topics.subject_resolver(t)(s) for t, s in asked]
    assert cached == fresh
    assert cached[0] is not None                           # Memphis resolves


def test_the_fallback_needs_only_the_lines_sharing_a_key(workspace, monkeypatch, tmp_path):
    """`subject_resolver` hands its last-resort grouping only the alias lines that share a folded
    key with the name; the grouping must come out as it does with every line."""
    from universe_builder import settings

    aliases = tmp_path / "aliases.txt"
    aliases.write_text("\n".join([
        "Thebes (Luxor) = Waset",
        "Akhetaten = Amarna = Tell el-Amarna",
        '"Abydos (Hellespont)" = Abydos on the Hellespont',
        "[Greek History]",
        "Thebes (Boeotia) = Thebes",
        "[*]",
        "Memphis = Men-nefer",
        "Alexandria = Rhakotis",
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(settings, "ALIASES_LIST", aliases)
    _corpus()
    lines = topics.load_aliases()
    by_key = {}
    for i, g in enumerate(lines):
        for k in g.keys():
            by_key.setdefault(k, []).append(i)
    names = ["Thebes", "Waset", "Thebes (Luxor)", "Amarna", "Tell el-Amarna", "Abydos (Hellespont)",
             "Abydos", "Men-nefer", "Rhakotis", "Nowhere", "Luxor"]
    for name in names:
        for context in (None, "Egyptian History", "Greek History"):
            some = [lines[i] for i in sorted({i for k in topics.alias_keys(name) for i in by_key.get(k, ())})]
            full = topics.group_occurrences([(name, context)], aliases=lines)
            assert topics.group_occurrences([(name, context)], aliases=some) == full, (name, context)


def test_overlapping_blocks_in_two_threads_leave_nothing_stale(workspace):
    """Builds in two threads that enter A, B and exit A, B (not nested) must not leave a listing behind,
    and a thread outside any block must never answer from another thread's listing."""
    import threading

    _corpus()
    a_in, b_in, a_out, b_done = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    seen_by_outsider = []

    def build_a():
        with store.read_snapshot():
            store.atoms_for("city", "Memphis")
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def build_b():
        a_in.wait(5)
        with store.read_snapshot():
            store.atoms_for("city", "Memphis")
            b_in.set()
            a_out.wait(5)
            # meanwhile, outside any block, a writer's new atom must be visible to itself
            _atom("city", "Abu Simbel")
            outsider = threading.Thread(target=lambda: seen_by_outsider.append(store.atoms_for("city", "Abu Simbel")))
            outsider.start()
            outsider.join(5)
        b_done.set()

    ta, tb = threading.Thread(target=build_a), threading.Thread(target=build_b)
    ta.start()
    tb.start()
    ta.join(5)
    tb.join(5)
    assert b_done.is_set()
    assert seen_by_outsider and seen_by_outsider[0]         # the outsider saw the new atom
    _atom("city", "Kom Ombo")
    assert store.atoms_for("city", "Kom Ombo")             # nothing stale left behind afterwards
