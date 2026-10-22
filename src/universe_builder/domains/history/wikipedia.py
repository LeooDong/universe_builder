"""Wikipedia as a free source of URLs and of identity evidence — history's.

Two of the history pack's resolvers live here, handed to core through
`DomainPack.url_resolvers` (`universe_builder sources --via wikipedia`) and
`DomainPack.identity_resolvers` (`universe_builder lint --review F --via wikipedia`).
Neither is engine machinery: an ancient-history corpus is one whose subjects
mostly have an English Wikipedia article under their common name, which is
what makes a title lookup worth doing; a formulary would resolve against a
drug database instead. Core carries the fetch/verify/append logic and knows
nothing about which encyclopaedia a domain trusts.

Moved out of `universe_builder.sources` when the engine was extracted. The
functions are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
#: Wikimedia's User-Agent policy wants a *meaningful* contact, and enforces it:
#: a bare `https://github.com/` with no repository reads as a placeholder and
#: earns `HTTP 429` on the first request, whatever the rate. That cost this
#: session three aborted runs and a batching rewrite before the cause turned
#: out to be the identification string rather than the request volume —
#: 50 titles fail with the placeholder and succeed with a real contact,
#: back to back. Overridable so a fork identifies as itself.
_UA = os.environ.get(
    "UNIVERSE_BUILDER_USER_AGENT",
    "universe_builder/1.0 (https://github.com/LeooDong/universe_builder)")


def wikipedia_lookup(lang: str = "en", per_atom: int = 1,
                     batch: int = 50, delay: float = 1.0):
    """An `ask` backed by Wikipedia's own API. Free, no key, no model.

    **The paid search was doing a title lookup.** Every URL it added in a
    trial was a Wikipedia article, at $10 per 1,000 search calls plus the
    input tokens of each. This does the same for nothing.

    **Titles are resolved in batches of 50, not one request per atom.** Asking
    once per atom in a row earned `HTTP 429 Too Many Requests` within a few
    dozen atoms, which is Wikipedia correctly objecting to being hammered.
    `action=query` accepts up to 50 titles at once and follows redirects, so the
    corpus takes one request per 50 atoms — politer *and* far faster. Call
    `ask.prefetch(subjects)` first to fill the cache.

    Anything the batch cannot match falls back to per-atom search, rate-limited
    and with one retry on 429. The gap that remains is judgement, not recall:
    for `Conquest of Egypt by Alexander the Great` the model returned
    `Alexander_the_Great#Egypt`, knowing which section was wanted, where this
    returns the article whose title matches.
    """
    import json as _json
    import time
    import urllib.parse
    import urllib.request

    cache: dict = {}
    #: Titles the API says are disambiguation pages. These must NOT fall
    #: through to search: the name is *known* ambiguous, and search answers
    #: with the most popular article, which for this corpus meant
    #: `Memphis` -> `Memphis, Tennessee`. Declining is the right answer; a
    #: wrong citation is worse than none.
    ambiguous: set = set()
    #: Titles abandoned because the API was rate-limiting. They must not be
    #: retried per atom — that is what caused the throttling.
    throttled: set = set()

    def _call(params: dict, attempts: int = 5):
        """One API call, backing off on 429.

        Wikipedia answers 429 when pushed, and having been pushed it stays
        cross for a while — a burst of per-atom requests earned a cooldown that
        outlasted the run that caused it. So this waits and retries rather than
        giving up, and the caller paces itself between calls as well.
        """
        query = urllib.parse.urlencode({**params, "format": "json"})
        request = urllib.request.Request(f"{WIKIPEDIA_API}?{query}",
                                         headers={"User-Agent": _UA})
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return _json.load(response)
            except Exception as e:  # noqa: BLE001
                last = e
                if "429" not in str(e) or attempt == attempts - 1:
                    raise
                wait = 5.0 * (3 ** attempt)   # 5s, 15s, 45s, 135s
                log.info("wikipedia asked us to slow down; waiting %.0fs", wait)
                time.sleep(wait)
        raise last

    def _url(title: str) -> str:
        return (f"https://{lang}.wikipedia.org/wiki/"
                + urllib.parse.quote(title.replace(" ", "_")))

    def prefetch(subjects) -> int:
        """Resolve many titles at once. Returns how many matched."""
        wanted = [s for s in dict.fromkeys(subjects) if s and s not in cache]
        found = 0
        for start in range(0, len(wanted), batch):
            chunk = wanted[start:start + batch]
            try:
                data = _call({"action": "query", "prop": "info|pageprops",
                              "inprop": "url", "titles": "|".join(chunk),
                              "redirects": 1})
            except Exception as e:  # noqa: BLE001
                if "429" in str(e):
                    # **Do not fall back to search when rate-limited.** The
                    # fallback is one request per atom, so a throttled batch of
                    # 50 becomes 50 more requests precisely when Wikipedia is
                    # asking for fewer — which is how the first run turned a
                    # brief 429 into a sustained one. Give up on the whole
                    # prefetch instead; the atoms simply go unsourced this run.
                    log.error("wikipedia is rate-limiting us; stopping the "
                              "prefetch rather than making it worse. %d title(s) "
                              "resolved before that.", found)
                    throttled.update(wanted[start:])
                    return found
                log.warning("wikipedia batch failed (%s); those fall back to search", e)
                continue
            result = data.get("query", {}) or {}
            # The API rewrites what you asked for: `normalized` for casing and
            # underscores, `redirects` for renames. Follow both so a requested
            # title maps to the page actually returned.
            hop = {}
            for kind in ("normalized", "redirects"):
                for entry in result.get(kind, []) or []:
                    hop[entry["from"]] = entry["to"]
            pages = {p["title"]: p for p in (result.get("pages") or {}).values()}
            for title in chunk:
                landed, seen = title, set()
                while landed in hop and landed not in seen:
                    seen.add(landed)
                    landed = hop[landed]
                page = pages.get(landed)
                if not page or "missing" in page:
                    continue
                if "disambiguation" in (page.get("pageprops") or {}):
                    # **A disambiguation page is not a source.** Most carry no
                    # marker in the URL — `Memphis`, `Thebes`, `Apollonia` and
                    # `Pausanias` are all disambiguation pages at their bare
                    # title — so the `(disambiguation)` pattern in
                    # `cites_something` catches almost none of them, and this
                    # corpus is made largely of exactly such names. Only the
                    # API knows, via `pageprops`.
                    log.info("%s resolves to a disambiguation page; skipped", title)
                    ambiguous.add(title)
                    continue
                cache[title] = [page.get("fullurl") or _url(page["title"])]
                found += 1
            time.sleep(delay)
        return found

    def _same_subject(subject: str, title: str, hints=()) -> bool:
        """Whether a search hit is the *same* subject, not merely a related one.

        Search is allowed to rescue spelling and word-order variants —
        `Ionian earthquake of 1953` finding `1953 Ionian earthquake` — and
        nothing else. A hit that introduces an identifying word the subject
        does not have is a different thing: `Memphis` returning
        `Memphis, Tennessee` is how an Egyptian city came to cite a city in
        the United States, on both the never-prefetched and batch-failed
        paths.
        """
        import re as _re
        import unicodedata

        def tokens(text):
            folded = "".join(c for c in unicodedata.normalize("NFKD", text or "")
                             if not unicodedata.combining(c)).lower()
            return {_re.sub(r"(ies|es|s)$", "", t)
                    for t in _re.sub(r"[^a-z0-9 ]", " ", folded).split()
                    if t and t not in ("the", "of", "at", "in", "by", "a", "an",
                                       "and", "from")}

        subject_tokens = tokens(subject)
        allowed = set(subject_tokens)
        for hint in hints or ():
            allowed |= tokens(hint)
        theirs = tokens(title)
        if not theirs or not theirs <= allowed:
            return False
        # **The result must still name the subject.** A subset rule alone lets
        # a hit be made entirely of context words: `Apollonia` with hints
        # `Albania, Fier County` matched `Fier`, and `Thebes` with `Egypt,
        # Luxor Governorate` matched `Luxor` — both real places, neither the
        # subject. Context may *narrow* a name, never replace it.
        return bool(theirs & subject_tokens)

    def _articles_only(titles):
        """Drop disambiguation pages. Search results do not say which they are."""
        if not titles:
            return []
        try:
            data = _call({"action": "query", "prop": "pageprops",
                          "titles": "|".join(titles), "redirects": 1})
        except Exception:  # noqa: BLE001 - if we cannot tell, do not guess
            return []
        bad = {page["title"] for page in
               (data.get("query", {}).get("pages") or {}).values()
               if "disambiguation" in (page.get("pageprops") or {})}
        return [t for t in titles if t not in bad]

    def _search(subject: str, hints=()):
        for attempt in (1, 2):
            try:
                terms = " ".join([subject, *(list(hints or [])[:1])])
                data = _call({"action": "query", "list": "search",
                              "srsearch": terms,
                              "srlimit": max(per_atom, 1) + 3})
                hits = [h["title"] for h in
                        (data.get("query", {}).get("search") or [])]
                hits = [t for t in hits if _same_subject(subject, t, hints)]
                return [_url(t) for t in _articles_only(hits)]
            except Exception as e:  # noqa: BLE001
                if "429" in str(e) and attempt == 1:
                    time.sleep(2.0)
                    continue
                log.warning("%s: wikipedia lookup failed: %s", subject, e)
                return []
        return []

    def ask(subject: str, atom_type: str, hints=()) -> List[str]:
        if subject in cache:
            return cache[subject][:per_atom]
        # An ambiguous title is no longer a dead end *if* the atom says where
        # it is: `Memphis` with `country='Egypt'` resolves, while `Memphis`
        # alone still declines rather than guessing at Tennessee.
        if subject in ambiguous and not hints:
            return []
        if subject in throttled:
            return []
        key = (subject, tuple(hints or ()))
        if key in cache:
            return cache[key][:per_atom]
        time.sleep(delay)
        found = _search(subject, hints)[:per_atom]
        cache[key] = found
        return found

    ask.prefetch = prefetch
    return ask


# ---------------------------------------------------------------------------
# Identity: does one name denote the same subject as another?
# ---------------------------------------------------------------------------

def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _wiki_api(params, attempts: int = 4):
    """One Wikipedia API call, backing off when asked to slow down.

    A module-level twin of the helper nested in `wikipedia_lookup`. Kept
    separate rather than hoisted: that one carries per-run caches closed over
    by its enclosing function, and untangling a working rate-limiter is not
    worth the risk for a second caller.
    """
    import time
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(f"{WIKIPEDIA_API}?{query}",
                                     headers={"User-Agent": _UA})
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response).get("query", {})
        except Exception as e:  # noqa: BLE001
            last = e
            if "429" not in str(e) or attempt == attempts - 1:
                raise
            wait = 5.0 * (3 ** attempt)
            log.info("wikipedia asked us to slow down; waiting %.0fs", wait)
            time.sleep(wait)
    raise last


def resolve_titles(titles, batch: int = 40, delay: float = 0.3) -> dict:
    """`title -> (pageid, resolved_title, is_disambiguation)`, following redirects.

    A redirect is Wikipedia asserting that two spellings reach one article, which
    is the closest thing to a free, checkable identity claim available. Titles are
    normalised and redirect-chased server-side; the chase is repeated locally
    because a normalisation can feed a redirect.
    """
    found: dict = {}
    titles = [t for t in dict.fromkeys(titles) if t and t.strip()]
    for start in range(0, len(titles), batch):
        chunk = titles[start:start + batch]
        params = {"action": "query", "titles": "|".join(chunk), "redirects": "1",
                  "prop": "pageprops", "format": "json"}
        try:
            payload = _wiki_api(params)
        except Exception as e:  # noqa: BLE001 - a failed lookup abstains, never guesses
            log.warning("title lookup failed for %d title(s): %s", len(chunk), e)
            for title in chunk:
                found[title] = (None, "", False)
            continue
        hop = {}
        for entry in payload.get("normalized", []):
            hop[entry["from"]] = entry["to"]
        for entry in payload.get("redirects", []):
            hop[entry["from"]] = entry["to"]
        by_title = {page["title"]: (page.get("pageid"),
                                    "disambiguation" in (page.get("pageprops") or {}))
                    for page in payload.get("pages", {}).values()}
        for title in chunk:
            landed = title
            for _ in range(4):
                landed = hop.get(landed, landed)
            page_id, ambiguous = by_title.get(landed, (None, False))
            found[title] = (page_id, landed, ambiguous)
        _sleep(delay)
    return found


def identity_verdict(subject: str, alias: str, resolved: dict):
    """`("same"|"wrong"|None, evidence)` for "is `alias` a name for `subject`?".

    **Redirect equality only, and it abstains rather than guesses.** Two other
    methods were measured against hand-adjudicated rows and both rejected:

    - *Article text contains the alias* — **too often wrong**. An article
      mentions relatives, successors and predecessors, so "mentioned" is not
      "is". Restricting to the intro scored identically.
    - *Asking a search-capable model* — wrong on some rows, and wrong while
      sounding certain: it assigned Ramesses VI's princely name to Ramesses V.

    Redirect equality made no error where it committed, and abstained on the
    rest — Egyptian throne names are not article titles, so nothing resolves.
    Some rows decided with no observed error beats all the rows decided with
    a fair share wrong, because an identity written here is permanent: `same`
    appends to `aliases.txt`, and that line stops a later run from regenerating
    the distinction.
    """
    subject_id, subject_at, subject_ambiguous = resolved.get(subject, (None, "", False))
    alias_id, alias_at, alias_ambiguous = resolved.get(alias, (None, "", False))
    if subject_id is None or alias_id is None:
        missing = subject if subject_id is None else alias
        return None, f"no Wikipedia article for {missing!r}"
    if subject_ambiguous or alias_ambiguous:
        return None, "resolves to a disambiguation page"
    if subject_id == alias_id:
        return "same", f"both redirect to {subject_at!r}"
    return "wrong", f"{subject_at!r} and {alias_at!r} are separate articles"


def subject_leads(titles, batch: int = 20, delay: float = 0.3) -> dict:
    """`title -> the opening of its Wikipedia article`, for rows nothing decided.

    Not a verdict — **evidence for a human**. Four automated readings of that
    same text were measured and every one of them was wrong more often than
    redirect equality (top-1 search, Wikidata aliases, top-3 search,
    first-sentence matching; redirect equality made none). The text is reliable; the
    machine reading of it is not.

    So it is attached to abstained rows rather than acted on. `Ramesses V`'s
    article opens "Usermaatre Sekheperenre Ramesses V", which settles that row
    the moment a person reads it and which no rule here got right.
    """
    found: dict = {}
    titles = [t for t in dict.fromkeys(titles) if t and t.strip()]
    for start in range(0, len(titles), batch):
        chunk = titles[start:start + batch]
        try:
            payload = _wiki_api({"action": "query", "titles": "|".join(chunk),
                                 "redirects": "1", "prop": "extracts",
                                 "explaintext": "1", "exintro": "1",
                                 "exlimit": "max"})
        except Exception as e:  # noqa: BLE001
            log.warning("lead lookup failed: %s", e)
            payload = {}
        hop = {}
        for entry in payload.get("normalized", []):
            hop[entry["from"]] = entry["to"]
        for entry in payload.get("redirects", []):
            hop[entry["from"]] = entry["to"]
        by_title = {page["title"]: " ".join((page.get("extract") or "").split())
                    for page in payload.get("pages", {}).values()}
        for title in chunk:
            landed = title
            for _ in range(4):
                landed = hop.get(landed, landed)
            found[title] = by_title.get(landed, "")
        _sleep(delay)
    return found


class WikipediaIdentity:
    """The identity source `lint --via wikipedia` runs: the three module
    functions above, in the shape `DomainPack.identity_resolvers` asks for."""

    resolve_titles = staticmethod(resolve_titles)
    subject_leads = staticmethod(subject_leads)
    identity_verdict = staticmethod(identity_verdict)


IDENTITY = WikipediaIdentity()
