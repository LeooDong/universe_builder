"""Appending verified source URLs. Offline: `ask` and `resolves` are injected."""

from __future__ import annotations

import json

import pytest

from universe_builder import sources, store
from universe_builder.domains.history import wikipedia

MODEL = "gpt-5-nano"


def _atom(atom_type, subject, payload=None):
    path = store.path_for(atom_type, None, subject, MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"name": {"common": subject}}),
                    encoding="utf-8")
    return path


def _ask(*urls):
    return lambda subject, atom_type: list(urls)


@pytest.fixture
def all_resolve(monkeypatch):
    monkeypatch.setattr(sources, "resolves", lambda u, timeout=15: (True, "HTTP 200"))


class TestUrlVerification:
    """A model asked for sources produces plausible URLs whether or not it
    visited them. The HTTP response is what promotes a candidate to a citation."""

    def test_a_typographic_dash_means_it_was_typed_not_visited(self):
        ok, why = sources.resolves("https://example.org/case‑law‑doc")

        assert ok is False and "typographic" in why

    def test_a_non_http_string_is_refused(self):
        assert sources.resolves("not-a-url")[0] is False

    def test_a_403_counts_as_existing(self, monkeypatch):
        """Institutional sites block scripts while serving browsers fine — the
        Met and Michigan both did during the fact-check audit. Treating that as
        fabricated would discard the citations most worth having."""
        from urllib.error import HTTPError

        def blocked(*a, **k):
            raise HTTPError("u", 403, "Forbidden", None, None)

        monkeypatch.setattr(sources, "urlopen", blocked)
        ok, why = sources.resolves("https://www.metmuseum.org/art/1")

        assert ok is True and "403" in why

    def test_a_404_is_a_real_failure(self, monkeypatch):
        from urllib.error import HTTPError

        def missing(*a, **k):
            raise HTTPError("u", 404, "Not Found", None, None)

        monkeypatch.setattr(sources, "urlopen", missing)

        assert sources.resolves("https://en.wikipedia.org/wiki/Nope")[0] is False

    @pytest.mark.parametrize("url", [
        "http://localhost:8080/admin/page",
        "http://127.0.0.1/secret/page",
        "http://192.168.1.1/router/page",
        "http://[::1]/x/page",
        "http://169.254.169.254/latest/meta-data",
        "http://printer.local/status/page",
    ])
    def test_a_non_public_host_is_refused_without_a_request(self, monkeypatch, url):
        """The URL is model-written; opening it must not reach this machine or its network."""
        def opened(*a, **k):
            raise AssertionError("a non-public host was requested")

        monkeypatch.setattr(sources, "urlopen", opened)

        assert sources.resolves(url) == (False, "not a public host")

    def test_a_redirect_to_a_non_public_host_is_not_followed(self):
        from email.message import Message
        from urllib.error import URLError
        from urllib.request import Request

        with pytest.raises(URLError, match="not a public host"):
            sources._PublicRedirects().redirect_request(
                Request("https://example.org/a/page"), None, 302, "Found", Message(),
                "http://127.0.0.1/admin")

    def test_a_redirect_off_http_is_not_followed(self):
        """urllib follows a redirect to `ftp:`, and its FTP handler resolves
        and connects without the guard."""
        from email.message import Message
        from urllib.error import URLError
        from urllib.request import Request

        with pytest.raises(URLError, match="not a public host"):
            sources._PublicRedirects().redirect_request(
                Request("https://example.org/a/page"), None, 302, "Found", Message(),
                "ftp://example.org/pub/page")

    @staticmethod
    def _dns(monkeypatch, answers):
        """Resolve each name to `answers[name]` and record every connection;
        no lookup or connection leaves the process."""
        import socket

        def getaddrinfo(host, port, *a, **k):
            return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                    for ip in answers[host]]

        connected = []

        def create_connection(address, *a, **k):
            connected.append(address)
            raise ConnectionRefusedError("test: no real connection")

        monkeypatch.setattr(sources.socket, "getaddrinfo", getaddrinfo)
        monkeypatch.setattr(sources.socket, "create_connection", create_connection)
        return connected

    @pytest.mark.parametrize("host, ip", [
        ("rebind.example.org", "127.0.0.1"),     # a public name pointing inward
        ("2130706433", "127.0.0.1"),             # 127.0.0.1 as one number
        ("0x7f.1", "127.0.0.1"),
        ("mapped.example.org", "::ffff:10.0.0.1"),
        ("mixed.example.org", "93.184.215.14 192.168.0.1"),
        ("multicast.example.org", "224.0.0.1"),             # is_global on 3.10-3.12
        ("sixtofour.example.org", "2002:7f00:1::1"),         # 6to4 around 127.0.0.1
        ("nat64.example.org", "64:ff9b::7f00:1"),            # NAT64 around 127.0.0.1
        ("compat.example.org", "::7f00:1"),                  # IPv4-compatible
    ])
    def test_a_name_resolving_to_a_non_public_address_is_never_connected(self, monkeypatch, host, ip):
        """What the URL says is not where it goes: the resolved address is
        what gets checked, before any connection."""
        connected = self._dns(monkeypatch, {host: ip.split()})

        assert sources.resolves(f"http://{host}/a/page") == (False, "not a public host")
        assert connected == []

    def test_a_public_name_connects_to_the_address_that_was_checked(self, monkeypatch):
        """Connecting by name would look it up again, and a second answer can differ."""
        connected = self._dns(monkeypatch, {"example.org": ["93.184.215.14"]})

        assert sources.resolves("https://example.org/a/page") == (False, "URLError")
        assert connected == [("93.184.215.14", 443)]

    @pytest.mark.parametrize("ip", ["64:ff9b::808:808", "::ffff:8.8.8.8"])
    def test_an_ipv6_form_of_a_public_ipv4_address_is_public(self, ip):
        import ipaddress

        assert sources._public_ip(ipaddress.ip_address(ip)) is True
        assert sources._non_public_host(f"http://[{ip}]/a/page") is False

    def test_an_unreachable_address_falls_back_to_the_next_public_one(self, monkeypatch):
        """An IPv6 answer on a network with no IPv6 route must not fail a host
        whose IPv4 address works."""
        connected = self._dns(monkeypatch, {"example.org": ["2606:2800:21f:cb07:6820:80da:af6b:8b2c",
                                                            "93.184.215.14"]})
        sock = object()

        def create_connection(address, *a, **k):
            connected.append(address)
            if ":" in address[0]:
                raise OSError("network is unreachable")
            return sock

        monkeypatch.setattr(sources.socket, "create_connection", create_connection)

        assert sources._connect_public(("example.org", 443), 15) is sock
        assert [ip for ip, _ in connected] == ["2606:2800:21f:cb07:6820:80da:af6b:8b2c", "93.184.215.14"]

    def test_the_request_identifies_itself(self, monkeypatch):
        seen = {}

        def capture(request, timeout):
            seen["ua"] = request.get_header("User-agent")
            raise sources.HTTPError("u", 404, "Not Found", None, None)

        monkeypatch.setattr(sources, "urlopen", capture)
        sources.resolves("https://en.wikipedia.org/wiki/Nope")

        assert seen["ua"].startswith("universe_builder/") and "Mozilla" not in seen["ua"]


class TestProposing:
    def test_verified_urls_are_separated_from_rejected(self, workspace, monkeypatch):
        monkeypatch.setattr(sources, "resolves",
                            lambda u, t=15: ("good" in u, "HTTP 200" if "good" in u
                                             else "HTTP 404"))
        path = _atom("city", "Memphis")

        p = sources.propose("city", path,
                            _ask("https://good.org/a", "https://bad.org/b"))

        assert p.verified == ["https://good.org/a"]
        assert p.rejected[0][0] == "https://bad.org/b"

    def test_a_url_already_cited_anywhere_is_not_offered_again(self, workspace,
                                                               all_resolve):
        path = _atom("city", "Memphis", payload={
            "name": {"common": "Memphis",
                     "sources": ["see https://en.wikipedia.org/wiki/Memphis"]}})

        p = sources.propose("city", path,
                            _ask("https://en.wikipedia.org/wiki/Memphis"))

        assert p.verified == []
        assert p.rejected == [("https://en.wikipedia.org/wiki/Memphis", "already cited")]

    def test_preferred_hosts_come_first(self, workspace, all_resolve):
        path = _atom("city", "Memphis")

        p = sources.propose("city", path,
                            _ask("https://random.blog/x", "https://en.wikipedia.org/wiki/M"))

        assert p.verified[0] == "https://en.wikipedia.org/wiki/M"

    def test_a_failing_ask_costs_one_atom_not_the_run(self, workspace):
        path = _atom("city", "Memphis")

        def boom(subject, atom_type):
            raise RuntimeError("no")

        assert sources.propose("city", path, boom).offered == []


class TestApplying:
    def test_it_appends_and_never_removes(self, workspace, all_resolve):
        """A complete atom, because `apply` refuses to write one that would no
        longer validate — the rule that keeps a citation fix from being the
        thing that makes an atom unreadable."""
        from universe_builder.domains.history.models.city import City
        from tests.factories import minimal_payload

        payload = minimal_payload(City)
        payload["name"]["common"] = "Memphis"
        payload["name"]["sources"] = ["Shaw, Oxford History"]
        path = _atom("city", "Memphis", payload=payload)
        p = sources.propose("city", path, _ask("https://en.wikipedia.org/wiki/M"))

        sources.apply(p, dry_run=False)

        after = json.loads(path.read_text())["name"]["sources"]
        assert after == ["Shaw, Oxford History", "https://en.wikipedia.org/wiki/M"]

    def test_a_dry_run_writes_nothing(self, workspace, all_resolve):
        path = _atom("city", "Memphis")
        p = sources.propose("city", path, _ask("https://en.wikipedia.org/wiki/M"))

        assert sources.apply(p) is False
        assert "wikipedia" not in path.read_text()

    def test_nothing_verified_means_nothing_written(self, workspace, monkeypatch):
        monkeypatch.setattr(sources, "resolves", lambda u, t=15: (False, "HTTP 404"))
        path = _atom("city", "Memphis")
        p = sources.propose("city", path, _ask("https://bad.org/x"))

        assert sources.apply(p, dry_run=False) is False

    def test_an_atom_that_would_not_validate_is_left_alone(self, workspace,
                                                           all_resolve, monkeypatch):
        path = _atom("city", "Memphis")
        p = sources.propose("city", path, _ask("https://en.wikipedia.org/wiki/M"))
        monkeypatch.setattr(sources, "_validate",
                            lambda at, pl: (_ for _ in ()).throw(ValueError("bad")))

        assert sources.apply(p, dry_run=False) is False
        assert "wikipedia" not in path.read_text()


class TestAResolvingUrlIsNotAutomaticallyASource:
    """HTTP 200 is necessary and not sufficient. A site search, a category
    listing and a raw search-engine query all answer 200 while containing no
    claim about the subject, and asking the model not to offer them does not
    stop it."""

    @pytest.mark.parametrize("url", [
        "https://en.wikipedia.org/wiki/Special:Search?search=x",
        "https://commons.wikimedia.org/wiki/Category:Gastraphetes",
        "https://en.wikipedia.org/wiki/Memphis_(disambiguation)",
        "https://www.britishmuseum.org/collection/search?keyword=stela",
        "https://example.org/search",
    ])
    def test_listing_and_query_pages_are_refused(self, url):
        assert sources.cites_something(url)[0] is False

    @pytest.mark.parametrize("url", [
        "https://www.google.com/search?q=egyptian+faience",
        "https://duckduckgo.com/?q=x",
        "https://www.youtube.com/watch?v=abc",
        "https://www.amazon.com/dp/B000",
        "https://www.pinterest.com/pin/123/",
        "https://medium.com/@someone/post",
    ])
    def test_hosts_that_never_carry_citations_are_refused(self, url):
        assert sources.cites_something(url)[0] is False

    @pytest.mark.parametrize("url", [
        "https://en.wikipedia.org/wiki/Talk:Memphis,_Egypt",
        "https://en.wikipedia.org/wiki/User:Someone",
        "https://en.wikipedia.org/wiki/Wikipedia:Verifiability",
    ])
    def test_wiki_machinery_namespaces_are_refused(self, url):
        assert sources.cites_something(url)[0] is False

    def test_a_listing_article_is_refused(self):
        """A list is not a source about any one thing it lists."""
        assert sources.cites_something(
            "https://en.wikipedia.org/wiki/List_of_ancient_Egyptian_sites")[0] is False

    def test_a_bare_homepage_is_refused(self):
        """A front page is not a page about the subject."""
        assert sources.cites_something("https://www.britishmuseum.org/")[0] is False

    def test_a_commons_file_page_is_kept(self):
        """`File:` is a media page for the object itself — a real citation. An
        earlier blanket namespace rule threw these away with the categories."""
        assert sources.cites_something(
            "https://commons.wikimedia.org/wiki/File:Rosetta_Stone.jpg")[0] is True

    def test_a_benign_s_parameter_is_not_a_search(self):
        """`s=` is a common session or style parameter; treating it as a query
        rejected `example.ac.uk/paper?s=2&id=44`, a real article."""
        assert sources.cites_something(
            "https://example.ac.uk/paper?s=2&id=44")[0] is True

    def test_a_wikipedia_index_php_article_url_is_kept(self):
        assert sources.cites_something(
            "https://en.wikipedia.org/w/index.php?title=Memphis,_Egypt")[0] is True

    def test_an_object_page_under_a_search_path_is_kept(self):
        """The Met serves individual objects at
        `/art/collection/search/100004` — a page about one thing, not a query.
        A blanket `/search` rule threw away exactly the citations worth most."""
        assert sources.cites_something(
            "https://www.metmuseum.org/art/collection/search/100004")[0] is True

    def test_ordinary_article_pages_are_kept(self):
        for url in ("https://en.wikipedia.org/wiki/Egyptian_faience",
                    "https://www.britannica.com/topic/faience",
                    "https://doi.org/10.1000/xyz"):
            assert sources.cites_something(url)[0] is True, url


class TestTheWikipediaResolver:
    """The paid search was doing a title lookup: every URL it added was a
    Wikipedia article, at $10/1k search calls plus the input tokens of each. This does the same lookup for nothing."""

    @staticmethod
    def _fake_api(monkeypatch, titles=None, error=None):
        import io
        import json as _json
        import urllib.request

        def fake_urlopen(request, timeout=None):
            if error:
                raise error
            body = _json.dumps({"query": {"search": [{"title": t}
                                                     for t in (titles or [])]}})
            return io.BytesIO(body.encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def test_it_builds_an_article_url_from_the_title(self, monkeypatch):
        self._fake_api(monkeypatch, ["Nika riots"])

        assert wikipedia.wikipedia_lookup()("Nika Riots", "event") == [
            "https://en.wikipedia.org/wiki/Nika_riots"]

    def test_it_returns_one_url_by_default(self, monkeypatch):
        """The model's habit of offering three produced `Justinian I` and
        `Narses` for the Nika Riots — real pages, about people *in* the subject
        rather than about it."""
        self._fake_api(monkeypatch, ["Nika riots", "Justinian I", "Narses"])

        assert len(wikipedia.wikipedia_lookup()("Nika Riots", "event")) == 1

    def test_per_atom_raises_the_cap(self, monkeypatch):
        # Titles must be the same subject as the query, or `_same_subject`
        # rejects them — search may rescue a variant, never substitute.
        self._fake_api(monkeypatch, ["Alpha Beta", "Alpha", "Beta"])

        assert len(wikipedia.wikipedia_lookup(per_atom=2)("Alpha Beta", "event")) == 2

    def test_titles_with_punctuation_are_quoted(self, monkeypatch):
        self._fake_api(monkeypatch, ["Euclid's theorem"])

        url = wikipedia.wikipedia_lookup()("Euclid's theorem", "discovery")[0]

        assert " " not in url and url.endswith("Euclid%27s_theorem")

    def test_no_results_is_an_empty_list_not_an_error(self, monkeypatch):
        self._fake_api(monkeypatch, [])

        assert wikipedia.wikipedia_lookup()("Nothing At All", "event") == []

    def test_a_failing_api_costs_one_atom_not_the_run(self, monkeypatch):
        self._fake_api(monkeypatch, error=OSError("network down"))

        assert wikipedia.wikipedia_lookup()("Memphis", "city") == []


class TestTheAlreadyCitedFilter:
    """Two spellings of one page defeated it, so the resolver could add a URL
    the atom already carried."""

    def test_an_apostrophe_does_not_truncate_a_url(self):
        """Apostrophes are legal in URLs. Excluding them cut
        `.../wiki/Euclid's_theorem` down to `.../wiki/Euclid`, and the filter
        then compared a fragment."""
        found = sources.existing_urls(
            {"a": "https://en.wikipedia.org/wiki/Euclid's_theorem"})

        assert found == {"https://en.wikipedia.org/wiki/Euclid's_theorem"}

    def test_percent_encoding_is_the_same_page(self):
        already = sources.existing_urls(
            {"a": "https://en.wikipedia.org/wiki/Euclid's_theorem"})

        assert sources.canonical_url(
            "https://en.wikipedia.org/wiki/Euclid%27s_theorem") in already

    def test_trailing_sentence_punctuation_is_not_part_of_the_url(self):
        found = sources.existing_urls(
            {"a": "see https://en.wikipedia.org/wiki/Memphis for more."})

        assert found == {"https://en.wikipedia.org/wiki/Memphis"}

    def test_a_title_keeps_its_parentheses(self):
        """`)` was stripped as sentence punctuation, so
        `.../Battle_of_Zela_(47_BC)` never matched itself and the resolver
        re-offered pages the atom already cited. A `)` closing
        a parenthesis the URL never opened is still prose."""
        found = sources.existing_urls({
            "a": "https://en.wikipedia.org/wiki/Battle_of_Zela_(47_BC)",
            "b": "(see https://en.wikipedia.org/wiki/Zela)",
            "c": "(see https://en.wikipedia.org/wiki/Cretan_revolt_(1866))."})

        assert found == {"https://en.wikipedia.org/wiki/Battle_of_Zela_(47_BC)",
                         "https://en.wikipedia.org/wiki/Zela",
                         "https://en.wikipedia.org/wiki/Cretan_revolt_(1866)"}

    def test_host_case_and_trailing_slash_do_not_matter(self):
        already = sources.existing_urls(
            {"a": "https://EN.wikipedia.org/wiki/Nika_riots/"})

        assert sources.canonical_url(
            "https://en.wikipedia.org/wiki/Nika_riots") in already

    def test_a_genuinely_new_url_is_still_new(self):
        already = sources.existing_urls(
            {"a": "https://en.wikipedia.org/wiki/Memphis"})

        assert sources.canonical_url(
            "https://en.wikipedia.org/wiki/Thebes") not in already

    def test_the_resolver_does_not_re_add_an_encoded_duplicate(self, workspace,
                                                               all_resolve):
        path = _atom("city", "Memphis", payload={"name": {
            "common": "Memphis",
            "sources": ["https://en.wikipedia.org/wiki/Euclid's_theorem"]}})

        p = sources.propose("city", path,
                            _ask("https://en.wikipedia.org/wiki/Euclid%27s_theorem"))

        assert p.verified == []
        assert p.rejected[0][1] == "already cited"


class TestCandidatesAreDedupedAgainstEachOther:
    """`dict.fromkeys` removes only byte-identical repeats, and a resolver can
    easily offer one page under two spellings — `Euclid's_theorem` beside
    `Euclid%27s_theorem`, or a trailing slash. Both passed verification and
    both were written: the same page, twice."""

    def test_two_spellings_of_one_page_yield_one_url(self, workspace, all_resolve):
        path = _atom("city", "Memphis")

        p = sources.propose("city", path, _ask(
            "https://en.wikipedia.org/wiki/Euclid's_theorem",
            "https://en.wikipedia.org/wiki/Euclid%27s_theorem"))

        assert len(p.verified) == 1
        assert p.rejected[0][1] == "already cited"

    def test_a_trailing_slash_is_the_same_page(self, workspace, all_resolve):
        path = _atom("city", "Memphis")

        p = sources.propose("city", path, _ask(
            "https://en.wikipedia.org/wiki/Nika_riots/",
            "https://en.wikipedia.org/wiki/Nika_riots"))

        assert len(p.verified) == 1

    def test_genuinely_distinct_pages_all_survive(self, workspace, all_resolve):
        path = _atom("city", "Memphis")

        p = sources.propose("city", path, _ask(
            "https://en.wikipedia.org/wiki/Memphis",
            "https://en.wikipedia.org/wiki/Thebes"))

        assert len(p.verified) == 2

    def test_the_write_also_compares_canonically(self, workspace, all_resolve):
        """`propose` reads the atom and `apply` re-reads it, so anything
        written in between must still match across encodings."""
        from universe_builder.domains.history.models.city import City
        from tests.factories import minimal_payload

        payload = minimal_payload(City)
        payload["name"]["common"] = "Memphis"
        payload["name"]["sources"] = ["https://en.wikipedia.org/wiki/Euclid's_theorem"]
        path = _atom("city", "Memphis", payload=payload)
        # A proposal built before that URL was present.
        stale = sources.Proposal(atom_type="city", file=path.name, subject="Memphis",
                                 verified=["https://en.wikipedia.org/wiki/Euclid%27s_theorem"])

        assert sources.apply(stale, dry_run=False) is False


class TestDisambiguationPages:
    """A disambiguation page is not a source, and almost none carry a marker in
    the URL — `Memphis`, `Thebes`, `Apollonia` and `Pausanias` are all
    disambiguation pages at their bare title. This corpus is made largely of
    exactly such names, so the URL-shape rule catches nearly none of them.
    Only the API knows, through `pageprops`."""

    @staticmethod
    def _api(monkeypatch, pages, search=None):
        import io
        import json as _json
        import urllib.request

        def fake_urlopen(request, timeout=None):
            url = request.full_url
            if "list=search" in url:
                body = {"query": {"search": [{"title": t} for t in (search or [])]}}
            else:
                body = {"query": {"pages": {str(i): p for i, p in enumerate(pages)}}}
            return io.BytesIO(_json.dumps(body).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def test_a_disambiguation_page_is_not_cached(self, monkeypatch):
        self._api(monkeypatch, [{"title": "Memphis",
                                 "fullurl": "https://en.wikipedia.org/wiki/Memphis",
                                 "pageprops": {"disambiguation": ""}}])
        ask = wikipedia.wikipedia_lookup()

        assert ask.prefetch(["Memphis"]) == 0

    def test_a_real_article_is_cached(self, monkeypatch):
        self._api(monkeypatch, [{"title": "Battle of Sellasia",
                                 "fullurl": "https://en.wikipedia.org/wiki/Battle_of_Sellasia"}])
        ask = wikipedia.wikipedia_lookup()

        assert ask.prefetch(["Battle of Sellasia"]) == 1
        assert ask("Battle of Sellasia", "event") == [
            "https://en.wikipedia.org/wiki/Battle_of_Sellasia"]

    def test_an_ambiguous_title_does_not_fall_through_to_search(self, monkeypatch):
        """Search answers a known-ambiguous name with the most popular article:
        `Memphis` returned `Memphis, Tennessee` for an Egyptian city. Declining
        is right — a wrong citation is worse than none."""
        self._api(monkeypatch,
                  [{"title": "Memphis",
                    "fullurl": "https://en.wikipedia.org/wiki/Memphis",
                    "pageprops": {"disambiguation": ""}}],
                  search=["Memphis, Tennessee"])
        ask = wikipedia.wikipedia_lookup()
        ask.prefetch(["Memphis"])

        assert ask("Memphis", "city") == []

    def test_search_results_are_filtered_too(self, monkeypatch):
        """A title never seen by the batch still must not return a
        disambiguation page."""
        self._api(monkeypatch,
                  [{"title": "Pausanias", "pageprops": {"disambiguation": ""}}],
                  search=["Pausanias"])
        ask = wikipedia.wikipedia_lookup()

        assert ask("Pausanias", "person") == []


class TestSearchMayNotSubstituteADifferentSubject:
    """Two failure paths both reached search unguarded — prefetch never called,
    and the batch call failing — and search answered `Memphis` with
    `Memphis, Tennessee`. Search may rescue a spelling or word-order variant
    and nothing else."""

    @staticmethod
    def _api(monkeypatch, search, batch_fails=False):
        import io
        import json as _json
        import urllib.request

        def fake_urlopen(request, timeout=None):
            url = request.full_url
            if "list=search" in url:
                body = {"query": {"search": [{"title": t} for t in search]}}
            elif batch_fails and "prop=info" in url:
                raise OSError("batch down")
            else:                       # the pageprops check
                body = {"query": {"pages": {str(i): {"title": t}
                                            for i, t in enumerate(search)}}}
            return io.BytesIO(_json.dumps(body).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def test_an_extra_identifying_word_means_a_different_subject(self, monkeypatch):
        self._api(monkeypatch, ["Memphis, Tennessee"])

        assert wikipedia.wikipedia_lookup()("Memphis", "city") == []

    def test_a_word_order_variant_is_still_rescued(self, monkeypatch):
        self._api(monkeypatch, ["1953 Ionian earthquake"])

        assert wikipedia.wikipedia_lookup()(
            "Ionian earthquake of 1953", "event") == [
                "https://en.wikipedia.org/wiki/1953_Ionian_earthquake"]

    def test_case_and_plurals_do_not_block_a_rescue(self, monkeypatch):
        self._api(monkeypatch, ["Nika riots"])

        assert wikipedia.wikipedia_lookup()("Nika Riots", "event") == [
            "https://en.wikipedia.org/wiki/Nika_riots"]

    def test_a_failed_batch_does_not_open_the_gate(self, monkeypatch):
        """The batch failing must not silently restore the unguarded search."""
        self._api(monkeypatch, ["Memphis, Tennessee"], batch_fails=True)
        ask = wikipedia.wikipedia_lookup()
        ask.prefetch(["Memphis"])

        assert ask("Memphis", "city") == []


class TestContextDisambiguatesAName:
    """An ancient corpus is full of names ambiguous on their own. Without
    context the resolver can only decline them, skipping exactly the atoms
    most worth citing."""

    @staticmethod
    def _api(monkeypatch, search):
        import io
        import json as _json
        import urllib.request

        def fake_urlopen(request, timeout=None):
            if "list=search" in request.full_url:
                body = {"query": {"search": [{"title": t} for t in search]}}
            else:
                body = {"query": {"pages": {str(i): {"title": t}
                                            for i, t in enumerate(search)}}}
            return io.BytesIO(_json.dumps(body).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    def test_country_admits_the_right_one(self, monkeypatch):
        self._api(monkeypatch, ["Memphis, Egypt"])

        assert wikipedia.wikipedia_lookup()("Memphis", "city", ["Egypt"]) == [
            "https://en.wikipedia.org/wiki/Memphis%2C_Egypt"]

    def test_the_wrong_one_is_still_refused(self, monkeypatch):
        self._api(monkeypatch, ["Memphis, Tennessee"])

        assert wikipedia.wikipedia_lookup()("Memphis", "city", ["Egypt"]) == []

    def test_context_may_narrow_a_name_never_replace_it(self, monkeypatch):
        """A subset rule alone let a hit be made entirely of context words:
        `Apollonia` with `Albania, Fier County` matched `Fier`, and `Thebes`
        with `Egypt, Luxor Governorate` matched `Luxor`. Both real places,
        neither the subject."""
        self._api(monkeypatch, ["Fier"])

        assert wikipedia.wikipedia_lookup()(
            "Apollonia", "city", ["Albania", "Fier County"]) == []

    def test_no_context_still_declines(self, monkeypatch):
        self._api(monkeypatch, ["Memphis, Tennessee"])

        assert wikipedia.wikipedia_lookup()("Memphis", "city") == []


class TestContextIsReadFromTheAtom:
    def test_a_city_uses_its_own_location(self):
        payload = {"location": {"country": "Egypt", "state": "Giza Governorate"}}

        assert sources.context_terms(payload, "city") == ["Egypt", "Giza Governorate"]

    def test_a_person_uses_birthplace(self):
        payload = {"birth": {"location": {"country": "Greece"}}}

        assert sources.context_terms(payload, "person") == ["Greece"]

    def test_an_artifact_prefers_where_it_was_found(self):
        payload = {"discoveryLocation": {"country": "Egypt"},
                   "currentLocation": {"country": "United Kingdom"}}

        assert sources.context_terms(payload, "artifact") == ["Egypt"]

    def test_no_location_is_no_context_not_an_error(self):
        assert sources.context_terms({"name": {}}, "city") == []

    def test_propose_passes_it_through(self, workspace, all_resolve):
        seen = {}

        def ask(subject, atom_type, hints=()):
            seen["hints"] = list(hints)
            return []

        path = _atom("city", "Memphis", payload={
            "name": {"common": "Memphis"}, "location": {"country": "Egypt"}})
        sources.propose("city", path, ask)

        assert seen["hints"] == ["Egypt"]

    def test_a_resolver_without_context_still_works(self, workspace, all_resolve):
        """`ask(subject, atom_type)` predates hints and must keep working."""
        path = _atom("city", "Memphis", payload={
            "name": {"common": "Memphis"}, "location": {"country": "Egypt"}})

        p = sources.propose("city", path,
                            lambda s, a: ["https://en.wikipedia.org/wiki/M"])

        assert p.verified == ["https://en.wikipedia.org/wiki/M"]


def test_has_link_distinguishes_cited_from_uncited():
    """`--missing` must key off a real URL, not merely a non-empty sources list.

    Almost every atom cites *something* -- "Encyclopaedia Britannica, 'Asyut'"
    -- so a truthy-sources test would mark the whole corpus as done and the
    paid pass would examine nothing.
    """
    assert not sources.has_link({"name": {"sources": ["Britannica, 'Asyut'"]}})
    assert not sources.has_link({"name": {"sources": []}})
    assert not sources.has_link({})
    assert sources.has_link(
        {"name": {"sources": ["https://en.wikipedia.org/wiki/Asyut"]}})


class TestIdentityViaWikipedia:
    """Redirect equality as a free identity check. It must abstain rather than
    guess: `same` writes a permanent line to aliases.txt, and that line stops a
    later run from regenerating the distinction."""

    def _resolved(self, monkeypatch, pages, redirects=(), disambig=()):
        def fake(params, attempts=4):
            titles = params["titles"].split("|")
            return {"redirects": [{"from": a, "to": b} for a, b in redirects],
                    "pages": {str(i): {"title": t, "pageid": pages.get(t),
                                       "pageprops": ({"disambiguation": ""}
                                                     if t in disambig else {})}
                              for i, t in enumerate(
                                  {redirects_map(redirects).get(x, x) for x in titles})}}

        def redirects_map(pairs):
            return {a: b for a, b in pairs}
        monkeypatch.setattr(wikipedia, "_wiki_api", fake)
        monkeypatch.setattr(wikipedia, "_sleep", lambda s: None)

    def test_two_spellings_that_redirect_together_are_the_same(self, monkeypatch):
        self._resolved(monkeypatch, {"Laskarina Bouboulina": 934118},
                       redirects=[("Bouboulina", "Laskarina Bouboulina")])
        r = wikipedia.resolve_titles(["Bouboulina", "Laskarina Bouboulina"])
        verdict, why = wikipedia.identity_verdict(
            "Laskarina Bouboulina", "Bouboulina", r)
        assert verdict == "same"
        assert "Laskarina Bouboulina" in why

    def test_separate_articles_are_different_subjects(self, monkeypatch):
        self._resolved(monkeypatch, {"Narmer Palette": 1, "Oxford Palette": 2},
                       redirects=[("Two Dog Palette", "Oxford Palette")])
        r = wikipedia.resolve_titles(["Narmer Palette", "Two Dog Palette"])
        assert wikipedia.identity_verdict("Narmer Palette", "Two Dog Palette", r)[0] == "wrong"

    def test_a_name_with_no_article_abstains(self, monkeypatch):
        """Egyptian throne names are not article titles. Abstaining is the
        measured-correct behaviour: text matching answered these and was often
        wrong."""
        self._resolved(monkeypatch, {"Ramesses II": 1, "Usermaatre Setepenre": None})
        r = wikipedia.resolve_titles(["Ramesses II", "Usermaatre Setepenre"])
        verdict, why = wikipedia.identity_verdict("Ramesses II", "Usermaatre Setepenre", r)
        assert verdict is None
        assert "no Wikipedia article" in why

    def test_a_disambiguation_page_abstains(self, monkeypatch):
        self._resolved(monkeypatch, {"Memphis, Egypt": 1, "Memphis": 2},
                       disambig={"Memphis"})
        r = wikipedia.resolve_titles(["Memphis, Egypt", "Memphis"])
        assert wikipedia.identity_verdict("Memphis, Egypt", "Memphis", r)[0] is None

    def test_a_failed_lookup_abstains_rather_than_raising(self, monkeypatch):
        def boom(params, attempts=4):
            raise RuntimeError("network is down")
        monkeypatch.setattr(wikipedia, "_wiki_api", boom)
        monkeypatch.setattr(wikipedia, "_sleep", lambda s: None)
        r = wikipedia.resolve_titles(["Anything"])
        assert r["Anything"] == (None, "", False)


# ---------------------------------------------------------------------------
# The rejected-sources memory and --only (v0.4.0)
# ---------------------------------------------------------------------------

HEADER = "atom\turl\treason\tdate\torigin\n"


def _key(atom_type, path):
    return sources.atom_key(atom_type, path)


def _rejects(tmp_path, *rows, name="rejected_sources.tsv"):
    path = tmp_path / name
    path.write_text(HEADER + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def never_fetch(monkeypatch):
    """`resolves` fails the test: a rejected URL must be dropped before any fetch."""
    fetched = []

    def resolves(url, timeout=15):
        fetched.append(url)
        if "Zela" in url:
            raise AssertionError(f"fetched a rejected URL: {url}")
        return True, "HTTP 200"

    monkeypatch.setattr(sources, "resolves", resolves)
    return fetched


class TestLoadingTheRejectedFile:
    def test_rows_are_keyed_by_atom_and_canonical_url(self, tmp_path):
        path = _rejects(tmp_path,
                        ("event:h0123456789abcdef", "https://en.wikipedia.org/wiki/Battle_of_Zela_%2847_BC%29",
                         "wrong battle", "2026-10-04", "lint review"))

        assert sources.load_rejected(path) == {
            "event:h0123456789abcdef": {"https://en.wikipedia.org/wiki/Battle_of_Zela_(47_BC)"}}

    def test_comments_and_blank_lines_are_allowed(self, tmp_path):
        path = tmp_path / "r.tsv"
        path.write_text("# rejects\n\n" + HEADER + "# one\n"
                        "event:h0a\thttps://x.org/a\twhy\t2026-10-04\tme\n\n", encoding="utf-8")

        assert sources.load_rejected(path) == {"event:h0a": {"https://x.org/a"}}

    def test_no_path_or_no_file_is_no_rejects(self, tmp_path):
        assert sources.load_rejected(None) == {}
        assert sources.load_rejected(tmp_path / "absent.tsv") == {}

    @pytest.mark.parametrize("line", [
        "event:h0a\thttps://x.org/a\twhy\t2026-10-04",            # a field short
        "event:h0a https://x.org/a why 2026-10-04 me",            # spaces, not tabs
        "Battle of Zela\thttps://x.org/a\twhy\t2026-10-04\tme",   # a name, not kind:hID
        "event:h0a\tx.org/a\twhy\t2026-10-04\tme",                # not an http URL
    ])
    def test_a_malformed_line_raises_rather_than_being_skipped(self, tmp_path, line):
        path = tmp_path / "r.tsv"
        path.write_text(HEADER + line + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"r\.tsv:2"):
            sources.load_rejected(path)

    def test_a_missing_header_raises(self, tmp_path):
        path = tmp_path / "r.tsv"
        path.write_text("event:h0a\thttps://x.org/a\twhy\t2026-10-04\tme\n", encoding="utf-8")

        with pytest.raises(ValueError, match="header"):
            sources.load_rejected(path)


class TestARejectedUrlIsNeverOfferedAgain:
    ZELA = "https://en.wikipedia.org/wiki/Battle_of_Zela_(47_BC)"
    ZELA_ENCODED = "https://en.wikipedia.org/wiki/Battle_of_Zela_%2847_BC%29"

    def test_another_spelling_is_neither_offered_nor_fetched(self, workspace, tmp_path, never_fetch):
        path = _atom("event", "Battle of Zela")
        rejected = sources.load_rejected(_rejects(
            tmp_path, (_key("event", path), self.ZELA_ENCODED, "the other Zela", "2026-10-04", "review")))

        p = sources.propose("event", path, _ask(self.ZELA, "https://good.org/zela-site"), rejected=rejected)

        assert p.offered == ["https://good.org/zela-site"]
        assert p.rejected_before == [self.ZELA]
        assert never_fetch == ["https://good.org/zela-site"]

    def test_a_reject_for_one_atom_does_not_block_another(self, workspace, tmp_path, monkeypatch):
        monkeypatch.setattr(sources, "resolves", lambda u, timeout=15: (True, "HTTP 200"))
        a = _atom("event", "Battle of Zela")
        b = _atom("event", "Zela campaign")
        rejected = {_key("event", a): {sources.canonical_url(self.ZELA)}}

        p = sources.propose("event", b, _ask(self.ZELA), rejected=rejected)

        assert p.verified == [self.ZELA] and p.rejected_before == []

    def test_no_rejects_means_everything_is_offered(self, workspace, all_resolve):
        path = _atom("event", "Battle of Zela")

        p = sources.propose("event", path, _ask(self.ZELA))

        assert p.verified == [self.ZELA] and p.rejected_before == []

    def test_apply_re_checks_the_packs_file(self, workspace, tmp_path, all_resolve, pack_overrides):
        """A proposal built without the rejects (or before one was recorded)
        still never writes a rejected URL."""
        path = _atom("event", "Battle of Zela")
        p = sources.propose("event", path, _ask(self.ZELA))
        pack_overrides(rejected_sources_path=_rejects(
            tmp_path, (_key("event", path), self.ZELA_ENCODED, "wrong", "2026-10-04", "review")))

        assert sources.apply(p, dry_run=False) is False
        assert "sources" not in json.loads(path.read_text())["name"]

    def test_run_reads_the_packs_file_and_counts_the_skips(self, workspace, tmp_path, never_fetch,
                                                           pack_overrides):
        path = _atom("event", "Battle of Zela")
        pack_overrides(rejected_sources_path=_rejects(
            tmp_path, (_key("event", path), self.ZELA_ENCODED, "wrong", "2026-10-04", "review")))

        report = sources.run(["event"], _ask(self.ZELA), write=True)

        assert report.examined == 1 and report.rejected_before == 1
        assert report.updated == 0 and never_fetch == []


class TestOnly:
    def test_only_limits_the_atoms(self, workspace, all_resolve):
        a = _atom("event", "Battle of Zela")
        _atom("event", "Battle of Pharsalus")
        _atom("person", "Pharnaces II")

        report = sources.run(["event", "person"], _ask("https://good.org/x"),
                             only={_key("event", a), "person:h0000000000000000"})

        assert [p.subject for p in report.proposals] == ["Battle of Zela"]
        assert report.examined == 1
        assert report.only_unmatched == ["person:h0000000000000000"]

    def test_the_ids_file_is_one_key_per_line(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("# a batch\nevent:h0a\n\nperson:h1b  # trailing note\n", encoding="utf-8")

        assert sources.load_only(path) == {"event:h0a", "person:h1b"}

    def test_a_line_that_is_not_a_key_raises(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("Battle of Zela\n", encoding="utf-8")

        with pytest.raises(ValueError, match="kind:hID"):
            sources.load_only(path)

    def test_the_cli_takes_only_and_prints_the_skipped_count(self, workspace, tmp_path, capsys,
                                                             monkeypatch, pack_overrides):
        from universe_builder import cli

        monkeypatch.setattr(sources, "resolves", lambda u, timeout=15: (True, "HTTP 200"))
        zela = _atom("event", "Battle of Zela")
        _atom("event", "Battle of Pharsalus")
        ids = tmp_path / "ids.txt"
        ids.write_text(_key("event", zela) + "\n", encoding="utf-8")
        pack_overrides(
            url_resolvers={"fake": lambda per_atom=1: _ask(TestARejectedUrlIsNeverOfferedAgain.ZELA,
                                                           "https://good.org/zela")},
            rejected_sources_path=_rejects(tmp_path, (
                _key("event", zela), TestARejectedUrlIsNeverOfferedAgain.ZELA_ENCODED,
                "wrong", "2026-10-04", "review")))

        rc = cli.main(["sources", "--via", "fake", "--atom", "event", "--only", str(ids)])
        out = capsys.readouterr().out

        assert rc == 1
        assert "Battle of Pharsalus" not in out
        assert "-- rejected before: " + TestARejectedUrlIsNeverOfferedAgain.ZELA in out
        assert "1 previously rejected URL(s) skipped unfetched (1 reject(s) on file)." in out
        assert "1 atom(s) examined;" in out

    def test_a_malformed_rejects_file_stops_the_cli(self, workspace, tmp_path, capsys, pack_overrides):
        from universe_builder import cli

        bad = tmp_path / "r.tsv"
        bad.write_text(HEADER + "nonsense\n", encoding="utf-8")
        pack_overrides(url_resolvers={"fake": lambda per_atom=1: _ask()}, rejected_sources_path=bad)

        assert cli.main(["sources", "--via", "fake"]) == 2
