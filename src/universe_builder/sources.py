"""Append verified, resolvable source URLs to stored atoms.

**Additive by construction.** Nothing is removed and no existing field is
rewritten: URLs are appended to `name.sources`. That is the whole reason this
pass is safe to run broadly where a searched re-verify is not — the worst case
is an atom with one more citation than it needed, not an atom whose prose was
replaced by something worse.

**Every URL is fetched before it is written.** A model asked for sources will
produce plausible URLs whether or not it visited them; this session has already
seen a fabricated UNODC link with non-breaking hyphens in it, which cannot
exist as typed. So a URL the model offers is a *candidate*, and the HTTP
response is what promotes it. That check costs nothing and is the only thing
standing between "cited" and "citable".

**Why lean on Wikipedia.** Stable URLs that rarely 404, and — the part that
matters — an article is a *pointer to* the scholarship rather than a claim to
be it. A Britannica or museum page is better when one exists for the subject;
a blog is worse than nothing, because it launders a guess into a citation.

The corpus's existing `sources` are recalled, not fetched: no generation call
has ever had search enabled. This does not fix that for the prose already
written — it adds something a reader can actually follow.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (HTTPHandler, HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request,
                            build_opener)

from pydantic import BaseModel, Field

from universe_builder import __version__, store
from universe_builder.registry import active_pack

log = logging.getLogger(__name__)


#: Hosts whose pages are worth citing, best first, are the domain's opinion —
#: `DomainPack.preferred_hosts`. Not a whitelist: anything that resolves is
#: kept, but the prompt asks for these and ties break here.
def _preferred_hosts():
    return active_pack().preferred_hosts


#: A URL the model typed rather than visited. Typographic dashes and quotes
#: cannot appear in a real URL, and their presence is a reliable tell — this
#: corpus produced exactly such a link during the fact-check audit.
_TYPOGRAPHIC = re.compile(r"[‐-―‘’“”\s]")

PROMPT = """Find authoritative web pages about this subject that a reader could
open to verify an encyclopaedia entry.

Subject: {subject}
Type: {atom_type}

{preference}
Do not offer blogs, travel sites, listicles, or search-result pages.

Return ONLY this JSON:
{{"urls": ["<url>", "<url>", "<url>"]}}

Rules:
- Only URLs you actually opened in a search result. Never construct or guess one.
- At most 3, and fewer is fine. An empty list is a valid answer.
- Each must be about THIS subject, not a category or a list that mentions it.
"""


@dataclass
class Proposal:
    """What one atom would gain."""

    atom_type: str
    file: str
    subject: str
    offered: List[str] = field(default_factory=list)
    verified: List[str] = field(default_factory=list)
    rejected: List[tuple] = field(default_factory=list)
    #: `kind:hID`, the key the rejected-sources file is read by ("" for a
    #: file whose name carries no content id).
    atom: str = ""
    #: What the resolver offered that a person had already rejected for this
    #: atom (`load_rejected`): dropped before any fetch, never in `offered`.
    rejected_before: List[str] = field(default_factory=list)

    @property
    def additions(self) -> List[str]:
        return self.verified


#: Hosts that never carry a citation about a historical subject: search
#: engines, video, marketplaces, social. A model offered a Commons category and
#: a raw Google query unprompted; nothing stops it offering these.
_NOT_SOURCE_HOSTS = (
    "google.", "bing.com", "duckduckgo.com", "search.yahoo",
    "webcache.googleusercontent", "youtube.com", "youtu.be", "vimeo.com",
    "amazon.", "ebay.", "etsy.com", "pinterest.", "facebook.com",
    "instagram.com", "twitter.com", "x.com/", "reddit.com", "quora.com",
    "tiktok.com", "medium.com", "blogspot.", "wordpress.com",
)

#: MediaWiki namespaces that are machinery or discussion rather than an
#: article. `File:` is deliberately **not** here — a Commons media page for the
#: object itself is a perfectly good citation, and an earlier blanket rule
#: threw those away along with the categories.
_WIKI_NAMESPACES = re.compile(
    r"/wiki/(Special|Category|Help|Portal|Template|Talk|User|Wikipedia"
    r"|Draft|Module|MediaWiki|Book|Course):", re.I)

#: URL shapes that resolve perfectly and cite nothing.
_NOT_A_SOURCE = (
    _WIKI_NAMESPACES,
    # `/search` only when it *is* the query, not when it is a path segment
    # before an object id: the Met serves individual objects at
    # `/art/collection/search/100004`, a page about one thing.
    re.compile(r"/(search|results|find)/?(\?|$)", re.I),
    # `q`/`query`/`search` only. `s=` was in here and is a common benign
    # parameter — it rejected `example.ac.uk/paper?s=2&id=44`, a real article.
    re.compile(r"[?&](q|query|search)=", re.I),
    re.compile(r"\(disambiguation\)", re.I),
    # A listing is not a source about any one of the things it lists, and the
    # prompt already asks for none.
    re.compile(r"/wiki/(List|Index|Outline|Timeline)_of_", re.I),
)


def cites_something(url: str) -> tuple:
    """`(ok, reason)` — whether a URL is a page about a subject at all."""
    lowered = url.lower()
    if any(host in lowered for host in _NOT_SOURCE_HOSTS):
        return False, "not a citable source"
    # A bare domain is a front page, not a page about the subject.
    after_host = lowered.split("//", 1)[-1]
    if "/" not in after_host or after_host.split("/", 1)[1].strip() in ("", "#"):
        return False, "a site homepage, not a page about the subject"
    for pattern in _NOT_A_SOURCE:
        if pattern.search(url):
            return False, "a search, category or listing page"
    return True, ""


#: Honest identification, with a contact, as Wikimedia's policy asks of any script.
_USER_AGENT = f"universe_builder/{__version__} (+https://github.com/LeooDong/universe_builder)"


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _public_ip(ip) -> bool:
    """Whether an address is on the public internet, judging an IPv6 address
    that carries an IPv4 one (mapped, 6to4, Teredo, NAT64) by that IPv4 one
    too. `is_global` alone counts multicast, and on 3.10-3.12 6to4, as public."""
    if ip.version == 6 and ip.ipv4_mapped:
        return _public_ip(ip.ipv4_mapped)
    if ip.version == 6 and ip in _NAT64:   # inside ::/8, so `is_reserved`; an IPv6-only network's route to IPv4
        return _public_ip(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    if not ip.is_global or ip.is_multicast or ip.is_reserved:
        return False
    if ip.version == 6:
        return all(_public_ip(e) for e in (ip.sixtofour, ip.teredo[1] if ip.teredo else None) if e)
    return True


def _non_public_host(url: str) -> bool:
    """True for a host no citation lives on: localhost, a `.local` or
    `.internal` name, or an IP literal outside the public internet. The URL is
    model-written, and `resolves` opens it, so it must not reach this machine
    or its network."""
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return True
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        return not _public_ip(ipaddress.ip_address(host))
    except ValueError:
        return False


class _NonPublicPeer(OSError):
    """A host that resolves, or redirects, outside the public internet."""


def _connect_public(address, *args, **kwargs):
    """`socket.create_connection`, but only to a public address.

    `_non_public_host` reads the URL as written, which a name pointing at
    127.0.0.1 or a numeric form like `2130706433` gets past. So the name is
    resolved here, refused if *any* address it gives is non-public, and the
    connection made to the checked address itself — a second lookup could
    answer differently. TLS still verifies against the URL's host name.
    """
    host, port = address
    addresses = []
    for *_, sockaddr in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        ip = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        if not _public_ip(ip):
            raise _NonPublicPeer(f"{host} resolves to {ip}, not a public address")
        if str(ip) not in addresses:
            addresses.append(str(ip))
    if not addresses:
        raise _NonPublicPeer(f"{host} resolves to nothing")
    # In order, as `create_connection` would: an unroutable IPv6 answer must
    # not fail a host whose IPv4 address works.
    for address in addresses:
        try:
            return socket.create_connection((address, port), *args, **kwargs)
        except OSError as e:
            error = e
    raise error


def _public_connection(http_class):
    class _Public(http_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._create_connection = _connect_public
    return _Public


class _PublicHTTPHandler(HTTPHandler):
    def do_open(self, http_class, req, **kwargs):
        return super().do_open(_public_connection(http_class), req, **kwargs)


class _PublicHTTPSHandler(HTTPSHandler):
    def do_open(self, http_class, req, **kwargs):
        return super().do_open(_public_connection(http_class), req, **kwargs)


class _PublicRedirects(HTTPRedirectHandler):
    """Follow a redirect only over http(s) to a public host
    (`_non_public_host`); the connection guard checks what the new host
    resolves to. urllib would follow one to `ftp:`, whose handler resolves and
    connects without that guard."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme not in ("http", "https") or _non_public_host(newurl):
            raise URLError(_NonPublicPeer(f"redirect to {newurl}, not a public host"))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


#: No proxies: a proxy resolves the host itself, out of `_connect_public`'s sight.
_OPENER = build_opener(ProxyHandler({}), _PublicHTTPHandler, _PublicHTTPSHandler, _PublicRedirects)


def urlopen(request, timeout):
    """`urllib`'s `urlopen` through `_OPENER`; a module name the tests replace."""
    return _OPENER.open(request, timeout=timeout)


def resolves(url: str, timeout: int = 15) -> tuple:
    """`(ok, reason)` for one candidate URL. Never raises.

    A 403 or 429 is *not* a failure: institutional sites routinely block
    scripted requests, and the Met and the University of Michigan both did so
    during the fact-check audit while serving the same pages fine to a browser.
    Treating those as fabricated would discard exactly the citations most worth
    having.
    """
    if not url.startswith("https://") and not url.startswith("http://"):
        return False, "not an http url"
    if _TYPOGRAPHIC.search(url):
        # An en-dash or curly quote means the model typed it rather than
        # visiting it.
        return False, "contains a typographic character"
    usable, why = cites_something(url)
    if not usable:
        return False, why
    if _non_public_host(url):
        return False, "not a public host"
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return (response.status < 400), f"HTTP {response.status}"
    except HTTPError as e:
        if e.code in (401, 403, 405, 429):
            return True, f"HTTP {e.code} (blocks scripts, page exists)"
        return False, f"HTTP {e.code}"
    except (URLError, OSError, ValueError) as e:
        if isinstance(e, _NonPublicPeer) or isinstance(getattr(e, "reason", None), _NonPublicPeer):
            return False, "not a public host"
        return False, type(e).__name__


def _preference_line() -> str:
    """The search prompt's "prefer these" line: the pack's hosts, best first,
    then the generic categories that hold for any domain. The categories were
    once the whole line and were written for history (a museum
    that holds the object, an excavation project); they are kept, in
    domain-free words, because a bare host list says nothing about what to do
    beyond it."""
    generic = ("prefer primary, institutional and peer-reviewed pages: an "
               "encyclopaedia, the body that holds or regulates the subject, a "
               "university page, a DOI.")
    hosts = _preferred_hosts()
    if not hosts:
        return "P" + generic[1:]
    return "Prefer, in this order: " + ", ".join(hosts) + ". Beyond those, " + generic


def _rank(url: str) -> int:
    hosts = _preferred_hosts()
    for position, host in enumerate(hosts):
        if host in url:
            return position
    return len(hosts)


#: A URL inside prose. Apostrophes are **legal** in URLs and excluding them
#: truncated `.../wiki/Euclid's_theorem` to `.../wiki/Euclid`, so the
#: already-cited check compared a fragment and let the same page in twice.
#: Trailing sentence punctuation is stripped instead, which is the real
#: ambiguity.
_URL_IN_TEXT = re.compile(r"https?://[^\s\"<>\]]+")
_TRAILING = ".,;:!?'\""


def _trim(url: str) -> str:
    """Strip what prose put after a URL: trailing punctuation, and a `)`
    closing a parenthesis the URL never opened. A title's own `)` stays:
    stripping every `)` made `Battle_of_Zela_(47_BC)` unmatchable."""
    while True:
        url = url.rstrip(_TRAILING)
        if url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
            continue
        return url


def canonical_url(url: str) -> str:
    """One spelling per page, for comparison only — never for writing.

    `Euclid%27s_theorem` and `Euclid's_theorem` are the same Wikipedia article,
    and comparing them raw let the resolver add a page the atom already cited.
    Percent-encoding is decoded, the host lowercased, and a trailing slash
    dropped.
    """
    from urllib.parse import unquote

    text = unquote(url or "").rstrip("/")
    if "://" in text:
        scheme, rest = text.split("://", 1)
        host, _, path = rest.partition("/")
        text = f"{scheme.lower()}://{host.lower()}" + (f"/{path}" if path else "")
    return text


def existing_urls(payload) -> set:
    """Every URL already cited anywhere in the atom, canonicalised."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            for raw in _URL_IN_TEXT.findall(node):
                found.add(canonical_url(_trim(raw)))

    walk(payload)
    return found


def context_terms(payload, atom_type: str) -> List[str]:
    """Words that say *which* subject this is — country, then region.

    An ancient corpus is full of names that are ambiguous on their own:
    `Memphis`, `Thebes`, `Apollonia`. Without context the resolver can only
    decline them, which skips exactly the atoms most worth citing. With
    `country='Egypt'` beside the name, `Memphis, Egypt` becomes admissible and
    `Memphis, Tennessee` stays refused — the result may use words from the
    subject or from its known context, and nothing else.
    """
    terms: List[str] = []
    # Where each atom type keeps the place that identifies it is the pack's
    # `coordinate_paths` — the same fields lint compares. First hit wins.
    for dotted in active_pack().coordinate_paths.get(atom_type, ()):
        node = payload
        for key in dotted.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            for context_field in ("country", "state"):
                value = node.get(context_field)
                if isinstance(value, str) and value.strip():
                    terms.append(value.strip())
            if terms:
                break
    return terms


#: The rejected-sources file's header (`DomainPack.rejected_sources_path`).
REJECTED_HEADER = ("atom", "url", "reason", "date", "origin")
_ATOM_KEY = re.compile(r"^[^\s:]+:h[0-9a-f]+$")


def atom_key(atom_type: str, path) -> str:
    """`kind:hID` for a stored atom — the journal's work item, the index's id —
    or "" when the filename carries no content id."""
    from universe_builder import journal

    return journal.work_item_from_path(atom_type, path) or ""


def load_rejected(path) -> Dict[str, Set[str]]:
    """`{atom_key: {canonical_url}}`: the URLs a person has rejected per atom.

    A TSV whose first line (after `#` comments and blank lines) is the header
    `atom url reason date origin`; `atom` is `kind:hID`. Each URL is compared
    by `canonical_url`, the spelling `sources` compares every URL by, so a
    reject written `%28…%29` also refuses the `(…)` spelling. No path, or no
    file at it, is no rejects. **A malformed line raises** `ValueError`
    naming it: a reject silently skipped is a rejected URL offered again.
    """
    if path is None or not Path(path).is_file():
        return {}
    found: Dict[str, Set[str]] = {}
    header_seen = False
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cells = line.split("\t")
        if not header_seen:
            if tuple(c.strip() for c in cells) != REJECTED_HEADER:
                raise ValueError(f"{path}:{number}: expected the header "
                                 f"{chr(9).join(REJECTED_HEADER)!r}, got {line!r}")
            header_seen = True
            continue
        if len(cells) != len(REJECTED_HEADER):
            raise ValueError(f"{path}:{number}: {len(cells)} tab-separated field(s), "
                             f"expected {len(REJECTED_HEADER)}: {line!r}")
        atom, url = cells[0].strip(), cells[1].strip()
        if not _ATOM_KEY.match(atom):
            raise ValueError(f"{path}:{number}: {atom!r} is not an atom key (kind:hID)")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{path}:{number}: {url!r} is not an http(s) URL")
        found.setdefault(atom, set()).add(canonical_url(url))
    return found


def load_only(path) -> Set[str]:
    """The atom keys an `--only` file names: one `kind:hID` per line, `#`
    comments and blank lines allowed. Anything else raises `ValueError`."""
    keys: Set[str] = set()
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not _ATOM_KEY.match(line):
            raise ValueError(f"{path}:{number}: {line!r} is not an atom key (kind:hID)")
        keys.add(line)
    return keys


def _pack_rejected() -> Dict[str, Set[str]]:
    return load_rejected(active_pack().rejected_sources_path)


def has_link(payload) -> bool:
    """Whether an atom already cites anything a reader could open."""
    return bool(existing_urls(payload)) if isinstance(payload, dict) else False


def propose(atom_type: str, path, ask, rejected: Optional[Dict[str, Set[str]]] = None) -> Proposal:
    """Ask for URLs, verify each, and report what would be added. Reads only.

    `ask(subject, atom_type) -> list[str]` is injected so the whole of this is
    testable without a network or a model. `rejected` is `load_rejected`'s
    map (None = none): a URL rejected before for *this* atom is dropped as
    soon as it is offered, before anything fetches it, and kept in
    `proposal.rejected_before`.
    """
    payload = store.read_json(path)
    subject = ((payload.get("name") or {}).get("common")
               if isinstance(payload, dict) else None) or path.name
    proposal = Proposal(atom_type=atom_type, file=path.name, subject=subject,
                        atom=atom_key(atom_type, path))
    if not isinstance(payload, dict):
        return proposal

    already = existing_urls(payload)
    hints = context_terms(payload, atom_type)
    try:
        try:
            offered = list(ask(subject, atom_type, hints))
        except TypeError:
            # A resolver that predates context still works.
            offered = list(ask(subject, atom_type))
    except Exception as e:  # noqa: BLE001 - one atom's failure costs one atom
        log.error("%s: could not ask for sources: %s", subject, e)
        return proposal
    blocked = (rejected or {}).get(proposal.atom, set()) if proposal.atom else set()
    for url in offered:
        if isinstance(url, str) and canonical_url(url) in blocked:
            proposal.rejected_before.append(url)
        else:
            proposal.offered.append(url)
    proposal.offered = proposal.offered[:5]

    # `already` grows as candidates are accepted, so the batch is deduplicated
    # against *itself* as well as against the atom. `dict.fromkeys` removes
    # only byte-identical repeats, and a resolver can easily offer one page
    # under two spellings — `Euclid's_theorem` beside `Euclid%27s_theorem`, or
    # a trailing slash — which is the same page written twice.
    seen = set(already)
    for url in sorted(dict.fromkeys(proposal.offered), key=_rank):
        if canonical_url(url) in seen:
            proposal.rejected.append((url, "already cited"))
            continue
        ok, reason = resolves(url)
        if ok:
            seen.add(canonical_url(url))
            proposal.verified.append(url)
        else:
            proposal.rejected.append((url, reason))
    return proposal


def apply(proposal: Proposal, *, dry_run: bool = True,
          rejected: Optional[Dict[str, Set[str]]] = None) -> bool:
    """Append the verified URLs to `name.sources`. Returns whether it wrote.

    Re-validates before writing, and refuses rather than persisting an atom
    that would no longer load — the same rule `lint.apply_review` follows.
    Re-checks the rejected-sources file too (`rejected`, else the pack's,
    read now): a URL rejected for this atom is never written, whoever built
    the proposal.
    """
    if dry_run or not proposal.additions:
        return False
    if rejected is None:
        rejected = _pack_rejected()
    blocked = rejected.get(proposal.atom, set()) if proposal.atom else set()
    path = store.dir_for(proposal.atom_type) / proposal.file
    payload = store.read_atom(path, proposal.atom_type)
    name = payload.setdefault("name", {})
    sources = list(name.get("sources") or [])
    # Canonical comparison here too, not `u not in sources`. `propose` reads
    # the atom and `apply` re-reads it, so anything written in between must
    # still be matched across encodings rather than byte for byte.
    held = {canonical_url(u) for u in sources if isinstance(u, str)}
    fresh = []
    for url in proposal.additions:
        if canonical_url(url) in held:
            continue
        if canonical_url(url) in blocked:
            log.warning("%s: %s was rejected before; not written", proposal.file, url)
            continue
        held.add(canonical_url(url))
        fresh.append(url)
    if not fresh:
        return False
    name["sources"] = sources + fresh
    try:
        _validate(proposal.atom_type, payload)
    except Exception as e:  # noqa: BLE001
        log.error("%s would not validate after adding sources; left alone: %s",
                  proposal.file, str(e)[:120])
        return False
    store.write_json(path, payload)
    return True


@dataclass
class Report:
    """What one `run` did: atoms examined, atoms written and URLs added, and
    every proposal that offered anything."""

    examined: int = 0
    updated: int = 0
    urls: int = 0
    #: Offered URLs dropped unfetched because a person rejected them before.
    rejected_before: int = 0
    #: `--only` keys that matched no stored atom of the types run.
    only_unmatched: List[str] = field(default_factory=list)
    #: `(matched, asked)` when the resolver prefetched titles, else None.
    prefetched: Optional[tuple] = None
    proposals: List[Proposal] = field(default_factory=list)


def _atoms(atom_types, only: Optional[Set[str]], seen: Optional[Set[str]] = None):
    """`(atom_type, path)` for every stored atom of `atom_types`, or only those
    whose `kind:hID` is in `only`; the keys met go into `seen`."""
    for atom_type in atom_types:
        if only is not None and not any(k.split(":", 1)[0] == atom_type for k in only):
            continue
        for path in store.iter_atoms(atom_type):
            if only is not None:
                key = atom_key(atom_type, path)
                if key not in only:
                    continue
                if seen is not None:
                    seen.add(key)
            yield atom_type, path


def run(atom_types, ask, *, limit: Optional[int] = None, missing: bool = False,
        write: bool = False, on_prefetch=None, on_proposal=None,
        rejected: Optional[Dict[str, Set[str]]] = None,
        only: Optional[Iterable[str]] = None) -> Report:
    """Propose (and with `write`, append) sources for every atom of `atom_types`.

    The loop `universe_builder sources` runs, out of `cli` since v0.2.0.
    `limit` counts atoms across every type (a falsy limit
    is none); `missing` skips atoms that already cite a URL. A resolver with a
    `prefetch` method is handed every subject first — one request per atom
    earned HTTP 429 from Wikipedia after a few dozen of them — and `on_prefetch(matched,
    asked)` hears how that went. `on_proposal(proposal)` hears each proposal
    that offered anything, before it is applied, so a caller can print as it
    goes.

    `rejected` is `load_rejected`'s map; None reads the pack's
    `rejected_sources_path` once, here, for the whole run (v0.4.0). `only`
    restricts the run to those `kind:hID` atoms (`sources --only FILE`); a key
    that matches nothing lands in `report.only_unmatched`.
    """
    report = Report()
    if rejected is None:
        rejected = _pack_rejected()
    only = set(only) if only is not None else None
    if hasattr(ask, "prefetch"):
        # Resolve every title in batches of 50 before the per-atom loop.
        subjects = []
        for atom_type, path in _atoms(atom_types, only):
            if limit and len(subjects) >= limit:
                break
            payload = store.read_json(path)
            if missing and has_link(payload):
                continue
            if isinstance(payload, dict):
                subjects.append(((payload.get("name") or {}).get("common")
                                 or path.name))
        matched = ask.prefetch(subjects)
        report.prefetched = (matched, len(subjects))
        if on_prefetch is not None:
            on_prefetch(matched, len(subjects))
    met: Set[str] = set()
    for atom_type, path in _atoms(atom_types, only, met):
        if limit and report.examined >= limit:
            break
        if missing and has_link(store.read_json(path)):
            continue
        report.examined += 1
        proposal = propose(atom_type, path, ask, rejected=rejected)
        report.rejected_before += len(proposal.rejected_before)
        if not proposal.offered:
            continue
        report.proposals.append(proposal)
        if on_proposal is not None:
            on_proposal(proposal)
        if apply(proposal, dry_run=not write, rejected=rejected):
            report.updated += 1
            report.urls += len(proposal.additions)
    if only is not None and not limit:
        report.only_unmatched = sorted(only - met)
    return report


def _validate(atom_type: str, payload) -> None:
    spec = next((s for s in active_pack().all_specs() if s.name == atom_type), None)
    if spec is None:
        return
    for predicate, model in spec.stages or ():
        if predicate(payload):
            model.model_validate(payload)
            return
    spec.model.model_validate(payload)


class SourceList(BaseModel):
    """What the search model returns: candidate URLs, each verified before use."""

    urls: List[str] = Field(default_factory=list)


#: The stable half of the search request; the subject goes in the user turn.
SEARCH_SYSTEM = "You find web pages a reference work can cite for one subject."
#: A search call is slow, so its per-attempt timeout is longer than a run's.
SEARCH_TIMEOUT_SECONDS = 180


def searcher(model: str):
    """An `ask` backed by the `openai-search` provider (Responses + `web_search`).

    Through `providers.run.complete`, like every other paid call, so the
    run's retry policy applies, the reply is schema-validated, and the spend is
    journalled for `costs` (`journal.record_call`, into the run `cmd_sources`
    opens). It used to call the SDK client directly, and every search went
    unrecorded. `model` is a model id; `cmd_sources`
    resolves a tier name against the `openai-search` table first.
    """
    from universe_builder import journal
    from universe_builder.providers import run as provider_run
    from universe_builder.providers.types import LLMRequest

    def ask(subject: str, atom_type: str, hints=None) -> List[str]:
        request = LLMRequest(
            model=model, schema_model=SourceList, system=SEARCH_SYSTEM,
            user=PROMPT.format(subject=subject, atom_type=atom_type,
                               preference=_preference_line()),
            max_tokens=4096, timeout=SEARCH_TIMEOUT_SECONDS, custom_id=subject,
            metadata={"operation": "sources-search", "atom_type": atom_type, "subject": subject},
        )
        response = provider_run.complete(request, provider="openai-search")
        journal.record_call(request, response, operation="sources-search",
                            atom_type=atom_type, subject=subject)
        if not isinstance(response.parsed, SourceList):
            log.warning("%s: no usable source list (%s)", subject, response.error)
            return []
        return list(response.parsed.urls)

    return ask
