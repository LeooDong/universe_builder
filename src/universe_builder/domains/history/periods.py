"""City-period consistency: lint, mechanical adjudication, and the split loop.

The one-period-one-government rule lives in the schema descriptions, but a
schema description is a soft instruction — a whole-file repair over many cities
split most periods and left Alexandria's untouched, because whether a
whole-file verify acts on a rule depends on the model noticing it, once per
city. Uniformity comes from taking the *decision* away from the model:

* **Label disagreements are adjudicated mechanically.** Identical period
  signatures across cities must agree on `typeOfGovernment`; where they
  disagree, the majority answer is applied to the minority *by data edit*,
  no model in the write path. One answer exists per signature, so drift is
  impossible. Ties are reported, never guessed.
* **Over-broad periods are found by a deterministic checker.** The regime
  table (`domains/history/regime_changes.txt`) lists enum-level government
  changes per civilisation; a period whose span strictly contains one is
  flagged. The model's only job is the history: a flagged period goes out as
  one `PeriodSplit` call and comes back as consecutive sub-periods, spliced
  in mechanically. The loop is done when the checker finds nothing — a
  machine verdict, not a model's mood.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from universe_builder import store, topics
from universe_builder.atoms import resolve_stored_class
from universe_builder.domains.history.models.city import CityExpanded, PeriodSplit, government_enum

log = logging.getLogger(__name__)

_CHANGE_LINE = re.compile(
    r"^(?P<year>-?\d+)\s*:\s*(?P<before>[a-z -]+?)\s*->\s*(?P<after>[a-z -]+?)"
    r"(?:\s*@(?P<country>[^#!]+?))?(?P<exempt>(?:\s*![^#!]+?)*)$"
)


@dataclass(frozen=True)
class RegimeChange:
    """One enum-level change of government, and who it applies to.

    `scope` is the civilisation whose History lists the city — the seed, not
    the geography. That is right for an ancient line: the Roman conquest of
    146 BC changed the government of Greek poleis wherever they stood, Massalia
    and Cyrene included.

    It is wrong for a modern one, and measurably so. A large share of the
    lint's flags **applied one country's twentieth-century
    politics to another country's city** — `'Republic of Türkiye'` flagged by
    the Greek colonels' junta of 1967, `'Kingdom of Italy'` by the Greek
    restoration of 1935. A Greek-seeded corpus reaches Marseille, Alexandria
    and Chalcedon, and those cities' modern governments have nothing to do with
    Athens.

    `country` is the fix, and it is deliberately opt-in per line rather than a
    rule in code: the table already exists to hold exactly the judgements a
    model cannot be trusted with, and whether a regime change is civilisational
    or national is such a judgement. A line with no `@country` applies to the
    whole scope, as every line did before, so the ancient entries need no edit.
    """

    scope: str          # civilisation, as written in the table
    year: int           # signed: negative is BCE
    before: str
    after: str
    #: Restrict to cities whose `location.country` matches, case-insensitively.
    #: `None` means the line applies to the whole civilisation scope.
    country: Optional[str] = None
    #: Polities whose periods the line skips (`!Kingdom of Italy`), matched
    #: against the period's own `polity`.
    #: `!city:NAME` instead exempts a named city's whole span from that change.
    exempt: Tuple[str, ...] = ()

    def applies_to(self, city_country: Optional[str]) -> bool:
        """Whether this line should be checked against a city in `city_country`.

        An unqualified line always applies. A qualified one needs the city to
        say where it is — and a city with no country recorded is *excluded*
        rather than included, because the alternative reintroduces exactly the
        false positives this field exists to remove.
        """
        if self.country is None:
            return True
        if not city_country:
            return False
        return _fold_country(city_country) == _fold_country(self.country)

    def exempts(self, period_polity: Optional[str], city_name: Optional[str] = None) -> bool:
        """Whether the period's polity or the named city is exempt."""
        exempt_polities = {e.casefold() for e in self.exempt if not e.casefold().startswith("city:")}
        exempt_cities = {e.split(":", 1)[1].strip().casefold() for e in self.exempt
                         if e.casefold().startswith("city:")}
        return (bool(period_polity) and period_polity.strip().casefold() in exempt_polities
                or bool(city_name) and city_name.strip().casefold() in exempt_cities)


def _fold_country(name: str) -> str:
    """Compare country names loosely enough to survive the corpus's spellings.

    `Turkey` and `Türkiye` both appear in the stored cities — 40 atoms under
    one and 3 under the other — so an exact match would silently exempt three
    cities from a table line that should cover them.
    """
    folded = (name or "").strip().casefold()
    for source, target in (("ü", "u"), ("ı", "i"), ("İ", "i"), ("ö", "o"), ("ç", "c")):
        folded = folded.replace(source, target)
    return _COUNTRY_ALIASES.get(folded, folded)


#: Spellings the corpus uses for one country. Small and hand-kept, like the
#: regime table itself; extend it when a new spelling actually shows up.
_COUNTRY_ALIASES = {
    "turkiye": "turkey",
    "republic of turkiye": "turkey",
    "republic of turkey": "turkey",
    "hellenic republic": "greece",
    "arab republic of egypt": "egypt",
    "italian republic": "italy",
}


#: The regime table ships beside this module (a history-domain resource), read
#: relative to it rather than from a core `settings` path.
REGIME_CHANGES = Path(__file__).resolve().parent / "regime_changes.txt"


def load_regime_changes(path=None) -> tuple:
    """Parse the regime table. A missing file means no lint, not an error."""
    if path is None:
        path = REGIME_CHANGES
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeDecodeError) as e:
        log.error("Cannot read the regime table %s (%s); the lint runs empty", path, e)
        return ()

    changes: list = []
    scope: Optional[str] = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            scope = line[1:-1].strip()
            continue
        match = _CHANGE_LINE.match(line)
        if not match or scope is None:
            log.warning("%s:%d is not a change line under a scope; skipped",
                        Path(path).name, number)
            continue
        before, after = match.group("before"), match.group("after")
        for value in (before, after):
            if value not in government_enum:
                log.warning("%s:%d names %r, which is not in government_enum; line skipped",
                            Path(path).name, number, value)
                break
        else:
            if before == after:
                log.warning("%s:%d changes nothing (%s -> %s); line skipped",
                            Path(path).name, number, before, after)
                continue
            country = (match.group("country") or "").strip() or None
            exempt = tuple(e.strip() for e in (match.group("exempt") or "").split("!") if e.strip())
            changes.append(RegimeChange(scope, int(match.group("year")), before, after,
                                        country, exempt))
    return tuple(changes)


def _signed_year(date: Optional[dict]) -> Optional[int]:
    """A period date as a signed year; None when it cannot be read."""
    if not isinstance(date, dict):
        return None
    year = date.get("year")
    if not isinstance(year, int):
        return None
    return -year if date.get("era") == "BCE" else year


def _signature(period: dict) -> Tuple[str, Optional[int], Optional[int]]:
    return (
        (period.get("periodName") or "").strip().casefold(),
        _signed_year(period.get("periodStart")),
        _signed_year(period.get("periodEnd")),
    )


def _scopes_by_city() -> Dict[str, set]:
    """Which civilisations' tables apply to which city files, by content id.

    A scope's lines apply to the cities its History lists — resolved through
    `topics.subject_resolver`, the same machinery as everything else, so the
    lint and the work list cannot disagree about which atom a listed name is.
    """
    scopes: Dict[str, set] = defaultdict(set)
    for topic in topics._raw_occurrences("city"):
        resolved = None
        if topic.context:
            resolver = _resolver_for(topic.context)
            resolved = resolver(topic.subject, topic.context)
        if resolved:
            scopes[resolved].add(topic.context)
    return scopes


_resolvers: dict = {}


def _resolver_for(_context: str):
    # One resolver per process invocation; contexts share it (subject_resolver
    # is context-aware internally). Cached because it walks the corpus.
    if "city" not in _resolvers:
        _resolvers["city"] = topics.subject_resolver("city")
    return _resolvers["city"]


@dataclass
class PeriodFlag:
    path: Path
    index: int          # position in cityPeriodsOfHistory
    period: dict
    crossings: List[RegimeChange]


@dataclass
class Disagreement:
    signature: Tuple[str, Optional[int], Optional[int]]
    votes: Dict[str, List[Path]]    # government value -> files holding it

    @property
    def majority(self) -> Optional[str]:
        """The winning value, or None on a tie — ties are reported, not guessed."""
        ranked = sorted(self.votes.items(), key=lambda kv: len(kv[1]), reverse=True)
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            return None
        return ranked[0][0]


@dataclass
class LintReport:
    flags: List[PeriodFlag] = field(default_factory=list)
    disagreements: List[Disagreement] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.flags and not self.disagreements


def lint(changes=None) -> LintReport:
    """Every over-broad period and every cross-city disagreement, read-only."""
    if changes is None:
        changes = load_regime_changes()
    scopes = _scopes_by_city() if changes else {}
    fold = topics._key

    report = LintReport()
    votes: dict = defaultdict(lambda: defaultdict(list))
    for path in store.iter_atoms("city"):
        data = store.read_json(path)
        if not isinstance(data, dict):
            continue
        if resolve_stored_class("city", data).__name__ != "CityExpanded":
            continue
        try:
            content_id = store.AtomName.decompose_filename(path.name).padded_id
        except ValueError:
            continue
        city_scopes = {fold(s) for s in scopes.get(content_id, ())}
        # Where the city actually is, for the `@country`-qualified lines. The
        # seed says which civilisation claims it; only the atom says which
        # modern state's politics apply to it.
        city_country = (data.get("location") or {}).get("country")
        city_name = (data.get("name") or {}).get("common")
        for i, period in enumerate(data.get("cityPeriodsOfHistory", [])):
            if not isinstance(period, dict):
                continue
            votes[_signature(period)][period.get("typeOfGovernment")].append(path)

            start = _signed_year(period.get("periodStart"))
            end = _signed_year(period.get("periodEnd"))
            if start is None:
                continue
            crossings = [
                c for c in changes
                if fold(c.scope) in city_scopes
                and c.applies_to(city_country)
                and not c.exempts(period.get("polity"), city_name)
                and start < c.year and (end is None or c.year < end)
            ]
            if crossings:
                report.flags.append(PeriodFlag(path, i, period, crossings))

    for signature, ballots in votes.items():
        if len(ballots) > 1:
            report.disagreements.append(Disagreement(signature, dict(ballots)))
    return report


def adjudicate(report: Optional[LintReport] = None, apply: bool = False) -> int:
    """Settle label disagreements by majority, mechanically. Returns edits.

    The minority files get the majority's value written through `store`, each
    edit validated as `CityExpanded` first and recorded in the journal as a
    mechanical adjudication (no payload — no model was involved). Ties change
    nothing and stay in the report.
    """
    from universe_builder import journal

    if report is None:
        report = lint()
    edits = 0
    for disagreement in report.disagreements:
        winner = disagreement.majority
        if winner is None:
            log.warning("Tie for %s — left for a human or a better majority",
                        disagreement.signature)
            continue
        for value, paths in disagreement.votes.items():
            if value == winner:
                continue
            for path in paths:
                if not apply:
                    log.warning("Would set %r to %r in %s", disagreement.signature[0],
                                winner, path.name)
                    edits += 1
                    continue
                data = store.read_json(path)
                changed = False
                for period in data.get("cityPeriodsOfHistory", []):
                    if isinstance(period, dict) and _signature(period) == disagreement.signature:
                        period["typeOfGovernment"] = winner
                        changed = True
                if not changed:
                    continue
                atom = CityExpanded.model_validate(data)
                store.write_atom(path, atom)
                edits += 1
                journal.current().record_attempt(
                    work_item_id=journal.work_item_from_path("city", path)
                    or journal.work_item_id("city", path.stem),
                    raw_payload=None,
                    raw_payload_unavailable_reason=(
                        "mechanical adjudication, no model involved: "
                        f"{disagreement.signature[0]!r} set to {winner!r} by "
                        f"majority across {sum(len(v) for v in disagreement.votes.values())} cities"
                    ),
                    operation="adjudicate",
                    atom_type="city",
                    subject=path.name,
                    validation_status="ok",
                    target_pointer=str(path),
                )
                log.warning("Adjudicated %r to %r in %s",
                            disagreement.signature[0], winner, path.name)
    return edits


def _acceptance_error(original: dict, replacement: List[dict]) -> Optional[str]:
    """Why a split reply may not be spliced, or None when it must be.

    The splice replaces recorded history, so the reply is held to a mechanical
    contract — the review gate caught the version without one, where a model
    returning sub-periods covering half the span silently deleted the rest
    *and still passed lint*, because the surviving periods crossed nothing.
    The contract: at least two periods (one period is not a split), in
    chronological order, contiguous (each starts where the previous ends,
    year-boundary tolerance of one), starting exactly at the original start
    and ending exactly at the original end — an open-ended original demands an
    open-ended final sub-period. Anything else is refused, logged, journaled,
    and the period stays flagged for the next run.
    """
    if len(replacement) < 2:
        return "one period back is not a split"
    starts = [_signed_year(p.get("periodStart")) for p in replacement]
    ends = [_signed_year(p.get("periodEnd")) for p in replacement]
    if starts[0] is None or any(s is None for s in starts[1:]) or any(e is None for e in ends[:-1]):
        return "a sub-period boundary has no readable year"
    original_start = _signed_year(original.get("periodStart"))
    original_end = _signed_year(original.get("periodEnd"))
    if original_start is not None and starts[0] != original_start:
        return (f"coverage starts at {starts[0]}, the original at {original_start} — "
                f"history before the first sub-period would be deleted")
    if (original_end is None) != (ends[-1] is None):
        return "the original and the final sub-period disagree about being open-ended"
    if original_end is not None and ends[-1] != original_end:
        return (f"coverage ends at {ends[-1]}, the original at {original_end} — "
                f"history after the last sub-period would be deleted")
    for i, (prev_end, next_start) in enumerate(zip(ends, starts[1:])):
        if not (0 <= next_start - prev_end <= 1):
            return (f"sub-periods {i} and {i + 1} are not contiguous "
                    f"({prev_end} -> {next_start}); a gap here is deleted history")
    return None


SPLIT_INSTRUCTION = (
    "The JSON below is ONE period of {city}'s history, but it spans these "
    "changes of government: {changes}. Split it into consecutive periods at "
    "each change, in chronological order. The sub-periods must jointly cover "
    "EXACTLY the original span: the first starts on the original start year, "
    "the last ends on the original end year (open-ended stays open-ended), "
    "and each starts where the previous one ends — a reply that covers less "
    "than the original span will be rejected. Carry every field truthfully "
    "for each sub-period."
)


def split_flagged(report: Optional[LintReport] = None, model: Optional[str] = None,
                  apply: bool = False) -> int:
    """Send each flagged period out as one `PeriodSplit` call and splice.

    The model decides the history (where exactly the sub-periods begin and
    end, what each looked like); the checker decided *that* a split happens
    and the splice is mechanical. Re-running `lint` afterwards is the
    convergence test — done means zero flags, a machine verdict.
    """
    from universe_builder import journal
    from universe_builder.config import base_system_text
    from universe_builder.providers import run as provider_run
    from universe_builder.providers.types import LLMRequest

    if report is None:
        report = lint()
    if model is None:
        from universe_builder import pipeline
        model = pipeline.load().resolve_model("fix").value

    splits = 0
    by_path: dict = defaultdict(list)
    for flag in report.flags:
        by_path[flag.path].append(flag)

    for path, flags in by_path.items():
        if not apply:
            for flag in flags:
                log.warning("Would split %r (%s) in %s at %s",
                            flag.period.get("periodName"), _signature(flag.period),
                            path.name, [c.year for c in flag.crossings])
                splits += 1
            continue
        data = store.read_json(path)
        periods = data.get("cityPeriodsOfHistory", [])
        changed = False
        # Splice from the end so earlier indices stay valid.
        for flag in sorted(flags, key=lambda f: f.index, reverse=True):
            changes_text = "; ".join(
                f"{abs(c.year)} {'BCE' if c.year < 0 else 'CE'} ({c.before} -> {c.after})"
                for c in flag.crossings
            )
            request = LLMRequest(
                model=model,
                schema_model=PeriodSplit,
                system=base_system_text,
                user=SPLIT_INSTRUCTION.format(
                    city=path.name, changes=changes_text,
                ) + "\n\n" + json.dumps(flag.period, indent=2, ensure_ascii=False),
                custom_id=f"split:{path.name}:{flag.index}",
            )
            response = provider_run.complete(request)
            refusal = None
            if not response.ok or not response.parsed or not response.parsed.periods:
                refusal = response.error or "no periods in the reply"
            else:
                replacement = [p.model_dump() for p in response.parsed.periods]
                refusal = _acceptance_error(flag.period, replacement)

            journal.current().record_attempt(
                work_item_id=(journal.work_item_from_path("city", path)
                              or journal.work_item_id("city", path.stem)),
                raw_payload=response.raw_text or None,
                raw_payload_unavailable_reason=(
                    None if response.raw_text else (response.error or "no text")),
                operation="split",
                atom_type="city",
                subject=f"{flag.period.get('periodName')} in {path.name}",
                requested_model=model,
                validation_status="failed" if refusal else "ok",
                validation_errors=[refusal] if refusal else None,
                target_pointer=str(path),
                # Without these the split was unpriceable (`costs` saw no
                # usage); every model call an operation makes passes them.
                **journal.usage_fields(response.usage, model),
            )
            if refusal:
                log.error("Split of %r in %s refused (%s); the period stays flagged",
                          flag.period.get("periodName"), path.name, refusal)
                continue
            periods[flag.index:flag.index + 1] = replacement
            splits += 1
            changed = True
            log.warning("Split %r in %s into %d periods",
                        flag.period.get("periodName"), path.name, len(replacement))
        if not changed:
            continue  # every split failed; do not churn the file for nothing
        atom = CityExpanded.model_validate(data)
        store.write_atom(path, atom)
    return splits


def run_periods(args) -> int:
    """The `universe_builder periods` operation for the history domain: lint city periods,
    optionally adjudicate labels (free) and split over-broad ones (paid).

    Registered on the history pack as the "periods" operation; `cli.cmd_periods`
    dispatches here through the active pack, so the CLI holds no period logic.
    Report-only exits non-zero while anything is flagged, so a script can gate on
    a clean corpus; the two apply modes are separate flags because one is free and
    mechanical and the other spends model calls.
    """
    from universe_builder import journal, pipeline
    from universe_builder.providers import run as provider_run

    if args.apply_labels or args.split:
        journal.start_run(config={"command": "periods",
                                  "apply_labels": bool(args.apply_labels),
                                  "split": bool(args.split),
                                  "model": args.model})
        provider_run.configure(pipeline.load().retry)

    report = lint()
    print(f"disagreeing signatures  {len(report.disagreements)}")
    for d in report.disagreements:
        verdict = d.majority or "TIE — needs a human"
        counts = {k: len(v) for k, v in d.votes.items()}
        print(f"    {d.signature[0]!r} {d.signature[1]}–{d.signature[2]}: "
              f"{counts} -> {verdict}")
    print(f"periods crossing a regime change  {len(report.flags)}")
    for flag in report.flags:
        print(f"    {flag.period.get('periodName')!r} in {flag.path.name} "
              f"crosses {[c.year for c in flag.crossings]}")

    if args.apply_labels:
        edits = adjudicate(report, apply=True)
        print(f"\nadjudicated {edits} minority label(s) to the majority")
    if args.split:
        model = None
        if args.model:
            model = pipeline.load().resolve_model("fix", override=args.model).value
        splits = split_flagged(report, model=model, apply=True)
        print(f"split {splits} period(s)")
    if args.apply_labels or args.split:
        after = lint()
        print(f"\nafter: {len(after.disagreements)} disagreement(s), "
              f"{len(after.flags)} flagged period(s)")
        return 0 if after.clean else 1
    return 0 if report.clean else 1
