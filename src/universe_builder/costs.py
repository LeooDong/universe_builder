"""What each run, atom and model actually cost, read back out of the journal.

**This stores nothing.** The journal already records every attempt — its
`work_item_id`, subject, atom type, model, execution mode, token counts and,
sometimes, a cost — so a cost ledger would be a second copy of facts already on
disk, and a second copy is a thing that can disagree with the first. This module
is a *reader*, in the same sense `universe_builder index` is: always recomputable, never
edited, and free to be thrown away. A full scan of every run takes a fraction
of a second, so there is nothing to cache and no database to keep in step.

The one real problem it solves is that **the journal's own cost field is mostly
empty**: most attempts in an older journal carry a null `estimated_cost`,
because the field was added after most of the corpus was generated. Tokens were
recorded faithfully throughout, so those attempts can be priced after the fact
against `model_pricing.json` — and that is what `Row.recomputed` marks. The
distinction is kept all the way to the report rather than averaged away: a
recomputed figure uses today's prices for a call billed at the prices of the
day, which is fine for "what does an artifact cost" and wrong for "what did I
spend in September".

Three states, and they are not collapsed:

* **reported** — the journal carried a cost. Authoritative.
* **recomputed** — priced here from stored tokens. Good enough to compare
  models, not an invoice.
* **unpriceable** — no usage recorded, so nothing to price from. Counted and
  shown, never silently dropped, because an unpriceable attempt is
  indistinguishable from a free one inside a total.

**The unpriceable attempts are historic, and they are not failures.** Nearly
all of them record `work_status=ok` and `validation_status=ok` — they produced
valid atoms and simply recorded no usage. Every one of them is synchronous
(`batch_id` unset) and predates the fix: batch attempts always carried usage,
and the synchronous path did not capture it until an early version. Nothing here can recover
that — the tokens were never written down — so those attempts are excluded
from every figure and reported separately. A corpus-wide total is therefore a
floor, not a bill.

**A mechanical write is not a call** (after v0.3.0). A pack verb
that rewrites an atom without a model (retire, relabel, respell) journals an
attempt line so provenance shows the write (`journal.record_mechanical`);
`costs` leaves those lines out of every row, total and footer and counts them
on their own. Until then they were counted as unpriceable sync calls and
blamed on the gap above, and they were most of the "no usage" attempts.
Lines written before the marker existed are recognised by
shape (`journal.is_mechanical`), so no journal is rewritten.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from universe_builder import journal, pipeline, settings
from universe_builder.providers import pricing
from universe_builder.providers.types import Usage

log = logging.getLogger(__name__)

#: Journal `execution_mode` values that mean the provider billed at batch rates.
#: Anything else — including a null, which is what the synchronous path writes —
#: is priced at list. Guessing the other way would under-report a real bill.
BATCH_MODES = frozenset({"batch"})

#: What `--by` accepts. `subject` is the finest: one row per atom generated,
#: which is the question "what did *this* atom cost" rather than "what did this
#: run cost". `effort` (v0.6.0) is model and effort together.
GROUPINGS = ("atom", "model", "run", "phase", "day", "subject", "mode", "effort")


@dataclass
class Row:
    """One aggregation bucket."""

    key: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0
    #: Attempts whose cost this module derived rather than read.
    recomputed: int = 0
    #: Attempts carrying no usage at all, so contributing nothing to `cost`.
    unpriceable: int = 0
    #: Attempts that failed validation. They cost money and produce no atom,
    #: which is the single most useful thing this report can surface.
    failed: int = 0
    models: set = field(default_factory=set)

    @property
    def cost_per_call(self) -> float:
        priced = self.calls - self.unpriceable
        return self.cost / priced if priced else 0.0

    @property
    def output_per_call(self) -> float:
        priced = self.calls - self.unpriceable
        return self.output_tokens / priced if priced else 0.0


@dataclass
class Report:
    rows: List[Row]
    total: Row
    grouping: str
    #: Runs scanned, for the "is this everything?" question a total invites.
    runs: int = 0
    #: Attempt lines no model made (`journal.is_mechanical`), after the
    #: filters: not calls, so in no row and no total; counted here instead.
    mechanical: int = 0

    def as_dict(self) -> dict:
        def row(r: Row) -> dict:
            return {
                "key": r.key, "calls": r.calls, "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens, "cache_read_tokens": r.cache_read_tokens,
                "cost": round(r.cost, 6), "cost_per_call": round(r.cost_per_call, 6),
                "recomputed": r.recomputed, "unpriceable": r.unpriceable,
                "failed": r.failed, "models": sorted(r.models),
            }
        return {
            "grouping": self.grouping,
            "runs": self.runs,
            "mechanical": self.mechanical,
            "total": row(self.total),
            "rows": [row(r) for r in self.rows],
        }


def _runs_dir() -> Path:
    return settings.data_dir() / "_runs"


def iter_attempts(runs_dir: Optional[Path] = None) -> Iterator[dict]:
    """Every attempt event across every run, oldest run first.

    A malformed line is skipped with a warning rather than aborting the scan:
    the journal is append-only and written during live runs, so a torn final
    line is an expected state, not a corrupt file.
    """
    root = runs_dir or _runs_dir()
    if not root.exists():
        return
    for path in sorted(root.glob("*/journal.jsonl")):
        for number, line in enumerate(path.open(encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s line %d is not JSON; skipped", path.parent.name, number)
                continue
            if event.get("event_type") == "attempt":
                yield event


def _run_configs(runs_dir: Optional[Path] = None) -> Dict[str, dict]:
    """run_id -> the `config` block `run.json` recorded, for phase grouping.

    The attempt events carry no phase: a phase is a property of the run. Runs
    predating `run.json` simply have no entry, and their attempts group under
    "(unrecorded)" rather than being dropped.
    """
    root = runs_dir or _runs_dir()
    out: Dict[str, dict] = {}
    if not root.exists():
        return out
    for meta in sorted(root.glob("*/run.json")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        run_id = data.get("run_id") or meta.parent.name
        out[run_id] = data.get("config") or {}
    return out


def _cost_of(event: dict) -> tuple[Optional[float], bool]:
    """`(cost, was_recomputed)`. `(None, False)` when it cannot be priced.

    A stored cost wins: it was computed against the prices in force when the
    call was billed. Only when it is absent do we reprice from tokens, which is
    today's price applied to an older call — accurate for comparing models,
    approximate for a historical total, and flagged either way.
    """
    stored = event.get("estimated_cost")
    if stored is not None:
        return float(stored), False

    usage = Usage(
        input_tokens=event.get("input_tokens") or 0,
        output_tokens=event.get("output_tokens") or 0,
        cache_read_tokens=event.get("cache_read_tokens") or 0,
        cache_write_tokens=event.get("cache_write_tokens") or 0,
    )
    if not (usage.input_tokens or usage.output_tokens):
        return None, False

    model = event.get("returned_model") or event.get("requested_model")
    if not model:
        return None, False
    batch = (event.get("execution_mode") or "") in BATCH_MODES
    try:
        return pricing.estimate_cost(usage, model, batch=batch), True
    except Exception:  # noqa: BLE001 - an unpriced model must not abort the scan
        return None, False


def _key_for(event: dict, grouping: str, configs: Dict[str, dict]) -> str:
    if grouping == "atom":
        return event.get("atom_type") or "(unknown)"
    if grouping == "model":
        return event.get("returned_model") or event.get("requested_model") or "(unknown)"
    if grouping == "run":
        return event.get("run_id") or "(unknown)"
    if grouping == "mode":
        return event.get("execution_mode") or "sync"
    if grouping == "effort":
        # Model and effort together (v0.6.0): an effort means nothing apart
        # from the model it was asked of. `default` is a line that sent none.
        model = event.get("returned_model") or event.get("requested_model") or "(unknown)"
        return f"{model} @ {event.get('sent_effort') or 'default'}"
    if grouping == "subject":
        return event.get("subject") or event.get("work_item_id") or "(unknown)"
    if grouping == "day":
        stamp = event.get("timestamp_utc") or ""
        return stamp[:10] or "(undated)"
    if grouping == "phase":
        return _phase_of(event, configs)
    raise ValueError(f"unknown grouping {grouping!r}; expected one of {', '.join(GROUPINGS)}")


def _phase_of(event: dict, configs: Dict[str, dict]) -> str:
    """Which pipeline phase an attempt belongs to.

    A phase is a property of the *run*, and attempts carry none — so a
    synchronous run answers directly from `run.json`. Batch work does not: the
    run that writes those attempts is a `collect`, hours or days after the
    submit, and "collect" is not a phase. Left unhandled it swallowed every
    batch attempt and most of the spend into one meaningless row, which is
    worse than no grouping at all.

    Now `batch._journal_result` records the operation the handle
    routed (`requests.build_requests` stamps it at submit), and that answers
    the question directly. Older entries lack it. For those, what can be
    recovered is recovered from the atom type, since `expand` and `append`
    generate under their own prefixed types; what cannot is said plainly: a
    batch-collected `create`, `fix` and `verify` all write a base atom type
    with the base schema, and nothing in such an entry separates them.
    """
    cfg = configs.get(event.get("run_id") or "", {})
    phase = cfg.get("phase")
    if phase:
        # Older runs call the seed phase `history`.
        return pipeline.modern_phase(phase)

    command = cfg.get("command")
    if command == "collect":
        operation = event.get("operation")
        if operation:
            return f"{operation} (batch)"
        atom = event.get("atom_type") or ""
        for prefix in ("expand_", "append_"):
            if atom.startswith(prefix):
                return f"{prefix[:-1]} (batch)"
        return "create/fix/verify (batch)"
    return command or "(unrecorded)"


def build(
    grouping: str = "atom",
    *,
    atom: Optional[str] = None,
    run: Optional[str] = None,
    since: Optional[str] = None,
    runs_dir: Optional[Path] = None,
    events: Optional[Iterable[dict]] = None,
) -> Report:
    """Aggregate the journal into one `Report`.

    `since` is compared as a string against the ISO timestamp, so `2026-09-24`
    and `2026-09-24T08:00:00Z` both work and neither needs a date parser.
    """
    if grouping not in GROUPINGS:
        raise ValueError(f"unknown grouping {grouping!r}; expected one of {', '.join(GROUPINGS)}")

    configs = _run_configs(runs_dir)
    source = events if events is not None else iter_attempts(runs_dir)
    buckets: Dict[str, Row] = defaultdict(lambda: Row(key=""))
    total = Row(key="total")
    seen_runs = set()
    mechanical = 0

    for event in source:
        if atom and event.get("atom_type") != atom:
            continue
        if run and event.get("run_id") != run:
            continue
        if since and (event.get("timestamp_utc") or "") < since:
            continue

        seen_runs.add(event.get("run_id"))
        if journal.is_mechanical(event):
            # A write no model made (a pack's list verb, a relabel): not a
            # call, so not unpriceable either. Until the marker existed these
            # were counted as sync calls with no usage and blamed on the
            # usage gap, and they were most of it.
            mechanical += 1
            continue
        key = _key_for(event, grouping, configs)
        row = buckets[key]
        row.key = key

        cost, recomputed = _cost_of(event)
        model = event.get("returned_model") or event.get("requested_model")
        failed = (event.get("validation_status") or "").lower() in ("invalid", "failed", "error")

        for target in (row, total):
            target.calls += 1
            target.input_tokens += event.get("input_tokens") or 0
            target.output_tokens += event.get("output_tokens") or 0
            target.cache_read_tokens += event.get("cache_read_tokens") or 0
            if model:
                target.models.add(model)
            if failed:
                target.failed += 1
            if cost is None:
                target.unpriceable += 1
            else:
                target.cost += cost
                if recomputed:
                    target.recomputed += 1

    rows = sorted(buckets.values(), key=lambda r: (-r.cost, r.key))
    return Report(rows=rows, total=total, grouping=grouping, runs=len(seen_runs - {None}),
                  mechanical=mechanical)


def format_report(report: Report, limit: Optional[int] = None) -> str:
    """The human-readable table, including the honesty line about provenance."""
    rows: Sequence[Row] = report.rows
    truncated = 0
    if limit is not None and len(rows) > limit:
        truncated = len(rows) - limit
        rows = rows[:limit]

    width = max([len(r.key) for r in rows] + [len(report.grouping), 5])
    width = min(width, 46)
    out = [
        f"{report.grouping:<{width}} {'calls':>7} {'out tok':>11} {'$ total':>9} {'$/call':>9} {'failed':>7}",
        "-" * (width + 47),
    ]
    for r in rows:
        key = r.key if len(r.key) <= width else r.key[: width - 1] + "…"
        out.append(f"{key:<{width}} {r.calls:>7} {r.output_tokens:>11,} "
                   f"{r.cost:>9.4f} {r.cost_per_call:>9.5f} {r.failed:>7}")
    if truncated:
        out.append(f"... and {truncated} more row(s); raise --limit to see them")

    t = report.total
    out.append("-" * (width + 47))
    out.append(f"{'total':<{width}} {t.calls:>7} {t.output_tokens:>11,} "
               f"{t.cost:>9.4f} {t.cost_per_call:>9.5f} {t.failed:>7}")
    out.append("")
    out.append(f"{report.runs} run(s) scanned.")

    # Provenance, always — a total whose provenance is invisible invites being
    # quoted as an invoice.
    if t.recomputed:
        share = 100 * t.recomputed / max(t.calls - t.unpriceable, 1)
        out.append(
            f"{t.recomputed} of {t.calls - t.unpriceable} priced attempts ({share:.0f}%) had no "
            "stored cost and were priced from tokens at today's rates. Good for comparing "
            "models; not an invoice."
        )
    if t.unpriceable:
        out.append(
            f"{t.unpriceable} attempt(s) recorded no usage and are excluded from every figure "
            "above, so the total is a floor rather than a bill. These are not failures: the "
            "synchronous path did not capture usage in early versions, and the "
            "tokens were never written down, so no report can recover them."
        )
    if report.mechanical:
        out.append(mechanical_line(report.mechanical))
    return "\n".join(out)


def mechanical_line(count: int) -> str:
    """The footer for writes no model made, which no figure above includes."""
    return (f"{count} mechanical write(s) journaled (no model involved: list verbs, relabels, "
            "respellings) are not calls and appear in no figure above.")
