"""Small helpers shared by the providers."""

from __future__ import annotations

from universe_builder.providers.types import BatchHandle


def expected_custom_ids(handle: BatchHandle) -> list[str]:
    """Every custom_id that was submitted in this batch.

    Used to reconcile what came back against what went out, and to synthesise
    per-request failures when a batch fails at the job level and returns nothing.
    """
    if handle.extra.get("order"):
        return [cid for cid in handle.extra["order"] if cid]
    return list(handle.routing)
