"""The arithmetic behind a prompt experiment's verdict (v0.5.0).

Pure functions, standard library only, so every number in a report can be
checked by hand. The tests pin each against hand-computed values.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p): the exact one-sided sign test."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def wilson(successes: float, n: float, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval for a proportion; (nan, nan) when n is 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


#: One-sided 95%: the z for a lower bound alone.
Z_ONE_SIDED_95 = 1.6448536269514722


def sign_flip_p(diffs: Sequence[float], *, greater: bool = True, permutations: int = 10_000,
                seed: int = 0) -> float:
    """One-sided paired permutation test of mean(diffs) against 0.

    Each subject's difference keeps its size and has its sign flipped at
    random; the p-value is the share of flips whose mean is at least as
    extreme as the observed one (with the observed counted, so never 0).
    `greater`: the alternative is mean > 0; otherwise mean < 0. Seeded, so a
    report reproduces exactly."""
    values = [float(d) for d in diffs]
    if not values or all(v == 0 for v in values):
        return 1.0
    observed = sum(values) / len(values)
    rng = random.Random(seed)
    hits = 1
    for _ in range(permutations):
        mean = sum(v if rng.random() < 0.5 else -v for v in values) / len(values)
        if (mean >= observed) if greater else (mean <= observed):
            hits += 1
    return hits / (permutations + 1)


def holm(pvalues: Sequence[float]) -> list:
    """Holm-adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def mean(values: Sequence[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def power(d: int, p: float, alpha: float = 0.05) -> float:
    """Power of the one-sided exact sign test at `d` decisive pairs when B's
    true win rate is `p`. For the plan's table and the report's footnote."""
    k = next((k for k in range(d + 1) if binom_sf(k, d) < alpha), d + 1)
    return binom_sf(k, d, p)
