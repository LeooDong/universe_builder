"""The closed-list coercion the atom types share (polity, monument, work, figure).

Case and whitespace are folded before the membership check, as the engine's
`subatomic._lower` does for Certainty: a model answering "Mythical" or
"Intact " means the listed value, and falling back over a capital letter
would record a wrong answer silently ("Mythical" became "legendary" in the
first draft).
"""

from typing import List


def coerce(value, allowed: List[str], fallback: str) -> str:
    """`value` folded if it is in `allowed`, else `fallback`."""
    folded = value.strip().lower() if isinstance(value, str) else value
    return folded if folded in allowed else fallback
