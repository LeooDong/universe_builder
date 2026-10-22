"""The history domain's Pydantic models.

Moved here from core `universe_builder/models/` so that core carries no
history-specific shapes. The shared primitives —
`Name`, `Date`, `Location`, `Judgments`, `Certainty`, `WireModel`, `EmptyModel`
— stay in `universe_builder.models.subatomic`, because a second domain (medications) builds
on them too; only the atom shapes that are *about history* live here.
"""
