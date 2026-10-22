"""Domain packs: pluggable problem domains the engine can run.

Each subpackage registers a `DomainPack` (see `universe_builder.registry`) on import, so
`registry.set_active_pack("<name>")` discovers it by importing
`universe_builder.domains.<name>`. `history` is the built-in pack (declared in
`registry.py`); `medications` is a second domain that exists to prove the core
carries no history-specific assumptions.
"""
