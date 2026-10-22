"""History's schema migrations.

Discovered and imported by the core migration loader (`universe_builder.migrations._loaded`)
when the history pack is active, the same way a second domain's migrations would
be — so migrate-on-read stays a core mechanism while the per-atom steps live with
their domain. Each module self-registers via `universe_builder.migrations.register`.
"""
