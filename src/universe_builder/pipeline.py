"""Resolved pipeline configuration, read from `pipeline.yaml`.

Model choices used to be a dict literal in the orchestrator, which meant that
changing which model runs a phase was a code edit — and that the values drifted
a generation out of date without anyone noticing. They are data now.

The resolution order is the point of this module. A phase can be overridden per
atom, and any of it can be overridden from the command line, so the answer to
"which model will actually run" depends on three layers. `resolve_model` is the
single place that answers it, and `PipelineConfig.describe` prints the answer
with its provenance, so a dry run says *why* as well as *what*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from universe_builder import settings

log = logging.getLogger(__name__)

DEFAULT_PATH = settings.PACKAGE_DIR / "pipeline.yaml"

#: Where a resolved value came from, for `--dry-run` output. Ordered strongest
#: first, matching the precedence documented in pipeline.yaml.
SOURCE_CLI = "cli"
#: The seed phase was once called `history`. A config that still says so
#: is read as `index`, once, with a warning — silently honouring both names
#: would let one file configure the same phase twice. The one table of old
#: phase names: the CLI, the work selection, the orchestrator and `costs`
#: (reading runs journalled under the old name) all read it.
LEGACY_PHASES = {"history": "index"}


def modern_phase(phase):
    """`phase` under its current name: `history` is `index`; anything else is itself."""
    return LEGACY_PHASES.get(phase, phase)


def _modern_phases(table: dict, path, atom=None) -> dict:
    for old, new in LEGACY_PHASES.items():
        if old not in table:
            continue
        where = f"atoms.{atom}" if atom else "phases"
        if new in table:
            raise ValueError(
                f"{path}: {where} names both {old!r} and {new!r}; they are one "
                f"phase ({new!r}) — remove the {old!r} line")
        log.warning("%s: %s.%s is now called %r; rename it", path, where, old, new)
        table[new] = table.pop(old)
    return table


SOURCE_ATOM = "atoms.<atom>.<phase>"
SOURCE_PHASE = "phases.<phase>"
SOURCE_DEFAULT = "defaults.model"
#: Where a resolved effort came from (v0.6.0), strongest first after the CLI.
SOURCE_EFFORT_ATOM = "effort.atoms.<atom>.<phase>"
SOURCE_EFFORT_PHASE = "effort.phases.<phase>"
SOURCE_EFFORT_DEFAULT = "effort.default"
#: No effort anywhere: none is sent, and the model runs at its provider's default.
SOURCE_EFFORT_UNSET = "unset: none sent, the provider's default"
#: Where a resolved task budget came from (v0.6.1), likewise.
SOURCE_TASK_BUDGET_ATOM = "task_budget.atoms.<atom>.<phase>"
SOURCE_TASK_BUDGET_PHASE = "task_budget.phases.<phase>"
SOURCE_TASK_BUDGET_DEFAULT = "task_budget.default"
#: No task budget anywhere: none is sent, and only `max_tokens` bounds a reply.
SOURCE_TASK_BUDGET_UNSET = "unset: none sent"

#: Capability tiers, cheapest last. Naming a tier rather than a model is the
#: point: model ids turn over every few months, and a config full of them is a
#: config that is always slightly wrong. A phase says how *hard* its job is; the
#: tier table says which model that means today, per vendor.
#:
#: Three tiers rather than five. Every extra tier is a judgement call someone has
#: to make per vendor, and vendors genuinely ship roughly three useful sizes.
TIER_FRONTIER = "frontier"
#: Between frontier and balanced (v0.1.18): OpenAI's gpt-6.1-sol is a fifth of
#: gpt-6-astra's price and about eight times gpt-5.6-luna's. Vendors with
#: nothing in that gap map it to their frontier model, never below balanced.
TIER_STRONG = "strong"
TIER_BALANCED = "balanced"
TIER_ECONOMY = "economy"
TIERS = (TIER_FRONTIER, TIER_STRONG, TIER_BALANCED, TIER_ECONOMY)


class UnknownTier(ValueError):
    """A phase named a tier the active provider does not define."""


@dataclass(frozen=True)
class Resolved:
    """A resolved value plus where it came from."""

    value: Any
    source: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.value}  ({self.source})"


@dataclass
class PipelineConfig:
    defaults: Dict[str, Any] = field(default_factory=dict)
    phases: Dict[str, str] = field(default_factory=dict)
    atoms: Dict[str, Dict[str, str]] = field(default_factory=dict)
    retry: Dict[str, Any] = field(default_factory=dict)
    #: provider name -> tier name -> model id.
    tiers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    #: Prompt experiments (v0.5.0): `max_output_price`, the dearest output
    #: price (USD per 1M tokens) a generating arm may run on.
    experiments: Dict[str, Any] = field(default_factory=dict)
    #: The reasoning effort (v0.6.0, `providers.effort`): `default`, `phases`
    #: (phase -> level) and `atoms` (atom -> phase -> level), resolved like the
    #: models (`resolve_effort`). Empty: none is sent anywhere.
    effort: Dict[str, Any] = field(default_factory=dict)
    #: The task budget (v0.6.1, `providers.task_budget`): the same shape as
    #: `effort`, the values token counts (`resolve_task_budget`). Empty: none
    #: is sent anywhere.
    task_budget: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def provider(self) -> Optional[str]:
        """The vendor whose tier table is in force. `defaults.provider`."""
        return self.defaults.get("provider")

    def tier_table(self, provider: Optional[str] = None) -> Dict[str, str]:
        name = provider or self.provider
        if not name:
            return {}
        try:
            return self.tiers[name]
        except KeyError as e:
            raise UnknownTier(
                f"No tier table for provider {name!r} in {self.path}. "
                f"Known: {', '.join(sorted(self.tiers)) or 'none'}"
            ) from e

    def resolve_tier(self, value: str, provider: Optional[str] = None) -> str:
        """Turn a tier name into a model id; pass a model id through untouched.

        The passthrough is deliberate and is the escape hatch: pinning one phase
        to an exact model — to reproduce a run, or because a specific model is
        the only one that gets a job right — must not require abandoning tiers
        everywhere else.
        """
        if value not in TIERS:
            return value
        table = self.tier_table(provider)
        try:
            return table[value]
        except KeyError as e:
            raise UnknownTier(
                f"Provider {provider or self.provider!r} defines no {value!r} tier "
                f"in {self.path}. It has: {', '.join(sorted(table)) or 'none'}"
            ) from e

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PipelineConfig":
        path = Path(path) if path else DEFAULT_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a mapping, got {type(raw).__name__}")
        return cls(
            defaults=dict(raw.get("defaults") or {}),
            phases=_modern_phases(dict(raw.get("phases") or {}), path),
            atoms={k: _modern_phases(dict(v or {}), path, atom=k)
                   for k, v in (raw.get("atoms") or {}).items()},
            retry=dict(raw.get("retry") or {}),
            tiers={k: dict(v or {}) for k, v in (raw.get("tiers") or {}).items()},
            experiments=dict(raw.get("experiments") or {}),
            effort=_effort_block(raw.get("effort"), path),
            task_budget=_task_budget_block(raw.get("task_budget"), path),
            path=path,
        )

    # -- resolution -------------------------------------------------------------

    def resolve_model(self, phase: str, atom: Optional[str] = None,
                      override: Optional[str] = None,
                      provider: Optional[str] = None) -> Resolved:
        """The model that will actually run, and why.

        Returning the provenance rather than a bare string is what makes
        `--dry-run` worth running: "claude-opus-5" alone does not tell you
        whether it came from your flag, a per-atom override or the global
        default, which is exactly what you need to know before a large run.
        """
        raw, source = self._raw_choice(phase, atom, override)
        model = self.resolve_tier(raw, provider)
        if model != raw:
            # Say both, so a dry run explains a model nobody wrote down anywhere.
            source = f"{source} -> tiers.{provider or self.provider}.{raw}"
        return Resolved(model, source)

    def _raw_choice(self, phase: str, atom: Optional[str], override: Optional[str]):
        """The configured value before tier resolution: a tier name or a model id."""
        # The old name of a phase resolves as the new one, so a caller still
        # saying `history` gets the seed phase's tier rather than silently
        # falling through to `defaults.model`.
        phase = modern_phase(phase)
        if override:
            return override, SOURCE_CLI
        if atom and phase in self.atoms.get(atom, {}):
            return self.atoms[atom][phase], SOURCE_ATOM
        if phase in self.phases:
            return self.phases[phase], SOURCE_PHASE
        if "model" in self.defaults:
            return self.defaults["model"], SOURCE_DEFAULT
        raise KeyError(
            f"No model configured for phase {phase!r} and no defaults.model in {self.path}"
        )

    def resolve_effort(self, phase: str, atom: Optional[str] = None,
                       override: Optional[str] = None) -> Resolved:
        """The effort a phase will ask for, and why (v0.6.0).

        Precedence as for models: the CLI's `--effort`, then
        `effort.atoms.<atom>.<phase>`, then `effort.phases.<phase>`, then
        `effort.default`. None at every level resolves to None, which sends
        nothing: the run is what it was before v0.6.0."""
        from universe_builder.providers.effort import validate

        return _resolve_setting(self.effort, phase, atom, override, validate,
                                (SOURCE_EFFORT_ATOM, SOURCE_EFFORT_PHASE, SOURCE_EFFORT_DEFAULT,
                                 SOURCE_EFFORT_UNSET))

    def resolve_task_budget(self, phase: str, atom: Optional[str] = None,
                            override: Optional[int] = None) -> Resolved:
        """The task budget a phase will send, and why (v0.6.1).

        Precedence as for effort: the CLI's `--task-budget`, then
        `task_budget.atoms.<atom>.<phase>`, then `task_budget.phases.<phase>`,
        then `task_budget.default`. None everywhere sends none."""
        from universe_builder.providers.task_budget import validate

        return _resolve_setting(self.task_budget, phase, atom, override, validate,
                                (SOURCE_TASK_BUDGET_ATOM, SOURCE_TASK_BUDGET_PHASE,
                                 SOURCE_TASK_BUDGET_DEFAULT, SOURCE_TASK_BUDGET_UNSET))

    def phase_models(self) -> Dict[str, str]:
        """The phase table in the shape the orchestrator's `PHASE_MODELS` uses."""
        return {phase: self.resolve_model(phase).value for phase in self.phases}

    def max_tokens(self, phase: str, atom: Optional[str] = None) -> int:
        return int(self.defaults.get("max_tokens", 16384))

    def describe(self, atom: Optional[str] = None) -> str:
        """Human-readable resolution table, for `--dry-run`."""
        lines = [f"pipeline config: {self.path}"]
        for phase in sorted(self.phases):
            line = f"  {phase:<8} {self.resolve_model(phase, atom)}"
            effort = self.resolve_effort(phase, atom)
            if effort.value:
                line += f"   effort {effort}"
            budget = self.resolve_task_budget(phase, atom)
            if budget.value:
                line += f"   task budget {budget}"
            lines.append(line)
        if self.retry:
            lines.append(f"  retry    {self.retry}")
        return "\n".join(lines)


def _resolve_setting(block: Mapping, phase: str, atom: Optional[str], override, validate,
                     sources: tuple) -> Resolved:
    """A per-phase setting (`effort`, `task_budget`) resolved as models are:
    `override` (the CLI's), then `atoms.<atom>.<phase>`, then
    `phases.<phase>`, then `default`; `sources` names the last three and the
    unset case."""
    atom_source, phase_source, default_source, unset_source = sources
    phase = modern_phase(phase)
    if override:
        return Resolved(validate(override), SOURCE_CLI)
    atoms = block.get("atoms") or {}
    if atom and (atoms.get(atom) or {}).get(phase):
        return Resolved(atoms[atom][phase], atom_source)
    phases = block.get("phases") or {}
    if phases.get(phase):
        return Resolved(phases[phase], phase_source)
    if block.get("default"):
        return Resolved(block["default"], default_source)
    return Resolved(None, unset_source)


def _effort_block(raw: Any, path) -> Dict[str, Any]:
    """`effort:` as read, every level validated (a typo must not silently mean
    "the provider's default") and old phase names modernised."""
    from universe_builder.providers.effort import validate

    return _setting_block(raw, path, "effort", validate)


def _task_budget_block(raw: Any, path) -> Dict[str, Any]:
    """`task_budget:` as read (v0.6.1), every total validated (at least
    20,000) and old phase names modernised."""
    from universe_builder.providers.task_budget import validate

    return _setting_block(raw, path, "task_budget", validate)


def _setting_block(raw: Any, path, name: str, validate) -> Dict[str, Any]:
    """A `default`/`phases`/`atoms` block as read, every value validated."""
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: {name} must be a mapping (default, phases, atoms)")
    unknown = set(raw) - {"default", "phases", "atoms"}
    if unknown:
        raise ValueError(f"{path}: {name} has unknown key(s) {sorted(unknown)}; "
                         "allowed: default, phases, atoms")

    def level(value, where):
        try:
            return validate(value)
        except ValueError as e:
            raise ValueError(f"{path}: {where}: {e}") from e

    phases = _modern_phases(dict(raw.get("phases") or {}), path)
    return {
        "default": level(raw.get("default"), f"{name}.default"),
        "phases": {p: level(v, f"{name}.phases.{p}") for p, v in phases.items()},
        "atoms": {a: {p: level(v, f"{name}.atoms.{a}.{p}")
                      for p, v in _modern_phases(dict(table or {}), path, atom=a).items()}
                  for a, table in (raw.get("atoms") or {}).items()},
    }


def load(path: Optional[Path] = None) -> PipelineConfig:
    return PipelineConfig.load(path)
