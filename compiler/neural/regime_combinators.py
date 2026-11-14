"""
Multi-regime combinators (RFC-012, Step 4).

A statute can impose several regimes on the same liability: a general rate, a
special rate, a cap, a credit. Step 4 defines how those regimes combine.

All four combinators operate on integer minor units and return one integer.
No intermediate rounding is permitted: the only rounding is the half-up
quantization of a rational rate, which is the USVM rule and is reproduced
exactly by :func:`rate_liability`.

The combinators lower to the SSA primitives that already exist in both engines
(``MIN``, ``MAX``, ``ADD``, ``CONDITIONAL``), so no new USVM opcode is
required and the Catala and USVM results must agree to the paisa.

OQ-3 (the source of preemption priority) is closed as follows.
``EXCLUSIVE_PREEMPT`` requires an explicit priority on every regime:

* a regime without a declared priority is ``ERR_PREEMPTION_PRIORITY_MISSING``;
* two regimes that share the top priority are ``ERR_PREEMPTION_TIE``, unless
  exactly one of them carries an explicit ``notwithstanding``.

The chosen regime and the source of its priority are recorded in the result,
so every preemption is auditable from the AST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BENEFICIAL_MIN = "BENEFICIAL_MIN"
PENAL_MAX = "PENAL_MAX"
EXCLUSIVE_PREEMPT = "EXCLUSIVE_PREEMPT"
INDEPENDENT_ACCUMULATE = "INDEPENDENT_ACCUMULATE"

COMBINATORS = (
    BENEFICIAL_MIN,
    PENAL_MAX,
    EXCLUSIVE_PREEMPT,
    INDEPENDENT_ACCUMULATE,
)

ERR_PREEMPTION_PRIORITY_MISSING = "ERR_PREEMPTION_PRIORITY_MISSING"
ERR_PREEMPTION_TIE = "ERR_PREEMPTION_TIE"
ERR_DUPLICATE_REGIME = "ERR_DUPLICATE_REGIME"
ERR_UNKNOWN_COMBINATOR = "ERR_UNKNOWN_COMBINATOR"
ERR_NO_REGIMES = "ERR_NO_REGIMES"


@dataclass(frozen=True)
class Regime:
    """One regime of a multi-regime provision."""

    id: str
    value: int | None = None          # the liability in minor units, when static
    value_ref: str | None = None      # the SSA reference of the liability
    priority: int | None = None       # the declared preemption priority
    notwithstanding: bool = False      # the regime overrides the others
    condition: str | None = None      # the SSA boolean reference of the regime
    section: str | None = None        # the section the regime comes from


@dataclass
class CombinatorDiagnostic:
    code: str
    detail: str


@dataclass
class CombinationResult:
    """The outcome of one combination, with its audit record."""

    combinator: str
    value: int | None = None
    expression: Any | None = None
    selected: list[str] = field(default_factory=list)
    selected_regime: str | None = None
    priority_source: str | None = None
    diagnostics: list[CombinatorDiagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def audit(self) -> dict[str, Any]:
        """The auditable record of the decision."""
        return {
            "combinator": self.combinator,
            "selected": list(self.selected),
            "selectedRegime": self.selected_regime,
            "prioritySource": self.priority_source,
        }


def rate_liability(base_minor: int, rate_bps: int, denominator: int = 10000) -> int:
    """The exact liability of a rate, in minor units.

    This reproduces the USVM ``RATIONAL_MULTIPLY`` contract: the quotient is
    truncated toward zero and the remainder is rounded half-up, away from zero.
    """
    numerator = base_minor * rate_bps
    quotient = abs(numerator) // denominator
    if numerator < 0:
        quotient = -quotient
    remainder = numerator - quotient * denominator
    if abs(remainder) * 2 >= denominator:
        quotient += -1 if remainder < 0 else 1
    return quotient


def _canonical(regimes: list[Regime]) -> list[Regime]:
    """The canonical regime order: by id, so the result never depends on the
    order in which the regimes were found."""
    return sorted(regimes, key=lambda r: r.id)


def _fold(op: str, regimes: list[Regime]) -> Any | None:
    """Chains a binary operator over the regime references."""
    refs = [
        r.value_ref if r.value_ref is not None else {"op": "TO_PAISE", "value": str(r.value)}
        for r in _canonical(regimes)
        if r.value_ref is not None or r.value is not None
    ]
    if not refs:
        return None
    expression = refs[0]
    for ref in refs[1:]:
        expression = {"op": op, "left": expression, "right": ref}
    return expression


def _select(result: CombinationResult, regimes: list[Regime], beneficial: bool) -> CombinationResult:
    """BENEFICIAL_MIN and PENAL_MAX: one regime, the more favourable one."""
    op = "MIN" if beneficial else "MAX"
    result.expression = _fold(op, regimes)
    values = [r.value for r in regimes]
    if any(value is None for value in values):
        # The engine computes the comparison; the winner is not known here.
        result.selected = [r.id for r in _canonical(regimes)]
        result.selected_regime = None
        return result
    best = min(values) if beneficial else max(values)
    candidates = sorted((r for r in regimes if r.value == best), key=lambda r: r.id)
    result.value = best
    result.selected_regime = candidates[0].id
    result.selected = [candidates[0].id]
    return result


def _preempt(result: CombinationResult, regimes: list[Regime]) -> CombinationResult:
    """EXCLUSIVE_PREEMPT: one regime, the one with the highest priority."""
    missing = sorted(r.id for r in regimes if r.priority is None)
    if missing:
        result.diagnostics.append(
            CombinatorDiagnostic(
                ERR_PREEMPTION_PRIORITY_MISSING,
                f"regimes {missing} declare no priority; the preemption cannot be resolved",
            )
        )
        return result
    top = max(r.priority for r in regimes if r.priority is not None)
    candidates = [r for r in regimes if r.priority == top]
    if len(candidates) > 1:
        overrides = [r for r in candidates if r.notwithstanding]
        if len(overrides) == 1:
            chosen = overrides[0]
            result.priority_source = "notwithstanding"
        else:
            result.diagnostics.append(
                CombinatorDiagnostic(
                    ERR_PREEMPTION_TIE,
                    (
                        f"regimes {sorted(r.id for r in candidates)} share priority {top} "
                        "and none of them overrides the others"
                    ),
                )
            )
            return result
    else:
        chosen = candidates[0]
        result.priority_source = "declared"
    result.selected_regime = chosen.id
    result.selected = [chosen.id]
    result.value = chosen.value
    result.expression = _preempt_expression(chosen, regimes)
    return result


def _preempt_expression(chosen: Regime, regimes: list[Regime]) -> Any | None:
    """The expression of a preemption: a condition chain, highest priority first."""
    ordered = sorted(
        regimes,
        key=lambda r: (-(r.priority or 0), r.id),
    )
    expression = None
    for regime in reversed(ordered):
        if regime.value_ref is None:
            continue
        if expression is None:
            expression = regime.value_ref
        elif regime.condition:
            expression = {
                "op": "CONDITIONAL",
                "test": regime.condition,
                "then": regime.value_ref,
                "else": expression,
            }
        else:
            # A regime without a condition always preempts the ones below it.
            expression = regime.value_ref
    return expression


def _accumulate(result: CombinationResult, regimes: list[Regime]) -> CombinationResult:
    """INDEPENDENT_ACCUMULATE: every regime contributes, at most once."""
    kept: list[Regime] = []
    seen: dict[str, Regime] = {}
    for regime in regimes:
        previous = seen.get(regime.id)
        if previous is not None:
            if previous.value != regime.value:
                result.diagnostics.append(
                    CombinatorDiagnostic(
                        ERR_DUPLICATE_REGIME,
                        (
                            f"regime {regime.id!r} contributes twice with different "
                            "values; a regime id may contribute at most once"
                        ),
                    )
                )
                return result
            # The same regime applied again contributes once.
            continue
        seen[regime.id] = regime
        kept.append(regime)
    result.selected = sorted(r.id for r in kept)
    result.expression = _fold("ADD", kept)
    if all(r.value is not None for r in kept):
        result.value = sum(r.value for r in kept)
    return result


def combine(combinator: str, regimes: list[Regime]) -> CombinationResult:
    """Combines the regimes of a multi-regime provision.

    The result carries the value (when every input is static), the expression
    to lower, the selected regimes and the audit record. A diagnostic means the
    combination is not well defined and no expression is produced.
    """
    result = CombinationResult(combinator=combinator)
    if combinator not in COMBINATORS:
        result.diagnostics.append(
            CombinatorDiagnostic(
                ERR_UNKNOWN_COMBINATOR,
                f"unknown combinator {combinator!r}; expected one of {list(COMBINATORS)}",
            )
        )
        return result
    if not regimes:
        result.diagnostics.append(
            CombinatorDiagnostic(ERR_NO_REGIMES, "a combination needs at least one regime")
        )
        return result
    if combinator == INDEPENDENT_ACCUMULATE:
        return _accumulate(result, regimes)
    if combinator == EXCLUSIVE_PREEMPT:
        return _preempt(result, regimes)
    return _select(result, regimes, beneficial=combinator == BENEFICIAL_MIN)
