"""
Single topological SSA walk (RFC-012, Step 3).

This module replaces the "first matching archetype wins" chain with one walk
over the resolved relation graph. Each relation is handled by a registered
handler; the handlers are pure functions that receive a node, its resolved
neighbours and a builder, and return one value reference.

The rules implemented here are the ones written down in
``docs/rfcs/RFC-012-lowerer-composition.md`` section 5:

* 5.2  the handler registry (relation -> handler);
* 5.3  one topological walk, deterministic order, fail closed on a cycle;
* 5.4  the composition acceptance test.

Two properties of the statutory text are load bearing and are handled here:

* a relation label alone does not decide the *sense* of a statutory
  adjustment. The neural annotator labels both ``capped at $10,000`` and
  ``reduced by a $500 credit`` as ``REL_CAP_CEILING``; the text decides which
  one is a cap (MIN) and which one is a reduction (SUB);
* the default order of the composed pipeline is base arithmetic, reductions,
  caps, floor at zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from compiler.neural.clause_linker import (
    base_phrase_after_rate,
    input_id_from_phrase,
    is_monetary_quantum_text,
    normalize_term,
)
from compiler.neural.inference import parse_minor_units, parse_rate_bps, parse_unit_and_type


# ── Diagnostic codes ────────────────────────────────────────────────────────

ERR_UNHANDLED_RELATION = "ERR_UNHANDLED_RELATION"
ERR_CYCLIC_DEPENDENCY = "ERR_CYCLIC_DEPENDENCY"
ERR_NO_BASE_OPERAND = "ERR_NO_BASE_OPERAND"
ERR_AMBIGUOUS_REFERENCE = "ERR_AMBIGUOUS_REFERENCE"
ERR_NO_RATE_HEAD = "ERR_NO_RATE_HEAD"
ERR_NO_THRESHOLD = "ERR_NO_THRESHOLD"


# ── Graph model ─────────────────────────────────────────────────────────────


@dataclass
class SsaNode:
    index: int
    role: str
    text: str
    char_range: tuple[int, int]
    value_ref: str | None = None


@dataclass(frozen=True)
class SsaEdge:
    source: int
    target: int
    relation: str


@dataclass
class SsaDiagnostic:
    code: str
    detail: str
    relation: str | None = None


@dataclass
class SsaBuilder:
    """Accumulates the SSA assignments of one clause."""

    clause_idx: int = 0
    currency_code: str | None = None
    steps: list[dict] = field(default_factory=list)
    handlers_used: list[str] = field(default_factory=list)
    inputs: dict[str, dict] = field(default_factory=dict)
    _counter: int = 0

    def name(self, semantic: str) -> str:
        self._counter += 1
        return f"clause_{self.clause_idx}_{semantic}_v{self._counter}"

    def emit(self, semantic: str, expr: Any, handler: str) -> str:
        """Emits one SSA assignment and returns its value reference."""
        name = self.name(semantic)
        self.steps.append({"assign": name, "expr": expr, "handler": handler})
        self.handlers_used.append(handler)
        return f"vars.{name}"

    def declare_input(self, input_id: str) -> str:
        if input_id not in self.inputs:
            ccy = self.currency_code
            if not ccy or ccy == "CURRENCY":
                ccy = "USD"
            self.inputs[input_id] = {
                "type": "MONEY_MINOR",
                "currency": ccy,
                "scale": 100,
                "required": True,
            }
        return f"inputs.{input_id}"


Handler = Callable[["SsaBuilder", SsaNode, list[SsaEdge], "WalkContext"], str | None]


@dataclass
class WalkContext:
    nodes: dict[int, SsaNode]
    edges: list[SsaEdge]
    clause_text: str
    builder: SsaBuilder
    base_ref: str | None = None
    base_name: str = "taxable"
    floor_amount: dict | str | None = None
    ceiling_amount: dict | None = None
    threshold_value: str | None = None
    rate_value: str | None = None


class HandlerRegistry:
    """The RFC-012 5.2 registry: relation -> handler."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, relation: str, handler: Handler) -> None:
        self._handlers[relation] = handler

    def get(self, relation: str) -> Handler | None:
        return self._handlers.get(relation)

    def relations(self) -> list[str]:
        return sorted(self._handlers)


# ── Text cues: the statute decides the sense of an adjustment ──────────────

_REDUCTION_CUES = (
    "reduced by", "reduced", "deducted", "deduct", "less", "minus",
    "credit", "allowance", "discount", "rebate", "subtract", "offset",
)
_CAP_CUES = (
    "capped", "cap", "not exceeding", "shall not exceed", "not to exceed",
    "maximum", "ceiling", "at most", "up to", "limited to",
)

_EXCESS_BASE_RE = re.compile(
    r"(?:excess|in\s+excess)\s+of\s+(?:the\s+|a\s+|an\s+|such\s+)?"
    r"(?P<phrase>[A-Za-z][A-Za-z ]{2,40}?)\s+"
    r"(?:over|above|exceeding|more\s+than|in\s+excess\s+of)\b",
    re.IGNORECASE,
)
_TRAILING_BASE_RE = re.compile(
    r"\b(?:of|on|to)\s+(?:the\s+|a\s+|an\s+|such\s+)?(?P<phrase>[A-Za-z][A-Za-z ]{2,40}?)\s*$",
    re.IGNORECASE,
)
_CONDITION_BASE_RE = re.compile(
    r"\b(?:if|where)\s+(?:the\s+|a\s+|an\s+|such\s+)?(?P<phrase>[A-Za-z][A-Za-z ]{2,40}?)\s+"
    r"(?:is\s+|are\s+)?(?:over|above|exceeding|more\s+than|not\s+over|not\s+exceeding|less\s+than)\b",
    re.IGNORECASE,
)


def relation_sense(text_before: str) -> str:
    """Returns ``"reduction"``, ``"cap"`` or ``""`` for a span context.

    The neural relation labels an adjustment, but the statutory text decides
    whether the adjustment bounds the liability or reduces it.
    """
    window = (text_before or "").lower()
    tail = window[-60:]
    reduction = max((tail.rfind(cue) for cue in _REDUCTION_CUES), default=-1)
    cap = max((tail.rfind(cue) for cue in _CAP_CUES), default=-1)
    if reduction < 0 and cap < 0:
        return ""
    if reduction >= cap:
        return "reduction"
    return "cap"


def resolve_base_phrase(clause_text: str, threshold_node: SsaNode) -> str:
    """Finds the amount a threshold applies to."""
    head = clause_text[: threshold_node.char_range[1]]
    match = _EXCESS_BASE_RE.search(head)
    if match:
        return match.group("phrase").strip()
    match = _TRAILING_BASE_RE.search(head)
    if match:
        return match.group("phrase").strip()
    match = _CONDITION_BASE_RE.search(clause_text)
    if match:
        return match.group("phrase").strip()
    return ""


# ── Handlers ────────────────────────────────────────────────────────────────


_BRACKET_CEILING_RE = re.compile(
    r"\b(?:not\s+over|not\s+exceeding|up\s+to|at\s+most)\s+(?P<ceiling>[$₹€£]\s*[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_BRACKET_BASE_TAX_RE = re.compile(
    r"(?:^|[\s(;])(?P<base_tax>[$₹€£]\s*[\d,]+(?:\.\d+)?)\s*,?\s*\bplus\b",
    re.IGNORECASE,
)


def _static_money(node: SsaNode) -> dict | None:
    if not is_monetary_quantum_text(node.text):
        return None
    minor = parse_minor_units(node.text)
    if minor is None:
        return None
    return {"op": "TO_PAISE", "value": str(minor)}


def _resolve_bracket_ceiling(node: SsaNode | None, clause_text: str) -> dict | None:
    """Resolves the upper monetary ceiling of a bounded bracket clause."""
    if node is not None:
        amount = _static_money(node)
        if amount is not None:
            return amount
    m = _BRACKET_CEILING_RE.search(clause_text or "")
    if m:
        raw = m.group("ceiling")
        if is_monetary_quantum_text(raw):
            minor = parse_minor_units(raw)
            if minor is not None:
                return {"op": "TO_PAISE", "value": str(minor)}
    return None


def _extract_bracket_base_tax(clause_text: str) -> dict | None:
    """Extracts a stated bracket base tax ('$<amount>, plus <rate>%') when present."""
    m = _BRACKET_BASE_TAX_RE.search(clause_text or "")
    if not m:
        return None
    raw = m.group("base_tax")
    if not is_monetary_quantum_text(raw):
        return None
    minor = parse_minor_units(raw)
    if minor is None or minor <= 0:
        return None
    return {"op": "TO_PAISE", "value": str(minor)}


def _handle_bracket_next(
    builder: SsaBuilder, node: SsaNode, neighbors: list[SsaEdge], ctx: WalkContext
) -> str | None:
    """Bounded bracket handler: adds stated baseTax (if any) and gates inside (floor, ceiling]."""
    ceiling = ctx.ceiling_amount or _resolve_bracket_ceiling(node, ctx.clause_text)
    if ceiling is None or ctx.rate_value is None or ctx.base_ref is None or ctx.floor_amount is None:
        return None
    base_tax = _extract_bracket_base_tax(ctx.clause_text)
    if base_tax is not None:
        with_base_expr = {
            "op": "ADD",
            "left": base_tax,
            "right": ctx.rate_value,
        }
        ctx.rate_value = builder.emit(
            f"{ctx.base_name}BracketWithBaseTax", with_base_expr, "bracket-base-tax"
        )
    cond_expr = {
        "op": "CONDITIONAL",
        "test": {
            "op": "AND",
            "left": {
                "op": "GT",
                "left": ctx.base_ref,
                "right": ctx.floor_amount,
            },
            "right": {
                "op": "LTE",
                "left": ctx.base_ref,
                "right": ceiling,
            },
        },
        "then": ctx.rate_value,
        "else": {"op": "TO_PAISE", "value": "0"},
    }
    name = builder.emit(f"{ctx.base_name}BracketTax", cond_expr, "bracket-next")
    ctx.rate_value = name
    return name


def _handle_rate_to_threshold(
    builder: SsaBuilder, node: SsaNode, neighbors: list[SsaEdge], ctx: WalkContext
) -> str | None:
    """Rate on base: the rate applied to the amount above the threshold."""
    bps = parse_rate_bps(node.text)
    if not bps or parse_unit_and_type(node.text)[0] != "PERCENTAGE_RATE":
        return None
    if ctx.threshold_value is None:
        return None
    expr = {
        "op": "RATIONAL_MULTIPLY",
        "base": ctx.threshold_value,
        "rateBps": bps,
        "denominator": 10000,
    }
    name = builder.emit(f"{ctx.base_name}Tax", expr, "rate-on-base")
    ctx.rate_value = name
    return name


def _handle_cap_ceiling(
    builder: SsaBuilder, node: SsaNode, neighbors: list[SsaEdge], ctx: WalkContext
) -> str | None:
    """A ceiling on the liability, unless the text marks it as a reduction."""
    amount = _static_money(node)
    if amount is None:
        return None
    text_before = ctx.clause_text[: node.char_range[0]]
    sense = relation_sense(text_before)
    if sense == "reduction":
        return _apply_reduction(builder, node, amount, ctx)
    if ctx.rate_value is None:
        return None
    expr = {"op": "MIN", "left": ctx.rate_value, "right": amount}
    name = builder.emit(f"{ctx.base_name}TaxCapped", expr, "cap")
    ctx.rate_value = name
    return name


def _apply_reduction(
    builder: SsaBuilder, node: SsaNode, amount: dict, ctx: WalkContext
) -> str | None:
    if ctx.rate_value is None:
        return None
    expr = {"op": "SUB", "left": ctx.rate_value, "right": amount}
    name = builder.emit(f"{ctx.base_name}TaxReduced", expr, "reduction")
    ctx.rate_value = name
    return name


def _handle_subtraction_base(
    builder: SsaBuilder, node: SsaNode, neighbors: list[SsaEdge], ctx: WalkContext
) -> str | None:
    """A subtraction of one amount from another."""
    amount = _static_money(node)
    if amount is None or ctx.rate_value is None:
        return None
    expr = {"op": "SUB", "left": ctx.rate_value, "right": amount}
    name = builder.emit(f"{ctx.base_name}TaxReduced", expr, "reduction")
    ctx.rate_value = name
    return name


def _regime_value(
    node: SsaNode, ctx: WalkContext, builder: SsaBuilder
) -> tuple[str | None, int | None]:
    """The liability of one regime: a rate on a base, or a stated amount."""
    if parse_unit_and_type(node.text)[0] == "PERCENTAGE_RATE":
        bps = parse_rate_bps(node.text)
        if not bps:
            return None, None
        phrase = base_phrase_after_rate(ctx.clause_text, node.char_range[1])
        base_id = input_id_from_phrase(phrase, builder.currency_code)
        if not base_id:
            return None, None
        base_ref = builder.declare_input(base_id)
        ctx.base_name = base_id
        expr = {
            "op": "RATIONAL_MULTIPLY",
            "base": base_ref,
            "rateBps": bps,
            "denominator": 10000,
        }
        return builder.emit(f"{base_id}RegimeTax", expr, "rate-on-base"), None
    amount = _static_money(node)
    if amount is None:
        return None, None
    expr = {"op": "TO_PAISE", "value": amount["value"]}
    return builder.emit("statedRegimeAmount", expr, "stated-amount"), int(amount["value"])


def _handle_combinator(
    builder: SsaBuilder,
    node: SsaNode,
    neighbors: list[SsaEdge],
    ctx: WalkContext,
    combinator: str,
) -> str | None:
    """A multi-regime provision: one regime is chosen out of several."""
    from compiler.neural.regime_combinators import Regime, combine

    edges = [edge for edge in ctx.edges if edge.relation in (
        "REL_COMBINATOR_MIN", "REL_COMBINATOR_MAX"
    )]
    if not edges:
        return None
    edge = edges[0]
    regimes = []
    for endpoint in (edge.source, edge.target):
        regime_node = ctx.nodes.get(endpoint)
        if regime_node is None:
            return None
        ref, value = _regime_value(regime_node, ctx, builder)
        if ref is None:
            return None
        regimes.append(Regime(id=f"regime_{endpoint}", value=value, value_ref=ref))
    result = combine(combinator, regimes)
    if not result.ok or result.expression is None:
        return None
    return builder.emit(
        f"{ctx.base_name}Combined",
        result.expression,
        f"combinator-{combinator.lower()}",
    )


# ── Dependency direction and cycle detection ────────────────────────────────

#: For each relation, which endpoint depends on which.
DEPENDENCY_DIRECTION: dict[str, str] = {
    "REL_RATE_TO_THRESHOLD": "source_depends_on_target",
    "REL_CAP_CEILING": "target_depends_on_source",
    "REL_SUBTRACTION_BASE": "target_depends_on_source",
    "REL_THRESHOLD_TO_TRANSACTION": "target_depends_on_source",
    "REL_COMBINATOR_MIN": "target_depends_on_source",
    "REL_COMBINATOR_MAX": "target_depends_on_source",
    "REL_BRACKET_NEXT": "target_depends_on_source",
    "REL_AGGREGATE_SUM": "target_depends_on_source",
    "REL_LOGICAL_AND": "target_depends_on_source",
    "REL_LOGICAL_OR": "target_depends_on_source",
    "REL_LOGICAL_NOT": "target_depends_on_source",
}


def build_dependency_graph(
    edges: list[SsaEdge],
) -> dict[int, set[int]]:
    """Returns node -> nodes it depends on."""
    graph: dict[int, set[int]] = {}
    for edge in edges:
        direction = DEPENDENCY_DIRECTION.get(edge.relation)
        if direction == "source_depends_on_target":
            graph.setdefault(edge.source, set()).add(edge.target)
        elif direction == "target_depends_on_source":
            graph.setdefault(edge.target, set()).add(edge.source)
    return graph


def find_dependency_cycle(graph: dict[int, set[int]]) -> list[int] | None:
    """Returns one cycle in the dependency graph, if any."""
    state: dict[int, int] = {}
    stack: list[int] = []

    def visit(node: int) -> list[int] | None:
        state[node] = 1
        stack.append(node)
        for neighbour in sorted(graph.get(node, ())):
            if state.get(neighbour) == 1:
                start = stack.index(neighbour)
                return stack[start:] + [neighbour]
            if state.get(neighbour, 0) == 0:
                found = visit(neighbour)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if state.get(node, 0) == 0:
            found = visit(node)
            if found:
                return found
    return None


def topological_order(
    nodes: list[SsaNode], graph: dict[int, set[int]]
) -> list[int]:
    """A deterministic topological order (ties break on the semantic name)."""
    incoming: dict[int, set[int]] = {node.index: set() for node in nodes}
    for node, dependencies in graph.items():
        incoming.setdefault(node, set())
        for dependency in dependencies:
            incoming.setdefault(dependency, set())
            incoming[node].add(dependency)
    by_index = {node.index: node for node in nodes}
    ready = [index for index, deps in incoming.items() if not deps]
    order: list[int] = []
    while ready:
        ready.sort(key=lambda i: (by_index[i].text if i in by_index else "", i))
        current = ready.pop(0)
        order.append(current)
        for node, deps in list(incoming.items()):
            if current in deps:
                deps.discard(current)
                if not deps and node not in order and node not in ready:
                    ready.append(node)
    return order


# ── The walk ────────────────────────────────────────────────────────────────


def _handle_combinator_min(
    builder: SsaBuilder, node: SsaNode, neighbors: list[SsaEdge], ctx: WalkContext
) -> str | None:
    """The beneficial regime of a multi-regime provision."""
    return _handle_combinator(builder, node, neighbors, ctx, "BENEFICIAL_MIN")


def _handle_combinator_max(
    builder: SsaBuilder, node: SsaNode, neighbors: list[SsaEdge], ctx: WalkContext
) -> str | None:
    """The penal regime of a multi-regime provision."""
    return _handle_combinator(builder, node, neighbors, ctx, "PENAL_MAX")


def _combinator_base_name(
    by_index: dict[int, SsaNode], edge: SsaEdge, clause_text: str
) -> str:
    """The name of the amount a multi-regime provision applies to."""
    for endpoint in (edge.source, edge.target):
        node = by_index.get(endpoint)
        if node is None:
            continue
        if parse_unit_and_type(node.text)[0] != "PERCENTAGE_RATE":
            continue
        phrase = base_phrase_after_rate(clause_text, node.char_range[1])
        base_id = input_id_from_phrase(phrase)
        if base_id:
            return base_id
    return ""


def default_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register("REL_RATE_TO_THRESHOLD", _handle_rate_to_threshold)
    registry.register("REL_BRACKET_NEXT", _handle_bracket_next)
    registry.register("REL_CAP_CEILING", _handle_cap_ceiling)
    registry.register("REL_SUBTRACTION_BASE", _handle_subtraction_base)
    registry.register("REL_COMBINATOR_MIN", _handle_combinator_min)
    registry.register("REL_COMBINATOR_MAX", _handle_combinator_max)
    return registry


def compose_clause(
    nodes: list[SsaNode],
    edges: list[SsaEdge],
    clause_text: str,
    clause_idx: int = 0,
    currency_code: str | None = None,
    inputs: dict[str, dict] | None = None,
    registry: HandlerRegistry | None = None,
    inherited_base_id: str | None = None,
    candidate_bases: list[str] | None = None,
    known_symbols: dict[str, str] | None = None,
) -> tuple[list[dict], list[str], list[SsaDiagnostic]]:
    """Composes one clause into an ordered list of SSA assignments.

    Returns ``(steps, handlers_used, diagnostics)``. An empty step list with
    an error diagnostic means the clause was not composed; the caller fails
    closed rather than guessing.
    """
    registry = registry or default_registry()
    builder = SsaBuilder(
        clause_idx=clause_idx,
        currency_code=currency_code,
        inputs=inputs if inputs is not None else {},
    )
    diagnostics: list[SsaDiagnostic] = []
    by_index = {node.index: node for node in nodes}

    # 1. every relation must have a handler
    for edge in edges:
        if registry.get(edge.relation) is None:
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_UNHANDLED_RELATION,
                    detail=(
                        f"relation {edge.relation} has no registered handler; "
                        "the clause cannot be composed"
                    ),
                    relation=edge.relation,
                )
            )
    if diagnostics:
        return [], [], diagnostics

    # 2. the dependency graph must be acyclic
    graph = build_dependency_graph(edges)
    cycle = find_dependency_cycle(graph)
    if cycle:
        diagnostics.append(
            SsaDiagnostic(
                code=ERR_CYCLIC_DEPENDENCY,
                detail="cyclic dependency: " + " -> ".join(str(i) for i in cycle),
            )
        )
        return [], [], diagnostics

    # 3. a deterministic visit order (data dependency, ties by name)
    topological_order(nodes, graph)

    # 4. the head of the clause
    combinator_edges = [
        edge for edge in edges
        if edge.relation in ("REL_COMBINATOR_MIN", "REL_COMBINATOR_MAX")
    ]
    if combinator_edges:
        # A multi-regime provision: the regimes are composed, then combined.
        ctx = WalkContext(
            nodes=by_index,
            edges=edges,
            clause_text=clause_text,
            builder=builder,
        )
        base_name = _combinator_base_name(by_index, combinator_edges[0], clause_text)
        if base_name:
            ctx.base_name = base_name
        handler = registry.get(combinator_edges[0].relation)
        combined = handler(builder, by_index[combinator_edges[0].source], [], ctx)
        if combined is None:
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_RATE_HEAD,
                    detail="the multi-regime provision could not be composed",
                )
            )
            return [], [], diagnostics
        builder.emit(
            f"{ctx.base_name}CombinedPaisa",
            {"op": "MAX", "left": combined, "right": {"op": "TO_PAISE", "value": "0"}},
            "floor-at-zero",
        )
        return builder.steps, builder.handlers_used, diagnostics

    rate_node = None
    threshold_node = None
    for edge in edges:
        if edge.relation != "REL_RATE_TO_THRESHOLD":
            continue
        source = by_index.get(edge.source)
        target = by_index.get(edge.target)
        if (
            source is not None
            and parse_rate_bps(source.text) is not None
            and parse_unit_and_type(source.text)[0] == "PERCENTAGE_RATE"
        ):
            rate_node = source
            if target is not None:
                is_money = _static_money(target) is not None
                is_known_sym = False
                if known_symbols and target.text:
                    norm_t = normalize_term(target.text)
                    is_known_sym = any(k in norm_t or norm_t in k for k in known_symbols)
                if is_money or is_known_sym:
                    threshold_node = target
            break
        elif target is not None:
            is_money = _static_money(target) is not None
            is_known_sym = False
            if known_symbols and target.text:
                norm_t = normalize_term(target.text)
                is_known_sym = any(k in norm_t or norm_t in k for k in known_symbols)
            if is_money or is_known_sym:
                threshold_node = target

    if rate_node is None and threshold_node is not None:
        for node in nodes:
            if (
                parse_rate_bps(node.text) is not None
                and parse_unit_and_type(node.text)[0] == "PERCENTAGE_RATE"
            ):
                rate_node = node
                break

    if rate_node is None:
        for node in nodes:
            if (
                parse_rate_bps(node.text) is not None
                and parse_unit_and_type(node.text)[0] == "PERCENTAGE_RATE"
            ):
                rate_node = node
                break

    if threshold_node is None:
        for node in nodes:
            if node.role == "ROLE_TRANSACTION_EXCESS_THRESHOLD":
                is_money = _static_money(node) is not None
                is_known_sym = False
                if known_symbols and node.text:
                    norm_t = normalize_term(node.text)
                    is_known_sym = any(k in norm_t or norm_t in k for k in known_symbols)
                if is_money or is_known_sym:
                    threshold_node = node
                    break

    if rate_node is None:
        diagnostics.append(
            SsaDiagnostic(
                code=ERR_NO_RATE_HEAD,
                detail="the clause has no rate to compose",
            )
        )
        return [], [], diagnostics

    # 4b. Flat rate: statutory rate without a threshold (e.g. IRC § 68(a)(2) or § 3301)
    if rate_node is not None and threshold_node is None:
        # Guard against silent degradation of an excess-over-threshold clause into a flat-rate clause.
        # If the clause contains excess or comparative cues (e.g. § 68(a)(1) "excess of AGI over applicable amount"),
        # dropping the threshold would falsely compute tax/reduction on the entire base.
        if re.search(
            r"\b(?:excess\s+(?:of)?|in\s+excess\s+of|over\s+the|over\s+such|exceeding|exceeds|more\s+than|greater\s+than)\b",
            clause_text,
            re.IGNORECASE,
        ):
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_THRESHOLD,
                    detail="clause specifies an excess calculation over a threshold, but no valid threshold node or symbol could be resolved",
                )
            )
            return [], [], diagnostics

        base_phrase = ""
        for node in nodes:
            if node.role == "ROLE_COMPUTATION_OPERAND":
                base_phrase = node.text
                break
        if not base_phrase:
            rate_pos = clause_text.find(rate_node.text)
            rate_end = (rate_pos + len(rate_node.text)) if rate_pos >= 0 else rate_node.char_range[1]
            base_phrase = base_phrase_after_rate(clause_text, rate_end)
        base_id = input_id_from_phrase(base_phrase, currency_code)
        # Inherited base is only for clauses with no distinct local base phrase.
        # If a clause explicitly names a base phrase that cannot be resolved (or refers to another paragraph/dollar amount),
        # it must NOT blindly fall back to an arbitrary inherited income base (e.g. § 63 200% multiplying grossIncome).
        if not base_id and not base_phrase and inherited_base_id:
            base_id = inherited_base_id
        if not base_id:
            if candidate_bases and len(candidate_bases) > 1:
                diagnostics.append(
                    SsaDiagnostic(
                        code=ERR_AMBIGUOUS_REFERENCE,
                        detail=f"clause has ambiguous candidate base amounts: {candidate_bases}",
                    )
                )
                return [], [], diagnostics
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_BASE_OPERAND,
                    detail=f"the clause has no resolvable base amount ({base_phrase!r})",
                )
            )
            return [], [], diagnostics

        ctx = WalkContext(
            nodes=by_index,
            edges=edges,
            clause_text=clause_text,
            builder=builder,
        )
        ctx.base_ref = builder.declare_input(base_id)
        ctx.base_name = base_id

        bps = parse_rate_bps(rate_node.text)
        if not bps:
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_RATE_HEAD,
                    detail="the rate of the clause could not be resolved",
                )
            )
            return [], [], diagnostics

        expr = {
            "op": "RATIONAL_MULTIPLY",
            "base": ctx.base_ref,
            "rateBps": bps,
            "denominator": 10000,
        }
        rate_val = builder.emit(f"{ctx.base_name}Tax", expr, "flat-rate")
        ctx.rate_value = rate_val

        applied = [
            edge for edge in edges
            if edge.relation in ("REL_CAP_CEILING", "REL_SUBTRACTION_BASE")
            and edge.target == rate_node.index
        ]
        reductions = []
        caps = []
        for edge in applied:
            node = by_index.get(edge.source)
            if node is None:
                continue
            text_before = clause_text[: node.char_range[0]]
            sense = relation_sense(text_before)
            if sense == "reduction":
                reductions.append(node)
            else:
                caps.append(node)
        reductions.sort(key=lambda n: n.char_range[0])
        caps.sort(key=lambda n: n.char_range[0])

        for node in reductions:
            handler = registry.get("REL_CAP_CEILING")
            if handler is not None:
                handler(builder, node, [], ctx)
            else:
                _handle_subtraction_base(builder, node, [], ctx)
        for node in caps:
            handler = registry.get("REL_CAP_CEILING")
            if handler is None:
                continue
            handler(builder, node, [], ctx)

        builder.emit(
            f"{ctx.base_name}TaxPaisa",
            {"op": "MAX", "left": ctx.rate_value, "right": {"op": "TO_PAISE", "value": "0"}},
            "floor-at-zero",
        )
        return builder.steps, builder.handlers_used, diagnostics

    # 5. the base the threshold applies to
    base_phrase = ""
    for node in nodes:
        if node.role == "ROLE_COMPUTATION_OPERAND":
            base_phrase = node.text
            break
    if not base_phrase:
        base_phrase = resolve_base_phrase(clause_text, threshold_node)
    base_id = input_id_from_phrase(base_phrase, currency_code)
    if not base_id and not base_phrase and inherited_base_id:
        base_id = inherited_base_id
    if not base_id:
        if candidate_bases and len(candidate_bases) > 1:
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_AMBIGUOUS_REFERENCE,
                    detail=f"clause has ambiguous candidate base amounts: {candidate_bases}",
                )
            )
            return [], [], diagnostics
        diagnostics.append(
            SsaDiagnostic(
                code=ERR_NO_BASE_OPERAND,
                detail=f"the clause has no resolvable base amount ({base_phrase!r})",
            )
        )
        return [], [], diagnostics

    ctx = WalkContext(
        nodes=by_index,
        edges=edges,
        clause_text=clause_text,
        builder=builder,
    )
    ctx.base_ref = builder.declare_input(base_id)
    ctx.base_name = base_id

    # 6. base arithmetic: the amount above the threshold (bounded by ceiling if REL_BRACKET_NEXT)
    threshold_amount = _static_money(threshold_node)
    if threshold_amount is None and known_symbols:
        norm_th = normalize_term(threshold_node.text)
        for sym_k, sym_ref in known_symbols.items():
            if sym_k in norm_th or norm_th in sym_k:
                threshold_amount = sym_ref
                break
    if threshold_amount is None:
        diagnostics.append(
            SsaDiagnostic(
                code=ERR_NO_THRESHOLD,
                detail=f"the threshold {threshold_node.text!r} is not a monetary amount",
            )
        )
        return [], [], diagnostics
    ctx.floor_amount = threshold_amount

    bracket_edges = [edge for edge in edges if edge.relation == "REL_BRACKET_NEXT"]
    ceiling_node = None
    if bracket_edges:
        for b_edge in bracket_edges:
            candidate_node = by_index.get(b_edge.target)
            resolved_c = _resolve_bracket_ceiling(candidate_node, clause_text)
            if resolved_c is not None:
                ceiling_node = candidate_node
                ctx.ceiling_amount = resolved_c
                break
        if ctx.ceiling_amount is None:
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_THRESHOLD,
                    detail="clause has REL_BRACKET_NEXT but no valid monetary ceiling could be resolved",
                )
            )
            return [], [], diagnostics
        if (
            isinstance(threshold_amount, dict)
            and threshold_amount.get("op") == "TO_PAISE"
            and int(ctx.ceiling_amount["value"]) <= int(threshold_amount["value"])
        ):
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_THRESHOLD,
                    detail="bracket ceiling must be strictly greater than the floor threshold",
                )
            )
            return [], [], diagnostics
        capped_base_expr = {
            "op": "MIN",
            "left": ctx.base_ref,
            "right": ctx.ceiling_amount,
        }
        excess_expr = {
            "op": "MAX",
            "left": {"op": "TO_PAISE", "value": "0"},
            "right": {"op": "SUB", "left": capped_base_expr, "right": threshold_amount},
        }
    else:
        excess_expr = {
            "op": "MAX",
            "left": {"op": "TO_PAISE", "value": "0"},
            "right": {"op": "SUB", "left": ctx.base_ref, "right": threshold_amount},
        }
    ctx.threshold_value = builder.emit(
        f"{ctx.base_name}Excess", excess_expr, "base-arithmetic"
    )

    # 7. the rate applied to the excess
    rate_handler = registry.get("REL_RATE_TO_THRESHOLD")
    rate_value = rate_handler(builder, rate_node, [], ctx)
    if rate_value is None:
        diagnostics.append(
            SsaDiagnostic(
                code=ERR_NO_RATE_HEAD,
                detail="the rate of the clause could not be resolved",
            )
        )
        return [], [], diagnostics

    # 7b. bounded bracket ceiling + baseTax (REL_BRACKET_NEXT) or open-ended top bracket baseTax
    if bracket_edges:
        bracket_handler = registry.get("REL_BRACKET_NEXT")
        if bracket_handler is None or bracket_handler(builder, ceiling_node, bracket_edges, ctx) is None:
            diagnostics.append(
                SsaDiagnostic(
                    code=ERR_NO_THRESHOLD,
                    detail="the bounded bracket clause could not be composed",
                )
            )
            return [], [], diagnostics
    else:
        base_tax = _extract_bracket_base_tax(clause_text)
        if base_tax is not None:
            with_base_expr = {
                "op": "ADD",
                "left": base_tax,
                "right": ctx.rate_value,
            }
            with_base_name = builder.emit(
                f"{ctx.base_name}BracketWithBaseTax", with_base_expr, "bracket-base-tax"
            )
            cond_expr = {
                "op": "CONDITIONAL",
                "test": {
                    "op": "GT",
                    "left": ctx.base_ref,
                    "right": threshold_amount,
                },
                "then": with_base_name,
                "else": {"op": "TO_PAISE", "value": "0"},
            }
            ctx.rate_value = builder.emit(
                f"{ctx.base_name}BracketTax", cond_expr, "bracket-open"
            )

    # 8. the adjustments that apply to the rate, in statutory order:
    #    reductions first, then caps, each ordered by its position in the text.
    applied = [
        edge for edge in edges
        if edge.relation in ("REL_CAP_CEILING", "REL_SUBTRACTION_BASE")
        and edge.target == rate_node.index
    ]
    reductions = []
    caps = []
    for edge in applied:
        node = by_index.get(edge.source)
        if node is None:
            continue
        text_before = clause_text[: node.char_range[0]]
        sense = relation_sense(text_before)
        if sense == "reduction":
            reductions.append(node)
        else:
            caps.append(node)
    reductions.sort(key=lambda n: n.char_range[0])
    caps.sort(key=lambda n: n.char_range[0])

    for node in reductions:
        handler = registry.get("REL_CAP_CEILING")
        if handler is not None:
            handler(builder, node, [], ctx)
        else:
            _handle_subtraction_base(builder, node, [], ctx)
    for node in caps:
        handler = registry.get("REL_CAP_CEILING")
        if handler is None:
            continue
        handler(builder, node, [], ctx)

    # 9. floor at zero
    builder.emit(
        f"{ctx.base_name}TaxPaisa",
        {"op": "MAX", "left": ctx.rate_value, "right": {"op": "TO_PAISE", "value": "0"}},
        "floor-at-zero",
    )

    return builder.steps, builder.handlers_used, diagnostics


# ============================================================================
# Model v2 Regex-Free Typed-Opcode Topological SSA Walker
# ============================================================================

from .literals import (
    derive_logic_fact_key as _v2_derive_logic_key,
    infer_rule_currency_info as _v2_infer_currency,
    parse_comparison_operator as _v2_parse_cmp_op,
    parse_days_window as _v2_parse_days_window,
    parse_iso_date_str as _v2_parse_iso_date,
    parse_minor_units as _v2_parse_minor_units,
    parse_numeric_quantity as _v2_parse_num,
    parse_rate_bps as _v2_parse_rate_bps,
    pluralize_collection_noun as _v2_pluralize_noun,
    _derive_fact_key_camel as _v2_fact_key_camel,
)


class V2LoweringError(ValueError):
    """Fail-closed diagnostic error raised during v2 topological SSA lowering."""


V2_EDGE_DIRECTION: dict[str, str] = {
    **DEPENDENCY_DIRECTION,
    "REL_MIN_CAP": "source_before_target",
    "REL_SUB_CREDIT": "source_before_target",
    "REL_AGGREGATE_COUNT": "target_before_source",
    "REL_BASE_TAX_ADD": "source_before_target",
    "REL_DEFINED_TERM_OF": "target_before_source",
    "REL_SUBSTITUTE_WHEN": "target_before_source",
    "REL_COREF_LINK": "target_before_source",
}


def _v2_norm_edge(edge: dict, valid_indices: set[int]) -> tuple[int, int, str] | None:
    s = edge.get("sourceIdx")
    if s is None:
        s = edge.get("source")
    t = edge.get("targetIdx")
    if t is None:
        t = edge.get("target")
    rel = edge.get("relationType") or edge.get("relation") or "REL_NO_EDGE"
    if s is None or t is None or rel == "REL_NO_EDGE":
        return None
    if int(s) not in valid_indices or int(t) not in valid_indices:
        return None
    return int(s), int(t), str(rel)


def _v2_derive_duration_key(raw_text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (raw_text or "").lower())
    drop = {
        "after", "before", "within", "under", "below", "above", "over", "less", "more", "than",
        "exceeding", "exceeds", "exceed", "not", "no", "at", "least", "most", "up", "to", "prior",
        "subsequent", "earlier", "later", "for", "of", "the", "a", "an", "days", "day", "months",
        "month", "years", "year", "weeks", "week", "hours", "hour", "taxable",
    }
    words = [w for w in cleaned.split() if not w.isdigit() and w not in drop and len(w) > 1]
    if not words:
        return "durationDays"
    chosen = words[:3]
    camel = chosen[0] + "".join(w.capitalize() for w in chosen[1:])
    return camel if camel.endswith("Days") else f"{camel}Days"


def _v2_parse_span_value(span_dict: dict) -> dict:
    """Extracts typed literal value purely from neural `unit` + deterministic literal parser."""
    text = str(span_dict.get("span") or "").strip()
    unit = str(span_dict.get("unit") or "UNIT_NONE")
    role = str(span_dict.get("role") or "ROLE_IGNORE")

    # Reject non-monetary spans strictly by neural unit class before touching money parsers
    if unit == "UNIT_MONEY":
        minor = _v2_parse_minor_units(text)
        return {"kind": "MONEY", "minor": minor, "cmpOp": _v2_parse_cmp_op(text), "text": text, "unit": unit, "role": role}
    if unit == "UNIT_RATE":
        bps = _v2_parse_rate_bps(text)
        return {"kind": "RATE", "rateBps": bps, "cmpOp": _v2_parse_cmp_op(text), "text": text, "unit": unit, "role": role}
    if unit == "UNIT_DURATION":
        qty = _v2_parse_num(text)
        days = _v2_parse_days_window(text)
        return {
            "kind": "DURATION",
            "num": qty,
            "days": days,
            "cmpOp": _v2_parse_cmp_op(text),
            "key": _v2_derive_duration_key(text),
            "text": text,
            "unit": unit,
            "role": role,
        }
    if unit == "UNIT_YEAR_DATE":
        iso_date = _v2_parse_iso_date(text)
        years = []
        for tok in text.replace(",", " ").replace(".", " ").split():
            if tok.isdigit() and len(tok) == 4 and 1800 <= int(tok) <= 2200:
                years.append(int(tok))
        return {
            "kind": "YEAR_DATE",
            "isoDate": iso_date,
            "years": years,
            "cmpOp": _v2_parse_cmp_op(text),
            "text": text,
            "unit": unit,
            "role": role,
        }
    if unit == "UNIT_COUNT":
        qty = _v2_parse_num(text)
        return {"kind": "COUNT", "num": int(round(float(qty))) if qty is not None else None, "text": text, "unit": unit, "role": role}
    if unit == "UNIT_BOOL_PREDICATE":
        return {"kind": "BOOL", "key": _v2_derive_logic_key(text), "text": text, "unit": unit, "role": role}
    if unit == "UNIT_SYMBOL_VAR":
        return {"kind": "SYMBOL", "key": _v2_fact_key_camel(text, "metricValue"), "text": text, "unit": unit, "role": role}
    return {"kind": "NONE", "text": text, "unit": unit, "role": role}


def _v2_build_clause_dag(
    span_indices: list[int],
    norm_edges: list[tuple[int, int, str]],
) -> dict[int, set[int]]:
    """Constructs directed dependency adjacency (node -> prerequisites) and verifies acyclicity."""
    adj: dict[int, set[int]] = {idx: set() for idx in span_indices}
    for s, t, rel in norm_edges:
        direction = V2_EDGE_DIRECTION.get(rel, "independent")
        if direction == "independent":
            continue
        if s == t:
            adj[s].add(t)
        elif direction == "target_before_source":
            adj[s].add(t)
        elif direction == "source_before_target":
            adj[t].add(s)
    return adj


def _v2_topo_sort_indices(adj: dict[int, set[int]]) -> list[int]:
    """Kahn bottom-up topological sort (prerequisites first); raises ERR_CYCLIC_DEPENDENCY on cycles."""
    in_deg = {node: len(deps) for node, deps in adj.items()}
    dependents: dict[int, list[int]] = {node: [] for node in adj}
    for node, deps in adj.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].append(node)
    ready = sorted([node for node, d in in_deg.items() if d == 0])
    ordered: list[int] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in sorted(dependents.get(cur, [])):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(ordered) != len(adj):
        raise V2LoweringError("ERR_CYCLIC_DEPENDENCY: cyclic dependency detected in v2 relation graph")
    return ordered


def compose_clause_v2(
    clause_idx: int,
    clause_spans: list[dict],
    all_edges: list[dict],
    full_text: str,
    currency_code: str | None = None,
    symbol_env: dict[str, dict] | None = None,
    resolved_corefs: dict[int, dict] | None = None,
    fallback_base_var: str | None = None,
    clause_modality: str | None = None,
) -> dict:
    """Regex-free bottom-up topological SSA composer for a single v2 clause.

    Reads predicted `SPAN_UNITS` + exact `V2_RELATION_TYPES` directly:
      - No `_BRACKET_BASE_TAX_RE`, no `relation_sense()` regex, no heuristic
        `is_monetary_quantum_text` guessing.
      - Actually executes `topological_order` bottom-up over computation nodes.
    """
    symbol_env = symbol_env or {}
    resolved_corefs = resolved_corefs or {}
    ccy = currency_code or _v2_infer_currency(clause_spans, full_text)[0] or "USD"

    sorted_spans = sorted(
        clause_spans,
        key=lambda sp: (
            int((sp.get("charRange") or [0, 0])[0]),
            int((sp.get("charRange") or [0, 0])[1]),
            str(sp.get("role") or ""),
            str(sp.get("unit") or ""),
        ),
    )
    if not sorted_spans:
        raise V2LoweringError("ERR_NO_COMPUTATION: clause has zero spans")

    all_ids = {int(sp.get("spanIdx", pos)) for pos, sp in enumerate(sorted_spans)}
    ignored_ids = {
        int(sp.get("spanIdx", pos))
        for pos, sp in enumerate(sorted_spans)
        if sp.get("role") == "ROLE_IGNORE"
    }
    for e in all_edges:
        ne_all = _v2_norm_edge(e, all_ids)
        if ne_all is not None and (ne_all[0] in ignored_ids or ne_all[1] in ignored_ids):
            raise V2LoweringError(
                f"ERR_EDGE_TO_IGNORED_SPAN: edge {ne_all[2]} touches ROLE_IGNORE span {ne_all[0]}->{ne_all[1]}"
            )

    active_spans = [
        sp for pos, sp in enumerate(sorted_spans)
        if int(sp.get("spanIdx", pos)) not in ignored_ids
    ]
    if not active_spans:
        raise V2LoweringError("ERR_NO_COMPUTATION: all clause spans have ROLE_IGNORE")

    span_by_idx: dict[int, dict] = {}
    parsed_by_idx: dict[int, dict] = {}
    for pos, sp in enumerate(active_spans):
        sid = int(sp.get("spanIdx", pos))
        span_by_idx[sid] = sp
        pv = _v2_parse_span_value(sp)
        if pv["kind"] == "RATE" and pv["rateBps"] is None:
            raise V2LoweringError(f"ERR_UNPARSEABLE_RATE: span '{pv['text']}' has UNIT_RATE without numeric value")
        if pv["kind"] == "MONEY" and pv["minor"] is None:
            if pv["role"] in (
                "ROLE_COMPUTATION_OPERAND",
                "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE",
                "ROLE_TAX_BASE",
                "ROLE_SUBJECT_PRINCIPAL",
            ):
                pv = {
                    "kind": "SYMBOL",
                    "key": _v2_fact_key_camel(pv["text"], "metricValue"),
                    "text": pv["text"],
                    "unit": pv["unit"],
                    "role": pv["role"],
                }
            else:
                raise V2LoweringError(f"ERR_UNPARSEABLE_MONEY: span '{pv['text']}' has UNIT_MONEY without monetary value")
        if pv["kind"] == "DURATION" and pv["days"] is None and pv["num"] is None:
            raise V2LoweringError(f"ERR_UNPARSEABLE_DURATION: span '{pv['text']}' has UNIT_DURATION without duration")
        if pv["kind"] == "YEAR_DATE" and pv["isoDate"] is None and not pv["years"]:
            raise V2LoweringError(f"ERR_UNPARSEABLE_DATE: span '{pv['text']}' has UNIT_YEAR_DATE without date/year")
        if pv["kind"] == "COUNT" and pv["num"] is None:
            raise V2LoweringError(f"ERR_UNPARSEABLE_COUNT: span '{pv['text']}' has UNIT_COUNT without integer")
        parsed_by_idx[sid] = pv

    valid_ids = set(span_by_idx.keys())
    in_clause_edges: list[tuple[int, int, str]] = []
    seen_e = set()
    for e in all_edges:
        ne = _v2_norm_edge(e, valid_ids)
        if ne is not None and ne not in seen_e:
            seen_e.add(ne)
            in_clause_edges.append(ne)
    in_clause_edges.sort(key=lambda item: (item[2], item[0], item[1]))

    adj = _v2_build_clause_dag(sorted(valid_ids), in_clause_edges)
    topo_ids = _v2_topo_sort_indices(adj)
    consumed_ids: set[int] = set()
    consumed_edges: set[tuple[int, int, str]] = set()

    incoming: dict[int, list[tuple[int, str]]] = {i: [] for i in valid_ids}
    outgoing: dict[int, list[tuple[int, str]]] = {i: [] for i in valid_ids}
    for s, t, rel in in_clause_edges:
        outgoing[s].append((t, rel))
        incoming[t].append((s, rel))

    steps: list[dict] = []
    handlers: list[str] = []
    inputs: dict[str, dict] = {}
    ssa_ver: dict[str, int] = {}

    def _emit(base_name: str, expr: dict, handler: str) -> str:
        ver = ssa_ver.get(base_name, 0) + 1
        ssa_ver[base_name] = ver
        var_name = f"clause_{clause_idx}_{base_name}_v{ver}"
        steps.append({"assign": var_name, "expr": expr})
        handlers.append(handler)
        return var_name

    threshold_role_or_targets = {
        t for _, t, rel in in_clause_edges if rel in ("REL_RATE_TO_THRESHOLD", "REL_SUBTRACTION_BASE")
    } | {
        sid for sid in topo_ids if span_by_idx[sid].get("role") == "ROLE_TRANSACTION_EXCESS_THRESHOLD"
    }

    # Resolve operand base variable deterministically (ROLE_COMPUTATION_OPERAND > coref > symbol_env > fallback)
    def _resolve_operand_expr(pref_id: int | None = None) -> tuple[str, str]:
        candidates = ([pref_id] if pref_id is not None else []) + topo_ids
        # Pass 1: explicitly tagged ROLE_COMPUTATION_OPERAND (never a threshold/excess reference)
        for sid in candidates:
            if sid in threshold_role_or_targets:
                continue
            sp = span_by_idx[sid]
            if sp.get("role") == "ROLE_COMPUTATION_OPERAND":
                if sid in resolved_corefs:
                    ref = resolved_corefs[sid]
                    return str(ref["refVar"]), str(ref["baseKey"])
                key = _v2_fact_key_camel(parsed_by_idx[sid]["text"], fallback_base_var or "amount")
                if key in symbol_env:
                    return symbol_env[key]["ref"], key
                inputs.setdefault(key, {"type": "MONEY_MINOR", "currency": ccy})
                return f"inputs.{key}", key
        # Pass 2: other non-threshold SYMBOL or resolved coref spans
        for sid in candidates:
            if sid in threshold_role_or_targets:
                continue
            if sid in resolved_corefs:
                ref = resolved_corefs[sid]
                return str(ref["refVar"]), str(ref["baseKey"])
            p = parsed_by_idx[sid]
            if p["kind"] == "SYMBOL":
                key = _v2_fact_key_camel(p["text"], fallback_base_var or "amount")
                if key in symbol_env:
                    return symbol_env[key]["ref"], key
                inputs.setdefault(key, {"type": "MONEY_MINOR", "currency": ccy})
                return f"inputs.{key}", key
        base_key = fallback_base_var or "amount"
        inputs.setdefault(base_key, {"type": "MONEY_MINOR", "currency": ccy})
        return f"inputs.{base_key}", base_key

    # Partition clause semantics strictly by predicted units & v2 edge opcodes (NO regex heuristics)
    fold_edges = [e for e in in_clause_edges if e[2] in ("REL_AGGREGATE_SUM", "REL_AGGREGATE_COUNT")]
    quant_edges = [e for e in in_clause_edges if e[2] in ("REL_QUANTIFIER_FORALL", "REL_QUANTIFIER_EXISTS")]
    rate_nodes = [sid for sid in topo_ids if parsed_by_idx[sid]["kind"] == "RATE" and parsed_by_idx[sid]["rateBps"] is not None]
    money_nodes = [sid for sid in topo_ids if parsed_by_idx[sid]["kind"] == "MONEY" and parsed_by_idx[sid]["minor"] is not None]
    bool_nodes = [sid for sid in topo_ids if parsed_by_idx[sid]["kind"] == "BOOL"]

    active_edge_endpoints = {s for s, _, _ in in_clause_edges} | {t for _, t, _ in in_clause_edges}
    cond_temporal_roles = {"ROLE_DEADLINE_WINDOW", "ROLE_TEMPORAL_CUTOVER_DATE", "ROLE_TEMPORAL_SUNSET_DATE", "ROLE_PRESCRIPTION_DEADLINE"}
    dur_cond_nodes = [
        sid for sid in topo_ids
        if parsed_by_idx[sid]["kind"] == "DURATION"
        and (sid in active_edge_endpoints or span_by_idx[sid].get("role") in cond_temporal_roles)
    ]
    date_cond_nodes = [
        sid for sid in topo_ids
        if parsed_by_idx[sid]["kind"] == "YEAR_DATE"
        and (sid in active_edge_endpoints or span_by_idx[sid].get("role") in cond_temporal_roles)
    ]

    must_consume_ids = {
        sid for sid, pv in parsed_by_idx.items()
        if pv["kind"] in ("RATE", "MONEY", "BOOL")
        or (
            pv["kind"] in ("DURATION", "YEAR_DATE", "COUNT")
            and (sid in active_edge_endpoints or span_by_idx[sid].get("role") in cond_temporal_roles)
        )
    }

    def _verify_affine_consumption() -> None:
        leftover = sorted(must_consume_ids - consumed_ids)
        if leftover:
            details = [f"{u}:{parsed_by_idx[u]['unit']}({parsed_by_idx[u]['text']!r})" for u in leftover]
            raise V2LoweringError(f"ERR_UNCONSUMED_SPAN: typed span(s) not lowered into AST: {', '.join(details)}")

    # Track node SSA bindings built bottom-up during topological traversal
    node_ssa: dict[int, str] = {}
    fold_var: str | None = None
    fold_key: str | None = None
    fold_type: str | None = None

    # 1. Bounded Collection Fold Path (`SUM_OVER` / `COUNT`) — composes downstream into arithmetic if present
    if fold_edges:
        s_id, t_id, fold_rel = fold_edges[0]
        consumed_ids.update((s_id, t_id))
        consumed_edges.add(fold_edges[0])
        coll_key = _v2_pluralize_noun(parsed_by_idx[t_id]["text"])
        elem_key = _v2_fact_key_camel(parsed_by_idx[s_id]["text"], "amount")
        inputs.setdefault(coll_key, {"type": "COLLECTION", "elementType": "OBJECT"})
        op_name = "COUNT" if fold_rel == "REL_AGGREGATE_COUNT" else "SUM_OVER"
        fold_expr = {
            "op": op_name,
            "iteratorVar": "it",
            "collection": f"inputs.{coll_key}",
            "predicate": True,
            "body": f"it.{elem_key}" if op_name == "SUM_OVER" else 1,
        }
        fold_key = "countTotal" if op_name == "COUNT" else "sumTotalPaisa"
        fold_type = "INTEGER" if op_name == "COUNT" else "MONEY_MINOR"
        fold_var = _emit(fold_key, fold_expr, "fold")
        if not rate_nodes and not money_nodes and not bool_nodes and not dur_cond_nodes and not date_cond_nodes and not quant_edges:
            _verify_affine_consumption()
            return {
                "clauseIdx": clause_idx,
                "steps": steps,
                "handlers": handlers,
                "outputVar": fold_var,
                "outputKey": fold_key,
                "inputs": inputs,
                "standardRateBps": None,
                "resultType": fold_type,
            }

    # 2. First-Order Quantifiers & Unified Condition/Comparison DAG (`AND` binds tighter than `OR`)
    gate_cond_expr: Any = None
    quant_expr: Any = None
    if quant_edges:
        s_id, t_id, q_rel = quant_edges[0]
        consumed_ids.update((s_id, t_id))
        consumed_edges.add(quant_edges[0])
        coll_key = _v2_pluralize_noun(parsed_by_idx[t_id]["text"])
        pred_key = _v2_derive_logic_key(parsed_by_idx[s_id]["text"], "conditionMet")
        not_sources = [
            sid for sid, tid, rel in in_clause_edges
            if rel == "REL_LOGICAL_NOT" and (tid in (s_id, t_id) or (sid not in bool_nodes and sid not in (s_id, t_id)))
        ]
        for ns in not_sources:
            consumed_ids.add(ns)
        has_q_not = bool(not_sources)
        base_q = "FORALL" if q_rel == "REL_QUANTIFIER_FORALL" else "EXISTS"
        inputs.setdefault(coll_key, {"type": "COLLECTION", "elementType": "OBJECT"})
        quant_expr = {
            "op": base_q,
            "iteratorVar": "it",
            "collection": f"inputs.{coll_key}",
            "predicate": {"op": "EQ", "left": f"it.{pred_key}", "right": True},
        }
        if has_q_not and not bool_nodes and not dur_cond_nodes and not date_cond_nodes:
            quant_expr = {"op": "NOT", "operand": quant_expr}
        if not rate_nodes and not money_nodes and not bool_nodes and not dur_cond_nodes and not date_cond_nodes and fold_var is None:
            _verify_affine_consumption()
            out_key = _v2_fact_key_camel(pred_key + "Satisfied", "ruleSatisfied")
            res_var = _emit(out_key, quant_expr, "quantifier-fold")
            return {
                "clauseIdx": clause_idx,
                "steps": steps,
                "handlers": handlers,
                "outputVar": res_var,
                "outputKey": out_key,
                "inputs": inputs,
                "standardRateBps": None,
                "resultType": "BOOLEAN",
            }

    bool_edges = [e for e in in_clause_edges if e[2] in ("REL_LOGICAL_AND", "REL_LOGICAL_OR", "REL_LOGICAL_NOT")]
    base_tax_ids_pre = {
        s for s, t, rel in in_clause_edges
        if rel == "REL_BASE_TAX_ADD" and parsed_by_idx[s]["kind"] == "MONEY"
    }
    cap_ids_pre = {
        s if parsed_by_idx[s]["kind"] == "MONEY" else t
        for s, t, rel in in_clause_edges
        if rel in ("REL_MIN_CAP", "REL_CAP_CEILING", "REL_COMBINATOR_MIN")
        for s, t in [(s, t)]
        if (parsed_by_idx[s]["kind"] == "MONEY") != (parsed_by_idx[t]["kind"] == "MONEY")
        or (parsed_by_idx[s]["kind"] == "MONEY" and rel in ("REL_MIN_CAP", "REL_CAP_CEILING"))
    }
    credit_ids_pre = {
        s if parsed_by_idx[s]["kind"] == "MONEY" else t
        for s, t, rel in in_clause_edges
        if rel == "REL_SUB_CREDIT"
        for s, t in [(s, t)]
        if parsed_by_idx[s]["kind"] == "MONEY" or parsed_by_idx[t]["kind"] == "MONEY"
    }
    ceiling_ids_pre = {
        t for s, t, rel in in_clause_edges
        if rel == "REL_BRACKET_NEXT" and parsed_by_idx[t]["kind"] == "MONEY"
    }

    is_deontic_modality = (
        clause_modality in ("MODALITY_PROHIBITION", "MODALITY_PERMISSION", "MODALITY_OBLIGATION")
        or str(clause_modality or "").startswith("MODALITY_DEONTIC_")
    )
    deontic_quant_mode = (
        is_deontic_modality
        and not base_tax_ids_pre
        and not cap_ids_pre
        and not credit_ids_pre
        and not ceiling_ids_pre
        and (bool_nodes or dur_cond_nodes or date_cond_nodes or quant_expr is not None)
        and (bool(rate_nodes) or bool(money_nodes))
    )

    atom_by_id: dict[int, Any] = {}
    cond_order: list[int] = []

    def _register_atom(nid: int, expr: Any) -> None:
        if any(rel == "REL_LOGICAL_NOT" and tid == nid for _, tid, rel in bool_edges):
            for ns, tid, rel in bool_edges:
                if rel == "REL_LOGICAL_NOT" and tid == nid:
                    consumed_ids.add(ns)
            expr = {"op": "NOT", "operand": expr}
        atom_by_id[nid] = expr
        if nid not in cond_order:
            cond_order.append(nid)

    for bid in bool_nodes:
        bk = parsed_by_idx[bid]["key"]
        inputs.setdefault(bk, {"type": "BOOLEAN"})
        _register_atom(bid, f"inputs.{bk}")

    for did in dur_cond_nodes:
        pv = parsed_by_idx[did]
        dk = pv.get("key") or "durationDays"
        inputs.setdefault(dk, {"type": "INTEGER"})
        dval = int(pv["days"] if pv["days"] is not None else round(float(pv["num"])))
        _register_atom(did, {"op": pv["cmpOp"], "left": f"inputs.{dk}", "right": dval})

    for dt_id in date_cond_nodes:
        pv = parsed_by_idx[dt_id]
        if pv.get("isoDate"):
            inputs.setdefault("transactionDate", {"type": "DATE"})
            _register_atom(dt_id, {"op": pv["cmpOp"], "left": "inputs.transactionDate", "right": pv["isoDate"]})
        elif pv.get("years"):
            inputs.setdefault("taxYear", {"type": "INTEGER"})
            _register_atom(dt_id, {"op": pv["cmpOp"], "left": "inputs.taxYear", "right": int(pv["years"][0])})

    if deontic_quant_mode:
        if rate_nodes:
            r_id = rate_nodes[0]
            pv_r = parsed_by_idx[r_id]
            sym_ids = [sid for sid in topo_ids if parsed_by_idx[sid]["kind"] == "SYMBOL"]
            base_sid = next(
                (t for s, t, rel in in_clause_edges if s == r_id and rel == "REL_RATE_TO_THRESHOLD" and parsed_by_idx[t]["kind"] == "SYMBOL"),
                None,
            )
            if base_sid is not None:
                op_key = parsed_by_idx[base_sid]["key"]
                inputs.setdefault(op_key, {"type": "MONEY_MINOR", "currency": ccy})
                op_ref = f"inputs.{op_key}"
                consumed_ids.add(base_sid)
            else:
                op_ref, op_key = _resolve_operand_expr()
            subj_sid = next((sid for sid in sym_ids if parsed_by_idx[sid]["key"] != op_key), None)
            if subj_sid is not None:
                subj_key = parsed_by_idx[subj_sid]["key"]
                inputs.setdefault(subj_key, {"type": "MONEY_MINOR", "currency": ccy})
                lhs_ref = f"inputs.{subj_key}"
                consumed_ids.add(subj_sid)
            else:
                inputs.setdefault("measuredAmount", {"type": "MONEY_MINOR", "currency": ccy})
                lhs_ref = "inputs.measuredAmount"
            tgt_var = _emit(
                "ratioThresholdPaisa",
                {"op": "RATIONAL_MULTIPLY", "base": op_ref, "rateBps": int(pv_r["rateBps"]), "denominator": 10000},
                "ratio-threshold",
            )
            _register_atom(r_id, {"op": pv_r["cmpOp"], "left": lhs_ref, "right": f"vars.{tgt_var}"})
            consumed_ids.add(r_id)
        elif money_nodes:
            m_id = money_nodes[0]
            pv_m = parsed_by_idx[m_id]
            op_ref, _ = _resolve_operand_expr()
            _register_atom(m_id, {"op": pv_m["cmpOp"], "left": op_ref, "right": {"op": "TO_PAISE", "value": str(pv_m["minor"])}})
            consumed_ids.add(m_id)

    # Identify node-scoped REL_CONDITION_TO_GATE targets (e.g. surcharge fixed fee gated by condition)
    base_tax_gate_map: dict[int, list[int]] = {}
    scoped_cond_ids: set[int] = set()
    for s, t, rel in in_clause_edges:
        if rel == "REL_CONDITION_TO_GATE" and t in base_tax_ids_pre and s in atom_by_id:
            base_tax_gate_map.setdefault(t, []).append(s)
            scoped_cond_ids.add(s)

    def _combine_cond_subset(subset_ids: list[int], include_quant: bool = False) -> Any:
        sub_set = set(subset_ids)
        and_parent: dict[int, int] = {cid: cid for cid in subset_ids}

        def _find(x: int) -> int:
            while and_parent[x] != x:
                and_parent[x] = and_parent[and_parent[x]]
                x = and_parent[x]
            return x

        for sid, tid, rel in bool_edges:
            if rel == "REL_LOGICAL_AND" and sid in and_parent and tid in and_parent:
                rx, ry = _find(sid), _find(tid)
                if rx != ry:
                    and_parent[max(rx, ry)] = min(rx, ry)

        comp_members: dict[int, list[int]] = {}
        for cid in subset_ids:
            comp_members.setdefault(_find(cid), []).append(cid)

        and_groups: list[Any] = []
        for root_id in sorted(comp_members.keys(), key=lambda r: subset_ids.index(r)):
            members = comp_members[root_id]
            grp_expr = atom_by_id[members[0]]
            for m_id in members[1:]:
                grp_expr = {"op": "AND", "left": grp_expr, "right": atom_by_id[m_id]}
            and_groups.append(grp_expr)

        if include_quant and quant_expr is not None:
            and_groups.append(quant_expr)

        if not and_groups:
            return None
        has_or = any(
            rel == "REL_LOGICAL_OR" and (sid in sub_set or tid in sub_set)
            for sid, tid, rel in bool_edges
        )
        join_top = "OR" if has_or else "AND"
        expr_out = and_groups[0]
        for nxt in and_groups[1:]:
            expr_out = {"op": join_top, "left": expr_out, "right": nxt}
        return expr_out

    if scoped_cond_ids:
        # Expand scoped surcharge conditions across any AND-linked companion predicates
        changed = True
        while changed:
            changed = False
            for sid, tid, rel in bool_edges:
                if rel in ("REL_LOGICAL_AND", "REL_LOGICAL_OR"):
                    if sid in scoped_cond_ids and tid in atom_by_id and tid not in scoped_cond_ids:
                        scoped_cond_ids.add(tid)
                        for bt_k, lst in base_tax_gate_map.items():
                            if sid in lst and tid not in lst:
                                lst.append(tid)
                        changed = True
                    elif tid in scoped_cond_ids and sid in atom_by_id and sid not in scoped_cond_ids:
                        scoped_cond_ids.add(sid)
                        for bt_k, lst in base_tax_gate_map.items():
                            if tid in lst and sid not in lst:
                                lst.append(sid)
                        changed = True

    remaining_cond_ids = [cid for cid in cond_order if cid not in scoped_cond_ids]
    if remaining_cond_ids or quant_expr is not None:
        gate_cond_expr = _combine_cond_subset(remaining_cond_ids, include_quant=True)
        for cid in remaining_cond_ids:
            consumed_ids.add(cid)

    # Pure boolean/quantifier/deontic predicate rule
    if deontic_quant_mode or (not rate_nodes and not money_nodes and fold_var is None):
        if gate_cond_expr is None:
            raise V2LoweringError("ERR_NO_COMPUTATION: no computational or boolean predicate spans resolved")
        _verify_affine_consumption()
        out_var = _emit("predicateResult", gate_cond_expr, "boolean-gate")
        return {
            "clauseIdx": clause_idx,
            "steps": steps,
            "handlers": handlers,
            "outputVar": out_var,
            "outputKey": "predicateResult",
            "inputs": inputs,
            "standardRateBps": None,
            "resultType": "BOOLEAN",
        }

    # 4. Quantitative Arithmetic Pipeline (bottom-up walk over `topo_ids`)
    # Resolve any symbolic excess-over-threshold reference via `resolved_corefs` or `symbol_env`
    sym_floor_ref: str | None = None
    for sid in topo_ids:
        if parsed_by_idx[sid]["kind"] == "MONEY":
            continue
        is_threshold_cand = (
            sid in threshold_role_or_targets
            or span_by_idx[sid].get("role") == "ROLE_TRANSACTION_EXCESS_THRESHOLD"
        )
        if not is_threshold_cand:
            continue
        if sid in resolved_corefs:
            sym_floor_ref = str(resolved_corefs[sid]["refVar"])
            break
        key = _v2_fact_key_camel(parsed_by_idx[sid]["text"], "")
        if key and key in symbol_env:
            sym_floor_ref = str(symbol_env[key]["ref"])
            break
        span_key = parsed_by_idx[sid].get("key")
        if span_key and span_key in symbol_env:
            sym_floor_ref = str(symbol_env[span_key]["ref"])
            break

    # Check explicit excess-over-threshold safety invariant without silent flat degradation
    excess_symbol_only = any(
        span_by_idx[sid].get("role") == "ROLE_TRANSACTION_EXCESS_THRESHOLD" and parsed_by_idx[sid]["kind"] != "MONEY"
        for sid in topo_ids
    )
    has_sub_base_edge = any(rel == "REL_SUBTRACTION_BASE" for _, _, rel in in_clause_edges)
    if rate_nodes and not money_nodes and (excess_symbol_only or has_sub_base_edge):
        if sym_floor_ref is None:
            raise V2LoweringError("ERR_NO_THRESHOLD: excess-over-threshold clause has no resolved monetary threshold")

    # Classify monetary spans cleanly using explicit v2 relation edges (zero text-guessing regexes)
    cap_ids = {
        s if parsed_by_idx[s]["kind"] == "MONEY" else t
        for s, t, rel in in_clause_edges
        if rel in ("REL_MIN_CAP", "REL_CAP_CEILING", "REL_COMBINATOR_MIN")
        for s, t in [(s, t)]
        if (parsed_by_idx[s]["kind"] == "MONEY") != (parsed_by_idx[t]["kind"] == "MONEY")
        or (parsed_by_idx[s]["kind"] == "MONEY" and rel in ("REL_MIN_CAP", "REL_CAP_CEILING"))
    }
    credit_ids = {
        s if parsed_by_idx[s]["kind"] == "MONEY" else t
        for s, t, rel in in_clause_edges
        if rel == "REL_SUB_CREDIT"
        for s, t in [(s, t)]
        if parsed_by_idx[s]["kind"] == "MONEY" or parsed_by_idx[t]["kind"] == "MONEY"
    }
    base_tax_ids = {
        s
        for s, t, rel in in_clause_edges
        if rel == "REL_BASE_TAX_ADD" and parsed_by_idx[s]["kind"] == "MONEY"
    }
    ceiling_ids = {
        t
        for s, t, rel in in_clause_edges
        if rel == "REL_BRACKET_NEXT" and parsed_by_idx[t]["kind"] == "MONEY"
    }

    if fold_var is not None:
        operand_ref = f"vars.{fold_var}"
        operand_key = (fold_key or "foldValue").removesuffix("Paisa")
    else:
        operand_ref, operand_key = _resolve_operand_expr()
    current_var: str | None = None
    active_floor_minor: int | None = None
    active_ceiling_minor: int | None = None

    # Identify primary floor threshold (MONEY connected by REL_RATE_TO_THRESHOLD / REL_BRACKET_NEXT)
    # Never silently promote unlinked MONEY spans when rate_nodes are present.
    floor_ids: list[int] = []
    for s, t, rel in in_clause_edges:
        if rel == "REL_RATE_TO_THRESHOLD" and parsed_by_idx[t]["kind"] == "MONEY" and t not in cap_ids and t not in credit_ids and t not in base_tax_ids:
            floor_ids.append(t)
        elif rel == "REL_BRACKET_NEXT" and parsed_by_idx[s]["kind"] == "MONEY":
            floor_ids.append(s)
    if not floor_ids and not rate_nodes:
        for sid in topo_ids:
            if (
                parsed_by_idx[sid]["kind"] == "MONEY"
                and sid not in cap_ids
                and sid not in credit_ids
                and sid not in base_tax_ids
                and sid not in ceiling_ids
            ):
                floor_ids.append(sid)

    if clause_modality == "MODALITY_COMPUTATIONAL_EXCESS" and rate_nodes and not floor_ids and sym_floor_ref is None:
        raise V2LoweringError("ERR_NO_THRESHOLD: MODALITY_COMPUTATIONAL_EXCESS clause lacks connected floor threshold")

    # Emit bottom-up arithmetic following topological node order
    primary_rate_bps = parsed_by_idx[rate_nodes[0]]["rateBps"] if rate_nodes else None
    gate_consumed = False

    def _detect_dual_rate_mode() -> str | None:
        if len(rate_nodes) < 2:
            return None
        r1, r2 = rate_nodes[0], rate_nodes[1]
        is_cond_fork = (
            gate_cond_expr is not None
            and (
                span_by_idx[r2].get("role") == "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"
                or any(
                    rel in ("REL_OVERRIDE_TO_GUARD", "REL_SUBSTITUTE_WHEN", "REL_EXCEPTION_TO_PROHIBITION", "REL_CONDITION_TO_GATE")
                    and (s == r2 or t == r2 or s == r1 or t == r1)
                    for s, t, rel in in_clause_edges
                )
            )
        )
        if is_cond_fork:
            return "conditional"
        if any(rel == "REL_AGGREGATE_SUM" and {s, t} <= set(rate_nodes) for s, t, rel in in_clause_edges):
            return "additive"
        return None

    dual_rate_mode = _detect_dual_rate_mode()

    def _emit_rate_application(base_expr_ref: str) -> str:
        nonlocal gate_consumed
        bps_0 = int(parsed_by_idx[rate_nodes[0]]["rateBps"])
        t0 = _emit(
            "taxAmountPaisa",
            {
                "op": "RATIONAL_MULTIPLY",
                "base": base_expr_ref,
                "rateBps": bps_0,
                "denominator": 10000,
            },
            "rate-on-base",
        )
        node_ssa[rate_nodes[0]] = f"vars.{t0}"
        consumed_ids.add(rate_nodes[0])
        if dual_rate_mode == "conditional":
            bps_1 = int(parsed_by_idx[rate_nodes[1]]["rateBps"])
            t1 = _emit(
                "altTaxAmountPaisa",
                {
                    "op": "RATIONAL_MULTIPLY",
                    "base": base_expr_ref,
                    "rateBps": bps_1,
                    "denominator": 10000,
                },
                "rate-on-base",
            )
            node_ssa[rate_nodes[1]] = f"vars.{t1}"
            consumed_ids.add(rate_nodes[1])
            gate_consumed = True
            return _emit(
                "branchedRateTaxPaisa",
                {
                    "op": "CONDITIONAL",
                    "test": gate_cond_expr,
                    "then": f"vars.{t0}",
                    "else": f"vars.{t1}",
                },
                "conditional-rate-branch",
            )
        if dual_rate_mode == "additive":
            acc = t0
            for r_extra in rate_nodes[1:]:
                bps_e = int(parsed_by_idx[r_extra]["rateBps"])
                te = _emit(
                    "extraRateTaxPaisa",
                    {
                        "op": "RATIONAL_MULTIPLY",
                        "base": base_expr_ref,
                        "rateBps": bps_e,
                        "denominator": 10000,
                    },
                    "rate-on-base",
                )
                node_ssa[r_extra] = f"vars.{te}"
                consumed_ids.add(r_extra)
                acc = _emit(
                    "additiveRatesTaxPaisa",
                    {"op": "ADD", "left": f"vars.{acc}", "right": f"vars.{te}"},
                    "additive-rate-sum",
                )
            return acc
        return t0

    if rate_nodes and (floor_ids or sym_floor_ref is not None):
        if floor_ids:
            floor_id = floor_ids[0]
            consumed_ids.add(floor_id)
            floor_minor: int | None = parsed_by_idx[floor_id]["minor"]
            floor_role = span_by_idx[floor_id].get("role", "")
            floor_span_text = span_by_idx[floor_id].get("span", "")
            active_floor_minor = floor_minor
            floor_expr: Any = {"op": "TO_PAISE", "value": str(floor_minor)}
        else:
            floor_minor = None
            floor_role = "ROLE_TRANSACTION_EXCESS_THRESHOLD"
            floor_span_text = ""
            floor_expr = sym_floor_ref

        # Excess base = MAX(0, operand - floor)
        excess_var = _emit(
            "excessBasePaisa",
            {
                "op": "MAX",
                "left": {"op": "TO_PAISE", "value": "0"},
                "right": {"op": "SUB", "left": operand_ref, "right": floor_expr},
            },
            "base-arithmetic",
        )
        base_for_rate = f"vars.{excess_var}"

        # Bounded bracket ceiling (`REL_BRACKET_NEXT`)
        if ceiling_ids:
            ceil_id = next(iter(ceiling_ids))
            consumed_ids.add(ceil_id)
            ceil_minor = parsed_by_idx[ceil_id]["minor"]
            active_ceiling_minor = ceil_minor
            if floor_minor is not None:
                width_expr: Any = {"op": "TO_PAISE", "value": str(max(0, ceil_minor - floor_minor))}
            else:
                width_expr = {
                    "op": "MAX",
                    "left": {"op": "TO_PAISE", "value": "0"},
                    "right": {"op": "SUB", "left": {"op": "TO_PAISE", "value": str(ceil_minor)}, "right": floor_expr},
                }
            slice_var = _emit(
                "bracketSlicePaisa",
                {
                    "op": "MIN",
                    "left": base_for_rate,
                    "right": width_expr,
                },
                "bracket-next",
            )
            base_for_rate = f"vars.{slice_var}"
        elif floor_role == "ROLE_TRANSACTION_CLIFF_THRESHOLD":
            # Cliff threshold: tax applies to ENTIRE operand only if operand > floor
            cmp_op = _v2_parse_cmp_op(floor_span_text)
            if cmp_op == "GTE":
                cliff_var = _emit(
                    "cliffBasePaisa",
                    {
                        "op": "CONDITIONAL",
                        "test": {"op": "GTE", "left": operand_ref, "right": floor_expr},
                        "then": operand_ref,
                        "else": {"op": "TO_PAISE", "value": "0"},
                    },
                    "base-arithmetic",
                )
            else:
                cliff_var = _emit(
                    "cliffBasePaisa",
                    {
                        "op": "CONDITIONAL",
                        "test": {"op": "GT", "left": operand_ref, "right": floor_expr},
                        "then": operand_ref,
                        "else": {"op": "TO_PAISE", "value": "0"},
                    },
                    "base-arithmetic",
                )
            base_for_rate = f"vars.{cliff_var}"

        current_var = _emit_rate_application(base_for_rate)

        # Window gate `(floor, ceiling]` when bounded bracket is specified:
        # unconditional bracket base taxes must sit inside the window gate so
        # amounts below floor or above ceiling evaluate to 0.
        if active_ceiling_minor is not None and active_floor_minor is not None:
            for sid in topo_ids:
                if sid in base_tax_ids and sid not in base_tax_gate_map and parsed_by_idx[sid]["minor"] is not None:
                    bt_minor = parsed_by_idx[sid]["minor"]
                    consumed_ids.add(sid)
                    current_var = _emit(
                        "bracketWithBaseTaxPaisa",
                        {
                            "op": "ADD",
                            "left": {"op": "TO_PAISE", "value": str(bt_minor)},
                            "right": f"vars.{current_var}",
                        },
                        "bracket-base-tax",
                    )
            gate_var = _emit(
                "boundedBracketTaxPaisa",
                {
                    "op": "CONDITIONAL",
                    "test": {
                        "op": "AND",
                        "left": {"op": "GT", "left": operand_ref, "right": {"op": "TO_PAISE", "value": str(active_floor_minor)}},
                        "right": {"op": "LTE", "left": operand_ref, "right": {"op": "TO_PAISE", "value": str(active_ceiling_minor)}},
                    },
                    "then": f"vars.{current_var}",
                    "else": {"op": "TO_PAISE", "value": "0"},
                },
                "bracket-window-gate",
            )
            current_var = gate_var

    elif rate_nodes:
        # Flat or branched rate over operand (without floor subtraction)
        current_var = _emit_rate_application(operand_ref)
    elif floor_ids:
        # Pure value/threshold definition clause (e.g. "$4,400", "fifty lakh rupees")
        m_id = floor_ids[0]
        consumed_ids.add(m_id)
        m_minor = parsed_by_idx[m_id]["minor"]
        val_expr = {"op": "TO_PAISE", "value": str(m_minor)}
        if span_by_idx[m_id].get("role") == "ROLE_TRANSACTION_EXCESS_THRESHOLD":
            val_expr = {
                "op": "MAX",
                "left": {"op": "TO_PAISE", "value": "0"},
                "right": {"op": "SUB", "left": operand_ref, "right": val_expr},
            }
        def_key = f"{operand_key}Paisa" if not operand_key.endswith("Paisa") else operand_key
        current_var = _emit(def_key, val_expr, "definition-value")
        _verify_affine_consumption()
        return {
            "clauseIdx": clause_idx,
            "steps": steps,
            "handlers": handlers,
            "outputVar": current_var,
            "outputKey": def_key,
            "inputs": inputs,
            "standardRateBps": None,
            "resultType": "MONEY_MINOR",
        }
    elif fold_var is not None:
        current_var = fold_var
    else:
        raise V2LoweringError("ERR_NO_COMPUTATION: no rate, money, or predicate structure to lower")

    # Add explicit `REL_BASE_TAX_ADD` node(s) cleanly for both excess/bracket and flat rates;
    # if targeted by `REL_CONDITION_TO_GATE`, gate only that surcharge node (`CONDITIONAL`).
    for sid in topo_ids:
        if sid in base_tax_ids and sid not in consumed_ids and parsed_by_idx[sid]["minor"] is not None:
            bt_minor = parsed_by_idx[sid]["minor"]
            consumed_ids.add(sid)
            bt_expr: Any = {"op": "TO_PAISE", "value": str(bt_minor)}
            if sid in base_tax_gate_map:
                sc_test = _combine_cond_subset(base_tax_gate_map[sid], include_quant=False)
                for cid in base_tax_gate_map[sid]:
                    consumed_ids.add(cid)
                bt_expr = {
                    "op": "CONDITIONAL",
                    "test": sc_test,
                    "then": bt_expr,
                    "else": {"op": "TO_PAISE", "value": "0"},
                }
            current_var = _emit(
                "bracketWithBaseTaxPaisa",
                {
                    "op": "ADD",
                    "left": bt_expr,
                    "right": f"vars.{current_var}",
                },
                "bracket-base-tax",
            )

    # Apply downstream topological reductions (`REL_SUB_CREDIT`) then caps (`REL_MIN_CAP`) in `topo_ids` order
    for sid in topo_ids:
        if sid in credit_ids and parsed_by_idx[sid]["minor"] is not None:
            consumed_ids.add(sid)
            c_minor = parsed_by_idx[sid]["minor"]
            current_var = _emit(
                "afterCreditPaisa",
                {
                    "op": "MAX",
                    "left": {"op": "TO_PAISE", "value": "0"},
                    "right": {"op": "SUB", "left": f"vars.{current_var}", "right": {"op": "TO_PAISE", "value": str(c_minor)}},
                },
                "reduction",
            )
    for sid in topo_ids:
        if sid in cap_ids and parsed_by_idx[sid]["minor"] is not None:
            consumed_ids.add(sid)
            cap_minor = parsed_by_idx[sid]["minor"]
            current_var = _emit(
                "afterCapPaisa",
                {
                    "op": "MIN",
                    "left": f"vars.{current_var}",
                    "right": {"op": "TO_PAISE", "value": str(cap_minor)},
                },
                "cap",
            )

    if gate_cond_expr is not None and not gate_consumed:
        zero_expr: Any = 0 if (fold_type == "INTEGER" and not rate_nodes and not money_nodes) else {"op": "TO_PAISE", "value": "0"}
        gate_key = "gatedCount" if (fold_type == "INTEGER" and not rate_nodes and not money_nodes) else "gatedResultPaisa"
        current_var = _emit(
            gate_key,
            {
                "op": "CONDITIONAL",
                "test": gate_cond_expr,
                "then": f"vars.{current_var}",
                "else": zero_expr,
            },
            "boolean-gate",
        )

    _verify_affine_consumption()

    out_type = fold_type if (fold_var is not None and not rate_nodes and not money_nodes) else "MONEY_MINOR"
    if out_type == "INTEGER":
        out_key = fold_key or "countTotal"
    else:
        out_key = operand_key if operand_key.endswith("TaxPaisa") else f"{operand_key}TaxPaisa"
    return {
        "clauseIdx": clause_idx,
        "steps": steps,
        "handlers": handlers,
        "outputVar": current_var,
        "outputKey": out_key,
        "inputs": inputs,
        "standardRateBps": primary_rate_bps,
        "resultType": out_type,
    }
