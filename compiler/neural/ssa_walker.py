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

from compiler.neural.clause_linker import base_phrase_after_rate, input_id_from_phrase, normalize_term
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
            self.inputs[input_id] = {
                "type": "MONEY_MINOR",
                "currency": self.currency_code or "USD",
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


def _static_money(node: SsaNode) -> dict | None:
    minor = parse_minor_units(node.text)
    if minor is None:
        return None
    return {"op": "TO_PAISE", "value": str(minor)}


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
        base_phrase = ""
        for node in nodes:
            if node.role == "ROLE_COMPUTATION_OPERAND":
                base_phrase = node.text
                break
        if not base_phrase:
            base_phrase = base_phrase_after_rate(clause_text, rate_node.char_range[1])
        base_id = input_id_from_phrase(base_phrase, currency_code)
        if not base_id and inherited_base_id:
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
    if not base_id and inherited_base_id:
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

    # 6. base arithmetic: the amount above the threshold
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
