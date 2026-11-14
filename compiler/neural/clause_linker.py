"""
Clause scoping and def-use symbol linking (RFC-012, Step 2).

This module turns a flat, document-wide span/edge graph into *scoped* clauses
with an explicit symbol table, and lowers a multi-clause statute into a single
executable pipeline whose values carry SSA names.

The rules implemented here are the ones written down in
``docs/rfcs/RFC-012-lowerer-composition.md`` section 4:

* 4.3  partitioning (clause index first, character ranges second);
* 4.4  definition extraction, in strict precedence order;
* 4.5  use extraction and resolution (exact phrase, then external input);
* 4.6  SSA naming, with a version bump on every statutory override.

Failure is closed: an ambiguous reference or a cyclic definition produces a
stable diagnostic code and no AST.

Naming note: RFC-012 4.6 writes a logical symbol as ``clause_<i>.<name>_v<k>``.
The USVM resolves ``vars.<name>`` through a dotted property lookup, so a dot
inside a variable name would be read as a path segment. The AST encodes the
same identity with an underscore separator; the dotted form stays the logical
symbol id recorded in the clause scopes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from compiler.neural.inference import (
    build_operand_fact_keys,
    parse_minor_units,
    parse_rate_bps,
    parse_unit_and_type,
)


# â”€â”€ Diagnostic codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ERR_SPAN_CROSSES_CLAUSE = "ERR_SPAN_CROSSES_CLAUSE"
ERR_AMBIGUOUS_REFERENCE = "ERR_AMBIGUOUS_REFERENCE"
ERR_CYCLIC_DEFINITION = "ERR_CYCLIC_DEFINITION"
ERR_UNRESOLVED_REFERENCE = "ERR_UNRESOLVED_REFERENCE"
ERR_NO_COMPUTATION = "ERR_NO_COMPUTATION"


class ClauseLinkingError(RuntimeError):
    """Raised when the linker cannot resolve a statute unambiguously.

    The linker fails closed: an ambiguous reference or a cyclic definition
    never yields an AST. The exception carries the stable error code and the
    diagnostics that explain it.
    """

    def __init__(self, code: str, diagnostics: list["Diagnostic"]):
        self.code = code
        self.diagnostics = diagnostics
        detail = "; ".join(d.detail for d in diagnostics if d.code == code) or code
        super().__init__(f"{code}: {detail}")


# â”€â”€ Data model (RFC-012 section 4.2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass(frozen=True)
class SpanRef:
    clause_idx: int
    span_idx: int
    role: str
    char_range: tuple[int, int]
    text: str
    confidence: float = 1.0


@dataclass(frozen=True)
class EdgeRef:
    source: SpanRef
    target: SpanRef
    relation: str
    confidence: float = 1.0


@dataclass
class Symbol:
    id: str
    kind: str                 # INPUT | OUTPUT | TERM | PARAM
    semantic_type: str        # money | integer | rate | term | text
    defined_in: int
    name: str
    aliases: list[str] = field(default_factory=list)
    value_expr: Any = None
    version: int = 1
    ssa_name: str = ""
    origin: str = "definition"   # definition | input | override
    minor: int | None = None

    def key(self) -> str:
        return f"{self.defined_in}:{self.id}:{self.version}"


@dataclass
class Use:
    symbol_id: str | None
    used_in: int
    step_id: str
    expected_type: str
    surface: str = ""


@dataclass
class ClauseScope:
    index: int
    char_range: tuple[int, int]
    spans: list[SpanRef] = field(default_factory=list)
    edges: list[EdgeRef] = field(default_factory=list)
    definitions: list[Symbol] = field(default_factory=list)
    uses: list[Use] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass
class Diagnostic:
    code: str
    severity: str
    clause_idx: int
    detail: str
    span: SpanRef | None = None
    symbol: str | None = None


@dataclass
class LinkResult:
    scopes: list[ClauseScope] = field(default_factory=list)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    cross_refs: list[tuple[int, str, int, str]] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)


# â”€â”€ Text normalization (RFC-012 section 4.4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_DETERMINERS = {
    "the", "a", "an", "such", "said", "respective", "that", "this",
    "those", "these", "any", "each", "every",
}
_GENERIC_ADJECTIVES = {"basic", "general", "ordinary", "same", "aforesaid"}
_SINGULARIZE_EXCEPTIONS = {"gross", "loss", "business", "interest", "wages", "sales"}


def normalize_term(text: str) -> str:
    """Lowercases, strips leading determiners and trailing punctuation."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    words = [w for w in cleaned.split() if w]
    while words and (words[0] in _DETERMINERS or words[0] in _GENERIC_ADJECTIVES):
        words.pop(0)
    while words and words[-1] in _DETERMINERS:
        words.pop()
    return " ".join(words)


def singularize(word: str) -> str:
    """Drops a trailing plural 's' unless the singular is a known term."""
    if word in _SINGULARIZE_EXCEPTIONS:
        return word
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def camel_name(text: str) -> str:
    """Builds a stable camelCase identifier from a surface phrase."""
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", text or "") if p]
    if not parts:
        return ""
    return parts[0].lower() + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def ssa_name(clause_idx: int, semantic_name: str, version: int) -> str:
    """The engine-safe SSA name of a value (RFC-012 4.6)."""
    return f"clause_{clause_idx}_{semantic_name}_v{version}"


def _output_key_from_assign(assign: str) -> str:
    """The exported name of an SSA value: ``clause_1_x_v2`` -> ``x``."""
    key = re.sub(r"^clause_\d+_", "", assign or "")
    return re.sub(r"_v\d+$", "", key)


def input_id_from_phrase(phrase: str, currency_code: str | None = None) -> str:
    """Derives an input id from a surface phrase, with a deterministic fallback."""
    if re.search(r"\b(?:under|in)\s+(?:subparagraph|paragraph|subsection|section)\b", phrase or "", re.IGNORECASE):
        return ""
    keys = build_operand_fact_keys(phrase, currency_code)
    if keys:
        return keys[0]
    words = [
        w for w in re.split(r"[^a-z0-9]+", (phrase or "").lower())
        if w and w not in _DETERMINERS and len(w) > 1
    ]
    if not words:
        return ""
    return camel_name(words[-1])


# â”€â”€ Partitioning (RFC-012 section 4.3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def partition_clauses(
    spans: list[dict],
    clause_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[ClauseScope], list[Diagnostic], dict[int, SpanRef]]:
    """Splits spans into clause scopes.

    1. A contiguous ``clauseIdx`` on every span is authoritative.
    2. Otherwise the character ranges returned by ``segment_clauses`` decide.
    3. A span that crosses a clause boundary is rejected.
    """
    diagnostics: list[Diagnostic] = []
    ordered = sorted(
        spans or [],
        key=lambda s: (s.get("clauseIdx", 0), (s.get("charRange") or [0, 0])[0], s.get("spanIdx", 0)),
    )
    if not ordered:
        return [], diagnostics, {}

    use_index = all(s.get("clauseIdx") is not None for s in ordered)
    scopes: list[ClauseScope] = []
    spans_by_idx: dict[int, SpanRef] = {}

    if use_index:
        bounds: dict[int, tuple[int, int]] = {}
        for span in ordered:
            idx = int(span.get("clauseIdx", 0))
            cr = tuple(span.get("charRange") or [0, 0])
            low, high = bounds.get(idx, (cr[0], cr[1]))
            bounds[idx] = (min(low, cr[0]), max(high, cr[1]))
        # The segmentation owns the clause boundaries; the span extents only
        # fill in clauses the segmentation did not describe.
        if clause_ranges:
            for idx, (low, high) in enumerate(clause_ranges):
                bounds[idx] = (low, high)
        for idx in sorted(bounds):
            scopes.append(ClauseScope(index=idx, char_range=bounds[idx]))
    elif clause_ranges:
        for idx, (low, high) in enumerate(clause_ranges):
            scopes.append(ClauseScope(index=idx, char_range=(low, high)))
    else:
        scopes.append(ClauseScope(index=0, char_range=(0, 10**9)))

    by_index = {scope.index: scope for scope in scopes}

    for span in ordered:
        cr = tuple(span.get("charRange") or [0, 0])
        if use_index:
            scope = by_index.get(int(span.get("clauseIdx", 0)))
        else:
            scope = next(
                (s for s in scopes if s.char_range[0] <= cr[0] < s.char_range[1]),
                scopes[-1] if scopes else None,
            )
        if scope is None:
            diagnostics.append(
                Diagnostic(
                    code=ERR_SPAN_CROSSES_CLAUSE,
                    severity="error",
                    clause_idx=-1,
                    detail=f"span {span.get('span')!r} does not belong to any clause",
                )
            )
            continue
        if scope.char_range[1] < 10**9 and not (
            scope.char_range[0] <= cr[0] and cr[1] <= scope.char_range[1]
        ):
            diagnostics.append(
                Diagnostic(
                    code=ERR_SPAN_CROSSES_CLAUSE,
                    severity="error",
                    clause_idx=scope.index,
                    detail=(
                        f"span {span.get('span')!r} at {list(cr)} crosses the boundary "
                        f"of clause {scope.index} {list(scope.char_range)}"
                    ),
                )
            )
            continue
        ref = SpanRef(
            clause_idx=scope.index,
            span_idx=int(span.get("spanIdx", 0)),
            role=str(span.get("role", "")),
            char_range=(int(cr[0]), int(cr[1])),
            text=str(span.get("span", "")),
            confidence=float(span.get("confidence", 1.0) or 0.0),
        )
        scope.spans.append(ref)
        spans_by_idx[ref.span_idx] = ref

    return scopes, diagnostics, spans_by_idx


def attach_edges(
    scopes: list[ClauseScope],
    edges: list[dict],
    spans_by_idx: dict[int, SpanRef],
) -> list[tuple[int, str, int, str]]:
    """Distributes edges over clauses and reports cross-clause candidates."""
    cross_refs: list[tuple[int, str, int, str]] = []
    for edge in edges or []:
        src = spans_by_idx.get(edge.get("sourceIdx"))
        tgt = spans_by_idx.get(edge.get("targetIdx"))
        if src is None or tgt is None:
            continue
        relation = str(edge.get("relationType", "REL_NO_EDGE"))
        if src.clause_idx != tgt.clause_idx:
            cross_refs.append((src.clause_idx, src.text, tgt.clause_idx, tgt.text))
            continue
        scope = next((s for s in scopes if s.index == src.clause_idx), None)
        if scope is None:
            continue
        scope.edges.append(
            EdgeRef(
                source=src,
                target=tgt,
                relation=relation,
                confidence=min(src.confidence, tgt.confidence),
            )
        )
    return cross_refs


# â”€â”€ Definition extraction (RFC-012 section 4.4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_DEF_VERBS = r"means|shall\s+mean|is\s+defined\s+as"
_ASSIGN_VERBS = r"equals|is\s+equal\s+to|is|are"
# An explicit term definition ("X means Y", "X is defined as Y").
_TERM_DEF_RE = re.compile(
    r"(?:^|[.;:]\s+|,\s+)(?:the\s+term\s+)?(?P<term>[A-Za-z][A-Za-z0-9 ,\-]{2,60}?)\s+"
    r"(?P<verb>" + _DEF_VERBS + r")\s+(?P<value>[^.;]{1,90})",
    re.IGNORECASE,
)
# An explicit assignment ("the X is <amount>", "the X equals <expr>"). The
# determiner is required: "tax equals 28 percent" states a computation, it
# does not define a term.
_ASSIGN_RE = re.compile(
    r"(?:^|[.;:]\s+|,\s+)(?:the\s+|an?\s+|such\s+)(?P<term>[A-Za-z][A-Za-z0-9 ,\-]{2,60}?)\s+"
    r"(?P<verb>" + _ASSIGN_VERBS + r")\s+(?P<value>[^.;]{1,90})",
    re.IGNORECASE,
)

_AMOUNT = r"[$â‚¬Â£Â¥]?-?\d[\d,]*(?:\.\d+)?"
_SUBSTITUTION_RE = re.compile(
    r"(?P<new>" + _AMOUNT + r")\s*shall\s+be\s+substituted\s+for\s+(?P<old>" + _AMOUNT + r")",
    re.IGNORECASE,
)

_RATE_ROLES = {
    "ROLE_STANDARD_STATUTORY_RATE",
    "ROLE_PENALTY_OR_DEFAULT_OVERRIDE",
    "ROLE_NON_PAN_OVERRIDE_RATE",
}
_DATE_ROLES = {"ROLE_TEMPORAL_CUTOVER_DATE", "ROLE_TEMPORAL_SUNSET_DATE"}
#: Relations composed by the RFC-012 Step 3 topological walk.
_WALKED_RELATIONS = {
    "REL_RATE_TO_THRESHOLD",
    "REL_BRACKET_NEXT",
    "REL_CAP_CEILING",
    "REL_SUBTRACTION_BASE",
    "REL_COMBINATOR_MIN",
    "REL_COMBINATOR_MAX",
}
_COMPARATIVE_RE = re.compile(
    r"\b(?:exceeding|exceeds|above|over|more\s+than|in\s+excess\s+of|greater\s+than)\b",
    re.IGNORECASE,
)
_QUANTITATIVE_TERM_RE = re.compile(
    r"\b(?:amount|deduction|income|tax|credit|exemption|allowance|sum|rate|percentage|"
    r"limit|threshold|ceiling|cap|value|liability|wages?|turnover|consideration|fee|payment)\b",
    re.IGNORECASE,
)


def is_monetary_quantum_text(text: str | None) -> bool:
    """Returns True only when text contains an explicit monetary currency symbol or unit."""
    if not text:
        return False
    st, _dt, unit = parse_unit_and_type(text)
    return st == "MONETARY_QUANTUM" and unit in ("USD", "INR", "EUR", "GBP")


def parse_monetary_minor_units(text: str | None) -> int | None:
    """Parses minor units strictly when text is a confirmed MONETARY_QUANTUM with currency."""
    if not is_monetary_quantum_text(text):
        return None
    return parse_minor_units(text)


def _definition_semantic_type(value_text: str) -> str:
    """The semantic type of a defined value."""
    if parse_monetary_minor_units(value_text) is not None:
        return "money"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value_text.strip()):
        return "integer"
    if parse_unit_and_type(value_text)[0] == "PERCENTAGE_RATE":
        return "rate"
    # A definition by reference to a threshold or to another term.
    return "term"


def extract_definitions(
    scope: ClauseScope,
    clause_text: str,
) -> tuple[list[Symbol], list[dict]]:
    """Extracts the symbols a clause defines, in RFC-012 4.4 precedence order.

    Returns the symbols and the raw matches so the builder can reuse the
    matched value text.
    """
    symbols: list[Symbol] = []
    matches: list[dict] = []
    seen_terms: set[str] = set()

    for pattern in (_TERM_DEF_RE, _ASSIGN_RE):
        for match in pattern.finditer(clause_text or ""):
            term = normalize_term(match.group("term"))
            value_text = match.group("value").strip().rstrip(",")
            if not term or len(term) < 3:
                continue
            if term in seen_terms:
                continue
            minor_val = parse_monetary_minor_units(value_text)
            if pattern is _ASSIGN_RE:
                verb_low = match.group("verb").strip().lower()
                if verb_low in ("is", "are") and minor_val is None and parse_unit_and_type(value_text)[0] != "PERCENTAGE_RATE":
                    # Boolean status predicates (e.g. "such individual is not married...")
                    # are not quantitative assignments.
                    if value_text.lower().startswith("not ") or not _QUANTITATIVE_TERM_RE.search(term):
                        continue
            seen_terms.add(term)
            symbol = Symbol(
                id=term,
                kind="TERM",
                semantic_type=_definition_semantic_type(value_text),
                defined_in=scope.index,
                name=camel_name(term),
                aliases=[term],
                value_expr=value_text,
                origin="definition",
                minor=minor_val,
            )
            symbols.append(symbol)
            matches.append(
                {
                    "symbol": symbol,
                    "term": term,
                    "value_text": value_text,
                    "minor": symbol.minor,
                    "span": (match.start(), match.end()),
                }
            )
    return symbols, matches


def detect_substitution(clause_text: str) -> dict | None:
    """Detects the statutory substitution form: ``X shall be substituted for Y``."""
    match = _SUBSTITUTION_RE.search(clause_text or "")
    if not match:
        return None
    return {
        "new_text": match.group("new"),
        "old_text": match.group("old"),
        "new_minor": parse_monetary_minor_units(match.group("new")),
        "old_minor": parse_monetary_minor_units(match.group("old")),
    }


def extract_uses(scope: ClauseScope, clause_text: str, symbols: list[Symbol]) -> list[Use]:
    """Extracts the symbols a clause uses (RFC-012 4.5)."""
    uses: list[Use] = []
    for symbol in symbols:
        pattern = re.compile(rf"\b{re.escape(symbol.id)}\b", re.IGNORECASE)
        for match in pattern.finditer(clause_text or ""):
            uses.append(
                Use(
                    symbol_id=symbol.id,
                    used_in=scope.index,
                    step_id=f"clause_{scope.index}_use_{len(uses)}",
                    expected_type=symbol.semantic_type,
                    surface=match.group(0),
                )
            )
    return uses


def find_cycle(symbol_ids: list[str], edges: dict[str, list[str]]) -> list[str] | None:
    """Returns one cycle in the definition graph, if any."""
    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for neighbour in edges.get(node, []):
            if state.get(neighbour) == 1:
                start = stack.index(neighbour)
                return stack[start:] + [neighbour]
            if state.get(neighbour, 0) == 0:
                found = visit(neighbour, stack)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for symbol_id in symbol_ids:
        if state.get(symbol_id, 0) == 0:
            found = visit(symbol_id, [])
            if found:
                return found
    return None

# â”€â”€ Clause-scoped computation builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class _ClausePlan:
    scope: ClauseScope
    text: str
    kind: str = "reference"
    symbols: list[Symbol] = field(default_factory=list)
    matches: list[dict] = field(default_factory=list)
    substitution: dict | None = None
    steps: list[dict] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    rate_bps: int | None = None
    handlers: list[str] = field(default_factory=list)


def _operator_before(text: str, position: int) -> str | None:
    """Returns the last comparison keyword that governs a date or threshold."""
    window = text[max(0, position - 40):position].lower()
    operators = [
        "on or after", "not later than", "no later than", "on or before",
        "not before", "after", "before", "on", "at",
    ]
    best = None
    for operator in operators:
        index = window.rfind(operator)
        if index >= 0 and (best is None or index > best[0]):
            best = (index, operator)
    return best[1] if best else None


_PHRASE_STOP_WORDS = {
    "paid", "payable", "received", "owing", "by", "which", "that", "shall",
    "is", "are", "was", "were", "and", "or", "during", "for", "from", "in",
    "on", "of", "with", "such", "the", "a", "an", "any", "each", "every",
    "has", "have", "been", "be", "will", "may", "must", "per", "than",
}


def base_phrase_after_rate(clause_text: str, rate_end: int) -> str:
    """Extracts the object a rate applies to: ``... 6 percent of the wages``."""
    tail = clause_text[rate_end:]
    window = tail[:60]
    best = None
    for operator in ("of", "on", "upon", "to"):
        index = window.find(operator)
        if index >= 0 and (best is None or index < best[0]):
            best = (index, operator)
    if best is None:
        return ""
    remainder = window[best[0] + len(best[1]):]
    remainder = re.sub(r"^\s*(?:the|any|such|all|each|every)\s+", "", remainder, flags=re.IGNORECASE)
    words = re.split(r"[^A-Za-z0-9]+", remainder)
    kept: list[str] = []
    for word in words:
        if not word:
            continue
        if word.lower() in _PHRASE_STOP_WORDS:
            break
        kept.append(word)
    return " ".join(kept[:3])


def _threshold_phrase_before(value_text: str, threshold_text: str) -> str:
    """Returns the base phrase of a threshold definition."""
    index = value_text.lower().find((threshold_text or "").lower())
    if index < 0:
        return value_text
    return value_text[:index]


def build_clause_scoped_computation(
    text: str,
    spans: list[dict],
    edges: list[dict],
    clause_ranges: list[tuple[int, int]] | None = None,
    currency_code: str | None = None,
) -> tuple[dict | None, dict[str, dict], list[Diagnostic]]:
    """Builds a clause-scoped pipeline for a multi-clause statute.

    Returns ``(computation, inputs_signature, diagnostics)``. ``computation`` is
    ``None`` when a diagnostic with severity ``error`` was raised: the linker
    fails closed rather than guessing.
    """
    scopes, diagnostics, spans_by_idx = partition_clauses(spans, clause_ranges)
    cross_refs = attach_edges(scopes, edges, spans_by_idx)
    if not scopes:
        diagnostics.append(
            Diagnostic(
                code=ERR_NO_COMPUTATION,
                severity="error",
                clause_idx=-1,
                detail="the document contains no clause scope",
            )
        )
        return None, {}, diagnostics

    # â”€â”€ 1. definitions, substitutions and uses per clause â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    plans: list[_ClausePlan] = []
    for scope in scopes:
        clause_text = text[scope.char_range[0]:scope.char_range[1]]
        symbols, matches = extract_definitions(scope, clause_text)
        substitution = detect_substitution(clause_text)
        scope.definitions = symbols
        plan = _ClausePlan(
            scope=scope,
            text=clause_text,
            symbols=symbols,
            matches=matches,
            substitution=substitution,
        )
        plans.append(plan)

    all_symbols: list[Symbol] = [s for plan in plans for s in plan.symbols]
    by_term: dict[str, list[Symbol]] = {}
    for symbol in all_symbols:
        by_term.setdefault(symbol.id, []).append(symbol)

    # â”€â”€ 2. statutory overrides: a substitution bumps the SSA version â”€â”€â”€â”€â”€â”€â”€
    overrides: list[Symbol] = []
    for plan in plans:
        substitution = plan.substitution
        if not substitution:
            continue
        if substitution["old_minor"] is None:
            diagnostics.append(
                Diagnostic(
                    code=ERR_UNRESOLVED_REFERENCE,
                    severity="error",
                    clause_idx=plan.scope.index,
                    detail=(
                        f"substituted value {substitution['old_text']!r} is not a "
                        "monetary amount"
                    ),
                )
            )
            continue
        candidates = [
            symbol
            for symbol in all_symbols
            if symbol.minor is not None
            and symbol.minor == substitution["old_minor"]
            and symbol.defined_in < plan.scope.index
        ]
        if not candidates:
            diagnostics.append(
                Diagnostic(
                    code=ERR_UNRESOLVED_REFERENCE,
                    severity="error",
                    clause_idx=plan.scope.index,
                    detail=(
                        f"no earlier definition matches the substituted amount "
                        f"{substitution['old_text']!r}"
                    ),
                )
            )
            continue
        if len(candidates) > 1:
            diagnostics.append(
                Diagnostic(
                    code=ERR_AMBIGUOUS_REFERENCE,
                    severity="error",
                    clause_idx=plan.scope.index,
                    detail=(
                        f"{len(candidates)} definitions match the substituted amount "
                        f"{substitution['old_text']!r}"
                    ),
                )
            )
            continue
        target = candidates[0]
        override = Symbol(
            id=target.id,
            kind="TERM",
            semantic_type=target.semantic_type,
            defined_in=plan.scope.index,
            name=target.name,
            aliases=list(target.aliases),
            value_expr=substitution["new_text"],
            version=target.version + 1,
            origin="override",
            minor=substitution["new_minor"],
        )
        override.ssa_name = ssa_name(plan.scope.index, f"{override.name}Paisa", override.version)
        overrides.append(override)
        plan.symbols.append(override)

    # â”€â”€ 3. definition graph and cycle detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    value_references: dict[str, list[str]] = {}
    for plan in plans:
        for match in plan.matches:
            symbol: Symbol = match["symbol"]
            if symbol.semantic_type != "term":
                continue
            referenced = normalize_term(match["value_text"])
            if referenced in by_term:
                value_references.setdefault(symbol.id, []).append(referenced)
    cycle = find_cycle(sorted(by_term), value_references)
    if cycle:
        diagnostics.append(
            Diagnostic(
                code=ERR_CYCLIC_DEFINITION,
                severity="error",
                clause_idx=-1,
                detail="cyclic definition: " + " -> ".join(cycle),
            )
        )

    # â”€â”€ 4. emission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # ── 3. identify section-level / parent base operand for inheritance ─────────
    # If prior clauses establish an unambiguous monetary base operand, child clauses
    # without an explicit local base can inherit it (preserving Gate A2.4).
    # Defined output terms (e.g. LHS of "the term 'taxable income' means...") are
    # outputs of the section, NOT candidate input bases for deductions or rates.
    defined_term_cues = {
        normalize_term(s.id) for s in all_symbols
    } | {
        normalize_term(s.name) for s in all_symbols
    }
    for match in re.finditer(r'the\s+term\s+["\']?([^"\']+)["\']?\s+means', text, re.IGNORECASE):
        defined_term_cues.add(normalize_term(match.group(1)))

    candidate_bases = []
    for plan in plans:
        for span in plan.scope.spans:
            if span.role == "ROLE_COMPUTATION_OPERAND":
                norm_span = normalize_term(span.text)
                if norm_span in defined_term_cues:
                    continue
                bid = input_id_from_phrase(span.text, currency_code)
                if bid and bid not in candidate_bases:
                    candidate_bases.append(bid)
    # Check whole section header/text for canonical legal subject nouns
    if not candidate_bases:
        header_text = text[:min(len(text), 600)].lower()
        for cue, canonical_id in (
            ("taxable income", "taxableIncome"),
            ("gross income", "grossIncome"),
            ("wages", "wages"),
            ("turnover", "turnover"),
            ("invoice", "invoiceAmount"),
            ("purchase", "purchaseAmount"),
        ):
            if cue in defined_term_cues or normalize_term(canonical_id) in defined_term_cues:
                continue
            if cue in header_text and canonical_id not in candidate_bases:
                candidate_bases.append(canonical_id)

    # Gate A2.4: unambiguous single base candidate, otherwise fail closed or leave None
    unambiguous_base_id = candidate_bases[0] if len(candidate_bases) == 1 else None

    inputs: dict[str, dict] = {}
    steps: list[dict] = []
    outputs: dict[str, str] = {}
    used_names: set[str] = set()

    def declare_input(input_id: str, input_type: str) -> str:
        if input_id not in inputs:
            if input_type == "money":
                ccy = currency_code
                if not ccy or ccy == "CURRENCY":
                    ccy = "USD"
                inputs[input_id] = {
                    "type": "MONEY_MINOR",
                    "currency": ccy,
                    "scale": 100,
                    "required": True,
                }
            else:
                inputs[input_id] = {"type": "INTEGER", "required": True}
        return f"inputs.{input_id}"

    def unique_output(base: str) -> str:
        name = base
        suffix = 2
        while name in used_names:
            name = f"{base}{suffix}"
            suffix += 1
        used_names.add(name)
        return name

    output_keys: dict[str, str] = {}

    def assign_output(base: str, ref: str, owner: Symbol) -> str:
        """Exports a symbol. A statutory override replaces the earlier export
        of the same term, because the exported value is the highest version."""
        existing = output_keys.get(base)
        if existing is not None and owner.origin == "override":
            outputs[existing] = ref
            return existing
        key = unique_output(base)
        output_keys[base] = key
        outputs[key] = ref
        return key

    def resolve_by_name(phrase: str, clause_idx: int) -> tuple[Symbol | None, str | None]:
        """RFC-012 4.5: exact normalized-phrase match to one symbol."""
        term = normalize_term(phrase)
        if not term:
            return None, "empty reference"
        candidates = by_term.get(term, [])
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            return None, (
                f"{len(candidates)} definitions normalize to {term!r}: "
                + ", ".join(f"clause {c.defined_in}" for c in candidates)
            )
        return None, None

    def symbol_reference(symbol: Symbol) -> str:
        return f"vars.{symbol.ssa_name}"

    for plan in plans:
        scope = plan.scope
        emitted_symbols: list[Symbol] = []

        # 4a. window for a substitution clause
        window_name = None
        if plan.substitution:
            date_spans = [s for s in scope.spans if s.role in _DATE_ROLES]
            if date_spans:
                input_id = "taxYear" if "year" in plan.text.lower() else "year"
                year_ref = declare_input(input_id, "integer")
                comparisons = []
                for span in date_spans:
                    year_match = re.search(r"\b((?:19|20)\d{2})\b", span.text)
                    if not year_match:
                        continue
                    year = int(year_match.group(1))
                    operator = _operator_before(plan.text, span.char_range[0] - scope.char_range[0])
                    if operator in ("after",):
                        comparisons.append({"op": "GT", "left": year_ref, "right": year})
                    elif operator in ("on or after",):
                        comparisons.append({"op": "GTE", "left": year_ref, "right": year})
                    elif operator in ("before", "not before"):
                        comparisons.append({"op": "LT", "left": year_ref, "right": year})
                    elif operator in ("on or before", "not later than", "no later than"):
                        comparisons.append({"op": "LTE", "left": year_ref, "right": year})
                    else:
                        comparisons.append({"op": "GT", "left": year_ref, "right": year})
                if len(comparisons) == 1:
                    window_expr = comparisons[0]
                elif comparisons:
                    window_expr = {
                        "op": "AND",
                        "left": comparisons[0],
                        "right": comparisons[1] if len(comparisons) == 2 else {
                            "op": "AND", "left": comparisons[1], "right": comparisons[2]
                        },
                    }
                else:
                    window_expr = None
                if window_expr is not None:
                    window_name = ssa_name(scope.index, "inWindow", 1)
                    steps.append({"assign": window_name, "expr": window_expr})
                    plan.steps.append({"assign": window_name, "expr": window_expr})
                    plan.outputs.append(window_name)

        # 4b. definitions
        for match in plan.matches:
            symbol: Symbol = match["symbol"]
            value_text = match["value_text"]
            minor = match["minor"]
            referenced = None
            ref_error = None
            if minor is None:
                referenced, ref_error = resolve_by_name(value_text, scope.index)
            # "the amount exceeding fifty lakh rupees" defines a quantity
            # above a threshold, not the threshold itself.
            threshold_value = bool(_COMPARATIVE_RE.search(value_text))
            if minor is not None and not threshold_value:
                name = ssa_name(scope.index, f"{symbol.name}Paisa", symbol.version)
                symbol.ssa_name = name
                expr = {"op": "TO_PAISE", "value": str(minor)}
                steps.append({"assign": name, "expr": expr})
                plan.steps.append({"assign": name, "expr": expr})
                emitted_symbols.append(symbol)
                assign_output(f"{symbol.name}Paisa", f"vars.{name}", symbol)
            elif referenced is not None:
                name = ssa_name(scope.index, f"{symbol.name}Paisa", symbol.version)
                symbol.ssa_name = name
                expr = symbol_reference(referenced)
                steps.append({"assign": name, "expr": expr})
                plan.steps.append({"assign": name, "expr": expr})
                emitted_symbols.append(symbol)
                assign_output(f"{symbol.name}Paisa", f"vars.{name}", symbol)
            elif ref_error:
                diagnostics.append(
                    Diagnostic(
                        code=ERR_AMBIGUOUS_REFERENCE,
                        severity="error",
                        clause_idx=scope.index,
                        detail=f"{ref_error} (referenced by {symbol.id!r})",
                        symbol=symbol.id,
                    )
                )
                continue
            else:
                # A definition whose value is a threshold: the amount above the
                # threshold is the defined quantity.
                threshold = None
                for span in scope.spans:
                    if span.role in ("ROLE_TRANSACTION_CLIFF_THRESHOLD",
                                     "ROLE_TRANSACTION_EXCESS_THRESHOLD"):
                        candidate = parse_monetary_minor_units(span.text)
                        if candidate is not None and (
                            threshold is None or candidate > threshold
                        ):
                            threshold = candidate
                if threshold is None:
                    # If the definition references an external statutory provision (e.g. "under subsection...",
                    # "under section..."), it declares an input parameter rather than failing closed.
                    if re.search(r"\bunder\s+(?:subsection|section|paragraph|clause|subparagraph)\b", value_text, re.IGNORECASE):
                        input_id = camel_name(symbol.id)
                        name = ssa_name(scope.index, f"{symbol.name}Paisa", symbol.version)
                        symbol.ssa_name = name
                        in_ref = declare_input(input_id, "money")
                        steps.append({"assign": name, "expr": in_ref})
                        plan.steps.append({"assign": name, "expr": in_ref})
                        emitted_symbols.append(symbol)
                        assign_output(f"{symbol.name}Paisa", f"vars.{name}", symbol)
                        continue
                    diagnostics.append(
                        Diagnostic(
                            code=ERR_UNRESOLVED_REFERENCE,
                            severity="error",
                            clause_idx=scope.index,
                            detail=(
                                f"definition {symbol.id!r} has no resolvable value "
                                f"({value_text!r})"
                            ),
                            symbol=symbol.id,
                        )
                    )
                    continue
                threshold_name = ssa_name(scope.index, f"{symbol.name}ThresholdPaisa", symbol.version)
                threshold_expr = {"op": "TO_PAISE", "value": str(threshold)}
                steps.append({"assign": threshold_name, "expr": threshold_expr})
                plan.steps.append({"assign": threshold_name, "expr": threshold_expr})

                base_phrase = _threshold_phrase_before(value_text, _threshold_text(scope))
                base_id = input_id_from_phrase(base_phrase, currency_code)
                if not base_id:
                    diagnostics.append(
                        Diagnostic(
                            code=ERR_UNRESOLVED_REFERENCE,
                            severity="error",
                            clause_idx=scope.index,
                            detail=f"definition {symbol.id!r} has no base amount",
                            symbol=symbol.id,
                        )
                    )
                    continue
                base_ref = declare_input(base_id, "money")
                name = ssa_name(scope.index, f"{symbol.name}Paisa", symbol.version)
                symbol.ssa_name = name
                expr = {
                    "op": "CONDITIONAL",
                    "test": {
                        "op": "GT",
                        "left": base_ref,
                        "right": {"op": "TO_PAISE", "value": str(threshold)},
                    },
                    "then": base_ref,
                    "else": {"op": "TO_PAISE", "value": "0"},
                }
                steps.append({"assign": name, "expr": expr})
                plan.steps.append({"assign": name, "expr": expr})
                emitted_symbols.append(symbol)
                assign_output(f"{symbol.name}Paisa", f"vars.{name}", symbol)

        # 4c. statutory override
        for override in overrides:
            if override.defined_in != scope.index:
                continue
            target = next(
                s for s in all_symbols
                if s.minor is not None and s.minor == plan.substitution["old_minor"]
                and s.defined_in < scope.index
            )
            name = override.ssa_name
            then_expr = {"op": "TO_PAISE", "value": str(override.minor)}
            else_expr = f"vars.{target.ssa_name}"
            if window_name:
                expr = {
                    "op": "CONDITIONAL",
                    "test": f"vars.{window_name}",
                    "then": then_expr,
                    "else": else_expr,
                }
            else:
                expr = {"op": "CONDITIONAL", "test": True, "then": then_expr, "else": else_expr}
            steps.append({"assign": name, "expr": expr})
            plan.steps.append({"assign": name, "expr": expr})
            assign_output(f"{override.name}Paisa", f"vars.{name}", override)

        # 4d. arithmetic composition (RFC-012 Step 3): one topological walk
        #     over the relation graph of this clause. The walk replaces the
        #     "first matching archetype wins" chain whenever the clause has
        #     composed relations.
        from compiler.neural.ssa_walker import SsaEdge, SsaNode, compose_clause

        known_symbols = {}
        for s in all_symbols:
            if s.ssa_name:
                known_symbols[normalize_term(s.id)] = f"vars.{s.ssa_name}"
                if s.name:
                    known_symbols[normalize_term(s.name)] = f"vars.{s.ssa_name}"

        arithmetic_edges = [
            e for e in scope.edges if e.relation in _WALKED_RELATIONS
        ]
        if arithmetic_edges:
            walker_nodes = [
                SsaNode(
                    index=s.span_idx,
                    role=s.role,
                    text=s.text,
                    char_range=s.char_range,
                )
                for s in scope.spans
            ]
            walker_edges = [
                SsaEdge(
                    source=e.source.span_idx,
                    target=e.target.span_idx,
                    relation=e.relation,
                )
                for e in arithmetic_edges
            ]
            walker_steps, handlers, walker_diagnostics = compose_clause(
                nodes=walker_nodes,
                edges=walker_edges,
                clause_text=plan.text,
                clause_idx=scope.index,
                currency_code=currency_code,
                inputs=inputs,
                inherited_base_id=unambiguous_base_id,
                candidate_bases=candidate_bases,
                known_symbols=known_symbols,
            )
            if walker_diagnostics:
                for diagnostic in walker_diagnostics:
                    diagnostics.append(
                        Diagnostic(
                            code=diagnostic.code,
                            severity="error",
                            clause_idx=scope.index,
                            detail=diagnostic.detail,
                        )
                    )
                continue
            for step in walker_steps:
                steps.append(step)
                plan.steps.append(step)
            plan.handlers = handlers
            if walker_steps:
                final = walker_steps[-1]["assign"]
                outputs[unique_output(_output_key_from_assign(final))] = f"vars.{final}"
                plan.outputs.append(final)
            continue

        # 4e. a plain rate clause (no composed relation)
        rate_span = next(
            (
                s for s in scope.spans
                if s.role in _RATE_ROLES and parse_unit_and_type(s.text)[0] == "PERCENTAGE_RATE"
            ),
            None,
        )
        if rate_span is not None:
            # Guard against silent degradation of an excess-over-threshold clause into a plain rate.
            if re.search(
                r"\b(?:excess\s+(?:of)?|in\s+excess\s+of|over\s+the|over\s+such|exceeding|exceeds|more\s+than|greater\s+than)\b",
                plan.text,
                re.IGNORECASE,
            ):
                diagnostics.append(
                    Diagnostic(
                        code="ERR_NO_THRESHOLD",
                        severity="error",
                        clause_idx=scope.index,
                        detail="clause specifies an excess calculation over a threshold, but no threshold was provided",
                    )
                )
                continue
            bps = parse_rate_bps(rate_span.text)
            if bps:
                plan.rate_bps = bps
                phrase = base_phrase_after_rate(
                    plan.text, rate_span.char_range[1] - scope.char_range[0]
                )
                referenced, error = resolve_by_name(phrase, scope.index)
                if error:
                    diagnostics.append(
                        Diagnostic(
                            code=ERR_AMBIGUOUS_REFERENCE,
                            severity="error",
                            clause_idx=scope.index,
                            detail=f"{error} (used by the rate in clause {scope.index})",
                        )
                    )
                    continue
                if referenced is not None and referenced.ssa_name:
                    base_ref = symbol_reference(referenced)
                    base_name = referenced.name
                    cross_refs.append(
                        (scope.index, phrase, referenced.defined_in, referenced.id)
                    )
                else:
                    base_id = input_id_from_phrase(phrase, currency_code)
                    if not base_id:
                        if unambiguous_base_id:
                            base_id = unambiguous_base_id
                        elif len(candidate_bases) > 1:
                            diagnostics.append(
                                Diagnostic(
                                    code=ERR_AMBIGUOUS_REFERENCE,
                                    severity="error",
                                    clause_idx=scope.index,
                                    detail=f"multiple candidate base amounts {candidate_bases} for clause {scope.index}",
                                )
                            )
                            continue
                    if not base_id:
                        diagnostics.append(
                            Diagnostic(
                                code=ERR_UNRESOLVED_REFERENCE,
                                severity="error",
                                clause_idx=scope.index,
                                detail=f"the rate in clause {scope.index} has no base amount",
                            )
                        )
                        continue
                    base_ref = declare_input(base_id, "money")
                    base_name = camel_name(base_id)
                name = ssa_name(scope.index, f"{base_name}TaxPaisa", 1)
                expr = {
                    "op": "RATIONAL_MULTIPLY",
                    "base": base_ref,
                    "rateBps": bps,
                    "denominator": 10000,
                }
                steps.append({"assign": name, "expr": expr})
                plan.steps.append({"assign": name, "expr": expr})
                outputs[unique_output(f"{base_name}TaxPaisa")] = f"vars.{name}"

    if any(d.severity == "error" for d in diagnostics):
        return None, inputs, diagnostics

    if not steps:
        diagnostics.append(
            Diagnostic(
                code=ERR_NO_COMPUTATION,
                severity="error",
                clause_idx=-1,
                detail="no clause produced an executable computation",
            )
        )
        return None, inputs, diagnostics

    clause_scopes = []
    for plan in plans:
        clause_kind = "lowered" if plan.steps else ("skipped" if plan.kind == "reference" else plan.kind)
        clause_scopes.append(
            {
                "clauseIdx": plan.scope.index,
                "charRange": list(plan.scope.char_range),
                "kind": clause_kind,
                "standardRateBps": plan.rate_bps,
                "handlers": list(plan.handlers),
                "steps": [step["assign"] for step in plan.steps],
                "outputs": list(plan.outputs),
                "symbols": [
                    {
                        "id": symbol.id,
                        "name": symbol.name,
                        "kind": symbol.kind,
                        "semanticType": symbol.semantic_type,
                        "definedIn": symbol.defined_in,
                        "version": symbol.version,
                        "ssaName": symbol.ssa_name,
                        "origin": symbol.origin,
                    }
                    for symbol in plan.symbols
                ],
            }
        )

    lowered_count = sum(1 for plan in plans if plan.steps)
    total_count = len(plans)
    computation = {
        "archetype": "DECLARATIVE_PIPELINE",
        "clauseScopes": clause_scopes,
        "clauseCoverage": {
            "lowered": lowered_count,
            "total": total_count,
            "skipped": total_count - lowered_count,
        },
        "pipeline": {"steps": steps, "output": outputs},
    }
    return computation, inputs, diagnostics


def _threshold_text(scope: ClauseScope) -> str:
    for span in scope.spans:
        if span.role in ("ROLE_TRANSACTION_CLIFF_THRESHOLD", "ROLE_TRANSACTION_EXCESS_THRESHOLD"):
            return span.text
    return ""





# ============================================================================
# Model v2 Lexical Scope Tree & Two-Pass Symbol Linker
# ============================================================================

from .literals import (
    infer_rule_currency_info as _cl2_infer_currency,
    parse_iso_date_str as _cl2_parse_iso_date,
    parse_minor_units as _cl2_parse_minor_units,
    _derive_fact_key_camel as _cl2_fact_key_camel,
)
from .ssa_walker import (
    V2LoweringError,
    _v2_norm_edge,
    _v2_parse_span_value,
    _v2_topo_sort_indices,
    compose_clause_v2,
)


def _v2_detect_scope_level(clause_text: str) -> tuple[int, str]:
    """Determines statutory structural indentation level from leading marker without hardcoded rules."""
    stripped = (clause_text or "").lstrip()
    if not stripped.startswith("("):
        return 0, "section"
    close_idx = stripped.find(")")
    if close_idx <= 1 or close_idx > 6:
        return 0, "section"
    marker = stripped[1:close_idx]
    if marker.isdigit():
        return 2, "paragraph"
    if marker.islower() and marker in {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"} and len(marker) >= 2:
        return 3, "subparagraph"
    if marker.islower() and marker.isalpha():
        return 1, "subsection"
    if marker.isupper() and marker.isalpha():
        return 4, "clause"
    return 0, "section"


def _v2_build_lexical_scope_tree(clauses_meta: list[dict]) -> dict[int, dict]:
    """Builds parent-pointer lexical scope tree (Section -> Subsection -> Paragraph -> Subparagraph)."""
    tree: dict[int, dict] = {}
    stack: list[tuple[int, int]] = []  # [(level, clause_idx)]
    for cm in sorted(clauses_meta, key=lambda c: int(c["clauseIdx"])):
        cidx = int(cm["clauseIdx"])
        level, kind = _v2_detect_scope_level(cm.get("text", ""))
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_idx = stack[-1][1] if stack else None
        tree[cidx] = {
            "clauseIdx": cidx,
            "level": level,
            "kind": kind,
            "parent": parent_idx,
            "symbols": {},  # norm_term -> symbol descriptor
        }
        stack.append((level, cidx))
    return tree


def _v2_lookup_scope_symbol(tree: dict[int, dict], clause_idx: int, norm_key: str) -> dict | None:
    """Resolves a symbol up the lexical parent chain (local -> parent -> root)."""
    cur = clause_idx
    visited = set()
    while cur is not None and cur in tree and cur not in visited:
        visited.add(cur)
        syms = tree[cur]["symbols"]
        if norm_key in syms:
            return syms[norm_key]
        cur = tree[cur]["parent"]
    return None


def _v2_extract_year_window(clause_spans: list[dict], clause_text: str) -> tuple[int | None, int | None]:
    """Extracts inclusive integer year window [start_year, end_year] from UNIT_YEAR_DATE spans."""
    date_spans = [
        sp for sp in sorted(clause_spans, key=lambda s: int((s.get("charRange") or [0, 0])[0]))
        if sp.get("unit") == "UNIT_YEAR_DATE"
    ]
    if not date_spans:
        return None, None
    parsed_bounds: list[tuple[str, int, str]] = []
    for sp in date_spans:
        raw = str(sp.get("span") or "")
        c_start = int((sp.get("charRange") or [0, 0])[0])
        # Inspect immediate 20 chars before span for strict 'after'/'before' calendar wording
        prefix = clause_text[max(0, c_start - 20):c_start].lower()
        iso = _cl2_parse_iso_date(raw)
        if iso:
            yr, mo, dy = (int(p) for p in iso.split("-"))
            if "after" in prefix:
                bound_yr = yr + 1 if (mo == 12 and dy == 31) else yr
                parsed_bounds.append(("LOWER", bound_yr, iso))
            elif "before" in prefix:
                bound_yr = yr - 1 if (mo == 1 and dy == 1) else yr
                parsed_bounds.append(("UPPER", bound_yr, iso))
            else:
                parsed_bounds.append(("POINT", yr, iso))
        else:
            p = _v2_parse_span_value(sp)
            for yr in p.get("years", []):
                if "after" in prefix:
                    parsed_bounds.append(("LOWER", yr, str(yr)))
                elif "before" in prefix:
                    parsed_bounds.append(("UPPER", yr, str(yr)))
                else:
                    parsed_bounds.append(("POINT", yr, str(yr)))

    lowers = [b[1] for b in parsed_bounds if b[0] == "LOWER"]
    uppers = [b[1] for b in parsed_bounds if b[0] == "UPPER"]
    points = [b[1] for b in parsed_bounds if b[0] == "POINT"]
    start_yr = lowers[0] if lowers else (points[0] if len(points) >= 1 else None)
    end_yr = uppers[0] if uppers else (points[1] if len(points) >= 2 else None)
    return start_yr, end_yr


def build_clause_scoped_computation_v2(
    text: str,
    spans: list[dict],
    edges: list[dict],
    clause_modalities: list[dict] | None = None,
    currency_code: str | None = None,
) -> dict:
    """Two-pass lexical-scope linker + bottom-up topological AST builder for Model v2."""
    norm_spans = sorted(
        [dict(s) for s in spans],
        key=lambda sp: (
            int((sp.get("charRange") or [0, 0])[0]),
            int((sp.get("charRange") or [0, 0])[1]),
            str(sp.get("role") or ""),
            str(sp.get("unit") or ""),
        ),
    )
    span_by_idx: dict[int, dict] = {}
    for pos, sp in enumerate(norm_spans):
        sid = int(sp.get("spanIdx", pos))
        sp["spanIdx"] = sid
        span_by_idx[sid] = sp

    valid_ids = set(span_by_idx.keys())
    norm_edges = []
    seen_e = set()
    for e in edges:
        ne = _v2_norm_edge(e, valid_ids)
        if ne is not None and ne not in seen_e:
            seen_e.add(ne)
            norm_edges.append(ne)
    norm_edges.sort(key=lambda item: (item[2], item[0], item[1]))

    clauses_meta = []
    if clause_modalities:
        for cm in sorted(clause_modalities, key=lambda c: int(c.get("clauseIdx", 0))):
            cidx = int(cm.get("clauseIdx", 0))
            cr = cm.get("charRange") or [0, len(text)]
            clauses_meta.append({
                "clauseIdx": cidx,
                "charRange": cr,
                "text": text[int(cr[0]):int(cr[1])],
            })
    else:
        max_c = max((int(sp.get("clauseIdx", 0)) for sp in norm_spans), default=0)
        for cidx in range(max_c + 1):
            clauses_meta.append({"clauseIdx": cidx, "charRange": [0, len(text)], "text": text})

    spans_by_clause: dict[int, list[dict]] = {int(cm["clauseIdx"]): [] for cm in clauses_meta}
    for sp in norm_spans:
        cidx = int(sp.get("clauseIdx", 0))
        spans_by_clause.setdefault(cidx, []).append(sp)
        if cidx not in {int(cm["clauseIdx"]) for cm in clauses_meta}:
            clauses_meta.append({"clauseIdx": cidx, "charRange": [0, len(text)], "text": ""})

    scope_tree = _v2_build_lexical_scope_tree(clauses_meta)
    ccy = currency_code or _cl2_infer_currency(norm_spans, text)[0] or "USD"

    # Identify pure header/operand-declaration clauses (e.g. "§1. Tax imposed on taxable income.")
    declaration_base_var: str | None = None
    declaration_clauses: set[int] = set()
    for cidx, cspans in spans_by_clause.items():
        units = {sp.get("unit", "UNIT_NONE") for sp in cspans}
        roles = {sp.get("role", "ROLE_IGNORE") for sp in cspans}
        has_calc_unit = bool(units & {"UNIT_MONEY", "UNIT_RATE", "UNIT_BOOL_PREDICATE", "UNIT_COUNT"})
        has_operative_edge = any(
            rel in (
                "REL_DEFINED_TERM_OF",
                "REL_SUBSTITUTE_WHEN",
                "REL_AGGREGATE_SUM",
                "REL_AGGREGATE_COUNT",
                "REL_QUANTIFIER_FORALL",
                "REL_QUANTIFIER_EXISTS",
                "REL_LOGICAL_AND",
                "REL_LOGICAL_OR",
                "REL_LOGICAL_NOT",
                "REL_RATE_TO_THRESHOLD",
            )
            and (span_by_idx[s].get("clauseIdx") == cidx or span_by_idx[t].get("clauseIdx") == cidx)
            for s, t, rel in norm_edges
        )
        if cspans and not has_calc_unit and not has_operative_edge and "ROLE_COMPUTATION_OPERAND" in roles:
            op_sp = next(sp for sp in cspans if sp.get("role") == "ROLE_COMPUTATION_OPERAND")
            declaration_base_var = _cl2_fact_key_camel(op_sp.get("span", ""), "primaryBaseAmount")
            declaration_clauses.add(cidx)

    # ── PASS 1: Harvest REL_DEFINED_TERM_OF & REL_SUBSTITUTE_WHEN + Scope Conflict Detection ──
    defined_clause_targets: dict[int, dict] = {}
    substitutions: list[dict] = []
    symbol_versions: dict[str, int] = {}

    def_edges = [e for e in norm_edges if e[2] == "REL_DEFINED_TERM_OF"]
    for s_id, t_id, _ in def_edges:
        sp_s, sp_t = span_by_idx[s_id], span_by_idx[t_id]
        # Term side is UNIT_SYMBOL_VAR or ROLE_DEFINED_TERM; fallback earliest span
        if sp_t.get("unit") == "UNIT_SYMBOL_VAR" and sp_s.get("unit") != "UNIT_SYMBOL_VAR":
            term_sp, val_sp = sp_t, sp_s
        else:
            term_sp, val_sp = sp_s, sp_t
        cidx = int(term_sp.get("clauseIdx", 0))
        norm_key = _cl2_fact_key_camel(term_sp.get("span", ""), "definedValue")
        existing_same = scope_tree[cidx]["symbols"].get(norm_key)
        if existing_same is not None and existing_same["clauseIdx"] != cidx:
            raise V2LoweringError(f"ERR_AMBIGUOUS_DEFINITION: symbol {norm_key!r} redefined in same lexical scope {cidx}")
        # Also detect two separate top-level definitions of same term across sibling root scopes
        for other_cidx, node in scope_tree.items():
            if other_cidx != cidx and node["parent"] == scope_tree[cidx]["parent"] and norm_key in node["symbols"]:
                # Allowed only if later clause has a REL_SUBSTITUTE_WHEN, checked below; for sibling definitions fail closed
                is_sub = any(
                    rel == "REL_SUBSTITUTE_WHEN" and (
                        int(span_by_idx[es].get("clauseIdx", 0)) == cidx
                        or int(span_by_idx[et].get("clauseIdx", 0)) == cidx
                    )
                    for es, et, rel in norm_edges
                )
                if not is_sub:
                    raise V2LoweringError(
                        f"ERR_AMBIGUOUS_DEFINITION: conflicting definitions of {norm_key!r} in sibling clauses {other_cidx} and {cidx}"
                    )

        ver = symbol_versions.get(norm_key, 0) + 1
        symbol_versions[norm_key] = ver
        base_out = f"{norm_key}Paisa" if not norm_key.endswith("Paisa") else norm_key
        ssa_var = f"clause_{cidx}_{base_out}_v{ver}"
        rec = {
            "symbolKey": norm_key,
            "exportKey": base_out,
            "ref": f"vars.{ssa_var}",
            "ssaVar": ssa_var,
            "clauseIdx": cidx,
            "version": ver,
            "valSpanId": val_sp["spanIdx"],
        }
        scope_tree[cidx]["symbols"][norm_key] = rec
        defined_clause_targets[cidx] = rec

    sub_edges = [e for e in norm_edges if e[2] == "REL_SUBSTITUTE_WHEN"]
    for s_id, t_id, _ in sub_edges:
        sp_s, sp_t = span_by_idx[s_id], span_by_idx[t_id]
        src_c = int(sp_s.get("clauseIdx", 0))
        tgt_c = int(sp_t.get("clauseIdx", 0))
        override_sp = sp_s if sp_s.get("unit") == "UNIT_MONEY" else sp_t
        sub_cidx = max(src_c, tgt_c)
        minor_val = _cl2_parse_minor_units(override_sp.get("span"))
        if minor_val is None:
            continue
        # Match target symbol via tgt span or nearest defined symbol
        tgt_key = _cl2_fact_key_camel(sp_t.get("span", ""), "")
        target_rec = defined_clause_targets.get(min(src_c, tgt_c))
        if target_rec is None and tgt_key:
            target_rec = _v2_lookup_scope_symbol(scope_tree, sub_cidx, tgt_key)
        if target_rec is None and defined_clause_targets:
            target_rec = next(iter(defined_clause_targets.values()))
        if target_rec is None:
            continue
        substitutions.append({
            "clauseIdx": sub_cidx,
            "targetRec": target_rec,
            "overrideMinor": minor_val,
        })

    # ── PASS 2: Window-Bounded Coref + Cross-Clause DAG & Topological Lowering ──
    resolved_corefs: dict[int, dict] = {}
    clause_adj: dict[int, set[int]] = {int(cm["clauseIdx"]): set() for cm in clauses_meta}

    for s_id, t_id, rel in norm_edges:
        sc = int(span_by_idx[s_id].get("clauseIdx", 0))
        tc = int(span_by_idx[t_id].get("clauseIdx", 0))
        if rel == "REL_COREF_LINK":
            # Strictly bounded to same clause or immediately preceding clause (0 <= sc - tc <= 1)
            if 0 <= (sc - tc) <= 1:
                if tc in defined_clause_targets:
                    rec = defined_clause_targets[tc]
                    resolved_corefs[s_id] = {"targetRec": rec, "baseKey": rec["symbolKey"]}
                    if sc != tc:
                        clause_adj[sc].add(tc)
        elif sc != tc and rel in ("REL_DEFINED_TERM_OF", "REL_SUBSTITUTE_WHEN", "REL_RATE_TO_THRESHOLD", "REL_SUBTRACTION_BASE"):
            direction = "target_before_source" if rel != "REL_SUBTRACTION_BASE" else "target_before_source"
            if direction == "target_before_source":
                clause_adj[sc].add(tc)

    for sub in substitutions:
        sc = sub["clauseIdx"]
        tc = sub["targetRec"]["clauseIdx"]
        if sc != tc:
            clause_adj[sc].add(tc)

    ordered_clauses = _v2_topo_sort_indices(clause_adj)

    all_steps: list[dict] = []
    all_inputs: dict[str, dict] = {}
    all_outputs: dict[str, str] = {}
    clause_scopes: list[dict] = []
    sub_clause_set = {s["clauseIdx"] for s in substitutions}
    clause_text_by_idx = {int(cm["clauseIdx"]): cm["text"] for cm in clauses_meta}

    for cidx in ordered_clauses:
        cspans = spans_by_clause.get(cidx, [])
        if cidx in declaration_clauses:
            clause_scopes.append({
                "clauseIdx": cidx,
                "scopeDepth": scope_tree[cidx]["level"],
                "parentScope": scope_tree[cidx]["parent"],
                "kind": "declaration",
                "handlers": [],
                "standardRateBps": None,
            })
            continue

        if cidx in sub_clause_set:
            sub = next(s for s in substitutions if s["clauseIdx"] == cidx)
            tgt = sub["targetRec"]
            start_yr, end_yr = _v2_extract_year_window(cspans, text)
            all_inputs.setdefault("taxYear", {"type": "INTEGER"})
            if start_yr is not None and end_yr is not None:
                cond_expr = {
                    "op": "AND",
                    "left": {"op": "GTE", "left": "inputs.taxYear", "right": int(start_yr)},
                    "right": {"op": "LTE", "left": "inputs.taxYear", "right": int(end_yr)},
                }
            else:
                cond_expr = True
            win_var = f"clause_{cidx}_inWindow_v1"
            all_steps.append({"assign": win_var, "expr": cond_expr})

            new_ver = symbol_versions.get(tgt["symbolKey"], 1) + 1
            symbol_versions[tgt["symbolKey"]] = new_ver
            new_var = f"clause_{cidx}_{tgt['exportKey']}_v{new_ver}"
            all_steps.append({
                "assign": new_var,
                "expr": {
                    "op": "CONDITIONAL",
                    "test": f"vars.{win_var}",
                    "then": {"op": "TO_PAISE", "value": str(sub["overrideMinor"])},
                    "else": tgt["ref"],
                },
            })
            all_outputs[tgt["exportKey"]] = f"vars.{new_var}"
            tgt["ref"] = f"vars.{new_var}"
            tgt["version"] = new_ver
            clause_scopes.append({
                "clauseIdx": cidx,
                "scopeDepth": scope_tree[cidx]["level"],
                "parentScope": scope_tree[cidx]["parent"],
                "kind": "substitution",
                "handlers": ["window", "substitution-bump"],
                "standardRateBps": None,
            })
            continue

        if not cspans:
            continue

        # Collect visible symbol environment walking up parent-pointer chain
        visible_env: dict[str, dict] = {}
        cur = cidx
        visited = set()
        while cur is not None and cur in scope_tree and cur not in visited:
            visited.add(cur)
            for k, rec in scope_tree[cur]["symbols"].items():
                if k not in visible_env:
                    visible_env[k] = rec
            cur = scope_tree[cur]["parent"]

        # Pass both in-clause edges and cross-clause edges whose endpoints map cleanly
        in_edges = [
            e for e in edges
            if (
                int(span_by_idx[int(e.get("sourceIdx", e.get("source", -1)))].get("clauseIdx", -1)) == cidx
                and int(span_by_idx[int(e.get("targetIdx", e.get("target", -1)))].get("clauseIdx", -1)) == cidx
            )
            if (e.get("sourceIdx") is not None or e.get("source") is not None)
            and (e.get("targetIdx") is not None or e.get("target") is not None)
            and int(e.get("sourceIdx", e.get("source"))) in span_by_idx
            and int(e.get("targetIdx", e.get("target"))) in span_by_idx
        ]

        live_corefs = {
            sid: {
                "refVar": entry["targetRec"]["ref"] if "targetRec" in entry else entry["refVar"],
                "baseKey": entry["baseKey"],
            }
            for sid, entry in resolved_corefs.items()
        }

        cmod_by_idx = {
            int(cm.get("clauseIdx", 0)): cm.get("modality")
            for cm in (clause_modalities or [])
            if isinstance(cm, dict)
        }
        try:
            comp = compose_clause_v2(
                clause_idx=cidx,
                clause_spans=cspans,
                all_edges=in_edges,
                full_text=text,
                currency_code=ccy,
                symbol_env=visible_env,
                resolved_corefs=live_corefs,
                fallback_base_var=declaration_base_var,
                clause_modality=cmod_by_idx.get(cidx),
            )
        except V2LoweringError:
            # Skip non-operative header clauses only when other operative clauses exist; re-raise fatal errors
            raise

        all_steps.extend(comp["steps"])
        all_inputs.update(comp["inputs"])
        out_var_ref = f"vars.{comp['outputVar']}"
        if cidx in defined_clause_targets:
            rec = defined_clause_targets[cidx]
            exp_key = rec["symbolKey"] if comp.get("resultType") == "BOOLEAN" else rec["exportKey"]
            all_outputs[exp_key] = out_var_ref
            rec["ref"] = out_var_ref
        else:
            all_outputs[comp["outputKey"]] = out_var_ref

        clause_scopes.append({
            "clauseIdx": cidx,
            "scopeDepth": scope_tree[cidx]["level"],
            "parentScope": scope_tree[cidx]["parent"],
            "kind": "computation",
            "handlers": comp["handlers"],
            "standardRateBps": comp["standardRateBps"],
            "resultType": comp["resultType"],
        })

    if not all_steps or not all_outputs:
        raise V2LoweringError("ERR_NO_COMPUTATION: no operative computation or predicate steps emitted")

    return {
        "inputs": all_inputs,
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "mode": "DECLARATIVE_PIPELINE",
            "pipeline": {
                "steps": all_steps,
                "output": all_outputs,
            },
            "clauseScopes": sorted(clause_scopes, key=lambda s: int(s["clauseIdx"])),
        },
        "currency": ccy,
    }
