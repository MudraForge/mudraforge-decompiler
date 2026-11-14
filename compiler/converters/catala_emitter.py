"""
MudraForge Statutory Decompiler - Canonical Catala Language Emitter
===================================================================
Lowers the canonical USVM ``DECLARATIVE_PIPELINE`` computation AST produced by
the dual-stream neural decompiler into literate Catala source (``.catala_en``).

The lowering is a faithful, operator-by-operator translation of the executable
pipeline evaluated by ``@mudraforge/statutory/usvm`` (RFC-009 Rev 3.1):

* Monetary values are exact ``money`` amounts. The USVM computes in integer
  paise and quantizes with commercial half-up rounding; Catala ``money``
  arithmetic rounds half away from zero to the cent, which is the same rule.
* Basis-point rates are ``integer`` values; ``RATIONAL_MULTIPLY`` with the
  statutory denominator of 10000 lowers to multiplication by an exact decimal
  percentage.
* Unsupported archetypes, operators, fact keys or rounding policies raise
  :class:`CatalaLoweringError`. The emitter never fabricates a rate, threshold
  or verdict that is not present in the executable AST.

Catala Reference:
  Merigoux, D., Chataing, N., & Protzenko, J. (2021).
  "Catala: A Programming Language for the Law", ACM PACMPL 5(ICFP):1-29.
  Specification: https://catala-lang.org/
"""

import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from compiler.converters.errors import CatalaLoweringError
from compiler.converters.archetype_strategies import normalize_ast
from compiler.converters.fact_bindings import FACT_BINDINGS as _FACT_BINDINGS


# Catala reserved words that must never be used as identifiers.
_CATALA_KEYWORDS = frozenset("""
scope declaration definition rule under condition consequence fulfilled
equals assertion fixed by decreases increases varies with state label
exception anything of if then else let in and or xor not true false
money decimal integer boolean date duration text struct enumeration
match pattern external module alias as nil among such that exists for
all we have an the input output internal context data content
""".split())

# Catala has no string type. Statutory text facts (customer type, country,
# goods category) are modelled as enumerations whose cases are documented by
# the statutory source (see tests/fixtures/french_tax_2026/*/circulars).
# Maps enum type name -> {IR literal value: Catala constructor name}.
_ENUM_DOMAINS: dict[str, dict[str, str]] = {
    "CustomerType": {"B2B": "B2B", "B2C": "B2C"},
    # FRANCE is distinct from EU: domestic B2B is not a reverse charge and
    # domestic B2C is not an export.
    "CustomerCountry": {"FRANCE": "France", "EU": "EU", "NON_EU": "NonEu"},
    "GoodsCategory": {
        "standard": "Standard",
        "intermediate": "Intermediate",
        "reduced": "Reduced",
        "superReduced": "SuperReduced",
        "zero": "Zero",
    },
}

_SUPPORTED_ARCHETYPES = frozenset({"DECLARATIVE_PIPELINE"})

_COMPARISON_OPS = {"EQ": "=", "NEQ": "!=", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}


def _to_snake_case(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", str(name or "")).strip()
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[\s\-]+", "_", s)
    return s.lower() or "statutory_rule"


def _to_pascal_case(name: str) -> str:
    clean = re.sub(r"[^\w\s\-]", " ", str(name or "")).strip()
    words = re.split(r"[\s_\-]+", clean)
    return "".join(w.capitalize() for w in words if w) or "StatutoryScope"


def _scope_identifier(name: str) -> str:
    """Returns a valid Catala scope name for an arbitrary rule/scenario name."""
    scope = _to_pascal_case(name)
    if not re.match(r"^[A-Z]", scope):
        scope = f"S{scope}"
    return scope


def _identifier(name: str) -> str:
    ident = _to_snake_case(name)
    if ident in _CATALA_KEYWORDS:
        ident = f"v_{ident}"
    if not re.match(r"^[a-z_]", ident):
        ident = f"v_{ident}"
    return ident


def _catala_type(typ: str) -> str:
    """Maps an internal type tag to a Catala type name."""
    if typ.startswith("enum:"):
        return typ[len("enum:"):]
    return typ


def _default_enum_constructor(literal: str) -> str:
    """Converts a bare uppercase/snake IR enum literal into a Catala UpperCamel constructor."""
    parts = [p for p in literal.strip().split("_") if p]
    if not parts:
        raise CatalaLoweringError(f"Empty enumeration literal {literal!r}")
    out_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"[A-Z]{2,3}", part) or re.fullmatch(r"[A-Z]\d[A-Za-z0-9]*", part):
            out_parts.append(part)
        else:
            out_parts.append(part[0].upper() + part[1:].lower())
    ctor = "".join(out_parts)
    if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", ctor):
        raise CatalaLoweringError(f"Invalid enumeration literal {literal!r} for Catala")
    return ctor


def _enum_constructor(
    enum_name: str,
    literal: str,
    domains: dict[str, dict[str, str]] | None = None,
) -> str:
    """Maps an IR literal to its declared Catala enumeration constructor."""
    source = domains if domains is not None else _ENUM_DOMAINS
    domain = source.get(enum_name)
    if domain is None:
        raise CatalaLoweringError(f"Unknown enumeration {enum_name!r}")
    constructor = domain.get(literal)
    if constructor is None:
        raise CatalaLoweringError(
            f"{literal!r} is not a case of enumeration {enum_name}; "
            f"expected one of {sorted(domain)}"
        )
    return constructor


def _money_literal(value: Fraction | int | str) -> str:
    """Render a major-unit amount as a Catala money literal.

    Catala money literals are written in major units: ``$1.23`` denotes 1
    major unit and 23 minor units, and ``decimal of $1.00`` evaluates to 1.0.
    Therefore a value expressed in minor units (paise/cents) must first be
    divided by 100 -- use :func:`_money_literal_minor`.
    """
    frac = value if isinstance(value, Fraction) else Fraction(str(value))
    sign = "-" if frac < 0 else ""
    frac = abs(frac)
    paise_num, rem = divmod(frac.numerator * 100, frac.denominator)
    if rem:
        raise CatalaLoweringError(
            f"Monetary literal {value} has sub-paisa precision; the USVM operates in whole paise"
        )
    whole, cents = divmod(paise_num, 100)
    return f"{sign}${whole}.{cents:02d}"


def _money_literal_minor(value: Fraction | int | str) -> str:
    """Render an amount expressed in minor units (paise/cents) as Catala money."""
    frac = value if isinstance(value, Fraction) else Fraction(str(value))
    return _money_literal(frac / 100)


def _decimal_literal(value: Fraction | int | float | str) -> str:
    frac = Fraction(str(value))
    sign = "-" if frac < 0 else ""
    frac = abs(frac)
    whole, rem = divmod(frac.numerator, frac.denominator)
    if rem == 0:
        return f"{sign}{whole}.0"
    # Denominators here are products of 2 and 5 (bps / 10000 / 100), so the
    # decimal expansion terminates; 12 places covers all statutory rates.
    digits = f"{frac.numerator * 10**12 // frac.denominator:012d}".rstrip("0")
    return f"{sign}{whole}.{digits}"


def _number(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise CatalaLoweringError(f"Boolean literal {value} used in numeric position")
    if isinstance(value, (int, float, Fraction)):
        return Fraction(str(value))
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", value.strip()):
        return Fraction(value.strip())
    raise CatalaLoweringError(f"Not a numeric literal: {value!r}")


class CatalaEmitter:
    """
    Lowers a canonical USVM DECLARATIVE_PIPELINE AST into literate Catala
    (.catala_en) source, optionally appending executable scenario scopes with
    concrete fact values for compiler-verified regression testing.
    """

    def __init__(self, indent_size: int = 2):
        self.indent = " " * indent_size

    # ── Public API ───────────────────────────────────────────────────────────

    def emit(
        self,
        ast_or_result: dict[str, Any],
        title: str | None = None,
        scenarios: list[dict[str, Any]] | None = None,
    ) -> str:
        ast = ast_or_result.get("synthesizedAst", ast_or_result)
        # Normalize non-canonical archetype layouts (e.g. French tax 2026
        # fixtures) into DECLARATIVE_PIPELINE before lowering. Canonical ASTs
        # pass through unchanged.
        ast = normalize_ast(ast)
        computation = ast.get("computation") or {}
        archetype = computation.get("archetype")
        if archetype not in _SUPPORTED_ARCHETYPES:
            raise CatalaLoweringError(
                f"Unsupported computation archetype {archetype!r}: only the canonical "
                "DECLARATIVE_PIPELINE emitted by synthesize_usvm_ast can be lowered to Catala"
            )
        pipeline = computation.get("pipeline") or {}
        steps = pipeline.get("steps")
        outputs = pipeline.get("output")
        if not isinstance(steps, list) or not steps:
            raise CatalaLoweringError("DECLARATIVE_PIPELINE is missing executable pipeline.steps")
        if not isinstance(outputs, dict) or not outputs:
            raise CatalaLoweringError("DECLARATIVE_PIPELINE is missing pipeline.output map")

        self._computation = computation
        self._vars: dict[str, str] = {}          # USVM var name -> catala type
        self._inputs: dict[str, str] = {}        # catala field name -> catala type
        self._output_types: dict[str, str] = {}  # catala output name -> catala type
        # Typed-input signature (RFC-012 v2 profile). ``inputs.*`` references
        # resolve against this map; ``_input_ids`` remembers the original id so
        # scenario facts can be supplied under either the id or the field name.
        self._input_specs: dict[str, dict[str, Any]] = ast.get("inputs") or {}
        self._input_ids: dict[str, str] = {}
        self._input_minor: set[str] = set()
        self._saw_inputs = False
        self._saw_facts = False
        self._enum_domains: dict[str, dict[str, str]] = {
            name: dict(cases) for name, cases in _ENUM_DOMAINS.items()
        }
        self._register_dynamic_enums(steps, outputs, scenarios)

        # Type inference pass over steps, then output expressions.
        step_types = []
        for step in steps:
            var = step.get("assign") or step.get("name")
            if not var:
                raise CatalaLoweringError(f"Pipeline step without assign target: {step!r}")
            expr = step.get("expr")
            null_path = self._find_null_path(expr)
            if null_path is not None:
                raise CatalaLoweringError(
                    f"ERR_NULL_EXPRESSION: null expression in step {var!r} at {null_path}"
                )
            typ = self._infer(expr)
            self._vars[var] = typ
            step_types.append((var, expr, typ))
        output_entries = []
        for key, expr in outputs.items():
            null_path = self._find_null_path(expr)
            if null_path is not None:
                raise CatalaLoweringError(
                    f"ERR_NULL_EXPRESSION: null expression in output {key!r} at {null_path}"
                )
            name = _identifier(key)
            typ = self._infer(expr)
            if name in self._output_types and self._output_types[name] != typ:
                raise CatalaLoweringError(f"Output name collision on {name!r}")
            self._output_types[name] = typ
            output_entries.append((name, expr, typ))
        if self._saw_inputs and self._saw_facts:
            raise CatalaLoweringError(
                "ERR_MIXED_INPUT_NAMESPACES: a typed-input AST must reference only "
                "inputs.*, never facts.*; the legacy fact registry is not consulted "
                "for v2 ASTs"
            )
        # A step variable re-exported under the same name is declared once, as
        # an output; Catala forbids duplicate variable declarations.
        internal_steps = []
        for var, expr, typ in step_types:
            name = _identifier(var)
            if name in self._output_types:
                if self._output_types[name] != typ:
                    raise CatalaLoweringError(
                        f"Step variable {var!r} and output {name!r} have conflicting types"
                    )
                continue
            internal_steps.append((var, expr, typ))

        rule_id = ast.get("sectionId") or ast.get("ruleId") or "StatutoryRule"
        scope_name = _scope_identifier(rule_id)
        struct_name = f"{scope_name}Input"
        raw_text = ast.get("rawText") or ast_or_result.get("text") or ""

        # Canonical statutory constants are optional: archetypes without a
        # single threshold or flat rate must not emit misleading placeholders.
        has_threshold = "thresholdPaisa" in computation
        has_rate = "standardRateBps" in computation
        threshold = _number(computation.get("thresholdPaisa", 0))
        rate_bps = _number(computation.get("standardRateBps", 0))
        if rate_bps.denominator != 1:
            raise CatalaLoweringError(f"standardRateBps must be an integer, got {rate_bps}")

        lines: list[str] = []
        doc_title = title or f"Statutory Formalization: {scope_name}"
        lines.append(f"# {doc_title}")
        lines.append("")
        if raw_text:
            clean_prose = raw_text.strip().replace("\n", " ")
            lines.append("> **Legislative Provision:**")
            lines.append(f'> "{clean_prose}"')
            lines.append("")
        lines.append("```catala")

        # Enumerations for text-valued facts. Catala has no string type, so
        # statutory categories are declared as enumerations before use.
        used_types = set(self._inputs.values()) | set(self._vars.values()) | set(
            self._output_types.values()
        )
        enum_names = sorted(
            typ[len("enum:"):] for typ in used_types if typ.startswith("enum:")
        )
        for enum_name in enum_names:
            domain = self._enum_domains.get(enum_name)
            if not domain:
                raise CatalaLoweringError(
                    f"Enumeration {enum_name!r} has no declared or inferred cases"
                )
            lines.append(f"declaration enumeration {enum_name}:")
            for constructor in domain.values():
                lines.append(f"{self.indent}-- {constructor}")
            lines.append("")

        # Input structure: only facts the executable pipeline actually reads.
        lines.append(f"declaration structure {struct_name}:")
        if not self._inputs:
            raise CatalaLoweringError("Pipeline reads no facts; nothing to lower")
        for field, typ in self._inputs.items():
            lines.append(f"{self.indent}data {field} content {_catala_type(typ)}")
        lines.append("")

        # Scope declaration.
        lines.append(f"declaration scope {scope_name}:")
        lines.append(f"{self.indent}input input_data content {struct_name}")
        if has_threshold:
            lines.append(f"{self.indent}internal threshold_paisa content money")
        if has_rate:
            lines.append(f"{self.indent}internal standard_rate_bps content integer")
        for var, _, typ in internal_steps:
            lines.append(f"{self.indent}internal {_identifier(var)} content {_catala_type(typ)}")
        for name, _, typ in output_entries:
            lines.append(f"{self.indent}output {name} content {_catala_type(typ)}")
        lines.append("")

        # Scope body: statutory constants, sequential steps, output map.
        lines.append(f"scope {scope_name}:")
        if has_threshold:
            # `thresholdPaisa` is expressed in minor units (paise/cents), so it
            # is converted to major units before becoming a Catala money literal.
            lines.append(
                f"{self.indent}definition threshold_paisa equals {_money_literal_minor(threshold)}"
            )
        if has_rate:
            lines.append(f"{self.indent}definition standard_rate_bps equals {int(rate_bps)}")
        defined_names = {_identifier(var) for var, _, _ in step_types}
        for var, expr, typ in step_types:
            rendered = self._render(expr, typ)
            lines.append(f"{self.indent}definition {_identifier(var)} equals {rendered}")
        for name, expr, typ in output_entries:
            if name in defined_names:
                # Re-export of a step variable under its own name: the step
                # definition already defines this scope variable.
                continue
            rendered = self._render(expr, typ)
            lines.append(f"{self.indent}definition {name} equals {rendered}")

        # Executable scenario scopes with concrete facts (compiler-verified tests).
        for idx, scenario in enumerate(scenarios or [], start=1):
            lines.append("")
            lines.extend(
                self._emit_scenario(scenario, idx, scope_name, struct_name, output_entries)
            )

        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    # ── Scenario emission ────────────────────────────────────────────────────

    def _emit_scenario(
        self,
        scenario: dict[str, Any],
        idx: int,
        scope_name: str,
        struct_name: str,
        output_entries: list[tuple[str, Any, str]],
    ) -> list[str]:
        name = _scope_identifier(scenario.get("name") or f"Scenario{idx}")
        facts = scenario.get("facts") or {}
        lines = [f"declaration scope {name}:", f"{self.indent}output result scope {scope_name}", ""]
        lines.append(f"scope {name}:")
        lines.append(f"{self.indent}definition result.input_data equals {struct_name} {{")
        rendered_fields = []
        for field, typ in self._inputs.items():
            source_key = self._input_ids.get(field)
            if source_key is not None and source_key in facts:
                value = facts[source_key]
            elif field in facts:
                value = facts[field]
            else:
                raise CatalaLoweringError(
                    f"Scenario {name!r} is missing fact {field!r} required by the pipeline"
                )
            rendered_fields.append(
                f"{self.indent}{self.indent}-- {field}: "
                f"{self._render_fact_value(value, typ, minor=field in self._input_minor)}"
            )
        lines.append("\n".join(rendered_fields))
        lines.append(f"{self.indent}}}")
        return lines

    def _render_fact_value(self, value: Any, typ: str, minor: bool = False) -> str:
        # Legacy scenario facts are expressed in major currency units. A v2
        # MONEY_MINOR input carries integer minor units instead, so it uses
        # the minor-unit literal helper.
        if typ == "money":
            return _money_literal_minor(_number(value)) if minor else _money_literal(_number(value))
        if typ == "integer":
            frac = _number(value)
            if frac.denominator != 1:
                raise CatalaLoweringError(f"Integer fact with fractional value: {value!r}")
            return str(int(frac))
        if typ == "decimal":
            return _decimal_literal(_number(value))
        if typ == "boolean":
            if not isinstance(value, bool):
                raise CatalaLoweringError(f"Boolean fact with non-boolean value: {value!r}")
            return "true" if value else "false"
        if typ == "date":
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
                raise CatalaLoweringError(f"Date fact requires ISO YYYY-MM-DD string, got {value!r}")
            return f"|{value.strip()}|"
        if typ.startswith("enum:"):
            if not isinstance(value, str):
                raise CatalaLoweringError(
                    f"Enumeration fact of type {typ} requires a string value, got {value!r}"
                )
            return _enum_constructor(typ[len("enum:"):], value, domains=self._enum_domains)
        raise CatalaLoweringError(f"Unsupported fact type {typ!r}")

    def _register_enum_case(self, enum_name: str, literal: str) -> None:
        if not isinstance(literal, str):
            return
        stripped = literal.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", stripped):
            return
        if stripped in ("true", "false") or stripped.startswith(("inputs.", "facts.", "vars.")):
            return
        domain = self._enum_domains.setdefault(enum_name, {})
        if stripped not in domain:
            domain[stripped] = _default_enum_constructor(stripped)

    def _register_dynamic_enums(
        self,
        steps: list[dict[str, Any]],
        outputs: dict[str, Any],
        scenarios: list[dict[str, Any]] | None,
    ) -> None:
        input_enums: dict[str, str] = {}
        for key, spec in self._input_specs.items():
            raw_type = str(spec.get("type") or "").strip()
            if raw_type.upper().startswith("ENUM:"):
                enum_name = raw_type.split(":", 1)[1].strip()
                if not enum_name:
                    continue
                self._enum_domains.setdefault(enum_name, {})
                input_enums[key] = enum_name
                spec_id = spec.get("id")
                if isinstance(spec_id, str) and spec_id:
                    input_enums[spec_id] = enum_name
                for val in (spec.get("values") or spec.get("cases") or spec.get("enumValues") or []):
                    self._register_enum_case(enum_name, val)

        for k, spec in _FACT_BINDINGS.items():
            ct = str(spec[1] if isinstance(spec, (tuple, list)) and len(spec) >= 2 else getattr(spec, "catala_type", "") or "")
            if ct.startswith("enum:"):
                input_enums.setdefault(k, ct[len("enum:"):])

        def _enum_for_ref(operand: Any) -> str | None:
            if not isinstance(operand, str):
                return None
            if operand.startswith("inputs."):
                return input_enums.get(operand[len("inputs."):])
            if operand.startswith("facts."):
                return input_enums.get(operand[len("facts."):])
            return None

        def _walk(node: Any) -> None:
            if isinstance(node, list):
                for item in node:
                    _walk(item)
                return
            if not isinstance(node, dict):
                return
            op = node.get("op")
            if op in ("EQ", "NEQ", "IN", "NOT_IN"):
                left = node.get("left", node.get("value"))
                right = node.get("right", node.get("values", node.get("items", node.get("list"))))
                l_enum = _enum_for_ref(left)
                r_enum = _enum_for_ref(right)
                if l_enum and isinstance(right, str):
                    self._register_enum_case(l_enum, right)
                elif l_enum and isinstance(right, list):
                    for item in right:
                        if isinstance(item, str):
                            self._register_enum_case(l_enum, item)
                elif r_enum and isinstance(left, str):
                    self._register_enum_case(r_enum, left)
            for v in node.values():
                _walk(v)

        _walk(steps)
        _walk(outputs)
        for scenario in (scenarios or []):
            facts = scenario.get("inputs") or scenario.get("facts") or {}
            if isinstance(facts, dict):
                for fk, fv in facts.items():
                    if fk in input_enums and isinstance(fv, str):
                        self._register_enum_case(input_enums[fk], fv)

    # ── Type inference ───────────────────────────────────────────────────────

    @staticmethod
    def _find_null_path(node: Any, path: str = "expr") -> str | None:
        """Returns the path of the first null expression, or None."""
        if node is None:
            return path
        if isinstance(node, dict):
            for key, value in node.items():
                found = CatalaEmitter._find_null_path(value, f"{path}.{key}")
                if found is not None:
                    return found
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found = CatalaEmitter._find_null_path(value, f"{path}[{index}]")
                if found is not None:
                    return found
        return None

    def _infer(self, node: Any) -> str:
        if node is None:
            raise CatalaLoweringError(
                "ERR_NULL_EXPRESSION: null expression in pipeline"
            )
        if isinstance(node, bool):
            return "boolean"
        if isinstance(node, (int, float)):
            return "integer" if isinstance(node, int) else "decimal"
        if isinstance(node, str):
            return self._infer_reference(node)
        if not isinstance(node, dict):
            raise CatalaLoweringError(f"Unsupported expression node: {node!r}")

        op = node.get("op") or node.get("operator")
        if op == "LITERAL":
            return self._infer(node["value"])
        if op in ("VARIABLE",) or (op is None and ("var" in node or "path" in node)):
            return self._infer_reference(node.get("var") or node.get("path"))

        if op in _COMPARISON_OPS:
            # Traverse both operands: this registers every fact they read and
            # rejects undefined references or mismatched types before any
            # Catala is written. Returning early left condition-only facts
            # undeclared in the input structure.
            self._comparison_operand_type(node["left"], node["right"])
            return "boolean"
        if op in ("AND", "OR", "XOR", "IMPLIES"):
            left, right = self._binary_operands(node)
            for operand in (left, right):
                operand_t = self._infer(operand)
                if operand_t != "boolean":
                    raise CatalaLoweringError(
                        f"{op} operand must be boolean, got {operand_t}"
                    )
            return "boolean"
        if op == "NOT":
            operand = node.get("operand", node.get("value"))
            if self._infer(operand) != "boolean":
                raise CatalaLoweringError("NOT operand must be boolean")
            return "boolean"
        if op in ("ADD", "SUB"):
            return self._unify_numeric(self._infer(node["left"]), self._infer(node["right"]), op)
        if op == "MUL":
            return self._unify_numeric(self._infer(node["left"]), self._infer(node["right"]), op)
        if op == "DIV":
            lt = self._infer(node["left"])
            rt = self._infer(node["right"])
            # Dividing money (integer minor units) by an integer stays in
            # minor units — there are no fractional paisa/cents.
            if lt == "money" and rt == "integer":
                return "money"
            return "decimal"
        if op in ("MIN", "MAX"):
            left, right = self._binary_operands(node)
            return self._unify_numeric(self._infer(left), self._infer(right), op)
        if op == "ABS":
            typ = self._infer(node.get("value", node.get("operand")))
            if typ not in ("money", "integer", "decimal"):
                raise CatalaLoweringError(f"ABS on non-numeric type {typ}")
            return typ
        if op == "TO_PAISE":
            operand_t = self._infer(node.get("value", node.get("operand")))
            if operand_t not in ("integer", "decimal", "money"):
                raise CatalaLoweringError(
                    f"TO_PAISE operand must be numeric, got {operand_t}"
                )
            return "money"
        if op == "FROM_PAISE":
            if self._infer(node.get("value", node.get("operand"))) != "money":
                raise CatalaLoweringError("FROM_PAISE expects a paise-denominated money value")
            return "money"
        if op == "RATIONAL_MULTIPLY":
            if self._infer(node["base"]) != "money":
                raise CatalaLoweringError("RATIONAL_MULTIPLY base must be money")
            if self._infer(node.get("rateBps", node.get("rate"))) != "integer":
                raise CatalaLoweringError("RATIONAL_MULTIPLY rateBps must be an integer")
            return "money"
        if op == "CONDITIONAL":
            if self._infer(node["test"]) != "boolean":
                raise CatalaLoweringError("CONDITIONAL test must be boolean")
            then_t = self._infer(node.get("then", node.get("consequent")))
            else_t = self._infer(node.get("else", node.get("alternate")))
            return self._unify_numeric(then_t, else_t, "CONDITIONAL") if then_t != else_t else then_t
        if op == "ROUND":
            policy = node.get("policy") or "COMMERCIAL_PAISA"
            if policy not in ("COMMERCIAL_PAISA", "NONE"):
                raise CatalaLoweringError(
                    f"ROUND policy {policy!r} has no faithful Catala lowering"
                )
            return self._infer(node["value"])
        if op == "PROGRESSIVE_SLABS":
            return self._infer_progressive_slabs(node)
        raise CatalaLoweringError(f"Unsupported pipeline operator {op!r}")

    def _infer_reference(self, path: str) -> str:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", path.strip()):
            return "integer" if "." not in path else "decimal"
        if path.startswith("inputs."):
            key = path[len("inputs."):]
            if key not in self._input_specs:
                raise CatalaLoweringError(
                    f"ERR_UNDECLARED_INPUT: {key!r} is not declared in the inputs signature"
                )
            self._saw_inputs = True
            spec = self._input_specs[key] or {}
            typ = self._input_catala_type(key, spec)
            field = _identifier(key)
            self._inputs[field] = typ
            self._input_ids[field] = key
            if str(spec.get("type", "")).upper() == "MONEY_MINOR":
                self._input_minor.add(field)
            return typ
        if path.startswith("facts."):
            self._saw_facts = True
            key = path[len("facts."):]
            if key not in _FACT_BINDINGS:
                raise CatalaLoweringError(
                    f"Fact {key!r} has no canonical Catala binding; the runtime injects only "
                    f"{sorted(_FACT_BINDINGS)}"
                )
            field, typ = _FACT_BINDINGS[key]
            self._inputs[field] = typ
            return typ
        if path.startswith("rules."):
            key = path[len("rules."):]
            if key == "thresholdPaisa":
                if "thresholdPaisa" not in self._computation:
                    raise CatalaLoweringError(
                        "Pipeline reads rules.thresholdPaisa but the AST declares no threshold"
                    )
                return "money"
            if key in ("standardRateBps", "effectiveRateBps"):
                if "standardRateBps" not in self._computation:
                    raise CatalaLoweringError(
                        f"Pipeline reads rules.{key} but the AST declares no standard rate"
                    )
                return "integer"
            raise CatalaLoweringError(f"Rule constant {key!r} has no canonical Catala binding")
        if path.startswith("vars."):
            key = path[len("vars."):]
            if key not in self._vars:
                raise CatalaLoweringError(f"Reference to undefined pipeline variable {key!r}")
            return self._vars[key]
        if path in self._vars:
            return self._vars[path]
        # Bare text literal (e.g. "B2B", "standard") — not a reference path.
        # Checked last so variable names in self._vars take priority.
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", path.strip()):
            return "text"
        raise CatalaLoweringError(f"Unresolvable expression reference {path!r}")

    def _input_catala_type(self, key: str, spec: dict[str, Any]) -> str:
        """Maps a v2 input signature entry to a Catala type tag."""
        raw = spec.get("type")
        if not isinstance(raw, str) or not raw.strip():
            raise CatalaLoweringError(
                f"ERR_UNSUPPORTED_INPUT_TYPE: input {key!r} declares no type"
            )
        upper = raw.strip().upper()
        if upper in ("MONEY_MINOR", "MONEY_MAJOR", "MONEY"):
            return "money"
        if upper in ("INTEGER", "INT", "COUNT", "RATE_BPS"):
            return "integer"
        if upper in ("DECIMAL", "NUMBER"):
            return "decimal"
        if upper in ("BOOLEAN", "BOOL"):
            return "boolean"
        if upper == "DATE":
            return "date"
        if upper.startswith("ENUM:"):
            return "enum:" + raw.split(":", 1)[1]
        raise CatalaLoweringError(
            f"ERR_UNSUPPORTED_INPUT_TYPE: input {key!r} declares unsupported type {raw!r}"
        )

    def _infer_progressive_slabs(self, node: dict[str, Any]) -> str:
        """Validates a PROGRESSIVE_SLABS node against the USVM contract."""
        if self._infer(node["base"]) != "money":
            raise CatalaLoweringError("PROGRESSIVE_SLABS base must be money")
        slabs = node.get("slabs")
        if not isinstance(slabs, list) or not slabs:
            raise CatalaLoweringError("PROGRESSIVE_SLABS requires a non-empty slabs list")
        for idx, slab in enumerate(slabs):
            if not isinstance(slab, dict):
                raise CatalaLoweringError(f"PROGRESSIVE_SLABS slab {idx} is not an object")
            rate = slab.get("rateBps", slab.get("rate"))
            if rate is None:
                raise CatalaLoweringError(f"PROGRESSIVE_SLABS slab {idx} has no rateBps")
            if self._infer(rate) != "integer":
                raise CatalaLoweringError(
                    f"PROGRESSIVE_SLABS slab {idx} rateBps must be an integer"
                )
            for key in ("floor", "upTo", "baseTax"):
                child = slab.get(key)
                if child is not None and self._infer(child) != "money":
                    raise CatalaLoweringError(
                        f"PROGRESSIVE_SLABS slab {idx} {key} must be money"
                    )
        return "money"

    @staticmethod
    def _unify_numeric(a: str, b: str, op: str) -> str:
        if a == b:
            if a == "boolean":
                raise CatalaLoweringError(f"{op} on boolean operands")
            return a
        pair = {a, b}
        if pair == {"money", "integer"}:
            return "money"
        if pair == {"money", "decimal"}:
            return "money"
        if pair == {"integer", "decimal"}:
            return "decimal"
        raise CatalaLoweringError(f"{op} on incompatible types {a} and {b}")

    @staticmethod
    def _binary_operands(node: dict[str, Any]) -> tuple[Any, Any]:
        if "operands" in node:
            operands = node["operands"]
            if not isinstance(operands, list) or len(operands) != 2:
                raise CatalaLoweringError(
                    f"{node.get('op')} with {len(operands) if isinstance(operands, list) else '?'} "
                    "operands; only binary form is supported"
                )
            return operands[0], operands[1]
        return node["left"], node["right"]

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self, node: Any, typ: str) -> str:
        if isinstance(node, bool):
            if typ != "boolean":
                raise CatalaLoweringError("Boolean literal in non-boolean position")
            return "true" if node else "false"
        if isinstance(node, (int, float)):
            return self._render_number(Fraction(str(node)), typ)
        if isinstance(node, str):
            return self._render_reference(node, typ)
        if not isinstance(node, dict):
            raise CatalaLoweringError(f"Unsupported expression node: {node!r}")

        op = node.get("op") or node.get("operator")
        if op == "LITERAL":
            return self._render(node["value"], typ)
        if op == "VARIABLE" or (op is None and ("var" in node or "path" in node)):
            return self._render_reference(node.get("var") or node.get("path"), typ)

        if op in _COMPARISON_OPS:
            left, right = node["left"], node["right"]
            operand_t = self._comparison_operand_type(left, right)
            lhs = self._render(left, operand_t)
            rhs = self._render(right, operand_t)
            return f"({lhs} {_COMPARISON_OPS[op]} {rhs})"
        if op in ("AND", "OR"):
            left, right = self._binary_operands(node)
            joiner = "and" if op == "AND" else "or"
            return f"({self._render(left, 'boolean')} {joiner} {self._render(right, 'boolean')})"
        if op == "NOT":
            operand = node.get("operand", node.get("value"))
            return f"(not {self._render(operand, 'boolean')})"
        if op in ("ADD", "SUB", "MUL"):
            left, right = node["left"], node["right"]
            lt, rt = self._infer(left), self._infer(right)
            result_t = self._unify_numeric(lt, rt, op)
            symbol = {"ADD": "+", "SUB": "-", "MUL": "*"}[op]
            if op == "MUL" and result_t == "money":
                money_side, rate_side = (left, right) if lt == "money" else (right, left)
                return (
                    f"({self._render(money_side, 'money')} * {self._as_decimal(rate_side)})"
                )
            return f"({self._render(left, result_t)} {symbol} {self._render(right, result_t)})"
        if op == "DIV":
            lt = self._infer(node["left"])
            rt = self._infer(node["right"])
            # Money divided by an integer stays money: Catala divides the
            # major-unit amount and keeps the money type.
            if lt == "money" and rt == "integer":
                return (
                    f"({self._render(node['left'], 'money')} / "
                    f"{self._render(node['right'], 'integer')})"
                )
            lhs = self._as_decimal(node["left"])
            rhs = self._as_decimal(node["right"])
            return f"({lhs} / {rhs})"
        if op in ("MIN", "MAX"):
            left, right = self._binary_operands(node)
            operand_t = self._unify_numeric(self._infer(left), self._infer(right), op)
            lhs = self._render(left, operand_t)
            rhs = self._render(right, operand_t)
            cmp_op = "<" if op == "MIN" else ">"
            return f"(if {lhs} {cmp_op} {rhs} then {lhs} else {rhs})"
        if op == "ABS":
            operand = node.get("value", node.get("operand"))
            inner_t = self._infer(operand)
            val = self._render(operand, inner_t)
            zero = self._render_number(Fraction(0), inner_t)
            return f"(if {val} < {zero} then ({zero} - {val}) else {val})"
        if op == "TO_PAISE":
            operand = node.get("value", node.get("operand"))
            frac = self._static_number(operand)
            if frac is not None:
                # RFC-012: a bare digit string is already minor units (the
                # USVM `toPaise` contract); a JSON number is legacy major
                # units. This is the single cast site for both profiles.
                if self._is_minor_unit_literal(operand):
                    return _money_literal_minor(frac)
                return _money_literal(frac)
            # Dynamic money value: Catala money is stored in minor units
            # (cents/paise) internally, so a money operand is already
            # minor-unit denominated. Render it as-is.
            typ = self._infer(operand)
            if typ == "money":
                return self._render(operand, "money")
            raise CatalaLoweringError(f"TO_PAISE of non-money dynamic value {typ}")
        if op == "FROM_PAISE":
            operand = node.get("value", node.get("operand"))
            # IR money values are already represented as Catala money, whose
            # `decimal of` view is the major-unit amount. Converting away from
            # minor units is therefore the identity here.
            return self._render(operand, "money")
        if op == "RATIONAL_MULTIPLY":
            base = self._render(node["base"], "money")
            rate = node.get("rateBps", node.get("rate"))
            denominator = node.get("denominator", 10000)
            denom_frac = self._static_number(denominator)
            if denom_frac is None:
                raise CatalaLoweringError("RATIONAL_MULTIPLY with dynamic denominator is not supported")
            rate_dec = self._as_decimal(rate)
            if denom_frac == 10000:
                return f"({base} * ({rate_dec} / 10000.0))"
            return f"({base} * ({rate_dec} / {_decimal_literal(denom_frac)}))"
        if op == "CONDITIONAL":
            test = self._render(node["test"], "boolean")
            then_v = self._render(node.get("then", node.get("consequent")), typ)
            else_v = self._render(node.get("else", node.get("alternate")), typ)
            return f"(if {test} then {then_v} else {else_v})"
        if op == "PROGRESSIVE_SLABS":
            return self._render_progressive_slabs(node)
        if op == "ROUND":
            # COMMERCIAL_PAISA / NONE: Catala money arithmetic already quantizes
            # to the paisa with the same half-up rule as the USVM.
            return self._render(node["value"], typ)
        raise CatalaLoweringError(f"Unsupported pipeline operator {op!r}")

    @classmethod
    def _is_minor_unit_literal(cls, node: Any) -> bool:
        """True when a static TO_PAISE operand uses the minor-unit convention."""
        if isinstance(node, str):
            return re.fullmatch(r"-?\d+", node.strip()) is not None
        if isinstance(node, dict):
            op = node.get("op") or node.get("operator")
            if op == "LITERAL":
                return cls._is_minor_unit_literal(node.get("value"))
        return False

    def _render_progressive_slabs(self, node: dict[str, Any]) -> str:
        """Lowers PROGRESSIVE_SLABS to a nested-let Catala expression.

        The expansion mirrors ``universal_evaluator.js`` (PROGRESSIVE_SLABS)
        exactly: the effective floor is the greater of the slab floor and the
        previous ceiling; the slab base is bounded by the slab ceiling; a
        non-positive slab base contributes nothing (and in particular never
        adds ``baseTax``); and an unbounded slab terminates the schedule.
        """
        slabs = node.get("slabs") or []
        base = self._render(node["base"], "money")
        zero = self._render_number(Fraction(0), "money")
        bindings: list[tuple[str, str]] = []
        tax_names: list[str] = []
        previous_ceiling: str | None = None
        for idx, slab in enumerate(slabs):
            prefix = f"slab_{idx}"
            ceiling_name = f"{prefix}_ceiling"
            previous_name = f"{prefix}_previous_ceiling"
            bindings.append(
                (previous_name, zero if previous_ceiling is None else previous_ceiling)
            )

            floor = slab.get("floor")
            if floor is not None:
                floor_expr = self._render(floor, "money")
                effective_floor = f"{prefix}_effective_floor"
                bindings.append(
                    (
                        effective_floor,
                        f"(if {floor_expr} > {previous_name} then {floor_expr} else {previous_name})",
                    )
                )
            else:
                effective_floor = previous_name

            up_to = slab.get("upTo")
            bindings.append(
                (ceiling_name, self._render(up_to, "money") if up_to is not None else base)
            )
            bounded_base = f"{prefix}_bounded_base"
            bindings.append(
                (
                    bounded_base,
                    f"(if {base} < {ceiling_name} then {base} else {ceiling_name})",
                )
            )
            slab_base = f"{prefix}_base"
            bindings.append((slab_base, f"({bounded_base} - {effective_floor})"))

            rate = slab.get("rateBps", slab.get("rate"))
            rate_tax = f"({slab_base} * ({self._as_decimal(rate)} / 10000.0))"
            base_tax = slab.get("baseTax")
            contribution = (
                f"({self._render(base_tax, 'money')} + {rate_tax})"
                if base_tax is not None
                else rate_tax
            )
            tax_name = f"{prefix}_tax"
            bindings.append(
                (tax_name, f"(if {slab_base} > {zero} then {contribution} else {zero})")
            )
            tax_names.append(tax_name)

            if up_to is None:
                # USVM breaks after the first unbounded slab.
                break
            previous_ceiling = ceiling_name

        if not tax_names:
            return zero
        total = tax_names[0]
        for name in tax_names[1:]:
            total = f"({total} + {name})"
        for name, expr in reversed(bindings):
            total = f"(let {name} equals {expr} in {total})"
        return total

    def _render_reference(self, path: str, typ: str) -> str:
        stripped = path.strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            return self._render_number(Fraction(stripped), typ)
        if path.startswith("inputs."):
            key = path[len("inputs."):]
            if key not in self._input_specs:
                raise CatalaLoweringError(
                    f"ERR_UNDECLARED_INPUT: {key!r} is not declared in the inputs signature"
                )
            return f"input_data.{_identifier(key)}"
        if path.startswith("facts."):
            field, _ = _FACT_BINDINGS[path[len("facts."):]]
            return f"input_data.{field}"
        if path.startswith("rules."):
            key = path[len("rules."):]
            if key == "thresholdPaisa":
                return "threshold_paisa"
            if key in ("standardRateBps", "effectiveRateBps"):
                # The canonical runtime binds effectiveRateBps to standardRateBps
                # unless a PAN-penalty override applies; overrides are outside the
                # pipeline and are not lowered here.
                return "standard_rate_bps"
            raise CatalaLoweringError(f"Rule constant {key!r} has no canonical Catala binding")
        if path.startswith("vars."):
            return _identifier(path[len("vars."):])
        if path in self._vars:
            return _identifier(path)
        # Bare text literal. Catala has no string type: it is only valid as an
        # enumeration case, resolved against the compared operand's enum type.
        if typ.startswith("enum:") and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", stripped):
            return _enum_constructor(typ[len("enum:"):], stripped, domains=self._enum_domains)
        raise CatalaLoweringError(f"Unresolvable expression reference {path!r}")

    def _render_number(self, frac: Fraction, typ: str) -> str:
        if typ == "money":
            # Numeric money literals in the IR are minor units (paise/cents),
            # matching the canonical `...Paisa` facts and `thresholdPaisa`.
            return _money_literal_minor(frac)
        if typ == "integer":
            if frac.denominator != 1:
                raise CatalaLoweringError(f"Fractional literal {frac} in integer position")
            return str(frac.numerator)
        if typ == "decimal":
            return _decimal_literal(frac)
        raise CatalaLoweringError(f"Numeric literal in {typ} position")

    def _as_decimal(self, node: Any) -> str:
        typ = self._infer(node)
        if typ == "decimal":
            return self._render(node, "decimal")
        if typ in ("integer", "money"):
            return f"(decimal of {self._render(node, typ)})"
        raise CatalaLoweringError(f"Non-numeric operand of type {typ} in decimal arithmetic")

    def _comparison_operand_type(self, left: Any, right: Any) -> str:
        lt, rt = self._infer(left), self._infer(right)
        if lt == rt:
            return lt
        # A bare text literal compared against an enum-typed operand adopts
        # that enumeration's type. _render_reference validates the case name.
        if lt.startswith("enum:") and self._is_text_literal(right):
            return lt
        if rt.startswith("enum:") and self._is_text_literal(left):
            return rt
        if {lt, rt} <= {"money", "integer", "decimal"}:
            return self._unify_numeric(lt, rt, "comparison")
        raise CatalaLoweringError(f"Comparison of incompatible types {lt} and {rt}")

    @staticmethod
    def _is_text_literal(node: Any) -> bool:
        if not isinstance(node, str):
            return False
        stripped = node.strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            return False
        return not stripped.startswith(("facts.", "vars.", "rules."))

    def _static_number(self, node: Any) -> Fraction | None:
        """Extracts a compile-time numeric constant, mirroring USVM toPaise()."""
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return Fraction(str(node))
        if isinstance(node, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", node.strip()):
            return Fraction(node.strip())
        if isinstance(node, dict):
            op = node.get("op") or node.get("operator")
            if op == "LITERAL":
                return self._static_number(node["value"])
            if op in ("TO_PAISE", "FROM_PAISE"):
                inner = self._static_number(node.get("value", node.get("operand")))
                return inner
        return None


def emit_catala_code(
    ast_or_result: dict[str, Any],
    title: str | None = None,
    scenarios: list[dict[str, Any]] | None = None,
) -> str:
    """
    Public entry point: lower a canonical USVM AST (or full decompiler result)
    into Catala (.catala_en) source. Raises CatalaLoweringError for constructs
    without a faithful translation.
    """
    emitter = CatalaEmitter()
    return emitter.emit(ast_or_result, title=title, scenarios=scenarios)


def save_catala_file(
    ast_or_result: dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
    scenarios: list[dict[str, Any]] | None = None,
) -> Path:
    """
    Exports decompiled Catala source to a file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    code = emit_catala_code(ast_or_result, title=title, scenarios=scenarios)
    out_p.write_text(code, encoding="utf-8")
    return out_p
