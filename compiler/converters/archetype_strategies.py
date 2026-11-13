"""
Archetype normalization strategies for non-canonical AST layouts.

Converts archetype-specific AST layouts (e.g. French tax 2026 fixtures)
into the canonical DECLARATIVE_PIPELINE layout understood by CatalaEmitter.

Each strategy is a pure function: raw_ast -> canonical_ast_dict.
The emitter remains the single Catala rendering path (no duplicate execution
paths per AGENTS.md Rule 3). Dispatch uses a table-driven registry (Rule 20),
never a monolithic if-else chain.
"""

from typing import Any

from compiler.converters.errors import CatalaLoweringError


# Operator alias map: non-canonical operator -> canonical operator.
# The emitter's _infer/_render only know canonical operators; aliases are
# resolved here so the emitter never sees non-canonical op names.
_OPERATOR_ALIASES: dict[str, str] = {
    "TO_CENTS": "TO_PAISE",
    "FROM_CENTS": "FROM_PAISE",
    "VAR": "VARIABLE",
    "MULTIPLY": "MUL",
    "LE": "LTE",
}

# Expression child keys that may contain nested sub-expressions.
_EXPR_CHILD_KEYS = (
    "left", "right", "value", "operand", "then", "else",
    "consequent", "alternate", "test", "base",
)


def _map_op(expr: Any) -> Any:
    """Recursively map non-canonical operators to canonical operators.

    Also renames VAR's ``value`` key to ``var`` so the emitter's existing
    VARIABLE handling (which reads ``node.get("var")``) works unchanged.
    """
    if not isinstance(expr, dict):
        return expr

    op = expr.get("op") or expr.get("operator")
    result = dict(expr)

    if op in _OPERATOR_ALIASES:
        result["op"] = _OPERATOR_ALIASES[op]
        result.pop("operator", None)
        # VAR uses "value"; the emitter reads "var" or "path" for VARIABLE.
        if op == "VAR" and "value" in result:
            result["var"] = result.pop("value")

    for key in _EXPR_CHILD_KEYS:
        if key in result:
            result[key] = _map_op(result[key])

    if "operands" in result:
        result["operands"] = [_map_op(o) for o in result["operands"]]

    return result


def _normalize_steps(computation_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a list of computation steps to canonical pipeline step format."""
    steps: list[dict[str, Any]] = []
    for step in computation_steps:
        assign = step.get("assign") or step.get("name")
        if not assign:
            raise CatalaLoweringError(
                f"Computation step without assign target: {step!r}"
            )
        expr = _map_op(step.get("expr", {}))
        steps.append({"assign": assign, "expr": expr})
    return steps


def _normalize_output(output_map: dict[str, Any]) -> dict[str, str]:
    """Normalize output map; wrap non-reference values in LITERAL expr form."""
    output: dict[str, str] = {}
    for key, val in output_map.items():
        if isinstance(val, str) and val.startswith("vars."):
            output[key] = val
        elif isinstance(val, dict):
            output[key] = _map_op(val)
        else:
            output[key] = {"op": "LITERAL", "value": val}
    return output


def _expand_progressive_slab(
    steps: list[dict[str, Any]],
    progressive_index: int,
) -> None:
    """Expand a PROGRESSIVE_SLAB step into canonical pipeline steps.

    Progressive tax is the sum over slabs of

        MAX(0, MIN(base, upper) - lower) * rateBps / 10000

    Open-ended slabs (upper is None) use ``base`` instead of ``MIN(base, upper)``.
    Zero-rate slabs contribute nothing and are skipped. The expansion is built
    as a standalone list and spliced in once, so ordering never depends on
    shifting indices while inserting.
    """
    progressive_step = steps[progressive_index]
    expr = progressive_step.get("expr", {})
    base_var = expr.get("base")
    slab_list = expr.get("slabs", [])
    assign_target = progressive_step.get("assign", "progressive_result")

    if not base_var:
        raise CatalaLoweringError("PROGRESSIVE_SLAB missing 'base' operand")
    if not isinstance(slab_list, list):
        raise CatalaLoweringError(
            f"PROGRESSIVE_SLAB 'slabs' must be a list, got {type(slab_list).__name__}"
        )

    expanded: list[dict[str, Any]] = []
    contribution_vars: list[str] = []
    for idx, slab in enumerate(slab_list):
        lower = slab.get("lowerCents", 0)
        upper = slab.get("upperCents")
        rate = slab.get("rateBps", 0)

        if rate == 0:
            continue  # Zero-rate slabs contribute nothing.

        # portion = MAX(0, MIN(base, upper) - lower)
        min_expr: Any = base_var
        if upper is not None:
            min_expr = {"op": "MIN", "left": base_var, "right": upper}
        portion_expr = {
            "op": "MAX",
            "left": 0,
            "right": {"op": "SUB", "left": min_expr, "right": lower},
        }

        portion_var = f"slab_{idx}_portion_cents"
        tax_var = f"slab_{idx}_tax_cents"
        expanded.append({"assign": portion_var, "expr": portion_expr})
        expanded.append(
            {
                "assign": tax_var,
                "expr": {
                    "op": "RATIONAL_MULTIPLY",
                    "base": portion_var,
                    "rateBps": rate,
                    "denominator": 10000,
                },
            }
        )
        contribution_vars.append(tax_var)

    # Sum all slab contributions into the assign target. This step comes after
    # every slab step so all its dependencies are defined first.
    if not contribution_vars:
        total_expr: Any = {"op": "LITERAL", "value": 0}
    elif len(contribution_vars) == 1:
        total_expr = {"op": "VARIABLE", "value": contribution_vars[0]}
    else:
        total_expr = {
            "op": "ADD",
            "left": contribution_vars[0],
            "right": contribution_vars[1],
        }
        for extra in contribution_vars[2:]:
            total_expr = {"op": "ADD", "left": total_expr, "right": extra}
    expanded.append({"assign": assign_target, "expr": total_expr})

    steps[progressive_index : progressive_index + 1] = expanded


def normalize_vat_rate_lookup(ast: dict[str, Any]) -> dict[str, Any]:
    """Normalize VAT_RATE_LOOKUP_AND_COMPUTE to DECLARATIVE_PIPELINE.

    VAT has no single canonical threshold or flat rate: the applicable rate
    depends on customer type, country, goods category and the EU distance-
    selling threshold, all of which the pipeline steps encode explicitly.
    """
    steps = _normalize_steps(ast.get("computationSteps", []))
    output = _normalize_output(ast.get("output", {}))

    return {
        "sectionId": ast.get("sectionId", "StatutoryRule"),
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "pipeline": {
                "steps": steps,
                "output": output,
            },
        },
    }


def normalize_tiered_corporate_tax(ast: dict[str, Any]) -> dict[str, Any]:
    """Normalize TIERED_CORPORATE_TAX to DECLARATIVE_PIPELINE.

    The tier split (SME rate on the first slice, normal rate above it) is
    encoded in the pipeline steps, so no single canonical threshold/rate pair
    is emitted.
    """
    steps = _normalize_steps(ast.get("computationSteps", []))
    output = _normalize_output(ast.get("output", {}))

    return {
        "sectionId": ast.get("sectionId", "StatutoryRule"),
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "pipeline": {
                "steps": steps,
                "output": output,
            },
        },
    }


def normalize_progressive_slab_tax(ast: dict[str, Any]) -> dict[str, Any]:
    """Normalize PROGRESSIVE_SLAB_TAX to DECLARATIVE_PIPELINE.

    Expands the PROGRESSIVE_SLAB operator into canonical steps so the emitter
    never needs to know about it.
    """
    steps = _normalize_steps(ast.get("computationSteps", []))
    output = _normalize_output(ast.get("output", {}))

    # Find and expand the PROGRESSIVE_SLAB step.
    progressive_index: int | None = None
    for i, step in enumerate(steps):
        if step.get("expr", {}).get("op") == "PROGRESSIVE_SLAB":
            progressive_index = i
            break

    if progressive_index is not None:
        _expand_progressive_slab(steps, progressive_index)

    return {
        "sectionId": ast.get("sectionId", "StatutoryRule"),
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "pipeline": {
                "steps": steps,
                "output": output,
            },
        },
    }


def normalize_payroll_contributions(ast: dict[str, Any]) -> dict[str, Any]:
    """Normalize PAYROLL_CONTRIBUTIONS to DECLARATIVE_PIPELINE.

    Payroll has no single statutory threshold or flat rate: contributions are
    a weighted sum of separately rated levies (plus ceiling-conditional ones),
    so the optional canonical constants are deliberately omitted.
    """
    steps = _normalize_steps(ast.get("computationSteps", []))
    output = _normalize_output(ast.get("output", {}))

    return {
        "sectionId": ast.get("sectionId", "StatutoryRule"),
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "pipeline": {
                "steps": steps,
                "output": output,
            },
        },
    }


def normalize_property_tax(ast: dict[str, Any]) -> dict[str, Any]:
    """Normalize AD_VALOREM_PROPERTY_TAX to DECLARATIVE_PIPELINE.

    The municipal rate is an explicit fact, never a hardcoded range: locality
    rates vary and the fixture only documents indicative ranges.
    """
    steps = _normalize_steps(ast.get("computationSteps", []))
    output = _normalize_output(ast.get("output", {}))

    return {
        "sectionId": ast.get("sectionId", "StatutoryRule"),
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "pipeline": {
                "steps": steps,
                "output": output,
            },
        },
    }


# Table-driven registry: archetype name -> normalizer function.
# New archetypes are added here, never via if-else chains in the emitter.
ARCHETYPE_NORMALIZERS: dict[str, Any] = {
    "VAT_RATE_LOOKUP_AND_COMPUTE": normalize_vat_rate_lookup,
    "TIERED_CORPORATE_TAX": normalize_tiered_corporate_tax,
    "PROGRESSIVE_SLAB_TAX": normalize_progressive_slab_tax,
    "PAYROLL_CONTRIBUTIONS": normalize_payroll_contributions,
    "AD_VALOREM_PROPERTY_TAX": normalize_property_tax,
}


def normalize_ast(ast: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the correct archetype normalizer.

    If the AST already uses the canonical DECLARATIVE_PIPELINE layout
    (computation.archetype == "DECLARATIVE_PIPELINE"), it is returned
    unchanged. Otherwise the top-level ``archetype`` field selects a
    normalizer from ARCHETYPE_NORMALIZERS.
    """
    computation = ast.get("computation") or {}
    if computation.get("archetype") == "DECLARATIVE_PIPELINE":
        return ast

    archetype = ast.get("archetype")
    if not archetype:
        raise CatalaLoweringError(
            "AST has no computation.archetype or top-level archetype; "
            "cannot determine lowering strategy"
        )

    normalizer = ARCHETYPE_NORMALIZERS.get(archetype)
    if normalizer is None:
        raise CatalaLoweringError(
            f"Unsupported computation archetype {archetype!r}; "
            f"supported: {sorted(ARCHETYPE_NORMALIZERS)}"
        )

    return normalizer(ast)
