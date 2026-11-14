"""Clean regex-free AST lowerer and type-compatibility masker for Model v2.

Consumes the native neural predictions emitted by Dual-Stream v2 (`SPAN_UNITS`,
`V2_SEMANTIC_ROLES`, `V2_RELATION_TYPES`) without heuristic pattern-matching,
hardcoded statutory section branches, or alias-probing fallbacks.
"""

from __future__ import annotations

import hashlib
import torch

from .clause_linker import build_clause_scoped_computation_v2
from .literals import infer_rule_currency_info
from .ssa_walker import V2LoweringError
from .statutory_neural_parser import SPAN_UNITS, V2_RELATION_TYPES


_NON_MONEY_TEMPORAL_BOOL = {
    "UNIT_DURATION",
    "UNIT_YEAR_DATE",
    "UNIT_COUNT",
    "UNIT_BOOL_PREDICATE",
}
_MONEY_COMPAT = {"UNIT_MONEY", "UNIT_SYMBOL_VAR"}
_ARITH_COMPAT = {"UNIT_MONEY", "UNIT_RATE", "UNIT_SYMBOL_VAR"}


def _is_compatible_triple(src_u: str, rel: str, tgt_u: str) -> bool:
    """Enforces type safety between span units and typed opcode relations."""
    if rel == "REL_NO_EDGE":
        return True
    if rel == "REL_RATE_TO_THRESHOLD":
        return src_u == "UNIT_RATE" and tgt_u in _MONEY_COMPAT
    if rel == "REL_BASE_TAX_ADD":
        return src_u in _MONEY_COMPAT and tgt_u in _ARITH_COMPAT
    if rel in ("REL_MIN_CAP", "REL_SUB_CREDIT", "REL_CAP_CEILING"):
        return (
            (src_u in _MONEY_COMPAT and tgt_u in _ARITH_COMPAT)
            or (tgt_u in _MONEY_COMPAT and src_u in _ARITH_COMPAT)
        )
    if rel == "REL_BRACKET_NEXT":
        return src_u in _ARITH_COMPAT and tgt_u in _ARITH_COMPAT
    if rel == "REL_SUBTRACTION_BASE":
        return src_u in _ARITH_COMPAT and tgt_u in _MONEY_COMPAT
    if rel == "REL_DEADLINE_TO_TRIGGER":
        return src_u in {"UNIT_DURATION", "UNIT_YEAR_DATE"}
    if rel in ("REL_AGGREGATE_COUNT", "REL_QUANTIFIER_FORALL", "REL_QUANTIFIER_EXISTS"):
        return src_u not in {"UNIT_RATE", "UNIT_YEAR_DATE"} and tgt_u not in {"UNIT_RATE", "UNIT_YEAR_DATE"}
    if rel == "REL_AGGREGATE_SUM":
        if src_u == "UNIT_RATE" and tgt_u == "UNIT_RATE":
            return True
        return src_u not in {"UNIT_RATE", "UNIT_YEAR_DATE", "UNIT_BOOL_PREDICATE"} and tgt_u not in {"UNIT_RATE", "UNIT_YEAR_DATE", "UNIT_BOOL_PREDICATE"}
    if rel == "REL_LOGICAL_NOT":
        return tgt_u in {"UNIT_BOOL_PREDICATE", "UNIT_DURATION", "UNIT_YEAR_DATE", "UNIT_RATE", "UNIT_MONEY", "UNIT_SYMBOL_VAR"}
    if rel in ("REL_LOGICAL_AND", "REL_LOGICAL_OR"):
        _logic_atoms = {"UNIT_BOOL_PREDICATE", "UNIT_DURATION", "UNIT_YEAR_DATE", "UNIT_RATE", "UNIT_MONEY"}
        return src_u in _logic_atoms and tgt_u in _logic_atoms
    return True


def build_type_compatibility_mask_tensor(
    unit_preds: list[int],
    id2unit: dict[int, str] | None = None,
    id2rel: dict[int, str] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Builds an additive mask `[S, S, R]` with `-1e9` on type-invalid triples."""
    u_map = id2unit or {i: u for i, u in enumerate(SPAN_UNITS)}
    r_map = id2rel or {i: r for i, r in enumerate(V2_RELATION_TYPES)}
    n_spans = len(unit_preds)
    n_rels = len(r_map)
    mask = torch.zeros((n_spans, n_spans, n_rels), dtype=torch.float32, device=device)

    for s_idx, su_id in enumerate(unit_preds):
        su = u_map.get(int(su_id), "UNIT_NONE")
        for t_idx, tu_id in enumerate(unit_preds):
            tu = u_map.get(int(tu_id), "UNIT_NONE")
            for r_idx, rel in r_map.items():
                if not _is_compatible_triple(su, rel, tu):
                    mask[s_idx, t_idx, r_idx] = -1e9
    return mask


def apply_type_compatibility_mask(
    rel_logits: torch.Tensor,
    unit_preds: list[int],
    id2unit: dict[int, str] | None = None,
    id2rel: dict[int, str] | None = None,
) -> torch.Tensor:
    """Masks impossible `(src_unit, rel, tgt_unit)` entries before Biaffine decoding."""
    if rel_logits is None or rel_logits.numel() == 0:
        return rel_logits
    u_map = id2unit or {i: u for i, u in enumerate(SPAN_UNITS)}
    r_map = id2rel or {i: r for i, r in enumerate(V2_RELATION_TYPES)}
    if len(r_map) != rel_logits.size(-1):
        r_map = {i: V2_RELATION_TYPES[i] for i in range(rel_logits.size(-1))}
    mask = build_type_compatibility_mask_tensor(
        unit_preds=unit_preds,
        id2unit=u_map,
        id2rel=r_map,
        device=rel_logits.device,
    )
    return rel_logits + mask


def synthesize_usvm_ast_v2(
    text: str,
    spans: list[dict],
    edges: list[dict],
    clause_modalities: list[dict] | None = None,
    dominant_modality: str | None = None,
    domain_abstain_prob: float = 0.0,
    rule_id: str | None = None,
) -> dict:
    """Regex-free v2 AST entry point routing neural predictions through two-pass linking + topological SSA."""
    if dominant_modality == "MODALITY_ABSTAIN_UNRESOLVED" and domain_abstain_prob >= 0.7:
        raise V2LoweringError("ERR_DOMAIN_ABSTAIN: statutory signal below confidence threshold")

    ccy, _ = infer_rule_currency_info(spans, text)
    scoped = build_clause_scoped_computation_v2(
        text=text,
        spans=spans,
        edges=edges,
        clause_modalities=clause_modalities,
        currency_code=ccy,
    )

    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    rid = rule_id or f"usvm_v2_{digest}"

    return {
        "ruleId": rid,
        "version": "2.0",
        "schemaVersion": "2.0",
        "astVersion": "v2",
        "archetype": "DECLARATIVE_PIPELINE",
        "ruleDomain": "STATUTORY_FINANCE",
        "modality": dominant_modality or "MODALITY_COMPUTATIONAL_EXCESS",
        "currency": scoped.get("currency", ccy or "USD"),
        "inputs": scoped["inputs"],
        "computation": scoped["computation"],
        "sourceLineage": {
            "rawText": text,
            "spanCount": len(spans),
            "edgeCount": len(edges),
        },
        "statutoryInvariants": {
            "isAuthorizedToDisburse": False,
        },
        "symbolLegend": {},
        "executionDiagnostics": {
            "lowerer": "v2_typed_opcode_topological",
            "regexHeuristicsUsed": False,
            "clauseCount": len(scoped["computation"].get("clauseScopes", [])),
            "stepCount": len(scoped["computation"]["pipeline"]["steps"]),
        },
    }
