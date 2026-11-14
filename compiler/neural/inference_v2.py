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


def _find_directed_cycle(n: int, edges: dict[tuple[int, int], tuple[int, int, int, float]]) -> list[tuple[int, int]] | None:
    """Return a list of directed graph keys (u, v) forming one cycle, or None if acyclic."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for (u, v) in sorted(edges):
        adj[u].append(v)

    state = [0] * n  # 0=unvisited, 1=visiting, 2=visited
    parent: dict[int, int] = {}

    def dfs(u: int) -> list[tuple[int, int]] | None:
        state[u] = 1
        for v in adj[u]:
            if state[v] == 1:
                cycle = [(u, v)]
                cur = u
                while cur != v and cur in parent:
                    prev = parent[cur]
                    cycle.append((prev, cur))
                    cur = prev
                return cycle
            if state[v] == 0:
                parent[v] = u
                res = dfs(v)
                if res is not None:
                    return res
        state[u] = 2
        return None

    for node in range(n):
        if state[node] == 0:
            res = dfs(node)
            if res is not None:
                return res
    return None


def decode_dag_relations_v2(
    rel_logits: torch.Tensor,
    id2rel: dict[int, str] | None = None,
    clause_ids: list[int] | None = None,
) -> list[list[int]]:
    """Decode an (N, N, R) relation logit tensor into an anti-symmetric DAG prediction matrix.

    Enforces:
    1. No self-loops (i == i -> 0).
    2. Anti-symmetry across pairs: if both i->j and j->i predict non-zero relations,
       keep only the direction with larger positive margin (logit[r] - logit[0]).
    3. Acyclicity across V2_EDGE_DIRECTION evaluation dependencies, raw directed
       edges, and (when clause_ids is provided) cross-clause dependency links:
       iteratively drop the lowest-margin edge on any residual cycle.
    """
    if rel_logits is None or rel_logits.numel() == 0:
        return []
    n = int(rel_logits.size(0))
    if n == 0:
        return []

    from .ssa_walker import V2_EDGE_DIRECTION

    CLAUSE_DEP_RELS = {
        "REL_COREF_LINK",
        "REL_DEFINED_TERM_OF",
        "REL_SUBSTITUTE_WHEN",
        "REL_RATE_TO_THRESHOLD",
        "REL_SUBTRACTION_BASE",
    }
    check_clauses = clause_ids is not None and len(clause_ids) == n
    num_clauses = (max(clause_ids) + 1) if check_clauses and clause_ids else 0

    logits_cpu = rel_logits.detach().cpu()
    preds: list[list[int]] = logits_cpu.argmax(dim=-1).tolist()

    for i in range(n):
        preds[i][i] = 0

    def _margin(s: int, t: int, rid: int) -> float:
        return float(logits_cpu[s, t, rid].item() - logits_cpu[s, t, 0].item())

    # 1. Anti-symmetry: resolve any bidirectional 2-cycle (i->j and j->i)
    for i in range(n):
        for j in range(i + 1, n):
            ri = preds[i][j]
            rj = preds[j][i]
            if ri != 0 and rj != 0:
                mi = _margin(i, j, ri)
                mj = _margin(j, i, rj)
                if mi >= mj:
                    preds[j][i] = 0
                else:
                    preds[i][j] = 0

    # 2. Break any longer cycles on V2_EDGE_DIRECTION eval graph, raw graph, or clause graph
    while True:
        eval_edges: dict[tuple[int, int], tuple[int, int, int, float]] = {}
        raw_edges: dict[tuple[int, int], tuple[int, int, int, float]] = {}
        clause_edges: dict[tuple[int, int], tuple[int, int, int, float]] = {}
        for s in range(n):
            for t in range(n):
                rid = preds[s][t]
                if rid == 0 or s == t:
                    continue
                m = _margin(s, t, rid)
                raw_edges[(s, t)] = (s, t, rid, m)
                rel_name = id2rel.get(rid, "REL_NO_EDGE") if id2rel else str(rid)
                direction = V2_EDGE_DIRECTION.get(rel_name, "independent")
                if direction == "source_before_target":
                    eval_edges[(s, t)] = (s, t, rid, m)
                elif direction == "target_before_source":
                    eval_edges[(t, s)] = (s, t, rid, m)

                if check_clauses and rel_name in CLAUSE_DEP_RELS:
                    sc = clause_ids[s]
                    tc = clause_ids[t]
                    if sc != tc and (rel_name != "REL_COREF_LINK" or sc - tc == 1):
                        prev = clause_edges.get((sc, tc))
                        if prev is None or m < prev[3]:
                            clause_edges[(sc, tc)] = (s, t, rid, m)

        cyc = _find_directed_cycle(n, eval_edges)
        graph_used = eval_edges
        if cyc is None:
            cyc = _find_directed_cycle(n, raw_edges)
            graph_used = raw_edges
        if cyc is None and check_clauses and clause_edges:
            cyc = _find_directed_cycle(num_clauses, clause_edges)
            graph_used = clause_edges
        if cyc is None:
            break

        weakest = min(
            (graph_used[k] for k in cyc),
            key=lambda item: (item[3], -item[0], -item[1]),
        )
        preds[weakest[0]][weakest[1]] = 0

    return preds


def remap_and_decode_dag_relations_v2(
    rel_logits: torch.Tensor,
    old_to_new_span_idx: dict[int, int],
    num_mapped_spans: int,
    id2rel: dict[int, str] | None = None,
    clause_ids: list[int] | None = None,
) -> list[list[int]]:
    """Project (N, N, R) relation logits through old_to_new_span_idx, then decode DAG.

    Running DAG anti-symmetry and cycle pruning AFTER span remapping guarantees that
    merging/stitching adjacent spans can never re-introduce self-loops or 2-cycles.
    """
    if rel_logits is None or rel_logits.numel() == 0 or num_mapped_spans <= 0:
        return []
    n = int(rel_logits.size(0))
    if n == 0:
        return []

    is_identity = (
        num_mapped_spans == n
        and all(old_to_new_span_idx.get(i, i) == i for i in range(n))
    )
    if is_identity:
        return decode_dag_relations_v2(
            rel_logits,
            id2rel=id2rel,
            clause_ids=clause_ids,
        )

    logits_cpu = rel_logits.detach().cpu()
    num_rels = int(logits_cpu.size(-1))
    mapped_logits = torch.zeros((num_mapped_spans, num_mapped_spans, num_rels), dtype=logits_cpu.dtype)
    mapped_logits[:, :, 0] = 1.0
    best_margin: dict[tuple[int, int], float] = {}

    for src in range(n):
        sm = old_to_new_span_idx.get(src, src)
        if sm < 0 or sm >= num_mapped_spans:
            continue
        for tgt in range(n):
            tm = old_to_new_span_idx.get(tgt, tgt)
            if tm < 0 or tm >= num_mapped_spans or sm == tm:
                continue
            cell = logits_cpu[src, tgt]
            margin = float(cell[1:].max().item() - cell[0].item()) if num_rels > 1 else 0.0
            key = (sm, tm)
            if key not in best_margin or margin > best_margin[key]:
                best_margin[key] = margin
                mapped_logits[sm, tm] = cell

    return decode_dag_relations_v2(
        mapped_logits,
        id2rel=id2rel,
        clause_ids=clause_ids,
    )


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
