"""Automated Intended-vs-Actual Discrepancy Verifier and Logger for Model v2.

Detects semantic, algebraic, type-mask, and USVM execution discrepancies
between intended statutory/combinatorial specs and actual lowered/predicted
ASTs without relying on token accuracy alone. Writes machine-readable JSONL
records to a separate discrepancy log alongside checkpoints.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from compiler.neural.inference_v2 import (
    apply_type_compatibility_mask,
    build_type_compatibility_mask_tensor,
    synthesize_usvm_ast_v2,
)
from compiler.neural.literals import segment_clauses
from compiler.neural.logic_reasoner import tokens_to_logic_ids
from compiler.neural.ssa_walker import V2LoweringError
from compiler.neural.statutory_neural_parser import (
    ID2MODALITY,
    ID2UNIT,
    SUFFIX2ID,
    V2_BIO_TAGS,
    V2_ID2BIO,
    V2_ID2REL,
    V2_ID2ROLE,
    canonical_suffix,
    canonicalize_token,
    decode_bio_spans,
)
from compiler.training.usvm_rl_environment import (
    StatutoryRlEnvironment,
    _extract_expected_outcome,
)
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator


def append_discrepancy_record(log_path: str | Path | None, record: dict[str, Any]) -> None:
    """Appends one structured discrepancy JSON record to `log_path` if configured."""
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(enriched, default=str) + "\n")


def _select_stratified_indices(items: list[dict], max_samples: int) -> list[int]:
    """Selects deterministic stratified sample indices covering every `stype` family."""
    if max_samples <= 0 or len(items) <= max_samples:
        return list(range(len(items)))
    buckets: dict[str, list[int]] = {}
    for idx, it in enumerate(items):
        if it.get("modality_name") == "MODALITY_ABSTAIN_UNRESOLVED":
            continue
        buckets.setdefault(str(it.get("stype", "UNKNOWN")), []).append(idx)
    picked: list[int] = []
    pos = 0
    keys = sorted(buckets.keys())
    while len(picked) < max_samples and keys:
        next_keys = []
        for k in keys:
            if pos < len(buckets[k]):
                picked.append(buckets[k][pos])
                next_keys.append(k)
                if len(picked) >= max_samples:
                    break
        pos += 1
        keys = next_keys
    return picked


def _build_spans_edges_from_indices(
    text: str,
    offsets: list[tuple[int, int]],
    span_boundaries: list[tuple[int, int]],
    role_ids: list[int],
    unit_ids: list[int],
    rel_matrix: list[list[int]] | torch.Tensor,
    clause_spans: list[tuple[int, int]],
    modality_name: str,
    id2role: dict[int, str],
    id2unit: dict[int, str],
    id2rel: dict[int, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Materializes `synthesize_usvm_ast_v2` span/edge/clause dicts from token indices."""
    spans: list[dict] = []
    for s_i, (ts, te) in enumerate(span_boundaries):
        if ts < 0 or te >= len(offsets) or ts > te:
            continue
        cs, ce = offs_s, offs_e = offsets[ts][0], offsets[te][1]
        c_idx = 0
        for ci, (ccs, cce) in enumerate(clause_spans):
            if ts >= ccs and te <= cce:
                c_idx = ci
                break
        spans.append({
            "spanIdx": len(spans),
            "span": text[offs_s:offs_e],
            "role": id2role.get(int(role_ids[s_i]), "ROLE_IGNORE"),
            "unit": id2unit.get(int(unit_ids[s_i]), "UNIT_NONE"),
            "charRange": [cs, ce],
            "clauseIdx": c_idx,
        })

    edges: list[dict] = []
    n_s = len(spans)
    for s_i in range(n_s):
        for t_i in range(n_s):
            if s_i == t_i:
                continue
            r_id = int(rel_matrix[s_i][t_i].item() if isinstance(rel_matrix, torch.Tensor) else rel_matrix[s_i][t_i])
            r_name = id2rel.get(r_id, "REL_NO_EDGE")
            if r_name != "REL_NO_EDGE":
                edges.append({"sourceIdx": s_i, "targetIdx": t_i, "relation": r_name})

    cmods = [
        {
            "clauseIdx": ci,
            "modality": modality_name,
            "charRange": [offsets[ccs][0], offsets[cce][1]],
        }
        for ci, (ccs, cce) in enumerate(clause_spans)
        if 0 <= ccs <= cce < len(offsets)
    ]
    return spans, edges, cmods


def check_ast_semantic_invariants(ast: dict[str, Any], dag_meta: dict[str, Any] | None = None) -> list[str]:
    """Checks structural & intended-DAG invariants on a lowered v2 AST."""
    violations: list[str] = []
    if ast.get("astVersion") != "v2":
        violations.append("AST_VERSION_NOT_V2")
    if ast.get("executionDiagnostics", {}).get("regexHeuristicsUsed", True):
        violations.append("REGEX_HEURISTICS_FLAGGED")

    for out_name, out_spec in (ast.get("outputs") or {}).items():
        if isinstance(out_spec, dict) and out_spec.get("type") == "BOOLEAN" and str(out_name).endswith("Paisa"):
            violations.append(f"BOOLEAN_PAISA_SUFFIX:{out_name}")

    if isinstance(dag_meta, dict):
        kind = dag_meta.get("kind")
        ops_str = json.dumps(ast.get("computation", {}).get("pipeline", {}).get("steps", []))
        if kind == "BRANCHED_RATES":
            mode = dag_meta.get("mode")
            if mode == "conditional" and '"CONDITIONAL"' not in ops_str:
                violations.append("BRANCHED_RATES_CONDITIONAL_MISSING_CONDITIONAL_OP")
            if mode == "additive" and '"ADD"' not in ops_str:
                violations.append("BRANCHED_RATES_ADDITIVE_MISSING_ADD_OP")
        elif kind == "FOLD_TO_ARITHMETIC":
            if '"SUM_OVER"' not in ops_str and '"COUNT"' not in ops_str:
                violations.append("FOLD_TO_ARITHMETIC_MISSING_FOLD_OP")
        elif kind == "GATED_ARITHMETIC":
            if '"CONDITIONAL"' not in ops_str:
                violations.append("GATED_ARITHMETIC_MISSING_CONDITIONAL_OP")
        elif kind == "MULTI_TIER_BRACKET":
            if '"ADD"' not in ops_str and '"MIN"' not in ops_str:
                violations.append("MULTI_TIER_BRACKET_MISSING_ADD_OR_MIN_OP")
    return violations


def _values_match(exp_val: Any, act_val: Any) -> bool:
    if isinstance(exp_val, bool) or isinstance(act_val, bool):
        return bool(exp_val) is bool(act_val)
    if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
        return abs(float(exp_val) - float(act_val)) < 1e-2
    return exp_val == act_val


def compare_ast_counterfactual_outputs(
    gold_ast: dict[str, Any],
    pred_ast: dict[str, Any],
    rl_env: StatutoryRlEnvironment,
    evaluator: CanonicalUsvmEvaluator,
    stype: str | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluates `pred_ast` against counterfactual vectors synthesized from `gold_ast`."""
    vectors = rl_env.build_sample_vectors(gold_ast, stype=stype)
    diffs: list[dict[str, Any]] = []
    for vec in vectors:
        facts = vec["facts"]
        expected = vec.get("expected") or {}
        try:
            pred_res = evaluator.evaluate(pred_ast, facts)
            pred_out = _extract_expected_outcome(pred_res, pred_ast)
        except Exception as exc:
            diffs.append({"vectorId": vec.get("id"), "error": str(exc)})
            continue
        for k, exp_v in expected.items():
            if k not in pred_out:
                # Match scalar output if only key naming shifted while single output value matches
                scalar_vals = [v for v in pred_out.values() if isinstance(v, (int, float, bool))]
                if len(scalar_vals) == 1 and _values_match(exp_v, scalar_vals[0]):
                    continue
                diffs.append({"vectorId": vec.get("id"), "key": k, "expected": exp_v, "actual": None})
            elif not _values_match(exp_v, pred_out[k]):
                diffs.append({"vectorId": vec.get("id"), "key": k, "expected": exp_v, "actual": pred_out[k]})
    return (len(diffs) == 0), diffs


def verify_v2_dataset_contract(
    payload: dict[str, Any],
    evaluator: CanonicalUsvmEvaluator | None = None,
    rl_env: StatutoryRlEnvironment | None = None,
    max_samples: int = 120,
    strict: bool = True,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audits processed dataset splits for mask, lowering, DAG invariant, and USVM alignment."""
    from compiler.training.dataset_generator import tokenize_with_offsets

    own_env = rl_env is None
    env = rl_env or StatutoryRlEnvironment()
    ev = evaluator or env.evaluator

    roles_list = payload.get("roles") or [V2_ID2ROLE[i] for i in sorted(V2_ID2ROLE)]
    rels_list = payload.get("relations") or [V2_ID2REL[i] for i in sorted(V2_ID2REL)]
    units_list = payload.get("span_units") or [ID2UNIT[i] for i in sorted(ID2UNIT)]
    id2role = {i: r for i, r in enumerate(roles_list)}
    id2rel = {i: r for i, r in enumerate(rels_list)}
    id2unit = {i: u for i, u in enumerate(units_list)}

    discrepancies: list[dict[str, Any]] = []
    checked = 0
    try:
        for split in ("train", "val"):
            items = payload.get(split) or []
            per_split = max(1, max_samples // 2)
            for idx in _select_stratified_indices(items, per_split):
                item = items[idx]
                if item.get("modality_name") == "MODALITY_ABSTAIN_UNRESOLVED":
                    continue
                checked += 1
                u_ids = item.get("unit_labels") or []
                rm = item.get("rel_matrix") or []
                mask = build_type_compatibility_mask_tensor(u_ids, id2unit=id2unit, id2rel=id2rel)

                mask_blocks = []
                for s_i in range(len(rm)):
                    for t_i in range(len(rm[s_i])):
                        r_id = int(rm[s_i][t_i])
                        if r_id != 0 and float(mask[s_i, t_i, r_id].item()) < -1e4:
                            mask_blocks.append(f"({id2unit.get(u_ids[s_i])}, {id2rel.get(r_id)}, {id2unit.get(u_ids[t_i])})")
                if mask_blocks:
                    rec = {
                        "recordType": "DATASET_CONTRACT_DISCREPANCY",
                        "split": split,
                        "sampleIdx": idx,
                        "stype": item.get("stype"),
                        "kind": "MASK_BLOCKED_GOLD_EDGE",
                        "detail": mask_blocks,
                        "text": item.get("text"),
                    }
                    discrepancies.append(rec)
                    append_discrepancy_record(log_path, rec)
                    continue

                _, offs = tokenize_with_offsets(item["text"])
                spans, edges, cmods = _build_spans_edges_from_indices(
                    text=item["text"],
                    offsets=offs,
                    span_boundaries=item["span_boundaries"],
                    role_ids=item["role_labels"],
                    unit_ids=u_ids,
                    rel_matrix=rm,
                    clause_spans=item.get("clause_spans") or [(0, len(offs) - 1)],
                    modality_name=item.get("modality_name") or "MODALITY_COMPUTATIONAL_EXCESS",
                    id2role=id2role,
                    id2unit=id2unit,
                    id2rel=id2rel,
                )
                try:
                    ast = synthesize_usvm_ast_v2(
                        text=item["text"],
                        spans=spans,
                        edges=edges,
                        clause_modalities=cmods,
                        dominant_modality=item.get("modality_name"),
                    )
                    inv_errs = check_ast_semantic_invariants(ast, dag_meta=item.get("dag"))
                    if inv_errs:
                        rec = {
                            "recordType": "DATASET_CONTRACT_DISCREPANCY",
                            "split": split,
                            "sampleIdx": idx,
                            "stype": item.get("stype"),
                            "kind": "DAG_CONTRACT_MISMATCH",
                            "detail": inv_errs,
                            "text": item.get("text"),
                        }
                        discrepancies.append(rec)
                        append_discrepancy_record(log_path, rec)
                        continue
                    vectors = env.build_sample_vectors(ast, stype=item.get("stype"))
                    if not vectors:
                        raise RuntimeError("empty counterfactual vectors")
                except Exception as exc:
                    rec = {
                        "recordType": "DATASET_CONTRACT_DISCREPANCY",
                        "split": split,
                        "sampleIdx": idx,
                        "stype": item.get("stype"),
                        "kind": "LOWER_OR_EVAL_ERROR",
                        "detail": str(exc),
                        "text": item.get("text"),
                    }
                    discrepancies.append(rec)
                    append_discrepancy_record(log_path, rec)
    finally:
        if own_env:
            env.evaluator.close()

    if strict and discrepancies:
        first = discrepancies[0]
        raise RuntimeError(
            f"ERR_DATASET_CONTRACT_DISCREPANCY ({len(discrepancies)}/{checked}): "
            f"[{first['kind']}] stype={first['stype']} detail={first['detail']}"
        )
    passed_cnt = checked - len(discrepancies)
    return {
        "checked": checked,
        "passed": passed_cnt,
        "errorCount": len(discrepancies),
        "discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
    }


@torch.no_grad()
def audit_v2_epoch_predictions(
    model: torch.nn.Module,
    val_items: list[dict],
    raw_vocab: dict[str, int],
    id2role: dict[int, str],
    id2rel: dict[int, str],
    id2bio: dict[int, str],
    id2unit: dict[int, str],
    evaluator: CanonicalUsvmEvaluator | None = None,
    rl_env: StatutoryRlEnvironment | None = None,
    log_path: str | Path | None = None,
    epoch: int = 1,
    device: torch.device | None = None,
    max_samples: int = 24,
) -> dict[str, Any]:
    """Audits live model predictions on stratified `val_items` against intended gold USVM outputs."""
    from compiler.training.dataset_generator import tokenize_with_offsets

    own_env = rl_env is None
    env = rl_env or StatutoryRlEnvironment()
    ev = evaluator or env.evaluator
    device = device or next(model.parameters()).device

    was_training = model.training
    model.eval()
    indices = _select_stratified_indices(val_items, max_samples)
    checked = 0
    passed = 0
    breakdown: dict[str, int] = {}

    def _bump(kind: str, stype: str, text: str, detail: Any) -> None:
        breakdown[kind] = breakdown.get(kind, 0) + 1
        append_discrepancy_record(
            log_path,
            {
                "recordType": "EPOCH_PRED_DISCREPANCY",
                "epoch": epoch,
                "stype": stype,
                "kind": kind,
                "detail": detail,
                "text": text,
            },
        )

    try:
        for idx in indices:
            item = val_items[idx]
            text = item["text"]
            stype = str(item.get("stype", "UNKNOWN"))
            checked += 1

            tokens, offsets = tokenize_with_offsets(text)
            if not tokens:
                _bump("EMPTY_TOKENS", stype, text, None)
                continue

            raw_ids = torch.tensor(
                [[
                    raw_vocab.get(
                        canonicalize_token(tokens[i], tokens[i - 1] if i > 0 else None, tokens[i + 1] if i + 1 < len(tokens) else None),
                        raw_vocab.get(tokens[i].lower(), raw_vocab.get("<UNK>", 1)),
                    )
                    for i in range(len(tokens))
                ]],
                dtype=torch.long,
                device=device,
            )
            log_ids = torch.tensor([tokens_to_logic_ids(tokens)], dtype=torch.long, device=device)
            suf_ids = torch.tensor(
                [[
                    SUFFIX2ID.get(
                        canonical_suffix(
                            tokens[i],
                            canonicalize_token(tokens[i], tokens[i - 1] if i > 0 else None, tokens[i + 1] if i + 1 < len(tokens) else None),
                        ),
                        SUFFIX2ID["<NONE>"],
                    )
                    for i in range(len(tokens))
                ]],
                dtype=torch.long,
                device=device,
            )
            pad_mask = torch.zeros_like(raw_ids, dtype=torch.bool)
            c_ranges = segment_clauses(text, tokens, offsets) or [(0, len(tokens) - 1)]

            # Step A: BIO span proposal pass
            bio_out = model(
                raw_token_ids=raw_ids,
                logic_token_ids=log_ids,
                suffix_ids=suf_ids,
                mask=pad_mask,
                clause_spans=c_ranges,
            )
            mod_vec = bio_out["modality_logits"].reshape(-1, bio_out["modality_logits"].size(-1))[0]
            pred_mod_id = int(mod_vec.argmax(dim=-1).item())
            pred_mod_name = ID2MODALITY.get(pred_mod_id, "MODALITY_COMPUTATIONAL_EXCESS")
            if pred_mod_name == "MODALITY_ABSTAIN_UNRESOLVED":
                _bump("FALSE_ABSTAIN", stype, text, {"goldModality": item.get("modality_name")})
                continue

            bio_preds = bio_out["bio_logits"].reshape(-1, bio_out["bio_logits"].size(-1)).argmax(dim=-1).tolist()
            bio_tags = [id2bio.get(p, "O") for p in bio_preds]
            disc_spans = decode_bio_spans(bio_tags, tokens, offsets)
            if not disc_spans:
                _bump("BIO_ZERO_SPANS", stype, text, {"goldSpans": len(item["span_boundaries"])})
                continue

            pred_span_bounds = [tuple(ds["span_tokens"]) for ds in disc_spans]
            full_out = model(
                raw_token_ids=raw_ids,
                logic_token_ids=log_ids,
                suffix_ids=suf_ids,
                mask=pad_mask,
                span_indices=pred_span_bounds,
                clause_spans=c_ranges,
            )
            pred_r_ids = full_out["role_logits"].argmax(dim=-1).tolist()
            pred_u_ids = full_out["unit_logits"].argmax(dim=-1).tolist() if full_out.get("unit_logits") is not None else [0] * len(pred_span_bounds)
            raw_rel = full_out["rel_logits"]
            masked_rel = apply_type_compatibility_mask(raw_rel, pred_u_ids, id2unit=id2unit, id2rel=id2rel)

            # Detect whether raw unmasked rel argmax disagreed with type compatibility mask
            if raw_rel is not None and masked_rel is not None:
                raw_argmax = raw_rel.argmax(dim=-1)
                mask_argmax = masked_rel.argmax(dim=-1)
                if not torch.equal(raw_argmax, mask_argmax):
                    breakdown["MASK_CONFLICT"] = breakdown.get("MASK_CONFLICT", 0) + 1

            pred_rm = masked_rel.argmax(dim=-1) if masked_rel is not None else torch.zeros((len(pred_span_bounds), len(pred_span_bounds)), dtype=torch.long)

            pred_spans, pred_edges, pred_cmods = _build_spans_edges_from_indices(
                text=text,
                offsets=offsets,
                span_boundaries=pred_span_bounds,
                role_ids=pred_r_ids,
                unit_ids=pred_u_ids,
                rel_matrix=pred_rm,
                clause_spans=c_ranges,
                modality_name=pred_mod_name,
                id2role=id2role,
                id2unit=id2unit,
                id2rel=id2rel,
            )

            try:
                pred_ast = synthesize_usvm_ast_v2(
                    text=text,
                    spans=pred_spans,
                    edges=pred_edges,
                    clause_modalities=pred_cmods,
                    dominant_modality=pred_mod_name,
                )
            except V2LoweringError as exc:
                _bump(
                    "LOWERING_ERROR",
                    stype,
                    text,
                    {
                        "error": str(exc),
                        "predUnits": [s["unit"] for s in pred_spans],
                        "predRoles": [s["role"] for s in pred_spans],
                        "goldUnits": [id2unit.get(int(u)) for u in (item.get("unit_labels") or [])],
                    },
                )
                continue

            # Build intended gold AST for semantic numeric equivalence check
            g_spans, g_edges, g_cmods = _build_spans_edges_from_indices(
                text=text,
                offsets=offsets,
                span_boundaries=item["span_boundaries"],
                role_ids=item["role_labels"],
                unit_ids=item["unit_labels"],
                rel_matrix=item["rel_matrix"],
                clause_spans=item.get("clause_spans") or c_ranges,
                modality_name=item.get("modality_name") or "MODALITY_COMPUTATIONAL_EXCESS",
                id2role=id2role,
                id2unit=id2unit,
                id2rel=id2rel,
            )
            gold_ast = synthesize_usvm_ast_v2(
                text=text,
                spans=g_spans,
                edges=g_edges,
                clause_modalities=g_cmods,
                dominant_modality=item.get("modality_name"),
            )
            eq_ok, diffs = compare_ast_counterfactual_outputs(gold_ast, pred_ast, env, ev, stype=stype)
            if eq_ok:
                passed += 1
            else:
                _bump(
                    "USVM_VALUE_MISMATCH",
                    stype,
                    text,
                    {
                        "diffs": diffs[:3],
                        "predUnits": [s["unit"] for s in pred_spans],
                        "goldUnits": [s["unit"] for s in g_spans],
                        "predEdges": [(e["sourceIdx"], e["targetIdx"], e["relation"]) for e in pred_edges],
                        "goldEdges": [(e["sourceIdx"], e["targetIdx"], e["relation"]) for e in g_edges],
                    },
                )
    finally:
        if was_training:
            model.train()
        if own_env:
            env.evaluator.close()

    disc_cnt = checked - passed
    sem_pass_pct = round((passed / max(1, checked)) * 100.0, 2)
    append_discrepancy_record(
        log_path,
        {
            "recordType": "EPOCH_SEMANTIC_AUDIT",
            "epoch": epoch,
            "checked": checked,
            "passed": passed,
            "semPassPct": sem_pass_pct,
            "discrepancyCount": disc_cnt,
            "breakdown": breakdown,
        },
    )
    return {
        "checked": checked,
        "passed": passed,
        "sem_pass_pct": sem_pass_pct,
        "discrepancy_count": disc_cnt,
        "breakdown": breakdown,
    }
