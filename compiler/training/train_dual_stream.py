"""
=============================================================================
TWO-STAGE DUAL-STREAM TRAINING:
  1. Stage 1 (Supervised Disentangled Co-Training):
     - Model B (DomainSensor) learns 9-class token-type & citation/preamble noise
       + rule-vs-chatter detection (zero gradient leakage from Logic heads).
     - Model A (LogicReasoner) learns BIO spans, SRL roles, biaffine relations,
       and rule modality strictly on closed-class LOGIC_VOCAB + detached 9-dim
       Domain token-type probabilities.
  2. Stage 2 (Execution-Guided GRPO RL on LogicReasoner ONLY):
     - Model B (DomainSensor) is 100% frozen so citation/NER sensing never drifts.
     - Model A updates via canonical JS USVM rewards + BIO KL anchor.
=============================================================================
"""

import os
import sys
import copy
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path

from compiler.training.dataset_generator import tokenize_with_offsets, generate_synthetic_dataset
from compiler.training.usvm_rl_environment import StatutoryRlEnvironment
from compiler.neural.statutory_neural_parser import (
    SEMANTIC_ROLES,
    RELATION_TYPES,
    BIO_TAGS,
    RULE_MODALITIES,
    SUFFIX2ID,
    ID2ROLE,
    ID2REL,
    ID2MODALITY,
    SPAN_UNITS,
    V2_SEMANTIC_ROLES,
    V2_RELATION_TYPES,
    V2_BIO_TAGS,
    V2_ID2ROLE,
    canonicalize_token,
    canonical_suffix,
)
from compiler.neural.inference import synthesize_usvm_ast, segment_clauses
from compiler.neural.inference_v2 import apply_type_compatibility_mask, synthesize_usvm_ast_v2
from compiler.neural.ssa_walker import V2LoweringError
from compiler.neural.domain_sensor import (
    DOMAIN_TOKEN_CLASSES,
    DOMAIN_CLASS2ID,
    derive_domain_token_targets,
)
from compiler.neural.logic_reasoner import (
    LOGIC_VOCAB,
    DualStreamStatutoryParser,
    tokens_to_logic_ids,
)

DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")
DEFAULT_SUP_CKPT = DEFAULT_CHECKPOINTS_DIR / "dual_stream_sup.pt"
DEFAULT_RL_CKPT = DEFAULT_CHECKPOINTS_DIR / "dual_stream_rl.pt"
DEFAULT_DATASET = DEFAULT_CHECKPOINTS_DIR / "dataset.pt"
DEFAULT_V2_SUP_CKPT = DEFAULT_CHECKPOINTS_DIR / "v2_dual_stream_sup.pt"
DEFAULT_V2_RL_CKPT = DEFAULT_CHECKPOINTS_DIR / "v2_dual_stream_rl.pt"
DEFAULT_V2_DATASET = DEFAULT_CHECKPOINTS_DIR / "v2_dataset.pt"


def _prepare_dual_samples(raw_samples: list[dict], is_v2: bool = False) -> list[dict]:
    prepared = []
    role_map = V2_ID2ROLE if is_v2 else ID2ROLE
    for sample in raw_samples:
        text = sample.get("text", "")
        tokens, offsets = tokenize_with_offsets(text)
        if not tokens or len(tokens) != len(sample["token_ids"]):
            continue
        logic_ids = tokens_to_logic_ids(tokens)
        dom_tok_targets, abstain_tgt = derive_domain_token_targets(
            tokens=tokens,
            offsets=offsets,
            raw_text=text,
            span_boundaries=sample.get("span_boundaries", []),
            role_labels=sample.get("role_labels", []),
            modality_id=int(sample.get("modality_id", 0)),
            id2role=role_map,
        )
        enriched = dict(sample)
        enriched["logic_token_ids"] = logic_ids
        enriched["domain_tok_targets"] = dom_tok_targets
        enriched["domain_abstain_target"] = abstain_tgt
        prepared.append(enriched)
    return prepared


def train_supervised_dual_stream(
    dataset_path: str | Path | dict = DEFAULT_DATASET,
    checkpoint_out: str | Path = DEFAULT_SUP_CKPT,
    epochs: int = 15,
    lr: float = 3e-4,
    accum_steps: int = 4,
    smoke_test: bool = False,
    v2: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_out = Path(checkpoint_out)
    print("=" * 75)
    print("MUDRAFORGE DECOMPILER - STAGE 1: SUPERVISED DISENTANGLED TRAINING")
    print("=" * 75)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if smoke_test:
        epochs = 1
        print(f"[SMOKE-TEST] Synthesizing 40 in-memory samples (v2={v2}) for fast sanity check...")
        payload = generate_synthetic_dataset(num_samples=40, v2=v2)
    elif isinstance(dataset_path, dict):
        payload = dataset_path
    elif os.path.exists(dataset_path):
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    else:
        print(f"[DATASET] Dataset not found at '{dataset_path}'. Synthesizing automatically (v2={v2})...")
        from compiler.training.dataset_generator import create_comprehensive_dataset
        payload = create_comprehensive_dataset(output_path=str(dataset_path), num_samples=3000, v2=v2)

    is_v2 = bool(v2 or payload.get("architecture_version") == "v2")
    roles_list = payload.get("roles") or (V2_SEMANTIC_ROLES if is_v2 else SEMANTIC_ROLES)
    rels_list = payload.get("relations") or (V2_RELATION_TYPES if is_v2 else RELATION_TYPES)
    bio_list = payload.get("bio_tags") or (V2_BIO_TAGS if is_v2 else BIO_TAGS)
    span_units_list = (payload.get("span_units") or SPAN_UNITS) if is_v2 else []
    num_span_units = len(span_units_list) if is_v2 else 0

    raw_vocab = payload["vocab"]
    train_data = _prepare_dual_samples(payload["train"], is_v2=is_v2)
    val_data = _prepare_dual_samples(payload["val"], is_v2=is_v2)
    print(f"Architecture: {'v2 (Typed-Opcode + SpanUnit)' if is_v2 else 'v1'} | Raw Vocab: {len(raw_vocab)} | Logic Vocab: {len(LOGIC_VOCAB)} tokens")
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    raw_vocab_size = max(len(raw_vocab) + 500, 2048)
    logic_vocab_size = len(LOGIC_VOCAB) + 16

    model = DualStreamStatutoryParser(
        raw_vocab_size=raw_vocab_size,
        logic_vocab_size=logic_vocab_size,
        dropout=0.15,
        num_roles=len(roles_list),
        num_relations=len(rels_list),
        num_bio_tags=len(bio_list),
        num_modalities=len(RULE_MODALITIES),
        num_span_units=num_span_units,
    ).to(device)

    dom_params = sum(p.numel() for p in model.domain_sensor.parameters())
    log_params = sum(p.numel() for p in model.logic_reasoner.parameters())
    print(f"Parameters -> DomainSensor (Model B): {dom_params:,} | LogicReasoner (Model A): {log_params:,} | Total: {dom_params + log_params:,}")

    opt_dom = AdamW(model.domain_sensor.parameters(), lr=lr, weight_decay=1e-3)
    opt_log = AdamW(model.logic_reasoner.parameters(), lr=lr, weight_decay=1e-3)
    sched_dom = CosineAnnealingLR(opt_dom, T_max=epochs, eta_min=1e-5)
    sched_log = CosineAnnealingLR(opt_log, T_max=epochs, eta_min=1e-5)

    rel_weights = torch.ones(len(rels_list), device=device)
    if is_v2:
        freq = [0] * len(rels_list)
        for s in train_data:
            for row in s.get("rel_matrix", []):
                for rid in row:
                    if 0 <= rid < len(freq):
                        freq[rid] += 1
        pos_freqs = [f for f in freq[1:] if f > 0]
        ref_freq = max(pos_freqs) if pos_freqs else 1
        for rid in range(1, len(rels_list)):
            w = math.sqrt(ref_freq / max(1, freq[rid])) if freq[rid] > 0 else 5.0
            rel_weights[rid] = min(5.0, max(1.0, w))
    else:
        rel_weights[1:] = 2.5

    # Up-weight rare but safety-critical domain token classes (citations, doc IDs, GL dest)
    dom_weights = torch.ones(len(DOMAIN_TOKEN_CLASSES), device=device)
    dom_weights[DOMAIN_CLASS2ID["DT_CITATION_HEADER_IGNORE"]] = 2.0
    dom_weights[DOMAIN_CLASS2ID["DT_DOCUMENT_EVIDENCE"]] = 1.8
    dom_weights[DOMAIN_CLASS2ID["DT_ACCOUNTING_DEST"]] = 1.8

    dom_tok_crit = nn.CrossEntropyLoss(weight=dom_weights)
    dom_abst_crit = nn.CrossEntropyLoss()
    bio_crit = nn.CrossEntropyLoss()
    unit_crit = nn.CrossEntropyLoss() if is_v2 else None
    val_role_crit = nn.CrossEntropyLoss()
    val_rel_crit = nn.CrossEntropyLoss(weight=rel_weights)
    val_mod_crit = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    best_score = -1e9
    patience = 6
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        smooth_role = max(0.01, 0.08 * (1.0 - (epoch - 1) / max(1, epochs - 1)))
        smooth_rel = max(0.01, 0.05 * (1.0 - (epoch - 1) / max(1, epochs - 1)))
        role_crit = nn.CrossEntropyLoss(label_smoothing=smooth_role)
        rel_crit = nn.CrossEntropyLoss(weight=rel_weights, label_smoothing=smooth_rel)
        mod_crit = nn.CrossEntropyLoss(label_smoothing=0.03)

        indices = torch.randperm(len(train_data)).tolist()
        opt_dom.zero_grad()
        opt_log.zero_grad()

        for step_i, idx in enumerate(indices):
            s = train_data[idx]
            raw_ids = torch.tensor([s["token_ids"]], dtype=torch.long, device=device)
            log_ids = torch.tensor([s["logic_token_ids"]], dtype=torch.long, device=device)
            suf_ids = torch.tensor([s["suffix_ids"]], dtype=torch.long, device=device)
            dom_tgt = torch.tensor(s["domain_tok_targets"], dtype=torch.long, device=device)
            abst_tgt = torch.tensor([s["domain_abstain_target"]], dtype=torch.long, device=device)
            bio_tgt = torch.tensor(s["bio_ids"], dtype=torch.long, device=device)
            mod_tgt = torch.tensor([s["modality_id"]], dtype=torch.long, device=device)

            spans = s["span_boundaries"]
            c_spans = s.get("clause_spans") or []
            c_mods = s.get("clause_modality_ids") or []
            has_spans = len(spans) > 0
            has_clauses = len(c_spans) > 0 and len(c_spans) == len(c_mods)

            out = model(
                raw_token_ids=raw_ids,
                logic_token_ids=log_ids,
                suffix_ids=suf_ids,
                span_indices=spans if has_spans else None,
                clause_spans=c_spans if has_clauses else None,
                detach_domain_for_logic=True,
                domain_msg_dropout=0.12,
            )

            loss_dom = dom_tok_crit(out["domain_tok_logits"].squeeze(0), dom_tgt) + 0.5 * dom_abst_crit(out["domain_abstain_logits"], abst_tgt)
            loss_bio = bio_crit(out["bio_logits"], bio_tgt)
            loss_mod = mod_crit(out["modality_logits"], mod_tgt)

            if has_spans:
                r_tgt = torch.tensor(s["role_labels"], dtype=torch.long, device=device)
                e_tgt = torch.tensor(s["rel_matrix"], dtype=torch.long, device=device)
                loss_log = (
                    loss_bio
                    + role_crit(out["role_logits"], r_tgt)
                    + 0.5 * rel_crit(out["rel_logits"].view(-1, len(rels_list)), e_tgt.view(-1))
                    + 0.75 * loss_mod
                )
                if is_v2 and out.get("unit_logits") is not None and "unit_labels" in s:
                    u_tgt = torch.tensor(s["unit_labels"], dtype=torch.long, device=device)
                    loss_log = loss_log + 0.75 * unit_crit(out["unit_logits"], u_tgt)
            else:
                loss_log = loss_bio + 0.75 * loss_mod

            if has_clauses and out["clause_mod_logits"] is not None:
                loss_log = loss_log + 0.5 * mod_crit(out["clause_mod_logits"], torch.tensor(c_mods, dtype=torch.long, device=device))

            ((loss_dom + loss_log) / accum_steps).backward()

            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(indices):
                nn.utils.clip_grad_norm_(model.domain_sensor.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(model.logic_reasoner.parameters(), max_norm=1.0)
                opt_dom.step()
                opt_log.step()
                opt_dom.zero_grad()
                opt_log.zero_grad()

        sched_dom.step()
        sched_log.step()

        # Validation
        model.eval()
        val_nll = 0.0
        dom_ok = dom_tot = bio_ok = bio_tot = mod_ok = mod_tot = em_ok = em_tot = unit_ok = unit_tot = 0
        tp_r = [0] * len(roles_list)
        fp_r = [0] * len(roles_list)
        fn_r = [0] * len(roles_list)
        sup_r = [0] * len(roles_list)
        rel_tp = rel_fp = rel_fn = 0

        with torch.no_grad():
            for s in val_data:
                raw_ids = torch.tensor([s["token_ids"]], dtype=torch.long, device=device)
                log_ids = torch.tensor([s["logic_token_ids"]], dtype=torch.long, device=device)
                suf_ids = torch.tensor([s["suffix_ids"]], dtype=torch.long, device=device)
                dom_tgt = torch.tensor(s["domain_tok_targets"], dtype=torch.long, device=device)
                bio_tgt = torch.tensor(s["bio_ids"], dtype=torch.long, device=device)
                mod_tgt = torch.tensor([s["modality_id"]], dtype=torch.long, device=device)
                spans = s["span_boundaries"]
                c_spans = s.get("clause_spans") or []
                c_mods = s.get("clause_modality_ids") or []
                has_spans = len(spans) > 0
                has_clauses = len(c_spans) > 0 and len(c_spans) == len(c_mods)

                out = model(
                    raw_token_ids=raw_ids,
                    logic_token_ids=log_ids,
                    suffix_ids=suf_ids,
                    span_indices=spans if has_spans else None,
                    clause_spans=c_spans if has_clauses else None,
                )
                dom_preds = out["domain_tok_logits"].squeeze(0).argmax(dim=-1)
                dom_ok += int((dom_preds == dom_tgt).sum().item())
                dom_tot += dom_tgt.size(0)

                bio_preds = out["bio_logits"].argmax(dim=-1)
                bio_ok += int((bio_preds == bio_tgt).sum().item())
                bio_tot += bio_tgt.size(0)

                mod_match = bool((out["modality_logits"].argmax(dim=-1) == mod_tgt).all().item())
                mod_ok += int(mod_match)
                mod_tot += 1

                l_v = bio_crit(out["bio_logits"], bio_tgt) + 0.75 * val_mod_crit(out["modality_logits"], mod_tgt)
                if has_clauses and out["clause_mod_logits"] is not None:
                    l_v = l_v + 0.5 * val_mod_crit(out["clause_mod_logits"], torch.tensor(c_mods, dtype=torch.long, device=device))

                if has_spans:
                    r_tgt = torch.tensor(s["role_labels"], dtype=torch.long, device=device)
                    e_tgt = torch.tensor(s["rel_matrix"], dtype=torch.long, device=device)
                    l_v = l_v + val_role_crit(out["role_logits"], r_tgt) + 0.5 * val_rel_crit(out["rel_logits"].view(-1, len(rels_list)), e_tgt.view(-1))

                    unit_match = True
                    if is_v2 and out.get("unit_logits") is not None and "unit_labels" in s:
                        u_tgt = torch.tensor(s["unit_labels"], dtype=torch.long, device=device)
                        l_v = l_v + 0.75 * unit_crit(out["unit_logits"], u_tgt)
                        p_units = out["unit_logits"].argmax(dim=-1)
                        unit_ok += int((p_units == u_tgt).sum().item())
                        unit_tot += u_tgt.size(0)
                        unit_match = bool((p_units == u_tgt).all().item())

                    p_roles = out["role_logits"].argmax(dim=-1)
                    p_rels = out["rel_logits"].argmax(dim=-1)
                    for p, t in zip(p_roles.tolist(), r_tgt.tolist()):
                        sup_r[t] += 1
                        if p == t:
                            tp_r[t] += 1
                        else:
                            fp_r[p] += 1
                            fn_r[t] += 1
                    for pr, tr in zip(p_rels.view(-1).tolist(), e_tgt.view(-1).tolist()):
                        if tr != 0:
                            if pr == tr:
                                rel_tp += 1
                            else:
                                rel_fn += 1
                        elif pr != 0:
                            rel_fp += 1
                    if bool((p_roles == r_tgt).all().item()) and bool((p_rels == e_tgt).all().item()) and mod_match and unit_match:
                        em_ok += 1
                else:
                    if bool((bio_preds == bio_tgt).all().item()) and mod_match:
                        em_ok += 1
                em_tot += 1
                val_nll += l_v.item()

        dom_acc = dom_ok / max(1, dom_tot) * 100.0
        bio_acc = bio_ok / max(1, bio_tot) * 100.0
        mod_acc = mod_ok / max(1, mod_tot) * 100.0
        unit_acc = unit_ok / max(1, unit_tot) * 100.0 if is_v2 else 0.0
        clause_em = em_ok / max(1, em_tot) * 100.0
        avg_nll = val_nll / max(1, len(val_data))

        f1s = []
        for c in range(len(roles_list)):
            if sup_r[c] > 0:
                pc = tp_r[c] / max(1, tp_r[c] + fp_r[c])
                rc = tp_r[c] / max(1, tp_r[c] + fn_r[c])
                f1s.append((2 * pc * rc) / max(1e-6, pc + rc))
        macro_f1 = (sum(f1s) / max(1, len(f1s))) * 100.0
        rel_p = rel_tp / max(1, rel_tp + rel_fp) * 100.0
        rel_r = rel_tp / max(1, rel_tp + rel_fn) * 100.0
        rel_f1 = (2 * rel_p * rel_r) / max(1e-6, rel_p + rel_r)

        if is_v2:
            comp_score = 0.35 * clause_em + 0.15 * macro_f1 + 0.15 * rel_f1 + 0.10 * unit_acc + 0.15 * mod_acc + 0.10 * bio_acc - 5.0 * avg_nll
        else:
            comp_score = 0.40 * clause_em + 0.20 * macro_f1 + 0.15 * rel_f1 + 0.15 * mod_acc + 0.10 * bio_acc - 5.0 * avg_nll
        saved = ""
        if comp_score > best_score:
            best_score = comp_score
            patience_cnt = 0
            ckpt_metrics = {
                "avg_val_nll": round(avg_nll, 4),
                "dom_acc": round(dom_acc, 2),
                "bio_acc": round(bio_acc, 2),
                "mod_acc": round(mod_acc, 2),
                "macro_f1": round(macro_f1, 2),
                "rel_f1": round(rel_f1, 2),
                "clause_em": round(clause_em, 2),
            }
            if is_v2:
                ckpt_metrics["unit_acc"] = round(unit_acc, 2)
            ckpt_cfg = {
                "raw_vocab_size": raw_vocab_size,
                "logic_vocab_size": logic_vocab_size,
                "architecture_version": "v2" if is_v2 else "v1",
                "num_roles": len(roles_list),
                "num_relations": len(rels_list),
                "num_bio_tags": len(bio_list),
                "num_span_units": num_span_units,
            }
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "raw_vocab": raw_vocab,
                "logic_vocab": LOGIC_VOCAB,
                "roles": roles_list,
                "relations": rels_list,
                "bio_tags": bio_list,
                "modalities": RULE_MODALITIES,
                "domain_classes": DOMAIN_TOKEN_CLASSES,
                "metrics": ckpt_metrics,
                "config": ckpt_cfg,
            }
            if is_v2:
                ckpt_payload["span_units"] = span_units_list
            torch.save(ckpt_payload, checkpoint_out)
            saved = " [BEST SAVED]"
        else:
            patience_cnt += 1

        unit_str = f" | Unit: {unit_acc:.2f}%" if is_v2 else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | NLL: {avg_nll:.4f} | DomTok: {dom_acc:.2f}% | "
            f"BIO: {bio_acc:.2f}% | Mod: {mod_acc:.2f}% | F1: {macro_f1:.2f}% | "
            f"RelF1: {rel_f1:.2f}%{unit_str} | EM: {clause_em:.2f}%{saved}",
            flush=True,
        )
        if patience_cnt >= patience:
            print(f"[EarlyStopping] Stopped at Epoch {epoch:02d}.", flush=True)
            break

    print(f"Stage 1 Supervised complete in {time.time() - t0:.1f}s -> {checkpoint_out}")
    return checkpoint_out


def train_grpo_logic_only(
    checkpoint_in: str | Path = DEFAULT_SUP_CKPT,
    checkpoint_out: str | Path = DEFAULT_RL_CKPT,
    dataset_path: str | Path | dict = DEFAULT_DATASET,
    epochs: int = 5,
    group_size: int = 4,
    lr: float = 3e-5,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    bio_kl_beta: float = 0.75,
    mod_kl_beta: float = 0.10,
    smoke_test: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_out = Path(checkpoint_out)
    print("\n" + "=" * 75)
    print("MUDRAFORGE DECOMPILER - STAGE 2: EXECUTION-GUIDED GRPO RL (LOGIC STREAM ONLY)")
    print("=" * 75)

    ckpt = torch.load(checkpoint_in, map_location=device, weights_only=True)
    raw_vocab = ckpt["raw_vocab"]
    cfg = ckpt["config"]
    arch_ver = cfg.get("architecture_version", "v1")
    is_v2 = (arch_ver == "v2")
    roles_list = ckpt.get("roles") or (V2_SEMANTIC_ROLES if is_v2 else SEMANTIC_ROLES)
    rels_list = ckpt.get("relations") or (V2_RELATION_TYPES if is_v2 else RELATION_TYPES)
    bio_list = ckpt.get("bio_tags") or (V2_BIO_TAGS if is_v2 else BIO_TAGS)
    span_units_list = ckpt.get("span_units") or (SPAN_UNITS if is_v2 else [])
    num_span_units = int(cfg.get("num_span_units", len(span_units_list) if is_v2 else 0))

    policy = DualStreamStatutoryParser(
        raw_vocab_size=cfg["raw_vocab_size"],
        logic_vocab_size=cfg["logic_vocab_size"],
        dropout=0.0,
        num_roles=len(roles_list),
        num_relations=len(rels_list),
        num_bio_tags=len(bio_list),
        num_modalities=len(RULE_MODALITIES),
        num_span_units=num_span_units,
    ).to(device)
    policy.load_state_dict(ckpt["model_state_dict"])

    # Strictly freeze Model B (DomainSensor) so RL cannot corrupt citation/header/domain sensing
    policy.domain_sensor.eval()
    for p in policy.domain_sensor.parameters():
        p.requires_grad = False
    policy.logic_reasoner.train()

    ref_logic = copy.deepcopy(policy.logic_reasoner).to(device)
    ref_logic.eval()
    for p in ref_logic.parameters():
        p.requires_grad = False

    optimizer = AdamW(policy.logic_reasoner.parameters(), lr=lr, weight_decay=1e-4)

    if smoke_test:
        epochs = 1
        print(f"[SMOKE-TEST] Synthesizing 30 in-memory samples (v2={is_v2}) for GRPO smoke test...")
        payload = generate_synthetic_dataset(num_samples=30, v2=is_v2)
    elif isinstance(dataset_path, dict):
        payload = dataset_path
    elif os.path.exists(dataset_path):
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    else:
        print(f"[DATASET] Dataset not found at '{dataset_path}'. Synthesizing automatically (v2={is_v2})...")
        from compiler.training.dataset_generator import create_comprehensive_dataset
        payload = create_comprehensive_dataset(output_path=str(dataset_path), num_samples=3000, v2=is_v2)

    train_data = _prepare_dual_samples(payload["train"], is_v2=is_v2)
    by_stype = {}
    for idx_s, s in enumerate(train_data):
        if len(s.get("span_boundaries", [])) == 0 or s.get("modality_name") == "MODALITY_ABSTAIN_UNRESOLVED":
            continue
        by_stype.setdefault(s.get("stype", "UNKNOWN"), []).append(idx_s)

    id2role = {i: r for i, r in enumerate(roles_list)}
    id2rel = {i: r for i, r in enumerate(rels_list)}
    id2mod = {i: m for i, m in enumerate(RULE_MODALITIES)}
    id2unit = {i: u for i, u in enumerate(span_units_list)}

    env = StatutoryRlEnvironment()
    vector_cache = {}
    meta_cache = {}
    best_score = -1e9
    t0 = time.time()

    try:
        for epoch in range(1, epochs + 1):
            epoch_rewards = []
            epoch_passes = 0
            epoch_total = 0
            bio_agrees = []

            per_stype_quota = max(3, min(6, 275 // max(1, len(by_stype))))
            sampled = []
            for st_ids in by_stype.values():
                perm = torch.randperm(len(st_ids))[:per_stype_quota].tolist()
                sampled.extend(st_ids[p] for p in perm)
            indices = [sampled[p] for p in torch.randperm(len(sampled)).tolist()]
            if smoke_test:
                indices = indices[:2]

            for s_idx in indices:
                s = train_data[s_idx]
                spans = s["span_boundaries"]
                raw_text = s.get("text", "")
                raw_toks, raw_offsets = tokenize_with_offsets(raw_text)
                if not spans or not raw_toks:
                    continue

                raw_ids = torch.tensor([s["token_ids"]], dtype=torch.long, device=device)
                log_ids = torch.tensor([s["logic_token_ids"]], dtype=torch.long, device=device)
                suf_ids = torch.tensor([s["suffix_ids"]], dtype=torch.long, device=device)

                with torch.no_grad():
                    _, dom_tok_logits, dom_abst_logits = policy.domain_sensor(raw_ids, suffix_ids=suf_ids)
                    dom_probs = F.softmax(dom_tok_logits, dim=-1)
                    dom_abst_probs = F.softmax(dom_abst_logits, dim=-1)

                if is_v2:
                    h1_pol, pad_m = policy.logic_reasoner.encode_stage1(log_ids, suffix_ids=suf_ids)
                    bio_logits, role_logits, rel_logits, mod_logits, _, unit_logits = policy.logic_reasoner.encode_stage2(
                        h1_pol,
                        domain_probs=dom_probs,
                        domain_abstain_probs=dom_abst_probs,
                        span_indices=spans,
                        mask=pad_m,
                        logic_token_ids=log_ids,
                        return_unit_logits=True,
                    )
                else:
                    bio_logits, role_logits, rel_logits, mod_logits, _ = policy.logic_reasoner(
                        logic_token_ids=log_ids,
                        suffix_ids=suf_ids,
                        domain_probs=dom_probs,
                        domain_abstain_probs=dom_abst_probs,
                        span_indices=spans,
                    )
                    unit_logits = None
                if role_logits is None:
                    continue

                with torch.no_grad():
                    if is_v2:
                        h1_ref, ref_pad_m = ref_logic.encode_stage1(log_ids, suffix_ids=suf_ids)
                        ref_bio_logits, ref_role_logits, ref_rel_logits, ref_mod_logits, _, ref_unit_logits = ref_logic.encode_stage2(
                            h1_ref,
                            domain_probs=dom_probs,
                            domain_abstain_probs=dom_abst_probs,
                            span_indices=spans,
                            mask=ref_pad_m,
                            logic_token_ids=log_ids,
                            return_unit_logits=True,
                        )
                    else:
                        ref_bio_logits, ref_role_logits, ref_rel_logits, ref_mod_logits, _ = ref_logic(
                            logic_token_ids=log_ids,
                            suffix_ids=suf_ids,
                            domain_probs=dom_probs,
                            domain_abstain_probs=dom_abst_probs,
                            span_indices=spans,
                        )
                        ref_unit_logits = None

                bio_kl = F.kl_div(F.log_softmax(bio_logits, dim=-1), F.softmax(ref_bio_logits, dim=-1), reduction="batchmean")
                mod_kl = F.kl_div(F.log_softmax(mod_logits, dim=-1), F.softmax(ref_mod_logits, dim=-1), reduction="batchmean")
                unit_kl = (
                    F.kl_div(F.log_softmax(unit_logits, dim=-1), F.softmax(ref_unit_logits, dim=-1), reduction="batchmean")
                    if (is_v2 and unit_logits is not None and ref_unit_logits is not None)
                    else torch.tensor(0.0, device=device)
                )
                with torch.no_grad():
                    bio_agrees.append((bio_logits.argmax(dim=-1) == ref_bio_logits.argmax(dim=-1)).float().mean().item())

                role_dist = Categorical(logits=role_logits)
                ref_role_dist = Categorical(logits=ref_role_logits)
                mod_dist = Categorical(logits=mod_logits.squeeze(0))
                ref_mod_dist = Categorical(logits=ref_mod_logits.squeeze(0))

                unit_preds = unit_logits.argmax(dim=-1).tolist() if (is_v2 and unit_logits is not None) else []
                eff_rel_logits = (
                    apply_type_compatibility_mask(rel_logits, unit_preds, id2unit=id2unit, id2rel=id2rel)
                    if (is_v2 and rel_logits is not None and unit_preds)
                    else rel_logits
                )
                eff_ref_rel_logits = (
                    apply_type_compatibility_mask(ref_rel_logits, unit_preds, id2unit=id2unit, id2rel=id2rel)
                    if (is_v2 and ref_rel_logits is not None and unit_preds)
                    else ref_rel_logits
                )

                n_sp = len(spans)
                off_diag = [(r_i, c_i) for r_i in range(n_sp) for c_i in range(n_sp) if r_i != c_i]
                has_rels = bool(off_diag and eff_rel_logits is not None and eff_ref_rel_logits is not None)
                if has_rels:
                    pr = torch.tensor([p[0] for p in off_diag], dtype=torch.long, device=device)
                    pc = torch.tensor([p[1] for p in off_diag], dtype=torch.long, device=device)
                    rel_dist = Categorical(logits=eff_rel_logits[pr, pc])
                    ref_rel_dist = Categorical(logits=eff_ref_rel_logits[pr, pc])
                    rel_weight = 0.5 / max(1, n_sp - 1)
                else:
                    rel_dist = ref_rel_dist = None
                    rel_weight = 0.0

                sample_stype = s.get("stype", "194Q_STYLE")
                rule_domain = "GENERAL_LOGIC" if (str(sample_stype).startswith("GENERAL_") or sample_stype == "OBLIGATION_CARVEOUT_GENERAL_STYLE") else "AUTO"

                if s_idx not in meta_cache:
                    clauses = segment_clauses(raw_text, raw_toks, raw_offsets)
                    clause_meta_list = [
                        {
                            "clauseIdx": ci,
                            "charRange": [raw_offsets[c0][0], raw_offsets[c1][1]] if raw_offsets else [0, len(raw_text)],
                        }
                        for ci, (c0, c1) in enumerate(clauses)
                    ] or [{"clauseIdx": 0, "charRange": [0, len(raw_text)]}]
                    meta_list = []
                    for i, (ts, te) in enumerate(spans):
                        cs, ce = raw_offsets[ts][0], raw_offsets[te][1]
                        mid = (ts + te) // 2
                        c_idx = next((ci for ci, (c0, c1) in enumerate(clauses) if c0 <= mid <= c1), 0)
                        meta_list.append({"spanIdx": i, "span": raw_text[cs:ce].strip(), "charRange": [cs, ce], "tokenRange": [ts, te], "clauseIdx": c_idx})
                    meta_cache[s_idx] = (meta_list, clause_meta_list)
                span_meta, clause_meta = meta_cache[s_idx]

                def _assemble(r_ids, e_ids, m_name, u_ids=None):
                    c_spans = [
                        dict(
                            m,
                            role=id2role.get(int(r_ids[i]), "ROLE_IGNORE"),
                            **(
                                {"unit": id2unit.get(int(u_ids[i]), "UNIT_NONE")}
                                if (is_v2 and u_ids is not None and i < len(u_ids))
                                else {}
                            ),
                        )
                        for i, m in enumerate(span_meta)
                    ]
                    if is_v2:
                        c_edges = [
                            {"sourceIdx": si, "targetIdx": ti, "sourceSpan": c_spans[si]["span"], "targetSpan": c_spans[ti]["span"], "relationType": id2rel.get(int(eid), "REL_NO_EDGE")}
                            for (si, ti), eid in zip(off_diag, e_ids)
                            if id2rel.get(int(eid), "REL_NO_EDGE") != "REL_NO_EDGE" and si != ti
                        ]
                        try:
                            return synthesize_usvm_ast_v2(
                                text=raw_text,
                                spans=c_spans,
                                edges=c_edges,
                                clause_modalities=clause_meta,
                                dominant_modality=m_name,
                            )
                        except V2LoweringError:
                            return {}
                    if m_name == "MODALITY_ABSTAIN_UNRESOLVED":
                        return {
                            "version": "0.1.0-draft",
                            "executionMode": "DEVELOPMENT",
                            "ruleDomain": rule_domain if rule_domain != "AUTO" else "GENERAL_LOGIC",
                            "predictedModality": m_name,
                            "statutoryInvariants": {"isAuthorizedToDisburse": False, "requiresHumanApprovalForOutcomes": True},
                            "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "PROHIBITION"},
                            "computation": {"archetype": "PURE_LOGIC_PIPELINE", "outputMode": "PURE_LOGIC", "pipeline": {"steps": [{"assign": "abstained", "expr": True}], "output": {"isApplicable": False, "isPermitted": False, "isCompliant": False, "isBlocked": True}}},
                        }
                    c_edges = [
                        {"sourceIdx": si, "targetIdx": ti, "sourceSpan": c_spans[si]["span"], "targetSpan": c_spans[ti]["span"], "relationType": id2rel.get(int(eid), "REL_NO_EDGE")}
                        for (si, ti), eid in zip(off_diag, e_ids)
                        if id2rel.get(int(eid), "REL_NO_EDGE") != "REL_NO_EDGE" and si != ti
                    ]
                    return synthesize_usvm_ast(text=raw_text, spans=c_spans, edges=c_edges, predicted_modality=m_name, rule_domain=rule_domain)

                if s_idx not in vector_cache:
                    gold_ast = _assemble(
                        s["role_labels"],
                        [int(s["rel_matrix"][ri][ci]) for (ri, ci) in off_diag],
                        s.get("modality_name") or id2mod.get(int(s.get("modality_id", 0))),
                        u_ids=s.get("unit_labels") if is_v2 else None,
                    )
                    vector_cache[s_idx] = env.build_sample_vectors(gold_ast, stype=sample_stype)
                sample_vectors = vector_cache[s_idx]

                g_log_probs = []
                g_ref_probs = []
                cand_asts = []
                for _ in range(group_size):
                    r_act = role_dist.sample()
                    m_act = mod_dist.sample()
                    lp = role_dist.log_prob(r_act).sum() + 0.5 * mod_dist.log_prob(m_act)
                    rlp = ref_role_dist.log_prob(r_act).sum() + 0.5 * ref_mod_dist.log_prob(m_act)
                    if has_rels:
                        e_act = rel_dist.sample()
                        lp = lp + rel_weight * rel_dist.log_prob(e_act).sum()
                        rlp = rlp + rel_weight * ref_rel_dist.log_prob(e_act).sum()
                        e_list = e_act.tolist()
                    else:
                        e_list = []
                    g_log_probs.append(lp)
                    g_ref_probs.append(rlp)
                    cand_asts.append(_assemble(r_act.tolist(), e_list, id2mod.get(int(m_act.item()), "MODALITY_COMPUTATIONAL_EXCESS"), u_ids=unit_preds if is_v2 else None))

                g_evals = env.compute_reward_group(cand_asts, vectors=sample_vectors, stype=sample_stype)
                g_rewards = []
                for rew, bd in g_evals:
                    g_rewards.append(rew)
                    epoch_rewards.append(rew)
                    epoch_passes += bd["vector_passes"]
                    epoch_total += bd["vector_total"]

                r_t = torch.tensor(g_rewards, dtype=torch.float32, device=device)
                advs = (r_t - r_t.mean()) / (r_t.std(unbiased=False) + 1e-6)
                losses = []
                for g in range(group_size):
                    ratio = torch.exp(g_log_probs[g] - g_ref_probs[g])
                    s1 = ratio * advs[g]
                    s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advs[g]
                    approx_kl = ratio - 1.0 - (g_log_probs[g] - g_ref_probs[g])
                    losses.append(-torch.min(s1, s2) + kl_beta * approx_kl)

                total_loss = torch.stack(losses).mean() + bio_kl_beta * bio_kl + mod_kl_beta * mod_kl + 0.1 * unit_kl
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.logic_reasoner.parameters(), max_norm=0.5)
                optimizer.step()

            avg_rew = sum(epoch_rewards) / max(1, len(epoch_rewards))
            pass_rate = epoch_passes / max(1, epoch_total) * 100.0
            bio_agree = sum(bio_agrees) / max(1, len(bio_agrees)) * 100.0
            score = round(avg_rew + 0.10 * pass_rate, 4)

            tag = ""
            if score > best_score:
                best_score = score
                rl_payload = {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "raw_vocab": raw_vocab,
                    "logic_vocab": LOGIC_VOCAB,
                    "roles": roles_list,
                    "relations": rels_list,
                    "bio_tags": bio_list,
                    "modalities": RULE_MODALITIES,
                    "domain_classes": DOMAIN_TOKEN_CLASSES,
                    "avg_reward": avg_rew,
                    "vector_pass_rate": pass_rate,
                    "bio_agree": bio_agree,
                    "config": cfg,
                }
                if is_v2:
                    rl_payload["span_units"] = span_units_list
                torch.save(rl_payload, checkpoint_out)
                tag = " [BEST SAVED]"

            print(
                f"RL Epoch {epoch:02d}/{epochs:02d} | Mean USVM Reward: {avg_rew:+.2f} | "
                f"Vector Pass: {pass_rate:.1f}% | BIO Agree: {bio_agree:.2f}%{tag}",
                flush=True,
            )
    finally:
        env.evaluator.close()

    print(f"Stage 2 RL (Logic Stream Only) completed in {time.time() - t0:.1f}s -> {checkpoint_out}")
    return checkpoint_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MudraForge Dual-Stream Statutory Training Engine")
    parser.add_argument("--stage", choices=["sup", "grpo", "all"], default="all", help="Training stage to execute")
    parser.add_argument("--v2", action="store_true", help="Train Model v2 typed-opcode architecture")
    parser.add_argument("--sup-ckpt", default=None, help="Supervised checkpoint output path")
    parser.add_argument("--rl-ckpt", default=None, help="GRPO RL checkpoint output path")
    parser.add_argument("--dataset", default=None, help="Dataset path")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for training (default: 15 for sup, 5 for grpo)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-epoch sanity check on synthetic samples")
    args = parser.parse_args()

    sup_ckpt_path = args.sup_ckpt or str(DEFAULT_V2_SUP_CKPT if args.v2 else DEFAULT_SUP_CKPT)
    rl_ckpt_path = args.rl_ckpt or str(DEFAULT_V2_RL_CKPT if args.v2 else DEFAULT_RL_CKPT)
    dataset_p = args.dataset or str(DEFAULT_V2_DATASET if args.v2 else DEFAULT_DATASET)

    if args.stage in ("sup", "all"):
        sup_kwargs = {
            "dataset_path": dataset_p,
            "checkpoint_out": sup_ckpt_path,
            "smoke_test": args.smoke_test,
            "v2": args.v2,
        }
        if args.epochs is not None and args.stage == "sup":
            sup_kwargs["epochs"] = args.epochs
        if args.lr is not None and args.stage == "sup":
            sup_kwargs["lr"] = args.lr
        sup_ckpt = train_supervised_dual_stream(**sup_kwargs)
    else:
        sup_ckpt = sup_ckpt_path

    if args.stage in ("grpo", "all"):
        grpo_kwargs = {
            "checkpoint_in": sup_ckpt,
            "checkpoint_out": rl_ckpt_path,
            "dataset_path": dataset_p,
            "smoke_test": args.smoke_test,
        }
        if args.epochs is not None and args.stage in ("grpo", "all"):
            grpo_kwargs["epochs"] = args.epochs
        if args.lr is not None and args.stage in ("grpo", "all"):
            grpo_kwargs["lr"] = args.lr
        train_grpo_logic_only(**grpo_kwargs)
