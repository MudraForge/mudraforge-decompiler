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
from compiler.neural.inference import synthesize_usvm_ast
from compiler.neural.literals import segment_clauses
from compiler.neural.inference_v2 import apply_type_compatibility_mask, synthesize_usvm_ast_v2
from compiler.neural.ssa_walker import V2LoweringError
from compiler.training.discrepancy_monitor import (
    verify_v2_dataset_contract,
    audit_v2_epoch_predictions,
    append_discrepancy_record,
)
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


def _collate_dual_batch(batch_items: list[dict], is_v2: bool, device: torch.device | None = None) -> dict:
    """Collates a variable-length minibatch into padded token tensors and packed span/clause lists."""
    B = len(batch_items)
    seq_lens = [len(s["token_ids"]) for s in batch_items]
    T_max = max(seq_lens)

    raw_ids = torch.zeros((B, T_max), dtype=torch.long)
    log_ids = torch.zeros((B, T_max), dtype=torch.long)
    suf_ids = torch.zeros((B, T_max), dtype=torch.long)
    pad_mask = torch.ones((B, T_max), dtype=torch.bool)
    dom_tgt = torch.full((B, T_max), -100, dtype=torch.long)
    bio_tgt = torch.full((B, T_max), -100, dtype=torch.long)
    abst_tgt = torch.zeros(B, dtype=torch.long)
    mod_tgt = torch.zeros(B, dtype=torch.long)

    batch_spans: list[list[tuple[int, int]]] = []
    batch_c_spans: list[list[tuple[int, int]]] = []
    r_list: list[int] = []
    u_list: list[int] = []
    e_parts: list[torch.Tensor] = []
    c_mod_list: list[int] = []
    item_span_slices: list[tuple[int, int]] = []
    span_cursor = 0

    for b, s in enumerate(batch_items):
        L = seq_lens[b]
        raw_ids[b, :L] = torch.as_tensor(s["token_ids"], dtype=torch.long)
        log_ids[b, :L] = torch.as_tensor(s["logic_token_ids"], dtype=torch.long)
        suf_ids[b, :L] = torch.as_tensor(s["suffix_ids"], dtype=torch.long)
        pad_mask[b, :L] = False
        dom_tgt[b, :L] = torch.as_tensor(s["domain_tok_targets"], dtype=torch.long)
        bio_tgt[b, :L] = torch.as_tensor(s["bio_ids"], dtype=torch.long)
        abst_tgt[b] = int(s["domain_abstain_target"])
        mod_tgt[b] = int(s["modality_id"])

        spans = s["span_boundaries"]
        batch_spans.append(spans)
        cnt = len(spans)
        item_span_slices.append((span_cursor, span_cursor + cnt))
        span_cursor += cnt
        if cnt > 0:
            r_list.extend(s["role_labels"])
            e_parts.append(torch.as_tensor(s["rel_matrix"], dtype=torch.long))
            if is_v2 and "unit_labels" in s:
                u_list.extend(s["unit_labels"])

        c_spans = s.get("clause_spans") or []
        c_mods = s.get("clause_modality_ids") or []
        if c_spans and len(c_spans) == len(c_mods):
            batch_c_spans.append(c_spans)
            c_mod_list.extend(c_mods)
        else:
            batch_c_spans.append([])

    out = {
        "raw_ids": raw_ids,
        "log_ids": log_ids,
        "suf_ids": suf_ids,
        "pad_mask": pad_mask,
        "seq_lens": seq_lens,
        "dom_tgt": dom_tgt,
        "bio_tgt": bio_tgt,
        "abst_tgt": abst_tgt,
        "mod_tgt": mod_tgt,
        "batch_spans": batch_spans,
        "batch_c_spans": batch_c_spans,
        "item_span_slices": item_span_slices,
        "r_tgt": torch.tensor(r_list, dtype=torch.long) if r_list else None,
        "u_tgt": torch.tensor(u_list, dtype=torch.long) if u_list else None,
        "e_tgt_parts": e_parts,
        "e_tgt_flat": torch.cat([ep.reshape(-1) for ep in e_parts], dim=0) if e_parts else None,
        "c_mod_tgt": torch.tensor(c_mod_list, dtype=torch.long) if c_mod_list else None,
    }
    if device is not None:
        nb = (device.type == "cuda")
        for k in ("raw_ids", "log_ids", "suf_ids", "pad_mask", "dom_tgt", "bio_tgt", "abst_tgt", "mod_tgt", "r_tgt", "u_tgt", "e_tgt_flat", "c_mod_tgt"):
            if out[k] is not None:
                out[k] = out[k].to(device, non_blocking=nb)
    return out


def _iter_length_bucketed_batches(items: list[dict], batch_size: int, is_v2: bool, shuffle: bool = True):
    n = len(items)
    if n == 0:
        return []
    if shuffle:
        # Small length jitter preserves mini-batch diversity while keeping padding overhead near zero
        jitter = torch.randint(0, 9, (n,)).tolist()
        order = sorted(range(n), key=lambda i: len(items[i]["token_ids"]) + jitter[i])
        chunks = [order[i : i + batch_size] for i in range(0, n, batch_size)]
        chunk_perm = torch.randperm(len(chunks)).tolist()
        return [_collate_dual_batch([items[idx] for idx in chunks[ci]], is_v2=is_v2) for ci in chunk_perm]
    order = sorted(range(n), key=lambda i: len(items[i]["token_ids"]))
    chunks = [order[i : i + batch_size] for i in range(0, n, batch_size)]
    return [_collate_dual_batch([items[idx] for idx in ch], is_v2=is_v2) for ch in chunks]


def train_supervised_dual_stream(
    dataset_path: str | Path | dict = DEFAULT_DATASET,
    checkpoint_out: str | Path = DEFAULT_SUP_CKPT,
    epochs: int = 15,
    lr: float = 5e-4,
    batch_size: int = 32,
    accum_steps: int = 1,
    smoke_test: bool = False,
    v2: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    checkpoint_out = Path(checkpoint_out)
    print("=" * 75)
    print("MUDRAFORGE DECOMPILER - STAGE 1: SUPERVISED DISENTANGLED TRAINING")
    print("=" * 75)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "") + f" | Batch: {batch_size} | AMP: {use_amp}")

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
    eff_batch_size = min(batch_size, max(1, len(train_data)))
    print(f"Architecture: {'v2 (Typed-Opcode + SpanUnit)' if is_v2 else 'v1'} | Raw Vocab: {len(raw_vocab)} | Logic Vocab: {len(LOGIC_VOCAB)} tokens")
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)} | Eff Batch: {eff_batch_size}")

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rel_weights = torch.ones(len(rels_list), device=device)
    if is_v2:
        rel_weights[0] = 0.35
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

    dom_tok_crit = nn.CrossEntropyLoss(weight=dom_weights, ignore_index=-100)
    dom_abst_crit = nn.CrossEntropyLoss()
    bio_weights = torch.ones(len(bio_list), device=device)
    if is_v2 and len(bio_list) > 0:
        bio_weights[0] = 0.40
    bio_crit = nn.CrossEntropyLoss(weight=bio_weights, ignore_index=-100)
    unit_crit = nn.CrossEntropyLoss() if is_v2 else None
    val_role_crit = nn.CrossEntropyLoss()
    val_rel_crit = nn.CrossEntropyLoss(weight=rel_weights)
    val_mod_crit = nn.CrossEntropyLoss()

    # Pre-collate validation batches once so zero CPU collation runs during epochs
    val_batches_cpu = _iter_length_bucketed_batches(val_data, batch_size=max(1, eff_batch_size), is_v2=is_v2, shuffle=False)
    val_batches = []
    for vb in val_batches_cpu:
        vb_dev = dict(vb)
        nb = (device.type == "cuda")
        for k in ("raw_ids", "log_ids", "suf_ids", "pad_mask", "dom_tgt", "bio_tgt", "abst_tgt", "mod_tgt", "r_tgt", "u_tgt", "e_tgt_flat", "c_mod_tgt"):
            if vb_dev[k] is not None:
                vb_dev[k] = vb_dev[k].to(device, non_blocking=nb)
        val_batches.append(vb_dev)

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    disc_log_path = os.path.splitext(os.path.abspath(str(checkpoint_out)))[0] + "_discrepancies.jsonl"
    if os.path.exists(disc_log_path):
        os.remove(disc_log_path)
    if is_v2:
        contract_summary = verify_v2_dataset_contract(
            payload,
            max_samples=(24 if smoke_test else 120),
            strict=True,
            log_path=disc_log_path,
        )
        print(
            f"[DISCREPANCY-CHECK] v2 dataset contract verified: "
            f"{contract_summary['passed']}/{contract_summary['checked']} clean "
            f"(errors={contract_summary['errorCount']})",
            flush=True,
        )
    id2bio_map = {i: b for i, b in enumerate(bio_list)}
    id2unit_map = {i: u for i, u in enumerate(span_units_list)}
    id2role_map = {i: r for i, r in enumerate(roles_list)}
    id2rel_map = {i: r for i, r in enumerate(rels_list)}
    best_score = -1e9
    patience = 8 if is_v2 else 6
    patience_cnt = 0
    t0 = time.time()
    num_rels = len(rels_list)
    num_roles = len(roles_list)
    num_bio = len(bio_list)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        smooth_role = max(0.01, 0.08 * (1.0 - (epoch - 1) / max(1, epochs - 1)))
        smooth_rel = max(0.01, 0.05 * (1.0 - (epoch - 1) / max(1, epochs - 1)))
        role_crit = nn.CrossEntropyLoss(label_smoothing=smooth_role)
        rel_crit = nn.CrossEntropyLoss(weight=rel_weights, label_smoothing=smooth_rel)
        mod_crit = nn.CrossEntropyLoss(label_smoothing=0.03)

        train_batches = _iter_length_bucketed_batches(train_data, batch_size=eff_batch_size, is_v2=is_v2, shuffle=True)
        opt_dom.zero_grad(set_to_none=True)
        opt_log.zero_grad(set_to_none=True)

        tr_bio_ok_t = torch.zeros(1, dtype=torch.long, device=device)
        tr_bio_tot_t = torch.zeros(1, dtype=torch.long, device=device)
        tr_rel_tp_t = torch.zeros(1, dtype=torch.long, device=device)
        tr_rel_fp_t = torch.zeros(1, dtype=torch.long, device=device)
        tr_rel_fn_t = torch.zeros(1, dtype=torch.long, device=device)

        nb = (device.type == "cuda")
        opt_stepped = False
        for step_i, b in enumerate(train_batches):
            raw_ids = b["raw_ids"].to(device, non_blocking=nb)
            log_ids = b["log_ids"].to(device, non_blocking=nb)
            suf_ids = b["suf_ids"].to(device, non_blocking=nb)
            pad_mask = b["pad_mask"].to(device, non_blocking=nb)
            dom_tgt = b["dom_tgt"].to(device, non_blocking=nb)
            abst_tgt = b["abst_tgt"].to(device, non_blocking=nb)
            bio_tgt = b["bio_tgt"].to(device, non_blocking=nb)
            mod_tgt = b["mod_tgt"].to(device, non_blocking=nb)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(
                    raw_token_ids=raw_ids,
                    logic_token_ids=log_ids,
                    suffix_ids=suf_ids,
                    mask=pad_mask,
                    batch_span_indices=b["batch_spans"],
                    batch_clause_spans=b["batch_c_spans"],
                    detach_domain_for_logic=True,
                    domain_msg_dropout=0.12,
                )

                loss_dom = (
                    dom_tok_crit(out["domain_tok_logits"].reshape(-1, len(DOMAIN_TOKEN_CLASSES)), dom_tgt.reshape(-1))
                    + 0.5 * dom_abst_crit(out["domain_abstain_logits"], abst_tgt)
                )
                loss_bio = bio_crit(out["bio_logits"].reshape(-1, num_bio), bio_tgt.reshape(-1))
                loss_mod = mod_crit(out["modality_logits"], mod_tgt)
                loss_log = (2.5 if is_v2 else 1.0) * loss_bio + 0.75 * loss_mod

                with torch.no_grad():
                    tr_bio_preds = out["bio_logits"].detach().argmax(dim=-1)
                    tr_valid_tok = ~pad_mask
                    tr_bio_ok_t += ((tr_bio_preds == bio_tgt) & tr_valid_tok).sum()
                    tr_bio_tot_t += tr_valid_tok.sum()

                if b["r_tgt"] is not None and out["role_logits"] is not None:
                    r_tgt = b["r_tgt"].to(device, non_blocking=nb)
                    loss_log = loss_log + role_crit(out["role_logits"], r_tgt)
                    if out["rel_logits"] is not None and b["e_tgt_flat"] is not None:
                        rel_flat = torch.cat([rl.reshape(-1, num_rels) for rl in out["rel_logits"] if rl is not None], dim=0)
                        e_tgt_flat = b["e_tgt_flat"].to(device, non_blocking=nb)
                        loss_log = loss_log + (1.5 if is_v2 else 0.5) * rel_crit(rel_flat, e_tgt_flat)
                        with torch.no_grad():
                            tr_r_preds = rel_flat.detach().argmax(dim=-1)
                            tr_rel_tp_t += ((tr_r_preds == e_tgt_flat) & (e_tgt_flat > 0)).sum()
                            tr_rel_fp_t += ((tr_r_preds != e_tgt_flat) & (tr_r_preds > 0)).sum()
                            tr_rel_fn_t += ((tr_r_preds != e_tgt_flat) & (e_tgt_flat > 0)).sum()
                    if is_v2 and out.get("unit_logits") is not None and b["u_tgt"] is not None:
                        u_tgt = b["u_tgt"].to(device, non_blocking=nb)
                        loss_log = loss_log + 0.75 * unit_crit(out["unit_logits"], u_tgt)

                if out["clause_mod_logits"] is not None and b["c_mod_tgt"] is not None:
                    c_mod_tgt = b["c_mod_tgt"].to(device, non_blocking=nb)
                    loss_log = loss_log + 0.5 * mod_crit(out["clause_mod_logits"], c_mod_tgt)

                total_step_loss = (loss_dom + loss_log) / accum_steps

            scaler.scale(total_step_loss).backward()
            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(train_batches):
                scaler.unscale_(opt_dom)
                scaler.unscale_(opt_log)
                nn.utils.clip_grad_norm_(model.domain_sensor.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(model.logic_reasoner.parameters(), max_norm=1.0)
                prev_scale = scaler.get_scale() if use_amp else 1.0
                scaler.step(opt_dom)
                scaler.step(opt_log)
                scaler.update()
                if not use_amp or scaler.get_scale() >= prev_scale:
                    opt_stepped = True
                opt_dom.zero_grad(set_to_none=True)
                opt_log.zero_grad(set_to_none=True)

        if opt_stepped:
            sched_dom.step()
            sched_log.step()

        # Vectorized batched validation (minimal CPU-GPU syncs)
        model.eval()
        val_nll_sum = 0.0
        dom_ok_t = torch.zeros(1, dtype=torch.long, device=device)
        dom_tot_t = torch.zeros(1, dtype=torch.long, device=device)
        bio_ok_t = torch.zeros(1, dtype=torch.long, device=device)
        bio_tot_t = torch.zeros(1, dtype=torch.long, device=device)
        mod_ok_t = torch.zeros(1, dtype=torch.long, device=device)
        mod_tot = 0
        unit_ok_t = torch.zeros(1, dtype=torch.long, device=device)
        unit_tot_t = torch.zeros(1, dtype=torch.long, device=device)
        em_ok = 0
        em_tot = 0
        tp_r_t = torch.zeros(num_roles, dtype=torch.long, device=device)
        fp_r_t = torch.zeros(num_roles, dtype=torch.long, device=device)
        fn_r_t = torch.zeros(num_roles, dtype=torch.long, device=device)
        sup_r_t = torch.zeros(num_roles, dtype=torch.long, device=device)
        rel_tp = rel_fp = rel_fn = 0

        with torch.no_grad():
            for vb in val_batches:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(
                        raw_token_ids=vb["raw_ids"],
                        logic_token_ids=vb["log_ids"],
                        suffix_ids=vb["suf_ids"],
                        mask=vb["pad_mask"],
                        batch_span_indices=vb["batch_spans"],
                        batch_clause_spans=vb["batch_c_spans"],
                    )
                    valid_tok = ~vb["pad_mask"]
                    l_v = (
                        bio_crit(out["bio_logits"].reshape(-1, num_bio), vb["bio_tgt"].reshape(-1))
                        + 0.75 * val_mod_crit(out["modality_logits"], vb["mod_tgt"])
                    )
                    if out["clause_mod_logits"] is not None and vb["c_mod_tgt"] is not None:
                        l_v = l_v + 0.5 * val_mod_crit(out["clause_mod_logits"], vb["c_mod_tgt"])
                    if vb["r_tgt"] is not None and out["role_logits"] is not None:
                        l_v = l_v + val_role_crit(out["role_logits"], vb["r_tgt"])
                        if out["rel_logits"] is not None and vb["e_tgt_flat"] is not None:
                            rel_flat = torch.cat([rl.reshape(-1, num_rels) for rl in out["rel_logits"] if rl is not None], dim=0)
                            l_v = l_v + 0.5 * val_rel_crit(rel_flat, vb["e_tgt_flat"])
                        if is_v2 and out.get("unit_logits") is not None and vb["u_tgt"] is not None:
                            l_v = l_v + 0.75 * unit_crit(out["unit_logits"], vb["u_tgt"])

                bs = vb["raw_ids"].size(0)
                val_nll_sum += float(l_v.item()) * bs

                dom_preds = out["domain_tok_logits"].argmax(dim=-1)
                dom_ok_t += ((dom_preds == vb["dom_tgt"]) & valid_tok).sum()
                dom_tot_t += valid_tok.sum()

                bio_preds = out["bio_logits"].argmax(dim=-1)
                bio_ok_t += ((bio_preds == vb["bio_tgt"]) & valid_tok).sum()
                bio_tot_t += valid_tok.sum()

                mod_matches = (out["modality_logits"].argmax(dim=-1) == vb["mod_tgt"])
                mod_ok_t += mod_matches.sum()
                mod_tot += bs

                p_roles = out["role_logits"].argmax(dim=-1) if out["role_logits"] is not None else None
                p_units = out["unit_logits"].argmax(dim=-1) if (is_v2 and out.get("unit_logits") is not None) else None
                if p_roles is not None and vb["r_tgt"] is not None:
                    r_t = vb["r_tgt"]
                    sup_r_t += torch.bincount(r_t, minlength=num_roles)
                    hit = (p_roles == r_t)
                    tp_r_t += torch.bincount(r_t[hit], minlength=num_roles)
                    fp_r_t += torch.bincount(p_roles[~hit], minlength=num_roles)
                    fn_r_t += torch.bincount(r_t[~hit], minlength=num_roles)
                if p_units is not None and vb["u_tgt"] is not None:
                    unit_ok_t += (p_units == vb["u_tgt"]).sum()
                    unit_tot_t += vb["u_tgt"].numel()

                mod_m_list = mod_matches.tolist()
                e_part_idx = 0
                for b_i, (s0, s1) in enumerate(vb["item_span_slices"]):
                    em_tot += 1
                    m_ok = bool(mod_m_list[b_i])
                    if s1 > s0:
                        role_ok = bool((p_roles[s0:s1] == vb["r_tgt"][s0:s1]).all().item())
                        unit_ok_item = bool((p_units[s0:s1] == vb["u_tgt"][s0:s1]).all().item()) if (p_units is not None and vb["u_tgt"] is not None) else True
                        rl_mat = out["rel_logits"][b_i].argmax(dim=-1)
                        et_mat = vb["e_tgt_parts"][e_part_idx].to(device)
                        e_part_idx += 1
                        rel_ok_item = bool((rl_mat == et_mat).all().item())
                        non_zero_t = (et_mat != 0)
                        rel_tp += int((non_zero_t & (rl_mat == et_mat)).sum().item())
                        rel_fn += int((non_zero_t & (rl_mat != et_mat)).sum().item())
                        rel_fp += int((~non_zero_t & (rl_mat != 0)).sum().item())
                        if role_ok and rel_ok_item and m_ok and unit_ok_item:
                            em_ok += 1
                    else:
                        L = vb["seq_lens"][b_i]
                        if m_ok and bool((bio_preds[b_i, :L] == vb["bio_tgt"][b_i, :L]).all().item()):
                            em_ok += 1

        dom_acc = float(dom_ok_t.item()) / max(1, int(dom_tot_t.item())) * 100.0
        bio_acc = float(bio_ok_t.item()) / max(1, int(bio_tot_t.item())) * 100.0
        mod_acc = float(mod_ok_t.item()) / max(1, mod_tot) * 100.0
        unit_acc = float(unit_ok_t.item()) / max(1, int(unit_tot_t.item())) * 100.0 if is_v2 else 0.0
        clause_em = em_ok / max(1, em_tot) * 100.0
        avg_nll = val_nll_sum / max(1, len(val_data))

        tp_r = tp_r_t.tolist()
        fp_r = fp_r_t.tolist()
        fn_r = fn_r_t.tolist()
        sup_r = sup_r_t.tolist()
        f1s = []
        for c in range(num_roles):
            if sup_r[c] > 0:
                pc = tp_r[c] / max(1, tp_r[c] + fp_r[c])
                rc = tp_r[c] / max(1, tp_r[c] + fn_r[c])
                f1s.append((2 * pc * rc) / max(1e-6, pc + rc))
        macro_f1 = (sum(f1s) / max(1, len(f1s))) * 100.0
        rel_p = rel_tp / max(1, rel_tp + rel_fp) * 100.0
        rel_r = rel_tp / max(1, rel_tp + rel_fn) * 100.0
        rel_f1 = (2 * rel_p * rel_r) / max(1e-6, rel_p + rel_r)

        audit_summary = (
            audit_v2_epoch_predictions(
                model=model,
                val_items=payload["val"],
                raw_vocab=raw_vocab,
                id2role=id2role_map,
                id2rel=id2rel_map,
                id2bio=id2bio_map,
                id2unit=id2unit_map,
                log_path=disc_log_path,
                epoch=epoch,
                device=device,
                max_samples=(6 if smoke_test else 20),
            )
            if is_v2
            else None
        )

        tr_bio_acc = float(tr_bio_ok_t.item()) / max(1, int(tr_bio_tot_t.item())) * 100.0
        tr_tp_f = float(tr_rel_tp_t.item())
        tr_fp_f = float(tr_rel_fp_t.item())
        tr_fn_f = float(tr_rel_fn_t.item())
        tr_rel_p = tr_tp_f / max(1e-6, tr_tp_f + tr_fp_f) * 100.0 if (tr_tp_f + tr_fp_f) > 0 else 0.0
        tr_rel_r = tr_tp_f / max(1e-6, tr_tp_f + tr_fn_f) * 100.0 if (tr_tp_f + tr_fn_f) > 0 else 0.0
        tr_rel_f1 = (2 * tr_rel_p * tr_rel_r) / max(1e-6, tr_rel_p + tr_rel_r)

        if is_v2:
            comp_score = 0.40 * rel_f1 + 0.30 * bio_acc + 0.20 * macro_f1 + 0.10 * mod_acc
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
                if audit_summary is not None:
                    ckpt_metrics["sem_pass_pct"] = audit_summary["sem_pass_pct"]
                    ckpt_metrics["disc_count"] = audit_summary["discrepancy_count"]
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

        ep_dt = time.time() - ep_t0
        unit_str = f" | Unit: {unit_acc:.2f}%" if is_v2 else ""
        sem_str = (
            f" | SemPass: {audit_summary['sem_pass_pct']:.1f}% | Disc: {audit_summary['discrepancy_count']}"
            if is_v2 and audit_summary is not None
            else ""
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({ep_dt:.1f}s) | NLL: {avg_nll:.4f} | DomTok: {dom_acc:.2f}% | "
            f"TrainBIO: {tr_bio_acc:.2f}% | TrainRelF1: {tr_rel_f1:.2f}% | "
            f"BIO: {bio_acc:.2f}% | Mod: {mod_acc:.2f}% | F1: {macro_f1:.2f}% | "
            f"RelF1: {rel_f1:.2f}%{unit_str} | EM: {clause_em:.2f}%{sem_str}{saved}",
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
    rl_disc_log_path = os.path.splitext(os.path.abspath(str(checkpoint_out)))[0] + "_discrepancies.jsonl"
    if os.path.exists(rl_disc_log_path):
        os.remove(rl_disc_log_path)
    best_score = -1e9
    t0 = time.time()

    try:
        for epoch in range(1, epochs + 1):
            epoch_rewards = []
            epoch_passes = 0
            epoch_total = 0
            bio_agrees = []
            rl_disc_counts = {"lower_fail": 0, "vector_mismatch": 0, "rollout_total": 0}

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
                for cand_ast, (rew, bd) in zip(cand_asts, g_evals):
                    g_rewards.append(rew)
                    epoch_rewards.append(rew)
                    epoch_passes += bd["vector_passes"]
                    epoch_total += bd["vector_total"]
                    rl_disc_counts["rollout_total"] += 1
                    if not cand_ast:
                        rl_disc_counts["lower_fail"] += 1
                    elif bd["vector_passes"] < bd["vector_total"]:
                        rl_disc_counts["vector_mismatch"] += 1

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
            if is_v2:
                append_discrepancy_record(
                    rl_disc_log_path,
                    {
                        "recordType": "GRPO_EPOCH_AUDIT",
                        "epoch": epoch,
                        "passRatePct": round(pass_rate, 2),
                        "meanReward": round(avg_rew, 3),
                        **rl_disc_counts,
                    },
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
    parser.add_argument("--batch-size", type=int, default=32, help="Minibatch size for Stage 1 supervised training")
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
            "batch_size": args.batch_size,
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
