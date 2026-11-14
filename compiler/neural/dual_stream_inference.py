"""
Inference adapter for the Dual-Stream Generalist Reasoner (DomainSensor || LogicReasoner).

Performs constrained Viterbi BIO decoding, span stitching, and neural graph extraction,
then delegates AST lowering strictly to canonical `synthesize_usvm_ast` in `inference.py`.
"""

import os
import re
import sys
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_THIS_DIR)
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from .tokenizer import tokenize_with_offsets
from .statutory_neural_parser import (
    SEMANTIC_ROLES,
    RELATION_TYPES,
    BIO_TAGS,
    RULE_MODALITIES,
    SPAN_UNITS,
    SUFFIX2ID,
    canonicalize_token,
    canonical_suffix,
    decode_bio_spans,
)
from .inference import (
    ALLOWED_RULE_DOMAINS,
    resolve_open_class_token,
    synthesize_usvm_ast,
)
from .literals import (
    segment_clauses,
    parse_numeric_quantity,
    parse_unit_and_type,
    stitch_classified_spans,
)
from .inference_v2 import (
    apply_type_compatibility_mask,
    synthesize_usvm_ast_v2,
)
from .ssa_walker import V2LoweringError
from .domain_sensor import DOMAIN_TOKEN_CLASSES, ID2DOMAIN_CLASS
from .logic_reasoner import (
    LOGIC_VOCAB,
    DualStreamStatutoryParser,
    tokens_to_logic_ids,
)


class DualStreamDecompilerInference:
    """Zero-shot Dual-Stream neuro-symbolic compiler wrapper."""

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Dual-stream checkpoint not found: {checkpoint_path}")

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self.architecture_version = cfg.get("architecture_version", "v1")
        self.vocab = ckpt["raw_vocab"]
        self.logic_vocab = ckpt.get("logic_vocab", LOGIC_VOCAB)
        self.roles = ckpt.get("roles", SEMANTIC_ROLES)
        self.relations = ckpt.get("relations", RELATION_TYPES)
        self.bio_tags = ckpt.get("bio_tags", BIO_TAGS)
        self.modalities = ckpt.get("modalities", RULE_MODALITIES)
        self.span_units = ckpt.get("span_units", SPAN_UNITS)
        self.num_span_units = int(cfg.get("num_span_units", 0))
        self.id2unit = {i: u for i, u in enumerate(self.span_units)}
        self.domain_classes = ckpt.get("domain_classes", DOMAIN_TOKEN_CLASSES)

        self.id2role = {i: r for i, r in enumerate(self.roles)}
        self.id2rel = {i: r for i, r in enumerate(self.relations)}
        self.id2bio = {i: b for i, b in enumerate(self.bio_tags)}
        self.id2mod = {i: m for i, m in enumerate(self.modalities)}
        self.id2dom = {i: d for i, d in enumerate(self.domain_classes)}

        n_tags = len(self.bio_tags)
        self.transition_mask = torch.zeros((n_tags, n_tags), dtype=torch.float32, device=self.device)
        for prev_id, prev_tag in self.id2bio.items():
            for curr_id, curr_tag in self.id2bio.items():
                if prev_tag == "O" and curr_tag.startswith("I-"):
                    self.transition_mask[prev_id, curr_id] = -1e9
                elif prev_tag.startswith(("B-", "I-")) and curr_tag.startswith("I-"):
                    if prev_tag[2:] != curr_tag[2:]:
                        self.transition_mask[prev_id, curr_id] = -1e9

        self.model = DualStreamStatutoryParser(
            raw_vocab_size=cfg["raw_vocab_size"],
            logic_vocab_size=cfg["logic_vocab_size"],
            num_roles=len(self.roles),
            num_relations=len(self.relations),
            num_bio_tags=len(self.bio_tags),
            num_modalities=len(self.modalities),
            num_span_units=self.num_span_units,
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _viterbi_decode(self, logits: torch.Tensor, tokens: list[str]) -> list[int]:
        seq_len, _ = logits.shape
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        dp = torch.empty_like(log_probs)
        backpointers = torch.empty((seq_len, log_probs.size(1)), dtype=torch.long, device=self.device)

        dp[0] = log_probs[0]
        for tag_id, tag in self.id2bio.items():
            if tag.startswith("I-"):
                dp[0, tag_id] = -1e9

        for t in range(1, seq_len):
            tok_lower = tokens[t].lower() if t < len(tokens) else ""
            next_lower = tokens[t + 1].lower() if t + 1 < len(tokens) else ""
            per_not_rate = (tok_lower == "per" and next_lower not in {"cent", "centum", "annum", "%"})
            step_mask = self.transition_mask.clone()
            if per_not_rate:
                for tag_id, tag in self.id2bio.items():
                    if "RATE" in tag:
                        step_mask[:, tag_id] = -1e9

            prev_tok = tokens[t - 1] if t > 0 else ""
            if prev_tok in ("$", "₹", "€", "£") and any(c.isdigit() for c in tokens[t]):
                for tid, tag in self.id2bio.items():
                    if tag in ("B-THRESHOLD", "I-THRESHOLD"):
                        for next_tid, next_tag in self.id2bio.items():
                            if next_tag != "I-THRESHOLD":
                                step_mask[tid, next_tid] = -1e9
                    elif tag in ("B-RATE", "I-RATE"):
                        for next_tid, next_tag in self.id2bio.items():
                            if next_tag != "I-RATE":
                                step_mask[tid, next_tid] = -1e9

            scores = dp[t - 1].unsqueeze(1) + step_mask + log_probs[t].unsqueeze(0)
            dp[t], backpointers[t] = scores.max(dim=0)

        curr = int(dp[-1].argmax().item())
        path = [curr]
        for t in range(seq_len - 1, 0, -1):
            curr = int(backpointers[t, curr].item())
            path.append(curr)
        path.reverse()
        return path

    def decompile_text(
        self,
        text: str,
        source_metadata: dict = None,
        rule_domain: str = "AUTO",
        typed_inputs: bool = False,
        clause_scopes: bool = False,
    ) -> dict:
        req_domain = str(rule_domain or "AUTO").strip().upper()
        if req_domain not in ALLOWED_RULE_DOMAINS:
            raise ValueError(f"Unsupported rule_domain '{rule_domain}'")

        tokens, offsets = tokenize_with_offsets(text)
        if not tokens:
            return {
                "status": "UNRESOLVED_STATUTORY_AMBIGUITY",
                "compiler": "DualStream-Generalist-Lab",
                "ruleDomain": req_domain,
                "calibratedConfidence": 0.0,
                "isAuthorizedToDisburse": False,
                "candidateSets": [],
            }

        raw_ids = []
        suffix_ids = []
        for idx, t in enumerate(tokens):
            p1 = tokens[idx - 1] if idx > 0 else None
            p2 = tokens[idx - 2] if idx > 1 else None
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            raw_ids.append(resolve_open_class_token(t, p1, p2, self.vocab, next_token=nxt))
            canon = canonicalize_token(t, p1, nxt)
            suffix_ids.append(SUFFIX2ID.get(canonical_suffix(t, canon), SUFFIX2ID["<NONE>"]))

        logic_ids = tokens_to_logic_ids(tokens)
        clause_ranges = segment_clauses(text, tokens, offsets)

        raw_tensor = torch.tensor([raw_ids], dtype=torch.long, device=self.device)
        logic_tensor = torch.tensor([logic_ids], dtype=torch.long, device=self.device)
        suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model(
                raw_token_ids=raw_tensor,
                logic_token_ids=logic_tensor,
                suffix_ids=suffix_tensor,
                clause_spans=clause_ranges if len(clause_ranges) > 1 else None,
            )
            mod_probs = torch.softmax(out["modality_logits"].squeeze(0), dim=-1)
            pred_mod_id = int(mod_probs.argmax(dim=-1).item())
            pred_modality = self.id2mod.get(pred_mod_id, "MODALITY_COMPUTATIONAL_EXCESS")
            modality_conf = float(mod_probs[pred_mod_id].item())

            dom_tok_preds = out["domain_probs"].squeeze(0).argmax(dim=-1).cpu().tolist()
            dom_abstain_prob = float(out["domain_abstain_probs"][0, 1].item())

            domain_signals = [
                {
                    "token": tok,
                    "domainClass": self.id2dom.get(cls_id, "DT_OPERATIVE_ENTITY"),
                }
                for tok, cls_id in zip(tokens, dom_tok_preds)
            ]

            clause_modalities = []
            c_mod_logits = out["clause_mod_logits"]
            if c_mod_logits is not None and c_mod_logits.numel() > 0:
                cm_probs = torch.softmax(c_mod_logits, dim=-1)
                for c_idx, (cs_tok, ce_tok) in enumerate(clause_ranges):
                    c_pid = int(cm_probs[c_idx].argmax(dim=-1).item())
                    clause_modalities.append({
                        "clauseIdx": c_idx,
                        "tokenRange": [cs_tok, ce_tok],
                        "charRange": [offsets[cs_tok][0], offsets[ce_tok][1]],
                        "modality": self.id2mod.get(c_pid, "MODALITY_COMPUTATIONAL_EXCESS"),
                        "confidence": round(float(cm_probs[c_idx, c_pid].item()), 4),
                    })
            elif clause_ranges:
                clause_modalities.append({
                    "clauseIdx": 0,
                    "tokenRange": [0, len(tokens) - 1],
                    "charRange": [offsets[0][0], offsets[-1][1]],
                    "modality": pred_modality,
                    "confidence": round(modality_conf, 4),
                })

            if len(clause_modalities) > 1 and pred_modality == "MODALITY_ABSTAIN_UNRESOLVED":
                comp_subs = [cm for cm in clause_modalities if cm["modality"].startswith("MODALITY_COMPUTATIONAL") and cm["confidence"] >= 0.50]
                non_abst = [cm for cm in clause_modalities if cm["modality"] != "MODALITY_ABSTAIN_UNRESOLVED" and cm["confidence"] >= 0.50]
                chosen_cm = max(comp_subs, key=lambda x: x["confidence"]) if comp_subs else (
                    max(non_abst, key=lambda x: x["confidence"]) if non_abst else None
                )
                if chosen_cm is not None:
                    pred_modality = chosen_cm["modality"]
                    modality_conf = max(modality_conf, chosen_cm["confidence"])

            bio_preds = self._viterbi_decode(out["bio_logits"], tokens)
            bio_tags = [self.id2bio[p] for p in bio_preds]

        if pred_modality == "MODALITY_ABSTAIN_UNRESOLVED":
            return {
                "status": "UNRESOLVED_STATUTORY_AMBIGUITY",
                "compiler": "DualStream-Generalist-Lab",
                "ruleDomain": req_domain,
                "predictedModality": pred_modality,
                "modalityConfidence": round(modality_conf, 4),
                "domainAbstainProb": round(dom_abstain_prob, 4),
                "domainSignals": domain_signals,
                "calibratedConfidence": 0.0,
                "statutoryInvariants": {"isAuthorizedToDisburse": False, "requiresHumanApprovalForOutcomes": True},
                "isAuthorizedToDisburse": False,
                "candidateSets": [],
            }

        discovered_spans = decode_bio_spans(bio_tags, tokens, offsets)
        if not discovered_spans:
            return {
                "status": "UNRESOLVED_STATUTORY_AMBIGUITY",
                "compiler": "DualStream-Generalist-Lab",
                "ruleDomain": req_domain,
                "predictedModality": pred_modality,
                "modalityConfidence": round(modality_conf, 4),
                "domainAbstainProb": round(dom_abstain_prob, 4),
                "domainSignals": domain_signals,
                "calibratedConfidence": 0.0,
                "statutoryInvariants": {"isAuthorizedToDisburse": False, "requiresHumanApprovalForOutcomes": True},
                "isAuthorizedToDisburse": False,
                "candidateSets": [],
            }

        span_boundaries = [(s["token_start"], s["token_end"]) for s in discovered_spans]
        with torch.no_grad():
            out2 = self.model(
                raw_token_ids=raw_tensor,
                logic_token_ids=logic_tensor,
                suffix_ids=suffix_tensor,
                span_indices=span_boundaries,
            )
            role_logits = out2["role_logits"]
            rel_logits = out2["rel_logits"]
            unit_logits = out2.get("unit_logits")
            role_probs = torch.softmax(role_logits, dim=-1)
            pred_roles = role_logits.argmax(dim=-1).cpu().tolist()
            pred_confs = role_probs.max(dim=-1).values.cpu().tolist()
            if unit_logits is not None and unit_logits.numel() > 0:
                unit_probs = torch.softmax(unit_logits, dim=-1)
                pred_units = unit_logits.argmax(dim=-1).cpu().tolist()
                pred_unit_confs = unit_probs.max(dim=-1).values.cpu().tolist()
            else:
                pred_units = None
                pred_unit_confs = None

        def _locate_clause_idx(ts: int, te: int) -> int:
            mid = (ts + te) // 2
            for cm in clause_modalities:
                if cm["tokenRange"][0] <= mid <= cm["tokenRange"][1]:
                    return cm["clauseIdx"]
            return 0

        classified_spans = []
        for i, s_info in enumerate(discovered_spans):
            cs, ce = s_info["char_start"], s_info["char_end"]
            entry = {
                "span": text[cs:ce].strip(),
                "bioType": s_info.get("type", "UNKNOWN"),
                "role": self.id2role.get(pred_roles[i], "ROLE_IGNORE"),
                "confidence": round(pred_confs[i], 4),
                "charRange": [cs, ce],
                "tokenRange": [s_info["token_start"], s_info["token_end"]],
                "clauseIdx": _locate_clause_idx(s_info["token_start"], s_info["token_end"]),
            }
            if pred_units is not None:
                entry["unit"] = self.id2unit.get(pred_units[i], "UNIT_NONE")
                entry["unitConfidence"] = round(pred_unit_confs[i], 4)
            classified_spans.append(entry)

        if self.architecture_version == "v2":
            # v2 BIO boundaries are trained end-to-end without legacy stitch heuristics
            old_to_new_span_idx = {i: i for i in range(len(classified_spans))}
        else:
            stitched_spans, old_to_new_span_idx = stitch_classified_spans(classified_spans, text)
            for curr_copy in stitched_spans:
                if curr_copy.get("role") in ("ROLE_COMPUTATION_OPERAND", "ROLE_TRANSACTION_EXCESS_THRESHOLD", "ROLE_TRANSACTION_CLIFF_THRESHOLD") or curr_copy["span"].strip().lower() in ("amount", "sum", "portion", "part", "total", "aggregate", "excess"):
                    c_end = curr_copy["charRange"][1]
                    trailing = text[c_end:]
                    m_prep = re.match(r"^\s+of\s+(?:the\s+)?([a-zA-Z\s_]+?)(?=\s+(?:otherwise\s+allowable|allowable|for\s+such|shall|must|may|over|above|exceeding|,|\.))", trailing, re.IGNORECASE)
                    if m_prep:
                        curr_copy["span"] = text[curr_copy["charRange"][0]:c_end + m_prep.end()].strip()
                        curr_copy["charRange"] = [curr_copy["charRange"][0], c_end + m_prep.end()]
                    elif re.search(r"\bexcess(?:\s+of)?\b", curr_copy["span"], re.IGNORECASE):
                        m_over = re.match(r"^\s+([a-zA-Z\s_]+?)(?=\s+(?:over|above|exceeding|more\s+than)\b)", trailing, re.IGNORECASE)
                        if m_over:
                            curr_copy["span"] = text[curr_copy["charRange"][0]:c_end + m_over.end()].strip()
                            curr_copy["charRange"] = [curr_copy["charRange"][0], c_end + m_over.end()]
            classified_spans = stitched_spans

        rel_edges = []
        seen_edges = set()
        if rel_logits is not None and rel_logits.numel() > 0:
            if self.architecture_version == "v2" and pred_units is not None:
                rel_logits = apply_type_compatibility_mask(
                    rel_logits,
                    pred_units,
                    id2unit=self.id2unit,
                    id2rel=self.id2rel,
                )
            rel_preds = rel_logits.argmax(dim=-1).cpu().tolist()
            num_spans = len(discovered_spans)
            for src in range(num_spans):
                for tgt in range(num_spans):
                    rel_id = rel_preds[src][tgt]
                    if rel_id != 0:
                        sm = old_to_new_span_idx.get(src, src)
                        tm = old_to_new_span_idx.get(tgt, tgt)
                        if sm < len(classified_spans) and tm < len(classified_spans) and sm != tm:
                            rel_name = self.id2rel.get(rel_id, "REL_NO_EDGE")
                            ek = (sm, tm, rel_name)
                            if ek not in seen_edges:
                                seen_edges.add(ek)
                                rel_edges.append({
                                    "sourceIdx": sm,
                                    "targetIdx": tm,
                                    "sourceSpan": classified_spans[sm]["span"],
                                    "targetSpan": classified_spans[tm]["span"],
                                    "relationType": rel_name,
                                })

        if self.architecture_version == "v2":
            try:
                ast_package = synthesize_usvm_ast_v2(
                    text=text,
                    spans=classified_spans,
                    edges=rel_edges,
                    clause_modalities=clause_modalities,
                    dominant_modality=pred_modality,
                    domain_abstain_prob=dom_abstain_prob,
                )
            except V2LoweringError as exc:
                return {
                    "status": "LOWERING_FAILED",
                    "compiler": "DualStream-Generalist-Lab",
                    "architectureVersion": self.architecture_version,
                    "ruleDomain": req_domain,
                    "predictedModality": pred_modality,
                    "modalityConfidence": round(modality_conf, 4),
                    "domainAbstainProb": round(dom_abstain_prob, 4),
                    "domainSignals": domain_signals,
                    "calibratedConfidence": 0.0,
                    "loweringError": str(exc),
                    "classifiedSpans": classified_spans,
                    "relationEdges": rel_edges,
                    "clauseModalities": clause_modalities,
                    "statutoryInvariants": {
                        "isAuthorizedToDisburse": False,
                        "requiresHumanApprovalForOutcomes": True,
                    },
                    "isAuthorizedToDisburse": False,
                }
        else:
            ast_package = synthesize_usvm_ast(
                text=text,
                spans=classified_spans,
                edges=rel_edges,
                source_metadata=source_metadata,
                predicted_modality=pred_modality,
                modality_conf=modality_conf,
                clause_modalities=clause_modalities,
                rule_domain=req_domain,
                typed_inputs=typed_inputs,
                clause_scopes=clause_scopes,
            )

        fn_words = {"in", "of", "to", "for", "by", "on", "at", "from", "the", "a", "an", "and", "or", "be", "that", "with", "as", "whether", "has", "been"}
        subst = [s for s in classified_spans if s["role"] != "ROLE_IGNORE" and not (len(s["span"].split()) == 1 and s["span"].lower() in fn_words)]
        cal_conf = round(sum(s["confidence"] for s in subst) / len(subst), 4) if subst else (
            round(sum(s["confidence"] for s in classified_spans) / len(classified_spans), 4) if classified_spans else 0.0
        )

        return {
            "status": "SUCCESS",
            "compiler": "DualStream-Generalist-Lab",
            "architectureVersion": self.architecture_version,
            "ruleDomain": ast_package.get("ruleDomain", req_domain),
            "predictedModality": pred_modality,
            "modalityConfidence": round(modality_conf, 4),
            "domainAbstainProb": round(dom_abstain_prob, 4),
            "domainSignals": domain_signals,
            "clauseModalities": clause_modalities,
            "calibratedConfidence": cal_conf,
            "sourceLineage": ast_package["sourceLineage"],
            "statutoryInvariants": ast_package["statutoryInvariants"],
            "synthesizedAst": ast_package,
            "symbolLegend": ast_package.get("symbolLegend", {}),
            "classifiedSpansCount": len(classified_spans),
            "classifiedSpans": classified_spans,
            "relationalEdgesCount": len(rel_edges),
            "relationalEdges": rel_edges,
            "isAuthorizedToDisburse": False,
        }
