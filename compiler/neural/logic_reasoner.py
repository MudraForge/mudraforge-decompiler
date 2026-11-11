"""
=============================================================================
MODEL A: CLOSED-VOCABULARY DOMAIN-AGNOSTIC LOGIC REASONER & DUAL-STREAM FUSION
=============================================================================

Model A (LogicReasonerModel) operates on a strict ~220-token structural
vocabulary (`LOGIC_VOCAB`) that contains ONLY:
  - Canonical slots (<NOUN_SLOT>, <VERB_SLOT>, <NUM_INT>, <NUM_DEC>, <YEAR_4D>,
    <CURRENCY>, <PERCENT>)
  - Closed-class English syntax (modals, connectives, quantifiers, comparators,
    combinators, time/number words, punctuation)

Domain strings such as "Section 194Q", "contractor", "vessel", or "remuneration"
literally cannot enter Model A's token embeddings. Instead, Model A runs in
parallel with Model B (DomainSensorModel) and receives only Model B's narrow
9-dimensional calibrated Domain Token-Type Probabilities across a controlled
residual gate before proposing BIO spans, SRL roles, biaffine relations, and
rule modalities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .statutory_neural_parser import (
    SPECIAL_TOKENS,
    SEMANTIC_ROLES,
    RELATION_TYPES,
    BIO_TAGS,
    LEGAL_SUFFIXES,
    RULE_MODALITIES,
    canonicalize_token,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
    SpanBoundaryRepresentation,
    DeepBiaffineAttention,
)
from .domain_sensor import DOMAIN_TOKEN_CLASSES, DomainSensorModel, CrossAttentionBlock

MODAL_AND_AUX = {
    "shall", "may", "must", "should", "cannot", "can", "will", "would", "could",
    "to", "not", "has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
}
DETERMINERS_AND_PREP = {
    "the", "a", "an", "this", "that", "these", "those", "whose", "any", "every", "each",
    "all", "no", "none", "some", "neither", "either", "nor", "or",
    "of", "for", "in", "by", "with", "from", "on", "at", "against",
}
VERB_SUFFIXES = ("ed", "ing", "ize", "ise", "ate", "ify")
NOUN_SUFFIXES = ("er", "or", "ee", "ion", "ment", "ity", "ance", "ence", "ship", "ist", "ism", "ness", "ure")

CLOSED_CLASS_LOGIC_WORDS = [
    # Punctuation
    ",", ".", ":", ";", "(", ")", "-", "–", "/", '"', "+", "§",
    # Determiners, articles, quantifiers & pronouns
    "the", "a", "an", "this", "that", "these", "those", "such", "same",
    "any", "all", "every", "each", "no", "none", "neither", "either", "both",
    "some", "many", "much", "few", "fewer", "more", "most", "several",
    "who", "whom", "whose", "which", "what", "when", "whenever", "where", "wherever",
    "whether", "while", "how", "why",
    "it", "its", "his", "her", "their", "they", "them", "he", "she",
    "himself", "herself", "itself", "themselves", "someone", "anyone", "everyone",
    # Logical & conditional connectives / prepositions
    "and", "or", "nor", "but", "if", "unless", "except", "excluding", "excluded", "exclude",
    "provided", "proviso", "notwithstanding", "subject", "until", "since",
    "before", "after", "during", "within", "beyond", "without", "with", "from",
    "into", "onto", "by", "for", "on", "at", "in", "of", "to", "than", "as",
    "against", "between", "among", "across", "under", "over", "above", "below",
    "through", "only", "there", "here", "thereof", "therein", "thereto",
    # Modals, auxiliaries & negation
    "shall", "may", "must", "should", "can", "cannot", "could", "will", "would",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "done", "not",
    # Comparators, arithmetic & statutory combinator keywords
    "less", "lesser", "lower", "smaller", "greater", "higher", "larger", "least",
    "equal", "equals", "equally", "exceed", "exceeds", "exceeded", "exceeding",
    "excess", "surplus", "minus", "plus", "times", "multiplied", "divided",
    "reduced", "increased", "scaled", "aggregate", "combined", "total", "count",
    "sum", "mean", "average", "cap", "capped", "ceiling", "floor",
    "part", "portion", "amount", "value", "slice", "bracket", "brackets", "tier",
    "rate", "base", "threshold", "earlier", "whichever", "otherwise", "allowable",
    "previously", "already", "directly", "respectively", "immediately",
    "term", "means", "mean", "include", "includes", "including",
    "substituted", "substitution", "lieu", "suspended", "suspension", "effective",
    "omitted", "omit", "applied", "apply", "applies", "applicable",
    # Generic temporal / unit & number words
    "day", "days", "week", "weeks", "month", "months", "year", "years", "annum",
    "hour", "hours", "turn", "turns", "round", "rounds", "cycle", "cycles",
    "step", "steps", "period", "periods", "time", "calendar", "financial", "previous",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "lakh", "lakhs", "lac", "lacs", "crore", "crores",
    "million", "billion", "half", "halves", "third", "thirds", "fourth", "fourths", "quarter",
    "per", "cent", "centum", "percent",
]

LOGIC_SPECIAL_TOKENS = dict(SPECIAL_TOKENS)
LOGIC_VOCAB = dict(LOGIC_SPECIAL_TOKENS)
for _w in CLOSED_CLASS_LOGIC_WORDS:
    if _w not in LOGIC_VOCAB:
        LOGIC_VOCAB[_w] = len(LOGIC_VOCAB)
ID2LOGIC_TOK = {v: k for k, v in LOGIC_VOCAB.items()}


def tokens_to_logic_ids(tokens: list[str]) -> list[int]:
    """
    Projects arbitrary raw tokens onto the closed structural `LOGIC_VOCAB`.
    Open-class domain words are forced into syntactic <LEGAL_NOUN> / <LEGAL_VERB>
    or canonical numerical/currency/year/percent slots, making section memorization
    or domain vocabulary leakage impossible by construction.
    """
    unk_id = LOGIC_VOCAB["<UNK>"]
    noun_id = LOGIC_VOCAB["<LEGAL_NOUN>"]
    verb_id = LOGIC_VOCAB["<LEGAL_VERB>"]
    out = []
    n = len(tokens)

    for i, tok in enumerate(tokens):
        if not tok:
            out.append(0)
            continue
        prev_t = tokens[i - 1] if i > 0 else None
        prev_prev_t = tokens[i - 2] if i > 1 else None
        next_t = tokens[i + 1] if i + 1 < n else None

        canon = canonicalize_token(tok, prev_t, next_t)
        if canon in LOGIC_VOCAB:
            out.append(LOGIC_VOCAB[canon])
            continue
        low = tok.lower()
        if low in LOGIC_VOCAB:
            out.append(LOGIC_VOCAB[low])
            continue
        if not tok.isalpha():
            out.append(unk_id)
            continue

        prev_low = prev_t.lower() if prev_t else ""
        prev_prev_low = prev_prev_t.lower() if prev_prev_t else ""
        if prev_low in MODAL_AND_AUX:
            out.append(verb_id)
        elif prev_prev_low in MODAL_AND_AUX and (prev_low.endswith("ly") or prev_low in ("not", "already", "previously", "directly")):
            out.append(verb_id)
        elif low.endswith(VERB_SUFFIXES) and prev_low not in DETERMINERS_AND_PREP:
            out.append(verb_id)
        elif prev_low in DETERMINERS_AND_PREP or low.endswith(NOUN_SUFFIXES):
            out.append(noun_id)
        else:
            out.append(noun_id)
    return out


class LogicReasonerModel(nn.Module):
    """
    Domain-agnostic structural Transformer Reasoner with a parallel typed fusion gate
    that consumes the 9-dimensional Domain Token-Type signal from DomainSensorModel.
    """

    def __init__(
        self,
        logic_vocab_size: int = len(LOGIC_VOCAB) + 16,
        d_model: int = 256,
        suffix_dim: int = 32,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_base_layers: int = 4,
        n_fusion_layers: int = 2,
        span_dim: int = 128,
        dropout: float = 0.1,
        d_domain: int = 192,
        num_domain_classes: int = len(DOMAIN_TOKEN_CLASSES),
        num_roles: int = len(SEMANTIC_ROLES),
        num_relations: int = len(RELATION_TYPES),
        num_bio_tags: int = len(BIO_TAGS),
        num_modalities: int = len(RULE_MODALITIES),
    ):
        super().__init__()
        self.d_model = d_model
        self.d_domain = d_domain
        self.word_dim = d_model - suffix_dim
        self.suffix_dim = suffix_dim
        self.num_domain_classes = num_domain_classes
        self.num_roles = num_roles
        self.num_relations = num_relations
        self.num_bio_tags = num_bio_tags
        self.num_modalities = num_modalities

        self.word_embedding = nn.Embedding(logic_vocab_size, self.word_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(len(LEGAL_SUFFIXES), suffix_dim, padding_idx=0)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        self.base_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_base_layers)
        ])
        self.cross_attn_from_domain = CrossAttentionBlock(d_q=d_model, d_kv=d_domain, n_heads=n_heads, dropout=dropout)

        # Parallel Domain->Logic Narrow Typed Bottleneck Bridge
        self.domain_msg_proj = nn.Sequential(
            nn.Linear(num_domain_classes, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )
        self.domain_gate = nn.Sequential(
            nn.Linear(d_model + num_domain_classes, d_model),
            nn.Sigmoid(),
        )
        self.fusion_ln = nn.LayerNorm(d_model)

        self.fusion_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_fusion_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        self.bio_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_bio_tags),
        )
        self.span_pooler = SpanBoundaryRepresentation(d_model, span_dim)
        self.role_classifier = nn.Sequential(
            nn.Linear(span_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(span_dim, num_roles),
        )
        self.relation_classifier = DeepBiaffineAttention(span_dim, num_relations)

        # Modality classifier incorporates sequence pooled logic context + pooled domain summary
        self.modality_classifier = nn.Sequential(
            nn.Linear(d_model * 2 + num_domain_classes + 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_modalities),
        )

    def encode_stage1(
        self,
        logic_token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        w_emb = self.word_embedding(logic_token_ids)
        if suffix_ids is not None:
            s_emb = self.suffix_embedding(suffix_ids)
        else:
            s_emb = torch.zeros(logic_token_ids.size(0), logic_token_ids.size(1), self.suffix_dim, device=logic_token_ids.device)

        x = torch.cat([w_emb, s_emb], dim=-1) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        pad_mask = mask
        if pad_mask is None and (logic_token_ids == 0).any():
            cand = (logic_token_ids == 0)
            if (~cand).any(dim=1).all():
                pad_mask = cand

        mid_pt = len(self.base_layers) // 2
        for layer in self.base_layers[:mid_pt]:
            x = layer(x, mask=pad_mask)
        return x, pad_mask

    def encode_stage2(
        self,
        h_stage1: torch.Tensor,
        domain_context: torch.Tensor = None,
        domain_probs: torch.Tensor = None,
        domain_abstain_probs: torch.Tensor = None,
        span_indices: list[tuple[int, int]] = None,
        clause_spans: list[tuple[int, int]] = None,
        mask: torch.Tensor = None,
        logic_token_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        x = h_stage1
        if domain_context is not None:
            x = self.cross_attn_from_domain(x, domain_context, key_padding_mask=mask)

        mid_pt = len(self.base_layers) // 2
        for layer in self.base_layers[mid_pt:]:
            x = layer(x, mask=mask)

        if domain_probs is not None:
            msg = self.domain_msg_proj(domain_probs)
            gate = self.domain_gate(torch.cat([x, domain_probs], dim=-1))
            x = self.fusion_ln(x + gate * msg)

        for layer in self.fusion_layers:
            x = layer(x, mask=mask)
        h_enc = self.final_ln(x)

        if logic_token_ids is not None:
            valid_mask = (logic_token_ids != 0)
        else:
            valid_mask = torch.ones(h_enc.size(0), h_enc.size(1), dtype=torch.bool, device=h_enc.device)
        if mask is not None:
            valid_mask = valid_mask & (~mask)
        m_float = valid_mask.unsqueeze(-1).type_as(h_enc)
        valid_counts = m_float.sum(dim=1).clamp(min=1.0)
        mean_pool = (h_enc * m_float).sum(dim=1) / valid_counts
        max_pool = h_enc.masked_fill(~valid_mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_pool = torch.where(valid_mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))

        if domain_probs is not None and domain_abstain_probs is not None:
            dom_mean_pool = (domain_probs * m_float).sum(dim=1) / valid_counts
            pooled_mod_in = torch.cat([mean_pool, max_pool, dom_mean_pool, domain_abstain_probs], dim=-1)
        else:
            dummy_dom = torch.zeros(h_enc.size(0), self.num_domain_classes + 2, device=h_enc.device)
            pooled_mod_in = torch.cat([mean_pool, max_pool, dummy_dom], dim=-1)
        modality_logits = self.modality_classifier(pooled_mod_in)

        h_seq = h_enc.squeeze(0) if h_enc.size(0) == 1 else h_enc[0]
        bio_logits = self.bio_classifier(h_seq)

        role_logits = None
        relation_logits = None
        if span_indices:
            span_vectors = self.span_pooler(h_seq, span_indices)
            role_logits = self.role_classifier(span_vectors)
            relation_logits = self.relation_classifier(span_vectors, span_vectors)

        clause_mod_logits = None
        if clause_spans:
            seq_len = h_seq.size(0)
            d_probs_seq = domain_probs.squeeze(0) if (domain_probs is not None and domain_probs.size(0) == 1) else (domain_probs[0] if domain_probs is not None else torch.zeros(seq_len, self.num_domain_classes, device=h_seq.device))
            abst_seq = domain_abstain_probs[0] if domain_abstain_probs is not None else torch.zeros(2, device=h_seq.device)
            clause_feats = []
            for cs, ce in clause_spans:
                c_s = max(0, min(int(cs), seq_len - 1))
                c_e = max(0, min(int(ce), seq_len - 1))
                if c_s > c_e:
                    c_s, c_e = c_e, c_s
                slice_h = h_seq[c_s:c_e + 1]
                slice_d = d_probs_seq[c_s:c_e + 1]
                clause_feats.append(torch.cat([
                    slice_h.mean(dim=0),
                    slice_h.max(dim=0).values,
                    slice_d.mean(dim=0),
                    abst_seq,
                ], dim=-1))
            if clause_feats:
                clause_mod_logits = self.modality_classifier(torch.stack(clause_feats, dim=0))

        return bio_logits, role_logits, relation_logits, modality_logits, clause_mod_logits

    def forward(
        self,
        logic_token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        domain_probs: torch.Tensor = None,
        domain_abstain_probs: torch.Tensor = None,
        domain_context: torch.Tensor = None,
        span_indices: list[tuple[int, int]] = None,
        clause_spans: list[tuple[int, int]] = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        h1, pad_mask = self.encode_stage1(logic_token_ids, suffix_ids=suffix_ids, mask=mask)
        return self.encode_stage2(
            h1,
            domain_context=domain_context,
            domain_probs=domain_probs,
            domain_abstain_probs=domain_abstain_probs,
            span_indices=span_indices,
            clause_spans=clause_spans,
            mask=pad_mask,
            logic_token_ids=logic_token_ids,
        )


class DualStreamStatutoryParser(nn.Module):
    """
    Bidirectional Mutual Co-Attention Dual-Stream Parser combining DomainSensorModel (Model A)
    and LogicReasonerModel (Model B).
    """

    def __init__(
        self,
        raw_vocab_size: int,
        logic_vocab_size: int = len(LOGIC_VOCAB) + 16,
        d_domain: int = 192,
        d_logic: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.domain_sensor = DomainSensorModel(vocab_size=raw_vocab_size, d_domain=d_domain, d_logic=d_logic, dropout=dropout)
        self.logic_reasoner = LogicReasonerModel(logic_vocab_size=logic_vocab_size, d_model=d_logic, d_domain=d_domain, dropout=dropout)

    def forward(
        self,
        raw_token_ids: torch.Tensor,
        logic_token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        span_indices: list[tuple[int, int]] = None,
        clause_spans: list[tuple[int, int]] = None,
        mask: torch.Tensor = None,
        detach_domain_for_logic: bool = True,
        domain_msg_dropout: float = 0.0,
    ) -> dict[str, torch.Tensor | None]:
        # Step 1: Parallel Stage-1 encoding
        h_dom_s1, pad_mask = self.domain_sensor.encode_stage1(raw_token_ids, suffix_ids=suffix_ids, mask=mask)
        h_log_s1, _ = self.logic_reasoner.encode_stage1(logic_token_ids, suffix_ids=suffix_ids, mask=pad_mask)

        # Step 2: Model A cross-attends to Model B's intermediate syntactic state
        h_dom_s2, domain_tok_logits, domain_abstain_logits = self.domain_sensor.encode_stage2(
            h_dom_s1,
            logic_context=h_log_s1.detach() if detach_domain_for_logic else h_log_s1,
            pad_mask=pad_mask,
            raw_token_ids=raw_token_ids,
        )

        dom_probs = F.softmax(domain_tok_logits, dim=-1)
        dom_abst_probs = F.softmax(domain_abstain_logits, dim=-1)
        if detach_domain_for_logic:
            dom_probs = dom_probs.detach()
            dom_abst_probs = dom_abst_probs.detach()

        if self.training and domain_msg_dropout > 0.0:
            drop_mask = (torch.rand(dom_probs.size(0), dom_probs.size(1), 1, device=dom_probs.device) < domain_msg_dropout)
            uniform_dom = torch.full_like(dom_probs, 1.0 / dom_probs.size(-1))
            dom_probs = torch.where(drop_mask, uniform_dom, dom_probs)

        # Step 3: Model B cross-attends to Model A's intermediate lexical state
        bio_logits, role_logits, rel_logits, modality_logits, clause_mod_logits = self.logic_reasoner.encode_stage2(
            h_log_s1,
            domain_context=h_dom_s1.detach() if detach_domain_for_logic else h_dom_s1,
            domain_probs=dom_probs,
            domain_abstain_probs=dom_abst_probs,
            span_indices=span_indices,
            clause_spans=clause_spans,
            mask=pad_mask,
            logic_token_ids=logic_token_ids,
        )

        return {
            "bio_logits": bio_logits,
            "role_logits": role_logits,
            "rel_logits": rel_logits,
            "modality_logits": modality_logits,
            "clause_mod_logits": clause_mod_logits,
            "domain_tok_logits": domain_tok_logits,
            "domain_abstain_logits": domain_abstain_logits,
            "domain_probs": dom_probs,
            "domain_abstain_probs": dom_abst_probs,
        }
