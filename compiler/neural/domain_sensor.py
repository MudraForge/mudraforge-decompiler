"""
=============================================================================
MODEL B: LEXICAL DOMAIN, CITATION & NOISE SENSOR (DomainSensorModel)
=============================================================================

Reads surface tokens and morphological suffixes in parallel with LogicReasoner.
Learns structural domain typing without memorizing legal constants from
section numbers:
  - Identifies statutory citations, gazette headers, and reference noise
    (DT_CITATION_HEADER_IGNORE) so LogicReasoner ignores non-operative dates
    and section numerals.
  - Identifies monetary values, rates, cutover dates, required documents,
    GL accounting destinations, and non-rule casual chatter.
"""

import re
import torch
import torch.nn as nn

from .statutory_neural_parser import (
    LEGAL_SUFFIXES,
    MODALITY2ID,
    ID2ROLE,
    SinusoidalPositionalEncoding,
    TransformerEncoderBlock,
)

DOMAIN_TOKEN_CLASSES = [
    "DT_OPERATIVE_ENTITY",       # 0: Active rule nouns/verbs/subjects/operands/predicates
    "DT_CITATION_HEADER_IGNORE", # 1: Gazette headers, Section/Rule/Act citations, reference boilerplate
    "DT_STRUCTURAL_CONNECTIVE",  # 2: Closed-class modals, logical connectives, prepositions, comparators
    "DT_MONETARY_AMOUNT",        # 3: Monetary values / thresholds / caps / floors
    "DT_PERCENT_RATE",           # 4: Tax / contribution / penal override percentage rates
    "DT_TEMPORAL_CUTOFF",        # 5: Operative deadlines, effective dates, and sunset cutovers
    "DT_DOCUMENT_EVIDENCE",      # 6: Statutory forms, certificates, returns, affidavits
    "DT_ACCOUNTING_DEST",        # 7: Ledger destinations (profit or loss, OCI, carrying amount)
    "DT_NON_RULE_CHIT_CHAT",     # 8: Non-legal / conversational chatter content
]
DOMAIN_CLASS2ID = {name: i for i, name in enumerate(DOMAIN_TOKEN_CLASSES)}
ID2DOMAIN_CLASS = {i: name for i, name in enumerate(DOMAIN_TOKEN_CLASSES)}

_CONNECTOR_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "such", "same",
    "any", "all", "every", "each", "no", "none", "neither", "either", "both", "some",
    "and", "or", "nor", "but", "if", "unless", "except", "provided", "notwithstanding",
    "subject", "until", "while", "where", "when", "whenever", "whether", "since",
    "before", "after", "during", "within", "beyond", "without", "with", "from",
    "into", "onto", "by", "for", "on", "at", "in", "of", "to", "than", "as",
    "against", "between", "among", "over", "under", "above", "below",
    "shall", "may", "must", "should", "can", "cannot", "could", "will", "would",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "not", "only", "less", "lesser", "lower", "smaller",
    "more", "most", "greater", "higher", "larger", "least", "equal", "equals",
    "exceed", "exceeds", "exceeded", "exceeding", "excess", "surplus",
    "minus", "plus", "reduced", "earlier", "whichever", "otherwise", "allowable",
    ",", ".", ":", ";", "(", ")", "-", "–", "/", '"', "+",
}

_PREAMBLE_PREFIX_RE = re.compile(
    r"^(?:MINISTRY\s+OF|CENTRAL\s+BOARD\s+OF|INSOLVENCY\s+AND\s+BANKRUPTCY\s+BOARD\s+OF|"
    r"SECURITIES\s+AND\s+EXCHANGE\s+BOARD|RESERVE\s+BANK\s+OF\s+INDIA)[^:]{10,280}:\s*",
    flags=re.IGNORECASE,
)


class CrossAttentionBlock(nn.Module):
    """
    Multihead cross-attention layer allowing stream X to query stream Y.
    """
    def __init__(self, d_q: int, d_kv: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_q, num_heads=n_heads, kdim=d_kv, vdim=d_kv, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        return self.norm(q + self.dropout(out))


class DomainSensorModel(nn.Module):
    """
    Compact bidirectional Transformer encoder over raw lexical + suffix embeddings
    emitting calibrated 9-class token-type logits and binary rule-vs-chatter logits.
    Incorporates cross-attention from LogicReasoner to preserve syntactic operands.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        d_domain: int = 192,
        d_logic: int = 256,
        suffix_dim: int = 32,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
        num_domain_classes: int = len(DOMAIN_TOKEN_CLASSES),
    ):
        super().__init__()
        self.d_domain = d_domain
        self.word_dim = d_domain - suffix_dim
        self.suffix_dim = suffix_dim
        self.num_domain_classes = num_domain_classes

        self.word_embedding = nn.Embedding(vocab_size, self.word_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(len(LEGAL_SUFFIXES), suffix_dim, padding_idx=0)
        self.pos_encoder = SinusoidalPositionalEncoding(d_domain)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_domain, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.cross_attn_from_logic = CrossAttentionBlock(d_q=d_domain, d_kv=d_logic, n_heads=n_heads, dropout=dropout)
        self.final_ln = nn.LayerNorm(d_domain)

        self.token_type_head = nn.Sequential(
            nn.Linear(d_domain, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_domain_classes),
        )
        self.abstain_head = nn.Sequential(
            nn.Linear(d_domain * 2, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 2),
        )

    def encode_stage1(
        self,
        raw_token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        w_emb = self.word_embedding(raw_token_ids)
        if suffix_ids is not None:
            s_emb = self.suffix_embedding(suffix_ids)
        else:
            s_emb = torch.zeros(raw_token_ids.size(0), raw_token_ids.size(1), self.suffix_dim, device=raw_token_ids.device)

        x = torch.cat([w_emb, s_emb], dim=-1) * (self.d_domain ** 0.5)
        x = self.pos_encoder(x)

        pad_mask = mask
        if pad_mask is None and (raw_token_ids == 0).any():
            cand = (raw_token_ids == 0)
            if (~cand).any(dim=1).all():
                pad_mask = cand

        mid_pt = len(self.layers) // 2
        for layer in self.layers[:mid_pt]:
            x = layer(x, mask=pad_mask)
        return x, pad_mask

    def encode_stage2(
        self,
        h_stage1: torch.Tensor,
        logic_context: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
        raw_token_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = h_stage1
        if logic_context is not None:
            x = self.cross_attn_from_logic(x, logic_context, key_padding_mask=pad_mask)

        mid_pt = len(self.layers) // 2
        for layer in self.layers[mid_pt:]:
            x = layer(x, mask=pad_mask)
        h_domain = self.final_ln(x)

        domain_tok_logits = self.token_type_head(h_domain)

        if raw_token_ids is not None:
            valid_mask = (raw_token_ids != 0)
        else:
            valid_mask = torch.ones(h_domain.size(0), h_domain.size(1), dtype=torch.bool, device=h_domain.device)
        if pad_mask is not None:
            valid_mask = valid_mask & (~pad_mask)
        m_float = valid_mask.unsqueeze(-1).type_as(h_domain)
        mean_pool = (h_domain * m_float).sum(dim=1) / m_float.sum(dim=1).clamp(min=1.0)
        neg_inf = torch.finfo(h_domain.dtype).min
        max_pool = h_domain.masked_fill(~valid_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        max_pool = torch.where(valid_mask.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        domain_abstain_logits = self.abstain_head(torch.cat([mean_pool, max_pool], dim=-1))

        return h_domain, domain_tok_logits, domain_abstain_logits

    def forward(
        self,
        raw_token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        logic_context: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1, pad_mask = self.encode_stage1(raw_token_ids, suffix_ids=suffix_ids, mask=mask)
        return self.encode_stage2(h1, logic_context=logic_context, pad_mask=pad_mask, raw_token_ids=raw_token_ids)


def derive_domain_token_targets(
    tokens: list[str],
    offsets: list[tuple[int, int]],
    raw_text: str,
    span_boundaries: list[tuple[int, int]],
    role_labels: list[int],
    modality_id: int,
    id2role: dict[int, str] | None = None,
) -> tuple[list[int], int]:
    abstain_id = MODALITY2ID["MODALITY_ABSTAIN_UNRESOLVED"]
    n_tok = len(tokens)

    if modality_id == abstain_id:
        targets = []
        for tok in tokens:
            if tok.lower() in _CONNECTOR_WORDS:
                targets.append(DOMAIN_CLASS2ID["DT_STRUCTURAL_CONNECTIVE"])
            else:
                targets.append(DOMAIN_CLASS2ID["DT_NON_RULE_CHIT_CHAT"])
        return targets, 1

    targets = [
        DOMAIN_CLASS2ID["DT_STRUCTURAL_CONNECTIVE"] if tok.lower() in _CONNECTOR_WORDS else DOMAIN_CLASS2ID["DT_OPERATIVE_ENTITY"]
        for tok in tokens
    ]

    # Mark gazette notification headers as DT_CITATION_HEADER_IGNORE
    m_pre = _PREAMBLE_PREFIX_RE.match(raw_text or "")
    if m_pre and offsets:
        pre_end_char = m_pre.end()
        for t_i, (cs, _ce) in enumerate(offsets):
            if cs < pre_end_char:
                targets[t_i] = DOMAIN_CLASS2ID["DT_CITATION_HEADER_IGNORE"]

    # Overlay role-guided semantic token types
    _CITATION_WORD_RE = re.compile(
        r"\b(?:section|sec\.?|sub-section|clause|sub-clause|rule|act|code|notification|circular|g\.s\.r\.|order|regulation|sched\.?|schedule)\b",
        re.IGNORECASE,
    )
    role_map = id2role or ID2ROLE
    for (ts, te), r_id in zip(span_boundaries, role_labels):
        r_name = role_map.get(int(r_id), "ROLE_IGNORE")
        span_toks = [tokens[k].lower() for k in range(max(0, ts), min(n_tok, te + 1))]
        span_str = " ".join(span_toks)
        is_actual_citation = bool(_CITATION_WORD_RE.search(span_str) or any(w in span_toks for w in ("section", "rule", "circular", "notification", "g.s.r.", "act")))
        for t_i in range(max(0, ts), min(n_tok, te + 1)):
            tok_low = tokens[t_i].lower()
            if r_name == "ROLE_IGNORE":
                if is_actual_citation:
                    targets[t_i] = DOMAIN_CLASS2ID["DT_CITATION_HEADER_IGNORE"]
            elif r_name in ("ROLE_STANDARD_STATUTORY_RATE", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"):
                targets[t_i] = DOMAIN_CLASS2ID["DT_PERCENT_RATE"]
            elif r_name in ("ROLE_TRANSACTION_EXCESS_THRESHOLD", "ROLE_TRANSACTION_CLIFF_THRESHOLD"):
                if tok_low in ("exceed", "exceeds", "exceeded", "exceeding", "over", "above", "under", "below", "not", "up", "to", "less", "more", "than"):
                    targets[t_i] = DOMAIN_CLASS2ID["DT_STRUCTURAL_CONNECTIVE"]
                else:
                    targets[t_i] = DOMAIN_CLASS2ID["DT_MONETARY_AMOUNT"]
            elif r_name in ("ROLE_TEMPORAL_CUTOVER_DATE", "ROLE_TEMPORAL_SUNSET_DATE", "ROLE_DEADLINE_WINDOW"):
                targets[t_i] = DOMAIN_CLASS2ID["DT_TEMPORAL_CUTOFF"]
            elif r_name == "ROLE_REQUIRED_EVIDENCE":
                targets[t_i] = DOMAIN_CLASS2ID["DT_DOCUMENT_EVIDENCE"]
            elif r_name == "ROLE_ACCOUNTING_DESTINATION":
                targets[t_i] = DOMAIN_CLASS2ID["DT_ACCOUNTING_DEST"]

    # Invariant: Numbers preceded or accompanied by currency symbols or relational comparators are DT_MONETARY_AMOUNT, NEVER citation ignore
    for t_i, tok in enumerate(tokens):
        if tok in ("$", "₹", "€", "£") or any(ch.isdigit() for ch in tok):
            prev_tok = tokens[t_i - 1].lower() if t_i > 0 else ""
            if tok in ("$", "₹", "€", "£") or prev_tok in ("$", "₹", "€", "£", "exceeds", "exceeding", "over", "above", "under", "below"):
                targets[t_i] = DOMAIN_CLASS2ID["DT_MONETARY_AMOUNT"]

    return targets, 0
