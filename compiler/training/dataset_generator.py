"""
=============================================================================
MUDRAFORGE FULL COMPREHENSIVE STATUTORY DATASET GENERATOR (v4.0 - UNIVERSAL)
=============================================================================

Generates high-fidelity, syntactically diverse statutory training samples
covering computational tax/financial codes and non-computational legal state machines:
- Direct Tax (194Q, 206C(1H), 194C, 194J, 194H, 194R, 115BAA, sunsets)
- Indirect Tax / GST (17(5) blocked credit, Sec 8 composite supply, Sec 51 TDS)
- Commercial & MSME (MSMED Sec 15 45-day rule, Sec 16 interest, Sec 43B(h))
- Corporate Law (Companies Act Sec 164 disqualifications, Sec 188 related party, Sec 177 audit committee, Sec 135 CSR)
- Insolvency & Bankruptcy (IBC Sec 4 threshold, Sec 29A resolution applicant ineligibility, Sec 7 utility default)
- Civil & Commercial Procedure (CPC Sec 80 notice, Commercial Courts Sec 12A mediation)
- Hard Negative Mining (Gazette dates, section numbers, Act citations labeled ROLE_IGNORE)

Strict Syntactic Generalization Split:
- Formulations 0 (Canonical Active) and 1 (Passive Voice) for Training.
- Formulations 2 (Inverted Condition-First) and 3 (Proviso-First Scrambled) for Validation.
"""

import os
import re
import json
import random
import torch

try:
    from compiler.neural.literals import parse_unit_and_type
    from compiler.neural.statutory_neural_parser import (
        ROLE2ID, ID2ROLE, REL2ID, ID2REL, SEMANTIC_ROLES, RELATION_TYPES,
        BIO2ID, ID2BIO, BIO_TAGS, SUFFIX2ID, extract_suffix,
        RULE_MODALITIES, MODALITY2ID, ID2MODALITY,
        SPECIAL_TOKENS, canonicalize_token, canonical_suffix,
        SPAN_UNITS, UNIT2ID, ID2UNIT,
        V2_SEMANTIC_ROLES, V2_ROLE2ID, V2_ID2ROLE,
        V2_BIO_TAGS, V2_BIO2ID, V2_ID2BIO,
        V2_RELATION_TYPES, V2_REL2ID, V2_ID2REL,
    )
    from compiler.training.ast_composer import CombinatorialAstComposer
except ImportError:
    from literals import parse_unit_and_type
    from statutory_neural_parser import (
        ROLE2ID, ID2ROLE, REL2ID, ID2REL, SEMANTIC_ROLES, RELATION_TYPES,
        BIO2ID, ID2BIO, BIO_TAGS, SUFFIX2ID, extract_suffix,
        RULE_MODALITIES, MODALITY2ID, ID2MODALITY,
        SPECIAL_TOKENS, canonicalize_token, canonical_suffix,
        SPAN_UNITS, UNIT2ID, ID2UNIT,
        V2_SEMANTIC_ROLES, V2_ROLE2ID, V2_ID2ROLE,
        V2_BIO_TAGS, V2_BIO2ID, V2_ID2BIO,
        V2_RELATION_TYPES, V2_REL2ID, V2_ID2REL,
    )
    from ast_composer import CombinatorialAstComposer

ARCHETYPE_MODALITY_MAP = {
    # 0: MODALITY_COMPUTATIONAL_EXCESS
    "194Q_STYLE": "MODALITY_COMPUTATIONAL_EXCESS",
    "194Q_CONCISE_STYLE": "MODALITY_COMPUTATIONAL_EXCESS",
    "206C1H_STYLE": "MODALITY_COMPUTATIONAL_EXCESS",
    "BIZARRE_TRIPLE_THRESHOLD_COLLISION": "MODALITY_COMPUTATIONAL_EXCESS",
    "SUBTRACTION_EXCESS_OPERAND_STYLE": "MODALITY_COMPUTATIONAL_EXCESS",
    # 1: MODALITY_COMPUTATIONAL_CLIFF
    "194C_STYLE": "MODALITY_COMPUTATIONAL_CLIFF",
    "194J_STYLE": "MODALITY_COMPUTATIONAL_CLIFF",
    "194H_STYLE": "MODALITY_COMPUTATIONAL_CLIFF",
    "194R_STYLE": "MODALITY_COMPUTATIONAL_CLIFF",
    "194I_STYLE": "MODALITY_COMPUTATIONAL_CLIFF",
    "STATUTORY_FLAT_COLLECTION_STYLE": "MODALITY_COMPUTATIONAL_CLIFF",
    "BIZARRE_194R_PERQUISITES_IN_KIND": "MODALITY_COMPUTATIONAL_CLIFF",
    "BIZARRE_ONLINE_GAMING_194BA": "MODALITY_COMPUTATIONAL_CLIFF",
    "BIZARRE_DIGITAL_SERVICES_EQUALISATION": "MODALITY_COMPUTATIONAL_CLIFF",
    "BIZARRE_194S_DUAL_CLIFF_CRYPTO_VDA": "MODALITY_COMPUTATIONAL_CLIFF",
    # 2: MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB
    "CORPORATE_TAX_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "MSME_DEADLINE_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "IND_AS_VALUATION_MEASUREMENT_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "BIZARRE_50B_SLUMP_SALE_NEGATIVE_NETWORTH": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "LABOR_GRATUITY_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "COMBINATOR_LESSER_OF_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "COMBINATOR_GREATER_OF_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "CAPPED_WAGE_BASE_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "PROGRESSIVE_BRACKET_CLAUSE_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "TEMPORAL_SUBSTITUTION_OVERRIDE_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "QUALIFICATION_THRESHOLD_DEFINITION_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "MULTI_CLAUSE_COMPOSED_STATUTE_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    # 3: MODALITY_DEONTIC_PROHIBITION
    "GST_BLOCKED_ITC_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    "SECURITIES_INSIDER_TRADING_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    "IND_AS_ELECTION_CARVEOUT_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    "IND_AS_OCI_EQUITY_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    "BIZARRE_269ST_CASH_RECEIPT_PROHIBITION": "MODALITY_DEONTIC_PROHIBITION",
    "SUNSET_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    "ENVIRONMENT_COMPLIANCE_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    "TEMPORAL_SUSPENSION_OVERRIDE_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    # 4: MODALITY_DEONTIC_OBLIGATION_WINDOW
    "PROCEDURAL_NOTICE_WINDOW_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "COMMERCIAL_COURT_MEDIATION_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "DATA_PROTECTION_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "IND_AS_RECOGNITION_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "COMPANIES_AUDIT_COMMITTEE_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "COMPANIES_CSR_MANDATE_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "OBLIGATION_CARVEOUT_GENERAL_STYLE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    # 5: MODALITY_DEONTIC_STATUS_OR_EVIDENCE
    "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "COMPANIES_RELATED_PARTY_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "IBC_RESOLUTION_INELIGIBILITY_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "IBC_INSOLVENCY_TRIGGER_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "SUBJECTIVE_EVIDENCE_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "BIZARRE_COMPANIES_QUORUM_FRACTION": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "STATUS_BAR_DISQUALIFICATION_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    # 6: MODALITY_ABSTAIN_UNRESOLVED
    "ABSTAIN_UNRESOLVED_STYLE": "MODALITY_ABSTAIN_UNRESOLVED",
    # 7: MODALITY_DEONTIC_PERMISSION + Compositional General Logic Archetypes
    "GENERAL_PERMISSION_CONDITIONAL_STYLE": "MODALITY_DEONTIC_PERMISSION",
    "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE": "MODALITY_DEONTIC_PERMISSION",
    "GENERAL_ENTITY_GRAPH_RELATION_STYLE": "MODALITY_DEONTIC_PERMISSION",
    "GENERAL_AGGREGATE_COLLECTION_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE": "MODALITY_DEONTIC_PROHIBITION",
    # Model v2 Typed-Opcode Contrast & Relational Disambiguation Archetypes
    "UNIT_CONTRAST_TEMPORAL_COUNT_STYLE": "MODALITY_COMPUTATIONAL_EXCESS",
    "CREDIT_VS_CAP_COMPOSED_STYLE": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "BOOLEAN_KINSHIP_DEFINED_TERM_STYLE": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    # Model v2 Universal AST-First Combinatorial Families (Zero Section Hacks)
    "V2_AST_COMPOSED_ARITHMETIC": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_BOUNDED_BRACKET": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_BOOLEAN_DAG": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
    "V2_AST_COLLECTION_FOLD": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_SCOPED_SUBSTITUTION": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_DEONTIC_GATE": "MODALITY_DEONTIC_OBLIGATION_WINDOW",
    "V2_AST_FOLD_TO_ARITHMETIC": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_GATED_ARITHMETIC": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_MULTI_TIER_BRACKET": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
    "V2_AST_BRANCHED_RATES": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
}

ROLE_TO_BIO_BASE = {
    "ROLE_PROHIBITION_GATE": "GATE",
    "ROLE_MANDATORY_ACTION_OBLIGATION": "OBLIGATION",
    "ROLE_AGENT_STATE_PRECONDITION": "PRECONDITION",
    "ROLE_BUYER_PRIOR_TURNOVER_GUARD": "PRECONDITION",
    "ROLE_SUBJECTIVE_CRITERION": "PRECONDITION",
    "ROLE_EXCEPTION_CARVEOUT": "EXCEPTION",
    "ROLE_STATUTORY_SANCTION": "SANCTION",
    "ROLE_TRANSACTION_EXCESS_THRESHOLD": "THRESHOLD",
    "ROLE_TRANSACTION_CLIFF_THRESHOLD": "THRESHOLD",
    "ROLE_STANDARD_STATUTORY_RATE": "RATE",
    "ROLE_PENALTY_OR_DEFAULT_OVERRIDE": "RATE",
    "ROLE_NON_PAN_OVERRIDE_RATE": "RATE",
    "ROLE_DEADLINE_WINDOW": "DEADLINE",
    "ROLE_REQUIRED_EVIDENCE": "EVIDENCE",
    "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE": "ENTITY",
    "ROLE_NON_RESIDENT_EXEMPTION_GUARD": "ENTITY",
    "ROLE_COMPUTATION_OPERAND": "ENTITY",
    "ROLE_TRIGGER_EARLIER_EVENT": "PRECONDITION",
    "ROLE_TRIGGER_PAYMENT_ONLY": "PRECONDITION",
    "ROLE_TEMPORAL_CUTOVER_DATE": "DEADLINE",
    "ROLE_TEMPORAL_SUNSET_DATE": "DEADLINE",
    "ROLE_ACCOUNTING_DESTINATION": "DESTINATION",
    "ROLE_IGNORE": "IGNORE"
}

V2_ROLE_TO_BIO_BASE = {
    **ROLE_TO_BIO_BASE,
    "ROLE_COMPUTATION_OPERAND": "OPERAND",
    "ROLE_DEFINED_TERM": "DEFINED_TERM",
    "ROLE_COUNT_QUANTIFIER": "COUNT",
}

_DATE_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)


def infer_span_unit_v2(span_text: str, role_name: str) -> str:
    """Deterministically assigns ground-truth `SPAN_UNITS` class from span text and role."""
    if role_name == "ROLE_IGNORE":
        return "UNIT_NONE"
    if role_name in ("ROLE_DEFINED_TERM", "ROLE_COMPUTATION_OPERAND"):
        return "UNIT_SYMBOL_VAR"
    if role_name in ("ROLE_STANDARD_STATUTORY_RATE", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE", "ROLE_NON_PAN_OVERRIDE_RATE"):
        return "UNIT_RATE"
    if role_name in ("ROLE_TEMPORAL_CUTOVER_DATE", "ROLE_TEMPORAL_SUNSET_DATE"):
        return "UNIT_YEAR_DATE"
    if role_name == "ROLE_DEADLINE_WINDOW":
        return "UNIT_DURATION"
    if role_name == "ROLE_COUNT_QUANTIFIER":
        if re.search(r"\b(?:years?|months?|days?|weeks?|hours?|turns?|cycles?|rounds?)\b", span_text, re.IGNORECASE):
            return "UNIT_DURATION"
        return "UNIT_COUNT"

    low = (span_text or "").strip().lower()
    if re.search(r"\b(?:section|clause|rule|table|article|part|paragraph)\b", low):
        return "UNIT_NONE"
    if _DATE_MONTH_RE.search(low) or re.fullmatch(r"(?:19|20)\d{2}", low):
        return "UNIT_YEAR_DATE"

    st, _, _ = parse_unit_and_type(span_text or "")
    if st == "PERCENTAGE_RATE":
        return "UNIT_RATE"
    if st == "TEMPORAL_DURATION":
        return "UNIT_DURATION"
    if st == "MONETARY_QUANTUM":
        return "UNIT_MONEY"
    if role_name in (
        "ROLE_PROHIBITION_GATE",
        "ROLE_AGENT_STATE_PRECONDITION",
        "ROLE_SUBJECTIVE_CRITERION",
        "ROLE_MANDATORY_ACTION_OBLIGATION",
        "ROLE_EXCEPTION_CARVEOUT",
        "ROLE_STATUTORY_SANCTION",
    ):
        return "UNIT_BOOL_PREDICATE"
    if re.search(r"\b(?:children|dependents?|members?|items?|directors?|shares?|violations?|counts?)\b", low):
        return "UNIT_COUNT"
    return "UNIT_NONE"


def _remap_v2_relation(rel_name: str, src_txt: str, tgt_txt: str, full_text: str, stype: str) -> str:
    """Upgrades ambiguous v1 relation labels into 1-to-1 opcode-exact v2 relations."""
    if rel_name == "REL_CAP_CEILING":
        ctx = f"{src_txt} {tgt_txt} {full_text}".lower()
        if any(tok in ctx for tok in ("reduced by", "minus", "subtract", "credit", "offset")):
            return "REL_SUB_CREDIT"
        return "REL_MIN_CAP"
    if rel_name == "REL_AGGREGATE_SUM":
        ctx = f"{src_txt} {tgt_txt}".lower()
        if any(tok in ctx for tok in ("count", "number of", "headcount", "occurrences")):
            return "REL_AGGREGATE_COUNT"
    if stype == "TEMPORAL_SUBSTITUTION_OVERRIDE_STYLE" and rel_name == "REL_EXCEPTION_TO_PROHIBITION":
        return "REL_SUBSTITUTE_WHEN"
    if rel_name == "REL_APPLIED_TO_BASE":
        return "REL_RATE_TO_THRESHOLD"
    return rel_name

OPEN_CLASS_NOUNS = {
    "buyer", "seller", "director", "company", "contractor", "resident",
    "person", "payee", "deductor", "applicant", "borrower", "recipient",
    "supplier", "shareholder", "member", "assessee", "taxpayer", "transferee",
    "transferor", "customer", "vendor", "client", "industry", "plant",
    "unit", "operator", "owner", "employee", "employer", "worker", "insider",
    "fiduciary", "principal", "dealer", "agent", "auditor", "promoter",
    "lessee", "lessor", "asset", "liability", "provision", "equity", "outflow",
    "inflow", "remeasurement", "actuary", "inventory", "goodwill", "contract",
    "player", "perquisite", "benefit", "winnings", "establishment", "quorum", "undertaking",
    "enemy", "piece", "board", "turn", "turns", "fine", "offence", "participant", "zone",
    "infraction", "violation", "breach", "misconduct", "position", "sector", "restart", "signal", "object", "item", "boundary",
    "node", "edge", "vertex", "milestone", "task", "record", "member", "permit", "badge", "bond", "fee", "deposit", "escrow",
    "tile", "corridor", "gate", "step", "steps", "round", "rounds", "cycle", "cycles", "claim", "allocation"
}

OPEN_CLASS_VERBS = {
    "pay", "paid", "pays", "deduct", "deducted", "deducts", "withhold", "withheld",
    "furnish", "furnished", "furnishes", "file", "filed", "files", "comply", "complied",
    "enter", "entered", "enters", "make", "made", "makes", "deliver", "delivered",
    "transfer", "transferred", "issue", "issued", "issues", "receive", "received", "receives",
    "fail", "failed", "fails", "omit", "omitted", "omits", "spend", "spent", "spends",
    "constitute", "constituted", "notify", "notified", "notifies", "disburse", "disbursed",
    "settle", "settled", "reconcile", "reconciled", "record", "recorded", "recognise", "recognised",
    "recognize", "recognized", "perform", "performed", "execute", "executed", "serve", "served",
    "render", "rendered", "apply", "applied", "incur", "incurred", "exceed", "exceeds", "exceeded",
    "require", "required", "provide", "provided", "disclose", "disclosed", "move", "moved",
    "strike", "strikes", "struck", "pass", "passed", "leave", "leaving",
    "collect", "attack", "attacked", "land", "landing", "paying", "receiving",
    "penalised", "penalized", "sanctioned", "disqualified", "suspended", "removed", "occupying", "handling", "crossing",
    "verify", "verified", "grant", "granted", "reach", "reachable", "connect", "connected", "contain", "contains", "hold", "holds"
}


TOKEN_REGEX = re.compile(
    r"\bper\s+cent(?:um)?\b"
    r"|\b\d{1,2}\.\d{1,2}\.\d{4}\b"
    r"|\b[A-Za-z]+-\d+[A-Za-z0-9\-]*\b"
    r"|\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b"
    r"|\b\d+\.\d+\b"
    r"|[A-Za-z0-9_]+"
    r"|[₹\$€£,\.\(\):;%–\-\/\"+§]",
    flags=re.IGNORECASE,
)

PREAMBLES = [
    "",
    "MINISTRY OF FINANCE (Department of Revenue) NOTIFICATION New Delhi. G.S.R. 320(E).—In exercise of the powers conferred by section 295 of the Income-tax Act, 1961: ",
    "CENTRAL BOARD OF INDIRECT TAXES AND CUSTOMS NOTIFICATION No. 14/2020 – Central Tax. G.S.R. 197(E).—In exercise of the powers conferred by section 164 of the Central Goods and Services Tax Act, 2017: ",
    "MINISTRY OF CORPORATE AFFAIRS NOTIFICATION New Delhi, the 1st June, 2016. S.O. 1934(E).—In exercise of the powers conferred by sub-section (1) and sub-section (2) of section 469 of the Companies Act, 2013: ",
    "INSOLVENCY AND BANKRUPTCY BOARD OF INDIA NOTIFICATION New Delhi, the 30th November, 2016. IBBI/2016-17/GN/REG004.—In exercise of powers conferred under section 240 of the Insolvency and Bankruptcy Code, 2016: ",
    "MINISTRY OF MICRO, SMALL AND MEDIUM ENTERPRISES NOTIFICATION New Delhi, the 26th June, 2020. S.O. 2119(E).—In exercise of powers conferred by sections 7 and 8 of the MSMED Act, 2006: ",
    "CENTRAL BOARD OF DIRECT TAXES CIRCULAR NO. 13 OF 2021 New Delhi, dated the 30th June, 2021. Subject: Guidelines under section 194Q of the Income-tax Act, 1961—reg. "
]


def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens = []
    offsets = []
    for match in TOKEN_REGEX.finditer(text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def char_span_to_token_span(
    char_start: int, char_end: int, offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    tok_start = 0
    tok_end = len(offsets) - 1

    for i, (cs, ce) in enumerate(offsets):
        if cs <= char_start < ce or (cs >= char_start and tok_start == 0):
            tok_start = i
            break

    for i in range(len(offsets) - 1, -1, -1):
        cs, ce = offsets[i]
        if cs < char_end <= ce or (ce <= char_end and tok_end == len(offsets) - 1):
            tok_end = i
            break

    if tok_start > tok_end:
        tok_end = tok_start
    return (tok_start, tok_end)


class ComprehensiveDatasetGenerator:
    """
    Generates multi-domain corporate finance, statutory tax, and non-computational legal samples.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.vocab = dict(SPECIAL_TOKENS)

    def _canonical_seq(self, tokens: list[str]) -> list[str]:
        out = []
        n = len(tokens)
        for i, t in enumerate(tokens):
            p = tokens[i - 1] if i > 0 else None
            nx = tokens[i + 1] if i + 1 < n else None
            out.append(canonicalize_token(t, p, nx))
        return out

    def build_vocab(self, texts: list[str], max_vocab: int = 16384):
        counts = {}
        for text in texts:
            tokens, _ = tokenize_with_offsets(text)
            for ctok in self._canonical_seq(tokens):
                counts[ctok] = counts.get(ctok, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens[:max_vocab - len(self.vocab)]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk_id = self.vocab.get("<UNK>", 1)
        return [self.vocab.get(ctok, unk_id) for ctok in self._canonical_seq(tokens)]

    def generate_all_samples(self, num_samples: int = 4600, v2: bool = False) -> tuple[list[dict], list[dict]]:
        train_samples = []
        val_samples = []

        turnover_gates = ["10 crore", "20 crore", "50 crore", "₹10,00,00,000", "500 crore", "100 crore", "5 crore", "one thousand crore rupees"]
        paidup_cap_gates = ["10 crore", "₹10,00,00,000", "50 crore", "100 crore"]
        networth_gates = ["500 crore", "₹500,00,00,000", "five hundred crore rupees"]
        excess_thresholds = ["50 lakh", "fifty lakh rupees", "₹50,00,000", "₹50 Lakhs", "one crore", "twenty lakh", "₹20,00,000", "one hundred crore rupees", "INR 7 Lakhs", "₹7,00,000"]
        cliff_thresholds = ["30,000 rupees", "₹20,000", "₹1,00,000", "₹50,000", "₹2,40,000", "₹2,50,000", "₹5,000", "one crore rupees", "₹1,00,00,000"]
        tds_goods_rates = ["0.1 per cent", "0.1%"]
        general_rates = ["0.5%", "0.5 per cent", "1 per cent", "1%", "1.5%", "2 per cent", "2%", "2.5%", "10 per cent", "10%", "22%", "15%"]
        penal_rates = ["5 per cent", "5%", "20 per cent", "20%"]
        dates = [
            "1st day of July, 2021", "1st April, 2022", "1st October, 2024", "1st April, 2020",
            "1st April, 2025", "1 April 2026", "1st April, 2026",
            "January 15, 2025", "May 1, 2022", "August 31, 2023", "September 30, 2024", "December 15, 2024"
        ]
        sunsets = [
            "1st day of April, 2025", "1 April 2025", "31st March, 2025",
            "1st October, 2024", "December 15, 2024", "January 15, 2025"
        ]
        deadlines = ["forty-five days", "45 days", "fifteen days", "15 days", "one hundred eighty days", "180 days", "two months", "thirty days", "30 days", "three months", "continuous period of three financial years"]
        evidence_docs = [
            "SUPPLIER_MASTER_SERVICE_AGREEMENT", "INDUSTRY_BUNDLING_PRACTICE_MEMO",
            "TRANSFER_PRICING_STUDY_REPORT", "FORM_3CEB", "EXPORT_LETTER_OF_UNDERTAKING",
            "FORM_MSME_1", "TAX_CLEARANCE_CERTIFICATE_SEC281", "Form DIR-8", "declaration in Form DIR-2",
            "record of default with information utility", "affidavit under section 29A",
            "Schedule K-3", "Form 1116", "Form 1065 Schedule K-1", "Form 26Q", "Form 10F", "Tax Residency Certificate"
        ]

        archetypes = [
            "194Q_STYLE", "194Q_STYLE", "194Q_CONCISE_STYLE", "194Q_CONCISE_STYLE", "206C1H_STYLE", "206C1H_STYLE",
            "194J_STYLE", "194C_STYLE", "194R_STYLE",
            "194I_STYLE", "194H_STYLE", "STATUTORY_FLAT_COLLECTION_STYLE", "STATUTORY_FLAT_COLLECTION_STYLE",
            "GST_BLOCKED_ITC_STYLE", "SUBJECTIVE_EVIDENCE_STYLE", "SUBJECTIVE_EVIDENCE_STYLE",
            "MSME_DEADLINE_STYLE", "CORPORATE_TAX_STYLE", "SUNSET_STYLE",
            "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE", "COMPANIES_RELATED_PARTY_STYLE",
            "COMPANIES_AUDIT_COMMITTEE_STYLE", "COMPANIES_CSR_MANDATE_STYLE",
            "IBC_RESOLUTION_INELIGIBILITY_STYLE", "IBC_INSOLVENCY_TRIGGER_STYLE",
            "PROCEDURAL_NOTICE_WINDOW_STYLE", "COMMERCIAL_COURT_MEDIATION_STYLE",
            "LABOR_GRATUITY_STYLE", "SECURITIES_INSIDER_TRADING_STYLE",
            "ENVIRONMENT_COMPLIANCE_STYLE", "DATA_PROTECTION_STYLE",
            "IND_AS_RECOGNITION_STYLE", "IND_AS_ELECTION_CARVEOUT_STYLE",
            "IND_AS_VALUATION_MEASUREMENT_STYLE", "IND_AS_OCI_EQUITY_STYLE",
            "OBLIGATION_CARVEOUT_GENERAL_STYLE", "OBLIGATION_CARVEOUT_GENERAL_STYLE",
            "OBLIGATION_CARVEOUT_GENERAL_STYLE", "OBLIGATION_CARVEOUT_GENERAL_STYLE",
            "BIZARRE_194R_PERQUISITES_IN_KIND", "BIZARRE_ONLINE_GAMING_194BA",
            "BIZARRE_TRIPLE_THRESHOLD_COLLISION", "BIZARRE_50B_SLUMP_SALE_NEGATIVE_NETWORTH",
            "BIZARRE_DIGITAL_SERVICES_EQUALISATION", "BIZARRE_COMPANIES_QUORUM_FRACTION", "BIZARRE_194S_DUAL_CLIFF_CRYPTO_VDA", "BIZARRE_269ST_CASH_RECEIPT_PROHIBITION",
            "COMBINATOR_LESSER_OF_STYLE", "COMBINATOR_LESSER_OF_STYLE",
            "COMBINATOR_GREATER_OF_STYLE",
            "SUBTRACTION_EXCESS_OPERAND_STYLE", "SUBTRACTION_EXCESS_OPERAND_STYLE",
            "CAPPED_WAGE_BASE_STYLE", "CAPPED_WAGE_BASE_STYLE",
            "PROGRESSIVE_BRACKET_CLAUSE_STYLE", "PROGRESSIVE_BRACKET_CLAUSE_STYLE",
            "TEMPORAL_SUSPENSION_OVERRIDE_STYLE", "TEMPORAL_SUSPENSION_OVERRIDE_STYLE",
            "TEMPORAL_SUBSTITUTION_OVERRIDE_STYLE", "TEMPORAL_SUBSTITUTION_OVERRIDE_STYLE",
            "QUALIFICATION_THRESHOLD_DEFINITION_STYLE",
            "STATUS_BAR_DISQUALIFICATION_STYLE",
            "MULTI_CLAUSE_COMPOSED_STATUTE_STYLE", "MULTI_CLAUSE_COMPOSED_STATUTE_STYLE",
            "MULTI_CLAUSE_COMPOSED_STATUTE_STYLE",
            "GENERAL_PERMISSION_CONDITIONAL_STYLE", "GENERAL_PERMISSION_CONDITIONAL_STYLE",
            "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE", "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE",
            "GENERAL_ENTITY_GRAPH_RELATION_STYLE", "GENERAL_ENTITY_GRAPH_RELATION_STYLE",
            "GENERAL_AGGREGATE_COLLECTION_STYLE", "GENERAL_AGGREGATE_COLLECTION_STYLE",
            "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE", "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE",
            "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE", "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE",
            "ABSTAIN_UNRESOLVED_STYLE", "ABSTAIN_UNRESOLVED_STYLE", "ABSTAIN_UNRESOLVED_STYLE",
            "ABSTAIN_UNRESOLVED_STYLE", "ABSTAIN_UNRESOLVED_STYLE"
        ]
        if v2:
            archetypes.extend([
                "UNIT_CONTRAST_TEMPORAL_COUNT_STYLE", "UNIT_CONTRAST_TEMPORAL_COUNT_STYLE",
                "CREDIT_VS_CAP_COMPOSED_STYLE", "CREDIT_VS_CAP_COMPOSED_STYLE",
                "BOOLEAN_KINSHIP_DEFINED_TERM_STYLE", "BOOLEAN_KINSHIP_DEFINED_TERM_STYLE",
                "PROGRESSIVE_BRACKET_CLAUSE_STYLE",
                "TEMPORAL_SUBSTITUTION_OVERRIDE_STYLE",
                "QUALIFICATION_THRESHOLD_DEFINITION_STYLE",
            ])

        abstain_train_texts = [
            "Good morning, can you help me summarize our quarterly roadmap slides?",
            "What is the weather forecast for Mumbai tomorrow afternoon?",
            "Please schedule a calendar invite for the product team on Friday.",
            "Click the settings gear in the top corner to refresh your workspace token.",
            "The cafeteria serves lunch between twelve thirty and two in the afternoon.",
            "Our distributed cluster uses worker isolates to minimize cold start latency.",
            "Please review the draft slide deck before Friday morning's sync.",
            "The backend server returned a gateway timeout error on port 8080.",
            "See Appendix B for historical diagrams and illustrative screenshots.",
            "Section headings in this handbook are inserted for reader convenience only.",
            "Decompiled rule under existing framework. Decompiled using neural parser.",
            "Here is a draft email thanking the client for attending yesterday's demo.",
            "Check if the deployment log shows any warning messages on staging.",
            "The board room projector needs a replacement cable before the townhall.",
            "Welcome to the documentation portal for developers and system integrators.",
            "Is there any printer paper left in the supply cabinet upstairs?",
            "Is there a quiet meeting room available on the second floor right now?",
            "Are there any vegetarian sandwiches left from the afternoon workshop?",
            "How many minutes does it take to walk from the railway station to the campus?",
            "How many attendees registered for next Wednesday's product walkthrough?",
            "How far is the nearest pharmacy from our downtown office building?",
            "Where can I find the guest wireless password for the third wing?",
            "What time does the shuttle depart from the north lobby today?",
            "Can someone verify if the espresso machine in the lounge has been refilled?",
            "Could you show me how to align two columns inside a CSS grid wrapper?",
            "Who is bringing the display adapters for the monitor in studio four?",
            "When does the cafeteria stop serving breakfast on rainy weekdays?",
            "Why did the automated nightly build pause during step three?",
            "Switch to the light sidebar mode under display preferences and hit save.",
            "Our weekly engineering standup moved from Tuesday noon to Wednesday eleven.",
            "Please remember to water the potted ferns near the main reception desk.",
            "This table of contents lists section numbers without substantive rules.",
            "Sub-section heading inserted solely for typographical formatting clarity.",
            "The shuttle bus usually arrives outside Gate 2 every twenty minutes.",
            "I forgot to pack my laptop charger before leaving the hotel earlier.",
            "Would you mind proof-reading the welcome newsletter before we publish it?"
        ]
        abstain_val_texts = [
            "Could you tell me how to center a div inside a flexbox container?",
            "Is there any coffee left in the pantry on the fourth floor?",
            "Let us catch up tomorrow morning to finalize the slide deck layout.",
            "Open the configuration panel and select your preferred dark theme.",
            "The conference bridge PIN has changed for this afternoon's standup.",
            "Download the CSV export and upload the attachment into shared storage.",
            "Please restart the staging container after editing the environment variables.",
            "The third chapter gives a brief historical background without sample calculations.",
            "Please remind everyone to submit their travel preferences before Thursday.",
            "Our database backup completed without any network timeout alerts.",
            "How many kilometers is the airport from the city center?",
            "This changelog summarizes UI bug fixes and minor dependency bumps.",
        ]

        if v2:
            # Universal AST-First Combinatorial Rule Composition (zero section-specific templates, 80/20 disjoint lexical split)
            composer = CombinatorialAstComposer(seed=self.seed)
            for idx in range(num_samples):
                is_train_split = (idx % 5 != 4)  # exact 80% Train / 20% Val split
                split_name = "train" if is_train_split else "val"
                formulation = random.choice([0, 1, 2, 3])
                if random.random() < 0.15:
                    text = random.choice(abstain_train_texts if is_train_split else abstain_val_texts)
                    sample_item = {
                        "stype": "ABSTAIN_UNRESOLVED_STYLE",
                        "modality_name": "MODALITY_ABSTAIN_UNRESOLVED",
                        "modality_id": MODALITY2ID["MODALITY_ABSTAIN_UNRESOLVED"],
                        "formulation": formulation,
                        "text": text,
                        "labeled_spans": [],
                        "spans_with_units": [],
                        "relations": [],
                        "clauses": [],
                    }
                else:
                    comp = composer.sample_rule(formulation=formulation, split=split_name)
                    text = comp["text"]
                    raw_swu = comp.get("spans_with_units") or []
                    valid_swu = []
                    valid_labeled = []
                    span_idx_map = {}
                    for o_i, (s_txt, r_name, u_name) in enumerate(raw_swu):
                        if s_txt in text:
                            span_idx_map[o_i] = len(valid_labeled)
                            valid_labeled.append((s_txt, r_name))
                            valid_swu.append((s_txt, r_name, u_name))
                    remapped_rels = [
                        (span_idx_map[s], span_idx_map[t], r)
                        for s, t, r in (comp.get("relations") or [])
                        if s in span_idx_map and t in span_idx_map
                    ]
                    mod_name = comp.get("modality_name", "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB")
                    sample_item = {
                        "stype": comp["stype"],
                        "modality_name": mod_name,
                        "modality_id": MODALITY2ID[mod_name],
                        "formulation": formulation,
                        "text": text,
                        "labeled_spans": valid_labeled,
                        "spans_with_units": valid_swu,
                        "relations": remapped_rels,
                        "clauses": comp.get("clauses") or [text],
                        "dag": comp.get("dag"),
                    }
                if is_train_split:
                    train_samples.append(sample_item)
                else:
                    val_samples.append(sample_item)
            random.shuffle(train_samples)
            random.shuffle(val_samples)
            return train_samples, val_samples

        for _ in range(num_samples):
            stype = random.choice(archetypes)
            to_amt = random.choice(turnover_gates)
            th_amt = random.choice(excess_thresholds)
            cf_amt = random.choice(cliff_thresholds)
            std_rate = random.choice(general_rates)
            goods_rate = random.choice(tds_goods_rates)
            penal_rate = random.choice(penal_rates)
            cut_date = random.choice(dates)
            sun_date = random.choice(sunsets)
            dl_str = random.choice(deadlines)
            ev_doc = random.choice(evidence_docs)

            preamble = random.choice(PREAMBLES)
            formulation = random.choice([0, 1, 2, 3])

            labeled_spans = []
            relations = []
            clauses = []
            modality_override = None
            body = ""

            # -----------------------------------------------------------------
            # 1. DIRECT TAX: 194Q
            # -----------------------------------------------------------------
            if stype == "194Q_STYLE":
                to_prefix = random.choice([
                    "turnover",
                    "total sales, gross receipts, or turnover from business"
                ])
                if formulation == 0:
                    body = (
                        f"Any buyer whose {to_prefix} exceeds {to_amt} in the preceding financial year "
                        f"shall deduct tax at {goods_rate} on purchase of goods exceeding {th_amt} from a resident seller. "
                        f"Deduction shall be made at the earlier of credit or payment. "
                        f"Provided that where the seller fails to furnish PAN, tax shall be deducted at {penal_rate}."
                    )
                    labeled_spans = [
                        (f"{to_prefix} exceeds {to_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("resident seller", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                        (f"{penal_rate}", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"),
                    ]
                    relations = [
                        (0, 1, "REL_PRECONDITION_TO_ENTITY"),
                        (2, 1, "REL_RATE_TO_THRESHOLD"),
                        (4, 1, "REL_DEADLINE_TO_TRIGGER"),
                        (5, 2, "REL_OVERRIDE_TO_GUARD")
                    ]
                elif formulation == 1:
                    body = (
                        f"Notwithstanding anything contained in this Act, any person being a buyer who is responsible for paying any sum to any resident "
                        f"for purchase of any goods of the value or aggregate of such value exceeding {th_amt} in any previous year, "
                        f"shall, at the time of credit of such sum to the account of the seller or at the time of payment thereof by any mode, whichever is earlier, "
                        f"deduct an amount equal to {goods_rate} of such excess sum as income-tax."
                    )
                    labeled_spans = [
                        ("Notwithstanding anything contained in this Act", "ROLE_PROHIBITION_GATE"),
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        ("at the time of credit of such sum to the account of the seller or at the time of payment thereof by any mode, whichever is earlier", "ROLE_TRIGGER_EARLIER_EVENT"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                    ]
                    relations = [(3, 1, "REL_RATE_TO_THRESHOLD"), (2, 1, "REL_DEADLINE_TO_TRIGGER")]
                elif formulation == 2:
                    body = (
                        f"Where the purchase of goods from a resident seller exceeding {th_amt} occurs during any financial year, "
                        f"any buyer whose {to_prefix} exceeds {to_amt} shall deduct tax at {goods_rate} at the time of credit of such sum or at the time of payment whichever is earlier. "
                        f"If the seller fails to furnish PAN, deduction shall be at {penal_rate}."
                    )
                    labeled_spans = [
                        (f"{to_prefix} exceeds {to_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("time of credit of such sum or at the time of payment whichever is earlier", "ROLE_TRIGGER_EARLIER_EVENT"),
                        (f"{penal_rate}", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"),
                    ]
                    relations = [
                        (0, 1, "REL_PRECONDITION_TO_ENTITY"),
                        (2, 1, "REL_RATE_TO_THRESHOLD"),
                        (3, 1, "REL_DEADLINE_TO_TRIGGER"),
                        (4, 2, "REL_OVERRIDE_TO_GUARD")
                    ]
                else:
                    body = (
                        f"Provided that where a resident seller fails to furnish PAN, tax shall be deducted at {penal_rate}. "
                        f"In all other cases, any buyer whose {to_prefix} exceeds {to_amt} shall deduct tax at {goods_rate} "
                        f"on purchase of goods exceeding {th_amt} at earlier of credit or payment."
                    )
                    labeled_spans = [
                        (f"{penal_rate}", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"),
                        (f"{to_prefix} exceeds {to_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [
                        (0, 2, "REL_OVERRIDE_TO_GUARD"),
                        (1, 3, "REL_PRECONDITION_TO_ENTITY"),
                        (2, 3, "REL_RATE_TO_THRESHOLD"),
                        (4, 3, "REL_DEADLINE_TO_TRIGGER")
                    ]
                if random.random() < 0.40:
                    cut_form = random.choice(["act2025_wef", "with_effect", "effective_parens"])
                    if cut_form == "act2025_wef":
                        body = body + f" Under the Income-tax Act, 2025 (w.e.f. {cut_date}), this provision continues under Section 393(1), Table 1, Sl. No. 8(ii)."
                        labeled_spans.append((f"{cut_date}", "ROLE_TEMPORAL_CUTOVER_DATE"))
                        labeled_spans.append(("Section 393(1), Table 1, Sl. No. 8(ii)", "ROLE_IGNORE"))
                    elif cut_form == "with_effect":
                        body = body[:-1] + f" with effect from {cut_date}."
                        labeled_spans.append((f"{cut_date}", "ROLE_TEMPORAL_CUTOVER_DATE"))
                    else:
                        body = body + f" Applicable under successor statute (effective {cut_date})."
                        labeled_spans.append((f"{cut_date}", "ROLE_TEMPORAL_CUTOVER_DATE"))

            # -----------------------------------------------------------------
            # 1B. DIRECT TAX: CONCISE SINGLE-CLAUSE EXCESS WITHHOLDING (194Q / 393)
            # -----------------------------------------------------------------
            elif stype == "194Q_CONCISE_STYLE":
                rate_expr = goods_rate if random.random() < 0.6 else f"tax at {goods_rate}"
                use_carveout_206c = (formulation in [1, 3] and random.random() < 0.5)
                if use_carveout_206c:
                    body = (
                        f"Provided that tax is deducted at {goods_rate} on consideration exceeding {th_amt} "
                        f"for purchase of goods by a buyer, no collection under section 206C shall apply."
                    )
                    labeled_spans = [
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        ("no collection under section 206C shall apply", "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [(0, 1, "REL_RATE_TO_THRESHOLD"), (2, 0, "REL_EXCEPTION_TO_PROHIBITION")]
                elif formulation == 0:
                    body = (
                        f"Any person, being a buyer who is responsible for paying any sum for purchase of goods "
                        f"exceeding {th_amt} in any financial year shall deduct {rate_expr} on the excess sum."
                    )
                    labeled_spans = [
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                    ]
                    relations = [(1, 0, "REL_RATE_TO_THRESHOLD")]
                elif formulation == 1:
                    body = (
                        f"Where a buyer pays consideration to a resident seller for purchase of goods in any financial year, "
                        f"income-tax at {goods_rate} shall be deducted on the sum exceeding {th_amt} "
                        f"at the earlier of credit or payment."
                    )
                    labeled_spans = [
                        ("resident seller", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [(1, 2, "REL_RATE_TO_THRESHOLD"), (3, 2, "REL_DEADLINE_TO_TRIGGER")]
                else:
                    body = (
                        f"On the sum exceeding {th_amt} for purchase of goods in any financial year, "
                        f"any person being a buyer shall deduct {rate_expr} at the earlier of credit or payment."
                    )
                    labeled_spans = [
                        (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [(1, 0, "REL_RATE_TO_THRESHOLD"), (2, 0, "REL_DEADLINE_TO_TRIGGER")]
                if not use_carveout_206c and random.random() < 0.40:
                    cut_form = random.choice(["with_effect", "wef_parens", "effective_parens"])
                    if cut_form == "with_effect":
                        body = body[:-1] + f" with effect from {cut_date}."
                    elif cut_form == "wef_parens":
                        body = f"Under Section 393(1), Table 1, Sl. No. 8(ii) of the Income-tax Act, 2025 (w.e.f. {cut_date}), " + body[0].lower() + body[1:]
                        labeled_spans.append(("Section 393(1), Table 1, Sl. No. 8(ii)", "ROLE_IGNORE"))
                    else:
                        body = f"Under Section 393(1), Table 1, Sl. No. 8(ii) of the Income-tax Act, 2025 (effective {cut_date}), " + body[0].lower() + body[1:]
                        labeled_spans.append(("Section 393(1), Table 1, Sl. No. 8(ii)", "ROLE_IGNORE"))
                    labeled_spans.append((f"{cut_date}", "ROLE_TEMPORAL_CUTOVER_DATE"))

            # -----------------------------------------------------------------
            # 2. DIRECT TAX: 206C(1H)
            # -----------------------------------------------------------------
            elif stype == "206C1H_STYLE":
                if formulation in [0, 1]:
                    body = (
                        f"A seller whose turnover exceeds {to_amt} in the preceding year shall collect tax at {goods_rate} "
                        f"on consideration exceeding {th_amt} received from a buyer at time of receipt. "
                        f"Provided that where the buyer fails to furnish PAN, collection shall be at {penal_rate}. "
                        f"This provision shall stand omitted with effect from the {sun_date}."
                    )
                else:
                    body = (
                        f"This provision stands omitted with effect from the {sun_date}. "
                        f"Prior to such omission, every seller whose turnover exceeds {to_amt} shall collect tax at {goods_rate} "
                        f"on consideration exceeding {th_amt} at time of receipt, or at {penal_rate} where the buyer fails to furnish PAN."
                    )
                labeled_spans = [
                    (f"turnover exceeds {to_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                    (f"exceeding {th_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    (f"{goods_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                    ("time of receipt", "ROLE_TRIGGER_PAYMENT_ONLY"),
                    (f"{penal_rate}", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"),
                    (f"{sun_date}", "ROLE_TEMPORAL_SUNSET_DATE"),
                ]
                relations = [(0, 1, "REL_PRECONDITION_TO_ENTITY"), (2, 1, "REL_RATE_TO_THRESHOLD"), (5, 0, "REL_SUNSET_TO_SECTION")]

            # -----------------------------------------------------------------
            # 3. DIRECT TAX: 194C
            # -----------------------------------------------------------------
            elif stype == "194C_STYLE":
                c_single = random.choice(["30,000 rupees", "₹30,000", "thirty thousand rupees"])
                c_agg = random.choice(["₹1,00,000", "one lakh rupees"])
                if formulation in [0, 1]:
                    body = (
                        f"Any person paying any sum to a resident contractor shall deduct tax at {std_rate} "
                        f"where any single invoice exceeds {c_single} or aggregate sums exceed {c_agg} in the financial year. "
                        f"Deduction shall be at earlier of credit or payment. "
                        f"In the absence of a valid PAN, tax shall be deducted at 20%."
                    )
                    labeled_spans = [
                        (f"exceeds {c_single}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"exceed {c_agg}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"{std_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                        ("20%", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"),
                    ]
                    relations = [
                        (2, 0, "REL_RATE_TO_THRESHOLD"),
                        (2, 1, "REL_RATE_TO_THRESHOLD"),
                        (0, 1, "REL_LOGICAL_OR"),
                        (3, 0, "REL_DEADLINE_TO_TRIGGER"),
                        (3, 1, "REL_DEADLINE_TO_TRIGGER"),
                    ]
                else:
                    body = (
                        f"Provided further that where the aggregate of the amounts of such sums credited or paid during the financial year exceeds {c_agg}, "
                        f"or any single invoice exceeds {c_single}, "
                        f"the person responsible shall deduct income-tax at {std_rate} at earlier of credit or payment."
                    )
                    labeled_spans = [
                        (f"exceeds {c_agg}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"exceeds {c_single}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"{std_rate}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [
                        (2, 0, "REL_RATE_TO_THRESHOLD"),
                        (2, 1, "REL_RATE_TO_THRESHOLD"),
                        (0, 1, "REL_LOGICAL_OR"),
                        (3, 0, "REL_DEADLINE_TO_TRIGGER"),
                        (3, 1, "REL_DEADLINE_TO_TRIGGER"),
                    ]

            # -----------------------------------------------------------------
            # 4. DIRECT TAX: 194J
            # -----------------------------------------------------------------
            elif stype == "194J_STYLE":
                body = (
                    f"Any person responsible for paying to a resident professional or technical consultant "
                    f"shall deduct tax at 10% on professional fees exceeding {cf_amt} in a financial year, "
                    f"at earlier of credit or payment."
                )
                labeled_spans = [
                    ("10%", "ROLE_STANDARD_STATUTORY_RATE"),
                    (f"exceeding {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                ]
                relations = [(0, 1, "REL_RATE_TO_THRESHOLD")]

            # -----------------------------------------------------------------
            # 5. DIRECT TAX: 194H
            # -----------------------------------------------------------------
            elif stype == "194H_STYLE":
                body = (
                    f"Any person responsible for paying to a resident commission or brokerage "
                    f"shall deduct income tax at the rate of 2% where the amount exceeds {cf_amt} in a financial year. "
                    f"Deduction shall be made at earlier of credit or payment."
                )
                labeled_spans = [
                    ("2%", "ROLE_STANDARD_STATUTORY_RATE"),
                    (f"exceeds {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                ]
                relations = [(0, 1, "REL_RATE_TO_THRESHOLD")]

            # -----------------------------------------------------------------
            # 6. DIRECT TAX: 194R
            # -----------------------------------------------------------------
            elif stype == "194R_STYLE":
                if formulation in [0, 1]:
                    body = (
                        f"Any person providing any benefit or perquisite to a resident arising from business "
                        f"shall deduct tax at 10% where the aggregate value exceeds {cf_amt} in a financial year."
                    )
                    labeled_spans = [
                        ("10%", "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"exceeds {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [(0, 1, "REL_RATE_TO_THRESHOLD")]
                elif formulation == 2:
                    body = (
                        f"Where the aggregate value of any benefit or perquisite arising from business exceeds {cf_amt} in a financial year, "
                        f"any person providing such benefit wholly in kind to a resident shall deduct tax at 10%."
                    )
                    labeled_spans = [
                        (f"exceeds {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("wholly in kind", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("10%", "ROLE_STANDARD_STATUTORY_RATE"),
                    ]
                    relations = [(2, 0, "REL_RATE_TO_THRESHOLD")]
                else:
                    body = (
                        f"Provided that where the aggregate value does not exceed {cf_amt} in a financial year no deduction shall be made, "
                        f"otherwise any person providing any benefit or perquisite wholly in kind to a resident shall deduct tax at 10%."
                    )
                    labeled_spans = [
                        (f"does not exceed {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("no deduction shall be made", "ROLE_EXCEPTION_CARVEOUT"),
                        ("wholly in kind", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("10%", "ROLE_STANDARD_STATUTORY_RATE"),
                    ]
                    relations = [(3, 0, "REL_RATE_TO_THRESHOLD"), (1, 3, "REL_EXCEPTION_TO_PROHIBITION")]

            # -----------------------------------------------------------------
            # 7. DIRECT TAX: 194I
            # -----------------------------------------------------------------
            elif stype == "194I_STYLE":
                if formulation == 0:
                    std_rate_i = random.choice(["10%", "2%", "10 per cent", "2 per cent"])
                    body = (
                        f"Any person paying rent to a resident for land or building shall deduct tax at {std_rate_i} "
                        f"where the rent exceeds {cf_amt} during the financial year, at earlier of credit or payment."
                    )
                    labeled_spans = [
                        (f"{std_rate_i}", "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"exceeds {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("earlier of credit or payment", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [(0, 1, "REL_RATE_TO_THRESHOLD"), (2, 1, "REL_DEADLINE_TO_TRIGGER")]
                elif formulation == 1:
                    rate_ib = random.choice(["5%", "5 per cent"])
                    rent_m_tr = random.choice(["₹50,000 per month", "fifty thousand rupees per month"])
                    body = (
                        f"Rent exceeding {rent_m_tr} payable to a resident by an individual or Hindu undivided family "
                        f"shall suffer income-tax deduction at {rate_ib} at the time of credit or payment whichever is earlier."
                    )
                    labeled_spans = [
                        (f"exceeding {rent_m_tr}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("resident", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        (f"{rate_ib}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("time of credit or payment whichever is earlier", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [(2, 0, "REL_RATE_TO_THRESHOLD"), (3, 0, "REL_DEADLINE_TO_TRIGGER")]
                else:
                    rent_m = "fifty thousand rupees per month" if formulation % 2 == 0 else "₹50,000 per month"
                    rate_ib = "5%" if formulation % 2 == 0 else "5 per cent"
                    body = (
                        f"Any individual or Hindu undivided family responsible for paying to a resident any rent "
                        f"exceeding {rent_m} shall deduct tax at {rate_ib} at the time of credit or payment whichever is earlier."
                    )
                    labeled_spans = [
                        (f"exceeding {rent_m}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"{rate_ib}", "ROLE_STANDARD_STATUTORY_RATE"),
                        ("time of credit or payment whichever is earlier", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ]
                    relations = [(1, 0, "REL_RATE_TO_THRESHOLD"), (2, 0, "REL_DEADLINE_TO_TRIGGER")]

            # -----------------------------------------------------------------
            # 7B. STATUTORY FLAT-RATE COLLECTION / MANDATE RULES (NO THRESHOLD)
            # -----------------------------------------------------------------
            elif stype == "STATUTORY_FLAT_COLLECTION_STYLE":
                flat_rate, sub_rate = random.choice([
                    ("1%", "0.5%"),
                    ("2%", "1%"),
                    ("3%", "1.5%"),
                    ("4%", "2%"),
                    ("1 per cent", "0.5 per cent"),
                    ("2 per cent", "1 per cent"),
                ])
                c_head, s_head, i_head = random.choice([
                    ("CGST", "SGST", "IGST"),
                    ("Central Tax", "State Tax", "Integrated Tax"),
                    ("federal levy", "provincial levy", "combined levy"),
                    ("primary component", "regional component", "interstate component"),
                ])
                mandate_verb = random.choice(["Mandates", "Requires", "Directs", "Obligates", "Authorises"])
                collect_phrase = random.choice([
                    "collect tax at source",
                    "deduct tax at source",
                    "collect statutory levy at source",
                    "withhold indirect tax at source",
                ])
                actor_phrase = random.choice([
                    "electronic commerce operators",
                    "digital marketplace intermediaries",
                    "designated settlement operators",
                    "authorised remittance dealers",
                    "clearing house facilitators",
                ])
                obj_scope = random.choice([
                    "net value of taxable supplies",
                    "gross value of qualifying transactions",
                    "net consideration received through the platform",
                    "eligible outbound remittance transfers",
                ])
                precond_clause = random.choice([
                    "the consideration is collected by the facilitator",
                    "the settlement proceeds pass through the platform",
                    "the transfer is routed via an authorised portal",
                    "payment is settled by the intermediary",
                ])
                with_breakdown = random.random() < 0.7
                if formulation == 0:
                    if with_breakdown:
                        body = (
                            f"{mandate_verb} {actor_phrase} to {collect_phrase} at {flat_rate} "
                            f"({sub_rate} {c_head} + {sub_rate} {s_head} or {flat_rate} {i_head}) "
                            f"on the {obj_scope} where {precond_clause}."
                        )
                        labeled_spans = [
                            (actor_phrase, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                            (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (sub_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (sub_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (precond_clause, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [(5, 0, "REL_PRECONDITION_TO_ENTITY")]
                    else:
                        body = (
                            f"{mandate_verb} {actor_phrase} to {collect_phrase} at {flat_rate} "
                            f"on the {obj_scope} where {precond_clause}."
                        )
                        labeled_spans = [
                            (actor_phrase, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                            (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (precond_clause, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [(2, 0, "REL_PRECONDITION_TO_ENTITY")]
                elif formulation == 1:
                    body = (
                        f"In covered transactions, {actor_phrase} shall {collect_phrase} at {flat_rate} "
                        f"on the {obj_scope} whenever {precond_clause}."
                    )
                    labeled_spans = [
                        (actor_phrase, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        (precond_clause, "ROLE_AGENT_STATE_PRECONDITION"),
                    ]
                    relations = [(2, 0, "REL_PRECONDITION_TO_ENTITY")]
                elif formulation == 2:
                    if with_breakdown:
                        body = (
                            f"Where {precond_clause}, this rule {mandate_verb.lower()} {actor_phrase} to {collect_phrase} "
                            f"at {flat_rate} ({sub_rate} {c_head} + {sub_rate} {s_head} or {flat_rate} {i_head}) "
                            f"on the {obj_scope}."
                        )
                        labeled_spans = [
                            (precond_clause, "ROLE_AGENT_STATE_PRECONDITION"),
                            (actor_phrase, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                            (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (sub_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (sub_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                            (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        ]
                        relations = [(0, 1, "REL_PRECONDITION_TO_ENTITY")]
                    else:
                        body = (
                            f"Where {precond_clause}, this rule {mandate_verb.lower()} {actor_phrase} to {collect_phrase} "
                            f"at {flat_rate} on the {obj_scope}."
                        )
                        labeled_spans = [
                            (precond_clause, "ROLE_AGENT_STATE_PRECONDITION"),
                            (actor_phrase, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                            (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        ]
                        relations = [(0, 1, "REL_PRECONDITION_TO_ENTITY")]
                else:
                    body = (
                        f"On the {obj_scope}, {actor_phrase} are required to {collect_phrase} at {flat_rate} "
                        f"whenever {precond_clause}."
                    )
                    labeled_spans = [
                        (actor_phrase, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        (flat_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        (precond_clause, "ROLE_AGENT_STATE_PRECONDITION"),
                    ]
                    relations = [(2, 0, "REL_PRECONDITION_TO_ENTITY")]

            # -----------------------------------------------------------------
            # 8. DIRECT TAX: CORPORATE TAX (115BAA)
            # -----------------------------------------------------------------
            elif stype == "CORPORATE_TAX_STYLE":
                cut_style = random.choice(["wef_parens", "wef_full", "wef_bare", "effective_parens"])
                if formulation in [0, 1]:
                    if cut_style == "wef_parens":
                        body = (
                            f"Under the updated statutory framework (w.e.f. {cut_date}), "
                            f"the income-tax payable by a domestic company for any previous year shall be computed at the rate of 22%."
                        )
                    elif cut_style == "effective_parens":
                        body = (
                            f"Under the updated statutory framework (effective {cut_date}), "
                            f"the income-tax payable by a domestic company for any previous year shall be computed at the rate of 22%."
                        )
                    elif cut_style == "wef_bare":
                        body = (
                            f"The income-tax payable in respect of total income of a domestic company for any previous year "
                            f"shall be computed at the rate of 22% with effect from {cut_date}."
                        )
                    else:
                        body = (
                            f"The income-tax payable in respect of total income of a domestic company for any previous year "
                            f"shall be computed at the rate of 22% with effect from the {cut_date}."
                        )
                else:
                    if cut_style == "wef_parens":
                        body = (
                            f"At the rate of 22%, the income-tax payable by a domestic company for any previous year "
                            f"shall be computed under the updated statutory framework (w.e.f. {cut_date})."
                        )
                    elif cut_style == "effective_parens":
                        body = (
                            f"At the rate of 22%, the income-tax payable by a domestic company for any previous year "
                            f"shall be computed under the updated statutory framework (effective {cut_date})."
                        )
                    elif cut_style == "wef_bare":
                        body = (
                            f"With effect from {cut_date}, the income-tax payable in respect of total income of a domestic company "
                            f"shall be computed at the rate of 22%."
                        )
                    else:
                        body = (
                            f"With effect from the {cut_date}, the income-tax payable in respect of total income of a domestic company "
                            f"shall be computed at the rate of 22%."
                        )
                labeled_spans = [
                    ("domestic company", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    ("22%", "ROLE_STANDARD_STATUTORY_RATE"),
                    (f"{cut_date}", "ROLE_TEMPORAL_CUTOVER_DATE"),
                ]
                relations = [(1, 0, "REL_PRECONDITION_TO_ENTITY")]

            # -----------------------------------------------------------------
            # 9. DIRECT TAX: SUNSET
            # -----------------------------------------------------------------
            elif stype == "SUNSET_STYLE":
                if formulation in [0, 1]:
                    body = (
                        f"Provided further that the provisions of this sub-section shall not apply to any transaction on or after {sun_date}, "
                        f"and the statutory provision shall stand omitted with effect from {sun_date}."
                    )
                    labeled_spans = [
                        (f"{sun_date}", "ROLE_TEMPORAL_SUNSET_DATE"),
                        (f"{sun_date}", "ROLE_TEMPORAL_SUNSET_DATE"),
                    ]
                else:
                    body = (
                        f"This statutory provision is omitted and ceases to apply to any sum credited or paid on or after {sun_date}."
                    )
                    labeled_spans = [
                        (f"{sun_date}", "ROLE_TEMPORAL_SUNSET_DATE"),
                    ]
                relations = []

            # -----------------------------------------------------------------
            # 10. COMMERCIAL & MSME: SECTION 15 (45-DAY DEADLINE)
            # -----------------------------------------------------------------
            elif stype == "MSME_DEADLINE_STYLE":
                if formulation in [0, 1]:
                    body = (
                        f"In no case shall the period agreed upon between the buyer and supplier exceed {dl_str} "
                        f"from the day of acceptance. Where there is no agreement, payment shall be made before the appointed day. "
                        f"Any buyer failing to make payment within the deadline shall be liable to pay compound interest at three times of the bank rate notified by the Reserve Bank of India."
                    )
                else:
                    body = (
                        f"Payment to micro and small enterprises shall be made before the appointed day, "
                        f"or within the period agreed in writing not to exceed {dl_str} from the day of acceptance. "
                        f"Default in payment triggers compound interest at three times of the bank rate."
                    )
                dl_match = f"exceed {dl_str}" if f"exceed {dl_str}" in body else dl_str
                labeled_spans = [
                    (dl_match, "ROLE_DEADLINE_WINDOW"),
                    ("day of acceptance", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ("appointed day", "ROLE_DEADLINE_WINDOW"),
                    ("compound interest at three times of the bank rate", "ROLE_STATUTORY_SANCTION"),
                ]
                relations = [
                    (0, 1, "REL_DEADLINE_TO_TRIGGER"),
                    (2, 1, "REL_DEADLINE_TO_TRIGGER"),
                    (3, 0, "REL_SANCTION_TO_VIOLATION")
                ]

            # -----------------------------------------------------------------
            # 11. GST: SECTION 17(5) BLOCKED CREDIT
            # -----------------------------------------------------------------
            elif stype == "GST_BLOCKED_ITC_STYLE":
                seats = random.choice(["thirteen", "13", "eight", "8", "five", "5"])
                if formulation in [0, 1]:
                    body = (
                        f"Input tax credit shall not be available in respect of motor vehicles for transportation of persons "
                        f"having approved seating capacity of not more than {seats} persons, "
                        f"except when they are used for making taxable supplies of such vehicles, transportation of passengers, or imparting training."
                    )
                else:
                    body = (
                        f"Notwithstanding Section 16, no credit shall be allowed for motor vehicles with seating capacity "
                        f"of not more than {seats} persons. Provided that credit shall not be available unless vehicles are used for "
                        f"transportation of passengers or except when they are used for making taxable supplies."
                    )
                prohibit_text = "shall not be available" if "shall not be available" in body else "no credit shall be allowed"
                labeled_spans = [
                    (prohibit_text, "ROLE_PROHIBITION_GATE"),
                    (f"not more than {seats} persons", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("transportation of passengers", "ROLE_EXCEPTION_CARVEOUT"),
                    ("taxable supplies", "ROLE_EXCEPTION_CARVEOUT"),
                ]
                relations = [
                    (2, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (3, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (1, 0, "REL_THRESHOLD_TO_TRANSACTION")
                ]

            # -----------------------------------------------------------------
            # 12. GST: SECTION 8 COMPOSITE SUPPLY & EVIDENCE / SUBSTANTIATION
            # -----------------------------------------------------------------
            elif stype == "SUBJECTIVE_EVIDENCE_STYLE":
                ev_doc2 = random.choice([d for d in evidence_docs if d != ev_doc] or evidence_docs)
                if formulation == 0:
                    body = (
                        f"A composite supply comprising two or more supplies naturally bundled in the ordinary course of business "
                        f"shall be treated as a supply of such principal supply. Every registered person shall maintain a {ev_doc} "
                        f"and documentary justification as prescribed."
                    )
                    labeled_spans = [
                        ("naturally bundled in the ordinary course of business", "ROLE_SUBJECTIVE_CRITERION"),
                        (f"{ev_doc}", "ROLE_REQUIRED_EVIDENCE"),
                    ]
                    relations = [(1, 0, "REL_EVIDENCE_TO_SUBJECTIVE")]
                elif formulation == 1:
                    claim_item = random.choice(["a foreign tax credit", "an exemption benefit", "a statutory deduction"])
                    filing_noun = random.choice(["returns", "eligibility filings"])
                    prelim_adj = random.choice(["Summary", "Preliminary"])
                    claimant_phrase = random.choice(["A taxpayer", "A claimant"])
                    body = (
                        f"{claimant_phrase} claiming {claim_item} based on items reported on {ev_doc} "
                        f"must substantiate the claim with {ev_doc2}. {prelim_adj} entries alone do not establish legal compliance, "
                        f"and prescribed schedules are statutory prerequisites before relief may be claimed on {filing_noun}."
                    )
                    labeled_spans = [
                        (f"claiming {claim_item}", "ROLE_AGENT_STATE_PRECONDITION"),
                        (f"{ev_doc}", "ROLE_REQUIRED_EVIDENCE"),
                        ("must substantiate the claim", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (f"{ev_doc2}", "ROLE_REQUIRED_EVIDENCE"),
                        ("statutory prerequisites before relief may be claimed", "ROLE_SUBJECTIVE_CRITERION"),
                    ]
                    relations = [
                        (3, 2, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (1, 2, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (2, 0, "REL_ACTION_TO_PRECONDITION"),
                        (4, 2, "REL_CONDITION_TO_GATE"),
                    ]
                elif formulation == 2:
                    body = (
                        f"Before any tax credit or exemption may be claimed on returns, a claimant must substantiate eligibility with {ev_doc} and {ev_doc2}. "
                        f"Preliminary reporting items do not establish legal compliance without statutory prerequisites."
                    )
                    labeled_spans = [
                        ("statutory prerequisites", "ROLE_SUBJECTIVE_CRITERION"),
                        ("must substantiate eligibility", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (f"{ev_doc}", "ROLE_REQUIRED_EVIDENCE"),
                        (f"{ev_doc2}", "ROLE_REQUIRED_EVIDENCE"),
                    ]
                    relations = [
                        (2, 1, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (3, 1, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (0, 1, "REL_CONDITION_TO_GATE"),
                    ]
                else:
                    body = (
                        f"Provided that {ev_doc} and {ev_doc2} are statutory prerequisites before relief may be claimed, "
                        f"a claimant must substantiate the claim with prescribed schedules."
                    )
                    labeled_spans = [
                        (f"{ev_doc}", "ROLE_REQUIRED_EVIDENCE"),
                        (f"{ev_doc2}", "ROLE_REQUIRED_EVIDENCE"),
                        ("statutory prerequisites before relief may be claimed", "ROLE_SUBJECTIVE_CRITERION"),
                        ("must substantiate the claim", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ]
                    relations = [
                        (0, 3, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (1, 3, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (2, 3, "REL_CONDITION_TO_GATE"),
                    ]

            # -----------------------------------------------------------------
            # 13. CORPORATE LAW: COMPANIES ACT SEC 164 (DIRECTOR DISQUALIFICATION)
            # -----------------------------------------------------------------
            elif stype == "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE":
                if formulation == 0:
                    body = (
                        f"No person shall be eligible for appointment as a director of a company "
                        f"if he has been convicted of an offence involving moral turpitude or if the company has failed to file financial statements for continuous three years. "
                        f"Every proposed director shall furnish a Form DIR-8 prior to appointment. "
                        f"Any disqualification shall continue for a period of five years."
                    )
                    labeled_spans = [
                        ("shall be eligible for appointment", "ROLE_PROHIBITION_GATE"),
                        ("convicted of an offence involving moral turpitude", "ROLE_SUBJECTIVE_CRITERION"),
                        ("failed to file financial statements for continuous three years", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("Form DIR-8", "ROLE_REQUIRED_EVIDENCE"),
                        ("period of five years", "ROLE_STATUTORY_SANCTION"),
                    ]
                    relations = [
                        (1, 0, "REL_CONDITION_TO_GATE"),
                        (2, 0, "REL_CONDITION_TO_GATE"),
                        (1, 2, "REL_LOGICAL_OR"),
                        (3, 0, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (4, 0, "REL_SANCTION_TO_VIOLATION")
                    ]
                elif formulation == 1:
                    body = (
                        f"A person is disqualified from appointment as director where such person is an undischarged insolvent, "
                        f"or convicted of an offence involving moral turpitude, or has failed to file financial statements for continuous three years. "
                        f"Disqualification shall continue for a period of five years unless cleared by filing Form DIR-8."
                    )
                    labeled_spans = [
                        ("disqualified from appointment", "ROLE_PROHIBITION_GATE"),
                        ("undischarged insolvent", "ROLE_SUBJECTIVE_CRITERION"),
                        ("convicted of an offence involving moral turpitude", "ROLE_SUBJECTIVE_CRITERION"),
                        ("failed to file financial statements for continuous three years", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("Form DIR-8", "ROLE_REQUIRED_EVIDENCE"),
                        ("period of five years", "ROLE_STATUTORY_SANCTION"),
                    ]
                    relations = [
                        (1, 0, "REL_CONDITION_TO_GATE"),
                        (2, 0, "REL_CONDITION_TO_GATE"),
                        (3, 0, "REL_CONDITION_TO_GATE"),
                        (1, 2, "REL_LOGICAL_OR"),
                        (2, 3, "REL_LOGICAL_OR"),
                        (4, 0, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (5, 0, "REL_SANCTION_TO_VIOLATION")
                    ]
                elif formulation == 2:
                    body = (
                        f"Where a company has failed to file financial statements for continuous three years, "
                        f"no person shall be eligible for appointment as a director unless he submits Form DIR-8. "
                        f"Such disqualification shall continue for a period of five years."
                    )
                    labeled_spans = [
                        ("failed to file financial statements for continuous three years", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall be eligible for appointment", "ROLE_PROHIBITION_GATE"),
                        ("Form DIR-8", "ROLE_REQUIRED_EVIDENCE"),
                        ("period of five years", "ROLE_STATUTORY_SANCTION"),
                    ]
                    relations = [
                        (0, 1, "REL_CONDITION_TO_GATE"),
                        (2, 1, "REL_EVIDENCE_TO_SUBJECTIVE"),
                        (3, 1, "REL_SANCTION_TO_VIOLATION")
                    ]
                else: # formulation == 3
                    body = (
                        f"Provided that disqualification shall continue for a period of five years from the date of conviction. "
                        f"No person shall be eligible for appointment as a director if convicted of an offence involving moral turpitude. "
                        f"Every director shall submit Form DIR-8."
                    )
                    labeled_spans = [
                        ("period of five years", "ROLE_STATUTORY_SANCTION"),
                        ("shall be eligible for appointment", "ROLE_PROHIBITION_GATE"),
                        ("convicted of an offence involving moral turpitude", "ROLE_SUBJECTIVE_CRITERION"),
                        ("Form DIR-8", "ROLE_REQUIRED_EVIDENCE"),
                    ]
                    relations = [
                        (0, 1, "REL_SANCTION_TO_VIOLATION"),
                        (2, 1, "REL_CONDITION_TO_GATE"),
                        (3, 1, "REL_EVIDENCE_TO_SUBJECTIVE")
                    ]

            # -----------------------------------------------------------------
            # 14. CORPORATE LAW: COMPANIES ACT SEC 188 (RELATED PARTY TRANSACTIONS)
            # -----------------------------------------------------------------
            elif stype == "COMPANIES_RELATED_PARTY_STYLE":
                rpt_th = th_amt if random.random() < 0.7 else "10 per cent of the turnover"
                if formulation in [0, 1]:
                    body = (
                        f"No company shall enter into any contract or arrangement with a related party "
                        f"exceeding {rpt_th} without prior approval by special resolution of the shareholders, "
                        f"except in ordinary course of business on an arms length basis. "
                        f"Every transaction requires consent of the Board of Directors."
                    )
                else:
                    body = (
                        f"Except in ordinary course of business on an arms length basis, no company shall enter into any contract "
                        f"with a related party exceeding {rpt_th} without prior approval by special resolution of the shareholders. "
                        f"Consent of the Board of Directors must be obtained."
                    )
                labeled_spans = [
                    ("shall enter into any contract", "ROLE_PROHIBITION_GATE"),
                    ("related party", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    (f"exceeding {rpt_th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    ("prior approval by special resolution of the shareholders", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("except in ordinary course of business on an arms length basis", "ROLE_EXCEPTION_CARVEOUT"),
                    ("consent of the Board of Directors", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                ]
                relations = [
                    (1, 0, "REL_PRECONDITION_TO_ENTITY"),
                    (2, 0, "REL_THRESHOLD_TO_TRANSACTION"),
                    (3, 2, "REL_ACTION_TO_PRECONDITION"),
                    (4, 0, "REL_EXCEPTION_TO_PROHIBITION")
                ]

            # -----------------------------------------------------------------
            # 15. CORPORATE LAW: COMPANIES ACT SEC 177 (AUDIT COMMITTEE)
            # -----------------------------------------------------------------
            elif stype == "COMPANIES_AUDIT_COMMITTEE_STYLE":
                cap_amt = random.choice(paidup_cap_gates)
                if formulation in [0, 1]:
                    body = (
                        f"Every listed public company and public company having paid-up share capital of {cap_amt} "
                        f"or turnover exceeds {to_amt} shall constitute an Audit Committee of the Board."
                    )
                else:
                    body = (
                        f"Where the paid-up share capital of {cap_amt} or turnover exceeds {to_amt} in a public company, "
                        f"the board shall constitute an Audit Committee of the Board as mandated."
                    )
                labeled_spans = [
                    ("public company", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    (f"paid-up share capital of {cap_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                    (f"turnover exceeds {to_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                    ("shall constitute an Audit Committee of the Board", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                ]
                relations = [
                    (1, 2, "REL_LOGICAL_OR"),
                    (3, 1, "REL_ACTION_TO_PRECONDITION")
                ]

            # -----------------------------------------------------------------
            # 16. CORPORATE LAW: COMPANIES ACT SEC 135 (CSR MANDATE)
            # -----------------------------------------------------------------
            elif stype == "COMPANIES_CSR_MANDATE_STYLE":
                nw_amt = random.choice(networth_gates)
                body = (
                    f"Every company having net worth of {nw_amt} or turnover exceeds {to_amt} "
                    f"shall constitute a Corporate Social Responsibility Committee and shall spend at least 2% of average net profits on CSR."
                )
                labeled_spans = [
                    (f"net worth of {nw_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                    (f"turnover exceeds {to_amt}", "ROLE_AGENT_STATE_PRECONDITION"),
                    ("shall constitute a Corporate Social Responsibility Committee", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("2%", "ROLE_STANDARD_STATUTORY_RATE"),
                ]
                relations = [
                    (0, 1, "REL_LOGICAL_OR"),
                    (2, 0, "REL_ACTION_TO_PRECONDITION"),
                    (3, 0, "REL_RATE_TO_THRESHOLD")
                ]

            # -----------------------------------------------------------------
            # 17. INSOLVENCY & BANKRUPTCY: SECTION 29A (RESOLUTION INELIGIBILITY)
            # -----------------------------------------------------------------
            elif stype == "IBC_RESOLUTION_INELIGIBILITY_STYLE":
                if formulation in [0, 1]:
                    body = (
                        f"A person shall not be eligible to submit a resolution plan if such person is an undischarged insolvent, "
                        f"or a wilful defaulter, or has been convicted of an offence. "
                        f"Every resolution applicant shall submit an affidavit under section 29A confirming eligibility."
                    )
                else:
                    body = (
                        f"Submission of an affidavit under section 29A is mandatory. A person shall not be eligible to submit a resolution plan "
                        f"if such person is a wilful defaulter or an undischarged insolvent."
                    )
                labeled_spans = [
                    ("shall not be eligible to submit a resolution plan", "ROLE_PROHIBITION_GATE"),
                    ("undischarged insolvent", "ROLE_SUBJECTIVE_CRITERION"),
                    ("wilful defaulter", "ROLE_SUBJECTIVE_CRITERION"),
                    ("affidavit under section 29A", "ROLE_REQUIRED_EVIDENCE"),
                ]
                relations = [
                    (1, 0, "REL_CONDITION_TO_GATE"),
                    (2, 0, "REL_CONDITION_TO_GATE"),
                    (1, 2, "REL_LOGICAL_OR"),
                    (3, 0, "REL_EVIDENCE_TO_SUBJECTIVE")
                ]

            # -----------------------------------------------------------------
            # 18. INSOLVENCY & BANKRUPTCY: SECTION 4 & 7 (DEFAULT THRESHOLD)
            # -----------------------------------------------------------------
            elif stype == "IBC_INSOLVENCY_TRIGGER_STYLE":
                body = (
                    f"A financial creditor may file an application for corporate insolvency resolution process "
                    f"where the default amount is not less than {cf_amt}. "
                    f"The application shall be accompanied by record of default with information utility."
                )
                labeled_spans = [
                    (f"not less than {cf_amt}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("record of default with information utility", "ROLE_REQUIRED_EVIDENCE"),
                    ("application for corporate insolvency resolution process", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                ]
                relations = [
                    (1, 2, "REL_EVIDENCE_TO_SUBJECTIVE"),
                    (2, 0, "REL_THRESHOLD_TO_TRANSACTION")
                ]

            # -----------------------------------------------------------------
            # 19. CIVIL PROCEDURE: SECTION 80 CPC (NOTICE WINDOW)
            # -----------------------------------------------------------------
            elif stype == "PROCEDURAL_NOTICE_WINDOW_STYLE":
                if formulation in [0, 1]:
                    body = (
                        f"No suit shall be instituted against the Government until the expiration of two months "
                        f"next after notice in writing has been delivered. "
                        f"A suit instituted without such notice shall be barred and plaint shall be rejected, "
                        f"except with leave of the court where urgent or immediate relief is sought."
                    )
                else:
                    body = (
                        f"Except with leave of the court where urgent or immediate relief is sought, "
                        f"no suit shall be instituted until expiration of two months next after notice in writing has been delivered. "
                        f"Non-compliance renders the plaint shall be rejected."
                    )
                two_mo_match = "expiration of two months" if "expiration of two months" in body else "two months"
                labeled_spans = [
                    ("shall be instituted", "ROLE_PROHIBITION_GATE"),
                    (two_mo_match, "ROLE_DEADLINE_WINDOW"),
                    ("notice in writing has been delivered", "ROLE_TRIGGER_EARLIER_EVENT"),
                    ("plaint shall be rejected", "ROLE_STATUTORY_SANCTION"),
                    ("urgent or immediate relief is sought", "ROLE_EXCEPTION_CARVEOUT"),
                ]
                relations = [
                    (1, 2, "REL_DEADLINE_TO_TRIGGER"),
                    (3, 0, "REL_SANCTION_TO_VIOLATION"),
                    (4, 0, "REL_EXCEPTION_TO_PROHIBITION")
                ]

            # -----------------------------------------------------------------
            # 20. COMMERCIAL COURTS: SECTION 12A (PRE-INSTITUTION MEDIATION)
            # -----------------------------------------------------------------
            elif stype == "COMMERCIAL_COURT_MEDIATION_STYLE":
                body = (
                    f"A suit which does not contemplate any urgent interim relief shall not be instituted "
                    f"unless the plaintiff exhausts the remedy of pre-institution mediation within three months. "
                    f"Any premature suit shall be dismissed as barred by law."
                )
                labeled_spans = [
                    ("shall not be instituted", "ROLE_PROHIBITION_GATE"),
                    ("urgent interim relief", "ROLE_EXCEPTION_CARVEOUT"),
                    ("three months", "ROLE_DEADLINE_WINDOW"),
                    ("suit shall be dismissed", "ROLE_STATUTORY_SANCTION"),
                ]
                relations = [
                    (1, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (2, 0, "REL_DEADLINE_TO_TRIGGER"),
                    (3, 0, "REL_SANCTION_TO_VIOLATION")
                ]

            # -----------------------------------------------------------------
            # 21. LABOR LAW: PAYMENT OF GRATUITY ACT (SECTION 4)
            # -----------------------------------------------------------------
            elif stype == "LABOR_GRATUITY_STYLE":
                prep = random.choice(["of", "for"])
                delim = random.choice([". Provided that", ": Provided that"])
                if formulation == 0:
                    body = (
                        "Gratuity shall be payable to an employee on the termination of his employment "
                        f"after he has rendered continuous service {prep} not less than five years{delim} "
                        "the completion of continuous service of five years shall not be necessary "
                        "where the termination of the employment of any employee is due to death or disablement."
                    )
                    ex_clause = "completion of continuous service of five years shall not be necessary"
                elif formulation == 1:
                    body = (
                        "Payment of gratuity is an affirmative obligation for every employer upon termination "
                        f"of an employee who has rendered continuous service {prep} not less than five years. "
                        "Continuous service of five years is not required where termination is due to death or disablement."
                    )
                    ex_clause = "Continuous service of five years is not required"
                elif formulation == 2:
                    body = (
                        f"Where an employee has rendered continuous service {prep} not less than five years, "
                        "gratuity shall be payable upon termination of his employment. "
                        "Provided that the completion of continuous service of five years shall not be necessary where termination is due to death or disablement."
                    )
                    ex_clause = "completion of continuous service of five years shall not be necessary"
                else: # formulation == 3
                    body = (
                        "Provided that the completion of continuous service of five years shall not be necessary where termination is due to death or disablement. "
                        "Gratuity shall be payable to an employee on termination of employment "
                        f"after he has rendered continuous service {prep} not less than five years."
                    )
                    ex_clause = "completion of continuous service of five years shall not be necessary"
                obligation_match = "Gratuity shall be payable" if "Gratuity shall be payable" in body else ("gratuity shall be payable" if "gratuity shall be payable" in body else "Payment of gratuity")
                labeled_spans = [
                    (obligation_match, "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("continuous service", "ROLE_AGENT_STATE_PRECONDITION"),
                    ("not less than five years", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    (ex_clause, "ROLE_EXCEPTION_CARVEOUT"),
                    ("due to death or disablement", "ROLE_EXCEPTION_CARVEOUT")
                ]
                relations = [
                    (1, 0, "REL_ACTION_TO_PRECONDITION"),
                    (2, 1, "REL_THRESHOLD_TO_TRANSACTION"),
                    (3, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (4, 0, "REL_EXCEPTION_TO_PROHIBITION")
                ]

            # -----------------------------------------------------------------
            # 22. SECURITIES LAW: SEBI PIT REGULATIONS (REGULATION 3)
            # -----------------------------------------------------------------
            elif stype == "SECURITIES_INSIDER_TRADING_STYLE":
                if formulation in [0, 1]:
                    body = (
                        "No insider shall communicate, provide, or allow access to any unpublished price sensitive information "
                        "relating to a company or securities listed or proposed to be listed, to any person including other insiders "
                        "except where such communication is in furtherance of legitimate purposes or discharge of legal obligations."
                    )
                else:
                    body = (
                        "Except where such communication is in furtherance of legitimate purposes or discharge of legal obligations, "
                        "no insider shall communicate, provide, or allow access to any unpublished price sensitive information "
                        "relating to a company or securities listed or proposed to be listed."
                    )
                gate_str = "No insider shall communicate, provide, or allow access" if "No insider shall communicate, provide, or allow access" in body else "no insider shall communicate, provide, or allow access"
                carve_str = "except where such communication is in furtherance of legitimate purposes or discharge of legal obligations" if "except where such communication is in furtherance of legitimate purposes or discharge of legal obligations" in body else "Except where such communication is in furtherance of legitimate purposes or discharge of legal obligations"
                labeled_spans = [
                    (gate_str, "ROLE_PROHIBITION_GATE"),
                    ("unpublished price sensitive information", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    (carve_str, "ROLE_EXCEPTION_CARVEOUT"),
                ]
                relations = [
                    (1, 0, "REL_PRECONDITION_TO_ENTITY"),
                    (2, 0, "REL_EXCEPTION_TO_PROHIBITION")
                ]

            # -----------------------------------------------------------------
            # 23. ENVIRONMENTAL LAW: AIR & WATER POLLUTION CONTROL ACT
            # -----------------------------------------------------------------
            elif stype == "ENVIRONMENT_COMPLIANCE_STYLE":
                if formulation in [0, 2]:
                    ppm_val = random.choice(["5 parts per million", "10 parts per million", "50 parts per million"])
                    fine_val = random.choice(["ten thousand rupees per day", "twenty thousand rupees per day", "₹10,000 per day"])
                    body = (
                        f"No industrial unit shall discharge trade effluent into any stream or well where the toxic contaminants exceed {ppm_val}, "
                        f"unless a prior consent order has been granted by the State Pollution Control Board. "
                        f"Contravention shall be punishable with a fine of {fine_val}."
                    )
                    labeled_spans = [
                        ("No industrial unit shall discharge", "ROLE_PROHIBITION_GATE"),
                        (f"exceed {ppm_val}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        ("unless a prior consent order has been granted by the State Pollution Control Board", "ROLE_EXCEPTION_CARVEOUT"),
                        (f"fine of {fine_val}", "ROLE_STATUTORY_SANCTION"),
                    ]
                    relations = [
                        (1, 0, "REL_CONDITION_TO_GATE"),
                        (2, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        (3, 0, "REL_SANCTION_TO_VIOLATION"),
                    ]
                else:
                    days_val = random.choice(["sixty days", "thirty days", "ninety days"])
                    fine_val = random.choice(["fifty thousand rupees per day", "twenty-five thousand rupees per day", "₹50,000 per day"])
                    body = (
                        f"Every industrial plant operating in an air pollution control area shall install an online continuous emission monitoring system. "
                        f"Where any industrial plant fails to install such monitoring system within {days_val}, "
                        f"the State Pollution Control Board shall order the immediate closure of the industry and impose a fine of {fine_val}."
                    )
                    install_match = "shall install an online continuous emission monitoring system"
                    fail_match = "fails to install such monitoring system"
                    labeled_spans = [
                        (install_match, "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (f"within {days_val}", "ROLE_DEADLINE_WINDOW"),
                        (fail_match, "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall order the immediate closure of the industry", "ROLE_STATUTORY_SANCTION"),
                        (f"fine of {fine_val}", "ROLE_STATUTORY_SANCTION"),
                    ]
                    relations = [
                        (1, 2, "REL_DEADLINE_TO_TRIGGER"),
                        (2, 0, "REL_ACTION_TO_PRECONDITION"),
                        (3, 2, "REL_SANCTION_TO_VIOLATION"),
                        (4, 3, "REL_LOGICAL_AND"),
                    ]

            # -----------------------------------------------------------------
            # 24. DATA PRIVACY LAW: DIGITAL PERSONAL DATA PROTECTION ACT
            # -----------------------------------------------------------------
            elif stype == "DATA_PROTECTION_STYLE":
                if formulation == 0:
                    body = (
                        "Every data fiduciary shall take reasonable security safeguards to prevent personal data breach. "
                        "Where a personal data breach occurs, the data fiduciary shall give notice to the Data Protection Board "
                        "within seventy-two hours from becoming aware of such breach."
                    )
                    labeled_spans = [
                        ("shall take reasonable security safeguards", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ("personal data breach occurs", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("within seventy-two hours", "ROLE_DEADLINE_WINDOW"),
                        ("shall give notice to the Data Protection Board", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ]
                    relations = [
                        (3, 1, "REL_ACTION_TO_PRECONDITION"),
                        (2, 3, "REL_DEADLINE_TO_TRIGGER"),
                        (3, 0, "REL_LOGICAL_AND"),
                    ]
                elif formulation == 1:
                    body = (
                        "In the case of a personal data breach, the data fiduciary shall notify the Board and affected data principals "
                        "without delay and not later than 72 hours after becoming aware of the breach, "
                        "unless the breach is unlikely to cause harm."
                    )
                    labeled_spans = [
                        ("case of a personal data breach", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall notify the Board and affected data principals", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ("not later than 72 hours", "ROLE_DEADLINE_WINDOW"),
                        ("unless the breach is unlikely to cause harm", "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [
                        (1, 0, "REL_ACTION_TO_PRECONDITION"),
                        (2, 1, "REL_DEADLINE_TO_TRIGGER"),
                        (3, 1, "REL_EXCEPTION_TO_PROHIBITION"),
                    ]
                elif formulation == 2:
                    body = (
                        "Where a personal data breach occurs, the data fiduciary shall give notice to the supervisory authority "
                        "within 72 hours after becoming aware of it, unless the personal data breach is unlikely to result in a risk to rights."
                    )
                    labeled_spans = [
                        ("personal data breach occurs", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall give notice to the supervisory authority", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ("within 72 hours", "ROLE_DEADLINE_WINDOW"),
                        ("unless the personal data breach is unlikely to result in a risk to rights", "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [
                        (1, 0, "REL_ACTION_TO_PRECONDITION"),
                        (2, 1, "REL_DEADLINE_TO_TRIGGER"),
                        (3, 1, "REL_EXCEPTION_TO_PROHIBITION"),
                    ]
                else:
                    body = (
                        "Every data fiduciary shall notify the Board of any personal data breach within 48 hours of detection, "
                        "and shall take reasonable security safeguards to prevent personal data breach."
                    )
                    labeled_spans = [
                        ("shall notify the Board", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ("personal data breach", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("within 48 hours", "ROLE_DEADLINE_WINDOW"),
                        ("shall take reasonable security safeguards", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ]
                    relations = [
                        (0, 1, "REL_ACTION_TO_PRECONDITION"),
                        (2, 0, "REL_DEADLINE_TO_TRIGGER"),
                        (3, 0, "REL_LOGICAL_AND"),
                    ]

            # -----------------------------------------------------------------
            # 25. IND AS 37 / IND AS 115: PROVISIONS & CONTRACT RECOGNITION
            # -----------------------------------------------------------------
            elif stype == "IND_AS_RECOGNITION_STYLE":
                if formulation in [0, 1]:
                    outflow_phrase = random.choice(["outflow of resources", "outflow of resources embodying economic benefits"])
                    body = (
                        f"A provision shall be recognised in profit or loss when an entity has a present obligation as a "
                        f"result of a past event, and it is probable that an {outflow_phrase} will be required to settle the obligation. "
                        f"Provided that where no reliable estimate can be made, no provision shall be recognised in the balance sheet "
                        f"and the item shall be disclosed in the notes as a contingent liability."
                    )
                    subj_phrase = f"probable that an {outflow_phrase} will be required"
                else:
                    body = (
                        "Where an entity has a present obligation as a result of a past event and it is probable that an outflow "
                        "of resources will be required to settle the obligation, a provision shall be recognised in profit or loss. "
                        "Provided that no provision shall be recognised in the balance sheet if a reliable estimate cannot be made."
                    )
                    subj_phrase = "probable that an outflow of resources will be required"
                labeled_spans = [
                    ("shall be recognised", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("in profit or loss", "ROLE_ACCOUNTING_DESTINATION"),
                    ("present obligation as a result of a past event", "ROLE_AGENT_STATE_PRECONDITION"),
                    (subj_phrase, "ROLE_SUBJECTIVE_CRITERION"),
                    ("no provision shall be recognised", "ROLE_EXCEPTION_CARVEOUT"),
                    ("in the balance sheet", "ROLE_ACCOUNTING_DESTINATION")
                ]
                relations = [
                    (1, 0, "REL_DESTINATION_TO_ACTION"),
                    (2, 0, "REL_ACTION_TO_PRECONDITION"),
                    (3, 0, "REL_ACTION_TO_PRECONDITION"),
                    (4, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (5, 4, "REL_DESTINATION_TO_ACTION")
                ]

            # -----------------------------------------------------------------
            # 26. IND AS 116: LEASES PRACTICAL EXPEDIENTS & CARVEOUTS
            # -----------------------------------------------------------------
            elif stype == "IND_AS_ELECTION_CARVEOUT_STYLE":
                if formulation == 0:
                    body = (
                        "A lessee may elect not to apply the requirements of lease recognition to short-term leases "
                        "having a lease term of twelve months or less, and leases for which the underlying asset is of low value. "
                        "The lease payments associated with those leases shall be recognised in profit or loss on a straight-line basis."
                    )
                elif formulation == 1:
                    body = (
                        "Short-term leases having a lease term of twelve months or less and low-value assets "
                        "may be excluded from lease recognition by election of the lessee. "
                        "Associated lease payments shall be recognised in profit or loss on a straight-line basis."
                    )
                else:
                    body = (
                        "In respect of short-term leases having a lease term of twelve months or less and low-value assets, "
                        "a lessee may elect not to apply lease recognition. "
                        "The lease payments associated with those leases shall be recognised in profit or loss on a straight-line basis."
                    )
                gate_str = (
                    "may elect not to apply the requirements of lease recognition"
                    if "may elect not to apply the requirements of lease recognition" in body
                    else (
                        "may elect not to apply lease recognition"
                        if "may elect not to apply lease recognition" in body
                        else "may be excluded from lease recognition by election"
                    )
                )
                carve1 = "short-term leases" if "short-term leases" in body else "Short-term leases"
                carve2 = "leases for which the underlying asset is of low value" if "leases for which the underlying asset is of low value" in body else "low-value assets"
                labeled_spans = [
                    (gate_str, "ROLE_PROHIBITION_GATE"),
                    (carve1, "ROLE_EXCEPTION_CARVEOUT"),
                    ("twelve months or less", "ROLE_DEADLINE_WINDOW"),
                    (carve2, "ROLE_EXCEPTION_CARVEOUT"),
                    ("shall be recognised", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("in profit or loss", "ROLE_ACCOUNTING_DESTINATION")
                ]
                relations = [
                    (1, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (2, 1, "REL_DEADLINE_TO_TRIGGER"),
                    (3, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (3, 1, "REL_LOGICAL_AND"),
                    (5, 4, "REL_DESTINATION_TO_ACTION")
                ]

            # -----------------------------------------------------------------
            # 27. IND AS 36 / IND AS 2: IMPAIRMENT & MEASUREMENT VALUATION
            # -----------------------------------------------------------------
            elif stype == "IND_AS_VALUATION_MEASUREMENT_STYLE":
                if formulation in [0, 1]:
                    body = (
                        "The recoverable amount of an asset shall be measured at the higher of its fair value less costs of disposal and its value in use. "
                        "Where the carrying amount exceeds the recoverable amount, the asset is impaired and an impairment loss shall be recognised immediately in profit or loss."
                    )
                else:
                    body = (
                        "Where the carrying amount exceeds the recoverable amount, an impairment loss shall be recognised immediately in profit or loss. "
                        "The recoverable amount of an asset shall be measured at the higher of its fair value less costs of disposal and its value in use."
                    )
                labeled_spans = [
                    ("shall be measured", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("higher of its fair value less costs of disposal and its value in use", "ROLE_SUBJECTIVE_CRITERION"),
                    ("carrying amount exceeds the recoverable amount", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    ("impairment loss shall be recognised immediately", "ROLE_STATUTORY_SANCTION"),
                    ("in profit or loss", "ROLE_ACCOUNTING_DESTINATION")
                ]
                relations = [
                    (1, 0, "REL_EVIDENCE_TO_SUBJECTIVE"),
                    (2, 3, "REL_THRESHOLD_TO_TRANSACTION"),
                    (3, 0, "REL_SANCTION_TO_VIOLATION"),
                    (4, 3, "REL_DESTINATION_TO_ACTION")
                ]

            # -----------------------------------------------------------------
            # 28. IND AS 19 / IND AS 109: OCI & EQUITY PRESENTATION
            # -----------------------------------------------------------------
            elif stype == "IND_AS_OCI_EQUITY_STYLE":
                if formulation in [0, 1]:
                    body = (
                        "Remeasurements of the net defined benefit liability comprising actuarial gains and losses shall be recognised immediately in other comprehensive income. "
                        "Such remeasurements shall not be reclassified to profit or loss in a subsequent period."
                    )
                else:
                    body = (
                        "Actuarial gains and losses on net defined benefit liability shall not be reclassified to profit or loss in a subsequent period. "
                        "All remeasurements shall be recognised immediately in other comprehensive income."
                    )
                labeled_spans = [
                    ("shall be recognised immediately", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ("in other comprehensive income", "ROLE_ACCOUNTING_DESTINATION"),
                    ("shall not be reclassified", "ROLE_PROHIBITION_GATE"),
                    ("to profit or loss", "ROLE_ACCOUNTING_DESTINATION")
                ]
                relations = [
                    (1, 0, "REL_DESTINATION_TO_ACTION"),
                    (3, 2, "REL_DESTINATION_TO_ACTION")
                ]

            # -----------------------------------------------------------------
            # 29. BIZARRE RULE 1: SECTION 194R PERQUISITES IN KIND
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_194R_PERQUISITES_IN_KIND":
                cliff = "20,000 rupees" if formulation % 2 == 0 else "₹20,000"
                to_gate = "100 lakh rupees" if formulation % 2 == 0 else "one crore rupees"
                if formulation == 0:
                    opt_appo = random.choice(["", ", whether convertible into money or not,"])
                    opt_src = random.choice(["arising from business", "arising from business or the exercise of a profession"])
                    body = (
                        f"Any person responsible for providing to a resident any benefit or perquisite{opt_appo} {opt_src} "
                        f"shall deduct tax at the rate of 10 per cent of the value of such benefit. "
                        f"Provided that no deduction shall be made if the aggregate value does not exceed {cliff} in a financial year. "
                        f"Provided further that the provisions of this section shall not apply to an individual whose total turnover does not exceed {to_gate}."
                    )
                    labeled_spans = [
                        ("resident", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        ("10 per cent", "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"does not exceed {cliff}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("no deduction shall be made", "ROLE_EXCEPTION_CARVEOUT"),
                        (f"total turnover does not exceed {to_gate}", "ROLE_BUYER_PRIOR_TURNOVER_GUARD"),
                    ]
                    relations = [
                        (1, 2, "REL_RATE_TO_THRESHOLD"),
                        (3, 1, "REL_EXCEPTION_TO_PROHIBITION"),
                        (4, 0, "REL_PRECONDITION_TO_ENTITY"),
                    ]
                elif formulation == 1:
                    opt_appo = random.choice(["", ", whether convertible into money or not,"])
                    body = (
                        f"Provided that in a case where the benefit or perquisite{opt_appo} is wholly in kind or partly in cash and partly in kind, "
                        f"the person responsible for providing such benefit or perquisite shall, before releasing the benefit, "
                        f"ensure that tax required to be deducted has been paid in respect of such benefit. "
                        f"Provided further that no deduction shall be made where aggregate value does not exceed {cliff}, "
                        f"or by an individual whose turnover does not exceed {to_gate}."
                    )
                    labeled_spans = [
                        ("wholly in kind or partly in cash", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("before releasing the benefit", "ROLE_PROHIBITION_GATE"),
                        ("ensure that tax required to be deducted has been paid", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ("no deduction shall be made", "ROLE_EXCEPTION_CARVEOUT"),
                        (f"does not exceed {cliff}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"turnover does not exceed {to_gate}", "ROLE_BUYER_PRIOR_TURNOVER_GUARD"),
                    ]
                    relations = [
                        (2, 0, "REL_ACTION_TO_PRECONDITION"),
                        (1, 2, "REL_CONDITION_TO_GATE"),
                        (3, 2, "REL_EXCEPTION_TO_PROHIBITION"),
                        (5, 0, "REL_PRECONDITION_TO_ENTITY"),
                    ]
                else:
                    body = (
                        f"Provided that no deduction shall be made where the aggregate value of any benefit does not exceed {cliff} in a financial year, "
                        f"or where total turnover does not exceed {to_gate}, "
                        f"any person responsible for providing to a resident any benefit or perquisite arising from business "
                        f"shall deduct tax at the rate of 10 per cent of the value of such benefit."
                    )
                    labeled_spans = [
                        (f"does not exceed {cliff}", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (f"total turnover does not exceed {to_gate}", "ROLE_BUYER_PRIOR_TURNOVER_GUARD"),
                        ("resident", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        ("10 per cent", "ROLE_STANDARD_STATUTORY_RATE"),
                    ]
                    relations = [
                        (3, 0, "REL_RATE_TO_THRESHOLD"),
                        (1, 2, "REL_PRECONDITION_TO_ENTITY"),
                    ]

            # -----------------------------------------------------------------
            # 30. BIZARRE RULE 2: SECTION 194BA ONLINE GAMING NET WINNINGS
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_ONLINE_GAMING_194BA":
                de_minimis = "one hundred rupees" if formulation % 2 == 0 else "₹100"
                rate_ba = "30 per cent" if formulation % 2 == 0 else "30%"
                if formulation == 0:
                    body = (
                        f"Any person responsible for paying to any person any income by way of winnings from any online game "
                        f"shall deduct income-tax on the net winnings in his user account at the rate of {rate_ba} at the time of withdrawal during the financial year. "
                        f"Provided that where the net winnings does not exceed {de_minimis}, no deduction shall be made."
                    )
                    labeled_spans = [
                        (rate_ba, "ROLE_STANDARD_STATUTORY_RATE"),
                        ("at the time of withdrawal", "ROLE_TRIGGER_PAYMENT_ONLY"),
                        (de_minimis, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("no deduction shall be made", "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [
                        (0, 2, "REL_RATE_TO_THRESHOLD"),
                        (3, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        (1, 0, "REL_DEADLINE_TO_TRIGGER"),
                    ]
                elif formulation == 1:
                    verb_summary = random.choice(["Mandates", "Imposes", "Prescribes"])
                    trigger_phrase = random.choice(["at the time of withdrawal", "upon user withdrawal"])
                    body = (
                        f"{verb_summary} {rate_ba} tax deduction on net winnings in any user account of online gaming platforms, "
                        f"computed {trigger_phrase} or at the end of the financial year: "
                        f"Provided that no deduction applies where winnings do not exceed {de_minimis}."
                    )
                    labeled_spans = [
                        (rate_ba, "ROLE_STANDARD_STATUTORY_RATE"),
                        ("net winnings", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        (trigger_phrase, "ROLE_TRIGGER_PAYMENT_ONLY"),
                        (de_minimis, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("no deduction applies", "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [
                        (0, 1, "REL_RATE_TO_THRESHOLD"),
                        (2, 0, "REL_DEADLINE_TO_TRIGGER"),
                        (4, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    ]
                elif formulation == 2:
                    body = (
                        f"Provided that where the net winnings does not exceed {de_minimis}, no deduction shall be made, "
                        f"any person responsible for paying to any person any income by way of winnings from any online game "
                        f"shall deduct income-tax on the net winnings in his user account at the rate of {rate_ba} at the time of withdrawal during the financial year."
                    )
                    labeled_spans = [
                        (de_minimis, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ("no deduction shall be made", "ROLE_EXCEPTION_CARVEOUT"),
                        (rate_ba, "ROLE_STANDARD_STATUTORY_RATE"),
                        ("at the time of withdrawal", "ROLE_TRIGGER_PAYMENT_ONLY"),
                    ]
                    relations = [
                        (2, 0, "REL_RATE_TO_THRESHOLD"),
                        (1, 2, "REL_EXCEPTION_TO_PROHIBITION"),
                        (3, 2, "REL_DEADLINE_TO_TRIGGER"),
                    ]
                else:
                    body = (
                        f"Computed at the time of withdrawal or at the end of the financial year, "
                        f"{rate_ba} withholding applies to net winnings in any gaming user account, "
                        f"except where aggregate winnings do not exceed {de_minimis}."
                    )
                    labeled_spans = [
                        ("at the time of withdrawal", "ROLE_TRIGGER_PAYMENT_ONLY"),
                        (rate_ba, "ROLE_STANDARD_STATUTORY_RATE"),
                        ("net winnings", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                        (de_minimis, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [
                        (1, 2, "REL_RATE_TO_THRESHOLD"),
                        (0, 1, "REL_DEADLINE_TO_TRIGGER"),
                    ]

            # -----------------------------------------------------------------
            # 31. BIZARRE RULE 3: TRIPLE THRESHOLD COLLISION
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_TRIPLE_THRESHOLD_COLLISION":
                nw = "500 crore" if formulation % 2 == 0 else "₹500,00,00,000"
                to = "100 crore" if formulation % 2 == 0 else "₹100,00,00,000"
                th = "50 lakh" if formulation % 2 == 0 else "₹50,00,000"
                rate = "2 per cent" if formulation % 2 == 0 else "2%"
                penal = "20 per cent" if formulation % 2 == 0 else "20%"
                if formulation in [0, 1]:
                    body = (
                        f"Every corporate buyer having a net worth exceeding {nw} or total turnover exceeding {to} "
                        f"shall deduct tax at the rate of {rate} on any sum exceeding {th} paid to a resident supplier. "
                        f"Provided that if the supplier fails to furnish permanent account number, tax shall be deducted at the rate of {penal}."
                    )
                else:
                    body = (
                        f"Provided that if the supplier fails to furnish permanent account number, tax shall be deducted at the rate of {penal}, "
                        f"every corporate buyer having a net worth exceeding {nw} or total turnover exceeding {to} "
                        f"shall deduct tax at the rate of {rate} on any sum exceeding {th} paid to a resident supplier."
                    )
                labeled_spans = [
                    (nw, "ROLE_AGENT_STATE_PRECONDITION"),
                    (to, "ROLE_AGENT_STATE_PRECONDITION"),
                    (rate, "ROLE_STANDARD_STATUTORY_RATE"),
                    (th, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    ("fails to furnish permanent account number", "ROLE_SUBJECTIVE_CRITERION"),
                    (penal, "ROLE_PENALTY_OR_DEFAULT_OVERRIDE"),
                ]
                relations = [
                    (1, 0, "REL_LOGICAL_OR"),
                    (2, 3, "REL_RATE_TO_THRESHOLD"),
                    (5, 4, "REL_OVERRIDE_TO_GUARD"),
                ]

            # -----------------------------------------------------------------
            # 32. BIZARRE RULE 4: SECTION 50B SLUMP SALE NEGATIVE NET WORTH
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_50B_SLUMP_SALE_NEGATIVE_NETWORTH":
                duration = "thirty-six months" if formulation % 2 == 0 else "36 months"
                if formulation in [0, 1]:
                    body = (
                        f"Any profits or gains arising from the slump sale of an undertaking shall be chargeable to income-tax as capital gains "
                        f"arising from the transfer of long-term capital assets if held for not more than {duration}. "
                        f"Provided that where the net worth of the undertaking is negative, the cost of acquisition shall be deemed to be nil."
                    )
                else:
                    body = (
                        f"Where the net worth of the undertaking is negative, the cost of acquisition shall be deemed to be nil. "
                        f"Any profits or gains arising from the slump sale of an undertaking shall be chargeable to income-tax as capital gains "
                        f"arising from the transfer of long-term capital assets if held for not more than {duration}."
                    )
                labeled_spans = [
                    ("capital gains", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    (f"not more than {duration}", "ROLE_DEADLINE_WINDOW"),
                    ("net worth of the undertaking is negative", "ROLE_AGENT_STATE_PRECONDITION"),
                    ("cost of acquisition shall be deemed to be nil", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                ]
                relations = [
                    (1, 0, "REL_DEADLINE_TO_TRIGGER"),
                    (3, 2, "REL_ACTION_TO_PRECONDITION"),
                ]

            # -----------------------------------------------------------------
            # 33. BIZARRE RULE 5: DIGITAL SERVICES EQUALISATION LEVY
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_DIGITAL_SERVICES_EQUALISATION":
                eq_rate = "2 per cent" if formulation % 2 == 0 else "2%"
                eq_turnover = "two crore rupees" if formulation % 2 == 0 else "₹2,00,00,000"
                if formulation in [0, 1]:
                    body = (
                        f"There shall be charged an equalisation levy at the rate of {eq_rate} of the amount of consideration "
                        f"received or receivable by an e-commerce operator from e-commerce supply or services made to a resident. "
                        f"Provided that the equalisation levy shall not be charged where the sales or turnover does not exceed {eq_turnover} during the previous year, "
                        f"or where the e-commerce operator has a permanent establishment in India."
                    )
                else:
                    body = (
                        f"The equalisation levy shall not be charged where the sales or turnover does not exceed {eq_turnover} during the previous year, "
                        f"or where the e-commerce operator has a permanent establishment in India. "
                        f"There shall be charged an equalisation levy at the rate of {eq_rate} of the amount of consideration "
                        f"received or receivable by an e-commerce operator from e-commerce supply or services made to a resident."
                    )
                not_charged_str = "equalisation levy shall not be charged" if "equalisation levy shall not be charged" in body else "The equalisation levy shall not be charged"
                labeled_spans = [
                    (eq_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                    ("resident", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    (not_charged_str, "ROLE_PROHIBITION_GATE"),
                    (eq_turnover, "ROLE_AGENT_STATE_PRECONDITION"),
                    ("has a permanent establishment in India", "ROLE_EXCEPTION_CARVEOUT"),
                ]
                relations = [
                    (0, 3, "REL_RATE_TO_THRESHOLD"),
                    (4, 2, "REL_EXCEPTION_TO_PROHIBITION"),
                ]

            # -----------------------------------------------------------------
            # 34. BIZARRE RULE 6: COMPANIES ACT FRACTIONAL QUORUM
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_COMPANIES_QUORUM_FRACTION":
                fraction_str = "two-thirds of the total strength" if formulation % 2 == 0 else "two-thirds of total strength"
                floor_str = "not less than two"
                if formulation in [0, 1]:
                    body = (
                        f"Where the number of interested directors exceeds or is equal to {fraction_str} of the Board of Directors, "
                        f"the number of directors who are not interested and who are present at the meeting, being {floor_str}, "
                        f"shall be the quorum during such time."
                    )
                else:
                    body = (
                        f"The number of directors who are not interested and who are present at the meeting, being {floor_str}, "
                        f"shall be the quorum during such time, where the number of interested directors exceeds or is equal to {fraction_str} of the Board of Directors."
                    )
                labeled_spans = [
                    ("interested directors", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    (fraction_str, "ROLE_AGENT_STATE_PRECONDITION"),
                    (floor_str, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("shall be the quorum", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                ]
                relations = [
                    (3, 2, "REL_ACTION_TO_PRECONDITION"),
                    (1, 0, "REL_CONDITION_TO_GATE"),
                ]

            # -----------------------------------------------------------------
            # 35. BIZARRE RULE 7: SECTION 194S DUAL-CLIFF VIRTUAL DIGITAL ASSETS
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_194S_DUAL_CLIFF_CRYPTO_VDA":
                cliff_specified = "fifty thousand rupees" if formulation % 2 == 0 else "₹50,000"
                cliff_general = "ten thousand rupees" if formulation % 2 == 0 else "₹10,000"
                if formulation in [0, 1]:
                    body = (
                        f"Any person responsible for paying to a resident any sum by way of consideration for transfer of a virtual digital asset "
                        f"shall deduct an amount equal to 1 per cent of such sum as income-tax. "
                        f"Provided that where the consideration is paid by a specified person, no deduction shall be made if such consideration does not exceed {cliff_specified} during the financial year. "
                        f"Provided further that in any other case, no deduction shall be made if such consideration does not exceed {cliff_general} during the financial year."
                    )
                else:
                    body = (
                        f"Provided that where consideration does not exceed {cliff_specified} in case of a specified person, "
                        f"or does not exceed {cliff_general} in any other case during the financial year, no deduction shall be made, "
                        f"any person responsible for paying to a resident consideration for transfer of a virtual digital asset "
                        f"shall deduct an amount equal to 1 per cent of such sum."
                    )
                labeled_spans = [
                    ("1 per cent", "ROLE_STANDARD_STATUTORY_RATE"),
                    ("resident", "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE"),
                    ("specified person", "ROLE_AGENT_STATE_PRECONDITION"),
                    (cliff_specified, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    (cliff_general, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("no deduction shall be made", "ROLE_EXCEPTION_CARVEOUT"),
                ]
                relations = [
                    (0, 3, "REL_RATE_TO_THRESHOLD"),
                    (0, 4, "REL_RATE_TO_THRESHOLD"),
                    (5, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (2, 3, "REL_PRECONDITION_TO_ENTITY"),
                ]

            # -----------------------------------------------------------------
            # 36. BIZARRE RULE 8: SECTION 269ST CASH TRANSACTION PROHIBITION & 100% PENALTY
            # -----------------------------------------------------------------
            elif stype == "BIZARRE_269ST_CASH_RECEIPT_PROHIBITION":
                cash_limit = "two lakh rupees" if formulation % 2 == 0 else "₹2,00,000"
                if formulation in [0, 1]:
                    body = (
                        f"No person shall receive an amount of {cash_limit} or more in aggregate from a person in a day "
                        f"or in respect of a single transaction otherwise than by an account payee cheque. "
                        f"Provided that if any person receives any sum in contravention of this section, "
                        f"he shall be liable to pay by way of penalty a sum equal to the amount of such receipt."
                    )
                else:
                    body = (
                        f"Except where receipt is otherwise than by an account payee cheque, "
                        f"no person shall receive an amount of {cash_limit} or more in aggregate from a person in a day. "
                        f"If any person receives any sum in contravention of this requirement, "
                        f"he shall be liable to pay by way of penalty a sum equal to the amount of such receipt."
                    )
                gate_str = "No person shall receive" if "No person shall receive" in body else "no person shall receive"
                labeled_spans = [
                    (gate_str, "ROLE_PROHIBITION_GATE"),
                    (cash_limit, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ("otherwise than by an account payee cheque", "ROLE_EXCEPTION_CARVEOUT"),
                    ("a sum equal to the amount of such receipt", "ROLE_STATUTORY_SANCTION"),
                ]
                relations = [
                    (1, 0, "REL_THRESHOLD_TO_TRANSACTION"),
                    (2, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    (3, 0, "REL_SANCTION_TO_VIOLATION"),
                ]

            # -----------------------------------------------------------------
            # 37. GENERAL OBLIGATION & CARVEOUT RULES (SPORTS / REGULATORY / GENERAL)
            # -----------------------------------------------------------------
            elif stype == "OBLIGATION_CARVEOUT_GENERAL_STYLE":
                gen_subj = random.choice(["player", "participant", "member", "operator", "delegate", "contender", "applicant", "custodian"])
                if formulation == 0:
                    variant = random.choice([0, 1, 2])
                    if variant == 0:
                        sanction_phrase = random.choice(["shall be penalised", "shall be sanctioned", "shall be disqualified or penalised", "shall be suspended"])
                        infraction_noun = random.choice(["for an offence", "for a violation", "for an infraction", "for a breach", "for a foul"])
                        trig_cond = random.choice([
                            "if entering a restricted zone",
                            "if crossing the boundary line during active operation",
                            "if committing a breach while occupying a prohibited sector",
                            "if receiving the item in a restricted position",
                            "if receiving a signal inside an unauthorised zone",
                            "if handling the baton in a forward sector",
                        ])
                        src1, src2, src3 = random.choice([
                            ("boundary restart", "side release", "designated pass"),
                            ("free clearance", "neutral zone", "authorised pass"),
                            ("official restart", "flank release", "baseline release"),
                        ])
                        carve_clause = random.choice([
                            "where the actor acts directly under an official clearance or umpire signal",
                            "where the movement followed an authorised restart protocol or exemption event",
                            f"where the {gen_subj} receives the item directly from a {src1}, a {src2}, or a {src3}",
                            "where the item passed directly from an approved exception boundary",
                        ])
                        body = (
                            f"A {gen_subj} {sanction_phrase} {infraction_noun} {trig_cond}: "
                            f"Provided that it shall not be an offence {carve_clause}."
                        )
                        labeled_spans = [
                            (sanction_phrase, "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (trig_cond, "ROLE_AGENT_STATE_PRECONDITION"),
                            (carve_clause, "ROLE_EXCEPTION_CARVEOUT"),
                        ]
                        relations = [
                            (0, 1, "REL_ACTION_TO_PRECONDITION"),
                            (2, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        ]
                    elif variant == 1:
                        act_out = random.choice([
                            "shall be excluded from eligibility",
                            "shall be judged out of compliance",
                            "shall be ruled out of contention",
                            "shall be deemed liable",
                            "shall forfeit the round",
                        ])
                        trig_cond = random.choice([
                            "if the object strikes his person inside the designated line",
                            "if the signal triggers outside the permitted channel",
                            "if the asset crosses outside the prescribed zone",
                            "if the marker contacts the barrier in line of the targets",
                        ])
                        carve_clause = random.choice([
                            "where the object originated outside the legitimate bound",
                            "where the object engaged outside the protected strip",
                            "where the transaction occurred outside the restricted perimeter",
                            "where official release occurred prior to cutoff",
                        ])
                        carve_deny = random.choice([
                            f"shall not suffer exclusion",
                            f"shall not be ruled out",
                            f"shall not be judged non-compliant",
                        ])
                        body = (
                            f"The {gen_subj} {act_out} {trig_cond}: "
                            f"Provided that the {gen_subj} {carve_deny} {carve_clause}."
                        )
                        labeled_spans = [
                            (act_out, "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (trig_cond, "ROLE_AGENT_STATE_PRECONDITION"),
                            (carve_clause, "ROLE_EXCEPTION_CARVEOUT"),
                        ]
                        relations = [
                            (0, 1, "REL_ACTION_TO_PRECONDITION"),
                            (2, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        ]
                    else:
                        sanction_phrase = random.choice(["shall be penalised", "shall be disqualified", "shall be declared ineligible"])
                        trig_cond = random.choice([
                            "if occupying a restricted zone when the restart begins",
                            "if receiving a transfer directly inside a prohibited boundary",
                            "if acting beyond the designated line during active play",
                        ])
                        carve_clause = random.choice([
                            "where the transfer came directly from a neutral restart, a permitted exchange, or an official release",
                            "where the movement followed an authorised clearance or designated exemption signal",
                        ])
                        body = (
                            f"Every {gen_subj} {sanction_phrase} {trig_cond}: "
                            f"Provided that it shall not constitute a violation {carve_clause}."
                        )
                        labeled_spans = [
                            (sanction_phrase, "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (trig_cond, "ROLE_AGENT_STATE_PRECONDITION"),
                            (carve_clause, "ROLE_EXCEPTION_CARVEOUT"),
                        ]
                        relations = [
                            (0, 1, "REL_ACTION_TO_PRECONDITION"),
                            (2, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        ]
                elif formulation == 1:
                    variant = random.choice([0, 1])
                    if variant == 0:
                        part_a, part_b = random.choice([
                            ("the primary seal", "the backup unit"),
                            ("the principal unit", "the auxiliary unit"),
                            ("the lead officer", "the deputy custodian"),
                            ("the first marker", "the anchor piece"),
                        ])
                        perm_verb = random.choice(["proceed", "relocate", "execute a swap", "activate transfer"])
                        state_verb = random.choice(["moved", "engaged", "shifted", "activated"])
                        prohib_verb = random.choice(["pass through", "bypass", "enter"])
                        zone_noun = random.choice(["guard sector", "channel", "corridor"])
                        adj_cond = random.choice(["contested", "controlled", "blocked"])
                        opp_noun = random.choice(["opposing unit", "rival party", "adversary node"])
                        quant_cond = f"any {zone_noun} {adj_cond} by an {opp_noun}"
                        body = (
                            f"A {gen_subj} may {perm_verb} provided that neither {part_a} nor {part_b} has previously {state_verb}, "
                            f"and {part_a} shall not {prohib_verb} {quant_cond}."
                        )
                        modality_override = "MODALITY_DEONTIC_PERMISSION"
                        labeled_spans = [
                            (f"may {perm_verb}", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (f"neither {part_a} nor {part_b} has previously {state_verb}", "ROLE_AGENT_STATE_PRECONDITION"),
                            (f"shall not {prohib_verb}", "ROLE_PROHIBITION_GATE"),
                            (quant_cond, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [
                            (0, 1, "REL_ACTION_TO_PRECONDITION"),
                            (2, 3, "REL_LOGICAL_NOT"),
                            (3, 2, "REL_QUANTIFIER_EXISTS"),
                        ]
                    else:
                        trig_evt = random.choice(["entering an exclusion zone", "triggering a compliance lock", "landing on a restricted space"])
                        duty_act = random.choice(["shall relocate directly to detention", "shall transfer immediately to escrow", "shall proceed directly to confinement"])
                        forbid_act = random.choice(["shall not collect bonus credits", "shall not draw settlement funds", "shall not receive two hundred credits"])
                        escape_cond = random.choice([
                            "upon paying an administrative fee of $50 or completing matching confirmations",
                            "upon posting a bond of $100 or completing dual verification",
                            "upon paying an administrative fee or completing matching confirmations",
                        ])
                        win_span = random.choice(["within three turns", "within forty-eight hours", "within five cycles"])
                        body = (
                            f"A {gen_subj} {trig_evt} {duty_act} and {forbid_act}: "
                            f"Provided that the {gen_subj} may resume {escape_cond} {win_span}."
                        )
                        modality_override = "MODALITY_DEONTIC_PROHIBITION"
                        labeled_spans = [
                            (trig_evt, "ROLE_AGENT_STATE_PRECONDITION"),
                            (duty_act, "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (forbid_act, "ROLE_PROHIBITION_GATE"),
                            (escape_cond, "ROLE_EXCEPTION_CARVEOUT"),
                            (win_span, "ROLE_DEADLINE_WINDOW"),
                        ]
                        relations = [
                            (1, 0, "REL_ACTION_TO_PRECONDITION"),
                            (2, 1, "REL_LOGICAL_AND"),
                            (3, 2, "REL_EXCEPTION_TO_PROHIBITION"),
                            (4, 3, "REL_DEADLINE_TO_TRIGGER"),
                        ]
                elif formulation == 2:
                    carve_clause = random.choice([
                        "where approved directly by an umpire signal",
                        "where authorized by official restart protocol",
                    ])
                    body = (
                        f"Provided that it shall not be a violation {carve_clause}, "
                        f"a {gen_subj} shall be penalised if crossing a restricted boundary."
                    )
                    labeled_spans = [
                        (carve_clause, "ROLE_EXCEPTION_CARVEOUT"),
                        ("shall be penalised", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ("if crossing a restricted boundary", "ROLE_AGENT_STATE_PRECONDITION"),
                    ]
                    relations = [
                        (0, 1, "REL_EXCEPTION_TO_PROHIBITION"),
                        (1, 2, "REL_ACTION_TO_PRECONDITION"),
                    ]
                else:
                    variant = random.choice([0, 1])
                    if variant == 0:
                        body = (
                            f"Provided that neither the primary unit nor the backup unit has previously shifted, "
                            f"a {gen_subj} may relocate or claim clearance, and the primary unit shall not bypass any active guard sector."
                        )
                        modality_override = "MODALITY_DEONTIC_PERMISSION"
                        labeled_spans = [
                            ("neither the primary unit nor the backup unit has previously shifted", "ROLE_AGENT_STATE_PRECONDITION"),
                            ("may relocate or claim clearance", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            ("shall not bypass", "ROLE_PROHIBITION_GATE"),
                            ("any active guard sector", "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [
                            (1, 0, "REL_ACTION_TO_PRECONDITION"),
                            (2, 3, "REL_LOGICAL_NOT"),
                            (3, 2, "REL_QUANTIFIER_EXISTS"),
                        ]
                    else:
                        escape_cond = random.choice([
                            "upon paying an administrative fee of $50 or completing dual verification",
                            "upon posting a bond of $100 or executing matching confirmations",
                        ])
                        win_span = random.choice(["within three rounds", "within forty-eight hours"])
                        body = (
                            f"Provided that the {gen_subj} may resume {escape_cond} {win_span}, "
                            f"a {gen_subj} triggering a compliance lock shall transfer immediately to escrow and shall not draw settlement funds."
                        )
                        modality_override = "MODALITY_DEONTIC_PROHIBITION"
                        labeled_spans = [
                            (escape_cond, "ROLE_EXCEPTION_CARVEOUT"),
                            (win_span, "ROLE_DEADLINE_WINDOW"),
                            ("triggering a compliance lock", "ROLE_AGENT_STATE_PRECONDITION"),
                            ("shall transfer immediately to escrow", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            ("shall not draw settlement funds", "ROLE_PROHIBITION_GATE"),
                        ]
                        relations = [
                            (3, 2, "REL_ACTION_TO_PRECONDITION"),
                            (4, 3, "REL_LOGICAL_AND"),
                            (0, 4, "REL_EXCEPTION_TO_PROHIBITION"),
                            (1, 0, "REL_DEADLINE_TO_TRIGGER"),
                        ]

            # -----------------------------------------------------------------
            # 38. COMBINATOR LESSER-OF PHASEOUT (MIN(A, B) MULTI-TERM)
            # -----------------------------------------------------------------
            elif stype == "COMBINATOR_LESSER_OF_STYLE":
                preamble = ""
                trig_op = random.choice([
                    "assessable revenue", "gross enterprise receipts",
                    "aggregate taxable turnover", "consolidated operating income",
                    "net assessable revenue", "qualifying composite turnover",
                ])
                base_op = random.choice([
                    "allowable operating expenses", "qualifying project outlays",
                    "eligible deductible charges", "assessable maintenance expenses",
                    "allowable capital deductions", "permitted project outlays",
                ])
                th = random.choice(["$120,000", "$240,000", "$450,000", "€95,000", "EUR 180,000", "£75,000", "USD 160,000"])
                r1 = random.choice(["4%", "7%", "9%", "4 percent", "7 per cent", "9 percent"])
                r2 = random.choice(["45%", "65%", "75%", "45 percent", "65 per cent", "75 percent"])
                if formulation == 0:
                    comb_v = random.choice([0, 1, 2])
                    if comb_v == 0:
                        body = (
                            f"Where the {trig_op} of a taxpayer exceeds {th}, the allowable {base_op} "
                            f"shall be reduced by the lesser of {r1} of the excess of {trig_op} over {th}, "
                            f"or {r2} of total {base_op}."
                        )
                        labeled_spans = [
                            (f"exceeds {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                            (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                            (f"excess of {trig_op}", "ROLE_COMPUTATION_OPERAND"),
                            (f"over {th}", "ROLE_IGNORE"),
                            (r2, "ROLE_STANDARD_STATUTORY_RATE"),
                            (f"total {base_op}", "ROLE_COMPUTATION_OPERAND"),
                        ]
                        relations = [
                            (1, 2, "REL_SUBTRACTION_BASE"),
                            (2, 0, "REL_THRESHOLD_TO_TRANSACTION"),
                            (4, 5, "REL_SUBTRACTION_BASE"),
                            (1, 4, "REL_COMBINATOR_MIN"),
                        ]
                    elif comb_v == 1:
                        body = (
                            f"In the case of an individual whose {trig_op} exceeds {th}, "
                            f"the amount of the {base_op} otherwise allowable for the calendar year "
                            f"shall be reduced by the lesser of {r1} of the excess of {trig_op} over {th}, "
                            f"or {r2} of the amount of the {base_op} otherwise allowable for such calendar year."
                        )
                        labeled_spans = [
                            (f"exceeds {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                            (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                            (f"excess of {trig_op}", "ROLE_COMPUTATION_OPERAND"),
                            (f"over {th}", "ROLE_IGNORE"),
                            (r2, "ROLE_STANDARD_STATUTORY_RATE"),
                            (f"amount of the {base_op}", "ROLE_COMPUTATION_OPERAND"),
                        ]
                        relations = [
                            (1, 2, "REL_SUBTRACTION_BASE"),
                            (2, 0, "REL_THRESHOLD_TO_TRANSACTION"),
                            (4, 5, "REL_SUBTRACTION_BASE"),
                            (1, 4, "REL_COMBINATOR_MIN"),
                        ]
                    else:
                        body = (
                            f"In the case of an enterprise whose {trig_op} exceeds {th}, "
                            f"the amount otherwise allowable as deductions shall be reduced by {r1} of the excess of {trig_op} over {th}, "
                            f"or {r2} of the amount of the {base_op} otherwise allowable, whichever is less."
                        )
                        labeled_spans = [
                            (f"exceeds {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                            (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                            (f"excess of {trig_op}", "ROLE_COMPUTATION_OPERAND"),
                            (f"over {th}", "ROLE_IGNORE"),
                            (r2, "ROLE_STANDARD_STATUTORY_RATE"),
                            (f"amount of the {base_op}", "ROLE_COMPUTATION_OPERAND"),
                        ]
                        relations = [
                            (1, 2, "REL_SUBTRACTION_BASE"),
                            (2, 0, "REL_THRESHOLD_TO_TRANSACTION"),
                            (4, 5, "REL_SUBTRACTION_BASE"),
                            (1, 4, "REL_COMBINATOR_MIN"),
                        ]
                elif formulation == 1:
                    comp_word = random.choice(["lower of", "smaller of"])
                    conj_word = "or" if comp_word == "lower of" else random.choice(["or", "and"])
                    body = (
                        f"A reduction in {base_op} shall equal the {comp_word} {r1} of the excess over {th} "
                        f"on {trig_op} {conj_word} {r2} of such {base_op}."
                    )
                    labeled_spans = [
                        (trig_op, "ROLE_COMPUTATION_OPERAND"),
                        (f"excess over {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                        (r2, "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"such {base_op}", "ROLE_COMPUTATION_OPERAND"),
                    ]
                    relations = [
                        (2, 1, "REL_RATE_TO_THRESHOLD"),
                        (1, 0, "REL_SUBTRACTION_BASE"),
                        (3, 4, "REL_SUBTRACTION_BASE"),
                        (2, 3, "REL_COMBINATOR_MIN"),
                    ]
                elif formulation == 2:
                    body = (
                        f"By the lesser of {r2} of total {base_op} or {r1} of the excess of {trig_op} over {th}, "
                        f"the allowable deductions shall be scaled down where receipts exceed {th}."
                    )
                    labeled_spans = [
                        (r2, "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"total {base_op}", "ROLE_COMPUTATION_OPERAND"),
                        (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"excess of {trig_op}", "ROLE_COMPUTATION_OPERAND"),
                        (f"over {th}", "ROLE_IGNORE"),
                        (f"exceed {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    ]
                    relations = [
                        (0, 1, "REL_SUBTRACTION_BASE"),
                        (2, 3, "REL_SUBTRACTION_BASE"),
                        (3, 5, "REL_THRESHOLD_TO_TRANSACTION"),
                        (0, 2, "REL_COMBINATOR_MIN"),
                    ]
                else:
                    body = (
                        f"Provided that aggregate turnover exceeds {th}, an entity shall apply a limitation "
                        f"equal to the smaller of {r1} of the surplus of {trig_op} above {th} and {r2} of eligible {base_op}."
                    )
                    labeled_spans = [
                        (f"exceeds {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"surplus of {trig_op}", "ROLE_COMPUTATION_OPERAND"),
                        (f"above {th}", "ROLE_IGNORE"),
                        (r2, "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"eligible {base_op}", "ROLE_COMPUTATION_OPERAND"),
                    ]
                    relations = [
                        (1, 2, "REL_SUBTRACTION_BASE"),
                        (2, 0, "REL_THRESHOLD_TO_TRANSACTION"),
                        (4, 5, "REL_SUBTRACTION_BASE"),
                        (1, 4, "REL_COMBINATOR_MIN"),
                    ]

            # -----------------------------------------------------------------
            # 39. COMBINATOR GREATER-OF FLOOR (MAX(A, B) MULTI-TERM)
            # -----------------------------------------------------------------
            elif stype == "COMBINATOR_GREATER_OF_STYLE":
                preamble = ""
                op_a = random.choice([
                    "adjusted book profit", "gross royalty base",
                    "assessed license fee", "consolidated platform receipts"
                ])
                r1 = random.choice(["12%", "18%", "25%", "12 percent", "18 per cent", "25 percent"])
                floor_amt = random.choice(["$14,500", "$35,000", "€22,000", "£18,500", "USD 48,000", "EUR 31,000"])
                if formulation == 0:
                    body = (
                        f"An entity shall remit a minimum charge equal to the greater of {r1} of {op_a} "
                        f"or {floor_amt} for each reporting cycle."
                    )
                elif formulation == 1:
                    body = (
                        f"The annual statutory levy payable on {op_a} is determined as the higher of {floor_amt} "
                        f"or {r1} of {op_a}."
                    )
                elif formulation == 2:
                    body = (
                        f"Computed as the higher of {r1} of {op_a} and {floor_amt}, "
                        f"every covered operator shall discharge the prescribed assessment."
                    )
                else:
                    body = (
                        f"Not less than the greater of {floor_amt} and {r1} of {op_a} "
                        f"shall be set aside in reserve by each regulated firm."
                    )
                if formulation in [1, 3]:
                    labeled_spans = [
                        (floor_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                        (op_a, "ROLE_COMPUTATION_OPERAND"),
                    ]
                    relations = [
                        (1, 2, "REL_SUBTRACTION_BASE"),
                        (1, 0, "REL_COMBINATOR_MAX"),
                    ]
                else:
                    labeled_spans = [
                        (r1, "ROLE_STANDARD_STATUTORY_RATE"),
                        (op_a, "ROLE_COMPUTATION_OPERAND"),
                        (floor_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [
                        (0, 1, "REL_SUBTRACTION_BASE"),
                        (0, 2, "REL_COMBINATOR_MAX"),
                    ]

            # -----------------------------------------------------------------
            # 40. EXPLICIT OPERAND SUBTRACTION EXCESS RULE (A - B) * R
            # -----------------------------------------------------------------
            elif stype == "SUBTRACTION_EXCESS_OPERAND_STYLE":
                preamble = ""
                op_a = random.choice(["gross export proceeds", "assessable trading surplus", "total freight earnings"])
                op_b = random.choice(["permitted capital outlay", "statutory abatement", "qualifying exemption floor"])
                th = random.choice(["$65,000", "$140,000", "€82,000", "£54,000", "USD 95,000"])
                rate = random.choice(["4.5%", "8.5%", "12%", "4.5 percent", "8.5 per cent", "12 percent"])
                if formulation == 0:
                    body = (
                        f"A levy of {rate} shall apply to the excess of {op_a} above {op_b} "
                        f"where aggregate valuation exceeds {th}."
                    )
                elif formulation == 1:
                    sub_prep = random.choice(["minus", "less"])
                    body = (
                        f"Where aggregate receipts exceed {th}, tax at {rate} is imposed on {op_a} {sub_prep} {op_b}."
                    )
                elif formulation == 2:
                    body = (
                        f"Computed on {op_a} less {op_b} where total volume exceeds {th}, "
                        f"a statutory charge of {rate} shall be levied."
                    )
                else:
                    body = (
                        f"Provided valuation exceeds {th}, every enterprise shall remit {rate} of the net amount "
                        f"by which {op_a} exceeds {op_b}."
                    )
                labeled_spans = [
                    (rate, "ROLE_STANDARD_STATUTORY_RATE"),
                    (op_a, "ROLE_COMPUTATION_OPERAND"),
                    (op_b, "ROLE_COMPUTATION_OPERAND"),
                    (f"exceeds {th}" if f"exceeds {th}" in body else f"exceed {th}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                ]
                relations = [
                    (0, 1, "REL_SUBTRACTION_BASE"),
                    (1, 2, "REL_SUBTRACTION_BASE"),
                    (0, 3, "REL_RATE_TO_THRESHOLD"),
                ]

            # -----------------------------------------------------------------
            # 41. CAPPED PER-UNIT BASE EXCISE RULE (RATE * MIN(VAR, CAP))
            # -----------------------------------------------------------------
            elif stype == "CAPPED_WAGE_BASE_STYLE":
                preamble = ""
                op_var, op_tail, base_term, act_scope = random.choice([
                    ("qualifying payroll", "payroll", "pensionable pay", "services"),
                    ("annual compensation", "compensation", "assessable pay", "employment"),
                    ("pensionable earnings", "earnings", "covered pay", "services"),
                    ("contractor remuneration", "remuneration", "qualifying pay", "engagement"),
                    ("gross commissions", "commissions", "commission base", "agency"),
                ])
                cap_amt = random.choice(["$9,500", "$14,200", "€18,000", "£11,500", "USD 16,000", "EUR 22,500"])
                rate = random.choice(["4.5%", "7%", "8.5%", "11%", "12.5%", "9 per cent"])
                is_rate_free_cap = (random.random() < 0.45)
                if is_rate_free_cap:
                    if formulation == 0:
                        body = (
                            f"The term {base_term} means all {op_tail} for {act_scope}, except that such term shall not "
                            f"include that part of the {op_tail} which, after {op_tail} equal to {cap_amt} with "
                            f"respect to {act_scope} has been paid to an individual by an employer during any calendar "
                            f"year, is paid to such individual by such employer during such calendar year."
                        )
                    elif formulation == 1:
                        body = (
                            f"Covered base comprises total {op_var} for {act_scope}, excluding that part of {op_tail} "
                            f"paid to a worker during a calendar year in excess of {cap_amt}."
                        )
                    elif formulation == 2:
                        body = (
                            f"Except that after {op_tail} equal to {cap_amt} has been paid during any calendar year "
                            f"excess sums are excluded, the term {base_term} means all {op_var} for {act_scope}."
                        )
                    else:
                        body = (
                            f"Subject to an upper ceiling equal to {cap_amt} per individual per calendar year, "
                            f"covered distributions include all {op_var} for {act_scope}."
                        )
                    op_span_txt = f"all {op_tail}" if f"all {op_tail}" in body else op_var
                    labeled_spans = [
                        (op_span_txt, "ROLE_COMPUTATION_OPERAND"),
                        (cap_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [
                        (1, 0, "REL_CAP_CEILING"),
                    ]
                else:
                    if formulation == 0:
                        if random.random() < 0.5:
                            body = (
                                f"Every employer shall pay a contribution equal to {rate} of {op_var} paid to each worker "
                                f"during the calendar year, except that such contribution base shall not include sums in excess of {cap_amt} "
                                f"paid to any single worker."
                            )
                        else:
                            body = (
                                f"A payer shall remit {rate} on {op_var}, except that such base shall not include "
                                f"that part of the {op_tail} which, after {op_tail} equal to {cap_amt} with respect to {act_scope} "
                                f"has been paid to an individual during any calendar year, is paid to such individual."
                            )
                    elif formulation == 1:
                        body = (
                            f"Contribution at {rate} applies to {op_var}, subject to a maximum per-recipient ceiling of {cap_amt} in any calendar period."
                        )
                    elif formulation == 2:
                        body = (
                            f"Excluding any part of {op_var} paid above {cap_amt} per employee in a calendar year, "
                            f"an assessment equal to {rate} shall be paid on covered payouts."
                        )
                    else:
                        body = (
                            f"Subject to a maximum per-recipient ceiling of {cap_amt}, "
                            f"tax at {rate} is imposed on total {op_var} disbursed during the year."
                        )
                    labeled_spans = [
                        (rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        (op_var, "ROLE_COMPUTATION_OPERAND"),
                        (cap_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [
                        (0, 1, "REL_RATE_TO_THRESHOLD"),
                        (2, 1, "REL_CAP_CEILING"),
                    ]

            # -----------------------------------------------------------------
            # 42. PROGRESSIVE TAX BRACKET CLAUSE (BASE_TAX + RATE * (X - FLOOR))
            # -----------------------------------------------------------------
            elif stype == "PROGRESSIVE_BRACKET_CLAUSE_STYLE":
                preamble = ""
                lo, hi, base_t, b_rate = random.choice([
                    ("$14,000", "$52,000", "$1,400", "24%"),
                    ("$25,000", "$92,000", "$2,750", "27%"),
                    ("€18,000", "€64,000", "€2,160", "25%"),
                    ("£12,500", "£58,000", "£1,500", "34%"),
                    ("USD 32,000", "USD 110,000", "USD 3,840", "24 percent"),
                ])
                if formulation == 0:
                    body = (
                        f"If taxable amount is over {lo} but not over {hi}, "
                        f"tax shall be {base_t}, plus {b_rate} of the excess over {lo}."
                    )
                    s_lo, s_hi, s_op = f"over {lo}", f"not over {hi}", "taxable amount"
                elif formulation == 1:
                    if random.random() < 0.5:
                        body = (
                            f"Where assessable income exceeds {lo} and does not exceed {hi}, "
                            f"the levy payable equals {base_t} plus {b_rate} of the amount above {lo}."
                        )
                        s_lo, s_hi, s_op = f"exceeds {lo}", f"does not exceed {hi}", "assessable income"
                    else:
                        body = (
                            f"For assessable income brackets between {lo} and {hi}, "
                            f"the levy equals {base_t} plus {b_rate} on any excess above {lo}."
                        )
                        s_lo, s_hi, s_op = lo, hi, "assessable income"
                elif formulation == 2:
                    body = (
                        f"Tax is computed as {base_t} plus {b_rate} of the slice exceeding {lo} "
                        f"whenever net taxable value is more than {lo} and up to {hi}."
                    )
                    s_lo, s_hi, s_op = f"exceeding {lo}", f"up to {hi}", "net taxable value"
                else:
                    body = (
                        f"For net taxable income brackets between {lo} and {hi}, total charge equals {base_t} "
                        f"together with {b_rate} on any excess above {lo}."
                    )
                    s_lo, s_hi, s_op = lo, hi, "net taxable income"

                if v2:
                    labeled_spans = [
                        (s_lo, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (s_hi, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (base_t, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (b_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        (s_op, "ROLE_COMPUTATION_OPERAND"),
                    ]
                    relations = [
                        (0, 1, "REL_BRACKET_NEXT"),
                        (3, 0, "REL_RATE_TO_THRESHOLD"),
                        (2, 3, "REL_BASE_TAX_ADD"),
                        (3, 4, "REL_SUBTRACTION_BASE"),
                    ]
                else:
                    labeled_spans = [
                        (s_lo, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (s_hi, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (base_t, "ROLE_COMPUTATION_OPERAND"),
                        (b_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                    ]
                    relations = [
                        (0, 1, "REL_BRACKET_NEXT"),
                        (3, 0, "REL_RATE_TO_THRESHOLD"),
                        (2, 3, "REL_SUBTRACTION_BASE"),
                    ]

            # -----------------------------------------------------------------
            # 42b. UNIT CONTRAST HARD-NEGATIVE TRAINING (V2 UNIT DISAMBIGUATION)
            # -----------------------------------------------------------------
            elif stype == "UNIT_CONTRAST_TEMPORAL_COUNT_STYLE":
                preamble = ""
                u_amt = random.choice(["$1,000", "$25,000", "₹50,000", "€4,500"])
                u_rate = random.choice(["5%", "12%", "18%", "25%"])
                u_dur = random.choice(["1 taxable year", "2 taxable years", "183 days", "6 months"])
                u_cnt = random.choice(["2 children", "3 dependents", "4 members"])
                if formulation in (0, 2):
                    body = (
                        f"Under section 1, an individual maintaining {u_cnt} during {u_dur} "
                        f"shall pay {u_rate} on income exceeding {u_amt}."
                    )
                else:
                    body = (
                        f"Where {u_cnt} qualify over a continuous period of {u_dur}, "
                        f"a rate of {u_rate} applies to taxable income over {u_amt}."
                    )
                op_kw = "taxable income" if "taxable income" in body else "income"
                labeled_spans = [
                    (u_cnt, "ROLE_COUNT_QUANTIFIER"),
                    (u_dur, "ROLE_DEADLINE_WINDOW"),
                    (u_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                    (op_kw, "ROLE_COMPUTATION_OPERAND"),
                    (u_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                ]
                relations = [
                    (2, 4, "REL_RATE_TO_THRESHOLD"),
                    (2, 3, "REL_SUBTRACTION_BASE"),
                ]

            # -----------------------------------------------------------------
            # 42c. CREDIT VS CAP EXACT-OPCODE COMPOSITION (V2)
            # -----------------------------------------------------------------
            elif stype == "CREDIT_VS_CAP_COMPOSED_STYLE":
                preamble = ""
                c_rate = random.choice(["28%", "22%", "15 percent"])
                c_floor = random.choice(["$36,900", "$24,000", "$18,500"])
                c_cred = random.choice(["$500", "$350", "$1,200"])
                c_cap = random.choice(["$10,000", "$15,000", "$8,500"])
                if formulation in (0, 2):
                    body = (
                        f"Tax equals {c_rate} of the excess of gross income over {c_floor}, "
                        f"reduced by a {c_cred} credit, and capped at {c_cap}."
                    )
                else:
                    body = (
                        f"Subject to a ceiling of {c_cap} and minus a {c_cred} credit, "
                        f"levy is {c_rate} on gross income exceeding {c_floor}."
                    )
                fl_lbl = f"over {c_floor}" if f"over {c_floor}" in body else f"exceeding {c_floor}"
                labeled_spans = [
                    (c_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                    ("gross income", "ROLE_COMPUTATION_OPERAND"),
                    (fl_lbl, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    (c_cred, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    (c_cap, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                ]
                relations = [
                    (0, 2, "REL_RATE_TO_THRESHOLD"),
                    (0, 1, "REL_SUBTRACTION_BASE"),
                    (3, 0, "REL_SUB_CREDIT"),
                    (4, 0, "REL_MIN_CAP"),
                ]

            # -----------------------------------------------------------------
            # 42d. BOOLEAN KINSHIP & DEFINED TERM GRAPH (V2)
            # -----------------------------------------------------------------
            elif stype == "BOOLEAN_KINSHIP_DEFINED_TERM_STYLE":
                preamble = ""
                d_term = random.choice(["head of a household", "qualifying spouse", "dependent beneficiary"])
                d_dur = random.choice(["1 taxable year", "more than half of the taxable year", "12 consecutive months"])
                if formulation in (0, 2):
                    body = (
                        f"(b) Definition of {d_term}. An individual is considered a {d_term} if such individual "
                        f"is not married and maintains a principal abode for {d_dur}."
                    )
                else:
                    body = (
                        f"The term {d_term} means any person who is not affiliated and maintains dependents for {d_dur}."
                    )
                p1 = "married" if "married" in body else "affiliated"
                p2 = "maintains a principal abode" if "maintains a principal abode" in body else "maintains dependents"
                labeled_spans = [
                    (d_term, "ROLE_DEFINED_TERM"),
                    ("not", "ROLE_PROHIBITION_GATE"),
                    (p1, "ROLE_AGENT_STATE_PRECONDITION"),
                    (p2, "ROLE_AGENT_STATE_PRECONDITION"),
                    (d_dur, "ROLE_DEADLINE_WINDOW"),
                ]
                relations = [
                    (1, 2, "REL_LOGICAL_NOT"),
                    (2, 3, "REL_LOGICAL_AND"),
                    (0, 2, "REL_DEFINED_TERM_OF"),
                    (4, 3, "REL_DEADLINE_TO_TRIGGER"),
                ]

            # -----------------------------------------------------------------
            # 43. TEMPORAL SUSPENSION WINDOW (NO RATE, SUSPENDED IN WINDOW)
            # -----------------------------------------------------------------
            elif stype == "TEMPORAL_SUSPENSION_OVERRIDE_STYLE":
                preamble = ""
                d_start, d_end = random.choice([
                    ("March 15, 2019", "October 1, 2023"),
                    ("June 30, 2021", "February 1, 2027"),
                    ("April 1, 2020", "November 30, 2024"),
                    ("July 1, 2022", "March 31, 2028"),
                    ("December 15, 2018", "January 15, 2025"),
                    ("May 10, 2020", "August 20, 2024"),
                    ("September 1, 2019", "December 20, 2023"),
                ])
                if formulation == 0:
                    body = (
                        f"This section shall not apply to any fiscal period beginning after {d_start}, "
                        f"and before {d_end}."
                    )
                    labeled_spans = [
                        ("shall not apply", "ROLE_PROHIBITION_GATE"),
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                    ]
                    relations = [(1, 0, "REL_CONDITION_TO_GATE"), (2, 0, "REL_SUNSET_TO_SECTION")]
                elif formulation == 1:
                    body = (
                        f"The operation of this sub-section stands suspended for any reporting year "
                        f"starting after {d_start} and ending before {d_end}."
                    )
                    labeled_spans = [
                        ("stands suspended", "ROLE_PROHIBITION_GATE"),
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                    ]
                    relations = [(1, 0, "REL_CONDITION_TO_GATE"), (2, 0, "REL_SUNSET_TO_SECTION")]
                elif formulation == 2:
                    body = (
                        f"For any period beginning after {d_start}, and before {d_end}, "
                        f"no allowance under this section shall be granted."
                    )
                    labeled_spans = [
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        ("no allowance under this section shall be granted", "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(0, 2, "REL_CONDITION_TO_GATE"), (1, 2, "REL_SUNSET_TO_SECTION")]
                else:
                    body = (
                        f"For cycles starting after {d_start} and ending before {d_end}, "
                        f"this provision shall be suspended and inoperative."
                    )
                    labeled_spans = [
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        ("shall be suspended and inoperative", "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(0, 2, "REL_CONDITION_TO_GATE"), (1, 2, "REL_SUNSET_TO_SECTION")]

            # -----------------------------------------------------------------
            # 43b. TEMPORAL SUBSTITUTION OVERRIDE WINDOW (MODALITY_COMPOUND_OR_SLAB)
            # -----------------------------------------------------------------
            elif stype == "TEMPORAL_SUBSTITUTION_OVERRIDE_STYLE":
                preamble = ""
                d_start, d_end = random.choice([
                    ("March 15, 2019", "October 1, 2023"),
                    ("June 30, 2021", "February 1, 2027"),
                    ("December 15, 2018", "January 15, 2025"),
                    ("May 10, 2020", "August 20, 2024"),
                    ("September 1, 2019", "December 20, 2023"),
                ])
                orig_amt, alt_amt = random.choice([
                    ("$9,200", "$24,000"),
                    ("€6,500", "€19,500"),
                    ("£8,400", "£21,000"),
                    ("$5,800", "$15,200"),
                    ("USD 11,000", "USD 27,500"),
                ])
                if formulation == 0:
                    body = (
                        f"In the case of a covered filer, the baseline threshold is {orig_amt}, "
                        f"except that for periods beginning after {d_start}, and before {d_end}, "
                        f"{alt_amt} shall be substituted for {orig_amt}."
                    )
                    labeled_spans = [
                        (orig_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        (alt_amt, "ROLE_EXCEPTION_CARVEOUT"),
                        (orig_amt, "ROLE_IGNORE"),
                    ]
                    relations = [
                        (3, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        (1, 3, "REL_CONDITION_TO_GATE"),
                        (2, 3, "REL_SUNSET_TO_SECTION"),
                    ]
                elif formulation == 1:
                    body = (
                        f"The basic statutory allowance equals {orig_amt}, provided that for any cycle "
                        f"starting after {d_start} and ending before {d_end}, {alt_amt} applies in lieu of {orig_amt}."
                    )
                    labeled_spans = [
                        (orig_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        (alt_amt, "ROLE_EXCEPTION_CARVEOUT"),
                        (orig_amt, "ROLE_IGNORE"),
                    ]
                    relations = [
                        (3, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                        (1, 3, "REL_CONDITION_TO_GATE"),
                        (2, 3, "REL_SUNSET_TO_SECTION"),
                    ]
                elif formulation == 2:
                    body = (
                        f"Except that for fiscal years beginning after {d_start}, and before {d_end}, "
                        f"{alt_amt} shall be substituted for {orig_amt}, the prescribed floor is {orig_amt}."
                    )
                    labeled_spans = [
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        (alt_amt, "ROLE_EXCEPTION_CARVEOUT"),
                        (orig_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (orig_amt, "ROLE_IGNORE"),
                    ]
                    relations = [
                        (2, 3, "REL_EXCEPTION_TO_PROHIBITION"),
                        (0, 2, "REL_CONDITION_TO_GATE"),
                        (1, 2, "REL_SUNSET_TO_SECTION"),
                    ]
                else:
                    body = (
                        f"For years starting after {d_start} and before {d_end}, "
                        f"{alt_amt} shall be substituted for {orig_amt}."
                    )
                    labeled_spans = [
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        (alt_amt, "ROLE_EXCEPTION_CARVEOUT"),
                        (orig_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [
                        (2, 3, "REL_EXCEPTION_TO_PROHIBITION"),
                        (0, 2, "REL_CONDITION_TO_GATE"),
                        (1, 2, "REL_SUNSET_TO_SECTION"),
                    ]

            # -----------------------------------------------------------------
            # 43c. QUALIFICATION THRESHOLD DEFINITION (METRIC >= THRESHOLD)
            # -----------------------------------------------------------------
            elif stype == "QUALIFICATION_THRESHOLD_DEFINITION_STYLE":
                preamble = ""
                q_actor = random.choice(["covered operator", "qualifying vendor", "registered carrier", "regulated distributor"])
                q_metric = random.choice(["freight charges", "commission payouts", "service fees", "contractor disbursements", "royalties"])
                q_th = random.choice(["$3,500", "$12,500", "$28,000", "€9,000", "£6,400"])
                if formulation == 0:
                    body = (
                        f"The term {q_actor} means, with respect to any calendar year, any person who "
                        f"during the calendar year or the preceding calendar year paid {q_metric} of {q_th} or more."
                    )
                elif formulation == 1:
                    body = (
                        f"In the case of commercial operations, the term {q_actor} means, with respect to any "
                        f"calendar year, any entity paying {q_metric} of {q_th} or more for such operations."
                    )
                elif formulation == 2:
                    body = (
                        f"Any entity that during the calendar year or prior calendar year disbursed {q_metric} of {q_th} or more "
                        f"is defined as a {q_actor}."
                    )
                else:
                    body = (
                        f"With respect to any calendar year, paying {q_metric} of {q_th} or more "
                        f"qualifies an entity as a {q_actor}."
                    )
                if v2:
                    labeled_spans = [
                        (q_metric, "ROLE_COMPUTATION_OPERAND"),
                        (f"{q_th} or more", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (q_actor, "ROLE_DEFINED_TERM"),
                    ]
                    relations = [
                        (0, 1, "REL_THRESHOLD_TO_TRANSACTION"),
                        (2, 1, "REL_DEFINED_TERM_OF"),
                    ]
                else:
                    labeled_spans = [
                        (q_metric, "ROLE_COMPUTATION_OPERAND"),
                        (f"{q_th} or more", "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [(0, 1, "REL_THRESHOLD_TO_TRANSACTION")]

            # -----------------------------------------------------------------
            # 43d. STATUS DISQUALIFICATION BAR (TWO-CLAUSE STATUS GATE)
            # -----------------------------------------------------------------
            elif stype == "STATUS_BAR_DISQUALIFICATION_STYLE":
                preamble = ""
                if formulation == 0:
                    body = (
                        "An applicant shall not be treated as an eligible beneficiary of a trust "
                        "if such applicant has executed a consolidated election with a co-holder for the reporting cycle."
                    )
                    labeled_spans = [
                        ("shall not be treated as", "ROLE_PROHIBITION_GATE"),
                        ("executed a consolidated election", "ROLE_AGENT_STATE_PRECONDITION"),
                    ]
                    relations = [(1, 0, "REL_CONDITION_TO_GATE")]
                elif formulation == 1:
                    body = (
                        "A participant legally released from an obligation under a decree of rescission or of separate covenant "
                        "shall not be considered as affiliated."
                    )
                    labeled_spans = [
                        ("decree of rescission or of separate covenant", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall not be considered as affiliated", "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(0, 1, "REL_CONDITION_TO_GATE")]
                elif formulation == 2:
                    body = (
                        "If such member has filed a joint waiver for the reporting cycle, "
                        "the member shall not be treated as an independent nominee."
                    )
                    labeled_spans = [
                        ("filed a joint waiver", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall not be treated as", "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(0, 1, "REL_CONDITION_TO_GATE")]
                else:
                    body = (
                        "Under a decree of annulment or of separate settlement, "
                        "an individual contractor shall not be considered as covered."
                    )
                    labeled_spans = [
                        ("decree of annulment or of separate settlement", "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall not be considered as covered", "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(0, 1, "REL_CONDITION_TO_GATE")]

            # -----------------------------------------------------------------
            # 44. MULTI-CLAUSE COMPOSED STATUTES (PER-CLAUSE MODALITY + DAG)
            # -----------------------------------------------------------------
            elif stype == "MULTI_CLAUSE_COMPOSED_STATUTE_STYLE":
                preamble = ""
                d_start, d_end = random.choice([
                    ("March 15, 2019", "October 1, 2023"),
                    ("June 30, 2021", "February 1, 2027"),
                    ("July 1, 2022", "March 31, 2028"),
                    ("December 15, 2018", "January 15, 2025"),
                    ("May 10, 2020", "August 20, 2024"),
                ])
                base_ded, sub_ded = random.choice([
                    ("$9,200", "$24,000"),
                    ("€6,500", "€19,500"),
                    ("£8,400", "£21,000"),
                ])
                c_amt = random.choice(["$120,000", "$240,000", "€95,000", "£75,000"])
                c_rate = random.choice(["4%", "7%", "9%", "12 percent"])
                doc_pair = formulation % 2
                if doc_pair == 0:
                    c1 = (
                        f"In the case of a qualified filer, the baseline deduction is {base_ded}, "
                        f"subject to an income phaseout of {c_rate} above {c_amt}."
                    )
                    c2 = (
                        f"Except that for taxable years beginning after {d_start} and before {d_end}, "
                        f"{sub_ded} shall apply as the adjusted deduction floor."
                    )
                    body = f"{c1} {c2}"
                    modality_override = "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB"
                    clauses = [
                        (c1, "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB"),
                        (c2, "MODALITY_DEONTIC_PROHIBITION"),
                    ]
                    labeled_spans = [
                        (base_ded, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (c_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        (f"above {c_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                        (sub_ded, "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [
                        (1, 2, "REL_RATE_TO_THRESHOLD"),
                        (5, 0, "REL_EXCEPTION_TO_PROHIBITION"),
                    ]
                else:
                    c1 = (
                        f"Any covered distributor paying commissions exceeding {c_amt} in any fiscal year "
                        f"shall deduct withholding at {c_rate} on the excess sum."
                    )
                    c2 = (
                        f"This sub-section shall not apply to any fiscal year beginning after {d_start} "
                        f"and before {d_end}."
                    )
                    body = f"{c1} {c2}"
                    modality_override = "MODALITY_COMPUTATIONAL_EXCESS"
                    clauses = [
                        (c1, "MODALITY_COMPUTATIONAL_EXCESS"),
                        (c2, "MODALITY_DEONTIC_PROHIBITION"),
                    ]
                    labeled_spans = [
                        (f"exceeding {c_amt}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (c_rate, "ROLE_STANDARD_STATUTORY_RATE"),
                        ("shall not apply", "ROLE_PROHIBITION_GATE"),
                        (d_start, "ROLE_TEMPORAL_CUTOVER_DATE"),
                        (d_end, "ROLE_TEMPORAL_SUNSET_DATE"),
                    ]
                    relations = [
                        (1, 0, "REL_RATE_TO_THRESHOLD"),
                        (3, 2, "REL_CONDITION_TO_GATE"),
                        (4, 2, "REL_SUNSET_TO_SECTION"),
                    ]

            # -----------------------------------------------------------------
            # 45. COMPOSITIONAL GENERAL LOGIC & QUANTIFIER ARCHETYPES
            # -----------------------------------------------------------------
            elif stype == "GENERAL_PERMISSION_CONDITIONAL_STYLE":
                preamble = ""
                g_actor = random.choice(["member", "operator", "delegate", "participant", "custodian", "agent"])
                g_act = random.choice(["execute a swap", "advance through the sector", "claim clearance", "initiate transfer", "activate access"])
                if formulation in [0, 2]:
                    cond_a = random.choice([
                        "the requester holds an active badge",
                        "the primary token is unlocked",
                        "the custodian holds a valid permit",
                    ])
                    cond_b = random.choice([
                        "no adjacent sector is blocked by a rival unit",
                        "neither gateway nor relay has previously faulted",
                        "every checkpoint is marked as cleared",
                    ])
                    if formulation == 0:
                        body = f"An authorized {g_actor} may {g_act} if {cond_a} and {cond_b}."
                        labeled_spans = [
                            (f"may {g_act}", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (cond_a, "ROLE_AGENT_STATE_PRECONDITION"),
                            (cond_b, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [(0, 1, "REL_ACTION_TO_PRECONDITION"), (0, 2, "REL_ACTION_TO_PRECONDITION"), (1, 2, "REL_LOGICAL_AND")]
                    else:
                        body = f"If {cond_a} and {cond_b}, an authorized {g_actor} may {g_act}."
                        labeled_spans = [
                            (cond_a, "ROLE_AGENT_STATE_PRECONDITION"),
                            (cond_b, "ROLE_AGENT_STATE_PRECONDITION"),
                            (f"may {g_act}", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ]
                        relations = [(2, 0, "REL_ACTION_TO_PRECONDITION"), (2, 1, "REL_ACTION_TO_PRECONDITION"), (0, 1, "REL_LOGICAL_AND")]
                else:
                    p1, p2 = random.choice([("the primary module", "the auxiliary module"), ("the lead seal", "the anchor unit"), ("the sender gate", "the receiver gate")])
                    sverb = random.choice(["moved", "triggered", "engaged", "faulted"])
                    pverb = random.choice(["cross", "traverse", "enter"])
                    qoperand = random.choice(["any tile contested by an adversary node", "any channel blocked by a rival unit", "any path flagged by a monitor"])
                    if formulation == 1:
                        body = (
                            f"A {g_actor} may {g_act} provided that neither {p1} nor {p2} has previously {sverb}, "
                            f"and {p1} shall not {pverb} {qoperand}."
                        )
                        labeled_spans = [
                            (f"may {g_act}", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (f"neither {p1} nor {p2} has previously {sverb}", "ROLE_AGENT_STATE_PRECONDITION"),
                            (f"shall not {pverb}", "ROLE_PROHIBITION_GATE"),
                            (qoperand, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [(0, 1, "REL_ACTION_TO_PRECONDITION"), (2, 3, "REL_LOGICAL_NOT"), (3, 2, "REL_QUANTIFIER_EXISTS")]
                    else:
                        body = (
                            f"Provided that neither {p1} nor {p2} has previously {sverb}, and {p1} shall not {pverb} {qoperand}, "
                            f"a {g_actor} may {g_act}."
                        )
                        labeled_spans = [
                            (f"neither {p1} nor {p2} has previously {sverb}", "ROLE_AGENT_STATE_PRECONDITION"),
                            (f"shall not {pverb}", "ROLE_PROHIBITION_GATE"),
                            (qoperand, "ROLE_AGENT_STATE_PRECONDITION"),
                            (f"may {g_act}", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ]
                        relations = [(3, 0, "REL_ACTION_TO_PRECONDITION"), (1, 2, "REL_LOGICAL_NOT"), (2, 1, "REL_QUANTIFIER_EXISTS")]

            elif stype == "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE":
                preamble = ""
                g_actor = random.choice(["operator", "node", "applicant", "custodian", "member"])
                q_item = random.choice(["milestone", "segment", "record", "credential", "checkpoint"])
                ok_adj = random.choice(["verified", "cleared", "validated", "signed", "approved"])
                bad_item = random.choice(["entry", "packet", "slot", "ticket", "signal"])
                bad_adj = random.choice(["flagged", "corrupted", "blocked", "expired", "revoked"])
                if formulation == 0:
                    q_phrase = f"every {q_item} is {ok_adj}"
                    e_phrase = f"no {bad_item} is {bad_adj}"
                    body = f"An {g_actor} may proceed if {q_phrase} and {e_phrase}."
                    labeled_spans = [
                        ("may proceed", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (q_phrase, "ROLE_AGENT_STATE_PRECONDITION"),
                        (e_phrase, "ROLE_AGENT_STATE_PRECONDITION"),
                    ]
                    relations = [(1, 0, "REL_QUANTIFIER_FORALL"), (0, 1, "REL_ACTION_TO_PRECONDITION"), (2, 0, "REL_QUANTIFIER_EXISTS"), (0, 2, "REL_LOGICAL_NOT")]
                elif formulation == 1:
                    q_phrase = f"all {q_item}s hold valid approval"
                    e_phrase = f"at least one inspector is {ok_adj}"
                    body = f"Release may be granted where {q_phrase} and {e_phrase}."
                    labeled_spans = [
                        ("may be granted", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (q_phrase, "ROLE_AGENT_STATE_PRECONDITION"),
                        (e_phrase, "ROLE_AGENT_STATE_PRECONDITION"),
                    ]
                    relations = [(1, 0, "REL_QUANTIFIER_FORALL"), (0, 1, "REL_ACTION_TO_PRECONDITION"), (2, 0, "REL_QUANTIFIER_EXISTS"), (0, 2, "REL_ACTION_TO_PRECONDITION")]
                elif formulation == 2:
                    modality_override = "MODALITY_DEONTIC_PROHIBITION"
                    q_phrase = f"each {q_item} is {ok_adj}"
                    e_phrase = f"none of the {bad_item}s is {bad_adj}"
                    body = f"No {g_actor} shall finalize transfer unless {q_phrase} or {e_phrase}."
                    labeled_spans = [
                        ("shall finalize transfer", "ROLE_PROHIBITION_GATE"),
                        (q_phrase, "ROLE_EXCEPTION_CARVEOUT"),
                        (e_phrase, "ROLE_EXCEPTION_CARVEOUT"),
                    ]
                    relations = [(1, 0, "REL_EXCEPTION_TO_PROHIBITION"), (0, 1, "REL_QUANTIFIER_FORALL"), (2, 0, "REL_EXCEPTION_TO_PROHIBITION"), (0, 2, "REL_QUANTIFIER_EXISTS")]
                else:
                    q_phrase = f"every {q_item} is {ok_adj}"
                    e_phrase = f"no {bad_item} is {bad_adj}"
                    body = f"Where {q_phrase} and {e_phrase}, an {g_actor} may proceed."
                    labeled_spans = [
                        (q_phrase, "ROLE_AGENT_STATE_PRECONDITION"),
                        (e_phrase, "ROLE_AGENT_STATE_PRECONDITION"),
                        ("may proceed", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                    ]
                    relations = [(0, 2, "REL_QUANTIFIER_FORALL"), (2, 0, "REL_ACTION_TO_PRECONDITION"), (1, 2, "REL_QUANTIFIER_EXISTS"), (2, 1, "REL_LOGICAL_NOT")]

            elif stype == "GENERAL_ENTITY_GRAPH_RELATION_STYLE":
                preamble = ""
                g_node = random.choice(["vertex", "node", "waypoint", "relay", "checkpoint"])
                g_edge = random.choice(["edge", "link", "bridge", "conduit"])
                cnt_w = random.choice(["two", "three", "four"])
                win_s = random.choice(["within three cycles", "within five steps", "within two rounds"])
                if formulation in [0, 2]:
                    pre_a = f"every predecessor {g_node} is verified"
                    pre_b = f"no {g_edge} along the route is disabled"
                    if formulation == 0:
                        body = f"A routing agent may dispatch traffic when {pre_a} and {pre_b}."
                        labeled_spans = [
                            ("may dispatch traffic", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                            (pre_a, "ROLE_AGENT_STATE_PRECONDITION"),
                            (pre_b, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [(1, 0, "REL_QUANTIFIER_FORALL"), (0, 1, "REL_ACTION_TO_PRECONDITION"), (2, 0, "REL_QUANTIFIER_EXISTS"), (0, 2, "REL_LOGICAL_NOT")]
                    else:
                        body = f"Provided that {pre_a} and {pre_b}, a routing agent may dispatch traffic."
                        labeled_spans = [
                            (pre_a, "ROLE_AGENT_STATE_PRECONDITION"),
                            (pre_b, "ROLE_AGENT_STATE_PRECONDITION"),
                            ("may dispatch traffic", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        ]
                        relations = [(0, 2, "REL_QUANTIFIER_FORALL"), (2, 0, "REL_ACTION_TO_PRECONDITION"), (1, 2, "REL_QUANTIFIER_EXISTS"), (2, 1, "REL_LOGICAL_NOT")]
                else:
                    modality_override = "MODALITY_DEONTIC_PROHIBITION"
                    blk_c = f"any intervening {g_node} is blocked"
                    esc_c = f"at least {cnt_w} backup {g_edge}s remain active"
                    if formulation == 1:
                        body = f"No controller shall route packets if {blk_c} unless {esc_c} {win_s}."
                        labeled_spans = [
                            ("shall route packets", "ROLE_PROHIBITION_GATE"),
                            (blk_c, "ROLE_AGENT_STATE_PRECONDITION"),
                            (esc_c, "ROLE_EXCEPTION_CARVEOUT"),
                            (win_s, "ROLE_DEADLINE_WINDOW"),
                        ]
                        relations = [(1, 0, "REL_QUANTIFIER_EXISTS"), (0, 1, "REL_CONDITION_TO_GATE"), (2, 0, "REL_EXCEPTION_TO_PROHIBITION"), (3, 2, "REL_DEADLINE_TO_TRIGGER")]
                    else:
                        body = f"Unless {esc_c} {win_s}, no controller shall route packets if {blk_c}."
                        labeled_spans = [
                            (esc_c, "ROLE_EXCEPTION_CARVEOUT"),
                            (win_s, "ROLE_DEADLINE_WINDOW"),
                            ("shall route packets", "ROLE_PROHIBITION_GATE"),
                            (blk_c, "ROLE_AGENT_STATE_PRECONDITION"),
                        ]
                        relations = [(0, 2, "REL_EXCEPTION_TO_PROHIBITION"), (1, 0, "REL_DEADLINE_TO_TRIGGER"), (3, 2, "REL_QUANTIFIER_EXISTS"), (2, 3, "REL_CONDITION_TO_GATE")]

            elif stype == "GENERAL_AGGREGATE_COLLECTION_STYLE":
                preamble = ""
                coll_op = random.choice(["qualifying claims", "verified allocations", "active entries", "eligible credits"])
                th_v = random.choice(["$15,000", "$25,000", "€18,000", "£12,000"])
                cap_v = random.choice(["$40,000", "$60,000", "€45,000", "£30,000"])
                r_v = random.choice(["10%", "15%", "20 percent", "25 per cent"])
                if formulation == 0:
                    body = f"Where the sum of {coll_op} exceeds {th_v}, an assessment at {r_v} applies on the excess capped at {cap_v}."
                    labeled_spans = [
                        (coll_op, "ROLE_COMPUTATION_OPERAND"),
                        (f"exceeds {th_v}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (r_v, "ROLE_STANDARD_STATUTORY_RATE"),
                        (cap_v, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [(1, 0, "REL_AGGREGATE_SUM"), (2, 1, "REL_RATE_TO_THRESHOLD"), (3, 2, "REL_CAP_CEILING")]
                elif formulation == 1:
                    body = f"An obligation equals {r_v} of the aggregate of {coll_op} exceeding {th_v}, capped at {cap_v}."
                    labeled_spans = [
                        (r_v, "ROLE_STANDARD_STATUTORY_RATE"),
                        (coll_op, "ROLE_COMPUTATION_OPERAND"),
                        (f"exceeding {th_v}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (cap_v, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [(2, 1, "REL_AGGREGATE_SUM"), (0, 2, "REL_RATE_TO_THRESHOLD"), (3, 0, "REL_CAP_CEILING")]
                elif formulation == 2:
                    body = f"Capped at {cap_v}, an assessment at {r_v} shall be levied where the total of {coll_op} exceeds {th_v}."
                    labeled_spans = [
                        (cap_v, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        (r_v, "ROLE_STANDARD_STATUTORY_RATE"),
                        (coll_op, "ROLE_COMPUTATION_OPERAND"),
                        (f"exceeds {th_v}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                    ]
                    relations = [(3, 2, "REL_AGGREGATE_SUM"), (1, 3, "REL_RATE_TO_THRESHOLD"), (0, 1, "REL_CAP_CEILING")]
                else:
                    body = f"Provided that the aggregate of {coll_op} exceeds {th_v}, the fee at {r_v} is capped at {cap_v}."
                    labeled_spans = [
                        (coll_op, "ROLE_COMPUTATION_OPERAND"),
                        (f"exceeds {th_v}", "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        (r_v, "ROLE_STANDARD_STATUTORY_RATE"),
                        (cap_v, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                    ]
                    relations = [(1, 0, "REL_AGGREGATE_SUM"), (2, 1, "REL_RATE_TO_THRESHOLD"), (3, 2, "REL_CAP_CEILING")]

            elif stype == "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE":
                preamble = ""
                op1 = random.choice(["gross assessment", "total intake", "measured load"])
                op2 = random.choice(["standard allowance", "exempt reserve", "baseline offset"])
                cap_amt = random.choice(["$12,000", "$18,500", "€14,000", "£9,500"])
                floor_amt = random.choice(["$3,500", "$5,000", "€4,000", "£2,500"])
                r_dag = random.choice(["15%", "25%", "30 percent"])
                if formulation in [0, 2]:
                    body = (
                        f"The net charge equals the lesser of {r_dag} of {op1} minus {op2}, or {cap_amt}."
                        if formulation == 0 else
                        f"Equal to the lesser of {cap_amt} or {r_dag} of {op1} minus {op2}, the net charge shall apply."
                    )
                    if formulation == 0:
                        labeled_spans = [
                            (r_dag, "ROLE_STANDARD_STATUTORY_RATE"),
                            (op1, "ROLE_COMPUTATION_OPERAND"),
                            (op2, "ROLE_COMPUTATION_OPERAND"),
                            (cap_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                        ]
                        relations = [(1, 2, "REL_SUBTRACTION_BASE"), (0, 1, "REL_SUBTRACTION_BASE"), (0, 3, "REL_COMBINATOR_MIN")]
                    else:
                        labeled_spans = [
                            (cap_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                            (r_dag, "ROLE_STANDARD_STATUTORY_RATE"),
                            (op1, "ROLE_COMPUTATION_OPERAND"),
                            (op2, "ROLE_COMPUTATION_OPERAND"),
                        ]
                        relations = [(2, 3, "REL_SUBTRACTION_BASE"), (1, 2, "REL_SUBTRACTION_BASE"), (0, 1, "REL_COMBINATOR_MIN")]
                else:
                    body = (
                        f"The required reserve shall be the greater of {r_dag} of {op1} minus {op2}, or {floor_amt}, capped at {cap_amt}."
                        if formulation == 1 else
                        f"Capped at {cap_amt}, the greater of {floor_amt} or {r_dag} of {op1} minus {op2} constitutes the required reserve."
                    )
                    if formulation == 1:
                        labeled_spans = [
                            (r_dag, "ROLE_STANDARD_STATUTORY_RATE"),
                            (op1, "ROLE_COMPUTATION_OPERAND"),
                            (op2, "ROLE_COMPUTATION_OPERAND"),
                            (floor_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                            (cap_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                        ]
                        relations = [(1, 2, "REL_SUBTRACTION_BASE"), (0, 1, "REL_SUBTRACTION_BASE"), (0, 3, "REL_COMBINATOR_MAX"), (4, 0, "REL_CAP_CEILING")]
                    else:
                        labeled_spans = [
                            (cap_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD"),
                            (floor_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD"),
                            (r_dag, "ROLE_STANDARD_STATUTORY_RATE"),
                            (op1, "ROLE_COMPUTATION_OPERAND"),
                            (op2, "ROLE_COMPUTATION_OPERAND"),
                        ]
                        relations = [(3, 4, "REL_SUBTRACTION_BASE"), (2, 3, "REL_SUBTRACTION_BASE"), (1, 2, "REL_COMBINATOR_MAX"), (0, 2, "REL_CAP_CEILING")]

            elif stype == "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE":
                preamble = ""
                g_actor = random.choice(["carrier", "operator", "custodian", "member", "applicant"])
                p_gate = random.choice(["shall dispatch cargo", "shall exit confinement", "shall unlock the account", "shall draw settlement funds"])
                c_pre = random.choice(["exceeding the threshold", "under an active freeze", "flagged for review"])
                esc_c = random.choice([
                    "paying a fee of $50 or completing dual verification",
                    "posting a bond of $100 or obtaining written clearance",
                    "depositing a fine of $75 or executing matching confirmations",
                ])
                w_dl = random.choice(["within 48 hours", "within three turns", "within five cycles"])
                if formulation == 0:
                    body = f"No {g_actor} {c_pre} {p_gate} unless {esc_c} {w_dl}."
                    labeled_spans = [
                        (c_pre, "ROLE_AGENT_STATE_PRECONDITION"),
                        (p_gate, "ROLE_PROHIBITION_GATE"),
                        (esc_c, "ROLE_EXCEPTION_CARVEOUT"),
                        (w_dl, "ROLE_DEADLINE_WINDOW"),
                    ]
                    relations = [(0, 1, "REL_CONDITION_TO_GATE"), (2, 1, "REL_EXCEPTION_TO_PROHIBITION"), (3, 2, "REL_DEADLINE_TO_TRIGGER")]
                elif formulation == 1:
                    body = f"A {g_actor} {c_pre} shall transfer to escrow and {p_gate.replace('shall ', 'shall not ')}: Provided that the {g_actor} may resume upon {esc_c} {w_dl}."
                    neg_gate = p_gate.replace("shall ", "shall not ")
                    labeled_spans = [
                        (c_pre, "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall transfer to escrow", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (neg_gate, "ROLE_PROHIBITION_GATE"),
                        (f"upon {esc_c}", "ROLE_EXCEPTION_CARVEOUT"),
                        (w_dl, "ROLE_DEADLINE_WINDOW"),
                    ]
                    relations = [(1, 0, "REL_ACTION_TO_PRECONDITION"), (2, 1, "REL_LOGICAL_AND"), (3, 2, "REL_EXCEPTION_TO_PROHIBITION"), (4, 3, "REL_DEADLINE_TO_TRIGGER")]
                elif formulation == 2:
                    body = f"Unless {esc_c} {w_dl}, no {g_actor} {c_pre} {p_gate}."
                    labeled_spans = [
                        (esc_c, "ROLE_EXCEPTION_CARVEOUT"),
                        (w_dl, "ROLE_DEADLINE_WINDOW"),
                        (c_pre, "ROLE_AGENT_STATE_PRECONDITION"),
                        (p_gate, "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(2, 3, "REL_CONDITION_TO_GATE"), (0, 3, "REL_EXCEPTION_TO_PROHIBITION"), (1, 0, "REL_DEADLINE_TO_TRIGGER")]
                else:
                    neg_gate = p_gate.replace("shall ", "shall not ")
                    body = f"Provided that the {g_actor} may resume upon {esc_c} {w_dl}, a {g_actor} {c_pre} shall transfer to escrow and {neg_gate}."
                    labeled_spans = [
                        (f"upon {esc_c}", "ROLE_EXCEPTION_CARVEOUT"),
                        (w_dl, "ROLE_DEADLINE_WINDOW"),
                        (c_pre, "ROLE_AGENT_STATE_PRECONDITION"),
                        ("shall transfer to escrow", "ROLE_MANDATORY_ACTION_OBLIGATION"),
                        (neg_gate, "ROLE_PROHIBITION_GATE"),
                    ]
                    relations = [(3, 2, "REL_ACTION_TO_PRECONDITION"), (4, 3, "REL_LOGICAL_AND"), (0, 4, "REL_EXCEPTION_TO_PROHIBITION"), (1, 0, "REL_DEADLINE_TO_TRIGGER")]

            # -----------------------------------------------------------------
            # 46. MODALITY_ABSTAIN_UNRESOLVED (NON-LEGAL / AMBIGUOUS FRAGMENTS)
            # -----------------------------------------------------------------
            elif stype == "ABSTAIN_UNRESOLVED_STYLE":
                preamble = ""
                if formulation in [0, 1]:
                    body = random.choice(abstain_train_texts)
                else:
                    body = random.choice(abstain_val_texts)
                labeled_spans = []
                relations = []

            # Complete text with preamble
            text = preamble + body

            # Hard Negative Injection (50% chance): add section number / gazette reference labeled ROLE_IGNORE
            if stype != "ABSTAIN_UNRESOLVED_STYLE" and random.random() < 0.5 and "Section 393(1)" not in body:
                hard_negatives = [
                    "Section 164 of Act", "clause (a) of sub-section (2)", "Act No. 18 of 2013",
                    "published on 30th August, 2013", "vide notification No. 12/2020", "Part II Section 3",
                    "Section 393(1), Table 1, Sl. No. 8(ii)", "Section 52(1) of the Act",
                    "Rule 42 and Rule 43", "Table 4(B)(1)",
                    "Section 397(2) Proviso", "Section 536(2)", "Section 16(2)", "Rule 48(4)"
                ]
                neg_span = random.choice(hard_negatives)
                neg_prefix = random.choice(["Reference:", "Continues under", "Read with"])
                text = text + f" {neg_prefix} {neg_span}."
                labeled_spans.append((neg_span, "ROLE_IGNORE"))

            # Defensive verification: Ensure every labeled span is physically present in text
            valid_labeled_spans = []
            span_idx_map = {}
            for original_idx, (s_txt, r_name) in enumerate(labeled_spans):
                if s_txt in text:
                    span_idx_map[original_idx] = len(valid_labeled_spans)
                    valid_labeled_spans.append((s_txt, r_name))

            # Remap relation indices defensively
            remapped_relations = []
            for src, tgt, rel_type in relations:
                if src in span_idx_map and tgt in span_idx_map:
                    remapped_relations.append((span_idx_map[src], span_idx_map[tgt], rel_type))

            modality_name = modality_override or ARCHETYPE_MODALITY_MAP.get(stype, "MODALITY_ABSTAIN_UNRESOLVED")
            sample_item = {
                "stype": stype,
                "modality_name": modality_name,
                "modality_id": MODALITY2ID[modality_name],
                "formulation": formulation,
                "text": text,
                "labeled_spans": valid_labeled_spans,
                "relations": remapped_relations,
                "clauses": clauses,
            }

            if formulation in [0, 1]:
                train_samples.append(sample_item)
            else:
                val_samples.append(sample_item)

        return train_samples, val_samples

    def process_samples(self, raw_samples: list[dict], is_train: bool = False, v2: bool = False) -> list[dict]:
        processed = []
        abstain_id = MODALITY2ID["MODALITY_ABSTAIN_UNRESOLVED"]
        role2id = V2_ROLE2ID if v2 else ROLE2ID
        id2role = V2_ID2ROLE if v2 else ID2ROLE
        rel2id = V2_REL2ID if v2 else REL2ID
        bio2id = V2_BIO2ID if v2 else BIO2ID
        role_to_bio = V2_ROLE_TO_BIO_BASE if v2 else ROLE_TO_BIO_BASE
        for sample in raw_samples:
            text = sample["text"]
            tokens, offsets = tokenize_with_offsets(text)
            if not tokens:
                continue
            token_ids = self.tokens_to_ids(tokens)
            canon_tokens = self._canonical_seq(tokens)
            suffix_ids = [
                SUFFIX2ID.get(canonical_suffix(tok, ctok), SUFFIX2ID["<NONE>"])
                for tok, ctok in zip(tokens, canon_tokens)
            ]
            modality_id = sample.get("modality_id")
            if modality_id is None:
                modality_name = ARCHETYPE_MODALITY_MAP.get(sample.get("stype", "194Q_STYLE"), "MODALITY_COMPUTATIONAL_EXCESS")
                modality_id = MODALITY2ID[modality_name]

            n_labeled = len(sample["labeled_spans"])
            swu_list = sample.get("spans_with_units") or []
            resolved_spans = [None] * n_labeled
            occupied_tokens = set()

            # Resolve longest spans first so embedded duplicate substrings find unoccupied occurrences
            resolution_order = sorted(
                range(n_labeled),
                key=lambda i: (-len(sample["labeled_spans"][i][0]), i)
            )
            for orig_i in resolution_order:
                span_text, role_name = sample["labeled_spans"][orig_i]
                explicit_unit = swu_list[orig_i][2] if orig_i < len(swu_list) and len(swu_list[orig_i]) >= 3 else None
                chosen_tok_range = None
                search_pos = 0
                while True:
                    idx = text.find(span_text, search_pos)
                    if idx == -1:
                        break
                    char_start = idx
                    char_end = idx + len(span_text)
                    tok_start, tok_end = char_span_to_token_span(char_start, char_end, offsets)
                    if all(t not in occupied_tokens for t in range(tok_start, tok_end + 1)):
                        chosen_tok_range = (tok_start, tok_end)
                        break
                    search_pos = idx + 1

                if chosen_tok_range is None:
                    idx = text.find(span_text)
                    if idx != -1:
                        chosen_tok_range = char_span_to_token_span(idx, idx + len(span_text), offsets)

                if chosen_tok_range is not None:
                    r_id = role2id.get(role_name, role2id["ROLE_IGNORE"])
                    resolved_spans[orig_i] = (chosen_tok_range, r_id, span_text, role_name, explicit_unit)
                    for t in range(chosen_tok_range[0], chosen_tok_range[1] + 1):
                        occupied_tokens.add(t)

            span_boundaries = []
            role_labels = []
            unit_labels = []
            kept_span_texts = []
            orig_to_kept = {}
            for orig_i, entry in enumerate(resolved_spans):
                if entry is not None:
                    orig_to_kept[orig_i] = len(span_boundaries)
                    span_boundaries.append(entry[0])
                    role_labels.append(entry[1])
                    kept_span_texts.append(entry[2])
                    if v2:
                        u_name = entry[4] if entry[4] in UNIT2ID else infer_span_unit_v2(entry[2], entry[3])
                        unit_labels.append(UNIT2ID[u_name])

            if not span_boundaries and modality_id != abstain_id:
                continue

            # Strictly assert zero overlapping token spans across the sample
            for i in range(len(span_boundaries)):
                s_i, e_i = span_boundaries[i]
                for j in range(i + 1, len(span_boundaries)):
                    s_j, e_j = span_boundaries[j]
                    if not (e_i < s_j or e_j < s_i):
                        raise ValueError(
                            f"Overlapping labeled spans in archetype '{sample.get('stype')}' "
                            f"(formulation {sample.get('formulation')}): {span_boundaries[i]} vs {span_boundaries[j]} in text: {text}"
                        )

            num_spans = len(span_boundaries)
            rel_matrix = [[rel2id["REL_NO_EDGE"] for _ in range(num_spans)] for _ in range(num_spans)]
            stype_name = sample.get("stype", "")
            for src_idx, tgt_idx, rel_name in sample["relations"]:
                if src_idx in orig_to_kept and tgt_idx in orig_to_kept:
                    k_s = orig_to_kept[src_idx]
                    k_t = orig_to_kept[tgt_idx]
                    eff_rel = (
                        _remap_v2_relation(rel_name, kept_span_texts[k_s], kept_span_texts[k_t], text, stype_name)
                        if v2 else rel_name
                    )
                    rel_matrix[k_s][k_t] = rel2id.get(eff_rel, rel2id["REL_NO_EDGE"])

            # Generate token-level BIO tags from labeled spans
            bio_labels = ["O"] * len(tokens)
            for (tok_s, tok_e), r_id in zip(span_boundaries, role_labels):
                role_name = id2role[r_id]
                base_tag = role_to_bio.get(role_name, "PRECONDITION")
                if base_tag == "IGNORE":
                    continue
                bio_labels[tok_s] = f"B-{base_tag}"
                for k in range(tok_s + 1, tok_e + 1):
                    bio_labels[k] = f"I-{base_tag}"

            bio_ids = [bio2id.get(tag, 0) for tag in bio_labels]

            # Balanced Open-Class Noun, Verb & OOD Lexical Slot Dropout (applied to training set only)
            # Eliminates any spurious slot-density cue for ABSTAIN while exposing non-ABSTAIN clauses
            # to the same 8%-32% <LEGAL_NOUN>/<LEGAL_VERB> density produced by OOD inference.
            PROTECTED_STATUTORY_MARKERS = {
                # Modals, auxiliaries, negation & conditional/logical connectives
                "shall", "may", "must", "should", "cannot", "could", "would", "will",
                "has", "have", "had", "been", "being", "does", "did", "done",
                "not", "neither", "nor", "none", "provided", "proviso", "unless", "except",
                "excluding", "excluded", "exclude", "notwithstanding", "subject", "until",
                "where", "while", "when", "whenever", "whose", "whom", "which", "what",
                "whether", "that", "this", "these", "those", "such", "same",
                "both", "either", "every", "each", "only", "some", "many", "much", "several",
                "into", "onto", "from", "with", "without", "within", "between", "among",
                "after", "before", "during", "since", "through", "against", "across",
                "substituted", "substitution", "lieu", "suspended", "suspension",
                "effective", "omitted", "omit", "calendar", "financial", "previous", "period", "periods",
                # Comparison, quantifier & arithmetic operators
                "exceed", "exceeds", "exceeded", "exceeding", "deduct", "deducted", "collect", "collected",
                "pay", "paying", "posting", "depositing", "mandates", "requires", "directs", "obligates", "authorises",
                "source", "supplies", "consideration", "reduced", "increased", "scaled",
                "lesser", "lower", "smaller", "greater", "higher", "larger", "less", "more", "most", "least", "fewer",
                "equal", "equals", "equally", "excess", "surplus", "over", "above", "below", "under", "beyond",
                "capped", "cap", "ceiling", "floor", "bracket", "brackets", "tier", "minus", "plus", "times",
                "sum", "combined", "aggregate", "count", "total", "mean", "average", "part", "portion", "amount", "value",
                "beginning", "ending", "earlier", "whichever", "otherwise", "allowable", "previously", "already", "directly",
                "respectively", "term", "means", "include", "includes", "including",
                # Currency, rate & temporal unit markers
                "per", "cent", "centum", "percent", "percentage", "rupee", "rupees", "lakh", "lakhs", "lac", "lacs",
                "crore", "crores", "thousand", "million", "billion", "dollar", "dollars", "usd", "inr",
                "euro", "euros", "eur", "pound", "pounds", "gbp", "paisa", "paise", "cents", "pence",
                "days", "weeks", "months", "years", "annum", "hours", "turns", "turn", "rounds", "round", "cycles", "cycle", "steps", "step",
                # Calendar month & number words
                "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "half", "third", "fourth", "quarter",
                # Citation/header & specialized financial GL markers
                "section", "subsection", "rule", "clause", "act", "notification", "circular", "table", "schedule", "form",
                "tax", "credit", "profit", "loss", "comprehensive", "income", "equity", "carrying", "pan", "resident",
                "fine", "bond", "fee", "deposit", "escrow",
            }
            PROTECTED_ABSTAIN_MARKERS = {
                "what", "where", "when", "which", "who", "whom", "whose", "why", "how",
                "is", "are", "was", "were", "can", "could", "would", "should", "will",
                "do", "does", "did", "has", "have", "had", "there", "here", "this", "that", "these", "those",
                "any", "some", "many", "much", "please", "from", "into", "before", "after",
                "with", "without", "for", "on", "in", "at", "to", "the", "a", "an", "of", "and", "or",
            }
            MODAL_AUX_CONTEXT = {
                "shall", "may", "must", "should", "cannot", "can", "will", "would", "could",
                "to", "not", "has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
            }
            DET_PREP_CONTEXT = {
                "the", "a", "an", "this", "that", "these", "those", "whose", "any", "every", "each",
                "all", "no", "none", "some", "neither", "either", "nor", "or",
                "of", "for", "in", "by", "with", "from", "on", "at", "against",
            }
            VERB_SUFFIXES = ("ed", "ing", "ize", "ise", "ate", "ify")
            if is_train:
                slot_dropped_ids = []
                legal_noun_id = self.vocab.get("<LEGAL_NOUN>", 2)
                legal_verb_id = self.vocab.get("<LEGAL_VERB>", 3)
                unk_id = self.vocab.get("<UNK>", 1)

                # Identical slot-dropout intensity schedules across ABSTAIN and NON-ABSTAIN
                abstain_drop_prob = random.choice([0.08, 0.14, 0.22])
                rule_drop_prob = random.choice([0.08, 0.18, 0.30])
                ignore_token_positions = {
                    t_i
                    for (t_s, t_e), r_id in zip(span_boundaries, role_labels)
                    if id2role.get(r_id) == "ROLE_IGNORE"
                    for t_i in range(t_s, t_e + 1)
                }

                for tok_idx, (tok, bio_tag, tid) in enumerate(zip(tokens, bio_labels, token_ids)):
                    low = tok.lower()
                    prev_low = tokens[tok_idx - 1].lower() if tok_idx > 0 else ""
                    prev_prev_low = tokens[tok_idx - 2].lower() if tok_idx > 1 else ""
                    is_verb_ctx = (
                        low in OPEN_CLASS_VERBS
                        or prev_low in MODAL_AUX_CONTEXT
                        or (
                            prev_prev_low in MODAL_AUX_CONTEXT
                            and (prev_low.endswith("ly") or prev_low in ("not", "already", "previously", "directly"))
                        )
                        or (low.endswith(VERB_SUFFIXES) and prev_low not in DET_PREP_CONTEXT)
                    )

                    if modality_id == abstain_id:
                        if tok.isalpha() and len(tok) >= 4 and low not in PROTECTED_ABSTAIN_MARKERS and random.random() < abstain_drop_prob:
                            if random.random() < 0.08:
                                slot_dropped_ids.append(unk_id)
                            elif is_verb_ctx:
                                slot_dropped_ids.append(legal_verb_id)
                            else:
                                slot_dropped_ids.append(legal_noun_id)
                        else:
                            slot_dropped_ids.append(tid)
                    elif low in PROTECTED_STATUTORY_MARKERS or tok_idx in ignore_token_positions:
                        slot_dropped_ids.append(tid)
                    elif low in OPEN_CLASS_NOUNS and random.random() < max(0.28, rule_drop_prob):
                        slot_dropped_ids.append(legal_noun_id)
                    elif low in OPEN_CLASS_VERBS and random.random() < max(0.28, rule_drop_prob):
                        slot_dropped_ids.append(legal_verb_id)
                    elif tok.isalpha() and len(tok) >= 4 and random.random() < rule_drop_prob:
                        if random.random() < 0.05:
                            slot_dropped_ids.append(unk_id)
                        elif is_verb_ctx:
                            slot_dropped_ids.append(legal_verb_id)
                        else:
                            slot_dropped_ids.append(legal_noun_id)
                    else:
                        slot_dropped_ids.append(tid)
                token_ids = slot_dropped_ids

            clause_spans = []
            clause_modality_ids = []
            raw_clauses = sample.get("clauses") or []
            if raw_clauses:
                c_pos = 0
                for c_entry in raw_clauses:
                    if isinstance(c_entry, (tuple, list)):
                        c_txt, c_mod = c_entry[0], c_entry[1]
                    else:
                        c_txt, c_mod = str(c_entry), ID2MODALITY[modality_id]
                    c_idx = text.find(c_txt, c_pos)
                    if c_idx == -1:
                        c_idx = text.find(c_txt)
                    if c_idx != -1:
                        c_tok_s, c_tok_e = char_span_to_token_span(c_idx, c_idx + len(c_txt), offsets)
                        clause_spans.append((c_tok_s, c_tok_e))
                        clause_modality_ids.append(MODALITY2ID[c_mod])
                        c_pos = c_idx + len(c_txt)

            item_dict = {
                "stype": sample.get("stype", "194Q_STYLE"),
                "modality_id": modality_id,
                "modality_name": ID2MODALITY[modality_id],
                "formulation": sample.get("formulation", 0),
                "text": text,
                "token_ids": token_ids,
                "suffix_ids": suffix_ids,
                "bio_ids": bio_ids,
                "span_boundaries": span_boundaries,
                "role_labels": role_labels,
                "rel_matrix": rel_matrix,
                "clause_spans": clause_spans,
                "clause_modality_ids": clause_modality_ids,
            }
            if v2:
                item_dict["unit_labels"] = unit_labels
            processed.append(item_dict)
        return processed


def generate_synthetic_dataset(num_samples: int = 60, seed: int = 42, v2: bool = False) -> dict:
    """
    Generates an in-memory synthetic dataset for training or smoke testing.
    Does not require external files or disk writes.
    """
    generator = ComprehensiveDatasetGenerator(seed=seed)
    raw_train, raw_val = generator.generate_all_samples(num_samples=num_samples, v2=v2)
    all_texts = [s["text"] for s in raw_train]
    generator.build_vocab(all_texts, max_vocab=4096)
    train_data = generator.process_samples(raw_train, is_train=True, v2=v2)
    val_data = generator.process_samples(raw_val, is_train=False, v2=v2)
    payload = {
        "vocab": generator.vocab,
        "modalities": RULE_MODALITIES,
        "train": train_data,
        "val": val_data,
    }
    if v2:
        payload["architecture_version"] = "v2"
        payload["span_units"] = SPAN_UNITS
        payload["roles"] = V2_SEMANTIC_ROLES
        payload["relations"] = V2_RELATION_TYPES
        payload["bio_tags"] = V2_BIO_TAGS
    return payload


def create_comprehensive_dataset(
    output_path: str = "checkpoints/dataset.pt",
    num_samples: int = 4000,
    seed: int = 42,
    v2: bool = False,
):
    print(f"[DATASET] Generating comprehensive statutory dataset ({num_samples} samples, v2={v2})...")
    generator = ComprehensiveDatasetGenerator(seed=seed)

    raw_train, raw_val = generator.generate_all_samples(num_samples=num_samples, v2=v2)
    all_texts = [s["text"] for s in raw_train]

    # Optional local OKF domain summaries if present
    possible_roots = ["enterprise_okf", "okf", "data"]
    for pr in possible_roots:
        if os.path.exists(pr):
            for root, _, files in os.walk(pr):
                for f in files:
                    if f == "section.json":
                        sec_path = os.path.join(root, f)
                        try:
                            with open(sec_path, "r", encoding="utf-8") as fh:
                                txt = json.load(fh).get("plainTextSummary")
                                if txt:
                                    all_texts.append(txt)
                        except Exception:
                            pass

    generator.build_vocab(all_texts, max_vocab=16384)
    print(f"[DATASET] Total Unified Vocabulary Size: {len(generator.vocab)} tokens")

    train_data = generator.process_samples(raw_train, is_train=True, v2=v2)
    val_data = generator.process_samples(raw_val, is_train=False, v2=v2)

    print(f"[DATASET] Split -> Train: {len(train_data)} | Val: {len(val_data)}")

    payload = {
        "vocab": generator.vocab,
        "modalities": RULE_MODALITIES,
        "train": train_data,
        "val": val_data,
    }
    if v2:
        payload["architecture_version"] = "v2"
        payload["span_units"] = SPAN_UNITS
        payload["roles"] = V2_SEMANTIC_ROLES
        payload["relations"] = V2_RELATION_TYPES
        payload["bio_tags"] = V2_BIO_TAGS
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(payload, output_path)
    print(f"[DATASET] Saved successfully to {output_path} ({os.path.getsize(output_path):,} bytes)")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Dual-Stream statutory dataset")
    parser.add_argument("--v2", action="store_true", help="Generate Model v2 dataset with span unit labels & v2 opcodes")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path")
    parser.add_argument("--num-samples", type=int, default=4000)
    args = parser.parse_args()
    out_p = args.output or ("checkpoints/v2_dataset.pt" if args.v2 else "checkpoints/dataset.pt")
    create_comprehensive_dataset(output_path=out_p, num_samples=args.num_samples, v2=args.v2)
