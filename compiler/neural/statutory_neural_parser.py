"""
=============================================================================
MUDRAFORGE NEURAL STATUTORY COMPILER (MUDRA-DECOMPILER-0)
=============================================================================

A dedicated, built-from-scratch neuro-symbolic semantic parser in pure PyTorch.
Translates natural-language legal text into structured USVM AST parameters
using contextual self-attention, token-level BIO sequence tagging,
span boundary representation (SBR) pooling, semantic role labeling (SRL),
and deep biaffine relation scoring.

Zero HuggingFace/LLM dependencies. Pure PyTorch tensor architecture.
Specification: docs/STATUTORY_AST_COMPILER_SPECIFICATION.md
"""

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<LEGAL_NOUN>": 2,
    "<LEGAL_VERB>": 3,
    "<LEGAL_ENTITY>": 4,
    "<NUM_INT>": 5,
    "<NUM_DEC>": 6,
    "<YEAR_4D>": 7,
    "<CURRENCY>": 8,
    "<PERCENT>": 9,
}

_CURRENCY_SURFACE_TOKENS = {
    "$", "₹", "€", "£", "inr", "usd", "eur", "gbp", "rs", "rs.",
    "rupee", "rupees", "dollar", "dollars", "euro", "euros", "pound", "pounds",
    "paisa", "paise", "cents",
}
_CURRENCY_CONTEXT_TOKENS = _CURRENCY_SURFACE_TOKENS | {
    "cent", "lakh", "lakhs", "lac", "lacs", "crore", "crores",
    "thousand", "million", "billion", "%", "percent", "percentage",
    "per cent", "per centum", "pct",
}
_PERCENT_SURFACE_TOKENS = {"%", "percent", "percentage", "per cent", "per centum", "pct"}


def canonicalize_token(raw_tok: str, prev_tok: str = None, next_tok: str = None) -> str:
    """
    Maps open-class numerical, currency, percentage, and calendar-year tokens
    to structural canonical slots so unseen OOD numbers and foreign currencies
    retain zero-UNK syntactic representations without changing parser tensor shapes.
    """
    if not raw_tok:
        return ""
    tok = raw_tok.strip()
    if not tok:
        return ""
    low = tok.lower()
    prev_low = prev_tok.strip().lower() if prev_tok else ""
    next_low = next_tok.strip().lower() if next_tok else ""

    if low in _PERCENT_SURFACE_TOKENS or re.fullmatch(r"per\s+cent(?:um)?", low):
        return "<PERCENT>"
    if low in {"cent", "centum"} and prev_low == "per":
        return "<PERCENT>"

    if low in _CURRENCY_SURFACE_TOKENS:
        return "<CURRENCY>"
    if low == "cent" and prev_low != "per":
        return "<CURRENCY>"

    # Dotted calendar dates (e.g. 01.04.2025)
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.(?:19|20)\d{2}", tok):
        return "<YEAR_4D>"

    # Bare 4-digit calendar year vs monetary integer disambiguation
    if tok.isdigit() and len(tok) == 4 and 1900 <= int(tok) <= 2099:
        if prev_low in _CURRENCY_SURFACE_TOKENS or next_low in _CURRENCY_CONTEXT_TOKENS:
            return "<NUM_INT>"
        return "<YEAR_4D>"

    if re.fullmatch(r"\d+(?:,\d{2,3})*\.\d+", tok):
        return "<NUM_DEC>"

    if re.fullmatch(r"\d+(?:,\d{2,3})*", tok):
        return "<NUM_INT>"

    return low


def canonical_suffix(raw_tok: str, canon_tok: str = None) -> str:
    """
    Returns <NONE> for canonicalized numeric/symbol tokens so equivalent forms
    like '%' vs 'percent' vs 'per cent' share identical token+suffix embeddings.
    """
    if canon_tok is None:
        canon_tok = canonicalize_token(raw_tok)
    if canon_tok in ("<NUM_INT>", "<NUM_DEC>", "<YEAR_4D>", "<CURRENCY>", "<PERCENT>"):
        return "<NONE>"
    return extract_suffix(raw_tok)

# 1. Semantic Role Categories for Statutory Spans
SEMANTIC_ROLES = [
    "ROLE_IGNORE",                           # 0: Non-statutory filler / narrative text / section numbers
    "ROLE_AGENT_STATE_PRECONDITION",         # 1: e.g. "turnover exceeds 10 crore", "paid-up capital of 10 crore", "net worth of 500 crore", "continuous service" (Legacy: ROLE_BUYER_PRIOR_TURNOVER_GUARD)
    "ROLE_TRANSACTION_EXCESS_THRESHOLD",     # 2: e.g. "purchase exceeding 50 lakh", "value exceeding 100 crore"
    "ROLE_TRANSACTION_CLIFF_THRESHOLD",      # 3: e.g. "contract exceeding 30,000", "default not less than one crore"
    "ROLE_STANDARD_STATUTORY_RATE",          # 4: e.g. "0.1 per cent", "10%", "2 per cent on CSR"
    "ROLE_PENALTY_OR_DEFAULT_OVERRIDE",      # 5: e.g. "5 per cent for missing PAN", "20%" (Legacy: ROLE_NON_PAN_OVERRIDE_RATE)
    "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", # 6: e.g. "resident seller", "import of goods", "related party", "public company", "unpublished price sensitive information" (Legacy: ROLE_NON_RESIDENT_EXEMPTION_GUARD)
    "ROLE_TEMPORAL_CUTOVER_DATE",            # 7: e.g. "1st day of July, 2021"
    "ROLE_TEMPORAL_SUNSET_DATE",             # 8: e.g. "omitted w.e.f. 1 April 2025"
    "ROLE_TRIGGER_EARLIER_EVENT",            # 9: e.g. "earlier of credit or payment", "date of occurrence", "day of acceptance"
    "ROLE_TRIGGER_PAYMENT_ONLY",             # 10: e.g. "at the time of receipt"
    "ROLE_DEADLINE_WINDOW",                  # 11: e.g. "forty-five days", "180 days", "two months", "continuous period of three financial years"
    "ROLE_PROHIBITION_GATE",                 # 12: e.g. "shall not be eligible", "no suit shall be instituted", "shall not enter into contract"
    "ROLE_EXCEPTION_CARVEOUT",               # 13: e.g. "except in ordinary course and arm's length", "unless approved by special resolution"
    "ROLE_REQUIRED_EVIDENCE",                # 14: e.g. "Form DIR-8", "Form MSME-1", "Form 3CEB", "e-way bill", "affidavit under section 29A"
    "ROLE_SUBJECTIVE_CRITERION",             # 15: e.g. "convicted of offence involving moral turpitude", "undischarged insolvent", "wilful default"
    "ROLE_MANDATORY_ACTION_OBLIGATION",      # 16: e.g. "shall constitute an audit committee", "shall deliver written notice", "shall spend two per cent"
    "ROLE_STATUTORY_SANCTION",               # 17: e.g. "disqualified for five years", "suit shall be barred", "dismissal of application", "void ab initio"
    "ROLE_ACCOUNTING_DESTINATION",           # 18: e.g. "in profit or loss", "in other comprehensive income", "in carrying amount of asset", "in notes to financial statements"
    "ROLE_COMPUTATION_OPERAND",              # 19: e.g. "adjusted gross income", "itemized deductions", "total wages", "qualifying capital"
]

ROLE2ID = {role: i for i, role in enumerate(SEMANTIC_ROLES)}
# Backward-compatibility aliases for legacy statutory role nomenclature
ROLE2ID["ROLE_BUYER_PRIOR_TURNOVER_GUARD"] = 1
ROLE2ID["ROLE_NON_PAN_OVERRIDE_RATE"] = 5
ROLE2ID["ROLE_NON_RESIDENT_EXEMPTION_GUARD"] = 6

ID2ROLE = {i: role for i, role in enumerate(SEMANTIC_ROLES)}

# 2. Pairwise Relational Edge Types between Spans
RELATION_TYPES = [
    "REL_NO_EDGE",                       # 0: Independent spans
    "REL_PRECONDITION_TO_ENTITY",        # 1: Connects prior turnover / capital gate to Entity
    "REL_THRESHOLD_TO_TRANSACTION",      # 2: Connects threshold to Goods/Services/Transactions
    "REL_RATE_TO_THRESHOLD",             # 3: Connects tax rate / percentage to threshold
    "REL_OVERRIDE_TO_GUARD",             # 4: Connects penal rate to non-PAN guard
    "REL_SUNSET_TO_SECTION",             # 5: Connects omission date to statutory section
    "REL_DEADLINE_TO_TRIGGER",           # 6: Connects duration window to triggering event
    "REL_EXCEPTION_TO_PROHIBITION",      # 7: Connects exception carveout to prohibition gate
    "REL_EVIDENCE_TO_SUBJECTIVE",        # 8: Connects required documentation to subjective standard or obligation
    "REL_LOGICAL_OR",                    # 9: Connects alternative conditions / thresholds / disqualifications
    "REL_LOGICAL_AND",                   # 10: Connects conjunctive prerequisite conditions
    "REL_SANCTION_TO_VIOLATION",         # 11: Connects statutory sanction to prohibition violation / deadline breach
    "REL_ACTION_TO_PRECONDITION",        # 12: Connects mandatory action to entity precondition
    "REL_CONDITION_TO_GATE",             # 13: Connects disqualification/default condition to prohibition gate
    "REL_DESTINATION_TO_ACTION",         # 14: Connects accounting destination to recognition obligation / carveout
    "REL_COMBINATOR_MIN",                # 15: Connects lesser/lower-of branches into MIN(A, B)
    "REL_COMBINATOR_MAX",                # 16: Connects greater/higher-of branches into MAX(A, B)
    "REL_CAP_CEILING",                   # 17: Connects per-unit cap ceiling to base variable
    "REL_BRACKET_NEXT",                  # 18: Links lower bracket boundary / base tax to upper bracket boundary
    "REL_SUBTRACTION_BASE",              # 19: Connects rate/excess threshold to explicit variable operand
    "REL_LOGICAL_NOT",                   # 20: Marks explicit logical negation ('neither', 'nor', 'not', 'none of') on a predicate or span
    "REL_QUANTIFIER_FORALL",             # 21: Connects universal quantifier ('every', 'all', 'each', 'none') to collection/predicate
    "REL_QUANTIFIER_EXISTS",             # 22: Connects existential quantifier ('any', 'at least one', 'some') to collection/predicate
    "REL_AGGREGATE_SUM",                 # 23: Connects summation/count aggregator over collection items to operand or threshold
]

REL2ID = {rel: i for i, rel in enumerate(RELATION_TYPES)}
ID2REL = {i: rel for i, rel in enumerate(RELATION_TYPES)}

# 3. Token-Level BIO Tags for Neural Span Boundary Proposing
BIO_TAGS = [
    "O",                    # 0: Outside statutory span
    "B-GATE",               # 1: Beginning of prohibition or disqualification gate
    "I-GATE",               # 2: Inside prohibition or disqualification gate
    "B-OBLIGATION",         # 3: Beginning of affirmative statutory obligation / duty
    "I-OBLIGATION",         # 4: Inside affirmative statutory obligation / duty
    "B-PRECONDITION",       # 5: Beginning of trigger condition, turnover guard, default condition
    "I-PRECONDITION",       # 6: Inside trigger condition, turnover guard, default condition
    "B-EXCEPTION",          # 7: Beginning of exception carveout, exemption, or proviso
    "I-EXCEPTION",          # 8: Inside exception carveout, exemption, or proviso
    "B-SANCTION",           # 9: Beginning of statutory sanction, penalty, or status bar
    "I-SANCTION",           # 10: Inside statutory sanction, penalty, or status bar
    "B-THRESHOLD",          # 11: Beginning of pecuniary or quantitative threshold
    "I-THRESHOLD",          # 12: Inside pecuniary or quantitative threshold
    "B-RATE",               # 13: Beginning of statutory tax rate, percentage, or compound rate
    "I-RATE",               # 14: Inside statutory tax rate, percentage, or compound rate
    "B-DEADLINE",           # 15: Beginning of prescription deadline or time window
    "I-DEADLINE",           # 16: Inside prescription deadline or time window
    "B-EVIDENCE",           # 17: Beginning of required statutory document, Form, or certification
    "I-EVIDENCE",           # 18: Inside required statutory document, Form, or certification
    "B-ENTITY",             # 19: Beginning of regulated entity scope
    "I-ENTITY",             # 20: Inside regulated entity scope
    "B-DESTINATION",        # 21: Beginning of accounting destination / general ledger routing
    "I-DESTINATION",        # 22: Inside accounting destination / general ledger routing
]

BIO2ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID2BIO = {i: tag for i, tag in enumerate(BIO_TAGS)}

# ── Model v2 Typed-Opcode Taxonomy (keeps v1 tables 100% untouched above) ────
SPAN_UNITS = [
    "UNIT_NONE",             # 0: Non-dimensional / structural / connective span
    "UNIT_MONEY",            # 1: Monetary currency quantum ("$36,900", "Rs. 50,00,000")
    "UNIT_RATE",             # 2: Percentage / basis-point rate ("28%", "3.1 percent")
    "UNIT_DURATION",         # 3: Elapsed duration ("1 taxable year", "183 days", "45 days")
    "UNIT_YEAR_DATE",        # 4: Calendar year or date ("2018", "April 1, 2026")
    "UNIT_COUNT",            # 5: Cardinal integer count or section index ("2 children", "(2)")
    "UNIT_BOOL_PREDICATE",   # 6: Truth-valued legal predicate ("is married", "maintains household")
    "UNIT_SYMBOL_VAR",       # 7: Named variable or defined term ("taxable income", "applicable amount")
]
UNIT2ID = {u: i for i, u in enumerate(SPAN_UNITS)}
ID2UNIT = {i: u for i, u in enumerate(SPAN_UNITS)}

V2_SEMANTIC_ROLES = list(SEMANTIC_ROLES) + [
    "ROLE_DEFINED_TERM",     # 20: LHS or RHS reference of a statutory term definition
    "ROLE_COUNT_QUANTIFIER", # 21: Cardinal count / collection-size threshold
]
V2_ROLE2ID = {role: i for i, role in enumerate(V2_SEMANTIC_ROLES)}
V2_ROLE2ID["ROLE_BUYER_PRIOR_TURNOVER_GUARD"] = 1
V2_ROLE2ID["ROLE_NON_PAN_OVERRIDE_RATE"] = 5
V2_ROLE2ID["ROLE_NON_RESIDENT_EXEMPTION_GUARD"] = 6
V2_ID2ROLE = {i: role for i, role in enumerate(V2_SEMANTIC_ROLES)}

V2_BIO_TAGS = list(BIO_TAGS) + [
    "B-OPERAND",       # 23: Beginning of quantitative base/variable operand
    "I-OPERAND",       # 24: Inside quantitative base/variable operand
    "B-DEFINED_TERM",  # 25: Beginning of defined statutory term ("applicable amount", "surviving spouse")
    "I-DEFINED_TERM",  # 26: Inside defined statutory term
    "B-COUNT",         # 27: Beginning of cardinal count / duration span
    "I-COUNT",         # 28: Inside cardinal count / duration span
]
V2_BIO2ID = {tag: i for i, tag in enumerate(V2_BIO_TAGS)}
V2_ID2BIO = {i: tag for i, tag in enumerate(V2_BIO_TAGS)}

V2_RELATION_TYPES = list(RELATION_TYPES) + [
    "REL_MIN_CAP",         # 24: 1-to-1 opcode-exact upper cap MIN(expr, cap)
    "REL_SUB_CREDIT",      # 25: 1-to-1 opcode-exact subtraction credit MAX(0, SUB(expr, credit))
    "REL_AGGREGATE_COUNT", # 26: 1-to-1 opcode-exact collection COUNT reduction
    "REL_BASE_TAX_ADD",    # 27: Connects bracket base tax ("$5,535") to marginal rate slice via ADD
    "REL_DEFINED_TERM_OF", # 28: Connects defined term symbol to its defining expression/value
    "REL_SUBSTITUTE_WHEN", # 29: Connects conditional/temporal override to target symbol
    "REL_COREF_LINK",      # 30: Window-bounded coreference/anaphora link ("such amount" -> target)
]
V2_REL2ID = {rel: i for i, rel in enumerate(V2_RELATION_TYPES)}
V2_ID2REL = {i: rel for i, rel in enumerate(V2_RELATION_TYPES)}

# 4. Legal Morphological Suffixes
LEGAL_SUFFIXES = [
    "<PAD>", "<UNK>", "-ity", "-ment", "-tion", "-sion", "-ee", "-or", "-er",
    "-able", "-ible", "-ed", "-ing", "-al", "-ive", "-ance", "-ence", "-ure",
    "-ant", "-ent", "-ize", "-ise", "-ly", "-ful", "-less", "-ness", "-ary",
    "-ory", "-ic", "-ous", "-ate", "<NONE>"
]
SUFFIX2ID = {s: i for i, s in enumerate(LEGAL_SUFFIXES)}

# 5. Native Rule-Modality Classification Taxonomy (Replaces Regex Routing)
RULE_MODALITIES = [
    "MODALITY_COMPUTATIONAL_EXCESS",           # 0: Marginal excess deductions/thresholds (e.g., 194Q, 206C(1H), triple threshold)
    "MODALITY_COMPUTATIONAL_CLIFF",            # 1: All-or-nothing cliff deductions (e.g., 194C, 194J, 194R, 194BA, 194S, EL 2.0)
    "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB", # 2: Compound interest, valuation min/max, fractional accruals, corporate tax, 50B
    "MODALITY_DEONTIC_PROHIBITION",            # 3: Prohibitions, election carveouts, OCI bans, safe-harbor provisos (e.g., 17(5), SEBI, 269ST, Ind AS 116)
    "MODALITY_DEONTIC_OBLIGATION_WINDOW",      # 4: Affirmative duties, recognition gates & procedural notice windows (e.g., CPC 80, 12A, Ind AS 37, CSR)
    "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",     # 5: Disqualifications, governance approvals, insolvency triggers, subjective evidence (e.g., 164, 188, 29A)
    "MODALITY_ABSTAIN_UNRESOLVED",             # 6: Non-legal text, conversational noise, ambiguous or operative-empty fragments
    "MODALITY_DEONTIC_PERMISSION",             # 7: Affirmative entitlements/permissions & conditional liberties ('may ... provided that ...')
]
MODALITY2ID = {m: i for i, m in enumerate(RULE_MODALITIES)}
ID2MODALITY = {i: m for i, m in enumerate(RULE_MODALITIES)}


def extract_suffix(word: str) -> str:
    """Extracts dominant legal morphological suffix for an input token."""
    w = word.lower()
    for s in ["-tion", "-sion", "-ment", "-ity", "-able", "-ible", "-ance", "-ence", "-ure", "-ant", "-ent", "-ing", "-ed", "-ee", "-or", "-er", "-al", "-ive", "-ly", "-ous", "-ate"]:
        clean_s = s[1:]
        if len(w) > len(clean_s) + 2 and w.endswith(clean_s):
            return s
    return "<NONE>"


def decode_bio_spans(
    bio_tags: list[str],
    tokens: list[str],
    offsets: list[tuple[int, int]],
    max_gap_bridge: int = 2
) -> list[dict]:
    """
    Decodes token-level BIO tags into contiguous legal spans.
    Enforces gap-bridging across commas and conjunctions ('and', 'or', ',')
    so multi-verb clauses like 'communicate, provide, or allow access' remain glued.
    """
    raw_spans = []
    current_type = None
    start_idx = None
    end_idx = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if current_type is not None:
                raw_spans.append((current_type, start_idx, end_idx))
            current_type = tag[2:]
            start_idx = i
            end_idx = i
        elif tag.startswith("I-"):
            itype = tag[2:]
            if current_type == itype:
                end_idx = i
            elif current_type is not None:
                # Transition mismatch: close previous and start new
                raw_spans.append((current_type, start_idx, end_idx))
                current_type = itype
                start_idx = i
                end_idx = i
            else:
                # Orphan I- tag becomes a B- tag
                current_type = itype
                start_idx = i
                end_idx = i
        else: # "O"
            if current_type is not None:
                raw_spans.append((current_type, start_idx, end_idx))
                current_type = None
                start_idx = None
                end_idx = None

    if current_type is not None:
        raw_spans.append((current_type, start_idx, end_idx))

    if not raw_spans:
        return []

    # 1. Proviso constituent absorption: absorb embedded threshold islands inside outer exception provisos
    absorbed_spans = []
    i = 0
    while i < len(raw_spans):
        stype, s_start, s_end = raw_spans[i]
        if stype == "EXCEPTION" and i + 2 < len(raw_spans):
            mid_type, mid_start, mid_end = raw_spans[i + 1]
            next_type, next_start, next_end = raw_spans[i + 2]
            # If an EXCEPTION span is followed by a short 1-2 token THRESHOLD, followed immediately by another EXCEPTION span
            if mid_type == "THRESHOLD" and (mid_end - mid_start <= 2) and next_type == "EXCEPTION":
                # Absorb the threshold island into the enclosing exception clause
                s_end = next_end
                absorbed_spans.append((stype, s_start, s_end))
                i += 3
                continue
        absorbed_spans.append((stype, s_start, s_end))
        i += 1

    # 2. Gap-bridging across coordinating conjunctions, commas, and subordinate connectors
    def _has_num_or_unit(tok_slice: list[str]) -> bool:
        joined = " ".join(tok_slice)
        if "per cent" in joined:
            return True
        for t in tok_slice:
            if any(ch.isdigit() for ch in t) or t in {"$", "₹", "€", "£", "%", "percent", "cents", "rupees", "dollars", "euros", "pounds"}:
                return True
        return False

    bridged_spans = []
    i = 0
    while i < len(absorbed_spans):
        stype, s_start, s_end = absorbed_spans[i]
        j = i + 1
        while j < len(absorbed_spans):
            next_type, next_start, next_end = absorbed_spans[j]
            gap = next_start - s_end - 1
            max_allowed = 8 if stype == "EXCEPTION" else max_gap_bridge
            if next_type == stype and 0 < gap <= max_allowed:
                # Check if intervening tokens are conjunctions, prepositions, or subordinate connectors
                intervening = [tokens[k].lower() for k in range(s_end + 1, next_start)]
                allowed_bridges = {
                    ",", "or", "and", "including", "other", "in", "of", "to", "such",
                    "where", "the", "is", "due", "on", "account", "by", "reason", "as",
                    "a", "an", "for", "any", "which", "that"
                }
                if all(tok in allowed_bridges for tok in intervening):
                    # Guard: never bridge two distinct rates, numeric thresholds, or calendar dates across commas/conjunctions
                    left_slice = [tokens[k].lower() for k in range(s_start, s_end + 1)]
                    right_slice = [tokens[k].lower() for k in range(next_start, next_end + 1)]
                    if stype == "RATE" and any(tok in {",", "or", "and"} for tok in intervening):
                        break
                    if (
                        stype in {"THRESHOLD", "DEADLINE"}
                        and any(tok in {",", "or", "and"} for tok in intervening)
                        and _has_num_or_unit(left_slice)
                        and _has_num_or_unit(right_slice)
                    ):
                        # Exception: allow "March 15" + "," + "2019" for DEADLINE
                        is_date_year_tail = (
                            stype == "DEADLINE"
                            and intervening == [","]
                            and len(right_slice) == 1
                            and right_slice[0].isdigit()
                            and len(right_slice[0]) == 4
                        )
                        if not is_date_year_tail:
                            break
                    s_end = next_end
                    j += 1
                    continue
            break
        bridged_spans.append((stype, s_start, s_end))
        i = j

    # 3. Unit and Quantifier Absorption: Keep numbers glued to their units/currencies without overlapping next span
    UNIT_TOKENS = {
        "parts", "per", "per cent", "%", "million", "billion", "ppm",
        "hours", "days", "weeks", "week", "months", "years",
        "rupees", "rupee", "lakh", "crore", "lakhs", "crores", "lacs", "lac", "thousand",
        "percent", "percentage", "cent", "cents", "paisa", "paise",
        "dollar", "dollars", "usd", "euro", "euros", "eur", "pound", "pounds", "gbp", "inr", "rs",
    }
    LEADING_CURRENCY_TOKENS = {"$", "₹", "€", "£", "usd", "inr", "eur", "gbp", "rs"}
    MONTH_TOKENS = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
    }

    def _is_day_tok(tok: str) -> bool:
        t_clean = tok.lower()
        for suf in ("st", "nd", "rd", "th"):
            if t_clean.endswith(suf) and t_clean[:-len(suf)].isdigit():
                t_clean = t_clean[:-len(suf)]
                break
        return t_clean.isdigit() and 1 <= int(t_clean) <= 31

    def _is_year_tok(tok: str) -> bool:
        return tok.isdigit() and len(tok) == 4 and tok.startswith(("19", "20"))

    trailing_absorbed = []
    num_bridged = len(bridged_spans)
    for idx, (stype, s_tok, e_tok) in enumerate(bridged_spans):
        next_start = bridged_spans[idx + 1][1] if idx + 1 < num_bridged else len(tokens)
        if stype in {"THRESHOLD", "RATE", "DEADLINE", "PRECONDITION", "ENTITY"}:
            curr_end = e_tok
            while curr_end + 1 < next_start:
                curr_low = tokens[curr_end].lower()
                nxt = tokens[curr_end + 1].lower()
                if nxt in UNIT_TOKENS:
                    curr_end += 1
                elif curr_end + 2 < next_start and tokens[curr_end + 2].lower() in UNIT_TOKENS and nxt in {"per", "of", "in"}:
                    curr_end += 2
                elif stype in {"THRESHOLD", "DEADLINE"} and nxt == "or" and curr_end + 2 < next_start and tokens[curr_end + 2].lower() in {"more", "less"}:
                    curr_end += 2
                elif stype == "DEADLINE" and curr_low in MONTH_TOKENS and _is_day_tok(nxt):
                    curr_end += 1
                elif stype == "DEADLINE" and curr_low in MONTH_TOKENS and _is_year_tok(nxt):
                    curr_end += 1
                elif stype == "DEADLINE" and curr_low in MONTH_TOKENS and nxt == "," and curr_end + 2 < next_start and _is_year_tok(tokens[curr_end + 2]):
                    curr_end += 2
                elif (
                    stype == "DEADLINE"
                    and _is_day_tok(curr_low)
                    and curr_end > s_tok
                    and tokens[curr_end - 1].lower() in MONTH_TOKENS
                    and nxt == ","
                    and curr_end + 2 < next_start
                    and _is_year_tok(tokens[curr_end + 2])
                ):
                    curr_end += 2
                elif (
                    stype == "DEADLINE"
                    and _is_day_tok(curr_low)
                    and curr_end > s_tok
                    and tokens[curr_end - 1].lower() in MONTH_TOKENS
                    and _is_year_tok(nxt)
                ):
                    curr_end += 1
                else:
                    break
            e_tok = curr_end
        trailing_absorbed.append((stype, s_tok, e_tok))

    # 3b. Leading currency symbol absorber (keeps '$36,900', 'EUR 50,000' atomic without overlapping previous span)
    unit_absorbed = []
    for stype, s_tok, e_tok in trailing_absorbed:
        prev_end = unit_absorbed[-1][2] if unit_absorbed else -1
        if (
            stype in {"THRESHOLD", "RATE", "PRECONDITION", "ENTITY"}
            and s_tok > 0
            and (s_tok - 1) > prev_end
            and tokens[s_tok - 1].lower() in LEADING_CURRENCY_TOKENS
        ):
            s_tok = s_tok - 1
        unit_absorbed.append((stype, s_tok, e_tok))

    # 3c. Month-first calendar date repair (prevents orphaned '31' or '31, 2017,' missing leading Month)
    date_repaired = []
    num_ua = len(unit_absorbed)
    for idx, (stype, s_tok, e_tok) in enumerate(unit_absorbed):
        prev_end = date_repaired[-1][2] if date_repaired else -1
        next_start = unit_absorbed[idx + 1][1] if idx + 1 < num_ua else len(tokens)

        # Strip dangling clause comma/semicolon from DEADLINE tails (e.g. '2017,')
        if stype in {"DEADLINE", "THRESHOLD", "RATE"} and e_tok > s_tok and tokens[e_tok] in {",", ";"}:
            e_tok -= 1

        # Case 1: Span starts on day token preceded by Month token and ends with (or precedes) a 4-digit year
        if (
            stype in {"DEADLINE", "THRESHOLD", "RATE", "PRECONDITION"}
            and _is_day_tok(tokens[s_tok])
            and s_tok > 0
            and (s_tok - 1) > prev_end
            and tokens[s_tok - 1].lower() in MONTH_TOKENS
        ):
            has_year_end = (e_tok > s_tok and _is_year_tok(tokens[e_tok]))
            new_end = e_tok
            if not has_year_end:
                if (
                    e_tok + 2 < next_start
                    and tokens[e_tok + 1] == ","
                    and _is_year_tok(tokens[e_tok + 2])
                ):
                    new_end = e_tok + 2
                    has_year_end = True
                elif e_tok + 1 < next_start and _is_year_tok(tokens[e_tok + 1]):
                    new_end = e_tok + 1
                    has_year_end = True
            if has_year_end:
                s_tok = s_tok - 1
                e_tok = new_end
                stype = "DEADLINE"

        # Case 2: Span starts on Month token followed by Day [, Year] even if mislabeled non-DEADLINE
        elif (
            stype in {"DEADLINE", "THRESHOLD", "RATE", "PRECONDITION"}
            and tokens[s_tok].lower() in MONTH_TOKENS
            and e_tok == s_tok
            and s_tok + 1 < next_start
            and _is_day_tok(tokens[s_tok + 1])
        ):
            if s_tok + 3 < next_start and tokens[s_tok + 2] == "," and _is_year_tok(tokens[s_tok + 3]):
                e_tok = s_tok + 3
                stype = "DEADLINE"
            elif s_tok + 2 < next_start and _is_year_tok(tokens[s_tok + 2]):
                e_tok = s_tok + 2
                stype = "DEADLINE"

        date_repaired.append((stype, s_tok, e_tok))
    bridged_spans = date_repaired

    # Convert token spans to text and character offsets
    decoded = []
    for stype, s_tok, e_tok in bridged_spans:
        span_tokens_list = tokens[s_tok : e_tok + 1]
        # Filter noise spans: purely punctuation or single stopword/preposition
        has_alnum = any(any(ch.isalnum() for ch in tok) for tok in span_tokens_list)
        if not has_alnum:
            continue
        if s_tok == e_tok and span_tokens_list[0].lower() in {
            "of", "to", "in", "by", "for", "at", "on", "a", "an", "the", "and", "or", "as", "is", "his"
        }:
            continue

        c_start = offsets[s_tok][0]
        c_end = offsets[e_tok][1]
        decoded.append({
            "type": stype,
            "token_start": s_tok,
            "token_end": e_tok,
            "char_start": c_start,
            "char_end": c_end,
            "span_tokens": (s_tok, e_tok)
        })

    return decoded


class SinusoidalPositionalEncoding(nn.Module):
    """Adds sinusoidal position embeddings to token representations."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            d_model = x.size(2)
            pe = torch.zeros(seq_len, d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0)
        return x + self.pe[:, :seq_len]


class TransformerEncoderBlock(nn.Module):
    """Standard pre-LayerNorm Transformer Encoder Layer with GELU."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, key_padding_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SpanBoundaryRepresentation(nn.Module):
    """
    Extracts fixed-dimensional span representations from variable-length token sequences.
    Combines start token embedding, end token embedding, span difference, and mean pooled vector:
    s_k = W_proj([h_start; h_end; h_end - h_start; mean_pool(h_start:h_end)])
    """
    def __init__(self, d_model: int, span_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4 * d_model, span_dim),
            nn.GELU(),
            nn.LayerNorm(span_dim)
        )

    def forward(self, h_tokens: torch.Tensor, span_indices: list[tuple[int, int]]) -> torch.Tensor:
        if not span_indices:
            return torch.empty(0, self.proj[0].out_features, device=h_tokens.device)

        span_vectors = []
        seq_len = h_tokens.size(0)

        for start, end in span_indices:
            s = max(0, min(start, seq_len - 1))
            e = max(0, min(end, seq_len - 1))
            if s > e:
                s, e = e, s

            h_start = h_tokens[s]
            h_end = h_tokens[e]
            h_diff = h_end - h_start
            h_mean = h_tokens[s : e + 1].mean(dim=0)

            fused = torch.cat([h_start, h_end, h_diff, h_mean], dim=-1)
            span_vectors.append(fused)

        stacked = torch.stack(span_vectors, dim=0)
        return self.proj(stacked)


class DeepBiaffineAttention(nn.Module):
    """
    Computes pairwise directed relational scores between statutory spans:
    Score(s_i, s_j) = s_i^T * W * s_j + U^T * [s_i; s_j] + b
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.bilinear = nn.Bilinear(in_features, in_features, num_classes, bias=False)
        self.linear = nn.Linear(2 * in_features, num_classes)

    def forward(self, spans_src: torch.Tensor, spans_tgt: torch.Tensor) -> torch.Tensor:
        N = spans_src.size(0)
        M = spans_tgt.size(0)

        if N == 0 or M == 0:
            return torch.empty(N, M, self.num_classes, device=spans_src.device)

        src_expanded = spans_src.unsqueeze(1).expand(-1, M, -1).contiguous()
        tgt_expanded = spans_tgt.unsqueeze(0).expand(N, -1, -1).contiguous()

        src_flat = src_expanded.view(N * M, -1)
        tgt_flat = tgt_expanded.view(N * M, -1)

        bilinear_out = self.bilinear(src_flat, tgt_flat)
        linear_out = self.linear(torch.cat([src_flat, tgt_flat], dim=-1))

        scores = (bilinear_out + linear_out).view(N, M, self.num_classes)
        return scores


class MudraStatutoryParser(nn.Module):
    """
    End-to-End Neural Statutory Semantic Parser (Mudra-Decompiler-0).
    Combines:
    1. Lexical word embedding (224-dim) + Morphological Suffix embedding (32-dim)
    2. Multi-Layer Bidirectional Transformer Encoder (256-dim)
    3. Token-Level BIO Sequence Tagging Head (proposes spans without regex)
    4. Span Boundary Representation (SBR) Pooling
    5. Semantic Role Labeling (SRL) Classification Head
    6. Deep Biaffine Relational Edge Head
    """
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 256,
        suffix_dim: int = 32,
        n_heads: int = 4,
        d_ff: int = 1024,
        n_layers: int = 6,
        span_dim: int = 128,
        dropout: float = 0.1,
        num_roles: int = None,
        num_relations: int = None,
        num_bio_tags: int = None,
        num_modalities: int = None
    ):
        super().__init__()
        self.d_model = d_model
        self.word_dim = d_model - suffix_dim
        self.suffix_dim = suffix_dim
        self.num_roles = num_roles if num_roles is not None else len(SEMANTIC_ROLES)
        self.num_relations = num_relations if num_relations is not None else len(RELATION_TYPES)
        self.num_bio_tags = num_bio_tags if num_bio_tags is not None else len(BIO_TAGS)
        self.num_modalities = num_modalities if num_modalities is not None else len(RULE_MODALITIES)

        self.word_embedding = nn.Embedding(vocab_size, self.word_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(len(LEGAL_SUFFIXES), suffix_dim, padding_idx=0)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # 1. Token-Level BIO Tagging Head (Span Proposer)
        self.bio_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_bio_tags)
        )

        # 2. Span Pooling (SBR)
        self.span_pooler = SpanBoundaryRepresentation(d_model, span_dim)

        # 3. Semantic Role Head
        self.role_classifier = nn.Sequential(
            nn.Linear(span_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(span_dim, self.num_roles)
        )

        # 4. Relational Edge Head
        self.relation_classifier = DeepBiaffineAttention(span_dim, self.num_relations)

        # 5. Native Rule-Modality Classifier (Pooled Contextual Sequence)
        self.modality_classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_modalities)
        )

    def encode(
        self,
        token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        token_ids: [batch_size, seq_len]
        suffix_ids: [batch_size, seq_len]
        returns: [batch_size, seq_len, d_model]
        """
        w_emb = self.word_embedding(token_ids)
        if suffix_ids is not None:
            s_emb = self.suffix_embedding(suffix_ids)
        else:
            s_emb = torch.zeros(token_ids.size(0), token_ids.size(1), self.suffix_dim, device=token_ids.device)

        x = torch.cat([w_emb, s_emb], dim=-1) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        pad_mask = mask
        if pad_mask is None and (token_ids == 0).any():
            candidate_pad = (token_ids == 0)
            if (~candidate_pad).any(dim=1).all():
                pad_mask = candidate_pad

        for layer in self.layers:
            x = layer(x, mask=pad_mask)

        return self.final_ln(x)

    def forward(
        self,
        token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        span_indices: list[tuple[int, int]] = None,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """
        Forward pass for a document.
        token_ids: [batch_size, seq_len] (typically [1, seq_len])
        suffix_ids: [batch_size, seq_len]
        span_indices: List of span (start, end) tuples (optional if BIO-only inference)

        Returns:
            bio_logits: [seq_len, len(BIO_TAGS)]
            role_logits: [num_spans, len(SEMANTIC_ROLES)] (or None)
            relation_logits: [num_spans, num_spans, len(RELATION_TYPES)] (or None)
            modality_logits: [batch_size, len(RULE_MODALITIES)]
        """
        h_enc = self.encode(token_ids, suffix_ids=suffix_ids, mask=mask) # [B, seq_len, d_model]

        # Strictly mask out <PAD> tokens (token_ids == 0) before BOTH mean_pool and max_pool
        # so sequence padding never shifts modality predictions between batched training and CLI inference.
        valid_mask = (token_ids != 0)
        if mask is not None:
            valid_mask = valid_mask & (~mask)

        mask_float = valid_mask.unsqueeze(-1).type_as(h_enc) # [B, seq_len, 1]
        valid_counts = mask_float.sum(dim=1).clamp(min=1.0)  # [B, 1]
        mean_pool = (h_enc * mask_float).sum(dim=1) / valid_counts

        h_masked_for_max = h_enc.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        max_pool = h_masked_for_max.max(dim=1).values
        has_valid_tokens = valid_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_valid_tokens, max_pool, torch.zeros_like(max_pool))

        pooled_context = torch.cat([mean_pool, max_pool], dim=-1) # [B, d_model * 2]
        modality_logits = self.modality_classifier(pooled_context) # [B, len(RULE_MODALITIES)]

        h_seq = h_enc.squeeze(0) if h_enc.size(0) == 1 else h_enc[0] # [seq_len, d_model]

        # 1. Predict token-level BIO tags across entire sequence
        bio_logits = self.bio_classifier(h_seq) # [seq_len, len(BIO_TAGS)]

        role_logits = None
        relation_logits = None

        # 2. If spans provided, compute roles and pairwise relations
        if span_indices is not None and len(span_indices) > 0:
            span_vectors = self.span_pooler(h_seq, span_indices) # [num_spans, span_dim]
            role_logits = self.role_classifier(span_vectors)     # [num_spans, len(SEMANTIC_ROLES)]
            relation_logits = self.relation_classifier(span_vectors, span_vectors) # [num_spans, num_spans, len(RELATION_TYPES)]

        return bio_logits, role_logits, relation_logits, modality_logits

    def classify_span_modality(
        self,
        h_seq: torch.Tensor,
        span_ranges: list[tuple[int, int]]
    ) -> torch.Tensor:
        """
        Pools [mean_pool(h_seq[s:e+1]); max_pool(h_seq[s:e+1])] per clause range
        and projects through the existing modality_classifier (d_model*2 -> 128 -> num_modalities)
        so each sub-clause in a multi-clause statute receives an independent modality prediction.
        """
        if not span_ranges:
            return torch.empty(0, self.num_modalities, device=h_seq.device)

        if h_seq.dim() == 3:
            h_seq = h_seq.squeeze(0) if h_seq.size(0) == 1 else h_seq[0]

        seq_len = h_seq.size(0)
        pooled = []
        for s_idx, e_idx in span_ranges:
            s = max(0, min(s_idx, seq_len - 1))
            e = max(0, min(e_idx, seq_len - 1))
            if s > e:
                s, e = e, s
            seg = h_seq[s : e + 1]
            mean_v = seg.mean(dim=0)
            max_v = seg.max(dim=0).values
            pooled.append(torch.cat([mean_v, max_v], dim=-1))

        return self.modality_classifier(torch.stack(pooled, dim=0))

    def forward_multitask(
        self,
        token_ids: torch.Tensor,
        suffix_ids: torch.Tensor = None,
        span_indices: list[tuple[int, int]] = None,
        mask: torch.Tensor = None,
        clause_spans: list[tuple[int, int]] = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """
        Single-encoder-pass multitask forward for training multi-clause modality supervision
        alongside document-level BIO/SRL/Biaffine/Modality heads.
        """
        h_enc = self.encode(token_ids, suffix_ids=suffix_ids, mask=mask)

        valid_mask = (token_ids != 0)
        if mask is not None:
            valid_mask = valid_mask & (~mask)

        mask_float = valid_mask.unsqueeze(-1).type_as(h_enc)
        valid_counts = mask_float.sum(dim=1).clamp(min=1.0)
        mean_pool = (h_enc * mask_float).sum(dim=1) / valid_counts

        h_masked_for_max = h_enc.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        max_pool = h_masked_for_max.max(dim=1).values
        has_valid_tokens = valid_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(has_valid_tokens, max_pool, torch.zeros_like(max_pool))

        pooled_context = torch.cat([mean_pool, max_pool], dim=-1)
        modality_logits = self.modality_classifier(pooled_context)

        h_seq = h_enc.squeeze(0) if h_enc.size(0) == 1 else h_enc[0]
        bio_logits = self.bio_classifier(h_seq)

        role_logits = None
        relation_logits = None
        if span_indices is not None and len(span_indices) > 0:
            span_vectors = self.span_pooler(h_seq, span_indices)
            role_logits = self.role_classifier(span_vectors)
            relation_logits = self.relation_classifier(span_vectors, span_vectors)

        clause_modality_logits = None
        if clause_spans is not None and len(clause_spans) > 0:
            clause_modality_logits = self.classify_span_modality(h_seq, clause_spans)

        return bio_logits, role_logits, relation_logits, modality_logits, clause_modality_logits
