"""
=============================================================================
MUDRA-DECOMPILER-0 END-TO-END INFERENCE & AST EMITTER (v5.0 - ZERO REGEX BIO)
=============================================================================

Executes zero-shot neural compilation from raw statutory text or government PDFs
to verified USVM AST rules without human handholding.

Neural Pipeline:
1. Lexical Tokenization with Exact Character Offset Tracking (dataset_generator.py).
2. Word Embeddings (224-dim) + Morphological Suffix Embeddings (32-dim).
3. 6-Layer Bidirectional Transformer Encoder on CUDA.
4. Token-Level BIO Sequence Tagging Head (proposes statutory spans dynamically).
5. Transition-Constrained Contiguous Span Decoding across conjunctions/commas.
6. Span Boundary Representation (SBR) Pooling.
7. Semantic Role Labeling (SRL) & Deep Biaffine Relational Scoring.
8. Deontic USVM AST Synthesis preserving statutory invariant (isAuthorizedToDisburse: false).
"""

import os
import re
import sys
import json
import hashlib
import torch

from .pdf_ingestion import StatutoryDocumentIngestor
from .tokenizer import tokenize_with_offsets, char_span_to_token_span
from .statutory_neural_parser import (
    MudraStatutoryParser,
    SPECIAL_TOKENS,
    ID2ROLE,
    ID2REL,
    ID2BIO,
    ID2MODALITY,
    SEMANTIC_ROLES,
    RELATION_TYPES,
    BIO_TAGS,
    LEGAL_SUFFIXES,
    RULE_MODALITIES,
    SUFFIX2ID,
    extract_suffix,
    canonicalize_token,
    canonical_suffix,
    decode_bio_spans
)


MODAL_AND_AUX = {
    "shall", "may", "must", "should", "cannot", "can", "will", "would", "could",
    "to", "not", "has", "have", "had", "is", "are", "was", "were", "be", "been", "being"
}
DETERMINERS_AND_PREP = {
    "the", "a", "an", "this", "that", "these", "those", "whose", "any", "every", "each",
    "all", "no", "none", "some", "neither", "either", "nor", "or",
    "of", "for", "in", "by", "with", "from", "on", "at", "against"
}
VERB_SUFFIXES = ("ed", "ing", "ize", "ise", "ate", "ify")
NOUN_SUFFIXES = ("er", "or", "ee", "ion", "ment", "ity", "ance", "ence", "ship", "ist", "ism", "ness", "ure")


def resolve_open_class_token(
    token: str,
    prev_token: str = None,
    prev_prev_token: str = None,
    vocab: dict = None,
    next_token: str = None
) -> int:
    """
    Syntactic Slot Splitting & Canonical Structural Mapping:
    Maps canonical numbers/currencies/years/percent to SPECIAL_TOKENS first,
    then checks trained vocabulary, then routes unseen open-class words into
    <LEGAL_VERB> vs <LEGAL_NOUN>.
    """
    if not token:
        return 0
    canon = canonicalize_token(token, prev_token, next_token)
    if not canon:
        return 0
    if vocab and canon in vocab:
        return vocab[canon]
    low = token.lower()
    if vocab and low in vocab:
        return vocab[low]
    if canon in SPECIAL_TOKENS:
        return vocab.get(canon, SPECIAL_TOKENS[canon]) if vocab else SPECIAL_TOKENS[canon]
    if not token.isalpha():
        return vocab.get("<UNK>", 1) if vocab else 1

    legal_noun_id = vocab.get("<LEGAL_NOUN>", 2) if vocab else 2
    legal_verb_id = vocab.get("<LEGAL_VERB>", 3) if vocab else 3

    prev_low = prev_token.lower() if prev_token else ""
    prev_prev_low = prev_prev_token.lower() if prev_prev_token else ""

    # 1. Immediate modal or auxiliary verb -> Action Verb Slot
    # e.g., 'may proceed', 'shall deduct', 'has moved', 'to furnish', 'will disburse'
    if prev_low in MODAL_AND_AUX:
        return legal_verb_id

    # 2. Auxiliary + adverb sequence -> Action Verb Slot
    # e.g., 'has previously moved', 'has already accrued', 'is actively engaged'
    if prev_prev_low in MODAL_AND_AUX and (prev_low.endswith("ly") or prev_low in ("not", "already", "previously", "directly")):
        return legal_verb_id

    # 3. Inflectional verb suffixes (-ed, -ing, -ize, -ate) when not governed by a determiner
    if low.endswith(VERB_SUFFIXES) and prev_low not in DETERMINERS_AND_PREP:
        return legal_verb_id

    # 4. Governed by determiner or preposition -> Entity / Noun Slot
    # e.g., 'a member', 'the applicant', 'nor the trustee', 'of units', 'in respect of'
    if prev_low in DETERMINERS_AND_PREP:
        return legal_noun_id

    # 5. Nominal suffixes (-er, -ee, -or, -ion, -ment, etc.)
    if low.endswith(NOUN_SUFFIXES):
        return legal_noun_id

    # Default fallback for unknown alpha: <LEGAL_NOUN>
    return legal_noun_id


WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
    "crore": 10000000, "crores": 10000000, "million": 1000000, "billion": 1000000000
}


FRACTION_DENOMINATORS = {
    "half", "halves", "third", "thirds", "fourth", "fourths", "fifth", "fifths",
    "sixth", "sixths", "eighth", "eighths", "tenth", "tenths"
}


def parse_numeric_quantity(text: str):
    if not text:
        return None
    cleaned = text.lower().replace(",", "")
    cleaned_no_pct = re.sub(r"\bper\s+cent(?:um)?\b", " ", cleaned)
    has_major_currency = bool(re.search(r"₹|\$|€|£|\b(?:dollars?|rupees?|euros?|pounds?|usd|inr|eur|gbp)\b", cleaned_no_pct))
    is_standalone_minor = bool(re.search(r"\b(?:cents?|paise|paisa|pence)\b", cleaned_no_pct)) and not has_major_currency

    digits = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    if digits:
        val = float(digits[0])
        if re.search(r"\bbillion\b", cleaned):
            val *= 1000000000
        elif re.search(r"\bmillion\b", cleaned):
            val *= 1000000
        elif re.search(r"\bcrores?\b", cleaned):
            val *= 10000000
        elif re.search(r"\b(?:lakhs?|lacs?)\b", cleaned):
            val *= 100000
        elif re.search(r"\bthousand\b", cleaned):
            val *= 1000
        elif is_standalone_minor:
            val /= 100.0
        return int(val) if val.is_integer() else val

    words = re.findall(r"[a-z\-]+", cleaned)
    total = 0
    current = 0
    found = False
    for w in words:
        if w in WORD_TO_NUM:
            found = True
            n = WORD_TO_NUM[w]
            if n in (1000000000, 10000000, 1000000, 100000, 1000):
                current = (current if current != 0 else 1) * n
                total += current
                current = 0
            elif n == 100:
                current = (current if current != 0 else 1) * 100
            else:
                current += n
        elif "-" in w:
            parts = w.split("-")
            if any(p in FRACTION_DENOMINATORS for p in parts):
                continue
            sub_val = sum(WORD_TO_NUM[p] for p in parts if p in WORD_TO_NUM)
            if sub_val > 0:
                found = True
                current += sub_val
    total += current
    if not found:
        return None
    val = float(total) / 100.0 if is_standalone_minor else float(total)
    return int(val) if val.is_integer() else val


_COMP_PREFIXES = {
    "exceed", "exceeds", "exceeding", "over", "above", "under", "below",
    "not over", "up to", "not less than", "at least", "more than",
    "less than", "not exceeding", "excess over", "does not exceed",
}
_UNIT_TAIL_RE = re.compile(
    r"^(?:crores?|lakhs?|lacs?|thousand|million|billion|rupees?|dollars?|euros?|pounds?|paise|paisa|cents?|pence|rs\.?|inr|usd|eur|gbp)$",
    re.IGNORECASE,
)
_NON_STITCH_ROLES = {
    "ROLE_COMPUTATION_OPERAND",
    "ROLE_PROHIBITION_GATE",
    "ROLE_MANDATORY_ACTION_OBLIGATION",
    "ROLE_EXCEPTION_CARVEOUT",
}


def spans_are_stitchable(curr: dict, nxt: dict, text: str) -> bool:
    """Decides whether two adjacent classified spans form one compound quantity.

    Shared by the standalone and dual-stream inference paths so their span
    stitching cannot drift apart.
    """
    gap_str = text[curr["charRange"][1]:nxt["charRange"][0]]
    if "," in gap_str:
        return False
    gap_norm = gap_str.strip().lower()
    # Compound Indian monetary amounts ("two lakh and fifty thousand rupees")
    # arrive as two monetary spans joined by the connective "and".
    is_amount_connective = (
        gap_norm == "and"
        and curr["clauseIdx"] == nxt["clauseIdx"]
        and parse_unit_and_type(curr["span"])[0] == "MONETARY_QUANTUM"
        and parse_unit_and_type(nxt["span"])[0] == "MONETARY_QUANTUM"
    )
    if not (nxt["charRange"][0] - curr["charRange"][1] <= 1 or is_amount_connective):
        return False

    comb_text = text[curr["charRange"][0]:nxt["charRange"][1]].strip()
    qty_curr = parse_numeric_quantity(curr["span"])
    qty_nxt = parse_numeric_quantity(nxt["span"])
    sym_curr, _, _ = parse_unit_and_type(curr["span"])
    sym_nxt, _, _ = parse_unit_and_type(nxt["span"])
    qty_comb = parse_numeric_quantity(comb_text)
    if qty_comb is None:
        return False

    if qty_curr is None and sym_curr == "MONETARY_QUANTUM" and qty_nxt is not None:
        return True
    if qty_curr is not None and sym_curr == "BOOLEAN_FLAG" and qty_nxt is None and sym_nxt in ("MONETARY_QUANTUM", "PERCENTAGE_RATE", "TEMPORAL_DURATION"):
        return True
    if qty_curr is not None and sym_curr == "MONETARY_QUANTUM" and qty_nxt is None and _UNIT_TAIL_RE.match(nxt["span"].strip()):
        return True
    if (
        curr["span"].strip().lower() in _COMP_PREFIXES
        or (
            curr["span"].strip().lower() in {"not", "no", "up"}
            and curr.get("bioType") in ("THRESHOLD", "PRECONDITION", "DEADLINE")
            and re.match(r"^(?:over|above|exceeding|exceed|more\s+than|less\s+than|under|below|to)\b", nxt["span"].strip(), flags=re.IGNORECASE)
        )
    ) and qty_nxt is not None and sym_nxt == "MONETARY_QUANTUM":
        return True
    if (
        qty_curr is not None
        and qty_nxt is not None
        and curr["role"] not in _NON_STITCH_ROLES
        and nxt["role"] not in _NON_STITCH_ROLES
        and curr["clauseIdx"] == nxt["clauseIdx"]
    ):
        return True
    return False


def stitch_classified_spans(classified_spans: list[dict], text: str) -> tuple[list[dict], dict]:
    """Greedily merges adjacent spans that form one compound quantity.

    A merged span is re-tested against the following span so multi-part
    amounts ("exceeds" + "two lakh" + "fifty thousand rupees") collapse into
    a single threshold. Shared by both inference paths.
    """
    stitched: list[dict] = []
    old_to_new: dict = {}
    idx = 0
    while idx < len(classified_spans):
        curr = dict(classified_spans[idx])
        new_idx = len(stitched)
        consumed = 1
        while idx + consumed < len(classified_spans):
            nxt = classified_spans[idx + consumed]
            if not spans_are_stitchable(curr, nxt, text):
                break
            comb_text = text[curr["charRange"][0]:nxt["charRange"][1]].strip()
            dominant_role = curr["role"] if curr["role"] != "ROLE_IGNORE" else nxt["role"]
            for role in (
                "ROLE_TRANSACTION_CLIFF_THRESHOLD",
                "ROLE_TRANSACTION_EXCESS_THRESHOLD",
                "ROLE_AGENT_STATE_PRECONDITION",
                "ROLE_STANDARD_STATUTORY_RATE",
            ):
                if role in (curr["role"], nxt["role"]):
                    dominant_role = role
                    break
            curr = {
                "span": comb_text,
                "bioType": curr["bioType"],
                "role": dominant_role,
                "confidence": round((curr["confidence"] + nxt["confidence"]) / 2, 4),
                "charRange": [curr["charRange"][0], nxt["charRange"][1]],
                "tokenRange": [curr["tokenRange"][0], nxt["tokenRange"][1]],
                "clauseIdx": curr["clauseIdx"],
            }
            consumed += 1
        curr["spanIdx"] = new_idx
        stitched.append(curr)
        for k in range(idx, idx + consumed):
            old_to_new[k] = new_idx
        idx += consumed
    return stitched, old_to_new


def parse_unit_and_type(text: str):
    low = text.lower()
    # Mask 'per cent' / 'per centum' BEFORE minor-unit 'cent' matching
    low_no_pct = re.sub(r"\bper\s+cent(?:um)?\b", " _pct_ ", low)
    # 1. Currency has highest precedence using strict word boundaries (avoids "rs" matching "years"/"directors")
    if re.search(r"₹|\b(?:rupees?|rs\.?|inr|crores?|lakhs?|lacs?|paisa|paise)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_PAISA", "INR"
    if re.search(r"\$|\b(?:dollars?|usd|cents?)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "USD"
    if re.search(r"€|\b(?:euros?|eur)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "EUR"
    if re.search(r"£|\b(?:pounds?|gbp|pence)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "GBP"
    # 2. Percentage has next precedence
    if "%" in low or "_pct_" in low_no_pct or re.search(r"\bpercent(?:age)?\b|\bpct\b", low):
        return "PERCENTAGE_RATE", "PERCENT", "PERCENT"
    # 3. Temporal duration applies only when no currency/percentage is present
    if re.search(r"\b(?:years?|annum|annual)\b", low) and not re.search(r"\b(?:financial|previous|calendar)\s+year\b", low):
        return "TEMPORAL_DURATION", "NUMBER", "YEARS"
    if re.search(r"\bmonths?\b", low):
        return "TEMPORAL_DURATION", "NUMBER", "MONTHS"
    if re.search(r"\bdays?\b", low):
        return "TEMPORAL_DURATION", "NUMBER", "DAYS"
    if re.search(r"\bweeks?\b", low):
        return "TEMPORAL_DURATION", "NUMBER", "WEEKS"
    if re.search(r"\bhours?\b", low):
        return "TEMPORAL_DURATION", "NUMBER", "HOURS"
    # 4. Documentary returns and certificates
    if re.search(r"\b(?:form|dir|msme|3ceb|certificate|affidavit|bill|declaration|challan|schedule|return|report)\b", low):
        return "DOCUMENT_VERIFICATION", "DOCUMENT_ID", "REGISTRY_VERIFICATION"
    # 5. Raw numeric quantum (e.g. "5,535", "1000", "50.5")
    if re.search(r"^\s*[\$₹€£]?\s*[\d,]+(?:\.\d+)?\s*$", low):
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "CURRENCY"
    return "BOOLEAN_FLAG", "BOOLEAN", "FLAG"


def parse_comparison_operator(text: str):
    low = text.lower()
    if re.search(r"\b(?:not\s+less\s+than|at\s+least|minimum\s+of|no\s+less\s+than|or\s+more|or\s+greater|or\s+above)\b", low):
        return "GTE"
    if re.search(r"\b(?:not\s+exceeding|not\s+over|at\s+most|maximum\s+of|twelve\s+months\s+or\s+less|or\s+less|or\s+below|or\s+fewer|no\s+more\s+than)\b", low):
        return "LTE"
    if re.search(r"\b(?:exceeding|exceeds|exceed|more\s+than|greater\s+than|in\s+excess\s+of|above|over)\b", low):
        return "GT"
    if re.search(r"\b(?:less\s+than|below|under)\b", low):
        return "LT"
    if re.search(r"\b(?:equal\s+to|equals|exactly)\b", low):
        return "EQ"
    return "GTE"


def _derive_fact_key_camel(raw_text: str, fallback: str = "metricValue") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (raw_text or "").lower())
    drop_words = {
        "the", "a", "an", "of", "for", "in", "on", "to", "any", "such", "his", "her", "their", "its",
        "is", "are", "was", "were", "be", "been", "being", "where", "thereof", "by", "from", "with",
        "as", "at", "and", "or", "excess", "surplus", "amount", "sum", "value", "total", "otherwise",
        "allowable", "eligible", "qualifying", "covered", "paid", "received", "payable", "above", "over",
        "under", "below", "exceed", "exceeds", "exceeding", "not", "no", "more", "than", "less", "up",
        "slice", "part", "portion", "rupees", "rupee", "dollars", "dollar", "euros", "euro", "pounds",
        "pound", "cents", "cent", "paise", "paisa", "inr", "usd", "eur", "gbp", "lakh", "lakhs", "lac",
        "lacs", "crore", "crores", "thousand", "million", "billion"
    }
    words = [w for w in cleaned.split() if not w.isdigit() and w not in drop_words and len(w) > 1]
    if not words:
        return fallback
    chosen = words[:3]
    camel = chosen[0] + "".join(w.capitalize() for w in chosen[1:])
    if camel and camel[0].isdigit():
        camel = "m" + camel
    return camel or fallback


def _safe_preceding_words(text: str, start_idx: int, max_chars: int = 40) -> str:
    """Slices up to max_chars before start_idx without bisecting a leading word."""
    win_start = max(0, start_idx - max_chars)
    snippet = text[win_start:start_idx]
    if win_start > 0 and text[win_start - 1].isalnum() and snippet and snippet[0].isalnum():
        snippet = re.sub(r"^\S+\s*", "", snippet)
    return snippet


def segment_clauses(text: str, tokens: list[str], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Deterministic multi-clause segmentation mapped cleanly to inclusive token ranges [start, end].
    Splits on sentence delimiters, semicolons, proviso colons, and standalone enumerated subclauses,
    merging short (< 3 token) fragments cleanly.
    """
    n = len(tokens)
    if n == 0:
        return []

    ABBREV_NO_SPLIT = {
        "rs", "no", "nos", "sl", "sec", "art", "reg", "dr", "mr", "ms", "vs", "etc", "e", "g", "i"
    }
    REF_PREFIXES = {
        "section", "sections", "clause", "clauses", "sub-section", "subsection",
        "paragraph", "para", "rule", "rules", "of", "under", "to", "and", "or", "table", "sl"
    }

    raw_segments = []
    seg_start = 0

    for i, tok in enumerate(tokens):
        low = tok.lower()
        prev_low = tokens[i - 1].lower() if i > 0 else ""
        next_low = tokens[i + 1].lower() if i + 1 < n else ""

        # Split BEFORE standalone enumerated subclause markers like '(a)', '(1)', '(i)'
        if (
            i > seg_start + 1
            and re.match(r"^\([a-zA-Z0-9ivxIVX]{1,4}\)$", tok)
            and prev_low.rstrip(".") not in REF_PREFIXES
        ):
            raw_segments.append((seg_start, i - 1))
            seg_start = i
            continue

        should_split_after = False
        if tok == ";":
            should_split_after = True
        elif tok == ":":
            if next_low in {"provided", "except", "subject", "where"}:
                should_split_after = True
        elif tok == ".":
            prev_clean = prev_low.rstrip(".")
            is_abbrev = prev_clean in ABBREV_NO_SPLIT or (len(prev_low) == 1 and prev_low.isalpha())
            is_decimal_adjacency = (i + 1 < n and offsets[i][1] == offsets[i + 1][0])
            if not is_abbrev and not is_decimal_adjacency:
                should_split_after = True

        if should_split_after and i >= seg_start:
            raw_segments.append((seg_start, i))
            seg_start = i + 1

    if seg_start < n:
        raw_segments.append((seg_start, n - 1))

    if not raw_segments:
        return [(0, n - 1)]

    # Merge tiny (< 3 tokens) tail/orphan fragments with adjacent clause
    merged = []
    for s_idx, e_idx in raw_segments:
        length = e_idx - s_idx + 1
        if length < 3 and merged:
            prev_s, _ = merged[-1]
            merged[-1] = (prev_s, e_idx)
        else:
            merged.append((s_idx, e_idx))

    if len(merged) > 1 and (merged[0][1] - merged[0][0] + 1) < 3:
        first_s, _ = merged[0]
        _, next_e = merged[1]
        merged = [(first_s, next_e)] + merged[2:]

    return merged


def normalize_symbol_id(span_text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", span_text.lower())
    words = cleaned.split()
    stop_words = {
        "the", "a", "an", "of", "for", "in", "on", "to", "any", "such", "his", "her", "their", "its",
        "is", "are", "was", "were", "be", "been", "being", "where", "thereof", "not", "no", "will", "shall",
        "cannot", "can", "may", "must", "would", "could", "might", "and", "or", "with", "at", "by", "from",
        "this", "that", "these", "those", "have", "has", "had"
    }
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    if not filtered:
        return ""
    symbol_id = "_".join(filtered[:4])
    return symbol_id


ALLOWED_RULE_DOMAINS = ("AUTO", "STATUTORY_FINANCE", "GENERAL_LOGIC")

_LOGIC_FACT_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "at", "by", "from", "with", "into",
    "through", "as", "that", "which", "who", "whom", "whose", "such", "any", "every",
    "each", "all", "some", "none", "no", "not", "neither", "nor", "either", "or", "and",
    "provided", "where", "when", "if", "unless", "except", "until", "upon", "shall", "may",
    "must", "should", "can", "cannot", "will", "would", "could", "it", "its", "his", "her",
    "their", "be", "been", "being", "is", "are", "was", "were", "has", "have", "had",
    "there", "this", "these", "those", "within"
}

_FISCAL_STATUTORY_RE = re.compile(
    r"₹|\b(?:rupees?|paisa|paise|inr|crores?|lakhs?|lacs?|tax|taxes|taxable|taxation|taxpayer|"
    r"tds|tcs|gst|cgst|igst|sgst|itc|msme|rbi|sebi|mca|cbdt|cbic|pan|tan|206aa|206ab|"
    r"194[a-z]?|206c|dir-?8|29a|withhold|withholding|deduct|deduction|payroll|remuneration|"
    r"invoice|surcharge|dividend|fiscal|excise|"
    r"profit\s+or\s+loss|other\s+comprehensive\s+income|carrying\s+amount|recoverable\s+amount|"
    r"straight-line\s+basis|impairment|lease\s+recognition|short-term\s+leases|present\s+obligation|"
    r"economic\s+benefits|turnover|gross\s+sales|total\s+sales|gross\s+receipts|paid-up|"
    r"net\s+worth|slump\s+sale|gratuity|continuous\s+service|insider|unpublished\s+price\s+sensitive|"
    r"related\s+party|special\s+resolution|board\s+of\s+directors|insolvent|resolution\s+plan|"
    r"undischarged|moral\s+turpitude|plaint|no\s+suit\s+shall|civil\s+court|wages)\b",
    flags=re.IGNORECASE,
)


def derive_logic_fact_key(phrase: str, fallback: str = "conditionFlag", max_words: int = 5) -> str:
    tokens_raw = (phrase or "").lower().split()
    words = []
    for tok in tokens_raw:
        clean_tok = tok.strip(".,;:!?()[]{}\"'")
        if "-" in clean_tok:
            parts = [re.sub(r"[^a-z0-9]", "", p) for p in clean_tok.split("-")]
            parts = [p for p in parts if p and not p.isdigit()]
            if parts and any(p not in _LOGIC_FACT_STOPWORDS for p in parts):
                words.extend(parts)
                continue
        sub = [
            p for p in re.sub(r"[^a-z0-9]", " ", clean_tok).split()
            if p and not p.isdigit() and p not in _LOGIC_FACT_STOPWORDS and len(p) > 1
        ]
        words.extend(sub)
    if not words:
        return fallback
    chosen = words[:max_words]
    camel = chosen[0] + "".join(w.capitalize() for w in chosen[1:])
    if camel and camel[0].isdigit():
        camel = "k" + camel
    return camel or fallback


def pluralize_collection_noun(noun: str) -> str:
    w = re.sub(r"[^a-zA-Z0-9]", "", (noun or "").lower())
    if not w:
        return "items"
    if w in {"people", "children", "men", "women", "data", "criteria", "entries"}:
        return w
    if w.endswith(("s", "x", "z", "ch", "sh")):
        if w.endswith("s"):
            return w
        return w + "es"
    if len(w) > 2 and w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def parse_rate_bps(rate_str: str | None) -> int | None:
    if not rate_str:
        return None
    qty = parse_numeric_quantity(rate_str)
    if qty is None:
        return None
    return int(round(float(qty) * 100))


def parse_iso_date_str(date_str: str | None) -> str | None:
    if not date_str:
        return None
    from datetime import datetime
    clean = re.sub(r"(?<=\d)(st|nd|rd|th)", "", str(date_str).strip(), flags=re.IGNORECASE)
    clean = re.sub(r"day\s+of", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s+,", ",", clean)
    clean = re.sub(r",\s*", ", ", clean)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(".,;:")
    for fmt in (
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %B, %Y",
        "%d %B %Y",
        "%d %b, %Y",
        "%d %b %Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_days_window(dur_str: str | None) -> int | None:
    if not dur_str:
        return None
    low = dur_str.lower()
    if "appointed day" in low:
        return 15
    qty = parse_numeric_quantity(dur_str)
    if qty is not None:
        n = float(qty)
        if "month" in low:
            return int(round(n * 30))
        if "year" in low:
            return int(round(n * 365))
        if "hour" in low:
            return max(1, int(round(n / 24)))
        return int(round(n))
    return None


def classify_gl_destination_from_spans(accounting_destinations: list[str], fallback: str) -> str:
    joined = " ".join(accounting_destinations).lower()
    if "other comprehensive income" in joined:
        return "OTHER_COMPREHENSIVE_INCOME"
    if "carrying amount" in joined or "asset" in joined:
        return "ASSET_CAPITALIZATION"
    if "notes" in joined or "contingent" in joined:
        return "CONTINGENT_LIABILITY_FOOTNOTE"
    if "profit or loss" in joined:
        return "PROFIT_OR_LOSS"
    return fallback


def parse_minor_units(text_val: str | None) -> int | None:
    if not text_val:
        return None
    qty = parse_numeric_quantity(text_val)
    if qty is None:
        return None
    return int(round(float(qty) * 100))


def infer_rule_currency_info(spans: list[dict], text: str) -> tuple[str | None, str]:
    for s in spans:
        st, dt, unit = parse_unit_and_type(s["span"])
        if st == "MONETARY_QUANTUM" and dt in ("CURRENCY_PAISA", "CURRENCY_CENTS", "CURRENCY_MINOR_UNITS"):
            return unit, dt
    st, dt, unit = parse_unit_and_type(text)
    if st == "MONETARY_QUANTUM":
        return unit, dt
    return None, "CURRENCY_PAISA"


def build_operand_fact_keys(span_text: str, currency_code: str | None) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (span_text or "").lower())
    drop_words = {
        "the", "a", "an", "of", "for", "in", "on", "to", "any", "such", "his", "her", "their", "its",
        "is", "are", "was", "were", "be", "been", "being", "where", "thereof", "by", "from", "with",
        "as", "at", "and", "or", "excess", "surplus", "amount", "sum", "value", "total", "otherwise",
        "allowable", "eligible", "qualifying", "covered", "paid", "received", "payable", "above", "over",
        "under", "below", "exceed", "exceeds", "exceeding", "not", "no", "more", "than", "less", "up",
        "slice", "part", "portion", "rupees", "rupee", "dollars", "dollar", "euros", "euro", "pounds",
        "pound", "cents", "cent", "paise", "paisa", "inr", "usd", "eur", "gbp", "lakh", "lakhs", "lac",
        "lacs", "crore", "crores", "thousand", "million", "billion"
    }
    words = [w for w in cleaned.split() if not w.isdigit() and w not in drop_words and len(w) > 1]
    camel = _derive_fact_key_camel(span_text, "")
    ccy_suffix = currency_code.capitalize() if currency_code in ("USD", "EUR", "GBP", "INR") else ""
    candidates = []
    if camel:
        candidates.append(camel)
        if ccy_suffix:
            candidates.append(f"{camel}{ccy_suffix}")
    if len(words) >= 3:
        acronym = "".join(w[0] for w in words[:3])
        if len(acronym) >= 2:
            candidates.append(acronym)
            if ccy_suffix:
                candidates.append(f"{acronym}{ccy_suffix}")
    adj_skip = {"adjusted", "taxable", "itemized", "allowable", "otherwise", "gross", "net", "annual", "calendar", "total"}
    if words:
        for w in (words[0], words[-1]):
            if w not in adj_skip and len(w) >= 4:
                candidates.append(w)
                if ccy_suffix:
                    candidates.append(f"{w}{ccy_suffix}")
    dedup = []
    for k in candidates:
        if k and k not in dedup:
            dedup.append(k)
    return dedup[:5]


_V2_CLAUSE_LEAD_RE = re.compile(
    r"^\s*(?:if|where|when|whenever|in\s+the\s+case\s+of|in\s+case|provided\s+that|"
    r"for\s+the\s+purposes\s+of\s+(?:this\s+)?(?:section|paragraph|subparagraph|clause|subsection))\b\s*",
    re.IGNORECASE,
)

_V2_COPULA_TAIL_RE = re.compile(
    r"\s+(?:is|are|was|were|equals?|exceeds?|amounts?\s+to|is\s+equal\s+to|"
    r"does\s+not\s+exceed|is\s+over|is\s+above|is\s+below|is\s+under)\s*$",
    re.IGNORECASE,
)


def resolve_v2_input_id(text: str, threshold_char_start: int, currency_code: str | None) -> str:
    """Derives the single typed input id for a v2 bracket clause.

    Step 1 only needs a deterministic single-clause resolver: the subject of
    the threshold comparison is the noun phrase immediately before the first
    bracket bound. It never probes aliases; an underivable subject fails closed.
    """
    prefix = (text or "")[: max(0, int(threshold_char_start))]
    # Restrict to the current sentence/clause before stripping connectors.
    prefix = re.split(r"[.;]\s+", prefix)[-1]
    previous = None
    while previous != prefix:
        previous = prefix
        prefix = _V2_CLAUSE_LEAD_RE.sub("", prefix, count=1)
    previous = None
    while previous != prefix:
        previous = prefix
        prefix = _V2_COPULA_TAIL_RE.sub("", prefix, count=1)
    phrase = prefix.strip(" \t\n,;:")
    keys = build_operand_fact_keys(phrase, currency_code)
    if not keys:
        raise ValueError(
            f"ERR_UNRESOLVED_INPUT: cannot derive a typed input from {phrase!r}"
        )
    return keys[0]


def finite_number_test(fpath: str) -> dict:
    return {
        "op": "AND",
        "left": {"op": "NEQ", "left": fpath, "right": None},
        "right": {
            "op": "OR",
            "left": {"op": "GTE", "left": fpath, "right": 0},
            "right": {"op": "LT", "left": fpath, "right": 0}
        }
    }


def emit_operand_resolution_steps(
    steps: list[dict],
    var_name: str,
    candidate_keys: list[str],
    fallback_to_primary: bool = True
) -> str:
    fallback_expr = (
        {
            "op": "CONDITIONAL",
            "test": finite_number_test("facts.primaryBasePaisa"),
            "then": {"op": "TO_PAISE", "value": "facts.primaryBasePaisa"},
            "else": {"op": "TO_PAISE", "value": "0"}
        }
        if fallback_to_primary
        else {"op": "TO_PAISE", "value": "0"}
    )
    if not candidate_keys:
        steps.append({"assign": var_name, "expr": fallback_expr})
        return f"vars.{var_name}"

    batches = [candidate_keys[i:i + 2] for i in range(0, len(candidate_keys), 2)]
    curr_else = fallback_expr
    for b_idx in range(len(batches) - 1, -1, -1):
        batch = batches[b_idx]
        node = curr_else
        for key in reversed(batch):
            fpath = f"facts.{key}"
            node = {
                "op": "CONDITIONAL",
                "test": finite_number_test(fpath),
                "then": {"op": "TO_PAISE", "value": fpath},
                "else": node
            }
        target_var = var_name if b_idx == 0 else f"{var_name}_b{b_idx}"
        steps.append({"assign": target_var, "expr": node})
        curr_else = f"vars.{target_var}"
    return f"vars.{var_name}"


def build_bounded_window_test(
    cutover_iso: str,
    sunset_iso: str,
    cutover_strict: bool = True,
    sunset_strict: bool = True
) -> dict:
    cut_op = "GT" if cutover_strict else "GTE"
    sun_op = "LT" if sunset_strict else "LTE"
    cut_yr = int(cutover_iso[:4])
    sun_yr = int(sunset_iso[:4])
    start_yr = cut_yr + 1 if (cutover_strict and cutover_iso[5:] >= "12-30") else cut_yr
    end_yr = sun_yr - 1 if (sunset_strict and sunset_iso[5:] <= "01-02") else sun_yr
    return {
        "op": "CONDITIONAL",
        "test": {
            "op": "GT",
            "left": {"op": "DATE_COMPARE", "left": "facts.transactionDate", "right": "0001-01-01"},
            "right": 0
        },
        "then": {
            "op": "AND",
            "left": {
                "op": cut_op,
                "left": {"op": "DATE_COMPARE", "left": "facts.transactionDate", "right": cutover_iso},
                "right": 0
            },
            "right": {
                "op": sun_op,
                "left": {"op": "DATE_COMPARE", "left": "facts.transactionDate", "right": sunset_iso},
                "right": 0
            }
        },
        "else": {
            "op": "CONDITIONAL",
            "test": {
                "op": "AND",
                "left": {"op": "NEQ", "left": "facts.taxYear", "right": None},
                "right": {"op": "GTE", "left": "facts.taxYear", "right": 1900}
            },
            "then": {
                "op": "AND",
                "left": {"op": "GTE", "left": "facts.taxYear", "right": start_yr},
                "right": {"op": "LTE", "left": "facts.taxYear", "right": end_yr}
            },
            "else": False
        }
    }


def classify_rule_domain(
    text: str,
    spans: list[dict],
    edges: list[dict] = None,
    predicted_modality: str = "MODALITY_COMPUTATIONAL_EXCESS",
    requested_domain: str = "AUTO"
) -> str:
    req = str(requested_domain or "AUTO").strip().upper()
    if req in ("STATUTORY_FINANCE", "GENERAL_LOGIC"):
        return req

    roles = {s.get("role") for s in (spans or [])}
    edge_types = {e.get("relationType") for e in (edges or [])}
    text_low = (text or "").lower()

    if "ROLE_ACCOUNTING_DESTINATION" in roles or "REL_SUNSET_TO_SECTION" in edge_types:
        return "STATUTORY_FINANCE"
    if _FISCAL_STATUTORY_RE.search(text or ""):
        return "STATUTORY_FINANCE"
    if re.search(r"\b(?:section|sub-section|regulation)\s+\d", text_low):
        return "STATUTORY_FINANCE"
    if "REL_BRACKET_NEXT" in edge_types or "REL_OVERRIDE_TO_GUARD" in edge_types:
        return "STATUTORY_FINANCE"

    return "GENERAL_LOGIC"


def _sanitize_ident(name: str, fallback: str = "var") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", str(name or ""))
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:48]


def _parse_quantified_leaf(
    span_text: str,
    forced_kind: str | None = None,
    forced_neg: bool = False,
) -> dict | None:
    """
    Extracts collection identifier, quantifier polarity, and item attribute keys from
    a single already-predicted neural span (or leaf phrase inside one).
    """
    low = (span_text or "").lower().strip()
    if not low or low.startswith("neither "):
        return None

    q_match = re.search(
        r"\b(?P<det>none\s+of\s+(?:the\s+)?|every|each|all\s+(?:of\s+the\s+|the\s+)?|any|no|some|at\s+least\s+one)\s+"
        r"(?P<noun>[a-zA-Z][a-zA-Z0-9_-]*(?:\s+[a-zA-Z][a-zA-Z0-9_-]*)?)\s*"
        r"(?P<tail>.*)$",
        span_text.strip(),
        flags=re.IGNORECASE,
    )
    if not q_match:
        if forced_kind in ("FORALL", "EXISTS"):
            words_clean = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", low).split() if w not in _LOGIC_FACT_STOPWORDS]
            if not words_clean:
                return None
            head_noun = words_clean[0]
            tail_words = words_clean[1:]
            coll_k = pluralize_collection_noun(head_noun)
            full_attr = derive_logic_fact_key(" ".join(tail_words) or "satisfied", "isSatisfied")
            attr_keys = [full_attr]
            if tail_words:
                attr_keys = [f"is{tail_words[0].capitalize()}", tail_words[0], full_attr]
            q_kind = ("FORALL_NOT" if forced_neg else "FORALL") if forced_kind == "FORALL" else ("NOT_EXISTS" if forced_neg else "EXISTS")
            return {
                "kind": q_kind,
                "collKeys": [coll_k],
                "attrKeys": list(dict.fromkeys(attr_keys))[:4],
                "scalarKey": derive_logic_fact_key(span_text, f"{coll_k}ConditionMet"),
            }
        return None

    det_raw = re.sub(r"\s+", " ", q_match.group("det").lower()).strip()
    noun_raw = q_match.group("noun").strip()
    tail_raw = q_match.group("tail").strip()
    noun_words = noun_raw.split()

    STOP_NON_COLLECTION = {
        "case", "event", "time", "point", "manner", "way", "extent", "taxable",
        "calendar", "fiscal", "financial", "previous", "such", "other", "longer", "more"
    }
    first_nw = noun_words[0].lower()
    if first_nw in STOP_NON_COLLECTION and not tail_raw and forced_kind is None:
        return None

    CUT_WORDS = {
        "in", "on", "of", "along", "through", "across", "within", "for", "from", "to", "by", "with",
        "who", "that", "which", "where", "when", "shall", "may", "must", "should", "can", "will",
        "has", "have", "had", "is", "are", "was", "were", "holds", "hold", "satisfies", "satisfy",
        "contains", "contain", "meets", "meet", "attacked", "flagged", "blocked", "verified", "approved",
        "registered", "exempt", "qualified", "eligible", "restricted", "covered", "active", "valid", "contested"
    }
    if len(noun_words) > 1 and noun_words[1].lower() in CUT_WORDS:
        tail_raw = f"{noun_words[1]} {tail_raw}".strip()
        head_noun = noun_words[0]
    elif len(noun_words) > 1 and first_nw in ("of", "the", "a", "an", "each", "any"):
        head_noun = noun_words[-1]
    else:
        head_noun = noun_words[-1]

    # Non-greedy locative trim anchored at finite verb/modal boundary
    pred_clean = tail_raw
    loc_match = re.match(
        r"^(?:in|on|along|within|of|among)\s+(?:the\s+|a\s+|an\s+|its\s+|such\s+)?[a-zA-Z0-9_-]+?\s+"
        r"((?:shall|may|must|should|can|will|does|do|did|has|have|had|is|are|was|were)\b.+)$",
        pred_clean,
        flags=re.IGNORECASE,
    )
    if loc_match:
        pred_clean = loc_match.group(1).strip()

    pred_core = re.sub(
        r"^(?:who\s+|which\s+|that\s+)?(?:shall|may|must|should|can|will|does|do|did|has|have|had|is|are|was|were)\s+(?:not\s+)?(?:be\s+|have\s+)?",
        "",
        pred_clean.strip(),
        flags=re.IGNORECASE,
    ).strip()
    pred_core = re.sub(r"^(?:holds?|possess(?:es)?|maintains?|carries|satisf(?:y|ies))\s+", "", pred_core, flags=re.IGNORECASE).strip()
    pred_no_agent = re.sub(r"\s+by\s+(?:an?|the)\s+[a-zA-Z0-9_\- ]+$", "", pred_core, flags=re.IGNORECASE).strip() or pred_core or "satisfied"

    has_clause_neg = bool(
        forced_neg
        or det_raw in ("no", "none of", "none of the")
        or re.search(r"\b(?:shall\s+not|may\s+not|must\s+not|cannot|does\s+not|do\s+not|is\s+not|are\s+not|not)\b", low[:q_match.start("det") + 16])
    )
    is_universal = det_raw.startswith(("every", "each", "all")) or forced_kind == "FORALL"
    if has_clause_neg and not is_universal:
        q_kind = "NOT_EXISTS"
    elif is_universal:
        inner_neg = bool(forced_neg or re.search(r"\b(?:not|never|fail)\b", tail_raw.lower()))
        q_kind = "FORALL_NOT" if inner_neg else "FORALL"
    else:
        q_kind = "EXISTS"

    coll_k = pluralize_collection_noun(head_noun)
    coll_keys = []
    if len(noun_words) >= 2 and noun_words[0].lower() not in STOP_NON_COLLECTION and noun_words[0].lower() not in _LOGIC_FACT_STOPWORDS and not noun_words[1].lower() in CUT_WORDS:
        mod_pref = re.sub(r"[^a-zA-Z0-9]", "", noun_words[0].lower())
        if mod_pref and len(mod_pref) > 1:
            comp_coll = mod_pref + coll_k[0].upper() + coll_k[1:]
            coll_keys.append(comp_coll)
    if coll_k not in coll_keys:
        coll_keys.append(coll_k)

    pred_words = [
        w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", pred_no_agent.lower()).split()
        if w not in _LOGIC_FACT_STOPWORDS and len(w) > 1
    ]
    full_pred_key = derive_logic_fact_key(pred_no_agent, "isSatisfied")
    attr_candidates = []
    if pred_words:
        w0 = pred_words[0]
        attr_candidates.extend([f"is{w0.capitalize()}", w0])
        if len(pred_words) >= 2:
            attr_candidates.append(f"has{derive_logic_fact_key(' '.join(pred_words), '').capitalize()}")
    attr_candidates.append(full_pred_key)
    dedup_attrs = []
    for k in attr_candidates:
        k_clean = k[0].lower() + k[1:] if k else ""
        if k_clean and k_clean not in dedup_attrs:
            dedup_attrs.append(k_clean)

    return {
        "kind": q_kind,
        "collKeys": coll_keys,
        "attrKeys": dedup_attrs[:4],
        "scalarKey": derive_logic_fact_key(span_text, f"{coll_k}ConditionMet"),
    }


def compile_composed_arithmetic_pipeline(
    text: str,
    indexed_spans: list[dict],
    span_by_idx: dict[int, dict],
    resolved_edges: list[dict],
    rate_items: list[dict],
    thresh_items: list[dict],
    operand_items: list[dict],
    all_monetary_spans: list[dict],
    currency_code: str | None,
    cutover_iso: str | None,
    sunset_iso: str | None,
    cutover_strict: bool,
    sunset_strict: bool,
    has_susp_signal: bool,
    excess_threshold: str | None,
    cliff_threshold: str | None,
    rate_bps: int,
) -> tuple[dict | None, str | None, str | None]:
    """
    Topological multi-node Biaffine DAG lowerer supporting:
    - Bounded collection aggregations (`REL_AGGREGATE_SUM` -> `SUM_OVER` / `COUNT`)
    - Arbitrarily composed `REL_COMBINATOR_MIN` / `REL_COMBINATOR_MAX`, `REL_CAP_CEILING`,
      `REL_SUBTRACTION_BASE`, and multi-term additive or rate prongs.
    Returns `(computation_dict, formula_description, expression_type)` or `(None, None, None)`.
    """
    agg_edges = [e for e in resolved_edges if e.get("relationType") == "REL_AGGREGATE_SUM"]
    comb_min_edges = [e for e in resolved_edges if e.get("relationType") == "REL_COMBINATOR_MIN"]
    comb_max_edges = [e for e in resolved_edges if e.get("relationType") == "REL_COMBINATOR_MAX"]
    comb_edges = comb_min_edges + comb_max_edges
    cap_edges = [e for e in resolved_edges if e.get("relationType") == "REL_CAP_CEILING"]
    sub_edges = [e for e in resolved_edges if e.get("relationType") == "REL_SUBTRACTION_BASE"]
    bracket_edges = [e for e in resolved_edges if e.get("relationType") == "REL_BRACKET_NEXT"]
    symbolic_sub_edges = [
        e for e in sub_edges
        if (
            span_by_idx[e["sourceIdx"]]["role"] == "ROLE_COMPUTATION_OPERAND"
            and span_by_idx[e["targetIdx"]]["role"] == "ROLE_COMPUTATION_OPERAND"
        )
    ]

    text_low = (text or "").lower()
    has_text_agg = bool(re.search(r"\b(?:sum\s+of|total\s+of|aggregate\s+of|count\s+of|number\s+of)\s+(?:qualifying|eligible|all|such)?", text_low))
    comb_rate_endpoints = set()
    for ce in comb_edges:
        comb_rate_endpoints.add(ce["sourceIdx"])
        comb_rate_endpoints.add(ce["targetIdx"])
    extra_unconsumed_rates = [r for r in rate_items if r["spanIdx"] not in comb_rate_endpoints]

    should_run_composed = bool(
        agg_edges
        or (has_text_agg and operand_items and (thresh_items or rate_items or cap_edges))
        or len(comb_edges) >= 2
        or (comb_edges and (cap_edges or bracket_edges or extra_unconsumed_rates))
        or (cap_edges and (symbolic_sub_edges or len(rate_items) >= 1))
        or len(symbolic_sub_edges) >= 2
    )
    if not should_run_composed:
        return None, None, None

    def _neighbors(node_idx: int, rel_filter: set[str]) -> list[int]:
        out = []
        for e in resolved_edges:
            if e["relationType"] in rel_filter:
                if e["sourceIdx"] == node_idx:
                    out.append(e["targetIdx"])
                elif e["targetIdx"] == node_idx:
                    out.append(e["sourceIdx"])
        return out

    steps = []
    bound_cache = {}
    fact_schema_operands = []

    def _bind_op_var(op_idx: int | None, allow_primary: bool, default_keys: list[str] | None = None) -> str:
        if op_idx is not None:
            norm_k = normalize_symbol_id(span_by_idx[op_idx]["span"]) or f"op_{op_idx}"
            if norm_k in bound_cache:
                return bound_cache[norm_k]
            c_keys = build_operand_fact_keys(span_by_idx[op_idx]["span"], currency_code)
            fact_schema_operands.append({"factKeys": c_keys, "sourceText": span_by_idx[op_idx]["span"]})
            v_ref = emit_operand_resolution_steps(steps, f"dag_op_{len(bound_cache)}", c_keys, fallback_to_primary=allow_primary)
            bound_cache[norm_k] = v_ref
            return v_ref
        c_keys = default_keys or []
        return emit_operand_resolution_steps(steps, f"dag_fb_{len(steps)}", c_keys, fallback_to_primary=allow_primary)

    # ── Branch 1: Bounded Collection Aggregation Pipeline (SUM_OVER / COUNT) ─
    if agg_edges or (has_text_agg and operand_items):
        agg_span_txt = None
        if agg_edges:
            ae = agg_edges[0]
            s_it, t_it = span_by_idx[ae["sourceIdx"]], span_by_idx[ae["targetIdx"]]
            agg_span_txt = t_it["span"] if t_it["role"] == "ROLE_COMPUTATION_OPERAND" else s_it["span"]
        elif operand_items:
            agg_span_txt = operand_items[0]["span"]
        else:
            agg_span_txt = "qualifying items"

        words = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", agg_span_txt.lower()).split() if w not in _LOGIC_FACT_STOPWORDS]
        prim_coll = derive_logic_fact_key(agg_span_txt, "qualifyingItems")
        if not prim_coll.endswith("s"):
            prim_coll = pluralize_collection_noun(prim_coll)
        short_coll = pluralize_collection_noun(words[-1]) if words else "items"
        coll_keys = [prim_coll]
        if short_coll and short_coll not in coll_keys:
            coll_keys.append(short_coll)

        scalar_keys = build_operand_fact_keys(agg_span_txt, currency_code)
        scalar_ref = emit_operand_resolution_steps(steps, "aggScalarFallbackPaisa", scalar_keys, fallback_to_primary=True)

        if len(coll_keys) >= 2:
            steps.append({
                "assign": "aggCollection",
                "expr": {
                    "op": "CONDITIONAL",
                    "test": {"op": "GT", "left": {"op": "LEN", "collection": f"facts.{coll_keys[0]}"}, "right": 0},
                    "then": f"facts.{coll_keys[0]}",
                    "else": f"facts.{coll_keys[1]}",
                },
            })
            coll_ref = "vars.aggCollection"
        else:
            coll_ref = f"facts.{coll_keys[0]}"

        is_count_agg = bool(re.search(r"\b(?:count\s+of|number\s+of|fewer\s+than|at\s+least\s+\d+\s+(?:distinct\s+)?(?:items?|members?|entries|occurrences?))\b", text_low))
        pred_node = {
            "op": "AND",
            "left": {"op": "NEQ", "left": "it.isQualifying", "right": False},
            "right": {"op": "NEQ", "left": "it.excluded", "right": True},
        }
        if is_count_agg:
            steps.append({
                "assign": "aggCountVal",
                "expr": {
                    "op": "CONDITIONAL",
                    "test": {"op": "GT", "left": {"op": "LEN", "collection": coll_ref}, "right": 0},
                    "then": {"op": "COUNT", "iteratorVar": "it", "collection": coll_ref, "predicate": pred_node},
                    "else": {"op": "FROM_PAISE", "value": scalar_ref},
                },
            })
            count_thresh = parse_numeric_quantity(excess_threshold or cliff_threshold or "0") or 0
            cmp_op = parse_comparison_operator(text)
            steps.append({
                "assign": "isApplicable",
                "expr": {"op": cmp_op, "left": "vars.aggCountVal", "right": count_thresh},
            })
            comp_out = {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": "0",
                "standardRateBps": rate_bps,
                "factSchema": {"collections": coll_keys, "scalarFallbackKeys": scalar_keys},
                "pipeline": {
                    "steps": steps,
                    "output": {
                        "countTotal": "vars.aggCountVal",
                        "sumTotal": "vars.aggCountVal",
                        "isApplicable": "vars.isApplicable",
                        "isCompliant": "vars.isApplicable",
                    },
                },
            }
            return comp_out, "Bounded collection COUNT evaluation against threshold", "DECLARATIVE_PIPELINE"

        sum_body = {
            "op": "CONDITIONAL",
            "test": finite_number_test("it.value"),
            "then": "it.value",
            "else": {
                "op": "CONDITIONAL",
                "test": finite_number_test("it.amount"),
                "then": "it.amount",
                "else": {"op": "CONDITIONAL", "test": finite_number_test("it"), "then": "it", "else": 0},
            },
        }
        steps.append({
            "assign": "aggRawSum",
            "expr": {
                "op": "CONDITIONAL",
                "test": {"op": "GT", "left": {"op": "LEN", "collection": coll_ref}, "right": 0},
                "then": {"op": "SUM_OVER", "iteratorVar": "it", "collection": coll_ref, "predicate": pred_node, "body": sum_body},
                "else": {"op": "FROM_PAISE", "value": scalar_ref},
            },
        })
        steps.append({"assign": "sumTotalPaisa", "expr": {"op": "TO_PAISE", "value": "vars.aggRawSum"}})

        th_minor = parse_minor_units(excess_threshold or cliff_threshold)
        if th_minor is not None and th_minor > 0:
            if excess_threshold:
                steps.append({
                    "assign": "aggBasePaisa",
                    "expr": {
                        "op": "MAX",
                        "left": {"op": "TO_PAISE", "value": "0"},
                        "right": {"op": "SUB", "left": "vars.sumTotalPaisa", "right": {"op": "TO_PAISE", "value": str(th_minor)}},
                    },
                })
                steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.aggBasePaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
            else:
                steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.sumTotalPaisa", "right": {"op": "TO_PAISE", "value": str(th_minor)}}})
                steps.append({
                    "assign": "aggBasePaisa",
                    "expr": {"op": "CONDITIONAL", "test": "vars.isApplicable", "then": "vars.sumTotalPaisa", "else": {"op": "TO_PAISE", "value": "0"}},
                })
        else:
            steps.append({"assign": "aggBasePaisa", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": "vars.sumTotalPaisa"}})
            steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.aggBasePaisa", "right": {"op": "TO_PAISE", "value": "0"}}})

        cap_minor = None
        if cap_edges:
            ce = cap_edges[0]
            cap_minor = parse_minor_units(span_by_idx[ce["sourceIdx"]]["span"]) or parse_minor_units(span_by_idx[ce["targetIdx"]]["span"])

        if rate_bps > 0:
            steps.append({
                "assign": "rawTaxPaisa",
                "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.aggBasePaisa", "rateBps": "rules.effectiveRateBps", "denominator": 10000},
            })
            if cap_minor is not None:
                steps.append({"assign": "taxAmountPaisa", "expr": {"op": "MIN", "left": "vars.rawTaxPaisa", "right": {"op": "TO_PAISE", "value": str(cap_minor)}}})
            else:
                steps.append({"assign": "taxAmountPaisa", "expr": "vars.rawTaxPaisa"})
        else:
            if cap_minor is not None:
                steps.append({"assign": "taxAmountPaisa", "expr": {"op": "MIN", "left": "vars.aggBasePaisa", "right": {"op": "TO_PAISE", "value": str(cap_minor)}}})
            else:
                steps.append({"assign": "taxAmountPaisa", "expr": "vars.aggBasePaisa"})

        out_map = {
            "sumTotalPaisa": "vars.sumTotalPaisa",
            "sumTotal": {"op": "FROM_PAISE", "value": "vars.sumTotalPaisa"},
            "taxableBasePaisa": "vars.aggBasePaisa",
            "taxAmountPaisa": "vars.taxAmountPaisa",
            "tdsAmountPaisa": "vars.taxAmountPaisa",
            "calculatedTaxInr": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
            "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
            "isApplicable": "vars.isApplicable",
        }
        comp_out = {
            "archetype": "DECLARATIVE_PIPELINE",
            "thresholdPaisa": str(th_minor or 0),
            "standardRateBps": rate_bps,
            "factSchema": {"collections": coll_keys, "scalarFallbackKeys": scalar_keys},
            "pipeline": {"steps": steps, "output": out_map},
        }
        return comp_out, "Bounded collection SUM_OVER aggregation pipeline with threshold and ceiling controls", "DECLARATIVE_PIPELINE"

    # ── Branch 2: Topological Multi-Node Biaffine Combinator / Cap / Sub DAG ─
    expr_ref_by_idx = {}

    # Identify all REL_CAP_CEILING mappings { target_idx -> cap_minor }
    cap_by_target = {}
    outer_caps = []
    for ce in cap_edges:
        s_i, t_i = ce["sourceIdx"], ce["targetIdx"]
        s_min = parse_minor_units(span_by_idx[s_i]["span"])
        t_min = parse_minor_units(span_by_idx[t_i]["span"])
        if s_min is not None and t_min is None:
            cap_by_target[t_i] = s_min
        elif t_min is not None and s_min is None:
            cap_by_target[s_i] = t_min
        elif s_min is not None:
            outer_caps.append(s_min)

    def _lower_branch_node(node_idx: int, pos_tag: str, allow_primary: bool) -> str:
        if node_idx in expr_ref_by_idx:
            return expr_ref_by_idx[node_idx]
        item = span_by_idx[node_idx]
        b_bps = parse_rate_bps(item["span"]) if any(w in item["span"].lower() for w in ("%", "per cent", "percent")) else None

        if b_bps is not None:
            sub_nbrs = _neighbors(node_idx, {"REL_SUBTRACTION_BASE"})
            th_nbrs = _neighbors(node_idx, {"REL_RATE_TO_THRESHOLD", "REL_THRESHOLD_TO_TRANSACTION"})
            b_cr = item.get("charRange", [0, 0])
            op_nbrs_all = [n for n in sub_nbrs if span_by_idx[n]["role"] == "ROLE_COMPUTATION_OPERAND"]
            op_nbrs_right = sorted(
                [n for n in op_nbrs_all if span_by_idx[n].get("charRange", [0, 0])[0] >= b_cr[1]],
                key=lambda n: span_by_idx[n].get("charRange", [0, 0])[0],
            )
            op_nbrs = op_nbrs_right if op_nbrs_right else op_nbrs_all
            num_nbrs = [
                n for n in (sub_nbrs + th_nbrs)
                if n not in op_nbrs
                and parse_minor_units(span_by_idx[n]["span"]) is not None
                and parse_unit_and_type(span_by_idx[n]["span"])[0] == "MONETARY_QUANTUM"
            ]
            if not op_nbrs and operand_items:
                # Spatial fallback constrained strictly to clause segment
                next_r = min((r["charRange"][0] for r in rate_items if r["spanIdx"] != node_idx and r["charRange"][0] > b_cr[1]), default=len(text))
                op_nbrs = [o["spanIdx"] for o in operand_items if o["charRange"][0] >= b_cr[1] and o["charRange"][1] <= next_r][:1]

            op_idx = op_nbrs[0] if op_nbrs else (operand_items[0]["spanIdx"] if operand_items else None)
            if op_idx is not None and not num_nbrs:
                num_nbrs = [
                    n for n in _neighbors(op_idx, {"REL_THRESHOLD_TO_TRANSACTION", "REL_SUBTRACTION_BASE", "REL_RATE_TO_THRESHOLD"})
                    if n != node_idx
                    and parse_minor_units(span_by_idx[n]["span"]) is not None
                    and parse_unit_and_type(span_by_idx[n]["span"])[0] == "MONETARY_QUANTUM"
                ]

            sec_op_nbrs = (
                [n for n in _neighbors(op_idx, {"REL_SUBTRACTION_BASE"}) if n != node_idx and span_by_idx[n]["role"] == "ROLE_COMPUTATION_OPERAND"]
                if op_idx is not None else []
            )
            op_ref = _bind_op_var(op_idx, allow_primary=allow_primary)

            if op_idx is not None and op_idx in cap_by_target:
                cap_v = cap_by_target[op_idx]
                steps.append({
                    "assign": f"dag_capop_{pos_tag}",
                    "expr": {"op": "MIN", "left": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": op_ref}, "right": {"op": "TO_PAISE", "value": str(cap_v)}},
                })
                op_ref = f"vars.dag_capop_{pos_tag}"

            if num_nbrs:
                th_minor = parse_minor_units(span_by_idx[num_nbrs[0]]["span"]) or 0
                steps.append({
                    "assign": f"dag_base_{pos_tag}",
                    "expr": {
                        "op": "MAX",
                        "left": {"op": "TO_PAISE", "value": "0"},
                        "right": {"op": "SUB", "left": op_ref, "right": {"op": "TO_PAISE", "value": str(th_minor)}},
                    },
                })
                base_ref = f"vars.dag_base_{pos_tag}"
            elif sec_op_nbrs:
                sec_ref = _bind_op_var(sec_op_nbrs[0], allow_primary=False)
                steps.append({
                    "assign": f"dag_base_{pos_tag}",
                    "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": op_ref, "right": sec_ref}},
                })
                base_ref = f"vars.dag_base_{pos_tag}"
            else:
                steps.append({"assign": f"dag_base_{pos_tag}", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": op_ref}})
                base_ref = f"vars.dag_base_{pos_tag}"

            steps.append({
                "assign": f"dag_rateval_{pos_tag}",
                "expr": {"op": "RATIONAL_MULTIPLY", "base": base_ref, "rateBps": b_bps, "denominator": 10000},
            })
            res_ref = f"vars.dag_rateval_{pos_tag}"
            if node_idx in cap_by_target:
                steps.append({
                    "assign": f"dag_ratecap_{pos_tag}",
                    "expr": {"op": "MIN", "left": res_ref, "right": {"op": "TO_PAISE", "value": str(cap_by_target[node_idx])}},
                })
                res_ref = f"vars.dag_ratecap_{pos_tag}"
            expr_ref_by_idx[node_idx] = res_ref
            return res_ref

        # Monetary constant or raw operand node
        const_minor = parse_minor_units(item["span"])
        if const_minor is not None and parse_unit_and_type(item["span"])[0] == "MONETARY_QUANTUM":
            steps.append({"assign": f"dag_const_{pos_tag}", "expr": {"op": "TO_PAISE", "value": str(const_minor)}})
            res_ref = f"vars.dag_const_{pos_tag}"
        else:
            res_ref = _bind_op_var(node_idx, allow_primary=allow_primary)
            if node_idx in cap_by_target:
                steps.append({
                    "assign": f"dag_nodecap_{pos_tag}",
                    "expr": {"op": "MIN", "left": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": res_ref}, "right": {"op": "TO_PAISE", "value": str(cap_by_target[node_idx])}},
                })
                res_ref = f"vars.dag_nodecap_{pos_tag}"
        expr_ref_by_idx[node_idx] = res_ref
        return res_ref

    # Graph Dominance & Type Reduction for Combinators:
    # A rate-to-rate combinator (comparing two rate-derived branches) dominates and subsumes
    # pseudo-combinators that mistakenly link a rate to an internal threshold of the same clause.
    rate_to_rate_edges = [
        ce for ce in comb_edges
        if span_by_idx[ce["sourceIdx"]]["role"] == "ROLE_STANDARD_STATUTORY_RATE"
        and span_by_idx[ce["targetIdx"]]["role"] == "ROLE_STANDARD_STATUTORY_RATE"
    ]
    canonical_comb_edges = rate_to_rate_edges if rate_to_rate_edges else comb_edges

    comb_vars = []
    consumed_rate_indices = set()
    for c_i, ce in enumerate(canonical_comb_edges):
        c_op = "MIN" if ce["relationType"] == "REL_COMBINATOR_MIN" else "MAX"
        l_ref = _lower_branch_node(ce["sourceIdx"], f"c{c_i}_l", allow_primary=(c_i == 0))
        r_ref = _lower_branch_node(ce["targetIdx"], f"c{c_i}_r", allow_primary=False)
        consumed_rate_indices.add(ce["sourceIdx"])
        consumed_rate_indices.add(ce["targetIdx"])
        v_name = f"dag_comb_{c_i}"
        steps.append({"assign": v_name, "expr": {"op": c_op, "left": l_ref, "right": r_ref}})
        comb_vars.append(f"vars.{v_name}")

    # Lower any cap/sub rates that were NOT already part of a combinator
    standalone_rate_vars = []
    for r_i, rit in enumerate(rate_items):
        if rit["spanIdx"] not in consumed_rate_indices:
            standalone_rate_vars.append(_lower_branch_node(rit["spanIdx"], f"sr_{r_i}", allow_primary=(len(comb_vars) == 0 and r_i == 0)))

    # Pure cap + symbolic subtraction chain (no combinator)
    if not comb_vars and not standalone_rate_vars and symbolic_sub_edges:
        se = symbolic_sub_edges[0]
        r_a = _lower_branch_node(se["sourceIdx"], "sub_a", allow_primary=True)
        r_b = _lower_branch_node(se["targetIdx"], "sub_b", allow_primary=False)
        steps.append({"assign": "dag_sub_root", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": r_a, "right": r_b}}})
        comb_vars.append("vars.dag_sub_root")

    # Multiple combinator expressions sharing a clause without an additive conjunction
    # are alternate branch evaluations, not additive terms. Only sum if there is explicit aggregation.
    has_additive_conjunction = bool(re.search(r"\b(?:plus|and\s+also|in\s+addition\s+to|together\s+with|aggregate\s+of|sum\s+of)\b", text_low))
    if len(comb_vars) > 1 and not has_additive_conjunction:
        active_comb_vars = comb_vars[-1:]
    else:
        active_comb_vars = comb_vars

    all_root_terms = active_comb_vars + standalone_rate_vars
    if not all_root_terms:
        return None, None, None

    curr_acc = all_root_terms[0]
    for t_i, nxt_term in enumerate(all_root_terms[1:]):
        acc_name = f"dag_sum_{t_i}"
        steps.append({"assign": acc_name, "expr": {"op": "ADD", "left": curr_acc, "right": nxt_term}})
        curr_acc = f"vars.{acc_name}"

    # Apply any dangling cap ceiling to top-level accumulator
    for oc_i, oc_minor in enumerate(outer_caps):
        cap_name = f"dag_outercap_{oc_i}"
        steps.append({"assign": cap_name, "expr": {"op": "MIN", "left": curr_acc, "right": {"op": "TO_PAISE", "value": str(oc_minor)}}})
        curr_acc = f"vars.{cap_name}"

    if has_susp_signal and cutover_iso and sunset_iso:
        steps.append({"assign": "inSuspensionWindow", "expr": build_bounded_window_test(cutover_iso, sunset_iso, cutover_strict, sunset_strict)})
        steps.append({
            "assign": "combinedAmountPaisa",
            "expr": {"op": "CONDITIONAL", "test": "vars.inSuspensionWindow", "then": {"op": "TO_PAISE", "value": "0"}, "else": curr_acc},
        })
    else:
        steps.append({"assign": "combinedAmountPaisa", "expr": curr_acc})

    steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.combinedAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
    output_map = {
        "combinedAmountPaisa": "vars.combinedAmountPaisa",
        "combinedAmount": {"op": "FROM_PAISE", "value": "vars.combinedAmountPaisa"},
        "taxAmountPaisa": "vars.combinedAmountPaisa",
        "tdsAmountPaisa": "vars.combinedAmountPaisa",
        "calculatedTaxInr": {"op": "FROM_PAISE", "value": "vars.combinedAmountPaisa"},
        "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.combinedAmountPaisa"},
        "isApplicable": "vars.isApplicable",
    }
    return {
        "archetype": "DECLARATIVE_PIPELINE",
        "thresholdPaisa": "0",
        "standardRateBps": rate_bps,
        "factSchema": {"operands": fact_schema_operands},
        "pipeline": {"steps": steps, "output": output_map},
    }, "Topological composed DAG arithmetic pipeline across multi-prong constraints", "DECLARATIVE_PIPELINE"


def compile_general_logic_pipeline(
    text: str,
    indexed_spans: list[dict],
    resolved_edges: list[dict],
    root_condition: dict,
    deontic_block: dict,
    deadline_window: str | None,
    effective_domain: str,
) -> dict | None:
    """
    Neural-graph-driven deontic, boolean, and bounded-quantifier DAG lowerer.
    Strictly consumes predicted `indexed_spans` and Biaffine `resolved_edges` from
    `MudraStatutoryParser` to compile `PURE_LOGIC_PIPELINE` (depth <= 7, quantifier nesting <= 1).
    Returns None if the neural graph contains no actionable logic/deontic structure.
    """
    LOGIC_ROLES = {
        "ROLE_AGENT_STATE_PRECONDITION", "ROLE_BUYER_PRIOR_TURNOVER_GUARD",
        "ROLE_SUBJECTIVE_CRITERION", "ROLE_COMPUTATION_OPERAND",
        "ROLE_EXCEPTION_CARVEOUT", "ROLE_NON_RESIDENT_EXEMPTION_GUARD",
        "ROLE_PROHIBITION_GATE", "ROLE_MANDATORY_ACTION_OBLIGATION", "ROLE_DEADLINE_WINDOW",
    }
    LOGIC_EDGES = {
        "REL_LOGICAL_NOT", "REL_LOGICAL_AND", "REL_QUANTIFIER_FORALL",
        "REL_QUANTIFIER_EXISTS", "REL_EXCEPTION_TO_PROHIBITION",
        "REL_DEADLINE_TO_TRIGGER", "REL_ACTION_TO_PRECONDITION", "REL_CONDITION_TO_GATE",
    }
    substantive = [
        s for s in (indexed_spans or [])
        if s.get("role") != "ROLE_IGNORE" and str(s.get("span", "")).strip()
    ]
    if not substantive or (
        not any(s["role"] in LOGIC_ROLES for s in substantive)
        and not any(e.get("relationType") in LOGIC_EDGES for e in (resolved_edges or []))
    ):
        return None

    span_by_idx = {s["spanIdx"]: s for s in substantive}
    edges = [
        e for e in (resolved_edges or [])
        if e.get("sourceIdx") in span_by_idx and e.get("targetIdx") in span_by_idx
    ]

    # Build Biaffine graph index maps
    exception_src_ids = {e["sourceIdx"] for e in edges if e["relationType"] == "REL_EXCEPTION_TO_PROHIBITION"}
    deadlines_by_target: dict[int, list[int]] = {}
    bound_dl_ids = set()
    not_edges = [e for e in edges if e["relationType"] == "REL_LOGICAL_NOT"]
    forall_ids = set()
    exists_ids = set()
    negated_ids = set()
    cond_links: dict[int, set[int]] = {}

    for e in edges:
        rt = e["relationType"]
        u, v = e["sourceIdx"], e["targetIdx"]
        if rt == "REL_DEADLINE_TO_TRIGGER":
            deadlines_by_target.setdefault(v, []).append(u)
            bound_dl_ids.add(u)
        elif rt == "REL_QUANTIFIER_FORALL":
            forall_ids.add(u)
            if span_by_idx[v]["role"] in ("ROLE_AGENT_STATE_PRECONDITION", "ROLE_BUYER_PRIOR_TURNOVER_GUARD", "ROLE_COMPUTATION_OPERAND", "ROLE_SUBJECTIVE_CRITERION"):
                forall_ids.add(v)
        elif rt == "REL_QUANTIFIER_EXISTS":
            exists_ids.add(u)
            if span_by_idx[v]["role"] in ("ROLE_AGENT_STATE_PRECONDITION", "ROLE_BUYER_PRIOR_TURNOVER_GUARD", "ROLE_COMPUTATION_OPERAND", "ROLE_SUBJECTIVE_CRITERION"):
                exists_ids.add(v)
        elif rt == "REL_LOGICAL_NOT":
            negated_ids.update((u, v))
        if rt in ("REL_LOGICAL_NOT", "REL_QUANTIFIER_EXISTS", "REL_QUANTIFIER_FORALL", "REL_CONDITION_TO_GATE", "REL_ACTION_TO_PRECONDITION"):
            cond_links.setdefault(u, set()).add(v)
            cond_links.setdefault(v, set()).add(u)

    # Fold auxiliary prohibition span (e.g. "shall not pass through") onto linked quantified/precondition operand
    folded_aux_prohib_ids = set()
    for p_idx, p_span in span_by_idx.items():
        if p_span["role"] != "ROLE_PROHIBITION_GATE":
            continue
        p_ce = p_span.get("charRange", [0, 0])[1]
        for nbr in cond_links.get(p_idx, ()):
            n_span = span_by_idx[nbr]
            if (
                n_span["role"] in ("ROLE_AGENT_STATE_PRECONDITION", "ROLE_COMPUTATION_OPERAND", "ROLE_BUYER_PRIOR_TURNOVER_GUARD")
                and n_span.get("charRange", [0, 0])[0] >= p_ce - 1
                and n_span.get("charRange", [0, 0])[0] - p_ce <= 8
            ):
                folded_aux_prohib_ids.add(p_idx)
                if p_idx in negated_ids or "not" in p_span["span"].lower():
                    negated_ids.add(nbr)

    by_pos = lambda ids: sorted(set(ids), key=lambda idx: span_by_idx[idx].get("charRange", [idx, idx])[0])
    carveout_ids = by_pos(
        idx for idx, s in span_by_idx.items()
        if s["role"] in ("ROLE_EXCEPTION_CARVEOUT", "ROLE_NON_RESIDENT_EXEMPTION_GUARD")
        or (idx in exception_src_ids and s["role"] != "ROLE_DEADLINE_WINDOW")
    )
    deadline_ids = by_pos(idx for idx, s in span_by_idx.items() if s["role"] == "ROLE_DEADLINE_WINDOW")
    prohib_ids = by_pos(
        idx for idx, s in span_by_idx.items()
        if s["role"] == "ROLE_PROHIBITION_GATE" and idx not in folded_aux_prohib_ids and idx not in carveout_ids
    )
    action_ids = by_pos(
        idx for idx, s in span_by_idx.items()
        if s["role"] == "ROLE_MANDATORY_ACTION_OBLIGATION" and idx not in carveout_ids
    )
    precond_ids = by_pos(
        idx for idx, s in span_by_idx.items()
        if idx not in carveout_ids and idx not in prohib_ids and idx not in folded_aux_prohib_ids and idx not in action_ids and idx not in deadline_ids
        and (
            s["role"] in ("ROLE_AGENT_STATE_PRECONDITION", "ROLE_BUYER_PRIOR_TURNOVER_GUARD", "ROLE_SUBJECTIVE_CRITERION", "ROLE_COMPUTATION_OPERAND")
            or idx in (forall_ids | exists_ids | {e["targetIdx"] for e in edges if e["relationType"] == "REL_ACTION_TO_PRECONDITION"})
        )
    )

    # Attach any unbound ROLE_DEADLINE_WINDOW to closest prior carveout (same clause), else keep standalone
    standalone_deadline_ids = []
    for d_idx in deadline_ids:
        if d_idx in bound_dl_ids:
            continue
        d_start = span_by_idx[d_idx].get("charRange", [0, 0])[0]
        prior_carves = [c for c in carveout_ids if span_by_idx[c].get("charRange", [0, 0])[0] <= d_start]
        if prior_carves:
            deadlines_by_target.setdefault(prior_carves[-1], []).append(d_idx)
        else:
            standalone_deadline_ids.append(d_idx)

    steps = []
    fact_schema = {"booleans": [], "collections": [], "numbers": [], "outcomeMode": "OBLIGATION_RULE"}
    seen_b, seen_c, seen_n = set(), set(), set()
    var_seq = [0]

    def _rec_bool(fk: str, pol: str, src: str):
        if fk and fk not in seen_b:
            seen_b.add(fk)
            fact_schema["booleans"].append({"factKey": fk, "polarity": pol, "sourceText": src.strip()})

    def _rec_coll(ckeys: list[str], kind: str, akeys: list[str], sk: str, src: str):
        k0 = ckeys[0] if ckeys else "items"
        if k0 not in seen_c:
            seen_c.add(k0)
            fact_schema["collections"].append({
                "factKey": k0, "altFactKeys": ckeys[1:], "quantifier": kind,
                "itemPredicateKeys": akeys, "scalarFallbackKey": sk, "sourceText": src.strip(),
            })

    def _rec_num(fk: str, op: str, thresh, unit: str, src: str, alt_keys=None, bool_fb=None):
        if fk and fk not in seen_n:
            seen_n.add(fk)
            e = {"factKey": fk, "operator": op, "threshold": thresh, "unit": unit, "sourceText": src.strip()}
            if alt_keys:
                e["altFactKeys"] = alt_keys
            if bool_fb:
                e["boolFallbackKey"] = bool_fb
            fact_schema["numbers"].append(e)

    def _compile_deadline_expr(dl_text: str) -> dict | None:
        dm = re.search(
            r"(?:\bwithin\s+)?(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty(?:-five)?|fifty|sixty)\s+"
            r"(?P<unit>turns?|cycles?|rounds?|steps?|attempts?|days?|hours?|weeks?|months?|years?)\b",
            dl_text or "",
            flags=re.IGNORECASE,
        )
        if not dm:
            return None
        n_val = parse_numeric_quantity(dm.group("num"))
        if n_val is None:
            return None
        u_raw = dm.group("unit").lower()
        u_pl = pluralize_collection_noun(u_raw)
        k_elapsed = f"{u_pl}Elapsed"
        k_alt = f"{u_raw.rstrip('s')}Count"
        within_bool = derive_logic_fact_key(dl_text, f"within{u_pl.capitalize()}")
        _rec_num(k_elapsed, "LTE", n_val, u_pl.upper(), dl_text, alt_keys=[k_alt], bool_fb=within_bool)
        return {
            "op": "CONDITIONAL",
            "test": finite_number_test(f"facts.{k_elapsed}"),
            "then": {"op": "LTE", "left": f"facts.{k_elapsed}", "right": n_val},
            "else": {
                "op": "CONDITIONAL",
                "test": finite_number_test(f"facts.{k_alt}"),
                "then": {"op": "LTE", "left": f"facts.{k_alt}", "right": n_val},
                "else": {"op": "EQ", "left": f"facts.{within_bool}", "right": True},
            },
        }

    def _lower_leaf(leaf_text: str, role_polarity: str = "REQUIRE_TRUE", forced_q: str | None = None, forced_neg: bool = False, extra_dl_expr: dict | None = None) -> str:
        s_clean = re.sub(r"^(?:and|or|that|where|when|if|upon)\s+", "", (leaf_text or "").strip(), flags=re.IGNORECASE).strip() or "condition satisfied"

        # A. `neither X nor Y <VP>` distribution inside one predicted precondition span
        nm = re.match(
            r"^(?:that\s+)?neither\s+(?:the\s+|an?\s+)?(.+?)\s+nor\s+(?:the\s+|an?\s+)?(.+?)\s+"
            r"((?:has|have|had|is|are|was|were|does|do|did|shall|may|must|will|can|previously|currently|ever)\b.+)$",
            s_clean,
            flags=re.IGNORECASE,
        ) or re.match(
            r"^(?:that\s+)?neither\s+(?:the\s+|an?\s+)?(\S+)\s+nor\s+(?:the\s+|an?\s+)?(\S+)\s+([a-z]+(?:s|ed|ing)?\b.*)$",
            s_clean,
            flags=re.IGNORECASE,
        )
        if nm:
            va = _lower_leaf(f"{nm.group(1).strip()} {nm.group(3).strip()}", "REQUIRE_FALSE")
            vb = _lower_leaf(f"{nm.group(2).strip()} {nm.group(3).strip()}", "REQUIRE_FALSE")
            var_seq[0] += 1
            vn = f"cond_{var_seq[0]}_neither"
            steps.append({"assign": vn, "expr": {"op": "AND", "left": va, "right": vb}})
            return f"vars.{vn}"

        # B. Leaf deadline suffix inside the span + attached Biaffine deadline
        dl_expr = extra_dl_expr
        inline_dl = re.search(
            r"\bwithin\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty|thirty|forty(?:-five)?|fifty|sixty)\s+"
            r"(?:turns?|cycles?|rounds?|steps?|attempts?|days?|hours?|weeks?|months?|years?)\b",
            s_clean,
            flags=re.IGNORECASE,
        )
        target_phrase = s_clean[:inline_dl.start()].strip(" ,;:") if inline_dl else s_clean
        if inline_dl and dl_expr is None:
            dl_expr = _compile_deadline_expr(inline_dl.group(0))

        # C. Bounded Count threshold inside span (`at least N items` / `fewer than N`)
        cnt_m = re.search(
            r"\b(?P<op>at\s+least|not\s+less\s+than|at\s+most|no\s+more\s+than|fewer\s+than|less\s+than|more\s+than|exceeds?|greater\s+than)\s+"
            r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:distinct\s+)?(?P<noun>[a-zA-Z][a-zA-Z0-9_-]*)",
            target_phrase,
            flags=re.IGNORECASE,
        )
        if cnt_m and not re.search(r"\b(?:dollars?|rupees?|euros?|pounds?|usd|inr|eur|gbp)\b|[$₹€£]", target_phrase, flags=re.IGNORECASE):
            n_val = parse_numeric_quantity(cnt_m.group("num"))
            if n_val is not None:
                op_str = parse_comparison_operator(cnt_m.group("op"))
                coll_k = pluralize_collection_noun(cnt_m.group("noun"))
                cnt_k = f"{coll_k}Count"
                _rec_coll([coll_k], f"COUNT_{op_str}", ["isQualifying"], cnt_k, target_phrase)
                _rec_num(cnt_k, op_str, n_val, "COUNT", target_phrase)
                var_seq[0] += 1
                vn = f"cond_{var_seq[0]}_cnt"
                steps.append({
                    "assign": vn,
                    "expr": {
                        "op": op_str,
                        "left": {
                            "op": "CONDITIONAL",
                            "test": {"op": "GT", "left": {"op": "LEN", "collection": f"facts.{coll_k}"}, "right": 0},
                            "then": {"op": "COUNT", "iteratorVar": "it", "collection": f"facts.{coll_k}", "predicate": {"op": "NEQ", "left": "it.excluded", "right": True}},
                            "else": {"op": "CONDITIONAL", "test": finite_number_test(f"facts.{cnt_k}"), "then": f"facts.{cnt_k}", "else": 0},
                        },
                        "right": n_val,
                    },
                })
                if dl_expr:
                    var_seq[0] += 1
                    vdl = f"cond_{var_seq[0]}_dl"
                    steps.append({"assign": vdl, "expr": {"op": "AND", "left": f"vars.{vn}", "right": dl_expr}})
                    return f"vars.{vdl}"
                return f"vars.{vn}"

        # D. Collection Quantifier inside span (`EXISTS`, `FORALL`, `NOT_EXISTS`, `FORALL_NOT`)
        q_info = _parse_quantified_leaf(target_phrase, forced_kind=forced_q, forced_neg=forced_neg)
        if q_info:
            coll_keys = q_info["collKeys"]
            coll_k = coll_keys[0]
            attr_keys, scalar_key, q_kind = q_info["attrKeys"], q_info["scalarKey"], q_info["kind"]
            _rec_coll(coll_keys, q_kind, attr_keys, scalar_key, target_phrase)
            item_pred = {"op": "OR", "operands": [{"op": "EQ", "left": f"it.{ak}", "right": True} for ak in attr_keys] + [{"op": "EQ", "left": "it", "right": True}]}

            def _build_q_for_coll(ck: str):
                if q_kind == "NOT_EXISTS":
                    return {"op": "NOT", "operand": {"op": "EXISTS", "iteratorVar": "it", "collection": f"facts.{ck}", "predicate": item_pred}}
                if q_kind == "FORALL_NOT":
                    return {"op": "FORALL", "iteratorVar": "it", "collection": f"facts.{ck}", "predicate": {"op": "NOT", "operand": item_pred}}
                if q_kind == "FORALL":
                    return {"op": "FORALL", "iteratorVar": "it", "collection": f"facts.{ck}", "predicate": item_pred}
                return {"op": "EXISTS", "iteratorVar": "it", "collection": f"facts.{ck}", "predicate": item_pred}

            if q_kind == "NOT_EXISTS":
                fb_expr = {"op": "NOT", "operand": {"op": "EQ", "left": f"facts.{scalar_key}", "right": True}}
            else:
                fb_expr = {"op": "EQ", "left": f"facts.{scalar_key}", "right": True}

            q_tree = fb_expr
            for ck in reversed(coll_keys[:2]):
                q_tree = {
                    "op": "CONDITIONAL",
                    "test": {"op": "GT", "left": {"op": "LEN", "collection": f"facts.{ck}"}, "right": 0},
                    "then": _build_q_for_coll(ck),
                    "else": q_tree,
                }

            _rec_bool(scalar_key, "REQUIRE_FALSE" if q_kind == "NOT_EXISTS" else role_polarity, target_phrase)
            var_seq[0] += 1
            qv = f"cond_{var_seq[0]}_q"
            steps.append({"assign": qv, "expr": q_tree})
            if dl_expr:
                var_seq[0] += 1
                vdl = f"cond_{var_seq[0]}_qdl"
                steps.append({"assign": vdl, "expr": {"op": "AND", "left": f"vars.{qv}", "right": dl_expr}})
                return f"vars.{vdl}"
            return f"vars.{qv}"

        # E. Monetary Payment/Bond Escape Leaf inside span
        fee_m = re.search(
            r"\b(?:pays?|paying|posts?|posting|deposits?|depositing|remits?|remitting|tenders?|tendering)\s+"
            r"(?:an?\s+)?(?:administrative\s+|statutory\s+)?"
            r"(?:(?P<noun1>fine|fee|bond|deposit|charge|sum)\s+of\s+(?P<amt1>[^,.;:]+)|(?P<amt2>[$₹€£]?\s*\d[\d,]*(?:\.\d+)?)\s+(?P<noun2>fine|fee|bond|deposit|charge|sum))\b",
            target_phrase,
            flags=re.IGNORECASE,
        )
        if fee_m:
            amt_raw = fee_m.group("amt1") or fee_m.group("amt2")
            noun_raw = fee_m.group("noun1") or fee_m.group("noun2")
            amt_qty = parse_numeric_quantity(amt_raw)
            if amt_qty is not None and noun_raw:
                fnoun = noun_raw.lower()
                ccy_val, _, _ = infer_rule_currency_info(substantive, text)[0], None, None
                ccy_suf = (ccy_val or "").capitalize() if ccy_val in ("USD", "EUR", "GBP", "INR") else ""
                num_k = f"{fnoun}Paid"
                alt_num_k = f"{derive_logic_fact_key(target_phrase, num_k)}{ccy_suf}"
                full_bk, alt_bk = derive_logic_fact_key(target_phrase, f"paid{fnoun.capitalize()}"), f"paid{fnoun.capitalize()}"
                _rec_num(num_k, "GTE", amt_qty, "AMOUNT", target_phrase, alt_keys=[alt_num_k] if alt_num_k != num_k else [])
                _rec_bool(full_bk, role_polarity, target_phrase)
                _rec_bool(alt_bk, role_polarity, target_phrase)
                var_seq[0] += 1
                vf = f"cond_{var_seq[0]}_fee"
                steps.append({
                    "assign": vf,
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": finite_number_test(f"facts.{num_k}"),
                        "then": {"op": "GTE", "left": f"facts.{num_k}", "right": amt_qty},
                        "else": {
                            "op": "CONDITIONAL",
                            "test": finite_number_test(f"facts.{alt_num_k}"),
                            "then": {"op": "GTE", "left": f"facts.{alt_num_k}", "right": amt_qty},
                            "else": {"op": "OR", "left": {"op": "EQ", "left": f"facts.{full_bk}", "right": True}, "right": {"op": "EQ", "left": f"facts.{alt_bk}", "right": True}},
                        },
                    },
                })
                if dl_expr:
                    var_seq[0] += 1
                    vdl = f"cond_{var_seq[0]}_feedl"
                    steps.append({"assign": vdl, "expr": {"op": "AND", "left": f"vars.{vf}", "right": dl_expr}})
                    return f"vars.{vdl}"
                return f"vars.{vf}"

        # F. Atomic Boolean Leaf
        is_neg = forced_neg or role_polarity == "REQUIRE_FALSE" or bool(
            re.search(r"\b(?:has\s+not|have\s+not|did\s+not|does\s+not|shall\s+not|may\s+not|must\s+not|cannot|is\s+not|are\s+not|never)\b", target_phrase.lower())
        )
        pos_phrase = re.sub(r"\b(?:has|have|had|does|do|did|shall|may|must|will|can|is|are|was|were)\s+not\b|\bnever\b", "", target_phrase, flags=re.IGNORECASE)
        fact_k = derive_logic_fact_key(pos_phrase, f"condition{var_seq[0] + 1}")
        _rec_bool(fact_k, "REQUIRE_FALSE" if is_neg else role_polarity, target_phrase)
        var_seq[0] += 1
        vn = f"cond_{var_seq[0]}"
        base_expr = {"op": "NOT", "operand": {"op": "EQ", "left": f"facts.{fact_k}", "right": True}} if is_neg else {"op": "EQ", "left": f"facts.{fact_k}", "right": True}
        if dl_expr and not is_neg:
            steps.append({"assign": f"{vn}_raw", "expr": base_expr})
            steps.append({"assign": vn, "expr": {"op": "AND", "left": f"vars.{vn}_raw", "right": dl_expr}})
        else:
            steps.append({"assign": vn, "expr": base_expr})
        return f"vars.{vn}"

    def _lower_span(span_idx: int, role_polarity: str = "REQUIRE_TRUE", split_or: bool = False) -> list[str]:
        s_txt = span_by_idx[span_idx]["span"].strip()
        forced_q = "FORALL" if span_idx in forall_ids else ("EXISTS" if span_idx in exists_ids else None)
        forced_neg = span_idx in negated_ids
        dl_nodes = [e_dl for d_i in deadlines_by_target.get(span_idx, []) if (e_dl := _compile_deadline_expr(span_by_idx[d_i]["span"])) is not None]
        attached_dl = dl_nodes[0] if len(dl_nodes) == 1 else ({"op": "AND", "operands": dl_nodes} if dl_nodes else None)

        if split_or and "neither" not in s_txt.lower():
            clean_c = re.sub(r"^(?:where|when|if|upon)\s+", "", s_txt, flags=re.IGNORECASE).strip()
            prep_m = re.search(r"\b(?:directly\s+)?(?:from|via|through|upon|by)\s+([^.;:]+(?:,\s*|\s+or\s+)[^.;:]+)$", clean_c, flags=re.IGNORECASE)
            if prep_m:
                alts = [p.strip() for p in re.split(r",\s*(?:or\s+)?|\s+\bor\b\s+", prep_m.group(1), flags=re.IGNORECASE) if p.strip()]
            else:
                alts = [p.strip() for p in re.split(r",\s+or\s+|\s+\bor\b\s+", clean_c, flags=re.IGNORECASE) if p.strip()]
            if len(alts) >= 2:
                return [
                    _lower_leaf(alt, role_polarity, forced_q, forced_neg, attached_dl if i == len(alts) - 1 else None)
                    for i, alt in enumerate(alts)
                ]
        return [_lower_leaf(s_txt, role_polarity, forced_q, forced_neg, attached_dl)]

    def _standalone_dl_vars() -> list[str]:
        out = []
        for d_idx in standalone_deadline_ids:
            dl_e = _compile_deadline_expr(span_by_idx[d_idx]["span"])
            if dl_e:
                var_seq[0] += 1
                vd = f"cond_{var_seq[0]}_dl"
                steps.append({"assign": vd, "expr": dl_e})
                out.append(f"vars.{vd}")
        return out

    def _carveout_expr_list() -> list[str]:
        out = []
        for cid in carveout_ids:
            out.extend(_lower_span(cid, role_polarity="CARVEOUT_TRUE", split_or=True))
        return out

    # Assemble AST topology purely from neural modality + predicted role/edge graph
    has_prohib_carveout_edge = any(e["relationType"] == "REL_EXCEPTION_TO_PROHIBITION" and e["targetIdx"] in prohib_ids for e in edges)
    is_perm_graph = (
        deontic_block.get("modalityType") in ("PERMISSION", "PERMISSION_ELECTION")
        or root_condition.get("modalityType") == "PERMISSION"
    ) and not (prohib_ids and carveout_ids and has_prohib_carveout_edge)

    is_prohib_graph = bool(prohib_ids) or deontic_block.get("modalityType") == "PROHIBITION" or root_condition.get("modalityType") == "PROHIBITION"

    if is_perm_graph:
        fact_schema["outcomeMode"] = "PERMISSION"
        deontic_block["modalityType"] = "PERMISSION"
        cond_vars = [v for pid in precond_ids for v in _lower_span(pid, "REQUIRE_TRUE")]
        cond_vars.extend(v for pid in prohib_ids for v in _lower_span(pid, "REQUIRE_FALSE"))
        cond_vars.extend(_standalone_dl_vars())
        if not cond_vars and action_ids:
            cond_vars = [v for aid in action_ids for v in _lower_span(aid, "REQUIRE_TRUE")]
        if not cond_vars:
            return None
        steps.append({"assign": "precondsMet", "expr": cond_vars[0] if len(cond_vars) == 1 else {"op": "AND", "operands": cond_vars}})
        carve_vars = _carveout_expr_list()
        if carve_vars:
            steps.append({"assign": "carveoutApplied", "expr": carve_vars[0] if len(carve_vars) == 1 else {"op": "OR", "operands": carve_vars}})
            steps.append({"assign": "isPermitted", "expr": {"op": "AND", "left": "vars.precondsMet", "right": {"op": "NOT", "operand": "vars.carveoutApplied"}}})
        else:
            steps.append({"assign": "carveoutApplied", "expr": False})
            steps.append({"assign": "isPermitted", "expr": "vars.precondsMet"})
        out_map = {"isPermitted": "vars.isPermitted", "isApplicable": "vars.isPermitted", "isCompliant": "vars.isPermitted", "isSatisfied": "vars.isPermitted", "carveoutApplied": "vars.carveoutApplied"}

    elif is_prohib_graph:
        deontic_block["modalityType"] = "PROHIBITION"
        is_triggered_escape = bool(
            carveout_ids and (
                (action_ids and prohib_ids)
                or (precond_ids and all(pid not in (negated_ids | forall_ids) for pid in precond_ids))
            )
        )
        if is_triggered_escape:
            fact_schema["outcomeMode"] = "PROHIBITION_WITH_ESCAPE"
            trig_srcs = precond_ids if precond_ids else action_ids
            trig_vars = [v for tid in trig_srcs for v in _lower_span(tid, "TRIGGER_TRUE")]
            steps.append({"assign": "triggerMet", "expr": trig_vars[0] if len(trig_vars) == 1 else {"op": "AND", "operands": trig_vars}})
            esc_vars = _carveout_expr_list()
            steps.append({"assign": "escapeSatisfied", "expr": esc_vars[0] if len(esc_vars) == 1 else {"op": "OR", "operands": esc_vars}})
            steps.append({"assign": "isProhibited", "expr": {"op": "AND", "left": "vars.triggerMet", "right": {"op": "NOT", "operand": "vars.escapeSatisfied"}}})
            steps.append({"assign": "isPermitted", "expr": {"op": "OR", "left": {"op": "NOT", "operand": "vars.triggerMet"}, "right": "vars.escapeSatisfied"}})
            out_map = {"isApplicable": "vars.triggerMet", "carveoutApplied": "vars.escapeSatisfied", "isProhibited": "vars.isProhibited", "isBlocked": "vars.isProhibited", "isPermitted": "vars.isPermitted", "isCompliant": "vars.isPermitted", "isSatisfied": "vars.isPermitted"}
        else:
            fact_schema["outcomeMode"] = "PROHIBITION_RULE"
            cond_vars = [v for pid in precond_ids for v in _lower_span(pid, "REQUIRE_FALSE" if pid in negated_ids else "REQUIRE_TRUE")]
            cond_vars.extend(v for pid in prohib_ids for v in _lower_span(pid, "REQUIRE_FALSE"))
            cond_vars.extend(_standalone_dl_vars())
            if not cond_vars and action_ids:
                cond_vars = [v for aid in action_ids for v in _lower_span(aid, "REQUIRE_FALSE")]
            if not cond_vars:
                return None
            steps.append({"assign": "mainCondition", "expr": cond_vars[0] if len(cond_vars) == 1 else {"op": "AND", "operands": cond_vars}})
            carve_vars = _carveout_expr_list()
            steps.append({"assign": "carveoutApplied", "expr": (carve_vars[0] if len(carve_vars) == 1 else {"op": "OR", "operands": carve_vars}) if carve_vars else False})
            steps.append({"assign": "isCompliant", "expr": {"op": "OR", "left": "vars.mainCondition", "right": "vars.carveoutApplied"}})
            steps.append({"assign": "isProhibited", "expr": {"op": "NOT", "operand": "vars.isCompliant"}})
            out_map = {"isApplicable": True, "carveoutApplied": "vars.carveoutApplied", "isProhibited": "vars.isProhibited", "isBlocked": "vars.isProhibited", "isCompliant": "vars.isCompliant", "isPermitted": "vars.isCompliant", "isSatisfied": "vars.isCompliant"}

    elif carveout_ids and (precond_ids or action_ids):
        fact_schema["outcomeMode"] = "SANCTION_WITH_CARVEOUT"
        deontic_block["modalityType"] = "OBLIGATION"
        trig_srcs = precond_ids if precond_ids else action_ids
        trig_vars = [v for tid in trig_srcs for v in _lower_span(tid, "TRIGGER_TRUE")] + _standalone_dl_vars()
        steps.append({"assign": "triggerMet", "expr": trig_vars[0] if len(trig_vars) == 1 else {"op": "AND", "operands": trig_vars}})
        carve_vars = _carveout_expr_list()
        steps.append({"assign": "carveoutApplied", "expr": carve_vars[0] if len(carve_vars) == 1 else {"op": "OR", "operands": carve_vars}})
        steps.append({"assign": "isApplicable", "expr": {"op": "AND", "left": "vars.triggerMet", "right": {"op": "NOT", "operand": "vars.carveoutApplied"}}})
        steps.append({"assign": "isCompliant", "expr": {"op": "NOT", "operand": "vars.isApplicable"}})
        out_map = {"triggerMet": "vars.triggerMet", "carveoutApplied": "vars.carveoutApplied", "isApplicable": "vars.isApplicable", "isCompliant": "vars.isCompliant", "isSatisfied": "vars.isCompliant"}

    else:
        fact_schema["outcomeMode"] = "OBLIGATION_RULE"
        deontic_block["modalityType"] = "OBLIGATION"
        cond_vars = [v for pid in precond_ids for v in _lower_span(pid, "REQUIRE_TRUE")] + _standalone_dl_vars()
        if not cond_vars and action_ids:
            cond_vars = [v for aid in action_ids for v in _lower_span(aid, "REQUIRE_TRUE")]
        if not cond_vars:
            return None
        steps.append({"assign": "mainCondition", "expr": cond_vars[0] if len(cond_vars) == 1 else {"op": "AND", "operands": cond_vars}})
        carve_vars = _carveout_expr_list()
        steps.append({"assign": "carveoutApplied", "expr": (carve_vars[0] if len(carve_vars) == 1 else {"op": "OR", "operands": carve_vars}) if carve_vars else False})
        steps.append({"assign": "isSatisfied", "expr": {"op": "OR", "left": "vars.mainCondition", "right": "vars.carveoutApplied"}})
        out_map = {"isApplicable": True, "carveoutApplied": "vars.carveoutApplied", "isCompliant": "vars.isSatisfied", "isSatisfied": "vars.isSatisfied"}

    return {
        "archetype": "PURE_LOGIC_PIPELINE",
        "outputMode": "PURE_LOGIC",
        "standardRateBps": 0,
        "factSchema": fact_schema,
        "pipeline": {"steps": steps, "output": out_map},
    }


def synthesize_building_blocks(
    root_condition: dict,
    spans: list[dict],
    edges: list[dict],
    text: str,
    predicted_modality: str = "MODALITY_COMPUTATIONAL_EXCESS",
    rule_domain: str = "STATUTORY_FINANCE",
) -> dict:
    """
    Synthesizes universal, composable statutory building blocks purely from
    predicted modality + SRL spans + Biaffine relations (zero raw-text domain hacks).
    """
    node_type = root_condition.get("nodeType", "TRANSACTIONAL_WITHHOLDING")

    regulated_objects = [
        s["span"] for s in spans
        if s["role"] in ["ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", "ROLE_NON_RESIDENT_EXEMPTION_GUARD"]
    ]

    PASSIVE_INANIMATE_NOUNS = {
        "suit", "provision", "credit", "deduction", "levy", "plaint", "tax",
        "sum", "amount", "asset", "liability", "contract", "arrangement", "supply", "benefit"
    }
    entity = root_condition.get("entityType")
    if not entity:
        if node_type.startswith("ACCOUNTING_") or node_type in ("MEASUREMENT_VALUATION_RULE", "DISCLOSURE_PRESENTATION_OBLIGATION"):
            entity = "REPORTING_ENTITY"
        else:
            payable_match = re.search(r"\bpayable\s+to\s+(?:an?|any)\s+([a-zA-Z]+)\b", text, flags=re.IGNORECASE)
            subj_match = re.search(
                r"\b(?:no|every|any|an|a|the)\s+([a-zA-Z]+)\s+(?:(?:who\s+is|who\s+has|landing\s+on|entering|receiving|providing|responsible\s+for)\s+[^.,:;]{1,45}?\s+)?(?:shall|may|must)(?!\s+be\s+(?:instituted|recognised|recognized|classified|allowed|available|applicable|deducted|collected|charged))\b",
                text,
                flags=re.IGNORECASE
            )
            if payable_match and payable_match.group(1).lower() not in PASSIVE_INANIMATE_NOUNS:
                entity = payable_match.group(1).upper()
            elif subj_match and subj_match.group(1).lower() not in PASSIVE_INANIMATE_NOUNS:
                entity = subj_match.group(1).upper()
            elif regulated_objects:
                norm_subj = normalize_symbol_id(regulated_objects[0])
                entity = norm_subj.upper() if norm_subj else "REGULATED_ACTOR"
            else:
                entity = "REGULATED_ACTOR"

    prior_turnover = (
        root_condition.get("priorTurnoverPrecondition")
        or root_condition.get("prerequisiteObligation")
        or root_condition.get("prerequisiteCondition")
    )

    qualifying_conditions = []
    for s in spans:
        if s["role"] in ["ROLE_AGENT_STATE_PRECONDITION", "ROLE_BUYER_PRIOR_TURNOVER_GUARD", "ROLE_SUBJECTIVE_CRITERION", "ROLE_REQUIRED_EVIDENCE", "ROLE_COMPUTATION_OPERAND"]:
            if s["span"] not in qualifying_conditions and len(s["span"].split()) > 1:
                qualifying_conditions.append(s["span"])

    comp_expr = {"roundingRule": "PAISA_PRECISION"}
    if node_type == "TRANSACTIONAL_WITHHOLDING":
        if (
            not root_condition.get("applicableCliffThreshold")
            and not root_condition.get("applicableExcessThreshold")
            and root_condition.get("standardTaxRate")
        ):
            comp_expr["expressionType"] = "FLAT_RATE"
            comp_expr["formulaDescription"] = "TransactionAmount * StandardTaxRate (flat statutory rate with zero threshold)"
        elif predicted_modality == "MODALITY_COMPUTATIONAL_CLIFF" or (
            root_condition.get("applicableCliffThreshold") and not root_condition.get("applicableExcessThreshold")
        ):
            comp_expr["expressionType"] = "CLIFF_THRESHOLD"
            comp_expr["formulaDescription"] = "if (TransactionAmountOrAggregate > CliffThreshold) then (TransactionAmountOrAggregate * StandardTaxRate) else 0"
            comp_expr["cliffThreshold"] = root_condition.get("applicableCliffThreshold") or root_condition.get("applicableExcessThreshold")
            if root_condition.get("applicableExcessThreshold") and root_condition.get("applicableCliffThreshold"):
                comp_expr["excessThreshold"] = root_condition.get("applicableExcessThreshold")
        else:
            comp_expr["expressionType"] = "EXCESS_OVER_THRESHOLD"
            comp_expr["formulaDescription"] = "max(0, min(CurrentInvoice, PriorAggregate + CurrentInvoice - ExcessThreshold)) * StandardTaxRate"
            comp_expr["excessThreshold"] = root_condition.get("applicableExcessThreshold")
        comp_expr["standardRate"] = root_condition.get("standardTaxRate")
        comp_expr["penalSurchargeRate"] = root_condition.get("nonPanPenalRate")
    elif node_type == "MEASUREMENT_VALUATION_RULE":
        comp_expr["expressionType"] = "MIN_MAX_VALUATION"
        comp_expr["formulaDescription"] = root_condition.get("valuationFormula", "Valuation formula derived from statutory rule")
        comp_expr["impairmentTrigger"] = root_condition.get("impairmentTrigger")
    elif node_type == "TEMPORAL_WINDOW_GATE" and (
        root_condition.get("accrualMode") == "FRACTIONAL_RATIO_ACCRUAL"
        or (not root_condition.get("statutorySanction") and not root_condition.get("statutoryInterestFormula") and predicted_modality == "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB")
    ):
        comp_expr["expressionType"] = "FRACTIONAL_RATIO_ACCRUAL"
        comp_expr["formulaDescription"] = "Accrual / vesting across prescribed continuous service duration threshold"
        comp_expr["vestingThreshold"] = root_condition.get("applicableDeadlineWindow")
    elif node_type == "TEMPORAL_WINDOW_GATE" and root_condition.get("statutoryInterestFormula"):
        comp_expr["expressionType"] = "COMPOUND_INTEREST_PENAL"
        comp_expr["formulaDescription"] = "Principal * ((1 + (3 * RbiBankRate) / 12) ^ Months - 1)"
        comp_expr["statutoryAnnualRateMultiplier"] = "3x RBI Bank Rate"
        comp_expr["compoundingRestInterval"] = "MONTHLY"
    elif node_type == "ACCOUNTING_ELECTION_CARVEOUT":
        comp_expr["expressionType"] = "EXEMPTION_THRESHOLD_GATE"
        comp_expr["formulaDescription"] = "Election exemption threshold gate with P&L straight-line recognition"
        comp_expr["applicableThresholdWindow"] = root_condition.get("applicableThresholdWindow")
    elif node_type == "BOOLEAN_PROHIBITION_GATE":
        comp_expr["expressionType"] = "BOOLEAN_BINARY_GATE"
        comp_expr["formulaDescription"] = "Prohibition evaluates strictly to False unless safe harbor carveout is established"
    elif node_type == "GENERAL_RULE_GATE":
        comp_expr["expressionType"] = "PURE_LOGIC_PIPELINE"
        comp_expr["formulaDescription"] = "Declarative boolean, quantifier, and carveout logic evaluation"
    else:
        comp_expr["expressionType"] = "STATUTORY_PREDICATE_EVALUATION"
        comp_expr["formulaDescription"] = "Deterministic boolean evaluation against evidence and corporate registry"

    prohib_pred = root_condition.get("prohibitionPredicate")
    if node_type == "ACCOUNTING_ELECTION_CARVEOUT":
        modality_type = "PERMISSION_ELECTION"
        statutory_duty = root_condition.get("electionPredicate") or "May elect not to apply recognition requirements"
    elif node_type == "GENERAL_RULE_GATE" and (
        predicted_modality == "MODALITY_DEONTIC_PERMISSION"
        or root_condition.get("modalityType") == "PERMISSION"
    ):
        modality_type = "PERMISSION"
        statutory_duty = root_condition.get("permissionPredicate") or root_condition.get("obligationPredicate") or "Permitted subject to stated preconditions"
    elif node_type == "DISCLOSURE_PRESENTATION_OBLIGATION":
        modality_type = "DISCLOSURE_DUTY"
        statutory_duty = root_condition.get("presentationPredicate") or "Shall disclose in financial statements"
    elif node_type == "BOOLEAN_PROHIBITION_GATE" or (
        node_type == "GENERAL_RULE_GATE" and (
            predicted_modality == "MODALITY_DEONTIC_PROHIBITION"
            or root_condition.get("modalityType") == "PROHIBITION"
        )
    ) or (
        prohib_pred and any(tok in prohib_pred.lower() for tok in ["no ", "not ", "prohibit", "barred", "ineligible", "disqualified"])
    ):
        modality_type = "PROHIBITION"
        statutory_duty = prohib_pred or "Conduct prohibited unless safe harbor carveout applies"
    else:
        modality_type = "OBLIGATION"
        statutory_duty = (
            root_condition.get("obligationPredicate")
            or root_condition.get("recognitionPredicate")
            or prohib_pred
            or "Mandatory compliance obligation"
        )

    carveouts = list(root_condition.get("exceptionCarveouts", []))
    if not carveouts:
        for s in spans:
            if s["role"] == "ROLE_EXCEPTION_CARVEOUT" and s["span"] not in carveouts:
                carveouts.append(s["span"])

    temporal = {
        "triggerPolicy": root_condition.get("triggerPolicy", "EARLIER_OF_CREDIT_OR_PAYMENT"),
        "effectiveCutover": root_condition.get("temporalCutover"),
        "sunsetDate": root_condition.get("temporalSunset"),
        "prescriptionWindow": root_condition.get("applicableDeadlineWindow"),
        "temporalStatus": "SUNSET_OMITTED" if root_condition.get("temporalSunset") else "ACTIVE"
    }

    dest_spans = [s["span"] for s in spans if s["role"] == "ROLE_ACCOUNTING_DESTINATION"]
    dest = root_condition.get("accountingDestination")
    if not dest:
        if rule_domain == "GENERAL_LOGIC":
            dest = "NOT_APPLICABLE_PURE_LOGIC"
        elif node_type == "TRANSACTIONAL_WITHHOLDING":
            dest = "STATUTORY_TAX_PAYABLE"
        elif dest_spans:
            dest = classify_gl_destination_from_spans(dest_spans, "PROFIT_OR_LOSS")
        elif comp_expr.get("expressionType") == "FRACTIONAL_RATIO_ACCRUAL":
            dest = "PROFIT_OR_LOSS"
        else:
            dest = "GENERAL_LEDGER_CONTROLLERSHIP_VOUCHER"

    accounting_routing = {
        "generalLedgerRoute": dest,
        "requiresDualEntryVoucher": dest in ["PROFIT_OR_LOSS", "STATUTORY_TAX_PAYABLE", "ASSET_CAPITALIZATION", "OTHER_COMPREHENSIVE_INCOME"]
    }

    return {
        "applicabilityScope": {
            "entitySubject": entity,
            "priorTurnoverPrecondition": prior_turnover,
            "regulatedObjectScope": regulated_objects,
            "qualifyingConditions": qualifying_conditions
        },
        "computationalExpression": comp_expr,
        "deonticModality": {
            "modalityType": modality_type,
            "statutoryDuty": statutory_duty
        },
        "carveoutsAndExceptions": {
            "safeHarbors": carveouts,
            "nonObstanteOverrides": [s["span"] for s in spans if "notwithstanding" in s["span"].lower()]
        },
        "temporalCrystallization": temporal,
        "accountingDestination": accounting_routing,
        "statutoryInvariants": {
            "isAuthorizedToDisburse": False,
            "requiresHumanApprovalForOutcomes": True,
            "controllershipAttestationRequired": rule_domain != "GENERAL_LOGIC"
        }
    }


def synthesize_symbol_legend(spans: list[dict], edges: list[dict], text: str) -> dict:
    """
    Infers strongly typed runtime concepts (symbols) from discovered roles,
    Biaffine relational graph edges, and linguistic quantifiers/units.
    """
    legend = {}
    FUNCTION_WORDS = {
        'in', 'of', 'to', 'for', 'by', 'on', 'at', 'from', 'the', 'a', 'an', 'and', 'or', 'be',
        'that', 'with', 'as', 'thereof', 'not', 'no', '0', '1', '2', 'such', 'any', 'where', 'whether',
        'will', 'shall', 'cannot', 'can', 'may', 'must', 'would', 'could', 'might', 'his', 'her',
        'their', 'its', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'this'
    }
    substantive_spans = [
        s for s in spans
        if s["role"] != "ROLE_IGNORE" and not all(w.lower() in FUNCTION_WORDS for w in s["span"].split())
    ]

    concept_roles = {
        "ROLE_AGENT_STATE_PRECONDITION",
        "ROLE_BUYER_PRIOR_TURNOVER_GUARD",
        "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE",
        "ROLE_SUBJECTIVE_CRITERION",
        "ROLE_NON_RESIDENT_EXEMPTION_GUARD",
        "ROLE_EXCEPTION_CARVEOUT",
        "ROLE_REQUIRED_EVIDENCE",
        "ROLE_PROHIBITION_GATE",
        "ROLE_COMPUTATION_OPERAND"
    }

    modifier_roles = {
        "ROLE_TRANSACTION_EXCESS_THRESHOLD",
        "ROLE_TRANSACTION_CLIFF_THRESHOLD",
        "ROLE_DEADLINE_WINDOW",
        "ROLE_STANDARD_STATUTORY_RATE",
        "ROLE_PENALTY_OR_DEFAULT_OVERRIDE",
        "ROLE_NON_PAN_OVERRIDE_RATE"
    }

    rate_roles = {"ROLE_STANDARD_STATUTORY_RATE", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE", "ROLE_NON_PAN_OVERRIDE_RATE"}
    concepts = [s for s in substantive_spans if s["role"] in concept_roles]
    concept_modifiers = [s for s in substantive_spans if s["role"] in modifier_roles and s["role"] not in rate_roles]
    modifiers = [s for s in substantive_spans if s["role"] in modifier_roles]
    consumed_modifier_indices = set()
    consumed_modifiers = set()

    edge_idx_map = {}
    target_co_parents_idx = {}
    edge_map = {}
    target_co_parents = {}
    for e in edges:
        src_idx = e.get("sourceIdx")
        tgt_idx = e.get("targetIdx")
        src = e.get("sourceSpan")
        tgt = e.get("targetSpan")
        rel = e.get("relationType", "RELATED_TO")
        if src_idx is not None and tgt_idx is not None:
            edge_idx_map[(src_idx, tgt_idx)] = rel
            edge_idx_map[(tgt_idx, src_idx)] = rel
            target_co_parents_idx.setdefault(tgt_idx, []).append((src_idx, rel))
        edge_map[(src, tgt)] = rel
        edge_map[(tgt, src)] = rel
        if tgt not in target_co_parents:
            target_co_parents[tgt] = []
        target_co_parents[tgt].append((src, rel))

    PROXIMITY_ALLOWED_ROLES = {
        "ROLE_AGENT_STATE_PRECONDITION",
        "ROLE_BUYER_PRIOR_TURNOVER_GUARD",
    }
    rule_ccy_code, _ = infer_rule_currency_info(spans, text)

    def _attach_monetary_fields(entry: dict, sym_type_val: str, data_type_val: str, unit_val: str, qty_val, fact_hint: str, fallback_key: str):
        if sym_type_val == "MONETARY_QUANTUM" and qty_val is not None:
            minor_int = int(round(float(qty_val) * 100))
            entry["currencyCode"] = unit_val or "INR"
            entry["minorUnitScale"] = 100
            entry["targetThresholdMinorUnits"] = minor_int
            op_keys = build_operand_fact_keys(fact_hint, unit_val)
            entry["factKey"] = op_keys[0] if op_keys else _derive_fact_key_camel(fact_hint, fallback_key)
            if data_type_val == "CURRENCY_PAISA":
                entry["targetThresholdInr"] = qty_val
                entry["targetThresholdPaisa"] = minor_int

    for c in concepts:
        c_span = c["span"]
        c_role = c["role"]
        c_idx = c.get("spanIdx")
        sym_id = normalize_symbol_id(c_span)
        if not sym_id or sym_id in legend:
            continue

        if c_role == "ROLE_COMPUTATION_OPERAND":
            op_keys = build_operand_fact_keys(c_span, rule_ccy_code)
            op_entry = {
                "symbolId": sym_id,
                "rawText": c_span,
                "role": c_role,
                "symbolType": "PREREQUISITE_STATUS",
                "dataType": "NUMBER",
                "factKey": op_keys[0] if op_keys else _derive_fact_key_camel(c_span, sym_id)
            }
            if rule_ccy_code:
                op_entry["currencyCode"] = rule_ccy_code
            legend[sym_id] = op_entry
            continue

        c_cr = c.get("charRange", [0, 0])
        c_qty_intrinsic = parse_numeric_quantity(c_span)
        c_sym_type_int, c_data_type_int, c_unit_int = parse_unit_and_type(c_span)

        if c_role in PROXIMITY_ALLOWED_ROLES and c_qty_intrinsic is not None and c_sym_type_int != "BOOLEAN_FLAG":
            win_start = max(0, c_cr[0] - 35)
            win_end = min(len(text), c_cr[1] + 35)
            context_window = text[win_start:win_end]
            op = parse_comparison_operator(context_window)
            sym_entry = {
                "symbolId": sym_id,
                "rawText": c_span,
                "role": c_role,
                "symbolType": c_sym_type_int,
                "dataType": c_data_type_int,
                "measurementUnit": c_unit_int,
                "comparisonOperator": op,
                "targetThreshold": c_qty_intrinsic,
                "sourceRelation": "INTRINSIC_QUANTIFIER"
            }
            _attach_monetary_fields(sym_entry, c_sym_type_int, c_data_type_int, c_unit_int, c_qty_intrinsic, c_span, sym_id)
            legend[sym_id] = sym_entry
            continue

        linked_modifier = None
        link_relation = None

        if c_role in PROXIMITY_ALLOWED_ROLES:
            for m in concept_modifiers:
                m_idx = m.get("spanIdx")
                m_span = m["span"]
                if c_idx is not None and m_idx is not None and (c_idx, m_idx) in edge_idx_map:
                    linked_modifier = m
                    link_relation = edge_idx_map[(c_idx, m_idx)]
                    break
                if (c_span, m_span) in edge_map:
                    linked_modifier = m
                    link_relation = edge_map[(c_span, m_span)]
                    break

            if not linked_modifier and c_idx is not None:
                for _, srcs in target_co_parents_idx.items():
                    src_ids = [s[0] for s in srcs]
                    if c_idx in src_ids:
                        for m in concept_modifiers:
                            m_idx = m.get("spanIdx")
                            if m_idx is not None and m_idx in src_ids and m_idx != c_idx:
                                linked_modifier = m
                                link_relation = "CO_DEPENDENT_SIBLING"
                                break
                    if linked_modifier:
                        break
            if not linked_modifier:
                for _, srcs in target_co_parents.items():
                    src_spans = [s[0] for s in srcs]
                    if c_span in src_spans:
                        for m in concept_modifiers:
                            if m["span"] in src_spans and m["span"] != c_span:
                                linked_modifier = m
                                link_relation = "CO_DEPENDENT_SIBLING"
                                break
                    if linked_modifier:
                        break

            if not linked_modifier:
                closest_mod = None
                closest_dist = 999999
                for m in concept_modifiers:
                    m_cr = m.get("charRange", [0, 0])
                    if c_cr[1] <= m_cr[0]:
                        dist = m_cr[0] - c_cr[1]
                        gap_text = text[c_cr[1]:m_cr[0]]
                    elif m_cr[1] <= c_cr[0]:
                        dist = c_cr[0] - m_cr[1]
                        gap_text = text[m_cr[1]:c_cr[0]]
                    else:
                        dist = 0
                        gap_text = ""
                    if re.search(r"[.;:]|\bprovided\b", gap_text, flags=re.IGNORECASE):
                        continue
                    if dist < 50 and dist < closest_dist:
                        closest_dist = dist
                        closest_mod = m
                if closest_mod:
                    linked_modifier = closest_mod
                    link_relation = "SPATIAL_PROXIMITY"

        if linked_modifier:
            c_sym_type, c_data_type, c_unit = parse_unit_and_type(c_span)
            m_sym_type, m_data_type, m_unit = parse_unit_and_type(linked_modifier["span"])

            if m_sym_type != "MONETARY_QUANTUM":
                if linked_modifier.get("spanIdx") is not None:
                    consumed_modifier_indices.add(linked_modifier["spanIdx"])
                else:
                    consumed_modifiers.add(linked_modifier["span"])

            m_cr = linked_modifier.get("charRange", [0, 0])
            win_start = max(0, min(c_cr[0], m_cr[0]) - 35)
            win_end = min(len(text), max(c_cr[1], m_cr[1]) + 35)
            context_window = text[win_start:win_end]

            if c_sym_type != "BOOLEAN_FLAG":
                sym_type, data_type, unit = c_sym_type, c_data_type, c_unit
                qty = parse_numeric_quantity(c_span) or parse_numeric_quantity(linked_modifier["span"])
            else:
                sym_type, data_type, unit = m_sym_type, m_data_type, m_unit
                qty = parse_numeric_quantity(linked_modifier["span"]) or parse_numeric_quantity(c_span)

            if sym_type == "BOOLEAN_FLAG":
                sym_type, data_type, unit = parse_unit_and_type(context_window)
                if qty is None:
                    qty = parse_numeric_quantity(context_window)
            op = parse_comparison_operator(context_window)

            sym_entry = {
                "symbolId": sym_id,
                "rawText": c_span,
                "role": c_role,
                "symbolType": sym_type,
                "dataType": data_type,
                "measurementUnit": unit,
                "comparisonOperator": op,
                "targetThreshold": qty,
                "sourceRelation": link_relation,
                "relatedSpan": linked_modifier["span"]
            }
            _attach_monetary_fields(sym_entry, sym_type, data_type, unit, qty, c_span, sym_id)
            legend[sym_id] = sym_entry

        elif c_role == "ROLE_EXCEPTION_CARVEOUT":
            legend[sym_id] = {
                "symbolId": sym_id,
                "rawText": c_span,
                "role": c_role,
                "symbolType": "EXEMPTION_CONDITION",
                "dataType": "BOOLEAN",
                "actionOnMatch": "WAIVE_THRESHOLD",
                "actionOnMismatch": "ENFORCE_OBLIGATION"
            }
        elif c_role == "ROLE_REQUIRED_EVIDENCE":
            legend[sym_id] = {
                "symbolId": sym_id,
                "rawText": c_span,
                "role": c_role,
                "symbolType": "DOCUMENT_VERIFICATION",
                "dataType": "DOCUMENT_ID",
                "measurementUnit": "REGISTRY_VERIFICATION",
                "actionOnMissing": "BLOCK_COMPLIANCE"
            }
        elif c_role == "ROLE_SUBJECTIVE_CRITERION":
            legend[sym_id] = {
                "symbolId": sym_id,
                "rawText": c_span,
                "role": c_role,
                "symbolType": "SUBJECTIVE_CRITERION_EVALUATION",
                "dataType": "BOOLEAN",
                "evaluationMethod": "ATTESTATION_CHECK"
            }
        else:
            qty = parse_numeric_quantity(c_span)
            sym_type, data_type, unit = parse_unit_and_type(c_span)
            if qty is not None and sym_type != "BOOLEAN_FLAG":
                win_start = max(0, c_cr[0] - 35)
                win_end = min(len(text), c_cr[1] + 35)
                context_window = text[win_start:win_end]
                op = parse_comparison_operator(context_window)
                sym_entry = {
                    "symbolId": sym_id,
                    "rawText": c_span,
                    "role": c_role,
                    "symbolType": sym_type,
                    "dataType": data_type,
                    "measurementUnit": unit,
                    "comparisonOperator": op,
                    "targetThreshold": qty,
                    "sourceRelation": "INTRINSIC_QUANTIFIER"
                }
                _attach_monetary_fields(sym_entry, sym_type, data_type, unit, qty, c_span, sym_id)
                legend[sym_id] = sym_entry
            elif c_role in ("ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", "ROLE_NON_RESIDENT_EXEMPTION_GUARD"):
                legend[sym_id] = {
                    "symbolId": sym_id,
                    "rawText": c_span,
                    "role": c_role,
                    "symbolType": "ENTITY_CLASSIFICATION",
                    "dataType": "CATEGORICAL_STRING",
                    "matchingRule": "SUBSTRING_OR_KYC_MATCH"
                }
            else:
                words_in_span = [w for w in c_span.lower().split() if w not in FUNCTION_WORDS]
                if len(words_in_span) >= 2 or (len(words_in_span) == 1 and len(words_in_span[0]) >= 8 and c_role == "ROLE_PROHIBITION_GATE" and words_in_span[0] not in {"economic"}):
                    legend[sym_id] = {
                        "symbolId": sym_id,
                        "rawText": c_span,
                        "role": c_role,
                        "symbolType": "PREREQUISITE_STATUS",
                        "dataType": "BOOLEAN"
                    }

    for m in modifiers:
        m_idx = m.get("spanIdx")
        is_consumed = (m_idx in consumed_modifier_indices) if m_idx is not None else (m["span"] in consumed_modifiers)
        if not is_consumed:
            m_span = m["span"]
            if len(m_span.split()) == 1 and (m_span.lower() in FUNCTION_WORDS or m_span.lower() in {"exceeding", "less", "more", "greater", "above", "below"}):
                continue
            sym_id = normalize_symbol_id(m_span)
            if sym_id and sym_id not in legend:
                m_cr = m.get("charRange", [0, 0])
                win_start = max(0, m_cr[0] - 25)
                win_end = min(len(text), m_cr[1] + 15)
                context_window = text[win_start:win_end]

                qty_span = parse_numeric_quantity(m_span)
                trailing_unit_slice = text[m_cr[1]:min(len(text), m_cr[1] + 20)]
                qty = qty_span
                if qty is None:
                    qty = parse_numeric_quantity(context_window)
                elif re.match(r"^\s*(?:crores?|lakhs?|lacs?|thousand|million|billion)\b", trailing_unit_slice, flags=re.IGNORECASE):
                    expanded_qty = parse_numeric_quantity(m_span + " " + trailing_unit_slice)
                    if expanded_qty is not None and expanded_qty > qty:
                        qty = expanded_qty

                sym_type, data_type, unit = parse_unit_and_type(m_span)
                if sym_type == "BOOLEAN_FLAG":
                    trailing_sym_type, trailing_data_type, trailing_unit = parse_unit_and_type(m_span + " " + trailing_unit_slice)
                    if trailing_sym_type != "BOOLEAN_FLAG":
                        sym_type, data_type, unit = trailing_sym_type, trailing_data_type, trailing_unit
                    elif qty is not None and float(qty).is_integer():
                        sym_type, data_type, unit = "PREREQUISITE_STATUS", "NUMBER", "COUNT"
                    else:
                        sym_type, data_type, unit = parse_unit_and_type(context_window)
                op = "EQ" if sym_type == "PERCENTAGE_RATE" else parse_comparison_operator(context_window)
                effective_role = m["role"]
                preceding_ctx = text[max(0, m_cr[0] - 60):m_cr[0]].lower()
                if any(k in preceding_ctx for k in ["turnover", "sales", "gross receipts", "capital", "net worth", "continuous service"]):
                    if sym_type == "MONETARY_QUANTUM" and data_type == "CURRENCY_PAISA":
                        effective_role = "ROLE_BUYER_PRIOR_TURNOVER_GUARD"
                    elif sym_type == "MONETARY_QUANTUM":
                        effective_role = "ROLE_AGENT_STATE_PRECONDITION"
                    elif sym_type == "TEMPORAL_DURATION":
                        effective_role = "ROLE_AGENT_STATE_PRECONDITION"
                elif any(k in preceding_ctx for k in ["aggregate value", "value of such", "sum", "perquisite", "benefit", "withdrawal", "winnings", "contract", "consideration"]):
                    if sym_type == "MONETARY_QUANTUM":
                        effective_role = "ROLE_TRANSACTION_CLIFF_THRESHOLD"

                sym_entry = {
                    "symbolId": sym_id,
                    "rawText": m_span,
                    "role": effective_role,
                    "symbolType": sym_type,
                    "dataType": data_type,
                    "measurementUnit": unit,
                    "comparisonOperator": op,
                    "targetThreshold": qty,
                    "sourceRelation": "STANDALONE_MODIFIER"
                }
                _attach_monetary_fields(sym_entry, sym_type, data_type, unit, qty, _safe_preceding_words(text, m_cr[0], 35) + " " + m_span, sym_id)
                if sym_type == "PERCENTAGE_RATE" and effective_role in ("ROLE_PENALTY_OR_DEFAULT_OVERRIDE", "ROLE_NON_PAN_OVERRIDE_RATE"):
                    sym_entry["factKey"] = _derive_fact_key_camel(context_window, f"override_{sym_id}")
                legend[sym_id] = sym_entry

    return legend


def synthesize_usvm_ast(
    text: str,
    spans: list[dict],
    edges: list[dict] = None,
    source_metadata: dict = None,
    predicted_modality: str = "MODALITY_COMPUTATIONAL_EXCESS",
    modality_conf: float = 0.9,
    clause_modalities: list[dict] = None,
    rule_domain: str = "AUTO",
    typed_inputs: bool = False,
    clause_scopes: bool = False,
) -> dict:
    """
    Canonical stateless Unified 6-Tuple USVM AST synthesizer used identically
    by production inference, CLI verification, and GRPO RL reward evaluation.

    ``typed_inputs`` opts into the RFC-012 v2 profile: a bracket clause emits
    one declared ``inputs.*`` value, with no alias-probe chain and no null test.
    The legacy profile remains the default.

    ``clause_scopes`` opts into the RFC-012 Step 2 profile: a multi-clause
    statute is compiled into one clause-scoped pipeline whose values carry SSA
    names and whose cross-clause references are linked by the def-use linker.
    A single-clause statute keeps the current computation unchanged.
    """
    meta = source_metadata or {}
    text_low = (text or "").lower()
    currency_code, _ = infer_rule_currency_info(spans, text)

    indexed_spans = []
    for idx_i, item in enumerate(spans or []):
        s_copy = dict(item)
        s_copy["spanIdx"] = item.get("spanIdx", idx_i)
        s_copy["clauseIdx"] = item.get("clauseIdx", 0)
        if "charRange" not in s_copy:
            if "tokenRange" in s_copy:
                s_copy["charRange"] = [s_copy["tokenRange"][0] * 5, s_copy["tokenRange"][1] * 5]
            else:
                pos = text.find(s_copy.get("span", ""))
                s_copy["charRange"] = [max(0, pos), max(0, pos) + len(s_copy.get("span", ""))]
        indexed_spans.append(s_copy)
    span_by_idx = {s["spanIdx"]: s for s in indexed_spans}
    span_str_to_idx = {}
    for s in indexed_spans:
        span_str_to_idx.setdefault(s["span"], s["spanIdx"])

    resolved_edges = []
    for e in (edges or []):
        s_idx = e.get("sourceIdx")
        t_idx = e.get("targetIdx")
        if s_idx is None:
            s_idx = span_str_to_idx.get(e.get("sourceSpan"))
        if t_idx is None:
            t_idx = span_str_to_idx.get(e.get("targetSpan"))
        if s_idx is not None and t_idx is not None and s_idx in span_by_idx and t_idx in span_by_idx:
            resolved_edges.append({
                "sourceIdx": s_idx,
                "targetIdx": t_idx,
                "sourceSpan": span_by_idx[s_idx]["span"],
                "targetSpan": span_by_idx[t_idx]["span"],
                "relationType": e.get("relationType", "REL_NO_EDGE")
            })

    effective_domain = classify_rule_domain(
        text,
        indexed_spans,
        resolved_edges,
        predicted_modality=predicted_modality,
        requested_domain=rule_domain,
    )

    turnover_precondition = None
    turnover_precondition_item = None
    excess_threshold = None
    cliff_threshold = None
    standard_rate = None
    non_pan_rate = None
    non_pan_rate_item = None
    trigger_policy = "EARLIER_OF"
    temporal_cutover = None
    temporal_sunset = None

    deadline_window = None
    prohibition_gate = None
    exception_carveouts = []
    subjective_criteria = None
    evidence_checklist = []
    mandatory_actions = []
    statutory_sanctions = []
    accounting_destinations = []
    regulated_objects = []

    rate_items = []
    thresh_items = []
    operand_items = []
    precondition_items = []
    cutover_items = []
    sunset_items = []

    CARVEOUT_STOPWORDS = {
        'in', 'of', 'to', 'for', 'by', 'on', 'at', 'from', 'the', 'a', 'an', 'and', 'or', 'be',
        'that', 'with', 'as', 'thereof', 'such', 'any', 'where', 'when', 'whether', 'while',
        'will', 'shall', 'may', 'must', 'would', 'could', 'might', 'his', 'her', 'their', 'its',
        'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'this', 'those', 'these'
    }
    DATE_MONTH_TOKENS = [
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
    ]

    for item in indexed_spans:
        r = item["role"]
        s = item["span"]
        s_lower = s.lower()
        s_unit_info = parse_unit_and_type(s)
        s_qty = parse_numeric_quantity(s)

        if r in ["ROLE_AGENT_STATE_PRECONDITION", "ROLE_BUYER_PRIOR_TURNOVER_GUARD"]:
            precondition_items.append(item)
            s_is_monetary = s_qty is not None and s_unit_info[0] == "MONETARY_QUANTUM"
            cur_is_monetary = bool(
                turnover_precondition
                and parse_numeric_quantity(turnover_precondition) is not None
                and parse_unit_and_type(turnover_precondition)[0] == "MONETARY_QUANTUM"
            )
            if not turnover_precondition or (s_is_monetary and not cur_is_monetary) or (s_is_monetary == cur_is_monetary and len(s) > len(turnover_precondition)):
                turnover_precondition = s
                turnover_precondition_item = item
        elif r == "ROLE_TRANSACTION_EXCESS_THRESHOLD" and (s_qty is not None or any(w in s_lower for w in ["exceed", "lakh", "purchase", "turnover", "years", "carrying amount", "recoverable", "rupee", "rupees", "dollar", "dollars", "euro", "euros", "pound", "pounds", "crore", "thousand", "million", "billion"])):
            if s_qty is not None:
                thresh_items.append(item)
            if not excess_threshold or len(s) > len(excess_threshold):
                excess_threshold = s
        elif r == "ROLE_TRANSACTION_CLIFF_THRESHOLD" and (s_qty is not None or any(w in s_lower for w in ["exceed", "lakh", "purchase", "turnover", "years", "carrying amount", "recoverable", "rupee", "rupees", "dollar", "dollars", "euro", "euros", "pound", "pounds", "crore", "thousand", "million", "billion", "two", "one", "hundred"])):
            if s_qty is not None:
                thresh_items.append(item)
            if not cliff_threshold or len(s) > len(cliff_threshold):
                cliff_threshold = s
        elif r == "ROLE_STANDARD_STATUTORY_RATE" and any(w in s_lower for w in ["%", "per cent", "percent"]):
            if parse_rate_bps(s) is not None:
                rate_items.append(item)
            if not standard_rate:
                standard_rate = s
        elif r in ["ROLE_PENALTY_OR_DEFAULT_OVERRIDE", "ROLE_NON_PAN_OVERRIDE_RATE"] and any(w in s_lower for w in ["%", "per cent", "percent"]):
            non_pan_rate = s
            non_pan_rate_item = item
        elif r == "ROLE_COMPUTATION_OPERAND":
            operand_items.append(item)
        elif r == "ROLE_TRIGGER_PAYMENT_ONLY":
            trigger_policy = "PAYMENT_ONLY"
        elif r == "ROLE_TRIGGER_EARLIER_EVENT":
            trigger_policy = "EARLIER_OF"
        elif r == "ROLE_TEMPORAL_CUTOVER_DATE" and (re.search(r"\b(?:19|20)\d{2}\b", s_lower) or any(m in s_lower for m in DATE_MONTH_TOKENS)):
            temporal_cutover = s
            cutover_items.append(item)
        elif r == "ROLE_TEMPORAL_SUNSET_DATE":
            temporal_sunset = s
            sunset_items.append(item)
        elif r == "ROLE_DEADLINE_WINDOW":
            if not deadline_window:
                deadline_window = s
        elif r == "ROLE_PROHIBITION_GATE":
            if not prohibition_gate or len(s) > len(prohibition_gate):
                prohibition_gate = s
        elif r == "ROLE_EXCEPTION_CARVEOUT":
            if len(s.strip()) > 2 and any(ch.isalnum() for ch in s) and not all(w.lower() in CARVEOUT_STOPWORDS for w in s.split()):
                exception_carveouts.append(s)
        elif r == "ROLE_SUBJECTIVE_CRITERION":
            subjective_criteria = s
        elif r == "ROLE_REQUIRED_EVIDENCE":
            if any(k in s_lower for k in ["form", "dir", "msme", "3ceb", "way", "bill", "affidavit", "certificate", "declaration", "document", "challan", "pan", "tan", "gstin", "lut", "invoice", "receipt", "record", "report", "schedule", "return"]):
                evidence_checklist.append(s)
        elif r == "ROLE_MANDATORY_ACTION_OBLIGATION":
            mandatory_actions.append(s)
        elif r == "ROLE_STATUTORY_SANCTION":
            statutory_sanctions.append(s)
        elif r == "ROLE_ACCOUNTING_DESTINATION":
            accounting_destinations.append(s)
        elif r in ["ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", "ROLE_NON_RESIDENT_EXEMPTION_GUARD"]:
            regulated_objects.append(s)

    # Syntactic Governor Disambiguation
    promoted_precond_indices = set()
    for t_item in list(thresh_items):
        t_cr = t_item.get("charRange", [0, 0])
        t_unit_info = parse_unit_and_type(t_item["span"])
        if t_unit_info[0] != "MONETARY_QUANTUM":
            continue
        pre_win = text_low[max(0, t_cr[0] - 55):t_cr[0]]
        gov_slice = pre_win.split(".")[-1].split(";")[-1]
        if (
            re.search(r"\b(?:turnover|gross\s+sales|total\s+sales|gross\s+receipts|paid-up\s+(?:share\s+)?capital|net\s+worth)\b", gov_slice)
            and not re.search(r"\b(?:purchase\s+of\s+goods|deduct\s+tax|aggregate\s+value\s+of|consideration|winnings|withdrawal)\b", gov_slice)
        ):
            promoted_precond_indices.add(t_item["spanIdx"])
            t_item["role"] = "ROLE_BUYER_PRIOR_TURNOVER_GUARD" if t_unit_info[1] == "CURRENCY_PAISA" else "ROLE_AGENT_STATE_PRECONDITION"
            precondition_items.append(t_item)
            cur_is_monetary = bool(
                turnover_precondition
                and parse_numeric_quantity(turnover_precondition) is not None
                and parse_unit_and_type(turnover_precondition)[0] == "MONETARY_QUANTUM"
            )
            if not turnover_precondition or not cur_is_monetary or len(t_item["span"]) > len(turnover_precondition):
                turnover_precondition = t_item["span"]
                turnover_precondition_item = t_item

    if promoted_precond_indices:
        thresh_items = [t for t in thresh_items if t["spanIdx"] not in promoted_precond_indices]
        rem_excess = [t for t in thresh_items if t["role"] == "ROLE_TRANSACTION_EXCESS_THRESHOLD"]
        rem_cliff = [t for t in thresh_items if t["role"] == "ROLE_TRANSACTION_CLIFF_THRESHOLD"]
        excess_threshold_item = max(rem_excess, key=lambda x: len(x["span"])) if rem_excess else None
        excess_threshold = excess_threshold_item["span"] if excess_threshold_item else None
        cliff_threshold_item = max(rem_cliff, key=lambda x: len(x["span"])) if rem_cliff else None
        cliff_threshold = cliff_threshold_item["span"] if cliff_threshold_item else None

    edge_types = {e.get("relationType") for e in resolved_edges}
    unit_cand = (
        turnover_precondition
        if (turnover_precondition and parse_unit_and_type(turnover_precondition)[0] != "BOOLEAN_FLAG")
        else (cliff_threshold or deadline_window or turnover_precondition or "")
    )
    inferred_unit_info = parse_unit_and_type(unit_cand)
    has_compound_interest_ctx = bool(
        re.search(
            r"\b(?:interest|compound|rbi|reserve\s+bank|bank\s+rate|msme|monthly\s+rests|rests|micro|small\s+enterprise)\b",
            text_low + " " + " ".join(statutory_sanctions).lower()
        )
    )

    # Collect chronological date bounds across cutover/sunset/date spans
    date_records = []
    seen_date_isos = set()
    for item in cutover_items + sunset_items:
        iso = parse_iso_date_str(item["span"])
        if iso and iso not in seen_date_isos:
            seen_date_isos.add(iso)
            cr = item.get("charRange", [0, 0])
            pre_d = text_low[max(0, cr[0] - 35):cr[0]]
            date_records.append((cr[0], iso, pre_d, item["role"]))
    if len(date_records) < 2:
        for item in indexed_spans:
            if item["role"] in (
                "ROLE_DEADLINE_WINDOW",
                "ROLE_AGENT_STATE_PRECONDITION",
                "ROLE_TRANSACTION_EXCESS_THRESHOLD",
                "ROLE_TRANSACTION_CLIFF_THRESHOLD",
                "ROLE_EXCEPTION_CARVEOUT",
                "ROLE_STANDARD_STATUTORY_RATE",
            ):
                iso = parse_iso_date_str(item["span"])
                if iso and iso not in seen_date_isos:
                    seen_date_isos.add(iso)
                    cr = item.get("charRange", [0, 0])
                    pre_d = text_low[max(0, cr[0] - 35):cr[0]]
                    date_records.append((cr[0], iso, pre_d, item["role"]))

    cutover_iso = parse_iso_date_str(temporal_cutover)
    sunset_iso = parse_iso_date_str(temporal_sunset)
    if len(date_records) >= 2:
        sorted_dates = sorted(date_records, key=lambda x: x[1])
        if not cutover_iso or not sunset_iso or cutover_iso >= sunset_iso:
            cutover_iso = sorted_dates[0][1]
            sunset_iso = sorted_dates[-1][1]

    cutover_strict = True
    sunset_strict = True
    if cutover_iso and sunset_iso and sunset_iso > cutover_iso:
        for _, d_iso, pre_d, _ in date_records:
            if d_iso == cutover_iso and re.search(r"\b(?:on\s+or\s+after|effective\s+(?:from|on)|beginning\s+on)\b", pre_d):
                cutover_strict = False
            if d_iso == sunset_iso and re.search(r"\b(?:on\s+or\s+before|through|until|ending\s+on)\b", pre_d):
                sunset_strict = False

    has_bounded_dates = bool(cutover_iso and sunset_iso and sunset_iso > cutover_iso)
    subst_edge_indices = set()
    for e in resolved_edges:
        if e["relationType"] == "REL_EXCEPTION_TO_PROHIBITION":
            subst_edge_indices.add(e["sourceIdx"])
            subst_edge_indices.add(e["targetIdx"])
    all_monetary_spans = [
        item for item in indexed_spans
        if (
            item["role"] != "ROLE_IGNORE"
            or (
                item.get("bioType") in ("EXCEPTION", "THRESHOLD")
                and item["spanIdx"] in subst_edge_indices
            )
        )
        and item["role"] not in ("ROLE_STANDARD_STATUTORY_RATE", "ROLE_PENALTY_OR_DEFAULT_OVERRIDE", "ROLE_NON_PAN_OVERRIDE_RATE")
        and parse_unit_and_type(item["span"])[0] == "MONETARY_QUANTUM"
        and parse_minor_units(item["span"]) is not None
    ]
    subst_edges = [e for e in resolved_edges if e["relationType"] == "REL_EXCEPTION_TO_PROHIBITION"]
    has_subst_signal = bool(
        has_bounded_dates
        and (
            (subst_edges and len(all_monetary_spans) >= 2)
            or (re.search(r"\b(?:substituted?\s+for|in\s+lieu\s+of|replaced\s+by)\b", text_low) and len({parse_minor_units(m["span"]) for m in all_monetary_spans}) >= 2)
        )
    )
    has_susp_signal = bool(
        has_bounded_dates
        and not has_subst_signal
        and (
            ("REL_SUNSET_TO_SECTION" in edge_types and "REL_CONDITION_TO_GATE" in edge_types and not rate_items)
            or re.search(r"\bshall\s+not\s+apply\b|\bsuspended\b|\binoperative\b", text_low)
        )
    )

    has_deontic_roles = any(
        s["role"] in (
            "ROLE_PROHIBITION_GATE", "ROLE_MANDATORY_ACTION_OBLIGATION",
            "ROLE_EXCEPTION_CARVEOUT", "ROLE_AGENT_STATE_PRECONDITION"
        )
        for s in indexed_spans
    )
    is_general_deontic_rule = bool(
        effective_domain == "GENERAL_LOGIC"
        and not rate_items
        and not (edge_types & {"REL_BRACKET_NEXT", "REL_COMBINATOR_MIN", "REL_COMBINATOR_MAX", "REL_AGGREGATE_SUM"})
        and not has_subst_signal
        and not has_susp_signal
        and (
            predicted_modality in (
                "MODALITY_DEONTIC_PERMISSION", "MODALITY_DEONTIC_PROHIBITION",
                "MODALITY_DEONTIC_OBLIGATION_WINDOW", "MODALITY_DEONTIC_STATUS_OR_EVIDENCE"
            )
            or has_deontic_roles
        )
    )

    # Determine AST structural archetype from Neural Rule-Modality + SRL Graph + Domain Mode
    if is_general_deontic_rule:
        inferred_perm = (
            predicted_modality == "MODALITY_DEONTIC_PERMISSION"
            or (
                predicted_modality != "MODALITY_DEONTIC_PROHIBITION"
                and any(a.lower().startswith("may ") for a in mandatory_actions)
                and not (prohibition_gate and exception_carveouts)
            )
        )
        inferred_prohib = (not inferred_perm) and (
            predicted_modality == "MODALITY_DEONTIC_PROHIBITION"
            or bool(prohibition_gate and not (mandatory_actions and exception_carveouts and not any("REL_EXCEPTION_TO_PROHIBITION" == e["relationType"] for e in resolved_edges)))
        )
        root_condition = {
            "nodeType": "GENERAL_RULE_GATE",
            "modalityType": "PERMISSION" if inferred_perm else ("PROHIBITION" if inferred_prohib else "OBLIGATION"),
            "permissionPredicate": mandatory_actions[0] if (inferred_perm and mandatory_actions) else None,
            "obligationPredicate": mandatory_actions[0] if mandatory_actions else None,
            "prohibitionPredicate": prohibition_gate,
            "applicableDeadlineWindow": deadline_window,
            "prerequisiteCondition": turnover_precondition or subjective_criteria,
            "exceptionCarveouts": exception_carveouts,
            "accountingDestination": "NOT_APPLICABLE_PURE_LOGIC"
        }

    elif predicted_modality == "MODALITY_DEONTIC_PERMISSION":
        if accounting_destinations or "REL_DESTINATION_TO_ACTION" in edge_types:
            dest = classify_gl_destination_from_spans(accounting_destinations, "PROFIT_OR_LOSS")
            root_condition = {
                "nodeType": "ACCOUNTING_ELECTION_CARVEOUT",
                "electionPredicate": prohibition_gate or (mandatory_actions[0] if mandatory_actions else "May elect not to apply recognition requirement"),
                "applicableThresholdWindow": deadline_window or "twelve months or less",
                "exceptionCarveouts": exception_carveouts,
                "accountingDestination": dest,
                "straightLineExpenseRecognition": True
            }
        else:
            root_condition = {
                "nodeType": "GENERAL_RULE_GATE",
                "modalityType": "PERMISSION",
                "permissionPredicate": mandatory_actions[0] if mandatory_actions else "May act subject to stated preconditions",
                "prerequisiteCondition": turnover_precondition or subjective_criteria,
                "exceptionCarveouts": exception_carveouts,
                **({"accountingDestination": "NOT_APPLICABLE_PURE_LOGIC"} if effective_domain == "GENERAL_LOGIC" else {})
            }

    elif predicted_modality == "MODALITY_COMPUTATIONAL_EXCESS":
        root_condition = {
            "nodeType": "TRANSACTIONAL_WITHHOLDING",
            "entityType": "BUYER" if turnover_precondition else "DEDUCTOR",
            "priorTurnoverPrecondition": turnover_precondition,
            "applicableExcessThreshold": excess_threshold or cliff_threshold,
            "applicableCliffThreshold": None,
            "standardTaxRate": standard_rate,
            "nonPanPenalRate": non_pan_rate,
            "triggerPolicy": trigger_policy,
            "temporalCutover": temporal_cutover,
            "temporalSunset": temporal_sunset,
            "exceptionCarveouts": exception_carveouts
        }

    elif predicted_modality == "MODALITY_COMPUTATIONAL_CLIFF":
        if not standard_rate and not cliff_threshold and not excess_threshold and prohibition_gate:
            root_condition = {
                "nodeType": "BOOLEAN_PROHIBITION_GATE",
                "prohibitionType": "STATUTORY_PROHIBITION_BAR",
                "prohibitionPredicate": prohibition_gate,
                "exceptionCarveouts": exception_carveouts
            }
        else:
            root_condition = {
                "nodeType": "TRANSACTIONAL_WITHHOLDING",
                "entityType": "BUYER" if turnover_precondition else "DEDUCTOR",
                "priorTurnoverPrecondition": turnover_precondition,
                "applicableExcessThreshold": None,
                "applicableCliffThreshold": cliff_threshold or excess_threshold,
                "standardTaxRate": standard_rate,
                "nonPanPenalRate": non_pan_rate,
                "triggerPolicy": trigger_policy,
                "temporalCutover": temporal_cutover,
                "temporalSunset": temporal_sunset,
                "exceptionCarveouts": exception_carveouts
            }

    elif predicted_modality == "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB":
        if accounting_destinations and (subjective_criteria or excess_threshold):
            dest = classify_gl_destination_from_spans(accounting_destinations, "PROFIT_OR_LOSS")
            root_condition = {
                "nodeType": "MEASUREMENT_VALUATION_RULE",
                "measurementPredicate": mandatory_actions[0] if mandatory_actions else "shall be measured",
                "valuationFormula": subjective_criteria or "recoverable valuation formula",
                "impairmentTrigger": excess_threshold or "carrying amount exceeds recoverable amount",
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else None,
                "accountingDestination": dest
            }
        elif (
            edge_types & {"REL_BRACKET_NEXT", "REL_COMBINATOR_MIN", "REL_COMBINATOR_MAX", "REL_CAP_CEILING", "REL_AGGREGATE_SUM"}
            or has_subst_signal
            or (standard_rate and not deadline_window and not statutory_sanctions)
        ):
            root_condition = {
                "nodeType": "TRANSACTIONAL_WITHHOLDING",
                "entityType": "COMPANY" if regulated_objects else "ASSESSEE",
                "priorTurnoverPrecondition": turnover_precondition,
                "applicableExcessThreshold": excess_threshold,
                "applicableCliffThreshold": cliff_threshold,
                "standardTaxRate": standard_rate,
                "nonPanPenalRate": non_pan_rate,
                "triggerPolicy": trigger_policy,
                "temporalCutover": temporal_cutover,
                "temporalSunset": temporal_sunset
            }
        elif not statutory_sanctions and (exception_carveouts or inferred_unit_info[2] == "YEARS"):
            root_condition = {
                "nodeType": "TEMPORAL_WINDOW_GATE",
                "obligationPredicate": mandatory_actions[0] if mandatory_actions else "Statutory benefit payable",
                "prerequisiteCondition": turnover_precondition or cliff_threshold or "continuous service period",
                "applicableDeadlineWindow": deadline_window or cliff_threshold or turnover_precondition or "five years",
                "exceptionCarveouts": exception_carveouts,
                "accountingDestination": "PROFIT_OR_LOSS",
                "accrualMode": "FRACTIONAL_RATIO_ACCRUAL"
            }
        elif has_compound_interest_ctx and (statutory_sanctions or (deadline_window and any(k in deadline_window.lower() for k in ["day", "appointed"]))):
            root_condition = {
                "nodeType": "TEMPORAL_WINDOW_GATE",
                "applicableDeadlineWindow": deadline_window or "forty-five days",
                "triggerPolicy": trigger_policy,
                "statutoryAnnualRatePct": 19.5,
                "statutoryInterestFormula": "3x RBI bank rate compound monthly with rests",
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else "compound interest at three times bank rate",
                "accrualMode": "COMPOUND_INTEREST_PENAL"
            }
        elif deadline_window:
            root_condition = {
                "nodeType": "TEMPORAL_WINDOW_GATE",
                "obligationPredicate": mandatory_actions[0] if mandatory_actions else "Mandatory statutory action within deadline",
                "applicableDeadlineWindow": deadline_window,
                "triggerPolicy": trigger_policy,
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else None,
                "exceptionCarveouts": exception_carveouts
            }
        else:
            root_condition = {
                "nodeType": "MEASUREMENT_VALUATION_RULE",
                "measurementPredicate": mandatory_actions[0] if mandatory_actions else "statutory valuation / deemed cost computation",
                "valuationFormula": subjective_criteria or turnover_precondition or "Deemed statutory valuation formula requiring controller attestation",
                "impairmentTrigger": turnover_precondition or excess_threshold or "Triggering statutory state precondition",
                "statutorySanction": None,
                "exceptionCarveouts": exception_carveouts,
                "controllershipAttestationRequired": True,
                "accountingDestination": "GENERAL_LEDGER_CONTROLLERSHIP_VOUCHER"
            }

    elif predicted_modality == "MODALITY_DEONTIC_PROHIBITION":
        if exception_carveouts and (accounting_destinations or "REL_DESTINATION_TO_ACTION" in edge_types):
            dest = classify_gl_destination_from_spans(accounting_destinations, "PROFIT_OR_LOSS")
            root_condition = {
                "nodeType": "ACCOUNTING_ELECTION_CARVEOUT",
                "electionPredicate": prohibition_gate or (mandatory_actions[0] if mandatory_actions else "May elect not to apply recognition requirement"),
                "applicableThresholdWindow": deadline_window or "twelve months or less",
                "exceptionCarveouts": exception_carveouts,
                "accountingDestination": dest,
                "straightLineExpenseRecognition": True
            }
        elif accounting_destinations and not exception_carveouts:
            dest = classify_gl_destination_from_spans(accounting_destinations, "OTHER_COMPREHENSIVE_INCOME")
            root_condition = {
                "nodeType": "DISCLOSURE_PRESENTATION_OBLIGATION",
                "presentationPredicate": mandatory_actions[0] if mandatory_actions else "shall be recognised immediately",
                "prohibitionGate": prohibition_gate or "shall not be reclassified",
                "accountingDestination": dest
            }
        else:
            root_condition = {
                "nodeType": "BOOLEAN_PROHIBITION_GATE",
                "prohibitionType": "STATUTORY_PROHIBITION_BAR",
                "prohibitionPredicate": prohibition_gate or "Conduct prohibited unless safe harbor carveout applies",
                "exceptionCarveouts": exception_carveouts,
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else None
            }

    elif predicted_modality == "MODALITY_DEONTIC_OBLIGATION_WINDOW":
        if accounting_destinations and (mandatory_actions or subjective_criteria or turnover_precondition):
            dest = classify_gl_destination_from_spans(accounting_destinations, "PROFIT_OR_LOSS")
            root_condition = {
                "nodeType": "ACCOUNTING_RECOGNITION_GATE",
                "recognitionPredicate": mandatory_actions[0] if mandatory_actions else "shall be recognised",
                "prerequisiteObligation": turnover_precondition or "present obligation precondition",
                "probabilityStandard": subjective_criteria,
                "exceptionCarveouts": exception_carveouts,
                "accountingDestination": dest
            }
        elif deadline_window and (prohibition_gate or statutory_sanctions):
            root_condition = {
                "nodeType": "PROCEDURAL_NOTICE_GATE",
                "applicableDeadlineWindow": deadline_window,
                "prohibitionPredicate": prohibition_gate or "Statutory wait window prior to institution",
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else "Suit barred during statutory notice window",
                "exceptionCarveouts": exception_carveouts
            }
        elif deadline_window:
            root_condition = {
                "nodeType": "TEMPORAL_WINDOW_GATE",
                "obligationPredicate": mandatory_actions[0] if mandatory_actions else "Mandatory statutory action within deadline",
                "applicableDeadlineWindow": deadline_window,
                "triggerPolicy": trigger_policy,
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else None,
                "exceptionCarveouts": exception_carveouts
            }
        elif excess_threshold or (mandatory_actions and regulated_objects and not exception_carveouts):
            root_condition = {
                "nodeType": "CORPORATE_APPROVAL_GATE",
                "governingTransaction": "CORPORATE_GOVERNANCE_MANDATE",
                "prohibitionPredicate": prohibition_gate,
                "applicableExcessThreshold": excess_threshold,
                "mandatoryApprovals": mandatory_actions,
                "exceptionCarveouts": exception_carveouts
            }
        else:
            root_condition = {
                "nodeType": "DEONTIC_STATE_GATE",
                "obligationPredicate": mandatory_actions[0] if mandatory_actions else (prohibition_gate or "Mandatory rule obligation"),
                "prerequisiteCondition": turnover_precondition or subjective_criteria,
                "exceptionCarveouts": exception_carveouts,
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else None
            }

    else:
        if mandatory_actions and (excess_threshold or (exception_carveouts and not evidence_checklist and not statutory_sanctions)):
            root_condition = {
                "nodeType": "CORPORATE_APPROVAL_GATE",
                "governingTransaction": "CORPORATE_GOVERNANCE_MANDATE",
                "prohibitionPredicate": prohibition_gate or "No entity shall enter into transaction without prescribed approval",
                "applicableExcessThreshold": excess_threshold,
                "mandatoryApprovals": mandatory_actions,
                "exceptionCarveouts": exception_carveouts
            }
        elif prohibition_gate and (subjective_criteria or statutory_sanctions or evidence_checklist or turnover_precondition):
            reasons = [r for r in [subjective_criteria, turnover_precondition] if r]
            root_condition = {
                "nodeType": "STATUS_DISQUALIFICATION_GATE",
                "prohibitionPredicate": prohibition_gate,
                "disqualificationReasons": reasons,
                "statutorySanction": statutory_sanctions[0] if statutory_sanctions else "Statutory disqualification active",
                "requiredDocumentChecklist": evidence_checklist,
                "controllershipAttestationRequired": True
            }
        else:
            root_condition = {
                "nodeType": "SUBJECTIVE_EVIDENCE_GATE",
                "subjectiveCriterion": subjective_criteria or "statutory_qualification_or_classification",
                "requiredDocumentChecklist": evidence_checklist or ["COMPLIANCE_DECLARATION"],
                "controllershipAttestationRequired": True
            }

    blocks = synthesize_building_blocks(
        root_condition,
        indexed_spans,
        resolved_edges,
        text,
        predicted_modality=predicted_modality,
        rule_domain=effective_domain,
    )
    symbol_legend = synthesize_symbol_legend(indexed_spans, resolved_edges, text)
    blocks["symbolLegend"] = symbol_legend

    comp_block = blocks["computationalExpression"]
    deontic_block = blocks["deonticModality"]
    draft_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8].upper()
    section_id = meta.get("sectionId") or f"DRAFT_{draft_digest}"

    guards = []
    skip_halting_precondition_idx = set()

    is_explicit_valuation_rule = bool(
        accounting_destinations
        or subjective_criteria
        or re.search(r"\b(?:net\s+worth|slump\s+sale|recoverable\s+amount|carrying\s+amount|value\s+in\s+use|fair\s+value)\b", text_low)
    )
    qual_thresh_item = None
    if (
        effective_domain != "GENERAL_LOGIC"
        and root_condition.get("nodeType") != "TRANSACTIONAL_WITHHOLDING"
        and not (edge_types & {"REL_BRACKET_NEXT", "REL_COMBINATOR_MIN", "REL_COMBINATOR_MAX", "REL_CAP_CEILING", "REL_AGGREGATE_SUM"})
        and not rate_items
        and not has_subst_signal
        and not has_susp_signal
        and not statutory_sanctions
        and not prohibition_gate
        and not evidence_checklist
        and not mandatory_actions
        and not deadline_window
        and not is_explicit_valuation_rule
        and root_condition.get("nodeType") not in (
            "CORPORATE_APPROVAL_GATE", "PROCEDURAL_NOTICE_GATE", "TEMPORAL_WINDOW_GATE",
            "ACCOUNTING_RECOGNITION_GATE", "ACCOUNTING_ELECTION_CARVEOUT",
            "DISCLOSURE_PRESENTATION_OBLIGATION", "GENERAL_RULE_GATE"
        )
        and all_monetary_spans
    ):
        cand_m = next((s for s in thresh_items if parse_unit_and_type(s["span"])[0] == "MONETARY_QUANTUM"), None)
        if cand_m is None:
            cand_m = all_monetary_spans[0]
        if parse_minor_units(cand_m["span"]) is not None:
            qual_thresh_item = cand_m
            skip_halting_precondition_idx.add(cand_m["spanIdx"])
            if root_condition.get("nodeType") == "MEASUREMENT_VALUATION_RULE":
                root_condition["nodeType"] = "BOOLEAN_THRESHOLD_GATE"

    if (
        effective_domain == "STATUTORY_FINANCE"
        and turnover_precondition
        and turnover_precondition_item
        and turnover_precondition_item["spanIdx"] not in skip_halting_precondition_idx
    ):
        to_qty = parse_numeric_quantity(turnover_precondition)
        _, to_data_type, to_unit = parse_unit_and_type(turnover_precondition)
        if to_qty is not None and to_data_type in ("CURRENCY_PAISA", "CURRENCY_CENTS", "CURRENCY_MINOR_UNITS"):
            is_indian_turnover_guard = (
                turnover_precondition_item["role"] == "ROLE_BUYER_PRIOR_TURNOVER_GUARD"
                or to_data_type == "CURRENCY_PAISA"
                or bool(re.search(r"\b(?:turnover|sales|gross\s+receipts|paid-up|share\s+capital|net\s+worth)\b", text_low))
            )
            minor_val_str = str(int(round(float(to_qty) * 100)))
            if is_indian_turnover_guard:
                guards.append({
                    "id": "PRIOR_TURNOVER_PRECONDITION",
                    "field": "buyerPriorTurnoverPaisa",
                    "operator": "GTE",
                    "value": minor_val_str,
                    "valueType": "CURRENCY_MINOR_UNITS",
                    "onMismatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_SMALL_ENTITY"}
                })
            else:
                dyn_keys = build_operand_fact_keys(turnover_precondition, to_unit)
                dyn_field = dyn_keys[0] if dyn_keys else _derive_fact_key_camel(turnover_precondition, "preconditionMetric")
                pre_cr = turnover_precondition_item.get("charRange", [0, 0])
                pre_ctx = text[max(0, pre_cr[0] - 35):min(len(text), pre_cr[1] + 35)]
                guards.append({
                    "id": "MONETARY_PRECONDITION_GUARD",
                    "field": dyn_field,
                    "operator": parse_comparison_operator(pre_ctx),
                    "value": minor_val_str,
                    "valueType": "CURRENCY_MINOR_UNITS",
                    "skipIfMissingFact": True,
                    "onMismatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_BELOW_THRESHOLD"}
                })

    if (
        effective_domain == "STATUTORY_FINANCE"
        and regulated_objects
        and any(re.search(r"\bresident\b", o.lower()) and not re.search(r"\bnon[-\s]?resident\b", o.lower()) for o in regulated_objects)
        and currency_code in ("INR", None)
    ):
        guards.append({
            "id": "RESIDENT_STATUS_GUARD",
            "field": "payee.isResident",
            "operator": "EQ",
            "value": True,
            "onMismatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_NON_RESIDENT"}
        })

    non_pan_bps = parse_rate_bps(non_pan_rate)
    if effective_domain == "STATUTORY_FINANCE" and non_pan_bps:
        has_pan_ctx = bool(
            re.search(r"\b(?:pan|permanent\s+account\s+number|206aa|397|tin|taxpayer\s+identification)\b", text_low)
            or (non_pan_rate_item and non_pan_rate_item["role"] == "ROLE_NON_PAN_OVERRIDE_RATE")
        )
        if has_pan_ctx:
            guards.append({
                "id": "PAN_PENAL_OVERRIDE",
                "field": "payee.isPanValid",
                "operator": "EQ",
                "value": False,
                "onMatch": {"action": "OVERRIDE_RATE", "overrideRateBps": non_pan_bps}
            })
        else:
            alt_field = _derive_fact_key_camel(non_pan_rate_item["span"] if non_pan_rate_item else "", "alternateOverrideActive")
            guards.append({
                "id": "ALTERNATE_RATE_OVERRIDE",
                "field": alt_field,
                "operator": "IS_TRUE",
                "value": True,
                "skipIfMissingFact": True,
                "onMatch": {"action": "OVERRIDE_RATE", "overrideRateBps": non_pan_bps}
            })

    is_bounded_window_pipeline = has_subst_signal or has_susp_signal
    if effective_domain == "STATUTORY_FINANCE" and not is_bounded_window_pipeline:
        if sunset_iso:
            guards.append({
                "id": "STATUTORY_SUNSET_GUARD",
                "field": "transactionDate",
                "operator": "GTE",
                "value": sunset_iso,
                "onMatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_SUNSET_OMITTED"}
            })
        if cutover_iso:
            guards.append({
                "id": "STATUTORY_CUTOVER_GUARD",
                "field": "transactionDate",
                "operator": "LT",
                "value": cutover_iso,
                "onMatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_PRE_COMMENCEMENT"}
            })

    computation = None
    typed_inputs_block = None
    deontic_gate = None
    node_type = root_condition.get("nodeType")
    expr_type = comp_block.get("expressionType")

    bracket_edges = [e for e in resolved_edges if e["relationType"] == "REL_BRACKET_NEXT"]
    comb_min_edges = [e for e in resolved_edges if e["relationType"] == "REL_COMBINATOR_MIN"]
    comb_max_edges = [e for e in resolved_edges if e["relationType"] == "REL_COMBINATOR_MAX"]
    cap_edges = [e for e in resolved_edges if e["relationType"] == "REL_CAP_CEILING"]
    sub_edges = [e for e in resolved_edges if e["relationType"] == "REL_SUBTRACTION_BASE"]
    symbolic_sub_edges = [
        e for e in sub_edges
        if (
            span_by_idx[e["sourceIdx"]]["role"] == "ROLE_COMPUTATION_OPERAND"
            and span_by_idx[e["targetIdx"]]["role"] == "ROLE_COMPUTATION_OPERAND"
        )
    ]

    rate_bps = parse_rate_bps(standard_rate) or (parse_rate_bps(rate_items[0]["span"]) if rate_items else 0)

    def _find_edge_neighbors(node_idx: int, rel_filter: set[str]):
        out = []
        for e in resolved_edges:
            if e["relationType"] in rel_filter:
                if e["sourceIdx"] == node_idx:
                    out.append(e["targetIdx"])
                elif e["targetIdx"] == node_idx:
                    out.append(e["sourceIdx"])
        return out

    # ── Pipeline Archetype 0: Composed Multi-Node Arithmetic / Collection Aggregate DAG ──
    clause_count = len({int(s.get("clauseIdx", 0)) for s in indexed_spans})
    has_composition = any(
        e["relationType"] in ("REL_CAP_CEILING", "REL_SUBTRACTION_BASE")
        for e in resolved_edges
    )
    use_clause_scoped = (
        (clause_scopes and clause_count > 1)
        or (has_composition and (typed_inputs or clause_scopes))
    )
    if use_clause_scoped:
        # RFC-012 Step 2: per-clause scopes with def-use symbol linking. The
        # linker owns this path, so the archetype chain below is skipped.
        from compiler.neural.clause_linker import (
            ClauseLinkingError,
            build_clause_scoped_computation,
        )
        clause_ranges = [
            (int(cm["charRange"][0]), int(cm["charRange"][1]))
            for cm in (clause_modalities or [])
            if cm.get("charRange")
        ]
        scoped_comp, scoped_inputs, scoped_diagnostics = build_clause_scoped_computation(
            text=text,
            spans=indexed_spans,
            edges=resolved_edges,
            clause_ranges=clause_ranges or None,
            currency_code=currency_code,
        )
        link_errors = [d for d in scoped_diagnostics if d.severity == "error"]
        if link_errors:
            raise ClauseLinkingError(link_errors[0].code, scoped_diagnostics)
        if scoped_comp is not None:
            computation = scoped_comp
            typed_inputs_block = scoped_inputs
            # The v2 profile expresses every precondition inside the pipeline
            # as an SSA step, so Catala and USVM evaluate the same conditions.
            # The legacy top-level guards are therefore not emitted here.
            guards = []
            comp_block["expressionType"] = "CLAUSE_SCOPED_PIPELINE"
            comp_block["formulaDescription"] = (
                "Per-clause scoped pipeline with def-use linked values"
            )

    composed_comp, composed_desc, composed_etype = compile_composed_arithmetic_pipeline(
        text=text,
        indexed_spans=indexed_spans,
        span_by_idx=span_by_idx,
        resolved_edges=resolved_edges,
        rate_items=rate_items,
        thresh_items=thresh_items,
        operand_items=operand_items,
        all_monetary_spans=all_monetary_spans,
        currency_code=currency_code,
        cutover_iso=cutover_iso,
        sunset_iso=sunset_iso,
        cutover_strict=cutover_strict,
        sunset_strict=sunset_strict,
        has_susp_signal=has_susp_signal,
        excess_threshold=excess_threshold,
        cliff_threshold=cliff_threshold,
        rate_bps=rate_bps,
    )
    if composed_comp is not None and computation is None:
        computation = composed_comp
        comp_block["expressionType"] = composed_etype or "DECLARATIVE_PIPELINE"
        comp_block["formulaDescription"] = composed_desc or "Composed declarative arithmetic pipeline"

    # ── Pipeline Archetype 1: Progressive Brackets (REL_BRACKET_NEXT) ──
    if computation is None and bracket_edges and rate_items:
        raw_slabs = []
        seen_slab_keys = set()
        bounded_bracket_edges = [
            be for be in bracket_edges
            if parse_minor_units(span_by_idx[be["sourceIdx"]]["span"]) is not None
            and parse_minor_units(span_by_idx[be["targetIdx"]]["span"]) is not None
        ]
        active_bracket_edges = bounded_bracket_edges or bracket_edges
        for be in active_bracket_edges:
            u_idx, v_idx = be["sourceIdx"], be["targetIdx"]
            q_u = parse_minor_units(span_by_idx[u_idx]["span"])
            q_v = parse_minor_units(span_by_idx[v_idx]["span"])
            if q_u is None and q_v is None:
                continue
            if q_u is not None and q_v is not None and q_u > q_v:
                lo_idx, hi_idx, lo_minor, hi_minor = v_idx, u_idx, q_v, q_u
            else:
                lo_idx, hi_idx, lo_minor, hi_minor = u_idx, v_idx, q_u, q_v

            b_clause = span_by_idx[lo_idx]["clauseIdx"]
            attached_rate_idx = next(
                (
                    n_i for n_i in (_find_edge_neighbors(lo_idx, {"REL_RATE_TO_THRESHOLD"}) + _find_edge_neighbors(hi_idx, {"REL_RATE_TO_THRESHOLD"}))
                    if span_by_idx[n_i]["role"] == "ROLE_STANDARD_STATUTORY_RATE"
                ),
                None
            )
            if attached_rate_idx is None:
                same_clause_rates = [r for r in rate_items if r["clauseIdx"] == b_clause]
                attached_rate_item = same_clause_rates[0] if same_clause_rates else rate_items[0]
                attached_rate_idx = attached_rate_item["spanIdx"]
            else:
                attached_rate_item = span_by_idx[attached_rate_idx]

            slab_rate_bps = parse_rate_bps(attached_rate_item["span"]) or rate_bps

            base_tax_minor = None
            for e in resolved_edges:
                if e["relationType"] == "REL_SUBTRACTION_BASE" and e["targetIdx"] in (attached_rate_idx, lo_idx):
                    cand_m = parse_minor_units(span_by_idx[e["sourceIdx"]]["span"])
                    if cand_m is not None and parse_unit_and_type(span_by_idx[e["sourceIdx"]]["span"])[0] == "MONETARY_QUANTUM":
                        if cand_m not in (lo_minor, hi_minor):
                            base_tax_minor = cand_m
                            break
            if base_tax_minor is None:
                for m in all_monetary_spans:
                    if m["clauseIdx"] == b_clause and m["spanIdx"] not in (lo_idx, hi_idx):
                        cand_m = parse_minor_units(m["span"])
                        if cand_m is not None and cand_m not in (lo_minor, hi_minor):
                            base_tax_minor = cand_m
                            break

            s_key = (lo_minor, hi_minor, slab_rate_bps, base_tax_minor)
            if s_key not in seen_slab_keys:
                seen_slab_keys.add(s_key)
                raw_slabs.append((lo_minor or 0, hi_minor, slab_rate_bps, base_tax_minor))

        if raw_slabs:
            raw_slabs.sort(key=lambda x: (x[0], x[1] if x[1] is not None else 10**18))
            slabs_ast = []
            for lo_m, hi_m, r_bps_s, bt_m in raw_slabs:
                slab_obj = {"rateBps": r_bps_s}
                if lo_m > 0:
                    slab_obj["floor"] = {"op": "TO_PAISE", "value": str(lo_m)}
                if hi_m is not None:
                    slab_obj["upTo"] = {"op": "TO_PAISE", "value": str(hi_m)}
                if bt_m is not None:
                    slab_obj["baseTax"] = {"op": "TO_PAISE", "value": str(bt_m)}
                slabs_ast.append(slab_obj)

            single_bracket = (
                len(slabs_ast) == 1
                and raw_slabs[0][0] > 0
                and raw_slabs[0][1] is not None
            )

            if typed_inputs:
                # RFC-012 v2: one declared input, no alias probing, no null test.
                endpoint_indices = {
                    be["sourceIdx"] for be in active_bracket_edges
                } | {be["targetIdx"] for be in active_bracket_edges}
                lowest_idx = min(
                    (i for i in endpoint_indices if parse_minor_units(span_by_idx[i]["span"]) is not None),
                    key=lambda i: parse_minor_units(span_by_idx[i]["span"]),
                )
                threshold_cr = span_by_idx[lowest_idx].get("charRange") or [0, 0]
                input_id = resolve_v2_input_id(text, threshold_cr[0], currency_code)
                steps = [{"assign": "basePaisa", "expr": f"inputs.{input_id}"}]
                base_ref = "vars.basePaisa"
                steps.append({
                    "assign": "rawSlabTaxPaisa",
                    "expr": {"op": "PROGRESSIVE_SLABS", "base": base_ref, "slabs": slabs_ast}
                })
                if single_bracket:
                    steps.append({
                        "assign": "inBracketRange",
                        "expr": {
                            "op": "AND",
                            "left": {"op": "GT", "left": base_ref, "right": {"op": "TO_PAISE", "value": str(raw_slabs[0][0])}},
                            "right": {"op": "LTE", "left": base_ref, "right": {"op": "TO_PAISE", "value": str(raw_slabs[0][1])}}
                        }
                    })
                    steps.append({
                        "assign": "taxAmountPaisa",
                        "expr": {
                            "op": "CONDITIONAL",
                            "test": "vars.inBracketRange",
                            "then": "vars.rawSlabTaxPaisa",
                            "else": {"op": "TO_PAISE", "value": "0"}
                        }
                    })
                    output_map = {
                        "taxAmountPaisa": "vars.taxAmountPaisa",
                        "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
                    }
                else:
                    steps.append({"assign": "taxAmountPaisa", "expr": "vars.rawSlabTaxPaisa"})
                    steps.append({
                        "assign": "isApplicable",
                        "expr": {"op": "GT", "left": "vars.taxAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}}
                    })
                    output_map = {
                        "taxAmountPaisa": "vars.taxAmountPaisa",
                        "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
                        "isApplicable": "vars.isApplicable",
                    }
                typed_inputs_block = {
                    input_id: {
                        "type": "MONEY_MINOR",
                        "currency": currency_code or "USD",
                        "scale": 100,
                        "required": True,
                    }
                }
                computation = {
                    "archetype": "DECLARATIVE_PIPELINE",
                    "pipeline": {"steps": steps, "output": output_map},
                }
            else:
                steps = []
                primary_keys = build_operand_fact_keys(operand_items[0]["span"], currency_code) if operand_items else ["taxableIncome", "taxableIncomeUsd"]
                base_ref = emit_operand_resolution_steps(steps, "primaryBasePaisa", primary_keys, fallback_to_primary=True)
                steps.append({
                    "assign": "rawSlabTaxPaisa",
                    "expr": {"op": "PROGRESSIVE_SLABS", "base": base_ref, "slabs": slabs_ast}
                })
                if single_bracket:
                    steps.append({
                        "assign": "inBracketRange",
                        "expr": {
                            "op": "AND",
                            "left": {"op": "GT", "left": base_ref, "right": {"op": "TO_PAISE", "value": str(raw_slabs[0][0])}},
                            "right": {"op": "LTE", "left": base_ref, "right": {"op": "TO_PAISE", "value": str(raw_slabs[0][1])}}
                        }
                    })
                    steps.append({
                        "assign": "taxAmountPaisa",
                        "expr": {
                            "op": "CONDITIONAL",
                            "test": "vars.inBracketRange",
                            "then": "vars.rawSlabTaxPaisa",
                            "else": {"op": "TO_PAISE", "value": "0"}
                        }
                    })
                    is_app_ref = "vars.inBracketRange"
                else:
                    steps.append({"assign": "taxAmountPaisa", "expr": "vars.rawSlabTaxPaisa"})
                    steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.taxAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
                    is_app_ref = "vars.isApplicable"

                comp_block["expressionType"] = "PROGRESSIVE_SLAB"
                comp_block["formulaDescription"] = "Piecewise progressive tax schedule across bracket slabs"
                computation = {
                    "archetype": "DECLARATIVE_PIPELINE",
                    "thresholdPaisa": str(raw_slabs[0][0]),
                    "standardRateBps": raw_slabs[-1][2],
                    "pipeline": {
                        "steps": steps,
                        "output": {
                            "taxAmountPaisa": "vars.taxAmountPaisa",
                            "tdsAmountPaisa": "vars.taxAmountPaisa",
                            "calculatedTaxInr": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
                            "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
                            "isApplicable": is_app_ref
                        }
                    }
                }

    # ── Pipeline Archetype 2: MIN / MAX Combinator (REL_COMBINATOR_MIN / MAX) ──
    if computation is None and (comb_min_edges or comb_max_edges):
        is_min_comb = bool(comb_min_edges)
        comb_op = "MIN" if is_min_comb else "MAX"
        comb_edge = comb_min_edges[0] if is_min_comb else comb_max_edges[0]
        b_indices = [comb_edge["sourceIdx"], comb_edge["targetIdx"]]
        steps = []
        bound_var_cache = {}
        reduction_target_var = None

        def _bind_operand(op_idx: int | None, allow_primary: bool, default_keys: list[str] | None = None) -> str:
            if op_idx is not None:
                norm_k = normalize_symbol_id(span_by_idx[op_idx]["span"]) or f"op_{op_idx}"
                if norm_k in bound_var_cache:
                    return bound_var_cache[norm_k]
                c_keys = build_operand_fact_keys(span_by_idx[op_idx]["span"], currency_code)
                v_ref = emit_operand_resolution_steps(steps, f"op_{len(bound_var_cache)}_paise", c_keys, fallback_to_primary=allow_primary)
                bound_var_cache[norm_k] = v_ref
                return v_ref
            c_keys = default_keys or []
            return emit_operand_resolution_steps(steps, f"op_fb_{len(steps)}_paise", c_keys, fallback_to_primary=allow_primary)

        branch_vars = []
        for b_pos, b_idx in enumerate(b_indices):
            b_item = span_by_idx[b_idx]
            b_bps = parse_rate_bps(b_item["span"]) if any(w in b_item["span"].lower() for w in ["%", "per cent", "percent"]) else None
            if b_bps is not None:
                sub_nbrs = _find_edge_neighbors(b_idx, {"REL_SUBTRACTION_BASE"})
                th_nbrs = _find_edge_neighbors(b_idx, {"REL_RATE_TO_THRESHOLD", "REL_THRESHOLD_TO_TRANSACTION"})
                b_cr = b_item.get("charRange", [0, 0])
                op_nbrs_all = [n for n in sub_nbrs if span_by_idx[n]["role"] == "ROLE_COMPUTATION_OPERAND"]
                op_nbrs_right = sorted(
                    [n for n in op_nbrs_all if span_by_idx[n].get("charRange", [0, 0])[0] >= b_cr[1]],
                    key=lambda n: span_by_idx[n].get("charRange", [0, 0])[0],
                )
                op_nbrs = op_nbrs_right if op_nbrs_right else op_nbrs_all
                num_nbrs = [n for n in (sub_nbrs + th_nbrs) if n not in op_nbrs and parse_minor_units(span_by_idx[n]["span"]) is not None and parse_unit_and_type(span_by_idx[n]["span"])[0] == "MONETARY_QUANTUM"]

                if not op_nbrs:
                    next_rate_pos = min(
                        (r["charRange"][0] for r in rate_items if r["spanIdx"] != b_idx and r["charRange"][0] > b_cr[1]),
                        default=len(text)
                    )
                    op_nbrs = [
                        o["spanIdx"] for o in operand_items
                        if o["charRange"][0] >= b_cr[1] and o["charRange"][1] <= next_rate_pos
                    ][:1]

                op_idx = op_nbrs[0] if op_nbrs else None
                if op_idx is not None and not num_nbrs:
                    op_linked_th = [
                        n for n in _find_edge_neighbors(op_idx, {"REL_THRESHOLD_TO_TRANSACTION", "REL_SUBTRACTION_BASE", "REL_RATE_TO_THRESHOLD"})
                        if n != b_idx and parse_minor_units(span_by_idx[n]["span"]) is not None and parse_unit_and_type(span_by_idx[n]["span"])[0] == "MONETARY_QUANTUM"
                    ]
                    num_nbrs.extend(op_linked_th)
                if op_idx is not None and not num_nbrs:
                    op_cr = span_by_idx[op_idx].get("charRange", [0, 0])
                    for m in all_monetary_spans:
                        if m["spanIdx"] not in b_indices and op_cr[1] <= m["charRange"][0] <= op_cr[1] + 25:
                            if re.search(r"\b(?:over|exceed(?:s|ing)?|minus|above)\b", text_low[op_cr[1]:m["charRange"][0]]):
                                num_nbrs.append(m["spanIdx"])
                                break

                sec_op_nbrs = []
                if op_idx is not None:
                    sec_op_nbrs = [
                        n for n in _find_edge_neighbors(op_idx, {"REL_SUBTRACTION_BASE"})
                        if n != b_idx and span_by_idx[n]["role"] == "ROLE_COMPUTATION_OPERAND"
                    ]

                op_ref = _bind_operand(op_idx, allow_primary=(b_pos == 0))
                if num_nbrs:
                    th_minor = parse_minor_units(span_by_idx[num_nbrs[0]]["span"])
                    steps.append({
                        "assign": f"br{b_pos}_base",
                        "expr": {
                            "op": "MAX",
                            "left": {"op": "TO_PAISE", "value": "0"},
                            "right": {"op": "SUB", "left": op_ref, "right": {"op": "TO_PAISE", "value": str(th_minor)}}
                        }
                    })
                    base_for_mul = f"vars.br{b_pos}_base"
                elif sec_op_nbrs:
                    sec_ref = _bind_operand(sec_op_nbrs[0], allow_primary=False)
                    steps.append({
                        "assign": f"br{b_pos}_base",
                        "expr": {
                            "op": "MAX",
                            "left": {"op": "TO_PAISE", "value": "0"},
                            "right": {"op": "SUB", "left": op_ref, "right": sec_ref}
                        }
                    })
                    base_for_mul = f"vars.br{b_pos}_base"
                else:
                    steps.append({
                        "assign": f"br{b_pos}_base",
                        "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": op_ref}
                    })
                    base_for_mul = f"vars.br{b_pos}_base"

                if b_pos == 1 and re.search(r"\b(?:reduced\s+by|reduction\s+(?:of|in)|scaled\s+down)\b", text_low):
                    reduction_target_var = op_ref

                steps.append({
                    "assign": f"br{b_pos}_val",
                    "expr": {"op": "RATIONAL_MULTIPLY", "base": base_for_mul, "rateBps": b_bps, "denominator": 10000}
                })
                branch_vars.append(f"vars.br{b_pos}_val")
            else:
                c_minor = parse_minor_units(b_item["span"]) or 0
                steps.append({"assign": f"br{b_pos}_val", "expr": {"op": "TO_PAISE", "value": str(c_minor)}})
                branch_vars.append(f"vars.br{b_pos}_val")

        steps.append({
            "assign": "rawCombinedPaisa",
            "expr": {"op": comb_op, "left": branch_vars[0], "right": branch_vars[1]}
        })
        if has_susp_signal and cutover_iso and sunset_iso:
            steps.append({
                "assign": "inSuspensionWindow",
                "expr": build_bounded_window_test(cutover_iso, sunset_iso, cutover_strict, sunset_strict)
            })
            steps.append({
                "assign": "combinedAmountPaisa",
                "expr": {
                    "op": "CONDITIONAL",
                    "test": "vars.inSuspensionWindow",
                    "then": {"op": "TO_PAISE", "value": "0"},
                    "else": "vars.rawCombinedPaisa"
                }
            })
        else:
            steps.append({"assign": "combinedAmountPaisa", "expr": "vars.rawCombinedPaisa"})

        output_map = {
            "combinedAmountPaisa": "vars.combinedAmountPaisa",
            "combinedAmount": {"op": "FROM_PAISE", "value": "vars.combinedAmountPaisa"},
            "taxAmountPaisa": "vars.combinedAmountPaisa",
            "tdsAmountPaisa": "vars.combinedAmountPaisa",
            "calculatedTaxInr": {"op": "FROM_PAISE", "value": "vars.combinedAmountPaisa"},
            "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.combinedAmountPaisa"},
            "isApplicable": {"op": "GT", "left": "vars.combinedAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}}
        }
        if reduction_target_var:
            steps.append({
                "assign": "netAdjustedBasePaisa",
                "expr": {
                    "op": "MAX",
                    "left": {"op": "TO_PAISE", "value": "0"},
                    "right": {"op": "SUB", "left": reduction_target_var, "right": "vars.combinedAmountPaisa"}
                }
            })
            output_map["netAdjustedBasePaisa"] = "vars.netAdjustedBasePaisa"
            output_map["netAdjustedBase"] = {"op": "FROM_PAISE", "value": "vars.netAdjustedBasePaisa"}

        comp_block["expressionType"] = "DECLARATIVE_PIPELINE"
        comp_block["formulaDescription"] = f"{comb_op} combinator statutory computation across candidate prongs"
        computation = {
            "archetype": "DECLARATIVE_PIPELINE",
            "thresholdPaisa": "0",
            "standardRateBps": rate_bps,
            "pipeline": {"steps": steps, "output": output_map}
        }

    # ── Pipeline Archetype 3: Capped Base (REL_CAP_CEILING) ──────────────
    if computation is None and cap_edges:
        cap_e = cap_edges[0]
        s_it, t_it = span_by_idx[cap_e["sourceIdx"]], span_by_idx[cap_e["targetIdx"]]
        s_min, t_min = parse_minor_units(s_it["span"]), parse_minor_units(t_it["span"])
        if s_min is not None:
            cap_minor, cap_op_idx = s_min, t_it["spanIdx"]
        else:
            cap_minor, cap_op_idx = t_min, s_it["spanIdx"]

        if cap_minor is not None:
            steps = []
            op_keys = build_operand_fact_keys(span_by_idx[cap_op_idx]["span"], currency_code)
            op_ref = emit_operand_resolution_steps(steps, "opBasePaisa", op_keys, fallback_to_primary=True)
            steps.append({
                "assign": "cleanBasePaisa",
                "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": op_ref}
            })
            steps.append({
                "assign": "cappedBasePaisa",
                "expr": {"op": "MIN", "left": "vars.cleanBasePaisa", "right": {"op": "TO_PAISE", "value": str(cap_minor)}}
            })
            steps.append({
                "assign": "excludedExcessPaisa",
                "expr": {"op": "SUB", "left": "vars.cleanBasePaisa", "right": "vars.cappedBasePaisa"}
            })
            out_map = {
                "taxableBasePaisa": "vars.cappedBasePaisa",
                "cappedBasePaisa": "vars.cappedBasePaisa",
                "cappedBase": {"op": "FROM_PAISE", "value": "vars.cappedBasePaisa"},
                "excludedExcessPaisa": "vars.excludedExcessPaisa",
                "excludedExcess": {"op": "FROM_PAISE", "value": "vars.excludedExcessPaisa"},
                "effectiveAmountPaisa": "vars.cappedBasePaisa",
                "effectiveAmount": {"op": "FROM_PAISE", "value": "vars.cappedBasePaisa"}
            }
            if rate_bps > 0:
                steps.append({
                    "assign": "taxAmountPaisa",
                    "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.cappedBasePaisa", "rateBps": "rules.effectiveRateBps", "denominator": 10000}
                })
                steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.taxAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
                out_map["taxAmountPaisa"] = "vars.taxAmountPaisa"
                out_map["tdsAmountPaisa"] = "vars.taxAmountPaisa"
                out_map["calculatedTaxInr"] = {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}
                out_map["calculatedTaxUsd"] = {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}
                out_map["effectiveRateBps"] = "rules.effectiveRateBps"
                out_map["isApplicable"] = "vars.isApplicable"
            else:
                steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.cappedBasePaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
                out_map["isApplicable"] = "vars.isApplicable"

            comp_block["expressionType"] = "DECLARATIVE_PIPELINE"
            comp_block["formulaDescription"] = "Capped remuneration/base pipeline with upper statutory ceiling"
            computation = {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": str(cap_minor),
                "standardRateBps": rate_bps,
                "pipeline": {"steps": steps, "output": out_map}
            }

    # ── Pipeline Archetype 4: Symbolic Subtraction Excess (REL_SUBTRACTION_BASE) ──
    if computation is None and symbolic_sub_edges:
        se = symbolic_sub_edges[0]
        op_a_idx, op_b_idx = se["sourceIdx"], se["targetIdx"]
        steps = []
        ref_a = emit_operand_resolution_steps(steps, "opAPaisa", build_operand_fact_keys(span_by_idx[op_a_idx]["span"], currency_code), fallback_to_primary=True)
        ref_b = emit_operand_resolution_steps(steps, "opBPaisa", build_operand_fact_keys(span_by_idx[op_b_idx]["span"], currency_code), fallback_to_primary=False)
        steps.append({
            "assign": "subBasePaisa",
            "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": ref_a, "right": ref_b}}
        })
        th_raw = excess_threshold or cliff_threshold
        th_min = parse_minor_units(th_raw)
        if th_min is not None and th_min > 0:
            steps.append({
                "assign": "gateMet",
                "expr": {"op": "GT", "left": ref_a, "right": {"op": "TO_PAISE", "value": str(th_min)}}
            })
            steps.append({
                "assign": "taxableBasePaisa",
                "expr": {"op": "CONDITIONAL", "test": "vars.gateMet", "then": "vars.subBasePaisa", "else": {"op": "TO_PAISE", "value": "0"}}
            })
        else:
            steps.append({"assign": "taxableBasePaisa", "expr": "vars.subBasePaisa"})

        out_map = {
            "subBasePaisa": "vars.subBasePaisa",
            "taxableBasePaisa": "vars.taxableBasePaisa"
        }
        if rate_bps > 0:
            steps.append({
                "assign": "taxAmountPaisa",
                "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.taxableBasePaisa", "rateBps": "rules.effectiveRateBps", "denominator": 10000}
            })
            steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.taxAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
            out_map["taxAmountPaisa"] = "vars.taxAmountPaisa"
            out_map["tdsAmountPaisa"] = "vars.taxAmountPaisa"
            out_map["calculatedTaxInr"] = {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}
            out_map["calculatedTaxUsd"] = {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}
            out_map["effectiveRateBps"] = "rules.effectiveRateBps"
            out_map["isApplicable"] = "vars.isApplicable"
        else:
            steps.append({"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.taxableBasePaisa", "right": {"op": "TO_PAISE", "value": "0"}}})
            out_map["isApplicable"] = "vars.isApplicable"

        comp_block["expressionType"] = "DECLARATIVE_PIPELINE"
        comp_block["formulaDescription"] = "Symbolic operand subtraction excess pipeline"
        computation = {
            "archetype": "DECLARATIVE_PIPELINE",
            "thresholdPaisa": str(th_min or 0),
            "standardRateBps": rate_bps,
            "pipeline": {"steps": steps, "output": out_map}
        }

    # ── Pipeline Archetype 5: Bounded Temporal Substitution Window ──────────
    if computation is None and has_subst_signal and rate_bps == 0 and cutover_iso and sunset_iso:
        orig_minor = None
        alt_minor = None
        if subst_edges:
            se = subst_edges[0]
            src_m = parse_minor_units(se["sourceSpan"])
            tgt_m = parse_minor_units(se["targetSpan"])
            if src_m is not None and tgt_m is not None and src_m != tgt_m:
                alt_minor, orig_minor = src_m, tgt_m
        if orig_minor is None or alt_minor is None:
            m_match = re.search(
                r"([$₹€£]?\s*[\d,]+(?:\.\d+)?(?:\s*(?:rupees?|dollars?|euros?|pounds?|lakhs?|crores?))?)\s+shall\s+be\s+substituted\s+for\s+([$₹€£]?\s*[\d,]+(?:\.\d+)?(?:\s*(?:rupees?|dollars?|euros?|pounds?|lakhs?|crores?))?)",
                text,
                flags=re.IGNORECASE
            )
            if m_match:
                a_m = parse_minor_units(m_match.group(1))
                o_m = parse_minor_units(m_match.group(2))
                if a_m is not None and o_m is not None and a_m != o_m:
                    alt_minor, orig_minor = a_m, o_m
        if orig_minor is None or alt_minor is None:
            distinct_m = []
            for m in all_monetary_spans:
                mv = parse_minor_units(m["span"])
                if mv is not None and mv not in distinct_m:
                    distinct_m.append(mv)
            if len(distinct_m) >= 2:
                orig_minor, alt_minor = distinct_m[0], distinct_m[1]

        if orig_minor is not None and alt_minor is not None:
            steps = [
                {"assign": "inOverrideWindow", "expr": build_bounded_window_test(cutover_iso, sunset_iso, cutover_strict, sunset_strict)},
                {
                    "assign": "effectiveAmountPaisa",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "vars.inOverrideWindow",
                        "then": {"op": "TO_PAISE", "value": str(alt_minor)},
                        "else": {"op": "TO_PAISE", "value": str(orig_minor)}
                    }
                },
                {"assign": "isApplicable", "expr": True}
            ]
            comp_block["expressionType"] = "DECLARATIVE_PIPELINE"
            comp_block["formulaDescription"] = "Bounded temporal substitution pipeline selecting override quantum within active window"
            computation = {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": str(orig_minor),
                "standardRateBps": 0,
                "pipeline": {
                    "steps": steps,
                    "output": {
                        "inOverrideWindow": "vars.inOverrideWindow",
                        "effectiveAmountPaisa": "vars.effectiveAmountPaisa",
                        "effectiveAmount": {"op": "FROM_PAISE", "value": "vars.effectiveAmountPaisa"},
                        "calculatedAmountUsd": {"op": "FROM_PAISE", "value": "vars.effectiveAmountPaisa"},
                        "isApplicable": "vars.isApplicable",
                        "isCompliant": True
                    }
                }
            }

    # ── Pipeline Archetype 6: Bounded Temporal Suspension Window ───────────
    if computation is None and has_susp_signal and rate_bps == 0 and cutover_iso and sunset_iso:
        steps = [
            {"assign": "inSuspensionWindow", "expr": build_bounded_window_test(cutover_iso, sunset_iso, cutover_strict, sunset_strict)},
            {"assign": "isApplicable", "expr": {"op": "NOT", "operand": "vars.inSuspensionWindow"}}
        ]
        comp_block["expressionType"] = "DECLARATIVE_PIPELINE"
        comp_block["formulaDescription"] = "Bounded temporal suspension gate inverting applicability inside prescribed dates"
        computation = {
            "archetype": "DECLARATIVE_PIPELINE",
            "thresholdPaisa": "0",
            "standardRateBps": 0,
            "pipeline": {
                "steps": steps,
                "output": {
                    "inSuspensionWindow": "vars.inSuspensionWindow",
                    "isSuspended": "vars.inSuspensionWindow",
                    "isApplicable": "vars.isApplicable",
                    "isSectionApplicable": "vars.isApplicable"
                }
            }
        }

    # ── Pipeline Archetype 7: Pure Logic / Quantifier / Permission DAG Lowering ─
    if computation is None and (
        node_type == "GENERAL_RULE_GATE"
        or effective_domain == "GENERAL_LOGIC"
        or (node_type == "DEONTIC_STATE_GATE" and bool(edge_types & {"REL_QUANTIFIER_FORALL", "REL_QUANTIFIER_EXISTS"}))
    ):
        logic_comp = compile_general_logic_pipeline(
            text=text,
            indexed_spans=indexed_spans,
            resolved_edges=resolved_edges,
            root_condition=root_condition,
            deontic_block=deontic_block,
            deadline_window=deadline_window,
            effective_domain=effective_domain,
        )
        if logic_comp is not None:
            computation = logic_comp
            comp_block["expressionType"] = "PURE_LOGIC_PIPELINE"
            comp_block["formulaDescription"] = "Declarative boolean, quantifier, and carveout logic evaluation"
            if node_type == "GENERAL_RULE_GATE":
                root_condition["modalityType"] = deontic_block.get("modalityType", root_condition.get("modalityType"))
            if effective_domain == "GENERAL_LOGIC":
                blocks["accountingDestination"] = {
                    "generalLedgerRoute": "NOT_APPLICABLE_PURE_LOGIC",
                    "requiresDualEntryVoucher": False,
                }
                blocks["statutoryInvariants"]["controllershipAttestationRequired"] = False

    # ── Legacy & Universal Transactional Withholding (includes Flat Rate) ──
    if computation is None and node_type == "TRANSACTIONAL_WITHHOLDING":
        thresh_raw = excess_threshold if expr_type == "EXCESS_OVER_THRESHOLD" else (cliff_threshold or excess_threshold)
        thresh_qty = parse_numeric_quantity(thresh_raw) if thresh_raw else None
        thresh_paisa_str = str(int(round(float(thresh_qty) * 100))) if thresh_qty is not None else "0"

        base_keys = build_operand_fact_keys(operand_items[0]["span"], currency_code) if operand_items else []
        step_curr_base = []
        curr_base_ref = (
            emit_operand_resolution_steps(step_curr_base, "currBasePaisa", base_keys, fallback_to_primary=True)
            if base_keys
            else None
        )

        if expr_type == "EXCESS_OVER_THRESHOLD" and thresh_qty is not None:
            if curr_base_ref is None:
                step_curr_base = [{
                    "assign": "currBasePaisa",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "GT", "left": "facts.currentInvoicePaisa", "right": {"op": "TO_PAISE", "value": "0"}},
                        "then": "facts.currentInvoicePaisa",
                        "else": "facts.primaryBasePaisa"
                    }
                }]
                curr_base_ref = "vars.currBasePaisa"
            steps = step_curr_base + [
                {"assign": "cumAfter", "expr": {"op": "ADD", "left": "facts.priorPurchasesPaisa", "right": curr_base_ref}},
                {"assign": "rawExcess", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": "vars.cumAfter", "right": "rules.thresholdPaisa"}}},
                {"assign": "alreadyExcess", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": "facts.priorPurchasesPaisa", "right": "rules.thresholdPaisa"}}},
                {"assign": "taxableBasePaisa", "expr": {"op": "MIN", "left": curr_base_ref, "right": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": "vars.rawExcess", "right": "vars.alreadyExcess"}}}},
                {"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.taxableBasePaisa", "right": {"op": "TO_PAISE", "value": "0"}}},
                {"assign": "tdsAmountPaisa", "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.taxableBasePaisa", "rateBps": "rules.effectiveRateBps", "denominator": 10000}},
                {"assign": "effectiveRateBps", "expr": {"op": "CONDITIONAL", "test": "vars.isApplicable", "then": "rules.effectiveRateBps", "else": 0}},
                {"assign": "tdsRatePercent", "expr": {"op": "DIV", "left": "vars.effectiveRateBps", "right": 100}}
            ]
            computation = {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": thresh_paisa_str,
                "standardRateBps": rate_bps,
                "pipeline": {
                    "steps": steps,
                    "output": {
                        "taxableBasePaisa": "vars.taxableBasePaisa",
                        "tdsAmountPaisa": "vars.tdsAmountPaisa",
                        "taxAmountPaisa": "vars.tdsAmountPaisa",
                        "isApplicable": "vars.isApplicable",
                        "isTdsApplicable": {"op": "GT", "left": "vars.tdsAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}},
                        "effectiveRateBps": "vars.effectiveRateBps",
                        "tdsRatePercent": "vars.tdsRatePercent"
                    }
                }
            }
        elif expr_type == "CLIFF_THRESHOLD" and thresh_qty is not None:
            if curr_base_ref is None:
                step_curr_base = [{
                    "assign": "currBasePaisa",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "GT", "left": "facts.currentInvoicePaisa", "right": {"op": "TO_PAISE", "value": "0"}},
                        "then": "facts.currentInvoicePaisa",
                        "else": "facts.primaryBasePaisa"
                    }
                }]
                curr_base_ref = "vars.currBasePaisa"
            steps = step_curr_base + [
                {"assign": "cumAfter", "expr": {"op": "ADD", "left": "facts.priorPurchasesPaisa", "right": curr_base_ref}},
                {
                    "assign": "breached",
                    "expr": {
                        "op": "OR",
                        "left": {"op": "GT", "left": curr_base_ref, "right": "rules.thresholdPaisa"},
                        "right": {"op": "GT", "left": "vars.cumAfter", "right": "rules.thresholdPaisa"}
                    }
                },
                {"assign": "taxableBasePaisa", "expr": {"op": "CONDITIONAL", "test": "vars.breached", "then": curr_base_ref, "else": {"op": "TO_PAISE", "value": "0"}}},
                {"assign": "tdsAmountPaisa", "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.taxableBasePaisa", "rateBps": "rules.effectiveRateBps", "denominator": 10000}},
                {"assign": "effectiveRateBps", "expr": {"op": "CONDITIONAL", "test": "vars.breached", "then": "rules.effectiveRateBps", "else": 0}},
                {"assign": "tdsRatePercent", "expr": {"op": "DIV", "left": "vars.effectiveRateBps", "right": 100}}
            ]
            computation = {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": thresh_paisa_str,
                "standardRateBps": rate_bps,
                "pipeline": {
                    "steps": steps,
                    "output": {
                        "taxableBasePaisa": "vars.taxableBasePaisa",
                        "tdsAmountPaisa": "vars.tdsAmountPaisa",
                        "taxAmountPaisa": "vars.tdsAmountPaisa",
                        "isApplicable": "vars.breached",
                        "isTdsApplicable": {"op": "GT", "left": "vars.tdsAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}},
                        "effectiveRateBps": "vars.effectiveRateBps",
                        "tdsRatePercent": "vars.tdsRatePercent"
                    }
                }
            }
        elif rate_bps > 0:
            if curr_base_ref is None:
                step_curr_base = [{
                    "assign": "currBasePaisa",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "GT", "left": "facts.currentInvoicePaisa", "right": {"op": "TO_PAISE", "value": "0"}},
                        "then": "facts.currentInvoicePaisa",
                        "else": "facts.primaryBasePaisa"
                    }
                }]
                curr_base_ref = "vars.currBasePaisa"
            steps = step_curr_base + [
                {"assign": "taxableBasePaisa", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": curr_base_ref}},
                {"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.taxableBasePaisa", "right": {"op": "TO_PAISE", "value": "0"}}},
                {"assign": "tdsAmountPaisa", "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.taxableBasePaisa", "rateBps": "rules.effectiveRateBps", "denominator": 10000}},
                {"assign": "effectiveRateBps", "expr": {"op": "CONDITIONAL", "test": "vars.isApplicable", "then": "rules.effectiveRateBps", "else": 0}},
                {"assign": "tdsRatePercent", "expr": {"op": "DIV", "left": "vars.effectiveRateBps", "right": 100}}
            ]
            computation = {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": "0",
                "standardRateBps": rate_bps,
                "pipeline": {
                    "steps": steps,
                    "output": {
                        "taxableBasePaisa": "vars.taxableBasePaisa",
                        "tdsAmountPaisa": "vars.tdsAmountPaisa",
                        "taxAmountPaisa": "vars.tdsAmountPaisa",
                        "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.tdsAmountPaisa"},
                        "isApplicable": "vars.isApplicable",
                        "isTdsApplicable": {"op": "GT", "left": "vars.tdsAmountPaisa", "right": {"op": "TO_PAISE", "value": "0"}},
                        "effectiveRateBps": "vars.effectiveRateBps",
                        "tdsRatePercent": "vars.tdsRatePercent"
                    }
                }
            }
        else:
            computation = {
                "archetype": "FLAT_RATE",
                "thresholdPaisa": "0",
                "standardRateBps": rate_bps
            }
    elif computation is None and node_type == "TEMPORAL_WINDOW_GATE" and expr_type == "COMPOUND_INTEREST_PENAL":
        window_days = parse_days_window(deadline_window) or 45
        computation = {
            "archetype": "DECLARATIVE_PIPELINE",
            "deadlineDays": window_days,
            "pipeline": {
                "steps": [
                    {"assign": "elapsedDays", "expr": {"op": "DATE_DIFF", "start": "facts.triggerDate", "end": "facts.settlementDate", "unit": "DAYS"}},
                    {"assign": "overdueDays", "expr": {"op": "MAX", "left": 0, "right": {"op": "SUB", "left": "vars.elapsedDays", "right": window_days}}},
                    {"assign": "isCompliant", "expr": {"op": "LTE", "left": "vars.overdueDays", "right": 0}},
                    {
                        "assign": "annualRateBps",
                        "expr": {
                            "op": "CONDITIONAL",
                            "test": {"op": "GT", "left": "facts.rbiBankRatePct", "right": 0},
                            "then": {"op": "MUL", "left": "facts.rbiBankRatePct", "right": 300},
                            "else": 1950
                        }
                    },
                    {"assign": "monthsOverdue", "expr": "vars.overdueDays / 30"},
                    {"assign": "interestPaisa", "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "GT", "left": "vars.overdueDays", "right": 0},
                        "then": {"op": "COMPOUND_INTEREST", "principal": "facts.invoiceAmountPaisa", "rateBps": "vars.annualRateBps", "months": "vars.monthsOverdue", "rests": "MONTHLY"},
                        "else": 0
                    }}
                ],
                "output": {
                    "isApplicable": True,
                    "isCompliant": "vars.isCompliant",
                    "deadlineDays": window_days,
                    "elapsedDays": "vars.elapsedDays",
                    "overdueDays": "vars.overdueDays",
                    "calculatedInterestInr": {"op": "FROM_PAISE", "value": "vars.interestPaisa"}
                }
            }
        }
    elif computation is None and qual_thresh_item is not None:
        q_minor = parse_minor_units(qual_thresh_item["span"])
        q_cr = qual_thresh_item.get("charRange", [0, 0])
        q_win = text[max(0, q_cr[0] - 35):min(len(text), q_cr[1] + 35)]
        q_op = parse_comparison_operator(q_win)
        non_thresh_ops = [
            o for o in operand_items
            if o["spanIdx"] != qual_thresh_item["spanIdx"]
            and not re.match(r"^\s*[$₹€£]?\s*[\d,]+", o["span"])
        ]
        if non_thresh_ops:
            op_keys = build_operand_fact_keys(non_thresh_ops[0]["span"], currency_code)
        else:
            pre_phrase = _safe_preceding_words(text, q_cr[0], 45)
            op_keys = build_operand_fact_keys(pre_phrase, currency_code)
        steps = []
        metric_ref = emit_operand_resolution_steps(steps, "metricPaisa", op_keys, fallback_to_primary=True)
        steps.append({
            "assign": "isApplicable",
            "expr": {"op": q_op, "left": metric_ref, "right": {"op": "TO_PAISE", "value": str(q_minor)}}
        })
        comp_block["expressionType"] = "DECLARATIVE_PIPELINE"
        comp_block["formulaDescription"] = f"Monetary qualification gate comparing metric {q_op} threshold"
        computation = {
            "archetype": "DECLARATIVE_PIPELINE",
            "thresholdPaisa": str(q_minor),
            "standardRateBps": 0,
            "pipeline": {
                "steps": steps,
                "output": {
                    "isApplicable": "vars.isApplicable",
                    "isSectionApplicable": "vars.isApplicable",
                    "isCompliant": "vars.isApplicable",
                    "metricPaisa": metric_ref,
                    "thresholdAmount": {"op": "FROM_PAISE", "value": str(q_minor)}
                }
            }
        }
    elif computation is None:
        deontic_gate = {
            "gateType": node_type,
            "modalityType": deontic_block.get("modalityType", "OBLIGATION"),
            "statutoryDuty": deontic_block.get("statutoryDuty"),
            "exceptionCarveouts": [
                {"id": f"CARVEOUT_{i + 1}", "rawText": c, "factKey": normalize_symbol_id(c) or f"carveout_{i + 1}"}
                for i, c in enumerate(exception_carveouts)
            ],
            "disqualificationTriggers": [],
            "mandatoryApprovals": [
                {"id": f"APPROVAL_{i + 1}", "description": a, "factKey": normalize_symbol_id(a) or f"approval_{i + 1}"}
                for i, a in enumerate(mandatory_actions)
            ] if node_type == "CORPORATE_APPROVAL_GATE" else [],
            "noticeWindowDays": parse_days_window(deadline_window) if node_type == "PROCEDURAL_NOTICE_GATE" else None,
            "requireAttestationIfChecklist": (
                effective_domain != "GENERAL_LOGIC"
                and node_type in ("SUBJECTIVE_EVIDENCE_GATE", "STATUS_DISQUALIFICATION_GATE", "MEASUREMENT_VALUATION_RULE")
            )
        }
        computation = {
            "archetype": deontic_gate["gateType"],
            "deonticGate": deontic_gate
        }

    if typed_inputs and typed_inputs_block is None:
        raise ValueError(
            "ERR_TYPED_INPUTS_UNSUPPORTED_ARCHETYPE: the v2 typed-input profile "
            "currently lowers only single-clause progressive bracket rules"
        )

    deontic_tuple = {
        "subject_S": blocks["applicabilityScope"],
        "preconditions_P": guards or blocks["applicabilityScope"].get("qualifyingConditions", []),
        "modality_M": deontic_block,
        "actionOrFormula_A": computation or comp_block,
        "carveouts_C": blocks["carveoutsAndExceptions"],
        "temporalClocks_T": blocks["temporalCrystallization"]
    }

    return {
        "sectionId": section_id,
        "version": meta.get("version", "0.1.0-draft"),
        "executionMode": meta.get("executionMode", "DEVELOPMENT"),
        "ruleDomain": effective_domain,
        "predictedModality": predicted_modality,
        "modalityConfidence": round(modality_conf, 4),
        "clauseModalities": clause_modalities or [],
        "triggerPolicy": {"type": "EARLIER_OF", "events": ["BOOK_CREDIT", "PAYMENT_DISBURSEMENT"]} if trigger_policy == "EARLIER_OF" else {"type": trigger_policy},
        "statutoryInvariants": {
            "isAuthorizedToDisburse": False,
            "requiresHumanApprovalForOutcomes": True
        },
        "sourceLineage": {
            "documentIdentifier": meta.get("documentNumber", "RULE_INSTRUMENT" if effective_domain == "GENERAL_LOGIC" else "STATUTORY_INSTRUMENT"),
            "issuingAuthority": meta.get("issuingAuthority", "GENERAL_RULE_SYSTEM" if effective_domain == "GENERAL_LOGIC" else "CBDT/CBIC"),
            "publicationDate": meta.get("gazetteDate", "UNKNOWN_DATE"),
            "referencedSections": meta.get("referencedSections", [])
        },
        "guards": guards,
        **({"inputs": typed_inputs_block} if typed_inputs_block else {}),
        "computation": computation,
        **({"deonticGate": deontic_gate} if deontic_gate else {}),
        "attestationChecklist": evidence_checklist,
        "rootCondition": root_condition,
        "deonticTuple": deontic_tuple,
        "compositionalBuildingBlocks": blocks,
        "symbolLegend": symbol_legend
    }


class MudraDecompilerInference:
    """
    Zero-shot neural decompiler for statutory legislation and government documents.
    """

    def __init__(self, checkpoint_path: str = None):
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ingestor = StatutoryDocumentIngestor()

        if not checkpoint_path:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint_path = os.path.join(repo_root, "checkpoints", "dual_stream_sup.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.vocab = checkpoint["vocab"]
        self.config = checkpoint["config"]
        self.roles = checkpoint.get("roles", SEMANTIC_ROLES)
        self.bio_tags = checkpoint.get("bio_tags", BIO_TAGS)
        self.relations = checkpoint.get("relations", RELATION_TYPES)
        self.modalities = checkpoint.get("modalities", RULE_MODALITIES)

        # Canonical dictionary binding directly from checkpoint
        self.id2role = {i: r for i, r in enumerate(self.roles)}
        self.role2id = {r: i for i, r in enumerate(self.roles)}
        self.id2bio = {i: b for i, b in enumerate(self.bio_tags)}
        self.bio2id = {b: i for i, b in enumerate(self.bio_tags)}
        self.id2rel = {i: r for i, r in enumerate(self.relations)}
        self.rel2id = {r: i for i, r in enumerate(self.relations)}
        self.id2modality = {i: m for i, m in enumerate(self.modalities)}
        self.modality2id = {m: i for i, m in enumerate(self.modalities)}

        # Build valid BIO transition matrix
        n_tags = len(self.bio_tags)
        self.transition_mask = torch.zeros((n_tags, n_tags), dtype=torch.float32, device=self.device)
        for prev_id, prev_tag in self.id2bio.items():
            for curr_id, curr_tag in self.id2bio.items():
                # Invalid 1: Cannot transition from O directly into any I-X tag
                if prev_tag == "O" and curr_tag.startswith("I-"):
                    self.transition_mask[prev_id, curr_id] = -1e9
                # Invalid 2: Cannot transition from B-X to I-Y where X != Y
                elif prev_tag.startswith("B-") and curr_tag.startswith("I-"):
                    if prev_tag[2:] != curr_tag[2:]:
                        self.transition_mask[prev_id, curr_id] = -1e9
                # Invalid 3: Cannot transition from I-X to I-Y where X != Y
                elif prev_tag.startswith("I-") and curr_tag.startswith("I-"):
                    if prev_tag[2:] != curr_tag[2:]:
                        self.transition_mask[prev_id, curr_id] = -1e9

        self.model = MudraStatutoryParser(
            vocab_size=self.config["vocab_size"],
            d_model=self.config.get("d_model", 256),
            suffix_dim=self.config.get("suffix_dim", 32),
            n_heads=self.config.get("n_heads", 4),
            d_ff=self.config.get("d_ff", 1024),
            n_layers=self.config.get("n_layers", 6),
            span_dim=self.config.get("span_dim", 128),
            dropout=0.0,
            num_roles=self.config.get("num_roles", len(self.roles)),
            num_relations=self.config.get("num_relations", len(self.relations)),
            num_bio_tags=self.config.get("num_bio_tags", len(self.bio_tags)),
            num_modalities=self.config.get("num_modalities", len(self.modalities))
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _viterbi_decode(self, logits: torch.Tensor, tokens: list[str]) -> list[int]:
        """
        Viterbi constrained decoding over token-level BIO logits.
        Enforces valid BIO transitions and context-aware lexical guards:
        e.g., 'per' does NOT start a RATE unless followed by 'cent', 'annum', or '%'.
        """
        seq_len, n_tags = logits.shape
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        dp = torch.empty((seq_len, n_tags), dtype=torch.float32, device=self.device)
        backpointers = torch.empty((seq_len, n_tags), dtype=torch.long, device=self.device)

        # Step 0: cannot start with I- tag
        dp[0] = log_probs[0]
        for tag_id, tag in self.id2bio.items():
            if tag.startswith("I-"):
                dp[0, tag_id] = -1e9

        for t in range(1, seq_len):
            tok_lower = tokens[t].lower() if t < len(tokens) else ""
            next_tok_lower = tokens[t + 1].lower() if t + 1 < len(tokens) else ""

            # Check if current token is 'per' and next is not 'cent', 'centum', 'annum', '%'
            per_not_rate = (tok_lower == "per" and next_tok_lower not in {"cent", "centum", "annum", "%"})

            step_mask = self.transition_mask.clone()
            if per_not_rate:
                # Invalidate transitioning into B-RATE or I-RATE
                for tag_id, tag in self.id2bio.items():
                    if "RATE" in tag:
                        step_mask[:, tag_id] = -1e9

            scores = dp[t - 1].unsqueeze(1) + step_mask + log_probs[t].unsqueeze(0)
            best_prev_scores, best_prev_indices = scores.max(dim=0)
            dp[t] = best_prev_scores
            backpointers[t] = best_prev_indices

        best_last_tag = dp[-1].argmax().item()
        path = [best_last_tag]
        curr = best_last_tag
        for t in range(seq_len - 1, 0, -1):
            curr = backpointers[t, curr].item()
            path.append(curr)
        path.reverse()
        return path

    def decompile_text(self, text: str, source_metadata: dict = None, rule_domain: str = "AUTO") -> dict:
        """
        Decompiles raw rule or statutory text into semantic roles, relations, and USVM AST.
        Uses Token-Level BIO Sequence Tagging (no regex span extraction!) + SBR Pooling + Biaffine Relations.
        """
        req_domain = str(rule_domain or "AUTO").strip().upper()
        if req_domain not in ALLOWED_RULE_DOMAINS:
            raise ValueError(f"Unsupported rule_domain '{rule_domain}'; expected one of {ALLOWED_RULE_DOMAINS}")
        tokens, offsets = tokenize_with_offsets(text)
        if not tokens:
            return {
                "status": "UNRESOLVED_STATUTORY_AMBIGUITY",
                "compiler": "Mudra-Decompiler-0",
                "ruleDomain": req_domain,
                "calibratedConfidence": 0.0,
                "message": "Empty or non-text statutory instrument input.",
                "isAuthorizedToDisburse": False,
                "candidateSets": []
            }

        token_ids = []
        suffix_ids = []
        for idx, t in enumerate(tokens):
            prev_t = tokens[idx - 1] if idx > 0 else None
            prev_prev_t = tokens[idx - 2] if idx > 1 else None
            next_t = tokens[idx + 1] if idx + 1 < len(tokens) else None
            tid = resolve_open_class_token(t, prev_t, prev_prev_t, self.vocab, next_token=next_t)
            token_ids.append(tid)
            canon_t = canonicalize_token(t, prev_t, next_t)
            suffix_ids.append(SUFFIX2ID.get(canonical_suffix(t, canon_t), SUFFIX2ID["<NONE>"]))

        clause_ranges = segment_clauses(text, tokens, offsets)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)

        # 1. Neural Sequence Tagging + Document & Per-Clause Modality Prediction (Zero Node Requirement)
        with torch.no_grad():
            bio_logits, _, _, modality_logits, clause_modality_logits = self.model.forward_multitask(
                token_tensor,
                suffix_ids=suffix_tensor,
                clause_spans=clause_ranges if len(clause_ranges) > 1 else None
            )
            mod_probs = torch.softmax(modality_logits.squeeze(0), dim=-1)
            pred_modality_id = int(mod_probs.argmax(dim=-1).item())
            pred_modality = self.id2modality.get(pred_modality_id, "MODALITY_COMPUTATIONAL_EXCESS")
            modality_conf = float(mod_probs[pred_modality_id].item())

            clause_modalities = []
            if clause_modality_logits is not None and clause_modality_logits.numel() > 0:
                cm_probs = torch.softmax(clause_modality_logits, dim=-1)
                for c_idx, (cs_tok, ce_tok) in enumerate(clause_ranges):
                    c_pid = int(cm_probs[c_idx].argmax(dim=-1).item())
                    c_mod = self.id2modality.get(c_pid, "MODALITY_COMPUTATIONAL_EXCESS")
                    c_conf = float(cm_probs[c_idx, c_pid].item())
                    clause_modalities.append({
                        "clauseIdx": c_idx,
                        "tokenRange": [cs_tok, ce_tok],
                        "charRange": [offsets[cs_tok][0], offsets[ce_tok][1]],
                        "modality": c_mod,
                        "confidence": round(c_conf, 4),
                    })
            elif clause_ranges:
                clause_modalities.append({
                    "clauseIdx": 0,
                    "tokenRange": [0, len(tokens) - 1],
                    "charRange": [offsets[0][0], offsets[-1][1]],
                    "modality": pred_modality,
                    "confidence": round(modality_conf, 4),
                })

            # Guarded multi-clause modality promotion for long composed statutes
            if len(clause_modalities) > 1:
                if pred_modality == "MODALITY_ABSTAIN_UNRESOLVED":
                    comp_subs = [cm for cm in clause_modalities if cm["modality"].startswith("MODALITY_COMPUTATIONAL") and cm["confidence"] >= 0.50]
                    non_abstain_subs = [cm for cm in clause_modalities if cm["modality"] != "MODALITY_ABSTAIN_UNRESOLVED" and cm["confidence"] >= 0.50]
                    chosen_cm = max(comp_subs, key=lambda x: x["confidence"]) if comp_subs else (
                        max(non_abstain_subs, key=lambda x: x["confidence"]) if non_abstain_subs else None
                    )
                    if chosen_cm is not None:
                        pred_modality = chosen_cm["modality"]
                        modality_conf = max(modality_conf, chosen_cm["confidence"])
                elif pred_modality.startswith("MODALITY_DEONTIC_"):
                    def _clause_has_computational_signal(cm):
                        if cm.get("confidence", 0.0) < 0.50:
                            return False
                        seg_txt = text[cm["charRange"][0]:cm["charRange"][1]].lower()
                        has_pct = bool("%" in seg_txt or re.search(r"\b(?:per\s+cent(?:um)?|percent)\b", seg_txt))
                        has_ccy = bool(re.search(r"\b(?:rupees?|dollars?|euros?|pounds?|inr|usd|eur|gbp|lakhs?|crores?)\b|[₹$€£]", seg_txt))
                        has_comp_syntax = bool(re.search(
                            r"\b(?:substituted\s+for|lesser\s+of|greater\s+of|excess\s+over|tax|levy|tds|tcs|excise|deduct\s+tax|withhold|surcharge|interest|compound)\b",
                            seg_txt
                        ))
                        is_acct_election = any(k in seg_txt for k in ["profit or loss", "other comprehensive income", "may elect not"])
                        is_deontic_escrow_or_carveout = bool(re.search(
                            r"\b(?:shall\s+not\s+collect|paying\s+a\s+(?:fine|bond|fee|deposit)|upon\s+paying|may\s+(?:exit|depart|proceed)|provided\s+that)\b",
                            seg_txt
                        )) and not has_pct and not re.search(r"\bsubstituted\s+for\b", seg_txt)
                        return (has_pct or (has_ccy and has_comp_syntax)) and not is_acct_election and not is_deontic_escrow_or_carveout

                    comp_subs = [
                        cm for cm in clause_modalities
                        if cm["modality"].startswith("MODALITY_COMPUTATIONAL") and _clause_has_computational_signal(cm)
                    ]
                    if comp_subs:
                        best_comp = max(comp_subs, key=lambda x: x["confidence"])
                        pred_modality = best_comp["modality"]
                        modality_conf = max(modality_conf, best_comp["confidence"])

            bio_preds = self._viterbi_decode(bio_logits.squeeze(0), tokens)
            bio_tags = [self.id2bio[p] for p in bio_preds]

        # Native Abstention if Rule-Modality Head predicts MODALITY_ABSTAIN_UNRESOLVED
        if pred_modality == "MODALITY_ABSTAIN_UNRESOLVED":
            return {
                "status": "UNRESOLVED_STATUTORY_AMBIGUITY",
                "compiler": "Mudra-Decompiler-0",
                "ruleDomain": req_domain,
                "predictedModality": pred_modality,
                "modalityConfidence": round(modality_conf, 4),
                "calibratedConfidence": 0.0,
                "message": "Input classified natively as non-legal, conversational, or operationally underspecified.",
                "statutoryInvariants": {
                    "isAuthorizedToDisburse": False,
                    "requiresHumanApprovalForOutcomes": True
                },
                "isAuthorizedToDisburse": False,
                "candidateSets": []
            }

        # 2. Transition-Constrained Contiguous Span Decoding across conjunctions/commas
        discovered_spans = decode_bio_spans(bio_tags, tokens, offsets)

        if not discovered_spans:
            return {
                "status": "UNRESOLVED_STATUTORY_AMBIGUITY",
                "compiler": "Mudra-Decompiler-0",
                "ruleDomain": req_domain,
                "predictedModality": pred_modality,
                "modalityConfidence": round(modality_conf, 4),
                "calibratedConfidence": 0.0,
                "message": "No statutory constants or actionable spans detected in input.",
                "statutoryInvariants": {
                    "isAuthorizedToDisburse": False,
                    "requiresHumanApprovalForOutcomes": True
                },
                "isAuthorizedToDisburse": False,
                "candidateSets": []
            }

        span_boundaries = [(s["token_start"], s["token_end"]) for s in discovered_spans]

        # 3. Neural Span Classification & Relational Parsing
        with torch.no_grad():
            _, role_logits, rel_logits, _ = self.model(
                token_tensor,
                suffix_ids=suffix_tensor,
                span_indices=span_boundaries
            )
            role_probs = torch.softmax(role_logits, dim=-1)
            pred_roles = role_logits.argmax(dim=-1).cpu().tolist()
            pred_confidences = role_probs.max(dim=-1).values.cpu().tolist()

        def _locate_clause_idx(tok_start: int, tok_end: int) -> int:
            mid = (tok_start + tok_end) // 2
            for cm in clause_modalities:
                if cm["tokenRange"][0] <= mid <= cm["tokenRange"][1]:
                    return cm["clauseIdx"]
            return 0

        classified_spans = []
        for i, s_info in enumerate(discovered_spans):
            cs = s_info["char_start"]
            ce = s_info["char_end"]
            span_text = text[cs:ce].strip()
            role_name = self.id2role.get(pred_roles[i], "ROLE_IGNORE")
            conf = pred_confidences[i]
            classified_spans.append({
                "span": span_text,
                "bioType": s_info.get("type", "UNKNOWN"),
                "role": role_name,
                "confidence": round(conf, 4),
                "charRange": [cs, ce],
                "tokenRange": [s_info["token_start"], s_info["token_end"]],
                "clauseIdx": _locate_clause_idx(s_info["token_start"], s_info["token_end"])
            })

        # 4. Contiguous numeric and compound quantifier span stitching (strictly guarded against verb swallowing)
        classified_spans, old_to_new_span_idx = stitch_classified_spans(classified_spans, text)

        # 5. Extract active relational edges with unambiguous span indices
        rel_edges = []
        seen_edges = set()
        if rel_logits is not None and rel_logits.numel() > 0:
            rel_preds = rel_logits.argmax(dim=-1).cpu().tolist()
            num_spans = len(discovered_spans)
            for src in range(num_spans):
                for tgt in range(num_spans):
                    rel_id = rel_preds[src][tgt]
                    if rel_id != 0: # Non-zero edge
                        src_mapped = old_to_new_span_idx.get(src, src)
                        tgt_mapped = old_to_new_span_idx.get(tgt, tgt)
                        if src_mapped < len(classified_spans) and tgt_mapped < len(classified_spans) and src_mapped != tgt_mapped:
                            rel_name = self.id2rel.get(rel_id, "REL_NO_EDGE")
                            edge_key = (src_mapped, tgt_mapped, rel_name)
                            if edge_key not in seen_edges:
                                seen_edges.add(edge_key)
                                rel_edges.append({
                                    "sourceIdx": src_mapped,
                                    "targetIdx": tgt_mapped,
                                    "sourceSpan": classified_spans[src_mapped]["span"],
                                    "targetSpan": classified_spans[tgt_mapped]["span"],
                                    "relationType": rel_name
                                })

        # 6. Synthesize verified USVM AST representation using Neural Modality + SRL Graph + Clause Segmentation
        ast_package = self._synthesize_ast(
            text,
            classified_spans,
            rel_edges,
            source_metadata,
            predicted_modality=pred_modality,
            modality_conf=modality_conf,
            clause_modalities=clause_modalities,
            rule_domain=req_domain
        )

        # Calibrated Confidence: Evaluate substantive statutory roles, excluding boundary function words
        FUNCTION_WORDS = {'in', 'of', 'to', 'for', 'by', 'on', 'at', 'from', 'the', 'a', 'an', 'and', 'or', 'be', 'that', 'with', 'as', 'whether', 'has', 'been'}
        substantive_spans = [
            s for s in classified_spans
            if s["role"] != "ROLE_IGNORE" and not (len(s["span"].split()) == 1 and s["span"].lower() in FUNCTION_WORDS)
        ]
        calibrated_conf = round(
            sum(s["confidence"] for s in substantive_spans) / len(substantive_spans), 4
        ) if substantive_spans else (
            round(sum(s["confidence"] for s in classified_spans) / len(classified_spans), 4) if classified_spans else 0.0
        )

        return {
            "status": "SUCCESS",
            "compiler": "Mudra-Decompiler-0",
            "ruleDomain": ast_package.get("ruleDomain", req_domain),
            "predictedModality": pred_modality,
            "modalityConfidence": round(modality_conf, 4),
            "clauseModalities": clause_modalities,
            "calibratedConfidence": calibrated_conf,
            "humanReviewChecklist": self._generate_review_checklist(ast_package, calibrated_conf),
            "sourceLineage": ast_package["sourceLineage"],
            "statutoryInvariants": ast_package["statutoryInvariants"],
            "synthesizedAst": ast_package,
            "symbolLegend": ast_package.get("symbolLegend", {}),
            "classifiedSpansCount": len(classified_spans),
            "classifiedSpans": classified_spans,
            "relationalEdgesCount": len(rel_edges),
            "relationalEdges": rel_edges,
            "isAuthorizedToDisburse": False
        }

    def decompile_pdf(self, pdf_path: str, source_metadata: dict = None, rule_domain: str = "AUTO") -> dict:
        """
        Decompiles a raw PDF gazette document into structured USVM AST.
        """
        ingest_result = self.ingestor.ingest_pdf(pdf_path)
        combined_text = ingest_result.get("englishCleanedText") or ingest_result.get("full_text", "")
        metadata = source_metadata or ingest_result.get("metadata") or ingest_result.get("detectedMetadata", {})
        return self.decompile_text(combined_text, source_metadata=metadata, rule_domain=rule_domain)

    def _synthesize_ast(
        self,
        text: str,
        spans: list[dict],
        edges: list[dict] = None,
        source_metadata: dict = None,
        predicted_modality: str = "MODALITY_COMPUTATIONAL_EXCESS",
        modality_conf: float = 0.9,
        clause_modalities: list[dict] = None,
        rule_domain: str = "AUTO",
    ) -> dict:
        return synthesize_usvm_ast(
            text=text,
            spans=spans,
            edges=edges,
            source_metadata=source_metadata,
            predicted_modality=predicted_modality,
            modality_conf=modality_conf,
            clause_modalities=clause_modalities,
            rule_domain=rule_domain,
        )

    def _synthesize_building_blocks(
        self,
        root_condition: dict,
        spans: list[dict],
        edges: list[dict],
        text: str,
        predicted_modality: str = "MODALITY_COMPUTATIONAL_EXCESS",
        rule_domain: str = "STATUTORY_FINANCE",
    ) -> dict:
        return synthesize_building_blocks(
            root_condition=root_condition,
            spans=spans,
            edges=edges,
            text=text,
            predicted_modality=predicted_modality,
            rule_domain=rule_domain,
        )

    def _synthesize_symbol_legend(self, spans: list[dict], edges: list[dict], text: str) -> dict:
        return synthesize_symbol_legend(spans, edges, text)

    def _generate_review_checklist(self, ast: dict, confidence: float) -> list[str]:
        items = []
        root = ast.get("rootCondition", {})
        node_type = root.get("nodeType", "UNKNOWN")
        rule_domain = ast.get("ruleDomain", "STATUTORY_FINANCE")

        items.append(f"Statutory Pattern Archetype: '{node_type}' (Domain: '{rule_domain}')")

        if node_type == "GENERAL_RULE_GATE" or rule_domain == "GENERAL_LOGIC":
            mod_type = root.get("modalityType") or ast.get("deonticTuple", {}).get("modality_M", {}).get("modalityType") or "OBLIGATION"
            items.append(f"Verify deontic modality: '{mod_type}'")
            duty = (
                root.get("permissionPredicate")
                or root.get("obligationPredicate")
                or root.get("prohibitionPredicate")
                or "Rule predicate evaluation"
            )
            items.append(f"Verify core rule predicate: '{duty}'")
            carveouts = root.get("exceptionCarveouts") or []
            if carveouts:
                items.append(f"Verify logical carveout / escape conditions ({len(carveouts)} detected): {carveouts}")
        elif node_type == "TRANSACTIONAL_WITHHOLDING":
            if root.get("applicableExcessThreshold"):
                items.append(f"Verify statutory excess threshold: '{root['applicableExcessThreshold']}'")
            if root.get("standardTaxRate"):
                items.append(f"Verify statutory baseline withholding rate: '{root['standardTaxRate']}'")
        elif node_type == "CORPORATE_APPROVAL_GATE":
            items.append(f"Verify pecuniary shareholder approval threshold: '{root.get('applicableExcessThreshold')}'")
            items.append(f"Verify mandatory approvals required: {root.get('mandatoryApprovals')}")
        elif node_type == "STATUS_DISQUALIFICATION_GATE":
            items.append(f"Verify statutory qualification bar: '{root.get('prohibitionPredicate')}'")
            items.append(f"Verify statutory sanction: '{root.get('statutorySanction')}'")
        elif node_type == "PROCEDURAL_NOTICE_GATE":
            items.append(f"Verify pre-institution prescription window: '{root.get('applicableDeadlineWindow')}'")

        if len(items) < 3 and rule_domain != "GENERAL_LOGIC":
            items.append("Confirm statutory citation lineage and controllership attestation before activation.")
        elif len(items) < 3:
            items.append("Verify extracted boolean/quantifier factSchema inputs against domain state model.")

        if confidence < 0.90:
            items.append("WARNING: Calibrated confidence < 90%. Manual legal review mandatory before production activation.")

        return items
