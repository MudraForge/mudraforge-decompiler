"""Pure deterministic literal, unit, date, and clause-segmentation helpers.

Shared by v1 inference (`inference.py`) and the regex-free v2 AST lowerer
(`inference_v2.py`). Zero ML model dependencies and zero domain-specific legal
hardcoding.
"""

from __future__ import annotations

import re
from datetime import datetime


WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
    "crore": 10000000, "crores": 10000000, "million": 1000000, "billion": 1000000000,
}


FRACTION_DENOMINATORS = {
    "half", "halves", "third", "thirds", "fourth", "fourths", "fifth", "fifths",
    "sixth", "sixths", "eighth", "eighths", "tenth", "tenths",
}


def parse_numeric_quantity(text: str):
    if not text:
        return None
    cleaned = text.lower().replace(",", "")
    cleaned_no_pct = re.sub(r"\bper\s+cent(?:um)?\b", " ", cleaned)
    has_major_currency = bool(
        re.search(
            r"₹|\$|€|£|¥|\b(?:dollars?|rupees?|euros?|pounds?|yen|dirhams?|francs?|usd|inr|eur|gbp|sgd|aed|chf|cad|aud|jpy)\b",
            cleaned_no_pct,
        )
    )
    is_standalone_minor = (
        bool(re.search(r"\b(?:cents?|paise|paisa|pence)\b", cleaned_no_pct))
        and not has_major_currency
    )

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
    """Decides whether two adjacent classified spans form one compound quantity."""
    gap_str = text[curr["charRange"][1]:nxt["charRange"][0]]
    if "," in gap_str:
        return False
    gap_norm = gap_str.strip().lower()
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
    if (
        qty_curr is not None
        and sym_curr == "BOOLEAN_FLAG"
        and qty_nxt is None
        and sym_nxt in ("MONETARY_QUANTUM", "PERCENTAGE_RATE", "TEMPORAL_DURATION")
    ):
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
    """Greedily merges adjacent spans that form one compound quantity."""
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
            merged_entry = {
                "span": comb_text,
                "bioType": curr["bioType"],
                "role": dominant_role,
                "confidence": round((curr["confidence"] + nxt["confidence"]) / 2, 4),
                "charRange": [curr["charRange"][0], nxt["charRange"][1]],
                "tokenRange": [curr["tokenRange"][0], nxt["tokenRange"][1]],
                "clauseIdx": curr["clauseIdx"],
            }
            if "unit" in curr:
                merged_entry["unit"] = curr["unit"]
            if "unitConfidence" in curr:
                merged_entry["unitConfidence"] = curr["unitConfidence"]
            curr = merged_entry
            consumed += 1
        curr["spanIdx"] = new_idx
        stitched.append(curr)
        for k in range(idx, idx + consumed):
            old_to_new[k] = new_idx
        idx += consumed
    return stitched, old_to_new


def parse_unit_and_type(text: str):
    low = text.lower()
    low_no_pct = re.sub(r"\bper\s+cent(?:um)?\b", " _pct_ ", low)
    if re.search(r"₹|\b(?:rupees?|rs\.?|inr|crores?|lakhs?|lacs?|paisa|paise)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_PAISA", "INR"
    if re.search(r"\b(?:sgd|singapore\s+dollars?|s\$)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "SGD"
    if re.search(r"\b(?:cad|canadian\s+dollars?|c\$)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "CAD"
    if re.search(r"\b(?:aud|australian\s+dollars?|a\$)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "AUD"
    if re.search(r"\b(?:aed|dirhams?)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "AED"
    if re.search(r"\b(?:chf|swiss\s+francs?|francs?)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "CHF"
    if re.search(r"¥|\b(?:jpy|yen)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "JPY"
    if re.search(r"\$|\b(?:dollars?|usd|cents?)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "USD"
    if re.search(r"€|\b(?:euros?|eur)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_CENTS", "EUR"
    if re.search(r"£|\b(?:pounds?|gbp|pence)\b", low_no_pct):
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "GBP"
    if "%" in low or "_pct_" in low_no_pct or re.search(r"\bpercent(?:age)?\b|\bpct\b", low):
        return "PERCENTAGE_RATE", "PERCENT", "PERCENT"
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
    if re.search(r"\b(?:form|dir|msme|3ceb|certificate|affidavit|bill|declaration|challan|schedule|return|report)\b", low):
        return "DOCUMENT_VERIFICATION", "DOCUMENT_ID", "REGISTRY_VERIFICATION"
    if re.search(r"^\s*[\$₹€£¥]?\s*[\d,]+(?:\.\d+)?\s*$", low):
        if "$" in low:
            return "MONETARY_QUANTUM", "CURRENCY_CENTS", "USD"
        if "₹" in low:
            return "MONETARY_QUANTUM", "CURRENCY_PAISA", "INR"
        if "€" in low:
            return "MONETARY_QUANTUM", "CURRENCY_CENTS", "EUR"
        if "£" in low:
            return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "GBP"
        if "¥" in low:
            return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", "JPY"
        return "MONETARY_QUANTUM", "CURRENCY_MINOR_UNITS", None
    return "BOOLEAN_FLAG", "BOOLEAN", "FLAG"


def parse_comparison_operator(text: str):
    low = text.lower()
    if re.search(
        r"\b(?:not\s+less\s+than|at\s+least|at\s+or\s+above|minimum\s+of|no\s+less\s+than|no\s+fewer\s+than|"
        r"equal\s+to\s+or\s+(?:greater|more)\s+than|not\s+below|or\s+more|or\s+greater|or\s+above)\b",
        low,
    ):
        return "GTE"
    if re.search(
        r"\b(?:not\s+exceeding|does\s+not\s+exceed|do\s+not\s+exceed|not\s+over|at\s+most|maximum\s+of|capped\s+at|"
        r"up\s+to|equal\s+to\s+or\s+(?:less|fewer)\s+than|twelve\s+months\s+or\s+less|or\s+less|or\s+below|or\s+fewer|no\s+more\s+than)\b",
        low,
    ):
        return "LTE"
    if re.search(
        r"\b(?:strictly\s+greater\s+than|exceeding|exceeds|exceed|more\s+than|greater\s+than|in\s+excess\s+of|above|over)\b",
        low,
    ):
        return "GT"
    if re.search(r"\b(?:strictly\s+less\s+than|less\s+than|fewer\s+than|falls\s+short\s+of|below|under)\b", low):
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
        "lacs", "crore", "crores", "thousand", "million", "billion",
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
    """Deterministic multi-clause segmentation mapped to inclusive token ranges [start, end]."""
    n = len(tokens)
    if n == 0:
        return []

    ABBREV_NO_SPLIT = {
        "rs", "no", "nos", "sl", "sec", "secs", "art", "arts", "reg", "regs", "sub", "cl", "para",
        "fig", "st", "dr", "mr", "ms", "vs", "etc", "e", "g", "i", "u.s", "u.k", "inc", "ltd", "llc",
        "co", "corp", "viz",
    }
    REF_PREFIXES = {
        "section", "sections", "sec", "sec.", "clause", "clauses", "cl", "cl.", "sub-section",
        "subsection", "subparagraph", "paragraph", "para", "para.", "rule", "rules", "article", "art",
        "art.", "regulation", "reg", "reg.", "of", "under", "to", "and", "or", "table", "sl", ",",
    }

    raw_segments = []
    seg_start = 0

    for i, tok in enumerate(tokens):
        low = tok.lower()
        prev_low = tokens[i - 1].lower() if i > 0 else ""
        next_low = tokens[i + 1].lower() if i + 1 < n else ""

        if (
            i > seg_start + 1
            and re.match(r"^\([a-zA-Z0-9ivxIVX]{1,4}\)$", tok)
            and prev_low.rstrip(".") not in REF_PREFIXES
            and next_low not in {"and", "or", "of", ",", ";"}
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
            is_decimal_adjacency = i + 1 < n and offsets[i][1] == offsets[i + 1][0]
            if not is_abbrev and not is_decimal_adjacency:
                should_split_after = True

        if should_split_after and i >= seg_start:
            raw_segments.append((seg_start, i))
            seg_start = i + 1

    if seg_start < n:
        raw_segments.append((seg_start, n - 1))

    if not raw_segments:
        return [(0, n - 1)]

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
        "this", "that", "these", "those", "have", "has", "had",
    }
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    if not filtered:
        return ""
    return "_".join(filtered[:4])


_LOGIC_FACT_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "at", "by", "from", "with", "into",
    "through", "as", "that", "which", "who", "whom", "whose", "such", "any", "every",
    "each", "all", "some", "none", "no", "not", "neither", "nor", "either", "or", "and",
    "provided", "where", "when", "if", "unless", "except", "until", "upon", "shall", "may",
    "must", "should", "can", "cannot", "will", "would", "could", "it", "its", "his", "her",
    "their", "be", "been", "being", "is", "are", "was", "were", "has", "have", "had",
    "there", "this", "these", "those", "within",
}


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


def parse_minor_units(text_val: str | None) -> int | None:
    if not text_val:
        return None
    qty = parse_numeric_quantity(text_val)
    if qty is None:
        return None
    return int(round(float(qty) * 100))


_EXTENDED_CURRENCIES = ("USD", "INR", "EUR", "GBP", "SGD", "AED", "CHF", "CAD", "AUD", "JPY")


def infer_rule_currency_info(spans: list[dict], text: str) -> tuple[str | None, str]:
    for s in spans:
        st, dt, unit = parse_unit_and_type(s["span"])
        if st == "MONETARY_QUANTUM" and dt in ("CURRENCY_PAISA", "CURRENCY_CENTS", "CURRENCY_MINOR_UNITS"):
            if unit in _EXTENDED_CURRENCIES:
                return unit, dt
    st, dt, unit = parse_unit_and_type(text)
    if st == "MONETARY_QUANTUM" and unit in _EXTENDED_CURRENCIES:
        return unit, dt
    if re.search(r"₹|\b(?:rupees?|rs\.?|inr|crores?|lakhs?|lacs?|paisa|paise)\b", text, re.IGNORECASE):
        return "INR", "CURRENCY_PAISA"
    if re.search(r"\b(?:sgd|singapore\s+dollars?|s\$)\b", text, re.IGNORECASE):
        return "SGD", "CURRENCY_CENTS"
    if re.search(r"\b(?:cad|canadian\s+dollars?|c\$)\b", text, re.IGNORECASE):
        return "CAD", "CURRENCY_CENTS"
    if re.search(r"\b(?:aud|australian\s+dollars?|a\$)\b", text, re.IGNORECASE):
        return "AUD", "CURRENCY_CENTS"
    if re.search(r"\b(?:aed|dirhams?)\b", text, re.IGNORECASE):
        return "AED", "CURRENCY_MINOR_UNITS"
    if re.search(r"\b(?:chf|swiss\s+francs?|francs?)\b", text, re.IGNORECASE):
        return "CHF", "CURRENCY_MINOR_UNITS"
    if re.search(r"¥|\b(?:jpy|yen)\b", text, re.IGNORECASE):
        return "JPY", "CURRENCY_MINOR_UNITS"
    if re.search(r"€|\b(?:euros?|eur)\b", text, re.IGNORECASE):
        return "EUR", "CURRENCY_CENTS"
    if re.search(r"£|\b(?:pounds?|gbp|pence)\b", text, re.IGNORECASE):
        return "GBP", "CURRENCY_MINOR_UNITS"
    if re.search(r"\$|\b(?:dollars?|usd|cents?)\b", text, re.IGNORECASE):
        return "USD", "CURRENCY_CENTS"
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
        "pound", "yen", "dirham", "dirhams", "franc", "francs", "cents", "cent", "paise", "paisa",
        "inr", "usd", "eur", "gbp", "sgd", "aed", "chf", "cad", "aud", "jpy",
        "lakh", "lakhs", "lac", "lacs", "crore", "crores", "thousand", "million", "billion",
    }
    words = [w for w in cleaned.split() if not w.isdigit() and w not in drop_words and len(w) > 1]
    camel = _derive_fact_key_camel(span_text, "")
    ccy_suffix = currency_code.capitalize() if currency_code in _EXTENDED_CURRENCIES else ""
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
