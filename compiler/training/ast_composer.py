"""AST-First Combinatorial Rule Composer for Model v2.

Generates arbitrary computational-law expression DAGs (depth 1..5+) directly from
universal algebraic and deontic building blocks (`LET`, `PIECEWISE`, `REDUCE`,
`FOLD`), then renders each DAG into varied, multi-level English legal phrasings
with exact span boundaries, ground-truth `SPAN_UNITS` classes, `V2_SEMANTIC_ROLES`,
and opcode-exact `V2_RELATION_TYPES`.

Design principles:
- Zero statute-specific or section-specific templates (no `194Q`, `IRC`, `MSMED`).
- Strictly disjoint formulation phrasing pools for `train` (`0, 1`) vs `val` (`2, 3`)
  so held-out validation tests syntactic & compositional generalization.
- Every composed computation sample aligns directly with `synthesize_usvm_ast_v2`
  without regex fallbacks.
"""

from __future__ import annotations

import random
from typing import Any


# Domain-agnostic commercial, regulatory, governance & financial nouns
_BASE_OPERANDS = [
    "gross invoice amount",
    "qualified revenue",
    "aggregate transaction value",
    "declared collateral balance",
    "net distributable surplus",
    "assessable contract sum",
    "annual royalty volume",
    "covered compensation base",
    "certified capital outlay",
    "cumulative disbursement total",
    "net operating proceeds",
    "eligible claim pool",
    "weighted emissions quota",
    "cross-border settlement volume",
    "insured cargo valuation",
    "metered throughput receipts",
]

_OOD_MODIFIERS = [
    "adjusted", "verified", "consolidated", "deferred", "residual", "audited",
    "primary", "marginal", "cross-border", "unencumbered", "statutory", "net",
]
_OOD_HEAD_NOUNS = [
    "allocation", "receivable", "exposure", "turnover", "contribution", "consideration",
    "tariff", "premium", "disbursement", "remittance", "reserve", "principal",
]

_DEFINED_TERMS = [
    "baseline allowance",
    "adjusted ceiling",
    "qualifying reserve",
    "net retention limit",
    "applicable floor",
    "minimum threshold",
    "eligible credit pool",
    "primary allocation",
    "statutory cap",
    "benchmark quota",
]

_BOOLEAN_PREDICATES = [
    ("the counterparty holds a valid certificate", "certificate valid"),
    ("audit verification is completed", "audit completed"),
    ("written notice is delivered", "notice delivered"),
    ("collateral coverage is maintained", "collateral maintained"),
    ("primary domicile is verified", "domicile verified"),
    ("regulatory exemption is waived", "exemption waived"),
    ("independent custody is established", "custody established"),
    ("material default remains uncured", "default uncured"),
    ("board ratification is recorded", "ratification recorded"),
    ("escrow confirmation is filed", "escrow filed"),
    ("electronic attestation is countersigned", "attestation countersigned"),
    ("continuous residency is proven", "residency proven"),
]

_COLLECTION_ITEMS = [
    ("approved vouchers", "claimed amount"),
    ("verified installments", "principal share"),
    ("covered policies", "insured benefit"),
    ("registered milestones", "payable fee"),
    ("active accounts", "reserved balance"),
    ("reporting units", "assessed charge"),
    ("shipped consignments", "declared freight"),
    ("settled contracts", "net consideration"),
]

_COLLECTION_BOOL_PREDS = [
    ("compliant status", "hasCompliantStatus"),
    ("timely filing", "hasTimelyFiling"),
    ("verified standing", "hasVerifiedStanding"),
    ("signed attestation", "hasSignedAttestation"),
]

_CURRENCY_RENDERS_BY_CODE = {
    "USD": [lambda k: f"${k:,}", lambda k: f"USD {k:,}"],
    "INR": [lambda k: f"₹{k:,}", lambda k: f"INR {k:,}"],
    "EUR": [lambda k: f"€{k:,}", lambda k: f"EUR {k:,}"],
    "GBP": [lambda k: f"£{k:,}", lambda k: f"GBP {k:,}"],
    "SGD": [lambda k: f"SGD {k:,}"],
    "CHF": [lambda k: f"CHF {k:,}"],
}

_WORD_AMOUNTS_BY_CODE = {
    "USD": {500000: "five hundred thousand dollars", 250000: "two hundred fifty thousand dollars", 40000: "forty thousand dollars"},
    "EUR": {15000: "fifteen thousand euros"},
}

_RATES = [
    "0.5%", "1.5%", "2%", "4%", "7.5%", "12%", "18%", "25%", "28%",
    "3 percent", "8 per cent", "15 percent", "22 per cent",
]

_DURATIONS = [
    "15 days", "30 days", "45 days", "60 days", "90 days",
    "1 calendar year", "2 reporting cycles", "6 months",
]

_COUNTS = [
    "2 inspections", "3 installments", "4 dependents", "5 directors", "2 violations",
]

_DATES_START = ["March 15, 2019", "January 1, 2020", "July 1, 2021", "April 1, 2022"]
_DATES_END = ["December 31, 2025", "January 1, 2027", "March 31, 2028", "October 15, 2026"]

_CITATION_NOISE = [
    "Schedule B clause 4",
    "Regulation 12(a)",
    "Section 4(2) Appendix",
    "Rule 19 table II",
    "Subdivision 8(c)",
    "Circular No. 14",
    "Article 7(b) Note",
    "Annex III Table 2",
]


def _pick_operand_phrase(rng: random.Random) -> str:
    if rng.random() < 0.35:
        return f"{rng.choice(_OOD_MODIFIERS)} {rng.choice(_OOD_HEAD_NOUNS)}"
    return rng.choice(_BASE_OPERANDS)


def _fmt_money(rng: random.Random, major_val: int, currency: str = "USD") -> str:
    word_map = _WORD_AMOUNTS_BY_CODE.get(currency) or {}
    if major_val in word_map and rng.random() < 0.20:
        return word_map[major_val]
    fns = _CURRENCY_RENDERS_BY_CODE.get(currency, _CURRENCY_RENDERS_BY_CODE["USD"])
    return rng.choice(fns)(major_val)


class CombinatorialAstComposer:
    """Samples random AST-first computational-law DAGs and renders Model v2 examples."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def sample_rule(self, formulation: int | None = None) -> dict[str, Any]:
        """Generates a single AST-first combinatorial v2 sample.

        Formulations 0 and 1 form the train phrasing distribution; 2 and 3 form
        the held-out validation phrasing distribution.
        """
        if formulation is None:
            formulation = self.rng.choice([0, 1]) if self.rng.random() < 0.80 else self.rng.choice([2, 3])
        ccy = self.rng.choice(["USD", "INR", "EUR", "GBP", "SGD", "CHF"])

        family = self.rng.choices(
            [
                "COMPOSED_ARITHMETIC",
                "BOUNDED_BRACKET",
                "BOOLEAN_DAG",
                "COLLECTION_FOLD",
                "SCOPED_SUBSTITUTION",
                "DEONTIC_GATE",
                "FOLD_TO_ARITHMETIC",
                "GATED_ARITHMETIC",
                "MULTI_TIER_BRACKET",
                "BRANCHED_RATES",
            ],
            weights=[0.18, 0.11, 0.13, 0.09, 0.09, 0.07, 0.09, 0.08, 0.07, 0.09],
            k=1,
        )[0]

        if family == "COMPOSED_ARITHMETIC":
            sample = self._gen_composed_arithmetic(formulation, currency=ccy)
        elif family == "BOUNDED_BRACKET":
            sample = self._gen_bounded_bracket(formulation, currency=ccy)
        elif family == "BOOLEAN_DAG":
            sample = self._gen_boolean_dag(formulation)
        elif family == "COLLECTION_FOLD":
            sample = self._gen_collection_fold(formulation)
        elif family == "SCOPED_SUBSTITUTION":
            sample = self._gen_scoped_substitution(formulation, currency=ccy)
        elif family == "FOLD_TO_ARITHMETIC":
            sample = self._gen_fold_to_arithmetic(formulation, currency=ccy)
        elif family == "GATED_ARITHMETIC":
            sample = self._gen_gated_arithmetic(formulation, currency=ccy)
        elif family == "MULTI_TIER_BRACKET":
            sample = self._gen_multi_tier_bracket(formulation, currency=ccy)
        elif family == "BRANCHED_RATES":
            sample = self._gen_branched_rates(formulation, currency=ccy)
        else:
            sample = self._gen_deontic_gate(formulation)

        return self._maybe_inject_distractor(sample, formulation)

    # ── Builder 1: Depth 1..5 Composed Arithmetic (`RATE`, `EXCESS`, `SUB_CREDIT`, `MIN_CAP`, `LET`) ──

    def _gen_composed_arithmetic(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        op_name = _pick_operand_phrase(self.rng)
        rate = self.rng.choice(_RATES)
        use_excess = self.rng.random() < 0.65
        floor_amt = _fmt_money(self.rng, self.rng.choice([5000, 12500, 25000, 50000, 120000]), currency=currency)
        use_let_wrapper = self.rng.random() < 0.40
        use_credit = self.rng.random() < 0.55
        use_cap = self.rng.random() < 0.55
        credit_amt = _fmt_money(self.rng, self.rng.choice([250, 500, 1200, 2500]), currency=currency)
        cap_amt = _fmt_money(self.rng, self.rng.choice([8000, 15000, 30000, 75000]), currency=currency)
        def_term = self.rng.choice(_DEFINED_TERMS)
        use_unit_contrast = self.rng.random() < 0.45
        dur_txt = self.rng.choice(_DURATIONS)
        cnt_txt = self.rng.choice(_COUNTS)

        spans: list[tuple[str, str, str]] = []  # (text, role, unit)
        rels: list[tuple[int, int, str]] = []
        clauses: list[str] = []

        # Optional Depth+1 LET definition header or cross-clause threshold definition
        if use_let_wrapper and use_excess:
            if formulation in (0, 1):
                c0 = f"(a) The term {def_term} means {floor_amt}."
            else:
                c0 = f"(a) {def_term.capitalize()} is fixed at {floor_amt}."
            clauses.append(c0)
            spans.append((def_term if def_term in c0 else def_term.capitalize(), "ROLE_DEFINED_TERM", "UNIT_SYMBOL_VAR"))
            spans.append((floor_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"))
            rels.append((0, 1, "REL_DEFINED_TERM_OF"))
            idx_floor_sym = 0
        else:
            idx_floor_sym = None

        prefix = "(b) " if use_let_wrapper and use_excess else ""
        rate_phrase = rate

        idx_rate = len(spans)
        spans.append((rate_phrase, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"))
        idx_op = len(spans)
        spans.append((op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"))

        if use_excess:
            if idx_floor_sym is not None:
                th_tok = f"such {def_term}"
                idx_th = len(spans)
                spans.append((th_tok, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_SYMBOL_VAR"))
                rels.append((idx_th, idx_floor_sym, "REL_COREF_LINK"))
            else:
                th_tok = f"exceeding {floor_amt}" if formulation in (0, 2) else f"over {floor_amt}"
                idx_th = len(spans)
                spans.append((th_tok, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"))
            rels.append((idx_rate, idx_th, "REL_RATE_TO_THRESHOLD"))
            rel_join = f"exceeding {th_tok}" if idx_floor_sym is not None else th_tok
            if formulation == 0:
                core_s = f"{prefix}The charge shall equal {rate_phrase} of the excess of {op_name} {rel_join}"
            elif formulation == 1:
                core_s = f"{prefix}On {op_name} {rel_join}, an assessment at {rate_phrase} applies"
            elif formulation == 2:
                core_s = f"{prefix}Liability is computed as {rate_phrase} on {op_name} {rel_join}"
            else:
                core_s = f"{prefix}A party shall remit {rate_phrase} of {op_name} {rel_join}"
        else:
            rels.append((idx_rate, idx_op, "REL_APPLIED_TO_BASE"))
            if formulation == 0:
                core_s = f"{prefix}The obligation shall equal {rate_phrase} of {op_name}"
            elif formulation == 1:
                core_s = f"{prefix}An assessment of {rate_phrase} applies to {op_name}"
            elif formulation == 2:
                core_s = f"{prefix}Payable amount is calculated at {rate_phrase} on {op_name}"
            else:
                core_s = f"{prefix}Every obligor shall contribute {rate_phrase} of {op_name}"

        tail_parts: list[str] = []
        # Compose arbitrary subset & order of SUB_CREDIT vs MIN_CAP opcodes
        mod_order = [("CREDIT", credit_amt) if use_credit else None, ("CAP", cap_amt) if use_cap else None]
        mod_items = [m for m in mod_order if m is not None]
        if len(mod_items) == 2 and self.rng.random() < 0.5:
            mod_items.reverse()

        for kind, amt in mod_items:
            idx_m = len(spans)
            spans.append((amt, "ROLE_EXCEPTION_CARVEOUT" if kind == "CREDIT" else "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"))
            if kind == "CREDIT":
                rels.append((idx_m, idx_rate, "REL_SUB_CREDIT"))
                tail_parts.append(
                    self.rng.choice([f"reduced by a credit of {amt}", f"minus an offset of {amt}"])
                    if formulation in (0, 1)
                    else self.rng.choice([f"less allowable rebate of {amt}", f"after subtracting {amt}"])
                )
            else:
                rels.append((idx_m, idx_rate, "REL_MIN_CAP"))
                tail_parts.append(
                    self.rng.choice([f"capped at a maximum of {amt}", f"not exceeding a ceiling of {amt}"])
                    if formulation in (0, 1)
                    else self.rng.choice([f"subject to an upper limit of {amt}", f"at most {amt}"])
                )

        if use_unit_contrast:
            contrast_tok = dur_txt if self.rng.random() < 0.5 else cnt_txt
            c_unit = "UNIT_DURATION" if contrast_tok == dur_txt else "UNIT_COUNT"
            c_role = "ROLE_DEADLINE_WINDOW" if c_unit == "UNIT_DURATION" else "ROLE_COUNT_QUANTIFIER"
            spans.append((contrast_tok, c_role, c_unit))
            tail_parts.append(f"across {contrast_tok}" if formulation in (0, 1) else f"over {contrast_tok}")

        if tail_parts:
            core_s = core_s + ", " + ", and ".join(tail_parts) + "."
        else:
            core_s = core_s + "."
        clauses.append(core_s)
        text = " ".join(clauses)

        dag = {
            "kind": "COMPOSED_ARITHMETIC",
            "depth": 1 + int(use_excess) + int(use_let_wrapper) + len(mod_items),
            "op": "CAP_MIN" if use_cap else ("SUB_CREDIT" if use_credit else ("EXCESS" if use_excess else "RATE")),
            "hasLet": use_let_wrapper and use_excess,
            "modifiers": [k for k, _ in mod_items],
        }
        return {
            "stype": "V2_AST_COMPOSED_ARITHMETIC",
            "modality_name": (
                "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB"
                if (use_cap or use_credit or use_let_wrapper)
                else ("MODALITY_COMPUTATIONAL_EXCESS" if use_excess else "MODALITY_COMPUTATIONAL_CLIFF")
            ),
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": clauses,
            "dag": dag,
        }

    # ── Builder 2: Bounded Brackets `(floor, ceiling]` + Optional `baseTax` ───────────────────────

    def _gen_bounded_bracket(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        op_name = self.rng.choice(_BASE_OPERANDS)
        lo_num = self.rng.choice([15000, 36900, 50000, 80000])
        hi_num = lo_num + self.rng.choice([25000, 52250, 90000])
        lo_amt = _fmt_money(self.rng, lo_num, currency=currency)
        hi_amt = _fmt_money(self.rng, hi_num, currency=currency)
        rate = self.rng.choice(_RATES)
        has_base_tax = self.rng.random() < 0.70
        base_tax = _fmt_money(self.rng, max(500, int(lo_num * 0.15)), currency=currency)
        use_decl_header = self.rng.random() < 0.55

        spans: list[tuple[str, str, str]] = []
        rels: list[tuple[int, int, str]] = []
        clauses: list[str] = []

        if use_decl_header:
            c0 = f"(a) Computation applies to {op_name}." if formulation in (0, 1) else f"(a) Base measured on {op_name}."
            clauses.append(c0)
            spans.append((op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"))
            pfx = "(b) "
        else:
            pfx = ""

        lo_span = f"over {lo_amt}" if formulation in (0, 2) else f"exceeding {lo_amt}"
        hi_span = f"not over {hi_amt}" if formulation in (0, 2) else f"at most {hi_amt}"

        if has_base_tax:
            idx_base = len(spans)
            spans.append((base_tax, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"))
        else:
            idx_base = None

        idx_rate = len(spans)
        spans.append((rate, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"))
        if not use_decl_header:
            spans.append((op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"))
        idx_lo = len(spans)
        spans.append((lo_span, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"))
        idx_hi = len(spans)
        spans.append((hi_span, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"))

        if idx_base is not None:
            rels.append((idx_base, idx_rate, "REL_BASE_TAX_ADD"))
        rels.append((idx_rate, idx_lo, "REL_RATE_TO_THRESHOLD"))
        rels.append((idx_lo, idx_hi, "REL_BRACKET_NEXT"))

        target_phrase = f"for {op_name} " if not use_decl_header else ""
        if has_base_tax:
            if formulation == 0:
                c1 = f"{pfx}Where {target_phrase}value is {lo_span} but {hi_span}, the liability is {base_tax} plus {rate} of the excess {lo_span}."
            elif formulation == 1:
                c1 = f"{pfx}For {target_phrase}tier {lo_span} and {hi_span}, charge equals {base_tax} together with {rate} on amount {lo_span}."
            elif formulation == 2:
                c1 = f"{pfx}Initial amount {base_tax} added to {rate} on slice {lo_span} up to {hi_span} applies {target_phrase}({lo_span}, {hi_span})."
            else:
                c1 = f"{pfx}Across band {lo_span} through {hi_span} {target_phrase}, base duty {base_tax} plus {rate} of excess is due."
            # Normalize hi_span presence cleanly in formulation 2/3
            if hi_span not in c1:
                c1 = f"{pfx}Within {lo_span} and {hi_span} {target_phrase}, {base_tax} plus {rate} on excess applies."
        else:
            if formulation in (0, 1):
                c1 = f"{pfx}If {op_name if not use_decl_header else 'base'} is {lo_span} and {hi_span}, liability is {rate} of excess {lo_span}."
            else:
                c1 = f"{pfx}For slice {lo_span} but {hi_span}, {rate} applies on {op_name if not use_decl_header else 'the excess'}."

        clauses.append(c1)
        text = " ".join(clauses)
        return {
            "stype": "V2_AST_BOUNDED_BRACKET",
            "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": clauses,
            "dag": {
                "kind": "BOUNDED_BRACKET",
                "depth": 3 if has_base_tax else 2,
                "hasBaseTax": has_base_tax,
                "hasDeclHeader": use_decl_header,
            },
        }

    # ── Builder 3: Arbitrary Depth Boolean Logic & Mixed AND/OR Precedence DAGs ─────────────────

    def _gen_boolean_dag(self, formulation: int) -> dict[str, Any]:
        preds = self.rng.sample(_BOOLEAN_PREDICATES, k=self.rng.randint(2, 4))
        use_def = self.rng.random() < 0.65
        term = self.rng.choice(_DEFINED_TERMS)
        use_mixed_dnf = len(preds) >= 3 and self.rng.random() < 0.40
        op_join = "DNF_MIXED" if use_mixed_dnf else self.rng.choice(["AND", "OR"])
        neg_mask = [self.rng.random() < 0.35 for _ in preds]
        if not any(neg_mask) and len(preds) >= 2 and self.rng.random() < 0.5:
            neg_mask[-1] = True

        spans: list[tuple[str, str, str]] = []
        rels: list[tuple[int, int, str]] = []

        if use_def:
            idx_def = 0
            spans.append((term, "ROLE_DEFINED_TERM", "UNIT_SYMBOL_VAR"))
        else:
            idx_def = None

        neg_markers = ["not the case that", "not satisfied that", "not established that", "false that"]
        pred_indices: list[int] = []
        pred_clauses: list[str] = []
        neg_edges: list[tuple[int, int]] = []
        for i, (p_text, _) in enumerate(preds):
            if neg_mask[i]:
                neg_tok = neg_markers[i % len(neg_markers)]
                idx_neg = len(spans)
                spans.append((neg_tok, "ROLE_PROHIBITION_GATE", "UNIT_NONE"))
                idx_p = len(spans)
                spans.append((p_text, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"))
                pred_indices.append(idx_p)
                neg_edges.append((idx_neg, idx_p))
                pred_clauses.append(f"it is {neg_tok} {p_text}")
            else:
                idx_p = len(spans)
                spans.append((p_text, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"))
                pred_indices.append(idx_p)
                pred_clauses.append(p_text)

        if use_mixed_dnf:
            # Group (P0 AND P1) OR (P2 [AND P3]) so AND binds tighter than OR
            split_k = 2
            lhs = f"({pred_clauses[0]} and {pred_clauses[1]})"
            rhs = f"({pred_clauses[2]} and {pred_clauses[3]})" if len(pred_clauses) == 4 else pred_clauses[2]
            joined = f"{lhs} or {rhs}"
            rels.append((pred_indices[0], pred_indices[1], "REL_LOGICAL_AND"))
            if len(pred_indices) == 4:
                rels.append((pred_indices[2], pred_indices[3], "REL_LOGICAL_AND"))
            rels.append((pred_indices[1], pred_indices[2], "REL_LOGICAL_OR"))
        else:
            conn = " and " if op_join == "AND" else " or "
            edge_op = "REL_LOGICAL_AND" if op_join == "AND" else "REL_LOGICAL_OR"
            joined = conn.join(pred_clauses)
            for i in range(1, len(pred_indices)):
                rels.append((pred_indices[i - 1], pred_indices[i], edge_op))

        for idx_neg, idx_p in neg_edges:
            rels.append((idx_neg, idx_p, "REL_LOGICAL_NOT"))
        if idx_def is not None:
            rels.append((idx_def, pred_indices[0], "REL_DEFINED_TERM_OF"))

        # Inject a contrast duration/count clause in 40% of boolean rules so numbers don't trigger TO_PAISE
        contrast = ""
        if self.rng.random() < 0.40:
            dur = self.rng.choice(_DURATIONS)
            spans.append((dur, "ROLE_COUNT_QUANTIFIER", "UNIT_DURATION"))
            contrast = f" throughout {dur}"

        if use_def:
            if formulation == 0:
                text = f"The term {term} is satisfied if {joined}{contrast}."
            elif formulation == 1:
                text = f"An entity holds {term} only where {joined}{contrast}."
            elif formulation == 2:
                text = f"Whenever {joined}{contrast}, {term} is established."
            else:
                text = f"{term.capitalize()} applies if {joined}{contrast}."
                # Ensure exact lowercase match when capitalized at start
                if term not in text:
                    spans[0] = (term.capitalize(), "ROLE_DEFINED_TERM", "UNIT_SYMBOL_VAR")
        else:
            if formulation in (0, 1):
                text = f"Eligibility holds provided {joined}{contrast}."
            else:
                text = f"Condition passes when {joined}{contrast}."

        return {
            "stype": "V2_AST_BOOLEAN_DAG",
            "modality_name": "MODALITY_DEONTIC_STATUS_OR_EVIDENCE",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": [text],
            "dag": {
                "kind": "BOOLEAN_DAG",
                "depth": len(preds),
                "op": op_join,
                "negatedCount": sum(neg_mask),
                "definedTerm": term if use_def else None,
            },
        }

    # ── Builder 4: Bounded Collection Folds & Quantifiers ──────────────────────────────────────

    def _gen_collection_fold(self, formulation: int) -> dict[str, Any]:
        coll_name, elem_name = self.rng.choice(_COLLECTION_ITEMS)
        pred_text, _ = self.rng.choice(_COLLECTION_BOOL_PREDS)
        mode = self.rng.choice(["SUM", "COUNT", "FORALL", "EXISTS", "NOT_FORALL"])

        spans: list[tuple[str, str, str]] = []
        rels: list[tuple[int, int, str]] = []

        if mode == "SUM":
            spans = [
                (elem_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
                (coll_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
            ]
            rels = [(0, 1, "REL_AGGREGATE_SUM")]
            if formulation in (0, 1):
                text = f"Total assessment equals the aggregate sum of {elem_name} across all {coll_name}."
            else:
                text = f"Combined value is obtained by summing {elem_name} over every item in {coll_name}."
        elif mode == "COUNT":
            spans = [
                (pred_text, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"),
                (coll_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
            ]
            rels = [(0, 1, "REL_AGGREGATE_COUNT")]
            if formulation in (0, 1):
                text = f"Determine the total count of {coll_name} satisfying {pred_text}."
            else:
                text = f"Score equals the number of {coll_name} with {pred_text}."
        else:
            spans = [
                (pred_text, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"),
                (coll_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
            ]
            if mode in ("FORALL", "NOT_FORALL"):
                rels.append((0, 1, "REL_QUANTIFIER_FORALL"))
                if mode == "NOT_FORALL":
                    rels.append((0, 1, "REL_LOGICAL_NOT"))
                    text = (
                        f"Compliance fails unless not every item in {coll_name} lacks {pred_text}."
                        if formulation in (0, 1)
                        else f"Condition triggers where it is not true that all {coll_name} hold {pred_text}."
                    )
                else:
                    text = (
                        f"Every record in {coll_name} shall demonstrate {pred_text}."
                        if formulation in (0, 1)
                        else f"All {coll_name} must maintain {pred_text}."
                    )
            else:
                rels.append((0, 1, "REL_QUANTIFIER_EXISTS"))
                text = (
                    f"Requirement is satisfied if at least one member of {coll_name} has {pred_text}."
                    if formulation in (0, 1)
                    else f"Rule passes whenever any entry in {coll_name} establishes {pred_text}."
                )

        return {
            "stype": "V2_AST_COLLECTION_FOLD",
            "modality_name": (
                "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB" if mode in ("SUM", "COUNT") else "MODALITY_DEONTIC_PERMISSION"
            ),
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": [text],
            "dag": {"kind": "COLLECTION_FOLD", "mode": mode, "depth": 2},
        }

    # ── Builder 5: Multi-Clause Lexical-Scope Substitution (`REL_DEFINED_TERM_OF` + `REL_SUBSTITUTE_WHEN`) ──

    def _gen_scoped_substitution(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        term = self.rng.choice(_DEFINED_TERMS)
        base_amt = _fmt_money(self.rng, self.rng.choice([4400, 12000, 25000, 60000]), currency=currency)
        sub_amt = _fmt_money(self.rng, self.rng.choice([18000, 35000, 75000, 150000]), currency=currency)
        d1 = self.rng.choice(_DATES_START)
        d2 = self.rng.choice(_DATES_END)

        c0 = (
            f"(a) The basic {term} equals {base_amt}."
            if formulation in (0, 1)
            else f"(a) Baseline {term} is fixed at {base_amt}."
        )
        if formulation == 0:
            c1 = f"(b) For periods beginning after {d1} and ending before {d2}, {sub_amt} shall be substituted for {base_amt}."
        elif formulation == 1:
            c1 = f"(b) After {d1} and before {d2}, the {term} is superseded by {sub_amt}."
        elif formulation == 2:
            c1 = f"(b) During cycles starting after {d1} and prior to {d2}, {sub_amt} applies in lieu of {base_amt}."
        else:
            c1 = f"(b) For reporting years after {d1} and before {d2}, {sub_amt} replaces the baseline {base_amt}."

        spans = [
            (term, "ROLE_DEFINED_TERM", "UNIT_SYMBOL_VAR"),
            (base_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"),
            (d1, "ROLE_TEMPORAL_CUTOVER_DATE", "UNIT_YEAR_DATE"),
            (d2, "ROLE_TEMPORAL_SUNSET_DATE", "UNIT_YEAR_DATE"),
            (sub_amt, "ROLE_EXCEPTION_CARVEOUT", "UNIT_MONEY"),
        ]
        rels = [
            (0, 1, "REL_DEFINED_TERM_OF"),
            (4, 0, "REL_SUBSTITUTE_WHEN"),
            (2, 4, "REL_CONDITION_TO_GATE"),
            (3, 4, "REL_SUNSET_TO_SECTION"),
        ]
        if base_amt in c1:
            spans.append((base_amt, "ROLE_IGNORE", "UNIT_MONEY"))

        clauses = [c0, c1]
        text = " ".join(clauses)
        return {
            "stype": "V2_AST_SCOPED_SUBSTITUTION",
            "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": clauses,
            "dag": {"kind": "SCOPED_SUBSTITUTION", "depth": 3, "term": term, "window": [d1, d2]},
        }

    # ── Builder 6: Multi-Clause Deontic / Obligation / Exception Gate ──────────────────────────

    def _gen_deontic_gate(self, formulation: int) -> dict[str, Any]:
        pre_txt, _ = self.rng.choice(_BOOLEAN_PREDICATES)
        dl_txt = self.rng.choice(_DURATIONS)
        use_exception = self.rng.random() < 0.65
        exc_txt, _ = self.rng.choice([p for p in _BOOLEAN_PREDICATES if p[0] != pre_txt])

        oblig_txt = self.rng.choice([
            "shall settle the balance",
            "must file an attestation",
            "shall place funds in reserve",
        ])
        prohib_txt = self.rng.choice([
            "shall not disburse proceeds",
            "is prohibited from transferring collateral",
        ])
        is_prohib = self.rng.random() < 0.50

        if is_prohib:
            spans = [
                (pre_txt, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"),
                (prohib_txt, "ROLE_PROHIBITION_GATE", "UNIT_BOOL_PREDICATE"),
            ]
            rels = [(0, 1, "REL_CONDITION_TO_GATE")]
            if use_exception:
                spans.append((exc_txt, "ROLE_EXCEPTION_CARVEOUT", "UNIT_BOOL_PREDICATE"))
                rels.append((2, 1, "REL_EXCEPTION_TO_PROHIBITION"))
                text = (
                    f"Where {pre_txt}, an obligor {prohib_txt}, except where {exc_txt}."
                    if formulation in (0, 1)
                    else f"Unless {exc_txt}, a party {prohib_txt} whenever {pre_txt}."
                )
            else:
                text = (
                    f"Whenever {pre_txt}, the entity {prohib_txt}."
                    if formulation in (0, 1)
                    else f"Provided {pre_txt}, no participant may waive and {prohib_txt}."
                )
            mod = "MODALITY_DEONTIC_PROHIBITION"
        else:
            spans = [
                (pre_txt, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"),
                (oblig_txt, "ROLE_MANDATORY_ACTION_OBLIGATION", "UNIT_BOOL_PREDICATE"),
                (dl_txt, "ROLE_DEADLINE_WINDOW", "UNIT_DURATION"),
            ]
            rels = [(1, 0, "REL_ACTION_TO_PRECONDITION"), (2, 1, "REL_DEADLINE_TO_TRIGGER")]
            text = (
                f"If {pre_txt}, the party {oblig_txt} within {dl_txt}."
                if formulation in (0, 1)
                else f"Within {dl_txt} after {pre_txt}, the party {oblig_txt}."
            )
            mod = "MODALITY_DEONTIC_OBLIGATION_WINDOW"

        return {
            "stype": "V2_AST_DEONTIC_GATE",
            "modality_name": mod,
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": [text],
            "dag": {"kind": "DEONTIC_GATE", "prohibition": is_prohib, "depth": len(spans)},
        }

    # ── Builder 7: Cross-Primitive Fold -> Excess -> Rate -> Cap/Credit Pipeline ───────────────

    def _gen_fold_to_arithmetic(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        coll_name, elem_name = self.rng.choice(_COLLECTION_ITEMS)
        rate = self.rng.choice(_RATES)
        floor_amt = _fmt_money(self.rng, self.rng.choice([10000, 25000, 50000]), currency=currency)
        cap_amt = _fmt_money(self.rng, self.rng.choice([12000, 30000, 80000]), currency=currency)
        use_cap = self.rng.random() < 0.65

        # Note: order guarantees no substring collisions
        spans: list[tuple[str, str, str]] = [
            (elem_name, "ROLE_COMPUTATION_OPERAND", "UNIT_MONEY"),
            (coll_name, "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", "UNIT_SYMBOL_VAR"),
            (rate, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"),
            (floor_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"),
        ]
        rels: list[tuple[int, int, str]] = [
            (0, 1, "REL_AGGREGATE_SUM"),
            (2, 3, "REL_RATE_TO_THRESHOLD"),
        ]
        if use_cap:
            spans.append((cap_amt, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"))
            rels.append((4, 2, "REL_MIN_CAP"))

        cap_phrase = f", capped at {cap_amt}" if use_cap else ""
        if formulation == 0:
            text = f"Charge equals {rate} of the aggregate sum of {elem_name} across {coll_name} exceeding {floor_amt}{cap_phrase}."
        elif formulation == 1:
            text = f"On total {elem_name} summed over all {coll_name} above {floor_amt}, {rate} applies{cap_phrase}."
        elif formulation == 2:
            text = f"Whenever summed {elem_name} across {coll_name} exceeds {floor_amt}, liability is {rate} of the excess{cap_phrase}."
        else:
            text = f"Combined {elem_name} of all {coll_name} over {floor_amt} is assessed at {rate}{cap_phrase}."

        return {
            "stype": "V2_AST_FOLD_TO_ARITHMETIC",
            "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": [text],
            "dag": {"kind": "FOLD_TO_ARITHMETIC", "depth": 3 + int(use_cap), "hasCap": use_cap},
        }

    # ── Builder 8: Hybrid Boolean / Quantifier Condition Gating Arithmetic ────────────────────

    def _gen_gated_arithmetic(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        preds = self.rng.sample(_BOOLEAN_PREDICATES, k=2)
        op_name = self.rng.choice(_BASE_OPERANDS)
        rate = self.rng.choice(_RATES)
        floor_amt = _fmt_money(self.rng, self.rng.choice([8000, 20000, 45000]), currency=currency)
        use_not = self.rng.random() < 0.50
        op_join = self.rng.choice(["AND", "OR"])

        spans: list[tuple[str, str, str]] = [
            (preds[0][0], "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"),
        ]
        rels: list[tuple[int, int, str]] = []
        if use_not:
            neg_tok = "not established that"
            spans.append((neg_tok, "ROLE_PROHIBITION_GATE", "UNIT_NONE"))
            spans.append((preds[1][0], "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"))
            idx_p2 = 2
            rels.append((1, 2, "REL_LOGICAL_NOT"))
            p2_str = f"it is {neg_tok} {preds[1][0]}"
        else:
            spans.append((preds[1][0], "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"))
            idx_p2 = 1
            p2_str = preds[1][0]

        rels.append((0, idx_p2, "REL_LOGICAL_AND" if op_join == "AND" else "REL_LOGICAL_OR"))
        idx_rate = len(spans)
        spans.append((rate, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"))
        idx_op = len(spans)
        spans.append((op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"))
        idx_fl = len(spans)
        spans.append((floor_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"))
        rels.append((idx_rate, idx_fl, "REL_RATE_TO_THRESHOLD"))

        conn = " and " if op_join == "AND" else " or "
        cond_s = f"{preds[0][0]}{conn}{p2_str}"
        if formulation == 0:
            text = f"Where {cond_s}, an assessment of {rate} applies to {op_name} exceeding {floor_amt}."
        elif formulation == 1:
            text = f"Provided {cond_s}, obligor shall pay {rate} on excess of {op_name} over {floor_amt}."
        elif formulation == 2:
            text = f"Liability is {rate} on {op_name} above {floor_amt} if {cond_s}."
        else:
            text = f"Only when {cond_s}, charge {rate} of {op_name} exceeding {floor_amt}."

        return {
            "stype": "V2_AST_GATED_ARITHMETIC",
            "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": [text],
            "dag": {"kind": "GATED_ARITHMETIC", "depth": 4, "gate": op_join, "negated": use_not},
        }

    # ── Builder 9: Multi-Clause Progressive Bracket Tiers (2..3 Tiers) ──────────────────────────

    def _gen_multi_tier_bracket(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        op_name = self.rng.choice(_BASE_OPERANDS)
        t1_lo, t1_hi, t2_hi = 10000, 40000, 90000
        a1 = _fmt_money(self.rng, t1_lo, currency=currency)
        a2 = _fmt_money(self.rng, t1_hi, currency=currency)
        a3 = _fmt_money(self.rng, t2_hi, currency=currency)
        r1 = self.rng.choice(["4%", "8%", "12%"])
        r2 = self.rng.choice(["18%", "22%", "28%"])
        base2 = _fmt_money(self.rng, 3600, currency=currency)

        c0 = f"(a) Computation applies to {op_name}." if formulation in (0, 1) else f"(a) Schedules evaluate {op_name}."
        c1 = f"(b) Over {a1} but not over {a2}, rate is {r1}." if formulation in (0, 1) else f"(b) Slice exceeding {a1} through {a2} bears {r1}."
        c2 = f"(c) Exceeding {a2} and not over {a3}, duty equals {base2} plus {r2}." if formulation in (0, 1) else f"(c) For tier over {a2} up to {a3}, {base2} plus {r2} applies."

        spans: list[tuple[str, str, str]] = [
            (op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
            (r1, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"),
            (a1, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"),
            (a2, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"),
            (base2, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"),
            (r2, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"),
            (a2, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"),
            (a3, "ROLE_TRANSACTION_CLIFF_THRESHOLD", "UNIT_MONEY"),
        ]
        rels: list[tuple[int, int, str]] = [
            (1, 2, "REL_RATE_TO_THRESHOLD"),
            (2, 3, "REL_BRACKET_NEXT"),
            (4, 5, "REL_BASE_TAX_ADD"),
            (5, 6, "REL_RATE_TO_THRESHOLD"),
            (6, 7, "REL_BRACKET_NEXT"),
        ]
        clauses = [c0, c1, c2]
        text = " ".join(clauses)
        return {
            "stype": "V2_AST_MULTI_TIER_BRACKET",
            "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": clauses,
            "dag": {"kind": "MULTI_TIER_BRACKET", "tiers": 2, "depth": 4},
        }

    # ── Builder 10: Conditional & Additive Dual-Rate Expressions (`IF cond THEN r1 ELSE r2` | `r1 + r2`)

    def _gen_branched_rates(self, formulation: int, currency: str = "USD") -> dict[str, Any]:
        op_name = _pick_operand_phrase(self.rng)
        mode = "conditional" if self.rng.random() < 0.65 else "additive"
        r1 = self.rng.choice(["4%", "7.5%", "12%"])
        r2 = self.rng.choice(["18%", "22%", "28%"])
        use_excess = self.rng.random() < 0.55
        floor_amt = _fmt_money(self.rng, self.rng.choice([10000, 25000, 50000]), currency=currency)

        if mode == "conditional":
            pred, _ = self.rng.choice(_BOOLEAN_PREDICATES)
            spans: list[tuple[str, str, str]] = [
                (pred, "ROLE_AGENT_STATE_PRECONDITION", "UNIT_BOOL_PREDICATE"),
                (r1, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"),
                (op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
            ]
            rels: list[tuple[int, int, str]] = [(1, 0, "REL_CONDITION_TO_GATE")]
            if use_excess:
                idx_fl = len(spans)
                spans.append((floor_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"))
                rels.append((1, idx_fl, "REL_RATE_TO_THRESHOLD"))
            idx_r2 = len(spans)
            spans.append((r2, "ROLE_PENALTY_OR_DEFAULT_OVERRIDE", "UNIT_RATE"))
            rels.append((idx_r2, 1, "REL_OVERRIDE_TO_GUARD"))

            base_phrase = f"{op_name} exceeding {floor_amt}" if use_excess else op_name
            if formulation == 0:
                text = f"If {pred}, apply {r1} to {base_phrase}; otherwise apply {r2}."
            elif formulation == 1:
                text = f"Where {pred}, charge equals {r1} of {base_phrase}, failing which {r2} applies."
            elif formulation == 2:
                text = f"The levy on {base_phrase} is {r1} whenever {pred}, or else {r2}."
            else:
                text = f"Provided {pred}, assess {r1} over {base_phrase}; in default thereof impose {r2}."
            return {
                "stype": "V2_AST_BRANCHED_RATES",
                "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
                "formulation": formulation,
                "text": text,
                "spans_with_units": spans,
                "relations": rels,
                "clauses": [text],
                "dag": {"kind": "BRANCHED_RATES", "mode": "conditional", "depth": 3},
            }

        # Additive dual-rate composition (`r1(base) + r2(base)`)
        spans = [
            (r1, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"),
            (op_name, "ROLE_COMPUTATION_OPERAND", "UNIT_SYMBOL_VAR"),
            (r2, "ROLE_STANDARD_STATUTORY_RATE", "UNIT_RATE"),
        ]
        rels = [(0, 2, "REL_AGGREGATE_SUM")]
        if use_excess:
            spans.append((floor_amt, "ROLE_TRANSACTION_EXCESS_THRESHOLD", "UNIT_MONEY"))
            rels.append((0, 3, "REL_RATE_TO_THRESHOLD"))
        base_phrase = f"{op_name} above {floor_amt}" if use_excess else op_name
        if formulation == 0:
            text = f"Total charge equals the sum of {r1} and {r2} on {base_phrase}."
        elif formulation == 1:
            text = f"An obligation of {r1} plus an additional {r2} applies to {base_phrase}."
        elif formulation == 2:
            text = f"Compute {r1} of {base_phrase} added to {r2} thereof."
        else:
            text = f"Both {r1} and {r2} aggregate on {base_phrase}."
        return {
            "stype": "V2_AST_BRANCHED_RATES",
            "modality_name": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
            "formulation": formulation,
            "text": text,
            "spans_with_units": spans,
            "relations": rels,
            "clauses": [text],
            "dag": {"kind": "BRANCHED_RATES", "mode": "additive", "depth": 2},
        }

    # ── Shared Distractor & Fronted/Trailing Clause Variation (`ROLE_IGNORE` + `UNIT_NONE`) ───

    def _maybe_inject_distractor(self, sample: dict[str, Any], formulation: int) -> dict[str, Any]:
        if self.rng.random() < 0.45:
            cite = self.rng.choice(_CITATION_NOISE)
            if self.rng.random() < 0.50:
                # Fronted citation preamble so spans occur at shifted sentence positions
                lead = "Pursuant to" if formulation in (0, 1) else "Subject to"
                sample["text"] = f"{lead} {cite}, {sample['text']}"
                if sample.get("clauses"):
                    sample["clauses"] = [f"{lead} {cite}, {sample['clauses'][0]}"] + sample["clauses"][1:]
            else:
                pfx = "Reference:" if formulation in (0, 1) else "See"
                sample["text"] = f"{sample['text']} {pfx} {cite}."
                if sample.get("clauses"):
                    sample["clauses"][-1] = f"{sample['clauses'][-1]} {pfx} {cite}."
            sample["spans_with_units"].append((cite, "ROLE_IGNORE", "UNIT_NONE"))
        return sample
