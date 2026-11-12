"""
=============================================================================
MUDRAFORGE STATUTORY RL ENVIRONMENT WITH USVM REWARD ORACLE
=============================================================================

Reinforcement learning environment that provides deterministic scalar rewards
by executing candidate AST programs in the canonical JavaScript Universal
Statutory Virtual Machine (USVM) via ONE persistent long-lived Node worker.

Reward Formulation:
  R_total = R_grammar + R_types + R_invariants + R_vectors
"""

import math
import re
from datetime import date, timedelta
try:
    from compiler.vm.usvm_evaluator import (
        CanonicalUsvmEvaluator,
        CANONICAL_TEST_VECTORS,
        DOMAIN_TEST_VECTORS,
        _parse_monetary_to_inr,
        _parse_duration_days,
    )
except ImportError:
    from usvm_evaluator import (
        CanonicalUsvmEvaluator,
        CANONICAL_TEST_VECTORS,
        DOMAIN_TEST_VECTORS,
        _parse_monetary_to_inr,
        _parse_duration_days,
    )

STYPE_VECTOR_MAP = {
    "194Q_STYLE": "194Q_STYLE",
    "194Q_CONCISE_STYLE": "194Q_STYLE",
    "206C1H_STYLE": "206C1H_STYLE",
    "194C_STYLE": "194C_STYLE",
    "194R_STYLE": "194R_STYLE",
    "BIZARRE_194R_PERQUISITES_IN_KIND": "194R_STYLE",
    "GST_BLOCKED_ITC_STYLE": "GST_17_5_STYLE",
    "SUBJECTIVE_EVIDENCE_STYLE": "CGST_SEC8_STYLE",
    "MSME_DEADLINE_STYLE": "MSME_15_STYLE",
    "SUNSET_STYLE": "SUNSET_STYLE",
    "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE": "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE",
    "COMPANIES_RELATED_PARTY_STYLE": "COMPANIES_RELATED_PARTY_STYLE",
    "COMPANIES_AUDIT_COMMITTEE_STYLE": "COMPANIES_RELATED_PARTY_STYLE",
    "COMPANIES_CSR_MANDATE_STYLE": "COMPANIES_RELATED_PARTY_STYLE",
    "IBC_RESOLUTION_INELIGIBILITY_STYLE": "IBC_RESOLUTION_INELIGIBILITY_STYLE",
    "IBC_INSOLVENCY_TRIGGER_STYLE": "IBC_RESOLUTION_INELIGIBILITY_STYLE",
    "PROCEDURAL_NOTICE_WINDOW_STYLE": "PROCEDURAL_NOTICE_WINDOW_STYLE",
    "COMMERCIAL_COURT_MEDIATION_STYLE": "PROCEDURAL_NOTICE_WINDOW_STYLE",
    "GENERAL_PERMISSION_CONDITIONAL_STYLE": "GENERAL_PERMISSION_CONDITIONAL_STYLE",
    "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE": "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE",
    "GENERAL_ENTITY_GRAPH_RELATION_STYLE": "GENERAL_ENTITY_GRAPH_RELATION_STYLE",
    "GENERAL_AGGREGATE_COLLECTION_STYLE": "GENERAL_AGGREGATE_COLLECTION_STYLE",
    "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE": "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE",
    "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE": "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE",
}

OUTCOME_NUMERIC_KEYS = (
    "calculatedTaxInr",
    "calculatedInterestInr",
    "effectiveAmount",
    "netAdjustedBase",
    "sumTotal",
    "countTotal",
)
OUTCOME_INT_KEYS = ("effectiveRateBps", "overdueDays", "deadlineDays")
OUTCOME_BOOL_OR_STR_KEYS = (
    "isApplicable",
    "isTdsApplicable",
    "isPermitted",
    "isCompliant",
    "isSatisfied",
    "isBlocked",
    "isProhibited",
    "isAdmissible",
    "carveoutApplied",
    "isEligible",
    "isDisqualified",
    "isApproved",
    "isSuitAdmissible",
    "isItcEligible",
    "inOverrideWindow",
    "inSuspensionWindow",
    "isSuspended",
    "gateType",
    "modalityType",
    "disposition",
    "gateStatus",
)

_BUILT_IN_FACT_KEYS = {
    "transactionDate", "creditDate", "paymentDate", "taxYear",
    "invoiceAmount", "invoiceAmountInr", "invoiceAmountPaisa",
    "currentInvoicePaisa", "primaryBasePaisa", "priorPurchasesPaisa",
    "priorPurchasesInCurrentFyInr", "buyerPriorTurnoverPaisa",
    "triggerDate", "settlementDate", "rbiBankRatePct",
    "payee", "payee.isResident", "payee.isPanValid",
}


def _collect_fact_refs(node, out_set: set[str]):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "collection" and isinstance(v, str) and v.startswith("facts."):
                out_set.add(v[6:])
            else:
                _collect_fact_refs(v, out_set)
    elif isinstance(node, list):
        for item in node:
            _collect_fact_refs(item, out_set)
    elif isinstance(node, str):
        for m in re.finditer(r"\bfacts\.([a-zA-Z_][a-zA-Z0-9_.]*)", node):
            out_set.add(m.group(1))


def _collect_op_nodes(node, op_name: str, out_list: list[dict]):
    if isinstance(node, dict):
        if node.get("op") == op_name:
            out_list.append(node)
        for v in node.values():
            _collect_op_nodes(v, op_name, out_list)
    elif isinstance(node, list):
        for item in node:
            _collect_op_nodes(item, op_name, out_list)


def _minor_str_to_major(raw) -> float:
    try:
        return float(raw) / 100.0
    except (TypeError, ValueError):
        return 0.0


def _iso_shift(iso_str: str, delta_days: int) -> str:
    d = date.fromisoformat(iso_str[:10])
    return (d + timedelta(days=delta_days)).isoformat()


def _extract_expected_outcome(gold_res: dict, ast: dict) -> dict:
    if not isinstance(gold_res, dict) or gold_res.get("error"):
        return {}
    comp = (ast or {}).get("computation") or {}
    pipe = comp.get("pipeline") if isinstance(comp.get("pipeline"), dict) else {}
    out_spec = pipe.get("output") if isinstance(pipe.get("output"), dict) else {}
    rc = (ast or {}).get("rootCondition") or {}
    node_type = rc.get("nodeType", "TRANSACTIONAL_WITHHOLDING")
    is_pure_logic = (
        node_type == "GENERAL_RULE_GATE"
        or comp.get("archetype") == "PURE_LOGIC_PIPELINE"
    )

    expected = {}
    if is_pure_logic:
        for k in (
            "isPermitted", "isProhibited", "isBlocked", "isCompliant",
            "isSatisfied", "isApplicable", "carveoutApplied",
        ):
            if k in out_spec or k in gold_res:
                if k in gold_res and isinstance(gold_res[k], bool):
                    expected[k] = gold_res[k]
        if not expected:
            expected["isCompliant"] = bool(gold_res.get("isCompliant", False))
        return expected

    # Gate / non-pipeline deontic rules
    if node_type == "STATUS_DISQUALIFICATION_GATE":
        for k in ("isDisqualified", "isEligible", "isBlocked"):
            if k in gold_res:
                expected[k] = bool(gold_res[k])
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        return expected
    if node_type == "CORPORATE_APPROVAL_GATE":
        for k in ("isApproved", "exemptionApplied", "requiresShareholderSpecialResolution"):
            if k in gold_res:
                expected[k] = bool(gold_res[k])
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        return expected
    if node_type == "PROCEDURAL_NOTICE_GATE":
        for k in ("isSuitAdmissible", "exemptionApplied", "isBlocked"):
            if k in gold_res:
                expected[k] = bool(gold_res[k])
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        return expected
    if node_type == "BOOLEAN_PROHIBITION_GATE" and not out_spec:
        for k in ("isBlocked", "isCompliant"):
            if k in gold_res:
                expected[k] = bool(gold_res[k])
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        return expected
    if node_type == "SUBJECTIVE_EVIDENCE_GATE":
        for k in ("isApproved", "isBlocked", "isCompliant"):
            if k in gold_res:
                expected[k] = bool(gold_res[k])
        if "gateStatus" in gold_res:
            expected["gateStatus"] = str(gold_res["gateStatus"])
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        return expected

    if (
        not out_spec
        and (
            "gateType" in gold_res
            or node_type in (
                "DEONTIC_STATE_GATE",
                "ACCOUNTING_ELECTION_CARVEOUT",
                "DISCLOSURE_PRESENTATION_OBLIGATION",
                "ACCOUNTING_RECOGNITION_GATE",
                "MEASUREMENT_VALUATION_RULE",
            )
        )
    ):
        for k in ("isBlocked", "isCompliant", "isAdmissible", "isPermitted"):
            if k in gold_res:
                expected[k] = bool(gold_res[k])
        for k in ("gateType", "modalityType", "disposition"):
            if k in gold_res and gold_res[k] is not None:
                expected[k] = str(gold_res[k])
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        return expected

    if node_type == "TEMPORAL_WINDOW_GATE" or ("overdueDays" in gold_res and "deadlineDays" in gold_res and not out_spec):
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        if "isCompliant" in gold_res:
            expected["isCompliant"] = bool(gold_res["isCompliant"])
        if "overdueDays" in gold_res:
            expected["overdueDays"] = int(gold_res["overdueDays"])
        if "deadlineDays" in gold_res:
            expected["deadlineDays"] = int(gold_res["deadlineDays"])
        if "calculatedInterestInr" in gold_res:
            expected["calculatedInterestInr"] = round(float(gold_res.get("calculatedInterestInr", 0.0) or 0.0), 2)
        return expected

    # Pipeline & computational rules
    if "isApplicable" in gold_res:
        expected["isApplicable"] = bool(gold_res["isApplicable"])
    if "isCompliant" in out_spec and "isCompliant" in gold_res:
        expected["isCompliant"] = bool(gold_res["isCompliant"])
    if "isSuspended" in out_spec and gold_res.get("isSuspended") is not None:
        expected["isSuspended"] = bool(gold_res["isSuspended"])
    if "inSuspensionWindow" in out_spec and gold_res.get("inSuspensionWindow") is not None:
        expected["inSuspensionWindow"] = bool(gold_res["inSuspensionWindow"])
    if "inOverrideWindow" in out_spec and gold_res.get("inOverrideWindow") is not None:
        expected["inOverrideWindow"] = bool(gold_res["inOverrideWindow"])

    if "calculatedInterestInr" in out_spec:
        expected["calculatedInterestInr"] = round(float(gold_res.get("calculatedInterestInr", 0.0) or 0.0), 2)
        if "overdueDays" in gold_res:
            expected["overdueDays"] = int(gold_res["overdueDays"])
        if "deadlineDays" in gold_res:
            expected["deadlineDays"] = int(gold_res["deadlineDays"])
    elif "effectiveAmount" in out_spec:
        expected["effectiveAmount"] = round(float(gold_res.get("effectiveAmount", 0.0) or 0.0), 2)
    elif "countTotal" in out_spec:
        expected["countTotal"] = float(gold_res.get("countTotal", 0.0) or 0.0)
    elif (
        "tdsAmountPaisa" in out_spec
        or "taxAmountPaisa" in out_spec
        or "combinedAmountPaisa" in out_spec
        or "sumTotalPaisa" in out_spec
        or (not out_spec and node_type == "TRANSACTIONAL_WITHHOLDING")
    ):
        expected["calculatedTaxInr"] = round(float(gold_res.get("calculatedTaxInr", 0.0) or 0.0), 2)
        if "effectiveRateBps" in gold_res and ("effectiveRateBps" in out_spec or not out_spec):
            expected["effectiveRateBps"] = int(gold_res.get("effectiveRateBps", 0) or 0)

    if not expected:
        expected["isApplicable"] = bool(gold_res.get("isApplicable", True))
        if "isCompliant" in gold_res:
            expected["isCompliant"] = bool(gold_res["isCompliant"])
    return expected


class StatutoryRlEnvironment:
    """
    Evaluates candidate AST actions via the canonical JS USVM and computes deterministic rewards.
    """

    def __init__(self):
        self.evaluator = CanonicalUsvmEvaluator()

    def _build_pure_logic_fact_candidates(self, ast: dict) -> list[tuple[str, dict]]:
        comp = (ast or {}).get("computation") or {}
        fs = comp.get("factSchema") or {}
        booleans = list(fs.get("booleans") or [])
        collections = list(fs.get("collections") or [])
        numbers = list(fs.get("numbers") or [])
        outcome_mode = str(fs.get("outcomeMode") or "OBLIGATION_RULE").upper()

        coll_keys = {c.get("factKey") for c in collections if c.get("factKey")}
        coll_scalars = {c.get("scalarFallbackKey") for c in collections if c.get("scalarFallbackKey")}

        def _populate_collection(base_facts: dict, col: dict, satisfy: bool):
            ck = col.get("factKey")
            if not ck:
                return
            q = str(col.get("quantifier") or "FORALL").upper()
            pred_keys = list(col.get("itemPredicateKeys") or ["isVerified"])
            sk = col.get("scalarFallbackKey")

            if q == "FORALL":
                if satisfy:
                    item_true = {ak: True for ak in pred_keys}
                    items = [dict(item_true), dict(item_true)]
                    sc_val = True
                else:
                    item_true = {ak: True for ak in pred_keys}
                    item_false = {ak: False for ak in pred_keys}
                    items = [item_true, item_false]
                    sc_val = False
            elif q in ("NOT_EXISTS", "FORALL_NOT"):
                # satisfy=True means NONE has bad predicate (good/clean)
                if satisfy:
                    item_clean = {ak: False for ak in pred_keys}
                    items = [dict(item_clean), dict(item_clean)]
                    sc_val = False
                else:
                    item_clean = {ak: False for ak in pred_keys}
                    item_bad = {ak: True for ak in pred_keys}
                    items = [item_clean, item_bad]
                    sc_val = True
            elif q.startswith("COUNT_"):
                # find matching number threshold
                n_entry = next((n for n in numbers if n.get("factKey") == sk), None)
                thresh = int(round(float((n_entry or {}).get("threshold") or 1)))
                op = str((n_entry or {}).get("operator") or "GTE").upper()
                want_high = (op in ("GT", "GTE")) == satisfy
                target_n = (thresh + 1) if want_high else max(0, thresh - 1)
                items = [{ak: True for ak in pred_keys} | {"excluded": False} for _ in range(max(0, target_n))]
                if not items:
                    items = [{"excluded": True}]
                sc_val = target_n
            else:
                # EXISTS
                if satisfy:
                    item_clean = {ak: False for ak in pred_keys}
                    item_hit = {ak: True for ak in pred_keys}
                    items = [item_clean, item_hit]
                    sc_val = True
                else:
                    item_clean = {ak: False for ak in pred_keys}
                    items = [dict(item_clean), dict(item_clean)]
                    sc_val = False

            base_facts[ck] = items
            for alt in (col.get("altFactKeys") or []):
                base_facts[alt] = items
            if sk:
                base_facts[sk] = sc_val

        def _base_satisfied_facts() -> dict:
            f = {}
            for b in booleans:
                fk = b.get("factKey")
                pol = str(b.get("polarity") or "REQUIRE_TRUE").upper()
                if not fk or fk in coll_keys or fk in coll_scalars:
                    continue
                if pol in ("REQUIRE_TRUE", "TRIGGER_TRUE"):
                    f[fk] = True
                elif pol == "REQUIRE_FALSE":
                    f[fk] = False
                elif pol == "CARVEOUT_TRUE":
                    f[fk] = (outcome_mode == "PROHIBITION_WITH_ESCAPE")
            for n in numbers:
                fk = n.get("factKey")
                if not fk or fk in coll_scalars:
                    continue
                th = float(n.get("threshold") or 0)
                op = str(n.get("operator") or "LTE").upper()
                if op in ("LTE", "LT"):
                    val = max(0, int(round(th - 1)))
                else:
                    val = int(round(th + 5)) if th > 0 else 5
                f[fk] = val
                for alt in (n.get("altFactKeys") or []):
                    f[alt] = val
                if n.get("boolFallbackKey"):
                    f[n["boolFallbackKey"]] = True
            for col in collections:
                _populate_collection(f, col, satisfy=True)
            return f

        v1 = _base_satisfied_facts()
        v2 = dict(v1)
        if outcome_mode == "PROHIBITION_WITH_ESCAPE":
            # v1 has trigger=True & escape=True; v2 breaks escape -> prohibited; v3 disables trigger -> exempt
            for b in booleans:
                if str(b.get("polarity") or "").upper() == "CARVEOUT_TRUE":
                    v2[b["factKey"]] = False
            for n in numbers:
                op = str(n.get("operator") or "").upper()
                if op in ("GTE", "GT"):
                    v2[n["factKey"]] = 0
                    for alt in (n.get("altFactKeys") or []):
                        v2[alt] = 0
                else:
                    v2[n["factKey"]] = int(round(float(n.get("threshold") or 10) + 20))
                    if n.get("boolFallbackKey"):
                        v2[n["boolFallbackKey"]] = False
            v3 = dict(v2)
            for b in booleans:
                if str(b.get("polarity") or "").upper() == "TRIGGER_TRUE":
                    v3[b["factKey"]] = False
            return [("VEC_LOGIC_ESCAPE_OK", v1), ("VEC_LOGIC_BLOCKED", v2), ("VEC_LOGIC_INACTIVE", v3)]

        # Flip one main collection or boolean condition for v2 counterexample
        flipped = False
        if collections:
            _populate_collection(v2, collections[0], satisfy=False)
            flipped = True
        else:
            for b in booleans:
                fk = b.get("factKey")
                pol = str(b.get("polarity") or "").upper()
                if fk in v2 and pol in ("REQUIRE_TRUE", "REQUIRE_FALSE", "TRIGGER_TRUE"):
                    v2[fk] = not bool(v2[fk])
                    flipped = True
                    break
        if not flipped:
            for n in numbers:
                fk = n.get("factKey")
                th = float(n.get("threshold") or 0)
                op = str(n.get("operator") or "LTE").upper()
                bad_v = int(round(th + 20)) if op in ("LTE", "LT") else max(0, int(round(th - 5)))
                v2[fk] = bad_v
                for alt in (n.get("altFactKeys") or []):
                    v2[alt] = bad_v
                flipped = True
                break

        out = [("VEC_LOGIC_SATISFIED", v1), ("VEC_LOGIC_COUNTEREXAMPLE", v2)]
        if outcome_mode == "PERMISSION" and any(str(b.get("polarity") or "").upper() == "CARVEOUT_TRUE" for b in booleans):
            v3 = dict(v1)
            for b in booleans:
                if str(b.get("polarity") or "").upper() == "CARVEOUT_TRUE":
                    v3[b["factKey"]] = True
            out.append(("VEC_LOGIC_CARVEOUT_BAR", v3))
        elif outcome_mode == "SANCTION_WITH_CARVEOUT":
            v3 = dict(v2)
            for b in booleans:
                if str(b.get("polarity") or "").upper() == "CARVEOUT_TRUE":
                    v3[b["factKey"]] = True
            out.append(("VEC_LOGIC_CARVEOUT_WAIVED", v3))
        return out

    def _build_computational_fact_candidates(self, ast: dict) -> list[tuple[str, dict]]:
        comp = (ast or {}).get("computation") or {}
        pipe = comp.get("pipeline") if isinstance(comp.get("pipeline"), dict) else {}
        steps = list(pipe.get("steps") or [])
        out_spec = pipe.get("output") if isinstance(pipe.get("output"), dict) else {}
        guards = list((ast or {}).get("guards") or [])
        rc = (ast or {}).get("rootCondition") or {}

        # Case A: Temporal window compound interest (e.g. MSME_DEADLINE_STYLE with randomized deadlineDays)
        if comp.get("deadlineDays") or ("elapsedDays" in out_spec and "calculatedInterestInr" in out_spec):
            dl_days = int(comp.get("deadlineDays") or 45)
            d_within = max(1, dl_days - 5)
            d_over = dl_days + 20
            return [
                ("VEC_WINDOW_COMPLIANT", {
                    "triggerDate": "2024-01-01",
                    "settlementDate": _iso_shift("2024-01-01", d_within),
                    "invoiceAmountInr": 1000000,
                    "rbiBankRatePct": 6.5,
                }),
                ("VEC_WINDOW_OVERDUE", {
                    "triggerDate": "2024-01-01",
                    "settlementDate": _iso_shift("2024-01-01", d_over),
                    "invoiceAmountInr": 1000000,
                    "rbiBankRatePct": 6.5,
                }),
            ]

        # Case B: Bounded temporal override / suspension window
        date_cmp_nodes = []
        _collect_op_nodes(steps, "DATE_COMPARE", date_cmp_nodes)
        bound_dates = sorted({
            str(n.get("right")) for n in date_cmp_nodes
            if isinstance(n.get("right"), str)
            and re.match(r"^\d{4}-\d{2}-\d{2}$", n.get("right"))
            and n.get("right") != "0001-01-01"
        })
        if ("inOverrideWindow" in out_spec or "inSuspensionWindow" in out_spec) and len(bound_dates) >= 2:
            d_lo, d_hi = date.fromisoformat(bound_dates[0]), date.fromisoformat(bound_dates[-1])
            mid_d = (d_lo + (d_hi - d_lo) // 2).isoformat()
            after_d = (d_hi + timedelta(days=30)).isoformat()
            return [
                ("VEC_BOUNDED_IN_WINDOW", {"transactionDate": mid_d, "isSellerResident": True, "hasValidPan": True}),
                ("VEC_BOUNDED_OUTSIDE_WINDOW", {"transactionDate": after_d, "isSellerResident": True, "hasValidPan": True}),
            ]

        # Case C: Progressive bracket slabs
        prog_nodes = []
        _collect_op_nodes(steps, "PROGRESSIVE_SLABS", prog_nodes)
        if prog_nodes:
            slabs = prog_nodes[0].get("slabs") or []
            lo_val = 0.0
            hi_val = 100000.0
            if slabs:
                s0 = slabs[0]
                if isinstance(s0.get("floor"), dict):
                    lo_val = _minor_str_to_major(s0["floor"].get("value", 0))
                if isinstance(s0.get("upTo"), dict):
                    hi_val = _minor_str_to_major(s0["upTo"].get("value", lo_val * 2 + 50000))
                elif isinstance(slabs[-1].get("floor"), dict):
                    hi_val = max(lo_val + 50000.0, _minor_str_to_major(slabs[-1]["floor"].get("value", 0)) * 1.5)
                else:
                    hi_val = max(lo_val * 2.0, 100000.0)
            in_amt = round(lo_val + 0.5 * max(1000.0, hi_val - lo_val), 2)
            below_amt = round(max(0.0, lo_val * 0.5), 2)
            return [
                ("VEC_SLAB_IN_BRACKET", {"primaryBasePaisa": str(int(round(in_amt * 100))), "invoiceAmount": in_amt}),
                ("VEC_SLAB_LOW", {"primaryBasePaisa": str(int(round(below_amt * 100))), "invoiceAmount": below_amt}),
            ]

        # Case D: Collection aggregate SUM_OVER / COUNT
        fs = comp.get("factSchema") or {}
        fs_colls = list(fs.get("collections") or [])
        if fs_colls or any(isinstance(st.get("expr"), dict) and ("SUM_OVER" in str(st) or "COUNT" in str(st)) for st in steps):
            fact_refs = set()
            _collect_fact_refs(pipe, fact_refs)
            coll_k = fs_colls[0] if fs_colls else next((r for r in sorted(fact_refs) if r not in _BUILT_IN_FACT_KEYS), "qualifyingClaims")
            th_amt = _minor_str_to_major(comp.get("thresholdPaisa", 0))
            if "countTotal" in out_spec:
                cnt_th = int(max(1, round(th_amt * 100 if th_amt < 1 else th_amt))) if th_amt else 2
                high_items = [{"value": 1000, "isQualifying": True} for _ in range(cnt_th + 2)]
                low_items = [{"value": 1000, "isQualifying": True} for _ in range(max(0, cnt_th - 1))]
                return [
                    ("VEC_COLL_COUNT_HIGH", {coll_k: high_items}),
                    ("VEC_COLL_COUNT_LOW", {coll_k: low_items}),
                ]
            ref_th = th_amt if th_amt > 0 else 20000.0
            high_v1 = round(ref_th * 0.9 + 5000.0, 2)
            high_v2 = round(ref_th * 0.9 + 6000.0, 2)
            low_v1 = round(ref_th * 0.25, 2)
            low_v2 = round(ref_th * 0.25, 2)
            return [
                ("VEC_COLL_SUM_HIGH", {coll_k: [{"value": high_v1, "isQualifying": True}, {"value": high_v2, "isQualifying": True}]}),
                ("VEC_COLL_SUM_LOW", {coll_k: [{"value": low_v1, "isQualifying": True}, {"value": low_v2, "isQualifying": True}]}),
            ]

        # Case E: Universal arithmetic/withholding/combinator/capped/subtraction pipelines
        fact_refs = set()
        _collect_fact_refs(pipe, fact_refs)

        th_major = _minor_str_to_major(comp.get("thresholdPaisa", 0))
        if th_major <= 0:
            rc_ex = _parse_monetary_to_inr(rc.get("applicableExcessThreshold")) or 0.0
            rc_cl = _parse_monetary_to_inr(rc.get("applicableCliffThreshold")) or 0.0
            raw_rc_th = max(rc_ex, rc_cl)
            th_major = raw_rc_th if raw_rc_th > 0 else 50000.0

        # Infer guard parameters so Vector 1 passes all preconditions reliably
        prior_to_major = 150000000.0
        tx_cutover = None
        tx_sunset = None
        has_pan_guard = False
        has_res_guard = False
        for g in guards:
            gid = str(g.get("id") or "")
            gf = str(g.get("field") or "")
            gv = g.get("value")
            if gf == "buyerPriorTurnoverPaisa":
                g_inr = _minor_str_to_major(gv)
                if g_inr > 0:
                    prior_to_major = round(g_inr * 1.5, 2)
            elif gf == "payee.isResident":
                has_res_guard = True
            elif gf == "payee.isPanValid":
                has_pan_guard = True
            elif gf == "transactionDate" and isinstance(gv, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", gv):
                if "SUNSET" in gid or g.get("operator") in ("GTE", "GT"):
                    tx_sunset = gv
                else:
                    tx_cutover = gv

        safe_date = "2024-06-15"
        if tx_cutover and tx_sunset and tx_sunset > tx_cutover:
            d0, d1 = date.fromisoformat(tx_cutover), date.fromisoformat(tx_sunset)
            safe_date = (d0 + (d1 - d0) // 2).isoformat()
        elif tx_sunset:
            safe_date = _iso_shift(tx_sunset, -60)
        elif tx_cutover:
            safe_date = _iso_shift(tx_cutover, 60)

        # Analyze var->factKey lineage inside pipeline so minuend vs subtrahend operands get distinct realistic values
        sub_nodes = []
        _collect_op_nodes(steps, "SUB", sub_nodes)
        sub_right_vars = {
            sn["right"][5:] for sn in sub_nodes
            if isinstance(sn.get("right"), str) and sn["right"].startswith("vars.")
        }
        var_to_facts = {}
        for st in steps:
            assign = st.get("assign")
            if assign:
                f_set = set()
                _collect_fact_refs(st.get("expr"), f_set)
                var_to_facts[assign] = f_set

        subtrahend_facts = set()
        for vn in sub_right_vars:
            for fk in var_to_facts.get(vn, ()):
                if fk not in _BUILT_IN_FACT_KEYS:
                    subtrahend_facts.add(fk)
        # Map any prefix-base variable names like opBPaisa_b0 -> opBPaisa
        for vn in list(var_to_facts.keys()):
            base_vn = re.sub(r"_b\d+$", "", vn)
            if base_vn != vn and base_vn in sub_right_vars:
                for fk in var_to_facts[vn]:
                    if fk not in _BUILT_IN_FACT_KEYS:
                        subtrahend_facts.add(fk)

        extra_facts_dyn = sorted(
            fk for fk in fact_refs
            if fk not in _BUILT_IN_FACT_KEYS and "." not in fk
        )

        # Also check any non-builtin monetary precondition guards
        guard_extra_high = {}
        for g in guards:
            gf = str(g.get("field") or "")
            if gf and gf not in _BUILT_IN_FACT_KEYS and "." not in gf:
                gv = g.get("value")
                if g.get("valueType") == "CURRENCY_MINOR_UNITS":
                    guard_extra_high[gf] = round(_minor_str_to_major(gv) * 1.5, 2)
                elif isinstance(gv, (int, float)):
                    guard_extra_high[gf] = float(gv) + 10.0

        rate_ratio = float(comp.get("standardRateBps") or 0) / 10000.0
        crossover_amt = (th_major / rate_ratio) if rate_ratio > 0 else th_major
        high_amt = round(max(th_major * 1.5, crossover_amt * 1.5, 100000.0), 2)
        sub_amt = round(max(th_major * 0.25, 2000.0), 2)
        if sub_amt >= high_amt:
            sub_amt = round(high_amt * 0.25, 2)
        low_amt = round(max(0.0, th_major * 0.3), 2)

        def _make_fact_dict(inv_amt: float, prior_amt: float, sub_val: float, dt: str, buyer_to: float, has_pan: bool) -> dict:
            d = {
                "transactionDate": dt,
                "buyerPriorYearTurnoverInr": buyer_to,
                "isSellerResident": True,
                "hasValidPan": has_pan,
                "cumulativePurchasesPriorInr": prior_amt,
                "currentInvoiceAmountInr": inv_amt,
                "invoiceAmount": inv_amt,
                "primaryBasePaisa": str(int(round(inv_amt * 100))),
            }
            for fk in extra_facts_dyn:
                d[fk] = sub_val if fk in subtrahend_facts else inv_amt
            for k, v in guard_extra_high.items():
                d[k] = v
            return d

        f_high = _make_fact_dict(
            inv_amt=high_amt,
            prior_amt=round(th_major * 0.2, 2) if rc.get("nodeType") == "TRANSACTIONAL_WITHHOLDING" else 0.0,
            sub_val=sub_amt,
            dt=safe_date,
            buyer_to=prior_to_major,
            has_pan=True,
        )
        f_low = _make_fact_dict(
            inv_amt=low_amt,
            prior_amt=0.0,
            sub_val=round(low_amt * 0.2, 2),
            dt=safe_date,
            buyer_to=prior_to_major,
            has_pan=True,
        )
        candidates = [("VEC_COMP_ABOVE_OR_ACTIVE", f_high), ("VEC_COMP_LOW_OR_BELOW", f_low)]
        if has_pan_guard:
            f_nonpan = dict(f_high)
            f_nonpan["hasValidPan"] = False
            candidates.append(("VEC_COMP_NON_PAN", f_nonpan))
        if tx_sunset:
            f_sunset = dict(f_high)
            f_sunset["transactionDate"] = _iso_shift(tx_sunset, 15)
            candidates.append(("VEC_COMP_POST_SUNSET", f_sunset))
        if has_res_guard:
            f_nonres = dict(f_high)
            f_nonres["isSellerResident"] = False
            candidates.append(("VEC_COMP_NON_RESIDENT", f_nonres))
        return candidates[:4]

    def _build_deontic_gate_fact_candidates(self, ast: dict) -> list[tuple[str, dict]]:
        rc = (ast or {}).get("rootCondition") or {}
        dg = (ast or {}).get("deonticGate") or ((ast or {}).get("computation") or {}).get("deonticGate") or {}
        guards = list((ast or {}).get("guards") or [])
        node_type = rc.get("nodeType", "DEONTIC_STATE_GATE")

        # Handle guard constraints (e.g. transactionDate sunset or turnover precondition)
        tx_date = "2024-06-15"
        sunset_iso = None
        for g in guards:
            if g.get("field") == "transactionDate" and isinstance(g.get("value"), str) and re.match(r"^\d{4}-\d{2}-\d{2}$", g["value"]):
                if "SUNSET" in str(g.get("id") or "") or g.get("operator") in ("GTE", "GT"):
                    sunset_iso = g["value"]
                    tx_date = _iso_shift(sunset_iso, -30)

        if node_type == "STATUS_DISQUALIFICATION_GATE":
            req_ev = " ".join(str(x).lower() for x in (rc.get("requiredDocumentChecklist") or []))
            f_bad = {
                "transactionDate": tx_date,
                "hasConvictionMoralTurpitude": True,
                "isUndischargedInsolvent": False,
                "isWilfulDefaulter": True,
                "hasFailedAnnualFilings3Years": False,
                "hasSubmittedFormDIR8": False,
                "hasSubmittedAffidavit29A": False,
            }
            f_clean = {
                "transactionDate": tx_date,
                "hasConvictionMoralTurpitude": False,
                "isUndischargedInsolvent": False,
                "isWilfulDefaulter": False,
                "hasFailedAnnualFilings3Years": False,
                "hasSubmittedFormDIR8": True,
                "hasSubmittedAffidavit29A": True,
                "hasControllerAttestation": True,
            }
            if "dir" not in req_ev and "29a" not in req_ev:
                f_bad["isWilfulDefaulter"] = True
            return [("VEC_GATE_DISQUALIFIED", f_bad), ("VEC_GATE_ELIGIBLE", f_clean)]

        if node_type == "CORPORATE_APPROVAL_GATE":
            th_amt = max(_parse_monetary_to_inr(rc.get("applicableExcessThreshold")) or 0.0, 100000000.0)
            return [
                ("VEC_GATE_APPROVAL_MISSING", {
                    "transactionDate": tx_date,
                    "isRelatedParty": True,
                    "transactionAmountInr": th_amt * 1.5,
                    "exceedsTenPctTurnover": True,
                    "buyerPriorYearTurnoverInr": 5000000000,
                    "isOrdinaryCourseOfBusiness": False,
                    "isArmLengthPrice": False,
                    "hasBoardConsent": True,
                    "hasShareholderSpecialResolution": False,
                }),
                ("VEC_GATE_APPROVAL_OR_EXEMPT_OK", {
                    "transactionDate": tx_date,
                    "isRelatedParty": True,
                    "transactionAmountInr": th_amt * 1.5,
                    "exceedsTenPctTurnover": True,
                    "buyerPriorYearTurnoverInr": 5000000000,
                    "isOrdinaryCourseOfBusiness": True,
                    "isArmLengthPrice": True,
                    "hasBoardConsent": True,
                    "hasShareholderSpecialResolution": True,
                }),
            ]

        if node_type == "PROCEDURAL_NOTICE_GATE":
            days = int(dg.get("noticeWindowDays") or 60)
            d_premature = max(1, days // 3)
            d_ready = days + 15
            return [
                ("VEC_NOTICE_PREMATURE", {
                    "transactionDate": tx_date,
                    "noticeDeliveryDate": "2024-01-01",
                    "suitFilingDate": _iso_shift("2024-01-01", d_premature),
                    "isUrgentInterimReliefSought": False,
                }),
                ("VEC_NOTICE_SATISFIED", {
                    "transactionDate": tx_date,
                    "noticeDeliveryDate": "2024-01-01",
                    "suitFilingDate": _iso_shift("2024-01-01", d_ready),
                    "isUrgentInterimReliefSought": False,
                }),
            ]

        if node_type == "BOOLEAN_PROHIBITION_GATE":
            if sunset_iso:
                return [
                    ("VEC_PROHIB_PRE_SUNSET", {"transactionDate": _iso_shift(sunset_iso, -30), "isBlocked": False}),
                    ("VEC_PROHIB_POST_SUNSET", {"transactionDate": _iso_shift(sunset_iso, 30), "isBlocked": False}),
                ]
            return [
                ("VEC_PROHIB_TRIGGERED", {
                    "transactionDate": tx_date,
                    "isMotorVehicle": True,
                    "seatingCapacity": 5,
                    "isUsedForPassengerTransport": False,
                    "isUsedForTaxableSupplyOfVehicles": False,
                    "isUsedForDrivingSchool": False,
                }),
                ("VEC_PROHIB_CARVEOUT_CLEAR", {
                    "transactionDate": tx_date,
                    "isMotorVehicle": False,
                    "seatingCapacity": 20,
                    "isUsedForPassengerTransport": True,
                    "isUsedForTaxableSupplyOfVehicles": True,
                    "isUsedForDrivingSchool": True,
                }),
            ]

        if node_type == "SUBJECTIVE_EVIDENCE_GATE":
            checklist = list((ast or {}).get("attestationChecklist") or rc.get("requiredDocumentChecklist") or [])
            docs = [str(c) for c in checklist]
            return [
                ("VEC_EVIDENCE_ATTESTED", {"transactionDate": tx_date, "uploadedDocuments": docs, "hasControllerAttestation": True}),
                ("VEC_EVIDENCE_MISSING", {"transactionDate": tx_date, "uploadedDocuments": [], "hasControllerAttestation": False}),
            ]

        if node_type == "TEMPORAL_WINDOW_GATE":
            dl_days = _parse_duration_days(rc.get("applicableDeadlineWindow") or rc.get("durationWindow")) or 45
            d_within = max(1, dl_days - 5)
            d_over = dl_days + 20
            return [
                ("VEC_WINDOW_COMPLIANT", {
                    "transactionDate": tx_date,
                    "triggerDate": "2020-01-01",
                    "settlementDate": _iso_shift("2020-01-01", d_within),
                    "invoiceAmountInr": 1000000,
                    "rbiBankRatePct": 6.5,
                }),
                ("VEC_WINDOW_OVERDUE", {
                    "transactionDate": tx_date,
                    "triggerDate": "2020-01-01",
                    "settlementDate": _iso_shift("2020-01-01", d_over),
                    "invoiceAmountInr": 1000000,
                    "rbiBankRatePct": 6.5,
                }),
            ]

        # Generic deontic / accounting / measurement gate
        carveout_keys = [
            c.get("factKey") for c in (dg.get("exceptionCarveouts") or [])
            if isinstance(c, dict) and c.get("factKey")
        ]
        trigger_keys = [
            t.get("factKey") for t in (dg.get("disqualificationTriggers") or [])
            if isinstance(t, dict) and t.get("factKey")
        ]
        approval_keys = [
            a.get("factKey") for a in (dg.get("mandatoryApprovals") or [])
            if isinstance(a, dict) and a.get("factKey")
        ]
        f_ok = {
            "transactionDate": tx_date,
            "buyerPriorYearTurnoverInr": 5000000000,
            "hasControllerAttestation": True,
            "continuousServiceYears": 10,
            "leaseTermMonths": 6,
        }
        f_alt = {
            "transactionDate": tx_date,
            "buyerPriorYearTurnoverInr": 1000000,
            "hasControllerAttestation": False,
            "continuousServiceYears": 1,
            "leaseTermMonths": 36,
        }
        for ck in carveout_keys:
            f_ok[ck] = True
            f_alt[ck] = False
        for tk in trigger_keys:
            f_ok[tk] = False
            f_alt[tk] = True
        for ak in approval_keys:
            f_ok[ak] = True
            f_alt[ak] = False
        return [("VEC_GENERIC_GATE_ATTESTED", f_ok), ("VEC_GENERIC_GATE_UNATTESTED", f_alt)]

    def build_sample_vectors(self, gold_ast: dict, stype: str = None) -> list[dict]:
        """
        Synthesizes 2-4 deterministic counterfactual fact vectors tailored directly
        to this training sample's gold AST (`factSchema`, `pipeline.steps`, `guards`,
        `deonticGate`), verifies expected outputs via the canonical JS USVM kernel,
        and returns strict expected outputs without any smoke-only pass-throughs.
        """
        if not isinstance(gold_ast, dict) or not gold_ast:
            return self._select_vectors(stype=stype)

        comp = gold_ast.get("computation") or {}
        pipe = comp.get("pipeline") if isinstance(comp.get("pipeline"), dict) else None
        is_pure_logic = (
            (gold_ast.get("rootCondition") or {}).get("nodeType") == "GENERAL_RULE_GATE"
            or comp.get("archetype") == "PURE_LOGIC_PIPELINE"
        )

        if is_pure_logic and pipe is not None:
            cand_list = self._build_pure_logic_fact_candidates(gold_ast)
        elif pipe is not None or comp.get("archetype") in ("FLAT_RATE", "EXCESS_OVER_THRESHOLD", "CLIFF_THRESHOLD", "DECLARATIVE_PIPELINE"):
            cand_list = self._build_computational_fact_candidates(gold_ast)
        else:
            cand_list = self._build_deontic_gate_fact_candidates(gold_ast)

        if not cand_list:
            return self._select_vectors(stype=stype)

        batch_items = [{"ast": gold_ast, "facts": f_dict} for _, f_dict in cand_list]
        try:
            gold_outputs = self.evaluator.evaluate_batch(batch_items)
        except Exception:
            return self._select_vectors(stype=stype)

        vectors = []
        for (vec_id, f_dict), gold_res in zip(cand_list, gold_outputs):
            if not isinstance(gold_res, dict) or gold_res.get("error"):
                continue
            expected = _extract_expected_outcome(gold_res, gold_ast)
            if expected:
                vectors.append({
                    "id": vec_id,
                    "description": f"Counterfactual oracle vector for {stype or gold_ast.get('sectionId', 'RULE')}",
                    "facts": f_dict,
                    "expected": expected,
                })

        if vectors:
            return vectors
        return self._select_vectors(stype=stype)

    def _select_vectors(self, vectors: list[dict] = None, stype: str = None) -> list[dict]:
        if vectors:
            return vectors
        if stype and stype in STYPE_VECTOR_MAP and STYPE_VECTOR_MAP[stype] in DOMAIN_TEST_VECTORS:
            return DOMAIN_TEST_VECTORS[STYPE_VECTOR_MAP[stype]]
        if stype and stype in DOMAIN_TEST_VECTORS:
            return DOMAIN_TEST_VECTORS[stype]
        return CANONICAL_TEST_VECTORS

    def _score_single_candidate(self, ast: dict, test_vectors: list[dict], eval_outputs: list[dict]) -> tuple[float, dict]:
        breakdown = {
            "r_grammar": 0.0,
            "r_types": 0.0,
            "r_invariants": 0.0,
            "r_vectors": 0.0,
            "vector_passes": 0,
            "vector_total": len(test_vectors)
        }

        if not ast.get("version"):
            breakdown["r_grammar"] -= 2.0
        else:
            breakdown["r_grammar"] += 1.0

        invariants = ast.get("statutoryInvariants", {})
        if invariants.get("isAuthorizedToDisburse") is True or ast.get("isAuthorizedToDisburse") is True:
            breakdown["r_invariants"] -= 100.0
            return -100.0, breakdown
        breakdown["r_invariants"] += 2.0

        root = ast.get("rootCondition", {})
        node_type = root.get("nodeType", "TRANSACTIONAL_WITHHOLDING")
        comp = ast.get("computation") or {}
        comp_arch = comp.get("archetype", "")
        has_pipeline_steps = bool(isinstance(comp.get("pipeline"), dict) and comp["pipeline"].get("steps"))

        if node_type == "GENERAL_RULE_GATE" or comp_arch == "PURE_LOGIC_PIPELINE":
            if has_pipeline_steps:
                breakdown["r_types"] += 2.0
                if comp.get("factSchema"):
                    breakdown["r_types"] += 1.0
            elif ast.get("deonticGate"):
                breakdown["r_types"] += 1.5
            else:
                breakdown["r_types"] -= 2.0
        elif node_type == "TRANSACTIONAL_WITHHOLDING":
            if comp_arch == "DECLARATIVE_PIPELINE" and has_pipeline_steps:
                breakdown["r_types"] += 2.0
                if comp.get("standardRateBps", 0) > 0 or comp.get("thresholdPaisa", "0") != "0":
                    breakdown["r_types"] += 1.0
            else:
                has_rate = bool(root.get("standardTaxRate"))
                has_threshold = bool(root.get("applicableExcessThreshold") or root.get("applicableCliffThreshold"))
                breakdown["r_types"] += 1.0 if has_rate else -2.0
                if has_threshold:
                    breakdown["r_types"] += 1.0
                std_rate = root.get("standardTaxRate")
                penal_rate = root.get("nonPanPenalRate")
                if std_rate and penal_rate and std_rate != penal_rate:
                    breakdown["r_types"] += 1.5
        else:
            breakdown["r_types"] += 2.0
            if (
                has_pipeline_steps
                or root.get("prohibitionPredicate")
                or root.get("applicableDeadlineWindow")
                or root.get("obligationPredicate")
                or root.get("valuationFormula")
                or root.get("recognitionPredicate")
                or root.get("electionPredicate")
                or root.get("presentationPredicate")
                or root.get("subjectiveCriterion")
            ):
                breakdown["r_types"] += 1.0

        vec_score = 0.0
        for vec, res in zip(test_vectors, eval_outputs):
            if res.get("error"):
                vec_score -= 5.0
                continue
            if res.get("isAuthorizedToDisburse") is True:
                breakdown["r_invariants"] -= 100.0
                return -100.0, breakdown

            expected = vec.get("expected") or {}
            if not expected:
                vec_score -= 2.0
                continue

            all_match = True
            matched_keys = 0
            total_keys = 0

            for k in OUTCOME_NUMERIC_KEYS:
                if k in expected:
                    total_keys += 1
                    exp_v = float(expected[k])
                    calc_v = float(res.get(k, 0.0) or 0.0)
                    delta = abs(calc_v - exp_v)
                    if delta < 0.01:
                        matched_keys += 1
                        vec_score += 2.5
                    else:
                        all_match = False
                        penalty = min(4.0, 1.0 + math.log(1.0 + delta / 100.0))
                        vec_score -= penalty

            for k in OUTCOME_INT_KEYS:
                if k in expected:
                    total_keys += 1
                    if int(round(float(res.get(k, 0) or 0))) == int(round(float(expected[k]))):
                        matched_keys += 1
                        vec_score += 0.75
                    else:
                        all_match = False
                        vec_score -= 1.0

            for k in OUTCOME_BOOL_OR_STR_KEYS:
                if k in expected:
                    total_keys += 1
                    if res.get(k) == expected[k]:
                        matched_keys += 1
                        vec_score += 1.0
                    else:
                        all_match = False
                        vec_score -= 1.5

            if total_keys == 0:
                all_match = False
                vec_score -= 2.0

            if all_match:
                breakdown["vector_passes"] += 1
                vec_score += 1.5

        breakdown["r_vectors"] = vec_score
        total_reward = (
            breakdown["r_grammar"]
            + breakdown["r_types"]
            + breakdown["r_invariants"]
            + breakdown["r_vectors"]
        )
        return total_reward, breakdown

    def compute_reward(self, ast: dict, vectors: list[dict] = None, stype: str = None) -> tuple[float, dict]:
        """
        Executes AST against verification vectors via a single batched IPC call to
        the persistent Node JS USVM worker and returns total scalar reward.
        """
        test_vectors = self._select_vectors(vectors=vectors, stype=stype)
        batch_items = [{"ast": ast, "facts": vec["facts"]} for vec in test_vectors]
        try:
            eval_outputs = self.evaluator.evaluate_batch(batch_items)
        except Exception as e:
            eval_outputs = [{"error": str(e)} for _ in test_vectors]
        return self._score_single_candidate(ast, test_vectors, eval_outputs)

    def compute_reward_group(self, ast_list: list[dict], vectors: list[dict] = None, stype: str = None) -> list[tuple[float, dict]]:
        """
        Evaluates an entire GRPO group of candidate ASTs in ONE persistent
        Node NDJSON batch request to eliminate IPC per-candidate overhead.
        """
        test_vectors = self._select_vectors(vectors=vectors, stype=stype)
        batch_items = []
        for ast in ast_list:
            for vec in test_vectors:
                batch_items.append({"ast": ast, "facts": vec["facts"]})
        try:
            all_outputs = self.evaluator.evaluate_batch(batch_items)
        except Exception as e:
            all_outputs = [{"error": str(e)} for _ in batch_items]

        n_vecs = len(test_vectors)
        results = []
        for i, ast in enumerate(ast_list):
            sub_out = all_outputs[i * n_vecs:(i + 1) * n_vecs]
            results.append(self._score_single_candidate(ast, test_vectors, sub_out))
        return results
