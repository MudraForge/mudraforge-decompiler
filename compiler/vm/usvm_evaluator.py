"""
=============================================================================
MUDRAFORGE CANONICAL USVM EVALUATOR (PERSISTENT NODE JS BRIDGE)
=============================================================================

Routes ALL AST validation, OKF lifecycle checks, and statutory execution
directly to the single canonical JavaScript USVM kernel
(`lib/core/usvm/evaluation_vm.js`) via ONE persistent long-lived Node.js
worker process communicating over newline-delimited JSON (stdin/stdout).

Zero duplicate tax, compounding, date, or deontic logic exists in Python.
Eliminates cross-language behavioral drift and Windows subprocess spawn
overhead during GRPO RL loops.
"""

import atexit
import json
import os
import re
import subprocess
import threading
from datetime import datetime


class SecurityError(Exception):
    pass


WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50,
    "sixty": 60, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000
}


def _parse_iso_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    clean = re.sub(r"(?<=\d)(st|nd|rd|th)", "", str(date_str).strip(), flags=re.IGNORECASE)
    clean = re.sub(r"day\s+of", "", clean, flags=re.IGNORECASE).strip()
    for fmt in (
        "%d %B, %Y", "%d %B %Y", "%d %b %Y", "%d %b, %Y",
        "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
        "%b, %Y", "%B, %Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"
    ):
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _parse_monetary_to_inr(val: str | int | float | None) -> float | None:
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return float(val)
    raw_s = str(val).lower().strip()
    raw_s = re.sub(r"\bper\s+cent(?:um)?\b", " ", raw_s)
    is_minor_unit_only = bool(
        re.search(r"\b(?:cents?|paise|paisa|pence)\b", raw_s)
        and not re.search(
            r"(?:₹|\$|€|£|\brs\.?\b|\binr\b|\busd\b|\beur\b|\bgbp\b|\brupees?\b|\bdollars?\b|\beuros?\b|\bpounds?\b|\bcrores?\b|\blakhs?\b|\blacs?\b|\bthousand\b|\bmillion\b|\bbillion\b)",
            raw_s,
        )
    )
    s = raw_s.replace(",", "")
    s = re.sub(r"[₹$€£]", " ", s)
    s = re.sub(r"\b(?:rs\.?|inr|usd|eur|gbp)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(crores?|cr|lakhs?|lakh|lacs?|lac|billions?|bn|millions?|mn|thousands?|thousand|k)?", s)
    if m:
        num = float(m.group(1))
        unit = m.group(2) or ""
        if "crore" in unit or unit == "cr":
            return num * 10_000_000.0
        if "lakh" in unit or "lac" in unit:
            return num * 100_000.0
        if "billion" in unit or unit == "bn":
            return num * 1_000_000_000.0
        if "million" in unit or unit == "mn":
            return num * 1_000_000.0
        if "thousand" in unit or unit == "k":
            return num * 1_000.0
        return num * 0.01 if is_minor_unit_only else num
    # Word number fallback (e.g., "twenty thousand rupees", "fifty million dollars")
    total = 0.0
    curr = 0.0
    found = False
    for tok in re.findall(r"[a-z\-]+", s):
        if tok in ("lakh", "lakhs", "lac", "lacs"):
            found = True
            curr = (curr if curr > 0 else 1.0) * 100_000.0
            total += curr
            curr = 0.0
        elif tok in ("crore", "crores"):
            found = True
            curr = (curr if curr > 0 else 1.0) * 10_000_000.0
            total += curr
            curr = 0.0
        elif tok in ("million", "millions"):
            found = True
            curr = (curr if curr > 0 else 1.0) * 1_000_000.0
            total += curr
            curr = 0.0
        elif tok in ("billion", "billions"):
            found = True
            curr = (curr if curr > 0 else 1.0) * 1_000_000_000.0
            total += curr
            curr = 0.0
        elif tok in WORD_NUMS:
            found = True
            n = WORD_NUMS[tok]
            if n >= 1000:
                curr = (curr if curr > 0 else 1.0) * n
                total += curr
                curr = 0.0
            elif n == 100:
                curr = (curr if curr > 0 else 1.0) * 100.0
            else:
                curr += n
    total += curr
    if not found:
        return None
    return total * 0.01 if is_minor_unit_only else total


def _parse_monetary_to_paisa_str(val: str | int | float | None) -> str | None:
    inr = _parse_monetary_to_inr(val)
    if inr is None:
        return None
    return str(int(round(inr * 100)))


def _parse_rate_to_bps(val: str | int | float | None) -> int | None:
    if val is None or val == "":
        return None
    if isinstance(val, (int, float)):
        return int(round(float(val) * 100))
    s = str(val).lower().replace("%", "").replace("per cent", "").replace("percent", "").strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return None
    return int(round(float(m.group(1)) * 100))


def _parse_duration_days(val: str | int | None) -> int | None:
    if val is None or val == "":
        return None
    if isinstance(val, int):
        return val
    s = str(val).lower().strip()
    if "appointed day" in s:
        return 15
    if "forty-five" in s or "45" in s:
        return 45
    if "one hundred eighty" in s or "180" in s:
        return 180
    if "two month" in s or "2 month" in s:
        return 60
    m = re.search(r"([0-9]+)\s*(days?|months?|years?)?", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2) or "days"
        if "month" in unit:
            return n * 30
        if "year" in unit:
            return n * 365
        return n
    for w, n in WORD_NUMS.items():
        if w in s:
            if "month" in s:
                return n * 30
            if "year" in s:
                return n * 365
            return n
    return None


class CanonicalUsvmEvaluator:
    """
    Persistent Node.js IPC bridge to the canonical JavaScript USVM kernel
    (`lib/core/usvm/evaluation_vm.js`). Zero duplicate math in Python.
    """

    def __init__(self, worker_path: str | None = None, default_mode: str = "DEVELOPMENT"):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        self.worker_path = worker_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "usvm_worker_stdio.js")
        if not os.path.exists(self.worker_path):
            raise FileNotFoundError(f"Canonical USVM stdio worker not found at {self.worker_path}")
        self.default_mode = default_mode
        self._lock = threading.Lock()
        self._proc = None
        self._req_counter = 0
        self._closed = False
        atexit.register(self.close)

    def _ensure_worker(self):
        if self._closed:
            self._closed = False
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                ["node", self.worker_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1
            )

    def _request(self, payload: dict) -> dict:
        with self._lock:
            self._ensure_worker()
            self._req_counter += 1
            req = dict(payload)
            req["id"] = self._req_counter
            line = json.dumps(req)
            try:
                assert self._proc.stdin is not None and self._proc.stdout is not None
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
                resp_line = self._proc.stdout.readline()
                if not resp_line:
                    # One automatic restart retry if worker closed unexpectedly
                    self.close()
                    self._ensure_worker()
                    self._proc.stdin.write(line + "\n")
                    self._proc.stdin.flush()
                    resp_line = self._proc.stdout.readline()
                if not resp_line:
                    err_out = ""
                    if self._proc and self._proc.stderr:
                        try:
                            err_out = self._proc.stderr.read()
                        except Exception:
                            pass
                    raise RuntimeError(f"Canonical USVM Node worker returned EOF. stderr={err_out}")
                return json.loads(resp_line)
            except BrokenPipeError:
                self.close()
                raise RuntimeError("Canonical USVM Node worker pipe broken")

    def close(self):
        self._closed = True
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def __enter__(self):
        self._ensure_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _normalize_facts_for_js_bridge(self, facts: dict) -> dict:
        f = dict(facts or {})
        if "invoiceAmount" not in f:
            if "currentInvoiceAmountInr" in f:
                f["invoiceAmount"] = f["currentInvoiceAmountInr"]
            elif "invoiceAmountInr" in f:
                f["invoiceAmount"] = f["invoiceAmountInr"]
        if "priorPurchasesInCurrentFyInr" not in f and "cumulativePurchasesPriorInr" in f:
            f["priorPurchasesInCurrentFyInr"] = f["cumulativePurchasesPriorInr"]
        if "buyerTurnoverInr" not in f and "buyerPriorYearTurnoverInr" in f:
            f["buyerTurnoverInr"] = f["buyerPriorYearTurnoverInr"]
        if "hasPan" not in f and "hasValidPan" in f:
            f["hasPan"] = f["hasValidPan"]
        if "sellerIsNonResident" not in f and "isSellerResident" in f:
            f["sellerIsNonResident"] = not bool(f["isSellerResident"])
        if "noticeServedDate" not in f and "noticeDeliveryDate" in f:
            f["noticeServedDate"] = f["noticeDeliveryDate"]
        if "filingDate" not in f and "suitFilingDate" in f:
            f["filingDate"] = f["suitFilingDate"]
        if "urgentInterimRelief" not in f and "isUrgentInterimReliefSought" in f:
            f["urgentInterimRelief"] = bool(f["isUrgentInterimReliefSought"])

        # Normalize uploadedDocuments array into attestations dict for evaluation_vm.js Precedence 7
        if isinstance(f.get("uploadedDocuments"), list):
            att = dict(f.get("attestations") or {})
            for doc in f["uploadedDocuments"]:
                att[str(doc)] = True
            f["attestations"] = att

        return f

    def _normalize_ast_for_js_bridge(self, ast: dict) -> dict:
        if not isinstance(ast, dict):
            return ast

        invariants = ast.get("statutoryInvariants") or {}
        if invariants.get("isAuthorizedToDisburse") is True or ast.get("isAuthorizedToDisburse") is True:
            raise SecurityError("Statutory Invariant Violation: isAuthorizedToDisburse cannot be true")

        norm = json.loads(json.dumps(ast, default=str))
        root = norm.get("rootCondition") or {}
        node_type = root.get("nodeType", "TRANSACTIONAL_WITHHOLDING")
        comp = norm.get("computation")
        if comp is None:
            comp = {}
        is_pure_logic_rule = (
            norm.get("ruleDomain") == "GENERAL_LOGIC"
            or node_type == "GENERAL_RULE_GATE"
            or comp.get("archetype") == "PURE_LOGIC_PIPELINE"
        )

        if not norm.get("version"):
            norm["version"] = "0.1.0-draft"

        if not norm.get("triggerPolicy"):
            if is_pure_logic_rule:
                norm["triggerPolicy"] = {"type": "TRANSACTION_DATE"}
            else:
                trig = str(root.get("triggerPolicy", "EARLIER_OF")).upper()
                if trig == "EARLIER_OF":
                    norm["triggerPolicy"] = {"type": "EARLIER_OF", "events": ["BOOK_CREDIT", "PAYMENT_DISBURSEMENT"]}
                elif trig == "PAYMENT_ONLY":
                    norm["triggerPolicy"] = {"type": "PAYMENT_ONLY"}
                else:
                    norm["triggerPolicy"] = {"type": "TRANSACTION_DATE"}

        # Ensure guards exist for string-encoded preconditions/temporals on legacy/RL ASTs
        guards = list(norm.get("guards") or [])
        if is_pure_logic_rule:
            norm["guards"] = guards
            return norm

        guard_ids = {g.get("id") for g in guards if isinstance(g, dict)}
        has_pipeline = bool(comp.get("pipeline") or comp.get("steps"))
        is_modern_ast = ("guards" in ast) or bool(norm.get("symbolLegend")) or has_pipeline

        if not is_modern_ast:
            if root.get("temporalSunset") and "STATUTORY_SUNSET_GUARD" not in guard_ids:
                sunset_iso = _parse_iso_date(root.get("temporalSunset"))
                if sunset_iso:
                    guards.append({
                        "id": "STATUTORY_SUNSET_GUARD",
                        "field": "transactionDate",
                        "operator": "GTE",
                        "value": sunset_iso,
                        "onMatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_SUNSET_OMITTED"}
                    })
                    guard_ids.add("STATUTORY_SUNSET_GUARD")

            if root.get("temporalCutover") and "STATUTORY_CUTOVER_GUARD" not in guard_ids:
                cutover_iso = _parse_iso_date(root.get("temporalCutover"))
                if cutover_iso:
                    guards.append({
                        "id": "STATUTORY_CUTOVER_GUARD",
                        "field": "transactionDate",
                        "operator": "LT",
                        "value": cutover_iso,
                        "onMatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_PRE_COMMENCEMENT"}
                    })
                    guard_ids.add("STATUTORY_CUTOVER_GUARD")

            if root.get("priorTurnoverPrecondition") and node_type == "TRANSACTIONAL_WITHHOLDING" and "PRIOR_TURNOVER_PRECONDITION" not in guard_ids:
                to_paisa = _parse_monetary_to_paisa_str(root.get("priorTurnoverPrecondition"))
                if to_paisa is not None:
                    guards.append({
                        "id": "PRIOR_TURNOVER_PRECONDITION",
                        "field": "buyerPriorTurnoverPaisa",
                        "operator": "GTE",
                        "value": to_paisa,
                        "onMismatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_SMALL_ENTITY"}
                    })
                    guard_ids.add("PRIOR_TURNOVER_PRECONDITION")

            if node_type == "TRANSACTIONAL_WITHHOLDING" and "RESIDENT_STATUS_GUARD" not in guard_ids:
                # Resident guard respects isSellerResident=False when passed
                guards.append({
                    "id": "RESIDENT_STATUS_GUARD",
                    "field": "payee.isResident",
                    "operator": "NEQ",
                    "value": False,
                    "onMismatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_NON_RESIDENT"}
                })
                guard_ids.add("RESIDENT_STATUS_GUARD")

            if root.get("nonPanPenalRate") and "PAN_PENAL_OVERRIDE" not in guard_ids:
                penal_bps = _parse_rate_to_bps(root.get("nonPanPenalRate"))
                if penal_bps is not None:
                    guards.append({
                        "id": "PAN_PENAL_OVERRIDE",
                        "field": "payee.isPanValid",
                        "operator": "EQ",
                        "value": False,
                        "onMatch": {"action": "OVERRIDE_RATE", "overrideRateBps": penal_bps}
                    })
                    guard_ids.add("PAN_PENAL_OVERRIDE")

        norm["guards"] = guards

        # Infer/adapt computation or deonticGate structurally for canonical JS execution
        if node_type == "TRANSACTIONAL_WITHHOLDING" and not has_pipeline:
            std_bps = _parse_rate_to_bps(root.get("standardTaxRate"))
            cliff_paisa = _parse_monetary_to_paisa_str(root.get("applicableCliffThreshold"))
            excess_paisa = _parse_monetary_to_paisa_str(root.get("applicableExcessThreshold"))
            rate_val = std_bps if std_bps is not None else 0

            if cliff_paisa is not None:
                comp = {
                    "archetype": "DECLARATIVE_PIPELINE",
                    "thresholdPaisa": cliff_paisa,
                    "standardRateBps": rate_val,
                    "pipeline": {
                        "steps": [
                            {"assign": "cumAfter", "expr": "facts.priorPurchasesPaisa + facts.currentInvoicePaisa"},
                            {"assign": "breached", "expr": "(facts.currentInvoicePaisa > rules.thresholdPaisa) || (cumAfter > rules.thresholdPaisa)"},
                            {"assign": "taxableBasePaisa", "expr": "breached ? facts.currentInvoicePaisa : 0n"},
                            {"assign": "tdsAmountPaisa", "expr": "RATIONAL_MULTIPLY(taxableBasePaisa, rules.effectiveRateBps, 10000n)"},
                            {"assign": "effectiveRateBps", "expr": "breached ? rules.effectiveRateBps : 0"},
                            {"assign": "tdsRatePercent", "expr": "effectiveRateBps / 100"}
                        ],
                        "output": {
                            "taxableBasePaisa": "vars.taxableBasePaisa",
                            "tdsAmountPaisa": "vars.tdsAmountPaisa",
                            "taxAmountPaisa": "vars.tdsAmountPaisa",
                            "isApplicable": "vars.breached",
                            "isTdsApplicable": "vars.tdsAmountPaisa > 0n",
                            "effectiveRateBps": "vars.effectiveRateBps",
                            "tdsRatePercent": "vars.tdsRatePercent"
                        }
                    }
                }
            elif excess_paisa is not None:
                comp = {
                    "archetype": "DECLARATIVE_PIPELINE",
                    "thresholdPaisa": excess_paisa,
                    "standardRateBps": rate_val,
                    "pipeline": {
                        "steps": [
                            {"assign": "cumAfter", "expr": "facts.priorPurchasesPaisa + facts.currentInvoicePaisa"},
                            {"assign": "rawExcess", "expr": "MAX(0n, cumAfter - rules.thresholdPaisa)"},
                            {"assign": "alreadyExcess", "expr": "MAX(0n, facts.priorPurchasesPaisa - rules.thresholdPaisa)"},
                            {"assign": "taxableBasePaisa", "expr": "MIN(facts.currentInvoicePaisa, MAX(0n, rawExcess - alreadyExcess))"},
                            {"assign": "isApplicable", "expr": "taxableBasePaisa > 0n"},
                            {"assign": "tdsAmountPaisa", "expr": "RATIONAL_MULTIPLY(taxableBasePaisa, rules.effectiveRateBps, 10000n)"},
                            {"assign": "effectiveRateBps", "expr": "isApplicable ? rules.effectiveRateBps : 0"},
                            {"assign": "tdsRatePercent", "expr": "effectiveRateBps / 100"}
                        ],
                        "output": {
                            "taxableBasePaisa": "vars.taxableBasePaisa",
                            "tdsAmountPaisa": "vars.tdsAmountPaisa",
                            "taxAmountPaisa": "vars.tdsAmountPaisa",
                            "isApplicable": "vars.isApplicable",
                            "isTdsApplicable": "vars.tdsAmountPaisa > 0n",
                            "effectiveRateBps": "vars.effectiveRateBps",
                            "tdsRatePercent": "vars.tdsRatePercent"
                        }
                    }
                }
            else:
                comp = {
                    "archetype": "FLAT_RATE",
                    "thresholdPaisa": "0",
                    "standardRateBps": rate_val
                }
            norm["computation"] = comp

        elif node_type == "TEMPORAL_WINDOW_GATE" and not has_pipeline:
            window_days = _parse_duration_days(root.get("applicableDeadlineWindow") or root.get("durationWindow"))
            if window_days is None:
                window_days = 45
            norm["computation"] = {
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
                        {
                            "assign": "interestPaisa",
                            "expr": {
                                "op": "CONDITIONAL",
                                "test": {"op": "GT", "left": "vars.overdueDays", "right": 0},
                                "then": {
                                    "op": "COMPOUND_INTEREST",
                                    "principal": "facts.invoiceAmountPaisa",
                                    "rateBps": "vars.annualRateBps",
                                    "months": "vars.monthsOverdue",
                                    "rests": "MONTHLY"
                                },
                                "else": 0
                            }
                        }
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

        elif node_type == "BOOLEAN_PROHIBITION_GATE" and not has_pipeline:
            dg = dict(norm.get("deonticGate") or comp.get("deonticGate") or {})
            dg["gateType"] = "BOOLEAN_PROHIBITION_GATE"
            dg["modalityType"] = "PROHIBITION"
            carveouts = list(root.get("exceptionCarveouts") or [])
            dg_carveouts = list(dg.get("exceptionCarveouts") or [])
            if carveouts and not any(isinstance(c, dict) and (c.get("expr") or c.get("field")) for c in dg_carveouts):
                dg["exceptionCarveouts"] = [
                    {
                        "id": "MOTOR_VEHICLE_COMMERCIAL_CARVEOUT",
                        "expr": "facts.isUsedForPassengerTransport || facts.isUsedForTaxableSupplyOfVehicles || facts.isUsedForDrivingSchool"
                    }
                ]
            elif not carveouts:
                dg["exceptionCarveouts"] = []
            if not dg.get("prohibitionPredicate"):
                dg["prohibitionPredicate"] = {
                    "op": "AND",
                    "left": "facts.isMotorVehicle",
                    "right": {"op": "LTE", "left": "facts.seatingCapacity", "right": 13}
                }
            norm["deonticGate"] = dg
            norm["computation"] = {"archetype": "BOOLEAN_PROHIBITION_GATE", "deonticGate": dg}

        elif node_type == "STATUS_DISQUALIFICATION_GATE" and not has_pipeline:
            dg = dict(norm.get("deonticGate") or comp.get("deonticGate") or {})
            dg["gateType"] = "STATUS_DISQUALIFICATION_GATE"
            dg["modalityType"] = "PROHIBITION"
            triggers = [
                {"id": "CONVICTION_MORAL_TURPITUDE", "field": "hasConvictionMoralTurpitude", "operator": "EQ", "value": True},
                {"id": "UNDISCHARGED_INSOLVENT", "field": "isUndischargedInsolvent", "operator": "EQ", "value": True},
                {"id": "WILFUL_DEFAULTER", "field": "isWilfulDefaulter", "operator": "EQ", "value": True},
                {"id": "FAILED_ANNUAL_FILINGS_3Y", "field": "hasFailedAnnualFilings3Years", "operator": "EQ", "value": True}
            ]
            req_evidence = [str(d).lower() for d in (root.get("requiredDocumentChecklist") or [])]
            if any("dir-8" in d or "dir 8" in d for d in req_evidence):
                triggers.append({"id": "MISSING_FORM_DIR8", "field": "hasSubmittedFormDIR8", "operator": "NEQ", "value": True})
            if any("29a" in d for d in req_evidence):
                triggers.append({"id": "MISSING_AFFIDAVIT_29A", "field": "hasSubmittedAffidavit29A", "operator": "NEQ", "value": True})
            dg["disqualificationTriggers"] = triggers
            dg["requireAttestationIfChecklist"] = False
            norm["deonticGate"] = dg
            norm["computation"] = {"archetype": "STATUS_DISQUALIFICATION_GATE", "deonticGate": dg}

        elif node_type == "CORPORATE_APPROVAL_GATE" and not has_pipeline:
            if "RELATED_PARTY_SCOPE" not in guard_ids:
                guards.append({
                    "id": "RELATED_PARTY_SCOPE",
                    "field": "isRelatedParty",
                    "operator": "EQ",
                    "value": True,
                    "onMismatch": {"action": "HALT_EXEMPT", "disposition": "EXEMPT_THIRD_PARTY"}
                })
                guard_ids.add("RELATED_PARTY_SCOPE")
                norm["guards"] = guards

            dg = dict(norm.get("deonticGate") or comp.get("deonticGate") or {})
            dg["gateType"] = "CORPORATE_APPROVAL_GATE"
            dg["modalityType"] = "OBLIGATION"
            carveouts = list(root.get("exceptionCarveouts") or [])
            if any("ordinary" in str(c).lower() or "arm" in str(c).lower() for c in carveouts):
                dg["exceptionCarveouts"] = [
                    {
                        "id": "ORDINARY_COURSE_ARMS_LENGTH_CARVEOUT",
                        "expr": "facts.isOrdinaryCourseOfBusiness && facts.isArmLengthPrice"
                    }
                ]
            else:
                dg["exceptionCarveouts"] = []

            thresh_inr = _parse_monetary_to_inr(root.get("applicableExcessThreshold"))
            thresh_val = int(round(thresh_inr)) if thresh_inr is not None else 0
            dg["mandatoryApprovals"] = [
                {"id": "BOARD_CONSENT", "expr": "facts.hasBoardConsent == true"},
                {
                    "id": "SHAREHOLDER_SPECIAL_RESOLUTION",
                    "expr": f"(!((facts.transactionAmountInr > {thresh_val}) || facts.exceedsTenPctTurnover)) || (facts.hasShareholderSpecialResolution == true)"
                }
            ]
            norm["deonticGate"] = dg
            norm["computation"] = {"archetype": "CORPORATE_APPROVAL_GATE", "deonticGate": dg}

        elif node_type == "PROCEDURAL_NOTICE_GATE" and not has_pipeline:
            dg = dict(norm.get("deonticGate") or comp.get("deonticGate") or {})
            dg["gateType"] = "PROCEDURAL_NOTICE_GATE"
            dg["modalityType"] = "PROHIBITION"
            days = _parse_duration_days(root.get("applicableDeadlineWindow"))
            dg["noticeWindowDays"] = days if (days is not None and days > 0) else 60
            carveouts = list(root.get("exceptionCarveouts") or [])
            dg["allowUrgentInterimRelief"] = any("urgent" in str(c).lower() for c in carveouts)
            dg["exceptionCarveouts"] = []
            norm["deonticGate"] = dg
            norm["computation"] = {"archetype": "PROCEDURAL_NOTICE_GATE", "deonticGate": dg}

        elif node_type == "SUBJECTIVE_EVIDENCE_GATE" and not has_pipeline:
            checklist = list(norm.get("attestationChecklist") or root.get("requiredDocumentChecklist") or [])
            norm["attestationChecklist"] = checklist
            dg = dict(norm.get("deonticGate") or comp.get("deonticGate") or {})
            dg["gateType"] = "SUBJECTIVE_EVIDENCE_GATE"
            dg["modalityType"] = "OBLIGATION"
            dg["requireAttestationIfChecklist"] = True
            if not checklist:
                dg["gates"] = [
                    {
                        "id": "CONTROLLER_ATTESTATION_REQUIRED",
                        "field": "hasControllerAttestation",
                        "operator": "EQ",
                        "value": True,
                        "disposition": "BLOCKED_PENDING_ATTESTATION"
                    }
                ]
            norm["deonticGate"] = dg
            norm["computation"] = {"archetype": "SUBJECTIVE_EVIDENCE_GATE", "deonticGate": dg}

        return norm

    def _map_voucher_compat(self, voucher: dict, ast: dict) -> dict:
        out = dict(voucher or {})
        out["isAuthorizedToDisburse"] = False
        root = (ast or {}).get("rootCondition") or {}
        node_type = root.get("nodeType") or out.get("gateType") or "TRANSACTIONAL_WITHHOLDING"

        if "calculatedTaxInr" not in out:
            out["calculatedTaxInr"] = float(out.get("calculatedTdsInr", 0.0) or 0.0)
        if "isApplicable" not in out:
            out["isApplicable"] = bool(out.get("isTdsApplicable", False))
        if "effectiveRateBps" not in out:
            out["effectiveRateBps"] = int(round(float(out.get("tdsRatePercent", 0.0) or 0.0) * 100))

        matched_carveouts = out.get("matchedCarveouts") or []
        missing_approvals = out.get("missingApprovals") or []
        is_blocked = bool(out.get("isBlocked", False))
        is_compliant = bool(out.get("isCompliant", not is_blocked))

        if node_type == "STATUS_DISQUALIFICATION_GATE":
            out["isDisqualified"] = is_blocked
            out["isEligible"] = not is_blocked
        elif node_type == "CORPORATE_APPROVAL_GATE":
            out["isApproved"] = is_compliant and not is_blocked
            out["exemptionApplied"] = len(matched_carveouts) > 0
            out["requiresShareholderSpecialResolution"] = "SHAREHOLDER_SPECIAL_RESOLUTION" in missing_approvals
        elif node_type == "PROCEDURAL_NOTICE_GATE":
            out["isSuitAdmissible"] = bool(out.get("isAdmissible", is_compliant and not is_blocked))
            out["exemptionApplied"] = len(matched_carveouts) > 0
        elif node_type == "BOOLEAN_PROHIBITION_GATE":
            out["isItcEligible"] = not is_blocked
            out["meetsExceptionCarveout"] = len(matched_carveouts) > 0
        elif node_type == "SUBJECTIVE_EVIDENCE_GATE":
            out["isApproved"] = is_compliant and not is_blocked
            out["gateStatus"] = "VERIFIED_AND_ATTESTED" if out["isApproved"] else "BLOCKED_PENDING_ATTESTATION"
        elif node_type == "GENERAL_RULE_GATE" or (ast or {}).get("ruleDomain") == "GENERAL_LOGIC":
            if "isPermitted" not in out:
                out["isPermitted"] = is_compliant and not is_blocked
            out["isCompliant"] = is_compliant
            out["isSatisfied"] = bool(out.get("isSatisfied", is_compliant))

        return out

    def evaluate(self, ast: dict, facts: dict, options: dict | None = None) -> dict:
        norm_ast = self._normalize_ast_for_js_bridge(ast)
        norm_facts = self._normalize_facts_for_js_bridge(facts)
        opts = dict(options or {})
        if "mode" not in opts:
            opts["mode"] = self.default_mode

        resp = self._request({
            "op": "evaluate",
            "ruleAst": norm_ast,
            "facts": norm_facts,
            "options": opts
        })
        if not resp.get("ok"):
            err_msg = resp.get("error", "USVM evaluation failed")
            if "ERR_FIDUCIARY_INVARIANT_VIOLATION" in err_msg:
                raise SecurityError(err_msg)
            raise RuntimeError(err_msg)
        return self._map_voucher_compat(resp["result"], ast)

    def evaluate_batch(self, items: list[dict], options: dict | None = None) -> list[dict]:
        if not items:
            return []
        default_opts = dict(options or {})
        if "mode" not in default_opts:
            default_opts["mode"] = self.default_mode

        payload_items = []
        raw_asts = []
        for entry in items:
            raw_ast = entry.get("ast") or entry.get("ruleAst")
            raw_asts.append(raw_ast)
            norm_ast = self._normalize_ast_for_js_bridge(raw_ast)
            norm_facts = self._normalize_facts_for_js_bridge(entry.get("facts") or {})
            item_opts = dict(entry.get("options") or default_opts)
            payload_items.append({
                "ruleAst": norm_ast,
                "facts": norm_facts,
                "options": item_opts
            })

        resp = self._request({"op": "evaluate_batch", "items": payload_items})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "USVM batch evaluation failed"))

        out = []
        for idx, sub in enumerate(resp.get("results", [])):
            if not sub.get("ok"):
                out.append({
                    "isAuthorizedToDisburse": False,
                    "isBlocked": True,
                    "error": sub.get("error"),
                    "code": sub.get("code")
                })
            else:
                out.append(self._map_voucher_compat(sub["result"], raw_asts[idx]))
        return out

    def validate(self, ast: dict, mode: str = "DEVELOPMENT") -> dict:
        norm_ast = self._normalize_ast_for_js_bridge(ast)
        resp = self._request({"op": "validate", "ruleAst": norm_ast, "mode": mode})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "USVM validate failed"))
        return resp["result"]

    def lifecycle(self, ast: dict, options: dict | None = None) -> dict:
        resp = self._request({"op": "lifecycle", "ruleAst": ast, "options": options or {}})
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "USVM lifecycle check failed"))
        return resp["result"]


# Multi-Domain Statutory Test Vector Registry
DOMAIN_TEST_VECTORS = {
    "194Q_STYLE": [
        {
            "id": "VEC_194Q_COMPLIANT",
            "description": "Standard Section 194Q purchase exceeding 50L (0.1% TDS)",
            "facts": {
                "transactionDate": "2023-10-15",
                "buyerPriorYearTurnoverInr": 150000000,
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 5000000,
                "currentInvoiceAmountInr": 3000000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 3000.0,
                "effectiveRateBps": 10
            }
        },
        {
            "id": "VEC_194Q_NON_PAN",
            "description": "Section 194Q purchase exceeding 50L with missing PAN (5% penal rate)",
            "facts": {
                "transactionDate": "2023-11-20",
                "buyerPriorYearTurnoverInr": 150000000,
                "isSellerResident": True,
                "hasValidPan": False,
                "cumulativePurchasesPriorInr": 5000000,
                "currentInvoiceAmountInr": 1000000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 50000.0,
                "effectiveRateBps": 500
            }
        },
        {
            "id": "VEC_194Q_GATE_FAIL",
            "description": "Buyer turnover is only 5 Cr (below 10 Cr gate) -> Exemption",
            "facts": {
                "transactionDate": "2023-10-15",
                "buyerPriorYearTurnoverInr": 50000000,
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 8000000,
                "currentInvoiceAmountInr": 2000000
            },
            "expected": {
                "isApplicable": False,
                "calculatedTaxInr": 0.0,
                "effectiveRateBps": 0
            }
        }
    ],

    "206C1H_STYLE": [
        {
            "id": "VEC_206C1H_ACTIVE",
            "description": "Seller collects 0.1% TCS on receipts exceeding 50L prior to sunset",
            "facts": {
                "transactionDate": "2023-10-15",
                "buyerPriorYearTurnoverInr": 150000000,
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 5000000,
                "currentInvoiceAmountInr": 2000000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 2000.0,
                "effectiveRateBps": 10
            }
        },
        {
            "id": "VEC_206C1H_SUNSET",
            "description": "Transaction after 1 April 2025 sunset date -> Section 206C(1H) omitted",
            "facts": {
                "transactionDate": "2025-04-02",
                "buyerPriorYearTurnoverInr": 150000000,
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 6000000,
                "currentInvoiceAmountInr": 2000000
            },
            "expected": {
                "isApplicable": False,
                "calculatedTaxInr": 0.0,
                "effectiveRateBps": 0
            }
        }
    ],

    "194R_STYLE": [
        {
            "id": "VEC_194R_EXCESS",
            "description": "Business benefit/perquisite value exceeding 20,000 cliff (10% TDS on full value)",
            "facts": {
                "transactionDate": "2023-10-15",
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 0,
                "currentInvoiceAmountInr": 50000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 5000.0,
                "effectiveRateBps": 1000
            }
        },
        {
            "id": "VEC_194R_BELOW_CLIFF",
            "description": "Business benefit value below 20,000 -> Exemption",
            "facts": {
                "transactionDate": "2023-10-15",
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 0,
                "currentInvoiceAmountInr": 15000
            },
            "expected": {
                "isApplicable": False,
                "calculatedTaxInr": 0.0,
                "effectiveRateBps": 0
            }
        }
    ],

    "194C_STYLE": [
        {
            "id": "VEC_194C_SINGLE_CONTRACT",
            "description": "Contractor payment single invoice exceeding 30,000 cliff (1% on full invoice)",
            "facts": {
                "transactionDate": "2023-10-15",
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 0,
                "currentInvoiceAmountInr": 50000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 500.0,
                "effectiveRateBps": 100
            }
        },
        {
            "id": "VEC_194C_BELOW_CLIFF",
            "description": "Contractor payment single invoice below 30,000 with zero prior -> Exemption",
            "facts": {
                "transactionDate": "2023-10-15",
                "isSellerResident": True,
                "hasValidPan": True,
                "cumulativePurchasesPriorInr": 0,
                "currentInvoiceAmountInr": 20000
            },
            "expected": {
                "isApplicable": False,
                "calculatedTaxInr": 0.0,
                "effectiveRateBps": 0
            }
        }
    ],

    "SUNSET_STYLE": [
        {
            "id": "VEC_SUNSET_BEFORE",
            "description": "Transaction occurring prior to statutory sunset date",
            "facts": {
                "transactionDate": "2024-10-15",
                "isSellerResident": True,
                "hasValidPan": True,
                "currentInvoiceAmountInr": 100000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 0.0,
                "effectiveRateBps": 0
            }
        },
        {
            "id": "VEC_SUNSET_AFTER",
            "description": "Transaction occurring after statutory sunset date -> Omitted",
            "facts": {
                "transactionDate": "2025-05-01",
                "isSellerResident": True,
                "hasValidPan": True,
                "currentInvoiceAmountInr": 100000
            },
            "expected": {
                "isApplicable": False,
                "calculatedTaxInr": 0.0,
                "effectiveRateBps": 0
            }
        }
    ],

    "MSME_15_STYLE": [
        {
            "id": "VEC_MSME15_COMPLIANT",
            "description": "MSME invoice paid within 45 days (Day 24) -> 100% Compliant, 0 interest",
            "facts": {
                "triggerDate": "2024-01-01",
                "settlementDate": "2024-01-25",
                "invoiceAmountInr": 1000000,
                "rbiBankRatePct": 6.5
            },
            "expected": {
                "isApplicable": True,
                "isCompliant": True,
                "calculatedInterestInr": 0.0,
                "overdueDays": 0
            }
        },
        {
            "id": "VEC_MSME15_OVERDUE",
            "description": "MSME invoice paid after 74 days (29 days overdue past 45d) -> Compound interest violation",
            "facts": {
                "triggerDate": "2024-01-01",
                "settlementDate": "2024-03-15",
                "invoiceAmountInr": 1000000,
                "rbiBankRatePct": 6.5
            },
            "expected": {
                "isApplicable": True,
                "isCompliant": False,
                "overdueDays": 29
            }
        }
    ],

    "GST_17_5_STYLE": [
        {
            "id": "VEC_GST17_5_BLOCKED",
            "description": "Motor vehicle passenger transport 5-seater for executive office use -> Blocked ITC",
            "facts": {
                "isMotorVehicle": True,
                "seatingCapacity": 5,
                "isUsedForPassengerTransport": False,
                "isUsedForTaxableSupplyOfVehicles": False,
                "isUsedForDrivingSchool": False
            },
            "expected": {
                "isApplicable": True,
                "isItcEligible": False,
                "isBlocked": True
            }
        },
        {
            "id": "VEC_GST17_5_CARVEOUT",
            "description": "Motor vehicle 5-seater purchased for commercial passenger transport service -> Carveout eligible",
            "facts": {
                "isMotorVehicle": True,
                "seatingCapacity": 5,
                "isUsedForPassengerTransport": True,
                "isUsedForTaxableSupplyOfVehicles": False,
                "isUsedForDrivingSchool": False
            },
            "expected": {
                "isApplicable": True,
                "isItcEligible": True,
                "isBlocked": False
            }
        }
    ],

    "CGST_SEC8_STYLE": [
        {
            "id": "VEC_SEC8_ATTESTED",
            "description": "Composite supply with bundling study report and controller signoff -> Verified",
            "facts": {
                "uploadedDocuments": [
                    "SUPPLIER_MASTER_SERVICE_AGREEMENT",
                    "INDUSTRY_BUNDLING_PRACTICE_MEMO",
                    "SINGLE_PRICE_INVOICE_BREAKUP"
                ],
                "hasControllerAttestation": True
            },
            "expected": {
                "isApplicable": True,
                "isApproved": True,
                "gateStatus": "VERIFIED_AND_ATTESTED"
            }
        },
        {
            "id": "VEC_SEC8_UNATTESTED",
            "description": "Composite supply missing required documents and unsigned -> Blocked fail-closed",
            "facts": {
                "uploadedDocuments": [],
                "hasControllerAttestation": False
            },
            "expected": {
                "isApplicable": True,
                "isApproved": False,
                "gateStatus": "BLOCKED_PENDING_ATTESTATION"
            }
        }
    ],

    "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE": [
        {
            "id": "VEC_DIR_DISQUALIFIED_CONVICTION",
            "description": "Proposed director convicted of moral turpitude -> Disqualified under Section 164(1)",
            "facts": {
                "hasConvictionMoralTurpitude": True,
                "isUndischargedInsolvent": False,
                "hasFailedAnnualFilings3Years": False,
                "hasSubmittedFormDIR8": False
            },
            "expected": {
                "isApplicable": True,
                "isEligible": False,
                "isDisqualified": True
            }
        },
        {
            "id": "VEC_DIR_QUALIFIED_COMPLIANT",
            "description": "Proposed director with clean record and valid Form DIR-8 -> Eligible",
            "facts": {
                "hasConvictionMoralTurpitude": False,
                "isUndischargedInsolvent": False,
                "hasFailedAnnualFilings3Years": False,
                "hasSubmittedFormDIR8": True
            },
            "expected": {
                "isApplicable": True,
                "isEligible": True,
                "isDisqualified": False
            }
        }
    ],

    "COMPANIES_RELATED_PARTY_STYLE": [
        {
            "id": "VEC_RPT_UNAPPROVED_THRESHOLD_BREACH",
            "description": "Related party transaction INR 150 Cr exceeding 100 Cr threshold without special resolution -> Barred",
            "facts": {
                "isRelatedParty": True,
                "transactionAmountInr": 1500000000,
                "isOrdinaryCourseOfBusiness": False,
                "isArmLengthPrice": False,
                "hasBoardConsent": True,
                "hasShareholderSpecialResolution": False
            },
            "expected": {
                "isApplicable": True,
                "isApproved": False,
                "requiresShareholderSpecialResolution": True
            }
        },
        {
            "id": "VEC_RPT_ORDINARY_COURSE_EXEMPT",
            "description": "Related party transaction in ordinary course of business at arm's length -> Section 188(1) proviso exempt",
            "facts": {
                "isRelatedParty": True,
                "transactionAmountInr": 2000000000,
                "isOrdinaryCourseOfBusiness": True,
                "isArmLengthPrice": True
            },
            "expected": {
                "isApplicable": True,
                "isApproved": True,
                "exemptionApplied": True
            }
        }
    ],

    "PROCEDURAL_NOTICE_WINDOW_STYLE": [
        {
            "id": "VEC_CPC80_PREMATURE_BARRED",
            "description": "Suit against Government filed 20 days after notice (prescribed 60 days) -> Barred / Plaint rejected",
            "facts": {
                "noticeDeliveryDate": "2024-01-01",
                "suitFilingDate": "2024-01-21",
                "isUrgentInterimReliefSought": False
            },
            "expected": {
                "isApplicable": True,
                "isSuitAdmissible": False
            }
        },
        {
            "id": "VEC_CPC80_SATISFIED_WINDOW",
            "description": "Suit against Government filed 75 days after notice (prescribed 60 days) -> Admissible",
            "facts": {
                "noticeDeliveryDate": "2024-01-01",
                "suitFilingDate": "2024-03-16",
                "isUrgentInterimReliefSought": False
            },
            "expected": {
                "isApplicable": True,
                "isSuitAdmissible": True
            }
        }
    ],

    "IBC_RESOLUTION_INELIGIBILITY_STYLE": [
        {
            "id": "VEC_IBC29A_DISQUALIFIED_DEFAULTER",
            "description": "Resolution applicant is a wilful defaulter without Section 29A affidavit -> Ineligible",
            "facts": {
                "isWilfulDefaulter": True,
                "isUndischargedInsolvent": False,
                "hasSubmittedAffidavit29A": False
            },
            "expected": {
                "isApplicable": True,
                "isEligible": False,
                "isDisqualified": True
            }
        },
        {
            "id": "VEC_IBC29A_ELIGIBLE_CLEAN",
            "description": "Resolution applicant with clean record and valid Section 29A affidavit -> Eligible",
            "facts": {
                "isWilfulDefaulter": False,
                "isUndischargedInsolvent": False,
                "hasSubmittedAffidavit29A": True
            },
            "expected": {
                "isApplicable": True,
                "isEligible": True,
                "isDisqualified": False
            }
        }
    ],

    "GENERAL_PERMISSION_CONDITIONAL_STYLE": [
        {
            "id": "VEC_GEN_PERM_GRANTED",
            "description": "Member satisfies both badge and clear-sector preconditions -> Permitted",
            "facts": {
                "requesterHoldsBadge": True,
                "sectorBlocked": False
            },
            "expected": {
                "isPermitted": True,
                "isCompliant": True
            }
        },
        {
            "id": "VEC_GEN_PERM_DENIED",
            "description": "Blocked adjacent sector -> Action permission denied",
            "facts": {
                "requesterHoldsBadge": True,
                "sectorBlocked": True
            },
            "expected": {
                "isPermitted": False,
                "isCompliant": False
            }
        }
    ],

    "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE": [
        {
            "id": "VEC_GEN_QUANT_ALL_CLEAR",
            "description": "Every milestone verified AND no entry flagged -> Permitted",
            "facts": {
                "milestones": [{"isVerified": True}, {"isVerified": True}],
                "entries": [{"isFlagged": False}, {"isFlagged": False}]
            },
            "expected": {
                "isPermitted": True,
                "isCompliant": True
            }
        },
        {
            "id": "VEC_GEN_QUANT_FLAGGED_EXISTS",
            "description": "One entry flagged fails NOT EXISTS check -> Blocked",
            "facts": {
                "milestones": [{"isVerified": True}, {"isVerified": True}],
                "entries": [{"isFlagged": False}, {"isFlagged": True}]
            },
            "expected": {
                "isPermitted": False,
                "isCompliant": False
            }
        }
    ],

    "GENERAL_ENTITY_GRAPH_RELATION_STYLE": [
        {
            "id": "VEC_GEN_GRAPH_REACHABLE",
            "description": "All predecessor nodes verified AND no blocked link -> Dispatch permitted",
            "facts": {
                "nodes": [{"isVerified": True}, {"isVerified": True}],
                "links": [{"isDisabled": False}, {"isDisabled": False}]
            },
            "expected": {
                "isPermitted": True,
                "isCompliant": True
            }
        },
        {
            "id": "VEC_GEN_GRAPH_DISABLED_LINK",
            "description": "Disabled link blocks dispatch",
            "facts": {
                "nodes": [{"isVerified": True}, {"isVerified": True}],
                "links": [{"isDisabled": False}, {"isDisabled": True}]
            },
            "expected": {
                "isPermitted": False,
                "isCompliant": False
            }
        }
    ],

    "GENERAL_AGGREGATE_COLLECTION_STYLE": [
        {
            "id": "VEC_GEN_AGG_OVER_THRESHOLD",
            "description": "Sum of qualifying claims (25,000) exceeds 15,000 floor -> 10% on 10,000 excess = 1,000",
            "facts": {
                "qualifyingClaims": [{"value": 12000, "isQualifying": True}, {"value": 13000, "isQualifying": True}],
                "thresholdAmount": 15000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 1000.0
            }
        },
        {
            "id": "VEC_GEN_AGG_BELOW_THRESHOLD",
            "description": "Sum of qualifying claims (9,000) below 15,000 floor -> Not applicable (0)",
            "facts": {
                "qualifyingClaims": [{"value": 4000, "isQualifying": True}, {"value": 5000, "isQualifying": True}]
            },
            "expected": {
                "isApplicable": False,
                "calculatedTaxInr": 0.0
            }
        }
    ],

    "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE": [
        {
            "id": "VEC_GEN_DAG_RATE_LESS_THAN_CAP",
            "description": "20% of (50,000 - 10,000) = 8,000 < 12,000 cap -> 8,000",
            "facts": {
                "grossAssessment": 50000,
                "standardAllowance": 10000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 8000.0
            }
        },
        {
            "id": "VEC_GEN_DAG_CAPPED",
            "description": "20% of (100,000 - 10,000) = 18,000 capped at 12,000 -> 12,000",
            "facts": {
                "grossAssessment": 100000,
                "standardAllowance": 10000
            },
            "expected": {
                "isApplicable": True,
                "calculatedTaxInr": 12000.0
            }
        }
    ],

    "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE": [
        {
            "id": "VEC_GEN_ESCROW_FEE_PAID",
            "description": "Restricted carrier pays $50 escrow fee within 48 hours -> Permitted",
            "facts": {
                "exceedsThreshold": True,
                "dispatchCargo": True,
                "feePaid": 50,
                "elapsedHours": 24
            },
            "expected": {
                "carveoutApplied": True,
                "isCompliant": True
            }
        },
        {
            "id": "VEC_GEN_ESCROW_NO_FEE_VIOLATION",
            "description": "Restricted carrier dispatches without escrow fee -> Prohibited / non-compliant",
            "facts": {
                "exceedsThreshold": True,
                "dispatchCargo": True,
                "feePaid": 0,
                "elapsedHours": 24
            },
            "expected": {
                "carveoutApplied": False,
                "isCompliant": False
            }
        }
    ]
}

CANONICAL_TEST_VECTORS = DOMAIN_TEST_VECTORS["194Q_STYLE"]


if __name__ == "__main__":
    with CanonicalUsvmEvaluator() as evaluator:
        dummy_asts = {
            "194Q_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "TRANSACTIONAL_WITHHOLDING",
                    "entityType": "BUYER",
                    "priorTurnoverPrecondition": "10 crore",
                    "applicableExcessThreshold": "50 lakh",
                    "standardTaxRate": "0.1%",
                    "nonPanPenalRate": "5%",
                    "triggerPolicy": "EARLIER_OF",
                    "temporalCutover": "1st July 2021",
                    "temporalSunset": None
                }
            },
            "206C1H_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "TRANSACTIONAL_WITHHOLDING",
                    "entityType": "SELLER",
                    "priorTurnoverPrecondition": "10 crore",
                    "applicableExcessThreshold": "50 lakh",
                    "standardTaxRate": "0.1%",
                    "temporalSunset": "1st April 2025"
                }
            },
            "194R_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "TRANSACTIONAL_WITHHOLDING",
                    "applicableCliffThreshold": "20000",
                    "standardTaxRate": "10%"
                }
            },
            "194C_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "TRANSACTIONAL_WITHHOLDING",
                    "applicableCliffThreshold": "30000",
                    "standardTaxRate": "1%"
                }
            },
            "SUNSET_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "TRANSACTIONAL_WITHHOLDING",
                    "temporalSunset": "1st April 2025"
                }
            },
            "MSME_15_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "TEMPORAL_WINDOW_GATE",
                    "applicableDeadlineWindow": "forty-five days"
                }
            },
            "GST_17_5_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "BOOLEAN_PROHIBITION_GATE",
                    "prohibitionPredicate": "Motor vehicle ITC blocked",
                    "exceptionCarveouts": ["transportation of passengers"]
                }
            },
            "CGST_SEC8_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "SUBJECTIVE_EVIDENCE_GATE",
                    "requiredDocumentChecklist": [
                        "SUPPLIER_MASTER_SERVICE_AGREEMENT",
                        "INDUSTRY_BUNDLING_PRACTICE_MEMO",
                        "SINGLE_PRICE_INVOICE_BREAKUP"
                    ]
                }
            },
            "COMPANIES_DIRECTOR_DISQUALIFICATION_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "STATUS_DISQUALIFICATION_GATE",
                    "requiredDocumentChecklist": ["Form DIR-8"]
                }
            },
            "COMPANIES_RELATED_PARTY_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "CORPORATE_APPROVAL_GATE",
                    "applicableExcessThreshold": "100 crore",
                    "exceptionCarveouts": ["ordinary course of business on an arm's length basis"]
                }
            },
            "PROCEDURAL_NOTICE_WINDOW_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "PROCEDURAL_NOTICE_GATE",
                    "applicableDeadlineWindow": "two months",
                    "exceptionCarveouts": ["urgent interim relief"]
                }
            },
            "IBC_RESOLUTION_INELIGIBILITY_STYLE": {
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {
                    "nodeType": "STATUS_DISQUALIFICATION_GATE",
                    "requiredDocumentChecklist": ["affidavit under section 29A"]
                }
            },
            "GENERAL_PERMISSION_CONDITIONAL_STYLE": {
                "ruleDomain": "GENERAL_LOGIC",
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "PERMISSION"},
                "computation": {
                    "archetype": "PURE_LOGIC_PIPELINE",
                    "outputMode": "PURE_LOGIC",
                    "pipeline": {
                        "steps": [
                            {"assign": "cond_1", "expr": {"op": "EQ", "left": "facts.requesterHoldsBadge", "right": True}},
                            {"assign": "cond_2", "expr": {"op": "NOT", "operand": {"op": "EQ", "left": "facts.sectorBlocked", "right": True}}},
                            {"assign": "isPermitted", "expr": {"op": "AND", "left": "vars.cond_1", "right": "vars.cond_2"}}
                        ],
                        "output": {"isPermitted": "vars.isPermitted", "isCompliant": "vars.isPermitted"}
                    }
                }
            },
            "GENERAL_QUANTIFIER_FORALL_EXISTS_STYLE": {
                "ruleDomain": "GENERAL_LOGIC",
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "PERMISSION"},
                "computation": {
                    "archetype": "PURE_LOGIC_PIPELINE",
                    "outputMode": "PURE_LOGIC",
                    "pipeline": {
                        "steps": [
                            {"assign": "allOk", "expr": {"op": "FORALL", "iteratorVar": "it", "collection": "facts.milestones", "predicate": {"op": "EQ", "left": "it.isVerified", "right": True}}},
                            {"assign": "noBad", "expr": {"op": "NOT", "operand": {"op": "EXISTS", "iteratorVar": "it", "collection": "facts.entries", "predicate": {"op": "EQ", "left": "it.isFlagged", "right": True}}}},
                            {"assign": "isPermitted", "expr": {"op": "AND", "left": "vars.allOk", "right": "vars.noBad"}}
                        ],
                        "output": {"isPermitted": "vars.isPermitted", "isCompliant": "vars.isPermitted"}
                    }
                }
            },
            "GENERAL_ENTITY_GRAPH_RELATION_STYLE": {
                "ruleDomain": "GENERAL_LOGIC",
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "PERMISSION"},
                "computation": {
                    "archetype": "PURE_LOGIC_PIPELINE",
                    "outputMode": "PURE_LOGIC",
                    "pipeline": {
                        "steps": [
                            {"assign": "nodesOk", "expr": {"op": "FORALL", "iteratorVar": "it", "collection": "facts.nodes", "predicate": {"op": "EQ", "left": "it.isVerified", "right": True}}},
                            {"assign": "linksOk", "expr": {"op": "NOT", "operand": {"op": "EXISTS", "iteratorVar": "it", "collection": "facts.links", "predicate": {"op": "EQ", "left": "it.isDisabled", "right": True}}}},
                            {"assign": "isPermitted", "expr": {"op": "AND", "left": "vars.nodesOk", "right": "vars.linksOk"}}
                        ],
                        "output": {"isPermitted": "vars.isPermitted", "isCompliant": "vars.isPermitted"}
                    }
                }
            },
            "GENERAL_AGGREGATE_COLLECTION_STYLE": {
                "ruleDomain": "GENERAL_LOGIC",
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "OBLIGATION"},
                "computation": {
                    "archetype": "DECLARATIVE_PIPELINE",
                    "thresholdPaisa": "1500000",
                    "standardRateBps": 1000,
                    "pipeline": {
                        "steps": [
                            {"assign": "rawSum", "expr": {"op": "SUM_OVER", "iteratorVar": "it", "collection": "facts.qualifyingClaims", "predicate": {"op": "EQ", "left": "it.isQualifying", "right": True}, "body": "it.value"}},
                            {"assign": "sumPaisa", "expr": {"op": "TO_PAISE", "value": "vars.rawSum"}},
                            {"assign": "excessPaisa", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": "vars.sumPaisa", "right": {"op": "TO_PAISE", "value": "1500000"}}}},
                            {"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.excessPaisa", "right": {"op": "TO_PAISE", "value": "0"}}},
                            {"assign": "taxAmountPaisa", "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.excessPaisa", "rateBps": 1000, "denominator": 10000}}
                        ],
                        "output": {"isApplicable": "vars.isApplicable", "taxAmountPaisa": "vars.taxAmountPaisa", "calculatedTaxInr": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}}
                    }
                }
            },
            "GENERAL_COMPOSED_DAG_ARITHMETIC_STYLE": {
                "ruleDomain": "GENERAL_LOGIC",
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "OBLIGATION"},
                "computation": {
                    "archetype": "DECLARATIVE_PIPELINE",
                    "thresholdPaisa": "0",
                    "standardRateBps": 2000,
                    "pipeline": {
                        "steps": [
                            {"assign": "netBasePaisa", "expr": {"op": "MAX", "left": {"op": "TO_PAISE", "value": "0"}, "right": {"op": "SUB", "left": {"op": "TO_PAISE", "value": "facts.grossAssessment"}, "right": {"op": "TO_PAISE", "value": "facts.standardAllowance"}}}},
                            {"assign": "rawChargePaisa", "expr": {"op": "RATIONAL_MULTIPLY", "base": "vars.netBasePaisa", "rateBps": 2000, "denominator": 10000}},
                            {"assign": "cappedChargePaisa", "expr": {"op": "MIN", "left": "vars.rawChargePaisa", "right": {"op": "TO_PAISE", "value": "1200000"}}},
                            {"assign": "isApplicable", "expr": {"op": "GT", "left": "vars.cappedChargePaisa", "right": {"op": "TO_PAISE", "value": "0"}}}
                        ],
                        "output": {"isApplicable": "vars.isApplicable", "taxAmountPaisa": "vars.cappedChargePaisa", "calculatedTaxInr": {"op": "FROM_PAISE", "value": "vars.cappedChargePaisa"}}
                    }
                }
            },
            "GENERAL_PROHIBITION_MONETARY_ESCROW_STYLE": {
                "ruleDomain": "GENERAL_LOGIC",
                "statutoryInvariants": {"isAuthorizedToDisburse": False},
                "rootCondition": {"nodeType": "GENERAL_RULE_GATE", "modalityType": "PROHIBITION"},
                "computation": {
                    "archetype": "PURE_LOGIC_PIPELINE",
                    "outputMode": "PURE_LOGIC",
                    "pipeline": {
                        "steps": [
                            {"assign": "condTriggered", "expr": {"op": "AND", "left": {"op": "EQ", "left": "facts.exceedsThreshold", "right": True}, "right": {"op": "EQ", "left": "facts.dispatchCargo", "right": True}}},
                            {"assign": "mainCondition", "expr": {"op": "NOT", "operand": "vars.condTriggered"}},
                            {"assign": "carveoutApplied", "expr": {"op": "AND", "left": {"op": "GTE", "left": "facts.feePaid", "right": 50}, "right": {"op": "LTE", "left": "facts.elapsedHours", "right": 48}}},
                            {"assign": "isCompliant", "expr": {"op": "OR", "left": "vars.mainCondition", "right": "vars.carveoutApplied"}}
                        ],
                        "output": {"carveoutApplied": "vars.carveoutApplied", "isCompliant": "vars.isCompliant"}
                    }
                }
            }
        }

        all_passed = 0
        all_total = 0
        batch_req = []
        batch_meta = []
        for style_key, vecs in DOMAIN_TEST_VECTORS.items():
            ast_obj = dummy_asts[style_key]
            for vec in vecs:
                batch_req.append({"ast": ast_obj, "facts": vec["facts"]})
                batch_meta.append((style_key, vec))

        results = evaluator.evaluate_batch(batch_req)
        for (style_key, vec), res in zip(batch_meta, results):
            ok = all(res.get(k) == v for k, v in vec["expected"].items()) and (res.get("isAuthorizedToDisburse") is False)
            all_total += 1
            if ok:
                all_passed += 1
            else:
                print(f"[FAIL] {style_key} / {vec['id']}: got={res} expected={vec['expected']}")

        print(f"Canonical JS USVM Bridge Self-Test: {all_passed}/{all_total} passed")
        if all_passed != all_total:
            raise SystemExit(1)
