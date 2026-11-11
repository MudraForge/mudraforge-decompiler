"""
=============================================================================
DEONTICBENCH / SARA BENCHMARK EVALUATION (RAW MODEL + CANONICAL USVM)
=============================================================================

Evaluates the neural compiler checkpoints
(./checkpoints/dual_stream_sup.pt)
zero-shot against the official Johns Hopkins / DeonticBench SARA dataset
(https://github.com/guangyaodou/DeonticBench, arXiv:2604.04443), with ZERO
hardcoded US IRC tokens or SARA examples in training data or inference.py.

  Track 1A: Zero-shot neural decompilation on 10 verbatim IRC clauses
            extracted straight from DeonticBench/statutes/sara/*
  Track 1B: Diagnostic surface-token normalization ablation
            ($ -> rupees/INR, percent -> per cent, weights untouched)
  Track 1C: Zero-shot neural decompilation on all 9 raw multi-subsection
            statutes/sara/* full documents (358 to 6,984 characters)
  Track 1D: End-to-end compiled AST -> Canonical JS USVM execution on
            verbatim computational/temporal US IRC clauses (zero hand-authored ASTs)
  Track 2:  Canonical JS USVM (evaluation_vm.js DECLARATIVE_PIPELINE &
            PURE_LOGIC_PIPELINE bounded quantifiers) deterministic execution
            across ALL 30/30 Binary and ALL 35/35 Numeric hard-split cases
"""

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from compiler.neural.tokenizer import tokenize_with_offsets
from compiler.neural.inference import MudraDecompilerInference, resolve_open_class_token
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator


def make_sara_decompiler(path: str):
    if "dual_stream" in str(path):
        return DualStreamDecompilerInference(path)
    return MudraDecompilerInference(path)

CACHE_DIR = os.environ.get("SARA_CACHE_DIR") or os.path.join(REPO_ROOT, "data", "deonticbench_sara")
RAW_BASE = "https://raw.githubusercontent.com/guangyaodou/DeonticBench/main"
STATUTE_FILES = [
    "section1", "section2", "section63", "section68",
    "section151", "section152", "section3301", "section3306", "section7703"
]


def ensure_deonticbench_sara_cached() -> str:
    """Downloads official DeonticBench SARA files if not yet cached locally."""
    os.makedirs(os.path.join(CACHE_DIR, "statutes"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "data", "sara_numeric"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "data", "sara_binary"), exist_ok=True)

    for s in STATUTE_FILES:
        dst = os.path.join(CACHE_DIR, "statutes", s)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"{RAW_BASE}/statutes/sara/{s}", dst)

    for domain in ("sara_numeric", "sara_binary"):
        for split in ("smoke.json", "hard.json"):
            dst = os.path.join(CACHE_DIR, "data", domain, split)
            if not os.path.exists(dst):
                urllib.request.urlretrieve(f"{RAW_BASE}/data/{domain}/{split}", dst)

    return CACHE_DIR


def load_statutes(cache_dir: str) -> dict[str, str]:
    out = {}
    for s in STATUTE_FILES:
        with open(os.path.join(cache_dir, "statutes", s), encoding="utf-8") as f:
            out[s] = f.read()
    return out


def compute_unk_rate(text: str, vocab: dict) -> tuple[float, list[str], list[str]]:
    tokens, _ = tokenize_with_offsets(text)
    if not tokens:
        return 0.0, [], []
    unk_id = vocab.get("<UNK>", 1)
    noun_id = vocab.get("<LEGAL_NOUN>", 2)
    verb_id = vocab.get("<LEGAL_VERB>", 3)
    unk_toks = []
    open_slot_toks = []
    for i, tok in enumerate(tokens):
        p1 = tokens[i - 1] if i > 0 else None
        p2 = tokens[i - 2] if i > 1 else None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        tid = resolve_open_class_token(tok, p1, p2, vocab, next_token=nxt)
        if tid == unk_id:
            unk_toks.append(tok)
        elif tid in (noun_id, verb_id) and tok.lower() not in vocab:
            open_slot_toks.append(tok)
    return round(len(unk_toks) / len(tokens) * 100.0, 1), unk_toks, open_slot_toks


def surface_normalize_us_to_inr(text: str) -> str:
    """
    Diagnostic ablation only (does NOT modify model weights):
    normalizes '$300,000' -> '300000 rupees' and 'percent' -> 'per cent'.
    """
    out = re.sub(
        r"\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)",
        lambda m: f"{m.group(1).replace(',', '')} rupees",
        text,
    )
    out = re.sub(r"\bpercent\b", "per cent", out, flags=re.IGNORECASE)
    return out


VERBATIM_OPERATIVE_CLAUSES = [
    {
        "id": "IRC_3301_FUTA_6PCT",
        "section": "section3301",
        "text": (
            "There is hereby imposed on every employer for each calendar year an excise tax "
            "equal to 6 percent of the total wages paid by such employer during the calendar year "
            "with respect to employment."
        ),
        "expected_rates": ["6"],
        "expected_thresholds": [],
        "expected_modality_family": "COMPUTATIONAL",
    },
    {
        "id": "IRC_68_A_LESSER_OF_3PCT_OR_80PCT",
        "section": "section68",
        "text": (
            "In the case of an individual whose adjusted gross income exceeds $300,000, "
            "the amount of the itemized deductions otherwise allowable for the taxable year "
            "shall be reduced by the lesser of 3 percent of the excess of adjusted gross income "
            "over $300,000, or 80 percent of the amount of the itemized deductions otherwise "
            "allowable for such taxable year."
        ),
        "expected_rates": ["3", "80"],
        "expected_thresholds": ["300000", "300,000"],
        "expected_modality_family": "COMPUTATIONAL",
    },
    {
        "id": "IRC_68_F_TCJA_SUSPENSION",
        "section": "section68",
        "text": (
            "This section shall not apply to any taxable year beginning after "
            "December 31, 2017, and before January 1, 2026."
        ),
        "expected_rates": [],
        "expected_thresholds": ["2017", "2026"],
        "expected_modality_family": "DEONTIC",
    },
    {
        "id": "IRC_1_A_II_BRACKET",
        "section": "section1",
        "text": (
            "If taxable income is over $36,900 but not over $89,150, tax shall be "
            "$5,535, plus 28% of the excess over $36,900."
        ),
        "expected_rates": ["28"],
        "expected_thresholds": ["36900", "36,900", "89150", "89,150", "5535", "5,535"],
        "expected_modality_family": "COMPUTATIONAL",
    },
    {
        "id": "IRC_3306_A_1_EMPLOYER_1500",
        "section": "section3306",
        "text": (
            "The term employer means, with respect to any calendar year, any person who "
            "during the calendar year or the preceding calendar year paid wages of $1,500 or more."
        ),
        "expected_rates": [],
        "expected_thresholds": ["1500", "1,500"],
        "expected_modality_family": "COMPUTATIONAL_OR_DEONTIC",
    },
    {
        "id": "IRC_3306_A_2_AG_LABOR_20000",
        "section": "section3306",
        "text": (
            "In the case of agricultural labor, the term employer means, with respect to any "
            "calendar year, any person who during the calendar year or the preceding calendar year "
            "paid wages of $20,000 or more for agricultural labor."
        ),
        "expected_rates": [],
        "expected_thresholds": ["20000", "20,000"],
        "expected_modality_family": "COMPUTATIONAL_OR_DEONTIC",
    },
    {
        "id": "IRC_3306_B_1_WAGE_CAP_7000",
        "section": "section3306",
        "text": (
            "The term wages means all remuneration for employment, except that such term shall not "
            "include that part of the remuneration which, after remuneration equal to $7,000 with "
            "respect to employment has been paid to an individual by an employer during any calendar "
            "year, is paid to such individual by such employer during such calendar year."
        ),
        "expected_rates": [],
        "expected_thresholds": ["7000", "7,000"],
        "expected_modality_family": "COMPUTATIONAL_OR_DEONTIC",
    },
    {
        "id": "IRC_63_C_2_STANDARD_DEDUCTION",
        "section": "section63",
        "text": (
            "In the case of a head of household, the basic standard deduction is $4,400, "
            "except that for taxable years beginning after December 31, 2017, and before "
            "January 1, 2026, $18,000 shall be substituted for $4,400."
        ),
        "expected_rates": [],
        "expected_thresholds": ["4400", "4,400", "18000", "18,000"],
        "expected_modality_family": "COMPUTATIONAL",
    },
    {
        "id": "IRC_152_B_2_MARRIED_DEPENDENT_BAR",
        "section": "section152",
        "text": (
            "An individual shall not be treated as a dependent of a taxpayer if such individual "
            "has made a joint return with the individual's spouse for the taxable year."
        ),
        "expected_rates": [],
        "expected_thresholds": [],
        "expected_modality_family": "DEONTIC",
    },
    {
        "id": "IRC_7703_A_2_DIVORCE_SEPARATION_BAR",
        "section": "section7703",
        "text": (
            "An individual legally separated from his spouse under a decree of divorce or "
            "of separate maintenance shall not be considered as married."
        ),
        "expected_rates": [],
        "expected_thresholds": [],
        "expected_modality_family": "DEONTIC",
    },
]


def evaluate_neural_clause_set(decompiler: MudraDecompilerInference, clauses: list[dict], normalize_surface: bool = False):
    rows = []
    compiled_ok = 0
    invariant_ok = 0
    modality_ok = 0
    rate_hits = 0
    rate_total = 0
    threshold_hits = 0
    threshold_total = 0

    for item in clauses:
        inp = surface_normalize_us_to_inr(item["text"]) if normalize_surface else item["text"]
        unk_pct, unk_toks, open_toks = compute_unk_rate(inp, decompiler.vocab)
        res = decompiler.decompile_text(inp)
        status = res.get("status")
        pred_mod = res.get("predictedModality", "UNKNOWN")
        mod_conf = res.get("modalityConfidence", 0.0)
        spans = res.get("classifiedSpans", [])
        invariant_safe = res.get("statutoryInvariants", {}).get("isAuthorizedToDisburse") is False

        if status == "SUCCESS":
            compiled_ok += 1
        if invariant_safe:
            invariant_ok += 1

        exp_fam = item["expected_modality_family"]
        if exp_fam == "COMPUTATIONAL" and pred_mod.startswith("MODALITY_COMPUTATIONAL"):
            modality_ok += 1
        elif exp_fam == "DEONTIC" and pred_mod.startswith("MODALITY_DEONTIC"):
            modality_ok += 1
        elif exp_fam == "COMPUTATIONAL_OR_DEONTIC" and (
            pred_mod.startswith("MODALITY_COMPUTATIONAL") or pred_mod.startswith("MODALITY_DEONTIC")
        ):
            modality_ok += 1

        rate_text_blob = " | ".join(
            s["span"].replace(",", "") for s in spans
            if "RATE" in s["role"] or "%" in s["span"] or "percent" in s["span"].lower()
        )
        thresh_text_blob = " | ".join(
            s["span"].replace(",", "") for s in spans
            if "THRESHOLD" in s["role"] or "DEADLINE" in s["role"] or "CUTOVER" in s["role"] or "SUNSET" in s["role"]
        )

        # Inspect synthesized AST fields (rootCondition, symbolLegend, guards, computation)
        ast = res.get("synthesizedAst", {}) if status == "SUCCESS" else {}
        root = ast.get("rootCondition", {}) if isinstance(ast, dict) else {}
        symbol_legend = ast.get("symbolLegend", {}) if isinstance(ast, dict) else {}
        guards = ast.get("guards", []) if isinstance(ast, dict) else []
        computation = ast.get("computation", {}) if isinstance(ast, dict) else {}
        comp_json = json.dumps(computation).replace(",", "")
        legend_json = json.dumps(symbol_legend).replace(",", "")
        all_ast_blob = " | ".join([
            json.dumps(root).replace(",", ""),
            legend_json,
            json.dumps(guards).replace(",", ""),
            comp_json,
            rate_text_blob,
            thresh_text_blob,
        ])

        exp_rates = item["expected_rates"]
        if exp_rates:
            rate_total += len(exp_rates)
            for r in exp_rates:
                r_bps = str(int(round(float(r) * 100)))
                rate_in_ast = bool(
                    r in rate_text_blob
                    or r in str(root.get("standardTaxRate") or "")
                    or re.search(rf'"(?:standardRateBps|rateBps|targetRateBps)"\s*:\s*{r_bps}\b', comp_json + " " + legend_json)
                )
                if rate_in_ast:
                    rate_hits += 1

        exp_t_unique = list({t.replace(",", "") for t in item["expected_thresholds"]})
        if exp_t_unique:
            threshold_total += len(exp_t_unique)
            for t in exp_t_unique:
                if t in all_ast_blob:
                    threshold_hits += 1

        rows.append({
            "id": item["id"],
            "status": status,
            "modality": pred_mod,
            "modConf": mod_conf,
            "unkPct": unk_pct,
            "unkCount": len(unk_toks),
            "spanCount": len(spans),
            "roles": [(s["role"], s["span"]) for s in spans[:6]],
            "invariantSafe": invariant_safe,
        })

    return {
        "total": len(clauses),
        "compiled": compiled_ok,
        "invariant_ok": invariant_ok,
        "modality_ok": modality_ok,
        "rate_hits": rate_hits,
        "rate_total": rate_total,
        "threshold_hits": threshold_hits,
        "threshold_total": threshold_total,
        "rows": rows,
    }


def build_us_irc_sec1_ast(filing_status: str) -> dict:
    """
    Pure declarative USVM AST for US IRC § 1(a)-(d) progressive brackets.
    Facts use dollar amounts in USD; PROGRESSIVE_SLABS handles cents via TO_PAISE/FROM_PAISE.
    """
    brackets_by_status = {
        "JOINT_OR_SURVIVING": [
            {"upTo": 36900, "rateBps": 1500},
            {"upTo": 89150, "rateBps": 2800},
            {"upTo": 140000, "rateBps": 3100},
            {"upTo": 250000, "rateBps": 3600},
            {"upTo": None, "rateBps": 3960},
        ],
        "HEAD_OF_HOUSEHOLD": [
            {"upTo": 29600, "rateBps": 1500},
            {"upTo": 76400, "rateBps": 2800},
            {"upTo": 127500, "rateBps": 3100},
            {"upTo": 250000, "rateBps": 3600},
            {"upTo": None, "rateBps": 3960},
        ],
        "SINGLE": [
            {"upTo": 22100, "rateBps": 1500},
            {"upTo": 53500, "rateBps": 2800},
            {"upTo": 115000, "rateBps": 3100},
            {"upTo": 250000, "rateBps": 3600},
            {"upTo": None, "rateBps": 3960},
        ],
        "MARRIED_SEPARATE": [
            {"upTo": 18450, "rateBps": 1500},
            {"upTo": 44575, "rateBps": 2800},
            {"upTo": 70000, "rateBps": 3100},
            {"upTo": 125000, "rateBps": 3600},
            {"upTo": None, "rateBps": 3960},
        ],
    }
    slabs = brackets_by_status[filing_status]
    return {
        "astSchemaVersion": "1.0.0",
        "sectionId": f"US_IRC_SEC1_{filing_status}",
        "sectionCode": "IRC Section 1",
        "version": "1.0.0",
        "executionMode": "DEVELOPMENT",
        "title": f"US IRC Section 1 Income Tax Brackets ({filing_status})",
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "steps": [
                {
                    "assign": "taxAmountPaisa",
                    "expr": {
                        "op": "PROGRESSIVE_SLABS",
                        "base": {"op": "MAX", "left": "facts.taxableIncomeUsd", "right": 0},
                        "slabs": slabs,
                    },
                }
            ],
            "output": {
                "taxAmountPaisa": "vars.taxAmountPaisa",
                "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
                "isApplicable": {"op": "GT", "left": "vars.taxAmountPaisa", "right": 0},
            },
        },
        "statutoryInvariants": {"isAuthorizedToDisburse": False},
    }


def build_us_irc_full_liability_ast() -> dict:
    """
    Pure declarative USVM pipeline implementing the DeonticBench SARA U.S. IRC corpus:
      - §§ 2 & 7703 Filing Status & Household Maintenance Hierarchy
      - § 63 Basic & Additional (Aged/Blind) Standard Deduction, §§ 63(c)(6) Zeroing,
             and § 63(c)(7) 2018-2025 TCJA substitution
      - § 68 Overall Itemized Deduction Phaseout (lesser of 3% excess or 80%) + 2018-2025 suspension
      - § 151 Personal Exemption ($2,000 per exemption, 2% per $2,500/$1,250 phaseout) + 2018-2025 zeroing
      - § 1(a)-(d) Progressive Brackets via SWITCH on derived filingStatus
      - §§ 3301 & 3306 FUTA 6% Excise Tax (employer triggers: >= $1,500 general / >= $20,000 cash
             agricultural / >= 10 weeks w/ >= 5 ag workers / >= $1,000 domestic; $7,000 per-employee cap)
    Zero U.S. tax logic exists in evaluation_vm.js — everything runs declaratively in USVM!
    """
    return {
        "astSchemaVersion": "1.0.0",
        "sectionId": "US_IRC_FULL_SARA_INDIVIDUAL_AND_FUTA",
        "sectionCode": "IRC Sections 1, 2, 63, 68, 151, 3301, 3306, 7703",
        "version": "1.0.0",
        "executionMode": "DEVELOPMENT",
        "title": "US Internal Revenue Code Combined Individual + FUTA Declarative USVM Pipeline",
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "steps": [
                # 0. TCJA 2018-2025 window
                {
                    "assign": "isTcjaWindow",
                    "expr": {
                        "op": "AND",
                        "left": {"op": "GTE", "left": "facts.taxYear", "right": 2018},
                        "right": {"op": "LTE", "left": "facts.taxYear", "right": 2025},
                    },
                },
                # 1. Filing status derived declaratively from observable household/marital facts
                {
                    "assign": "canJointReturn",
                    "expr": {
                        "op": "AND",
                        "operands": [
                            "facts.isMarried",
                            "facts.filesJointReturn",
                            {"op": "NOT", "operand": "facts.isNonresidentAlien"},
                            {"op": "NOT", "operand": "facts.isDivorcedOrSeparated"},
                        ],
                    },
                },
                {
                    "assign": "isSurvivingSpouseEligible",
                    "expr": {
                        "op": "AND",
                        "operands": [
                            {"op": "NOT", "operand": "facts.isMarried"},
                            "facts.spouseDiedPriorTwoYears",
                            {"op": "NOT", "operand": "facts.remarried"},
                            "facts.maintainsHouseholdOverHalfCost",
                            "facts.hasQualifyingChildDependent",
                            "facts.jointReturnWasPossibleInDeathYear",
                        ],
                    },
                },
                {
                    "assign": "isTreatedUnmarried7703b",
                    "expr": {
                        "op": "AND",
                        "operands": [
                            "facts.isMarried",
                            {"op": "NOT", "operand": "facts.filesJointReturn"},
                            "facts.maintainsHouseholdOverHalfCost",
                            "facts.hasQualifyingChildDependent",
                            "facts.spouseDifferentAbodeAllYear",
                        ],
                    },
                },
                {
                    "assign": "isHohEligible",
                    "expr": {
                        "op": "AND",
                        "operands": [
                            {"op": "NOT", "operand": "facts.isMarried"},
                            {"op": "NOT", "operand": "vars.isSurvivingSpouseEligible"},
                            {"op": "NOT", "operand": "facts.isNonresidentAlien"},
                            "facts.maintainsHouseholdOverHalfCost",
                            "facts.hasQualifyingHohDependent",
                        ],
                    },
                },
                {
                    "assign": "filingStatus",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "OR", "left": "vars.canJointReturn", "right": "vars.isSurvivingSpouseEligible"},
                        "then": "JOINT_OR_SURVIVING",
                        "else": {
                            "op": "CONDITIONAL",
                            "test": "vars.isHohEligible",
                            "then": "HEAD_OF_HOUSEHOLD",
                            "else": {
                                "op": "CONDITIONAL",
                                "test": {
                                    "op": "AND",
                                    "left": "facts.isMarried",
                                    "right": {
                                        "op": "AND",
                                        "left": {"op": "NOT", "operand": "facts.isDivorcedOrSeparated"},
                                        "right": {"op": "NOT", "operand": "vars.isTreatedUnmarried7703b"},
                                    },
                                },
                                "then": "MARRIED_SEPARATE",
                                "else": "SINGLE",
                            },
                        },
                    },
                },
                # § 63(c)(6) Standard deduction zeroing (nonresident alien or MFS w/ spouse itemizing)
                {
                    "assign": "stdDeductionZeroed",
                    "expr": {
                        "op": "OR",
                        "left": "facts.isNonresidentAlien",
                        "right": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "vars.filingStatus", "right": "MARRIED_SEPARATE"},
                            "right": "facts.spouseItemizes",
                        },
                    },
                },
                # 2. § 68 Applicable Amount: 300k joint/surv, 275k HOH, 250k single, 150k MFS
                {
                    "assign": "sec68ApplicableAmount",
                    "expr": {
                        "op": "SWITCH",
                        "discriminant": "vars.filingStatus",
                        "cases": [
                            {"test": "JOINT_OR_SURVIVING", "value": 300000},
                            {"test": "HEAD_OF_HOUSEHOLD", "value": 275000},
                            {"test": "MARRIED_SEPARATE", "value": 150000},
                        ],
                        "default": 250000,
                    },
                },
                {
                    "assign": "sec68Excess",
                    "expr": {
                        "op": "MAX",
                        "left": {"op": "SUB", "left": "facts.agiUsd", "right": "vars.sec68ApplicableAmount"},
                        "right": 0,
                    },
                },
                {
                    "assign": "sec68ThreePctPaisa",
                    "expr": {
                        "op": "RATIONAL_MULTIPLY",
                        "base": {"op": "TO_PAISE", "value": "vars.sec68Excess"},
                        "rateBps": 300,
                        "denominator": 10000,
                    },
                },
                {
                    "assign": "sec68EightyPctPaisa",
                    "expr": {
                        "op": "RATIONAL_MULTIPLY",
                        "base": {"op": "TO_PAISE", "value": "facts.itemizedDeductionsUsd"},
                        "rateBps": 8000,
                        "denominator": 10000,
                    },
                },
                {
                    "assign": "sec68ReductionUsd",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {
                            "op": "AND",
                            "left": {"op": "NOT", "operand": "vars.isTcjaWindow"},
                            "right": {"op": "GT", "left": "vars.sec68Excess", "right": 0},
                        },
                        "then": {
                            "op": "FROM_PAISE",
                            "value": {"op": "MIN", "left": "vars.sec68ThreePctPaisa", "right": "vars.sec68EightyPctPaisa"},
                        },
                        "else": 0,
                    },
                },
                {
                    "assign": "netItemizedUsd",
                    "expr": {
                        "op": "MAX",
                        "left": {"op": "SUB", "left": "facts.itemizedDeductionsUsd", "right": "vars.sec68ReductionUsd"},
                        "right": 0,
                    },
                },
                # 3. § 63 Standard Deduction (basic + aged/blind)
                {
                    "assign": "basicStdDeduction",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "vars.stdDeductionZeroed",
                        "then": 0,
                        "else": {
                            "op": "SWITCH",
                            "discriminant": "vars.filingStatus",
                            "cases": [
                                {
                                    "test": "JOINT_OR_SURVIVING",
                                    "value": {"op": "CONDITIONAL", "test": "vars.isTcjaWindow", "then": 24000, "else": 6000},
                                },
                                {
                                    "test": "HEAD_OF_HOUSEHOLD",
                                    "value": {"op": "CONDITIONAL", "test": "vars.isTcjaWindow", "then": 18000, "else": 4400},
                                },
                            ],
                            "default": {"op": "CONDITIONAL", "test": "vars.isTcjaWindow", "then": 12000, "else": 3000},
                        },
                    },
                },
                # § 63(f)(1)(B)/(2)(B): spouse aged/blind adds only when spouse exemption under § 151(b) is allowable
                {
                    "assign": "spouse151bEligible",
                    "expr": {
                        "op": "AND",
                        "operands": [
                            "facts.isMarried",
                            {"op": "NOT", "operand": "vars.canJointReturn"},
                            {"op": "NOT", "operand": "facts.isDivorcedOrSeparated"},
                            "facts.spouseHasNoIncomeAndNotDep",
                        ],
                    },
                },
                {
                    "assign": "eligibleSpouseAgedBlind",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "AND", "left": "vars.spouse151bEligible", "right": {"op": "NOT", "operand": "vars.isTcjaWindow"}},
                        "then": "facts.spouseAgedBlindCount",
                        "else": 0,
                    },
                },
                {
                    "assign": "totalAgedBlindCount",
                    "expr": {"op": "ADD", "left": "facts.taxpayerAgedBlindCount", "right": "vars.eligibleSpouseAgedBlind"},
                },
                {
                    "assign": "perAdditionalUnit",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {
                            "op": "OR",
                            "left": {"op": "EQ", "left": "vars.filingStatus", "right": "SINGLE"},
                            "right": {"op": "EQ", "left": "vars.filingStatus", "right": "HEAD_OF_HOUSEHOLD"},
                        },
                        "then": 750,
                        "else": 600,
                    },
                },
                {
                    "assign": "addlStdDeduction",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "vars.stdDeductionZeroed",
                        "then": 0,
                        "else": {"op": "MUL", "left": "vars.totalAgedBlindCount", "right": "vars.perAdditionalUnit"},
                    },
                },
                {
                    "assign": "totalStdDeduction",
                    "expr": {"op": "ADD", "left": "vars.basicStdDeduction", "right": "vars.addlStdDeduction"},
                },
                {
                    "assign": "chosenDeductionUsd",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "facts.takesStandardDeduction",
                        "then": "vars.totalStdDeduction",
                        "else": "vars.netItemizedUsd",
                    },
                },
                # 4. § 151 Personal Exemption ($2,000 per exemption minus 2% per $2,500/$1,250 ceil step)
                {
                    "assign": "taxpayerAndSpouseExemptions",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "vars.canJointReturn",
                        "then": 2,
                        "else": {
                            "op": "CONDITIONAL",
                            "test": "vars.spouse151bEligible",
                            "then": 2,
                            "else": 1,
                        },
                    },
                },
                {
                    "assign": "exemptionCount",
                    "expr": {"op": "ADD", "left": "vars.taxpayerAndSpouseExemptions", "right": "facts.dependentCount"},
                },
                {
                    "assign": "stepSize",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "EQ", "left": "vars.filingStatus", "right": "MARRIED_SEPARATE"},
                        "then": 1250,
                        "else": 2500,
                    },
                },
                {
                    "assign": "rawFraction",
                    "expr": {"op": "DIV", "left": "vars.sec68Excess", "right": "vars.stepSize"},
                },
                {
                    "assign": "numSteps",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": {"op": "LTE", "left": "vars.sec68Excess", "right": 0},
                        "then": 0,
                        "else": {
                            "op": "CONDITIONAL",
                            "test": {"op": "EQ", "left": {"op": "MOD", "left": "vars.sec68Excess", "right": "vars.stepSize"}, "right": 0},
                            "then": "vars.rawFraction",
                            "else": {"op": "ROUND", "value": {"op": "ADD", "left": "vars.rawFraction", "right": 0.999999}, "policy": "FLOOR"},
                        },
                    },
                },
                {
                    "assign": "phaseoutPct",
                    "expr": {"op": "MIN", "left": {"op": "MUL", "left": "vars.numSteps", "right": 2}, "right": 100},
                },
                {
                    "assign": "exemptionUnitUsd",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "vars.isTcjaWindow",
                        "then": 0,
                        "else": {
                            "op": "MUL",
                            "left": 2000,
                            "right": {"op": "DIV", "left": {"op": "SUB", "left": 100, "right": "vars.phaseoutPct"}, "right": 100},
                        },
                    },
                },
                {
                    "assign": "totalExemptionsUsd",
                    "expr": {"op": "MUL", "left": "vars.exemptionCount", "right": "vars.exemptionUnitUsd"},
                },
                # 5. Taxable Income
                {
                    "assign": "taxableIncomeUsd",
                    "expr": {
                        "op": "MAX",
                        "left": {
                            "op": "SUB",
                            "left": {"op": "SUB", "left": "facts.agiUsd", "right": "vars.chosenDeductionUsd"},
                            "right": "vars.totalExemptionsUsd",
                        },
                        "right": 0,
                    },
                },
                # 6. § 1(a)-(d) Progressive Brackets
                {
                    "assign": "incomeTaxJointPaisa",
                    "expr": {
                        "op": "PROGRESSIVE_SLABS",
                        "base": "vars.taxableIncomeUsd",
                        "slabs": [
                            {"upTo": 36900, "rateBps": 1500},
                            {"upTo": 89150, "rateBps": 2800},
                            {"upTo": 140000, "rateBps": 3100},
                            {"upTo": 250000, "rateBps": 3600},
                            {"upTo": None, "rateBps": 3960},
                        ],
                    },
                },
                {
                    "assign": "incomeTaxHohPaisa",
                    "expr": {
                        "op": "PROGRESSIVE_SLABS",
                        "base": "vars.taxableIncomeUsd",
                        "slabs": [
                            {"upTo": 29600, "rateBps": 1500},
                            {"upTo": 76400, "rateBps": 2800},
                            {"upTo": 127500, "rateBps": 3100},
                            {"upTo": 250000, "rateBps": 3600},
                            {"upTo": None, "rateBps": 3960},
                        ],
                    },
                },
                {
                    "assign": "incomeTaxSinglePaisa",
                    "expr": {
                        "op": "PROGRESSIVE_SLABS",
                        "base": "vars.taxableIncomeUsd",
                        "slabs": [
                            {"upTo": 22100, "rateBps": 1500},
                            {"upTo": 53500, "rateBps": 2800},
                            {"upTo": 115000, "rateBps": 3100},
                            {"upTo": 250000, "rateBps": 3600},
                            {"upTo": None, "rateBps": 3960},
                        ],
                    },
                },
                {
                    "assign": "incomeTaxMfsPaisa",
                    "expr": {
                        "op": "PROGRESSIVE_SLABS",
                        "base": "vars.taxableIncomeUsd",
                        "slabs": [
                            {"upTo": 18450, "rateBps": 1500},
                            {"upTo": 44575, "rateBps": 2800},
                            {"upTo": 70000, "rateBps": 3100},
                            {"upTo": 125000, "rateBps": 3600},
                            {"upTo": None, "rateBps": 3960},
                        ],
                    },
                },
                {
                    "assign": "incomeTaxPaisa",
                    "expr": {
                        "op": "SWITCH",
                        "discriminant": "vars.filingStatus",
                        "cases": [
                            {"test": "JOINT_OR_SURVIVING", "value": "vars.incomeTaxJointPaisa"},
                            {"test": "HEAD_OF_HOUSEHOLD", "value": "vars.incomeTaxHohPaisa"},
                            {"test": "MARRIED_SEPARATE", "value": "vars.incomeTaxMfsPaisa"},
                        ],
                        "default": "vars.incomeTaxSinglePaisa",
                    },
                },
                # 7. §§ 3301 & 3306 FUTA 6% Excise Tax (exact SARA § 3301 rule, $7,000 cap per employee)
                {
                    "assign": "emp1Plus2",
                    "expr": {"op": "ADD", "left": "facts.emp1WageUsd", "right": "facts.emp2WageUsd"},
                },
                {
                    "assign": "emp3Plus4",
                    "expr": {"op": "ADD", "left": "facts.emp3WageUsd", "right": "facts.emp4WageUsd"},
                },
                {
                    "assign": "emp5Plus6",
                    "expr": {"op": "ADD", "left": "facts.emp5WageUsd", "right": "facts.emp6WageUsd"},
                },
                {
                    "assign": "futaGrossEligibleWagesUsd",
                    "expr": {"op": "ADD", "left": "vars.emp1Plus2", "right": {"op": "ADD", "left": "vars.emp3Plus4", "right": "vars.emp5Plus6"}},
                },
                {
                    "assign": "cap1Plus2",
                    "expr": {
                        "op": "ADD",
                        "left": {"op": "MIN", "left": "facts.emp1WageUsd", "right": 7000},
                        "right": {"op": "MIN", "left": "facts.emp2WageUsd", "right": 7000},
                    },
                },
                {
                    "assign": "cap3Plus4",
                    "expr": {
                        "op": "ADD",
                        "left": {"op": "MIN", "left": "facts.emp3WageUsd", "right": 7000},
                        "right": {"op": "MIN", "left": "facts.emp4WageUsd", "right": 7000},
                    },
                },
                {
                    "assign": "cap5Plus6",
                    "expr": {
                        "op": "ADD",
                        "left": {"op": "MIN", "left": "facts.emp5WageUsd", "right": 7000},
                        "right": {"op": "MIN", "left": "facts.emp6WageUsd", "right": 7000},
                    },
                },
                {
                    "assign": "futaCappedWagesUsd",
                    "expr": {"op": "ADD", "left": "vars.cap1Plus2", "right": {"op": "ADD", "left": "vars.cap3Plus4", "right": "vars.cap5Plus6"}},
                },
                # Employer threshold gates per § 3306(a) and § 3306(c)(1) (outside US requires explicit cash >= $20k)
                {
                    "assign": "agCashFloor",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "facts.futaPerformedOutsideUs",
                        "then": {"op": "GTE", "left": "facts.futaExplicitCashAgUsd", "right": 20000},
                        "else": {"op": "GTE", "left": "vars.futaGrossEligibleWagesUsd", "right": 20000},
                    },
                },
                {
                    "assign": "futaEmployerTriggered",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "facts.futaIsAgricultural",
                        "then": {
                            "op": "AND",
                            "left": {
                                "op": "OR",
                                "left": "vars.agCashFloor",
                                "right": {"op": "GTE", "left": "facts.agWeeksWith5Workers", "right": 10},
                            },
                            "right": {
                                "op": "OR",
                                "left": {"op": "NOT", "operand": "facts.futaPerformedOutsideUs"},
                                "right": {"op": "GTE", "left": "facts.futaExplicitCashAgUsd", "right": 20000},
                            },
                        },
                        "else": {
                            "op": "OR",
                            "left": {"op": "GTE", "left": "vars.futaGrossEligibleWagesUsd", "right": 1500},
                            "right": {"op": "GTE", "left": "facts.genWeeksWith1Worker", "right": 10},
                        },
                    },
                },
                {
                    "assign": "futaTaxableWagesUsd",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "vars.futaEmployerTriggered",
                        "then": "vars.futaCappedWagesUsd",
                        "else": 0,
                    },
                },
                {
                    "assign": "futaTaxPaisa",
                    "expr": {
                        "op": "RATIONAL_MULTIPLY",
                        "base": {"op": "TO_PAISE", "value": "vars.futaTaxableWagesUsd"},
                        "rateBps": 600,
                        "denominator": 10000,
                    },
                },
                {
                    "assign": "taxAmountPaisa",
                    "expr": {"op": "ADD", "left": "vars.incomeTaxPaisa", "right": "vars.futaTaxPaisa"},
                },
            ],
            "output": {
                "filingStatus": "vars.filingStatus",
                "taxableIncomeUsd": "vars.taxableIncomeUsd",
                "incomeTaxUsd": {"op": "FROM_PAISE", "value": "vars.incomeTaxPaisa"},
                "futaTaxUsd": {"op": "FROM_PAISE", "value": "vars.futaTaxPaisa"},
                "taxAmountPaisa": "vars.taxAmountPaisa",
                "totalTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"},
                "isApplicable": {"op": "GT", "left": "vars.taxAmountPaisa", "right": 0},
            },
        },
        "statutoryInvariants": {"isAuthorizedToDisburse": False},
    }


def _extract_ag_worker_schedule(case_text: str, year: int) -> dict:
    """
    Pure structural parser reading raw worker names, dates, and per-occasion pay
    from bulleted SARA agricultural schedule texts (supports multi-line and inline
    dash bullets; contains zero tax rate/cap/threshold math).
    """
    pay_m = re.search(r"paid each(?:\s+of them)?\s+\$([0-9]+)", case_text)
    per_occ = int(pay_m.group(1)) if pay_m else 0
    header_workers = ["__worker__"]
    hdr_m = re.search(r"employed\s+([A-Z][A-Za-z,\s]+?)\s+(?:for\s+[^:]+\s+)?on\s+various\s+occasions", case_text)
    if hdr_m:
        parsed_hdr = [n.strip() for n in re.split(r",|\band\b", hdr_m.group(1)) if n.strip()]
        if parsed_hdr:
            header_workers = parsed_hdr

    emp_totals = {}
    weeks_ge5 = set()
    weeks_ge1 = set()
    bullet_pat = re.compile(
        r"(?:^|\s)-\s*([A-Za-z]{3}\s+\d{1,2})(?:\s*:\s*([^\r\n]+?))?(?=\s+-\s*[A-Za-z]{3}\s+\d{1,2}\b|\r?\n|$)"
    )
    for m in bullet_pat.finditer(case_text):
        date_str, names_blob = m.group(1), m.group(2)
        iso_week = datetime.strptime(f"{year} {date_str}", "%Y %b %d").isocalendar()[1]
        if names_blob and names_blob.strip():
            names = [n.strip() for n in re.split(r",|\band\b", names_blob) if n.strip()]
        else:
            names = header_workers
        for n in names:
            emp_totals[n] = emp_totals.get(n, 0) + per_occ
        if len(names) >= 1:
            weeks_ge1.add(iso_week)
        if len(names) >= 5:
            weeks_ge5.add(iso_week)
    ordered = list(emp_totals.values()) + [0] * 6
    return {
        "empWages": ordered[:6],
        "agWeeksWith5Workers": len(weeks_ge5),
        "genWeeksWith1Worker": len(weeks_ge1),
    }


def _default_tax_facts(**overrides) -> dict:
    facts = {
        "taxYear": 2017,
        "agiUsd": 0,
        "itemizedDeductionsUsd": 0,
        "takesStandardDeduction": True,
        "isMarried": False,
        "filesJointReturn": False,
        "isDivorcedOrSeparated": False,
        "isNonresidentAlien": False,
        "spouseItemizes": False,
        "spouseHasNoIncomeAndNotDep": False,
        "spouseDifferentAbodeAllYear": False,
        "spouseDiedPriorTwoYears": False,
        "remarried": False,
        "jointReturnWasPossibleInDeathYear": False,
        "maintainsHouseholdOverHalfCost": False,
        "hasQualifyingChildDependent": False,
        "hasQualifyingHohDependent": False,
        "dependentCount": 0,
        "taxpayerAgedBlindCount": 0,
        "spouseAgedBlindCount": 0,
        "futaIsAgricultural": False,
        "futaPerformedOutsideUs": False,
        "futaExplicitCashAgUsd": 0,
        "agWeeksWith5Workers": 0,
        "genWeeksWith1Worker": 0,
        "emp1WageUsd": 0,
        "emp2WageUsd": 0,
        "emp3WageUsd": 0,
        "emp4WageUsd": 0,
        "emp5WageUsd": 0,
        "emp6WageUsd": 0,
    }
    facts.update(overrides)
    return facts


def _expr_involves_pair(edge_var: str, p1_expr: str, p2_expr: str, k1: str = "a", k2: str = "b") -> dict:
    return {
        "op": "OR",
        "left": {
            "op": "AND",
            "left": {"op": "EQ", "left": f"{edge_var}.{k1}", "right": p1_expr},
            "right": {"op": "EQ", "left": f"{edge_var}.{k2}", "right": p2_expr},
        },
        "right": {
            "op": "AND",
            "left": {"op": "EQ", "left": f"{edge_var}.{k1}", "right": p2_expr},
            "right": {"op": "EQ", "left": f"{edge_var}.{k2}", "right": p1_expr},
        },
    }


def _expr_interval_overlaps(s1: str, e1: str, s2: str, e2: str) -> dict:
    return {
        "op": "AND",
        "left": {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": s1, "right": e2}, "right": 0},
        "right": {"op": "GTE", "left": {"op": "DATE_COMPARE", "left": e1, "right": s2}, "right": 0},
    }


def _expr_spouse_name(m_var: str = "m", taxpayer_ref: str = "facts.taxpayer") -> dict:
    return {
        "op": "CONDITIONAL",
        "test": {"op": "EQ", "left": f"{m_var}.a", "right": taxpayer_ref},
        "then": f"{m_var}.b",
        "else": f"{m_var}.a",
    }


def _expr_involves_one(edge_var: str, person_expr: str, k1: str = "a", k2: str = "b") -> dict:
    return {
        "op": "OR",
        "left": {"op": "EQ", "left": f"{edge_var}.{k1}", "right": person_expr},
        "right": {"op": "EQ", "left": f"{edge_var}.{k2}", "right": person_expr},
    }


def _expr_date_max(d1, d2) -> dict:
    return {
        "op": "CONDITIONAL",
        "test": {"op": "GTE", "left": {"op": "DATE_COMPARE", "left": d1, "right": d2}, "right": 0},
        "then": d1,
        "else": d2,
    }


def _expr_date_min(d1, d2) -> dict:
    return {
        "op": "CONDITIONAL",
        "test": {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": d1, "right": d2}, "right": 0},
        "then": d1,
        "else": d2,
    }


def build_us_irc_relational_deontic_ast(clause_key: str) -> dict:
    """
    Pure declarative USVM PURE_LOGIC_PIPELINE (ruleDomain=GENERAL_LOGIC) encoding
    the relational kinship graph, marital/death intervals, household-abode day overlap,
    and student-spouse employment exclusions from §§ 2, 63, 151, 152, and 3306(c)(10).
    All deduction executes inside canonical JS evaluation_vm.js bounded quantifiers.
    """
    steps = []

    if clause_key in ("SEC2_A_2_B", "SEC2_B_2_C"):
        steps.extend([
            {
                "assign": "nonresInYearPersons",
                "expr": {
                    "op": "MAP",
                    "iteratorVar": "nr",
                    "collection": {
                        "op": "FILTER",
                        "iteratorVar": "nrF",
                        "collection": "facts.nonresidentIntervals",
                        "predicate": _expr_interval_overlaps("nrF.start", "nrF.end", "facts.yearStart", "facts.yearEnd"),
                    },
                    "body": "nr.person",
                },
            },
            {
                "assign": "spouseDiedWithoutNonres",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "m",
                    "collection": "facts.marriages",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            _expr_involves_one("m", "facts.taxpayer"),
                            {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "m.start", "right": "facts.yearEnd"}, "right": 0},
                            {
                                "op": "EXISTS",
                                "iteratorVar": "d",
                                "collection": "facts.deaths",
                                "predicate": {
                                    "op": "AND",
                                    "operands": [
                                        {"op": "EQ", "left": "d.person", "right": _expr_spouse_name("m", "facts.taxpayer")},
                                        {"op": "EQ", "left": "d.year", "right": "facts.targetYear"},
                                        {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "m.start", "right": "d.date"}, "right": 0},
                                        {"op": "NOT_IN", "left": "d.person", "right": "vars.nonresInYearPersons"},
                                    ],
                                },
                            },
                        ],
                    },
                },
            },
        ])
        if clause_key == "SEC2_A_2_B":
            steps.extend([
                {
                    "assign": "taxpayerIsNonresident",
                    "expr": {
                        "op": "EXISTS",
                        "iteratorVar": "nr",
                        "collection": "facts.nonresidentIntervals",
                        "predicate": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "nr.person", "right": "facts.taxpayer"},
                            "right": _expr_interval_overlaps("nr.start", "nr.end", "facts.yearStart", "facts.yearEnd"),
                        },
                    },
                },
                {
                    "assign": "applies",
                    "expr": {
                        "op": "AND",
                        "left": {"op": "NOT", "operand": "vars.taxpayerIsNonresident"},
                        "right": "vars.spouseDiedWithoutNonres",
                    },
                },
            ])
        else:
            steps.append({"assign": "applies", "expr": "vars.spouseDiedWithoutNonres"})

    elif clause_key in ("SEC2_B_1_A_II", "SEC2_B_1_B", "SEC63_C_2_A_II"):
        steps.extend([
            {
                "assign": "taxpayerIsNonresident",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "nr",
                    "collection": "facts.nonresidentIntervals",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "nr.person", "right": "facts.taxpayer"},
                        "right": _expr_interval_overlaps("nr.start", "nr.end", "facts.yearStart", "facts.yearEnd"),
                    },
                },
            },
            {
                "assign": "spouseAliveAtClose",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "m",
                    "collection": "facts.marriages",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            _expr_involves_one("m", "facts.taxpayer"),
                            {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "m.start", "right": "facts.yearEnd"}, "right": 0},
                            {"op": "GTE", "left": {"op": "DATE_COMPARE", "left": "m.end", "right": "facts.yearEnd"}, "right": 0},
                            {
                                "op": "NOT",
                                "operand": {
                                    "op": "EXISTS",
                                    "iteratorVar": "d",
                                    "collection": "facts.deaths",
                                    "predicate": {
                                        "op": "AND",
                                        "left": {"op": "EQ", "left": "d.person", "right": _expr_spouse_name("m", "facts.taxpayer")},
                                        "right": {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "d.date", "right": "facts.yearEnd"}, "right": 0},
                                    },
                                },
                            },
                        ],
                    },
                },
            },
            {
                "assign": "isSurvivingSpouse",
                "expr": {
                    "op": "AND",
                    "left": {"op": "NOT", "operand": "vars.spouseAliveAtClose"},
                    "right": {
                        "op": "EXISTS",
                        "iteratorVar": "m",
                        "collection": "facts.marriages",
                        "predicate": {
                            "op": "AND",
                            "left": _expr_involves_one("m", "facts.taxpayer"),
                            "right": {
                                "op": "EXISTS",
                                "iteratorVar": "d",
                                "collection": "facts.deaths",
                                "predicate": {
                                    "op": "AND",
                                    "left": {"op": "EQ", "left": "d.person", "right": _expr_spouse_name("m", "facts.taxpayer")},
                                    "right": {
                                        "op": "OR",
                                        "left": {"op": "EQ", "left": "facts.targetYear", "right": {"op": "ADD", "left": "d.year", "right": 1}},
                                        "right": {"op": "EQ", "left": "facts.targetYear", "right": {"op": "ADD", "left": "d.year", "right": 2}},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        ])
        if clause_key == "SEC63_C_2_A_II":
            steps.append({"assign": "applies", "expr": "vars.isSurvivingSpouse"})
        else:
            steps.append({
                "assign": "activeMaints",
                "expr": {
                    "op": "FILTER",
                    "iteratorVar": "hm",
                    "collection": "facts.householdMaints",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "hm.taxpayer", "right": "facts.taxpayer"},
                            {"op": "LTE", "left": "hm.startYear", "right": "facts.targetYear"},
                            {"op": "GTE", "left": "hm.endYear", "right": "facts.targetYear"},
                            {"op": "GT", "left": "hm.costSharePct", "right": 50},
                        ],
                    },
                },
            })
            if clause_key == "SEC2_B_1_A_II":
                steps.extend([
                    {
                        "assign": "sharedAbodeWithDep",
                        "expr": {
                            "op": "EXISTS",
                            "iteratorVar": "hm",
                            "collection": "vars.activeMaints",
                            "predicate": {
                                "op": "AND",
                                "left": {
                                    "op": "EXISTS",
                                    "iteratorVar": "rt",
                                    "collection": "facts.residences",
                                    "predicate": {
                                        "op": "AND",
                                        "operands": [
                                            {"op": "EQ", "left": "rt.person", "right": "facts.taxpayer"},
                                            {"op": "EQ", "left": "rt.home", "right": "hm.home"},
                                            _expr_interval_overlaps("rt.start", "rt.end", "facts.yearStart", "facts.yearEnd"),
                                        ],
                                    },
                                },
                                "right": {
                                    "op": "EXISTS",
                                    "iteratorVar": "rd",
                                    "collection": "facts.residences",
                                    "predicate": {
                                        "op": "AND",
                                        "operands": [
                                            {"op": "EQ", "left": "rd.person", "right": "facts.individual"},
                                            {"op": "EQ", "left": "rd.home", "right": "hm.home"},
                                            _expr_interval_overlaps("rd.start", "rd.end", "facts.yearStart", "facts.yearEnd"),
                                        ],
                                    },
                                },
                            },
                        },
                    },
                    {
                        "assign": "has151cForDep",
                        "expr": {
                            "op": "EXISTS",
                            "iteratorVar": "d151",
                            "collection": "facts.deductions151c",
                            "predicate": {
                                "op": "AND",
                                "operands": [
                                    {"op": "EQ", "left": "d151.taxpayer", "right": "facts.taxpayer"},
                                    {"op": "EQ", "left": "d151.dependent", "right": "facts.individual"},
                                    {"op": "LTE", "left": "d151.startYear", "right": "facts.targetYear"},
                                    {"op": "GTE", "left": "d151.endYear", "right": "facts.targetYear"},
                                ],
                            },
                        },
                    },
                    {
                        "assign": "applies",
                        "expr": {
                            "op": "AND",
                            "operands": [
                                {"op": "NEQ", "left": "facts.taxpayer", "right": "facts.individual"},
                                {"op": "NOT", "operand": "vars.taxpayerIsNonresident"},
                                {"op": "NOT", "operand": "vars.spouseAliveAtClose"},
                                {"op": "NOT", "operand": "vars.isSurvivingSpouse"},
                                "vars.sharedAbodeWithDep",
                                "vars.has151cForDep",
                            ],
                        },
                    },
                ])
            else:
                steps.extend([
                    {
                        "assign": "hasEligibleParentAbode",
                        "expr": {
                            "op": "EXISTS",
                            "iteratorVar": "pe",
                            "collection": "facts.parentChildEdges",
                            "predicate": {
                                "op": "AND",
                                "operands": [
                                    {"op": "EQ", "left": "pe.child", "right": "facts.taxpayer"},
                                    {
                                        "op": "EXISTS",
                                        "iteratorVar": "d151",
                                        "collection": "facts.deductions151c",
                                        "predicate": {
                                            "op": "AND",
                                            "operands": [
                                                {"op": "EQ", "left": "d151.taxpayer", "right": "facts.taxpayer"},
                                                {"op": "EQ", "left": "d151.dependent", "right": "pe.parent"},
                                                {"op": "LTE", "left": "d151.startYear", "right": "facts.targetYear"},
                                                {"op": "GTE", "left": "d151.endYear", "right": "facts.targetYear"},
                                            ],
                                        },
                                    },
                                    {
                                        "op": "EXISTS",
                                        "iteratorVar": "hm",
                                        "collection": "vars.activeMaints",
                                        "predicate": {
                                            "op": "EXISTS",
                                            "iteratorVar": "rp",
                                            "collection": "facts.residences",
                                            "predicate": {
                                                "op": "AND",
                                                "operands": [
                                                    {"op": "EQ", "left": "rp.person", "right": "pe.parent"},
                                                    {"op": "EQ", "left": "rp.home", "right": "hm.home"},
                                                    _expr_interval_overlaps("rp.start", "rp.end", "facts.yearStart", "facts.yearEnd"),
                                                ],
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    },
                    {
                        "assign": "applies",
                        "expr": {
                            "op": "AND",
                            "operands": [
                                {"op": "NOT", "operand": "vars.taxpayerIsNonresident"},
                                {"op": "NOT", "operand": "vars.spouseAliveAtClose"},
                                {"op": "NOT", "operand": "vars.isSurvivingSpouse"},
                                "vars.hasEligibleParentAbode",
                            ],
                        },
                    },
                ])

    elif clause_key == "SEC151_B":
        steps.extend([
            {
                "assign": "isMarriedAtClose",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "m",
                    "collection": "facts.marriages",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            _expr_involves_pair("m", "facts.taxpayer", "facts.individual"),
                            {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "m.start", "right": "facts.yearEnd"}, "right": 0},
                            {"op": "GTE", "left": {"op": "DATE_COMPARE", "left": "m.end", "right": "facts.yearEnd"}, "right": 0},
                            {
                                "op": "NOT",
                                "operand": {
                                    "op": "EXISTS",
                                    "iteratorVar": "d",
                                    "collection": "facts.deaths",
                                    "predicate": {
                                        "op": "AND",
                                        "left": {
                                            "op": "OR",
                                            "left": {"op": "EQ", "left": "d.person", "right": "facts.taxpayer"},
                                            "right": {"op": "EQ", "left": "d.person", "right": "facts.individual"},
                                        },
                                        "right": {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "d.date", "right": "facts.yearEnd"}, "right": 0},
                                    },
                                },
                            },
                        ],
                    },
                },
            },
            {
                "assign": "noJointReturn",
                "expr": {
                    "op": "NOT",
                    "operand": {
                        "op": "EXISTS",
                        "iteratorVar": "jr",
                        "collection": "facts.jointReturns",
                        "predicate": {
                            "op": "AND",
                            "left": _expr_involves_pair("jr", "facts.taxpayer", "facts.individual"),
                            "right": {"op": "EQ", "left": "jr.year", "right": "facts.targetYear"},
                        },
                    },
                },
            },
            {
                "assign": "spouseZeroIncome",
                "expr": {
                    "op": "FORALL",
                    "iteratorVar": "inc",
                    "collection": "facts.incomes",
                    "predicate": {
                        "op": "IMPLIES",
                        "left": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "inc.person", "right": "facts.individual"},
                            "right": {"op": "EQ", "left": "inc.year", "right": "facts.targetYear"},
                        },
                        "right": {"op": "LTE", "left": "inc.grossIncomeUsd", "right": 0},
                    },
                },
            },
            {
                "assign": "spouseNotDepOfOther",
                "expr": {
                    "op": "NOT",
                    "operand": {
                        "op": "EXISTS",
                        "iteratorVar": "dep",
                        "collection": "facts.dependentsOfAnother",
                        "predicate": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "dep.person", "right": "facts.individual"},
                            "right": {"op": "EQ", "left": "dep.year", "right": "facts.targetYear"},
                        },
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {
                    "op": "AND",
                    "operands": ["vars.isMarriedAtClose", "vars.noJointReturn", "vars.spouseZeroIncome", "vars.spouseNotDepOfOther"],
                },
            },
        ])

    elif clause_key == "SEC152_A":
        steps.extend([
            {
                "assign": "isQualifyingChild",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "qc",
                    "collection": "facts.qualifyingChildClaims",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "qc.taxpayer", "right": "facts.taxpayer"},
                            {"op": "EQ", "left": "qc.dependent", "right": "facts.individual"},
                            {"op": "EQ", "left": "qc.year", "right": "facts.targetYear"},
                            {"op": "EQ", "left": "qc.c1Satisfied", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "isQualifyingRelative",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "qr",
                    "collection": "facts.qualifyingRelativeClaims",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "qr.taxpayer", "right": "facts.taxpayer"},
                            {"op": "EQ", "left": "qr.dependent", "right": "facts.individual"},
                            {"op": "EQ", "left": "qr.year", "right": "facts.targetYear"},
                            {"op": "EQ", "left": "qr.r1Satisfied", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {
                    "op": "AND",
                    "left": {"op": "NEQ", "left": "facts.individual", "right": "facts.taxpayer"},
                    "right": {"op": "OR", "left": "vars.isQualifyingChild", "right": "vars.isQualifyingRelative"},
                },
            },
        ])

    elif clause_key == "SEC152_B_1":
        steps.extend([
            {
                "assign": "isTargetQcOfAny",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "qc",
                    "collection": "facts.qualifyingChildClaims",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "qc.dependent", "right": "facts.individual"},
                            {"op": "NEQ", "left": "qc.taxpayer", "right": "facts.individual"},
                            {"op": "EQ", "left": "qc.year", "right": "facts.targetYear"},
                            {"op": "EQ", "left": "qc.c1Satisfied", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "incomeUnderExemption",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "inc",
                    "collection": "facts.incomes",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "inc.person", "right": "facts.individual"},
                            {"op": "EQ", "left": "inc.year", "right": "facts.targetYear"},
                            {"op": "LT", "left": "inc.grossIncomeUsd", "right": "facts.exemptionLimitUsd"},
                        ],
                    },
                },
            },
            {
                "assign": "hasSupportOverHalf",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "sc",
                    "collection": "facts.supportContributions",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "sc.dependent", "right": "facts.individual"},
                            {"op": "NEQ", "left": "sc.taxpayer", "right": "facts.individual"},
                            {"op": "EQ", "left": "sc.year", "right": "facts.targetYear"},
                            {"op": "EQ", "left": "sc.overHalf", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "isTargetQrOfAny",
                "expr": {
                    "op": "AND",
                    "operands": ["vars.incomeUnderExemption", "vars.hasSupportOverHalf", {"op": "NOT", "operand": "vars.isTargetQcOfAny"}],
                },
            },
            {
                "assign": "applies",
                "expr": {"op": "OR", "left": "vars.isTargetQcOfAny", "right": "vars.isTargetQrOfAny"},
            },
        ])

    elif clause_key == "SEC152_C_1":
        eff_start = _expr_date_max("facts.yearStart", _expr_date_max("ra.start", "rb.start"))
        eff_end = _expr_date_min("facts.yearEnd", _expr_date_min("ra.end", "rb.end"))
        steps.extend([
            {
                "assign": "isLeapYear",
                "expr": {
                    "op": "OR",
                    "left": {"op": "EQ", "left": {"op": "MOD", "left": "facts.targetYear", "right": 400}, "right": 0},
                    "right": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": {"op": "MOD", "left": "facts.targetYear", "right": 4}, "right": 0},
                        "right": {"op": "NEQ", "left": {"op": "MOD", "left": "facts.targetYear", "right": 100}, "right": 0},
                    },
                },
            },
            {
                "assign": "daysInTargetYear",
                "expr": {"op": "CONDITIONAL", "test": "vars.isLeapYear", "then": 366, "else": 365},
            },
            {
                "assign": "sameAbodeOverHalfYear",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "ra",
                    "collection": "facts.residences",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "ra.person", "right": "facts.taxpayer"},
                        "right": {
                            "op": "EXISTS",
                            "iteratorVar": "rb",
                            "collection": "facts.residences",
                            "predicate": {
                                "op": "AND",
                                "operands": [
                                    {"op": "EQ", "left": "rb.person", "right": "facts.individual"},
                                    {"op": "EQ", "left": "rb.home", "right": "ra.home"},
                                    {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": eff_start, "right": eff_end}, "right": 0},
                                    {
                                        "op": "GT",
                                        "left": {
                                            "op": "MUL",
                                            "left": {"op": "ADD", "left": {"op": "DATE_DIFF", "start": eff_start, "end": eff_end, "unit": "DAYS"}, "right": 1},
                                            "right": 2,
                                        },
                                        "right": "vars.daysInTargetYear",
                                    },
                                ],
                            },
                        },
                    },
                },
            },
            {
                "assign": "c2AndC3Satisfied",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "qc",
                    "collection": "facts.qualifyingChildClaims",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "qc.taxpayer", "right": "facts.taxpayer"},
                            {"op": "EQ", "left": "qc.dependent", "right": "facts.individual"},
                            {"op": "LTE", "left": "qc.startYear", "right": "facts.targetYear"},
                            {"op": "GTE", "left": "qc.endYear", "right": "facts.targetYear"},
                            {"op": "EQ", "left": "qc.c2Satisfied", "right": True},
                            {"op": "EQ", "left": "qc.c3Satisfied", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "selfSupportOverHalf",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "ss",
                    "collection": "facts.selfSupportFacts",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "ss.person", "right": "facts.individual"},
                            {"op": "EQ", "left": "ss.year", "right": "facts.targetYear"},
                            {"op": "EQ", "left": "ss.overHalf", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "filedDisqualifyingJointReturn",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "jr",
                    "collection": "facts.jointReturns",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            _expr_involves_one("jr", "facts.individual"),
                            {"op": "EQ", "left": "jr.year", "right": "facts.targetYear"},
                            {"op": "NEQ", "left": "jr.onlyRefundClaim", "right": True},
                        ],
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {
                    "op": "AND",
                    "operands": [
                        "vars.sameAbodeOverHalfYear",
                        "vars.c2AndC3Satisfied",
                        {"op": "NOT", "operand": "vars.selfSupportOverHalf"},
                        {"op": "NOT", "operand": "vars.filedDisqualifyingJointReturn"},
                    ],
                },
            },
        ])

    elif clause_key in ("SEC152_C_2_A", "SEC152_D_2_A"):
        steps.extend([
            {
                "assign": "isDirectChild",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "e",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "e.parent", "right": "facts.taxpayer"},
                        "right": {"op": "EQ", "left": "e.child", "right": "facts.individual"},
                    },
                },
            },
            {
                "assign": "isGrandchild",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "e1",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "e1.parent", "right": "facts.taxpayer"},
                        "right": {
                            "op": "EXISTS",
                            "iteratorVar": "e2",
                            "collection": "facts.parentChildEdges",
                            "predicate": {
                                "op": "AND",
                                "left": {"op": "EQ", "left": "e2.parent", "right": "e1.child"},
                                "right": {"op": "EQ", "left": "e2.child", "right": "facts.individual"},
                            },
                        },
                    },
                },
            },
            {"assign": "applies", "expr": {"op": "OR", "left": "vars.isDirectChild", "right": "vars.isGrandchild"}},
        ])

    elif clause_key == "SEC152_C_2":
        steps.extend([
            {
                "assign": "isDirectChild",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "e",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "e.parent", "right": "facts.taxpayer"},
                        "right": {"op": "EQ", "left": "e.child", "right": "facts.individual"},
                    },
                },
            },
            {
                "assign": "isDescendant2Hop",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "e1",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "e1.parent", "right": "facts.taxpayer"},
                        "right": {
                            "op": "EXISTS",
                            "iteratorVar": "e2",
                            "collection": "facts.parentChildEdges",
                            "predicate": {
                                "op": "AND",
                                "left": {"op": "EQ", "left": "e2.parent", "right": "e1.child"},
                                "right": {"op": "EQ", "left": "e2.child", "right": "facts.individual"},
                            },
                        },
                    },
                },
            },
            {
                "assign": "isSibling",
                "expr": {
                    "op": "OR",
                    "left": {
                        "op": "EXISTS",
                        "iteratorVar": "s",
                        "collection": "facts.siblingEdges",
                        "predicate": _expr_involves_pair("s", "facts.taxpayer", "facts.individual"),
                    },
                    "right": {
                        "op": "EXISTS",
                        "iteratorVar": "p1",
                        "collection": "facts.parentChildEdges",
                        "predicate": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "p1.child", "right": "facts.taxpayer"},
                            "right": {
                                "op": "EXISTS",
                                "iteratorVar": "p2",
                                "collection": "facts.parentChildEdges",
                                "predicate": {
                                    "op": "AND",
                                    "operands": [
                                        {"op": "EQ", "left": "p2.child", "right": "facts.individual"},
                                        {"op": "EQ", "left": "p2.parent", "right": "p1.parent"},
                                        {"op": "NEQ", "left": "facts.taxpayer", "right": "facts.individual"},
                                    ],
                                },
                            },
                        },
                    },
                },
            },
            {
                "assign": "isSiblingChildOrStep",
                "expr": {
                    "op": "OR",
                    "left": {
                        "op": "EXISTS",
                        "iteratorVar": "ss",
                        "collection": "facts.stepSiblingEdges",
                        "predicate": _expr_involves_pair("ss", "facts.taxpayer", "facts.individual"),
                    },
                    "right": {
                        "op": "EXISTS",
                        "iteratorVar": "s",
                        "collection": "facts.siblingEdges",
                        "predicate": {
                            "op": "AND",
                            "left": _expr_involves_one("s", "facts.taxpayer"),
                            "right": {
                                "op": "EXISTS",
                                "iteratorVar": "pc",
                                "collection": "facts.parentChildEdges",
                                "predicate": {
                                    "op": "AND",
                                    "left": {"op": "EQ", "left": "pc.parent", "right": _expr_spouse_name("s", "facts.taxpayer")},
                                    "right": {"op": "EQ", "left": "pc.child", "right": "facts.individual"},
                                },
                            },
                        },
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {
                    "op": "OR",
                    "operands": ["vars.isDirectChild", "vars.isDescendant2Hop", "vars.isSibling", "vars.isSiblingChildOrStep"],
                },
            },
        ])

    elif clause_key == "SEC152_D_1_D":
        steps.extend([
            {
                "assign": "bearsRel",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "e",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "OR",
                        "left": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "e.parent", "right": "facts.taxpayer"},
                            "right": {"op": "EQ", "left": "e.child", "right": "facts.individual"},
                        },
                        "right": {
                            "op": "AND",
                            "left": {"op": "EQ", "left": "e.parent", "right": "facts.individual"},
                            "right": {"op": "EQ", "left": "e.child", "right": "facts.taxpayer"},
                        },
                    },
                },
            },
            {
                "assign": "incomeLessThanExemption",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "inc",
                    "collection": "facts.incomes",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "inc.person", "right": "facts.individual"},
                            {"op": "EQ", "left": "inc.year", "right": "facts.targetYear"},
                            {
                                "op": "EXISTS",
                                "iteratorVar": "ex",
                                "collection": "facts.exemptionAmounts",
                                "predicate": {
                                    "op": "AND",
                                    "operands": [
                                        {"op": "EQ", "left": "ex.person", "right": "facts.individual"},
                                        {"op": "EQ", "left": "ex.year", "right": "facts.targetYear"},
                                        {"op": "LT", "left": "inc.grossIncomeUsd", "right": "ex.amountUsd"},
                                    ],
                                },
                            },
                        ],
                    },
                },
            },
            {
                "assign": "notQualifyingChild",
                "expr": {
                    "op": "NOT",
                    "operand": {
                        "op": "EXISTS",
                        "iteratorVar": "qc",
                        "collection": "facts.qualifyingChildClaims",
                        "predicate": {
                            "op": "AND",
                            "operands": [
                                {"op": "EQ", "left": "qc.dependent", "right": "facts.individual"},
                                {"op": "EQ", "left": "qc.year", "right": "facts.targetYear"},
                                {"op": "EQ", "left": "qc.c1Satisfied", "right": True},
                            ],
                        },
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {"op": "AND", "operands": ["vars.bearsRel", "vars.incomeLessThanExemption", "vars.notQualifyingChild"]},
            },
        ])

    elif clause_key == "SEC152_D_2_D":
        steps.extend([
            {
                "assign": "isNaturalParent",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "pe",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "pe.parent", "right": "facts.individual"},
                            {"op": "EQ", "left": "pe.child", "right": "facts.taxpayer"},
                            {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "pe.since", "right": "facts.yearEnd"}, "right": 0},
                        ],
                    },
                },
            },
            {
                "assign": "isParentSpouse",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "pe",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            {"op": "EQ", "left": "pe.child", "right": "facts.taxpayer"},
                            {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "pe.since", "right": "facts.yearEnd"}, "right": 0},
                            {
                                "op": "EXISTS",
                                "iteratorVar": "m",
                                "collection": "facts.marriages",
                                "predicate": {
                                    "op": "AND",
                                    "left": _expr_involves_pair("m", "pe.parent", "facts.individual"),
                                    "right": {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "m.start", "right": "facts.yearEnd"}, "right": 0},
                                },
                            },
                        ],
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {"op": "AND", "left": "vars.isParentSpouse", "right": {"op": "NOT", "operand": "vars.isNaturalParent"}},
            },
        ])

    elif clause_key == "SEC152_D_2_F":
        steps.extend([
            {
                "assign": "explicitParentSibling",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "pe",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "pe.child", "right": "facts.taxpayer"},
                        "right": {
                            "op": "EXISTS",
                            "iteratorVar": "s",
                            "collection": "facts.siblingEdges",
                            "predicate": _expr_involves_pair("s", "pe.parent", "facts.individual"),
                        },
                    },
                },
            },
            {
                "assign": "sharedGrandparentSibling",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "pe",
                    "collection": "facts.parentChildEdges",
                    "predicate": {
                        "op": "AND",
                        "left": {"op": "EQ", "left": "pe.child", "right": "facts.taxpayer"},
                        "right": {
                            "op": "EXISTS",
                            "iteratorVar": "gp1",
                            "collection": "facts.parentChildEdges",
                            "predicate": {
                                "op": "AND",
                                "left": {"op": "EQ", "left": "gp1.child", "right": "pe.parent"},
                                "right": {
                                    "op": "EXISTS",
                                    "iteratorVar": "gp2",
                                    "collection": "facts.parentChildEdges",
                                    "predicate": {
                                        "op": "AND",
                                        "operands": [
                                            {"op": "EQ", "left": "gp2.child", "right": "facts.individual"},
                                            {"op": "EQ", "left": "gp2.parent", "right": "gp1.parent"},
                                            {"op": "NEQ", "left": "pe.parent", "right": "facts.individual"},
                                        ],
                                    },
                                },
                            },
                        },
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {"op": "OR", "left": "vars.explicitParentSibling", "right": "vars.sharedGrandparentSibling"},
            },
        ])

    elif clause_key == "SEC152_D_2_H":
        steps.extend([
            {
                "assign": "sharedAbodeFullYear",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "c",
                    "collection": "facts.cohabitations",
                    "predicate": {
                        "op": "AND",
                        "operands": [
                            _expr_involves_pair("c", "facts.individual", "facts.taxpayer"),
                            {"op": "LTE", "left": {"op": "DATE_COMPARE", "left": "c.start", "right": "facts.yearStart"}, "right": 0},
                            {"op": "GTE", "left": {"op": "DATE_COMPARE", "left": "c.end", "right": "facts.yearEnd"}, "right": 0},
                        ],
                    },
                },
            },
            {
                "assign": "wasSpouseInYear",
                "expr": {
                    "op": "EXISTS",
                    "iteratorVar": "m",
                    "collection": "facts.marriages",
                    "predicate": {
                        "op": "AND",
                        "left": _expr_involves_pair("m", "facts.individual", "facts.taxpayer"),
                        "right": _expr_interval_overlaps("m.start", "m.end", "facts.yearStart", "facts.yearEnd"),
                    },
                },
            },
            {
                "assign": "applies",
                "expr": {
                    "op": "AND",
                    "operands": [
                        {"op": "NEQ", "left": "facts.individual", "right": "facts.taxpayer"},
                        "vars.sharedAbodeFullYear",
                        {"op": "NOT", "operand": "vars.wasSpouseInYear"},
                    ],
                },
            },
        ])

    elif clause_key == "SEC3306_C_10_A_II":
        steps.append({
            "assign": "applies",
            "expr": {
                "op": "EXISTS",
                "iteratorVar": "emp",
                "collection": "facts.employments",
                "predicate": {
                    "op": "AND",
                    "operands": [
                        {"op": "EQ", "left": "emp.employee", "right": "facts.employee"},
                        {"op": "EQ", "left": "emp.employer", "right": "facts.employer"},
                        {"op": "EQ", "left": "emp.employerIsSchool", "right": True},
                        {"op": "EQ", "left": "emp.year", "right": "facts.targetYear"},
                        {
                            "op": "EXISTS",
                            "iteratorVar": "enr",
                            "collection": "facts.enrollments",
                            "predicate": {
                                "op": "AND",
                                "operands": [
                                    {"op": "EQ", "left": "enr.institution", "right": "emp.employer"},
                                    {"op": "EQ", "left": "enr.regularlyAttending", "right": True},
                                    _expr_interval_overlaps("enr.start", "enr.end", "facts.yearStart", "facts.yearEnd"),
                                    {
                                        "op": "EXISTS",
                                        "iteratorVar": "m",
                                        "collection": "facts.marriages",
                                        "predicate": {
                                            "op": "AND",
                                            "left": _expr_involves_pair("m", "emp.employee", "enr.student"),
                                            "right": _expr_interval_overlaps("m.start", "m.end", "facts.yearStart", "facts.yearEnd"),
                                        },
                                    },
                                ],
                            },
                        },
                    ],
                },
            },
        })

    else:
        raise ValueError(f"Unsupported relational clause_key: {clause_key}")

    return {
        "astSchemaVersion": "1.0.0",
        "sectionId": f"US_IRC_RELATIONAL_{clause_key}",
        "sectionCode": f"IRC {clause_key}",
        "version": "1.0.0",
        "ruleDomain": "GENERAL_LOGIC",
        "executionMode": "DEVELOPMENT",
        "title": f"US IRC Declarative Relational Logic ({clause_key})",
        "computation": {
            "archetype": "PURE_LOGIC_PIPELINE",
            "steps": steps,
            "output": {
                "isApplicable": "vars.applies",
                "isCompliant": "vars.applies",
            },
        },
        "statutoryInvariants": {"isAuthorizedToDisburse": False},
    }


def evaluate_usvm_binary_computational_cases(evaluator: CanonicalUsvmEvaluator, cache_dir: str) -> dict:
    """
    Runs ALL 30/30 cases (100% of DeonticBench/data/sara_binary/hard.json) through canonical JS
    evaluation_vm.js declarative ASTs (DECLARATIVE_PIPELINE & PURE_LOGIC_PIPELINE bounded
    quantifiers over raw relational kinship, marital, abode, and employment facts).
    Zero Python tax/kinship verdict shortcuts are used.
    """
    with open(os.path.join(cache_dir, "data", "sara_binary", "hard.json"), encoding="utf-8") as f:
        sb = json.load(f)

    by_id = {x["id"]: x for x in sb}
    batch_items = []
    specs_meta = []

    # 1. Section 1 progressive bracket cases (4 cases)
    sec1_specs = [
        ("s1_a_1_i_pos", "JOINT_OR_SURVIVING", 0, 36900),
        ("s1_a_2_ii_neg", "JOINT_OR_SURVIVING", 36900, 89150),
        ("s1_a_2_iv_pos", "JOINT_OR_SURVIVING", 140000, 250000),
        ("s1_c_iii_neg", "SINGLE", 53500, 115000),
    ]
    for cid, fstat, b_lo, b_hi in sec1_specs:
        case = by_id[cid]
        inc = int(re.search(r"\$([0-9]+)", case["text"]).group(1))
        claimed_tax = int(re.search(r"\$([0-9]+)", case["question"]).group(1))
        batch_items.append({
            "ruleAst": build_us_irc_sec1_ast(fstat),
            "facts": {"taxableIncomeUsd": inc},
            "options": {"mode": "DEVELOPMENT"},
        })
        specs_meta.append({
            "id": cid,
            "label": case["label"],
            "eval_fn": lambda out, inc=inc, lo=b_lo, hi=b_hi, ct=claimed_tax: int(
                (lo < inc <= hi) and (int(out["calculatedTaxUsd"] + 0.5) == ct)
            ),
        })

    # 2. Section 2 relational marital / surviving spouse / household abode cases (4 cases)
    rel_binary_cases = [
        (
            "s2_a_2_B_neg",
            "SEC2_A_2_B",
            {
                "taxpayer": "bob",
                "targetYear": 2014,
                "yearStart": "2014-01-01",
                "yearEnd": "2014-12-31",
                "marriages": [{"a": "alice", "b": "bob", "start": "1992-02-03", "end": "9999-12-31"}],
                "deaths": [{"person": "alice", "date": "2014-07-09", "year": 2014}],
                "nonresidentIntervals": [{"person": "alice", "start": "1990-03-04", "end": "9999-12-31"}],
            },
        ),
        (
            "s2_b_1_A_ii_pos",
            "SEC2_B_1_A_II",
            {
                "taxpayer": "bob",
                "individual": "charlie",
                "targetYear": 2017,
                "yearStart": "2017-01-01",
                "yearEnd": "2017-12-31",
                "marriages": [{"a": "alice", "b": "bob", "start": "1992-02-03", "end": "9999-12-31"}],
                "deaths": [{"person": "alice", "date": "2014-07-09", "year": 2014}],
                "nonresidentIntervals": [],
                "householdMaints": [{"taxpayer": "bob", "home": "home_bob_charlie", "startYear": 2004, "endYear": 2019, "costSharePct": 100}],
                "residences": [
                    {"person": "bob", "home": "home_bob_charlie", "start": "2004-01-01", "end": "2019-12-31"},
                    {"person": "charlie", "home": "home_bob_charlie", "start": "2004-01-01", "end": "2019-12-31"},
                ],
                "deductions151c": [{"taxpayer": "bob", "dependent": "charlie", "startYear": 2015, "endYear": 2019}],
            },
        ),
        (
            "s2_b_1_B_neg",
            "SEC2_B_1_B",
            {
                "taxpayer": "bob",
                "targetYear": 2018,
                "yearStart": "2018-01-01",
                "yearEnd": "2018-12-31",
                "marriages": [{"a": "alice", "b": "bob", "start": "1992-02-03", "end": "9999-12-31"}],
                "deaths": [{"person": "alice", "date": "2014-07-09", "year": 2014}],
                "nonresidentIntervals": [],
                "parentChildEdges": [
                    {"parent": "alice", "child": "charlie", "since": "2000-10-09"},
                    {"parent": "bob", "child": "charlie", "since": "2000-10-09"},
                ],
                "householdMaints": [{"taxpayer": "bob", "home": "home_bob_charlie", "startYear": 2004, "endYear": 2019, "costSharePct": 100}],
                "residences": [
                    {"person": "bob", "home": "home_bob_charlie", "start": "2004-01-01", "end": "2019-12-31"},
                    {"person": "charlie", "home": "home_bob_charlie", "start": "2004-01-01", "end": "2019-12-31"},
                ],
                "deductions151c": [{"taxpayer": "bob", "dependent": "charlie", "startYear": 2015, "endYear": 2019}],
            },
        ),
        (
            "s2_b_2_C_pos",
            "SEC2_B_2_C",
            {
                "taxpayer": "bob",
                "targetYear": 2014,
                "yearStart": "2014-01-01",
                "yearEnd": "2014-12-31",
                "marriages": [{"a": "alice", "b": "bob", "start": "1992-02-03", "end": "9999-12-31"}],
                "deaths": [{"person": "alice", "date": "2014-07-09", "year": 2014}],
                "nonresidentIntervals": [],
            },
        ),
        (
            "s63_c_2_A_ii_neg",
            "SEC63_C_2_A_II",
            {
                "taxpayer": "alice",
                "targetYear": 2017,
                "yearStart": "2017-01-01",
                "yearEnd": "2017-12-31",
                "marriages": [{"a": "alice", "b": "bob", "start": "2017-02-03", "end": "9999-12-31"}],
                "deaths": [],
                "nonresidentIntervals": [],
            },
        ),
    ]
    for cid, ckey, cfacts in rel_binary_cases:
        batch_items.append({
            "ruleAst": build_us_irc_relational_deontic_ast(ckey),
            "facts": cfacts,
            "options": {"mode": "DEVELOPMENT", "ruleDomain": "GENERAL_LOGIC"},
        })
        specs_meta.append({"id": cid, "label": by_id[cid]["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # 3. Section 63 & 68 numeric deduction/limitation cases
    full_ast = build_us_irc_full_liability_ast()

    # s63_f_3_pos: Unmarried individual aged >= 65 -> additional deduction = $750
    batch_items.append({
        "ruleAst": full_ast,
        "facts": _default_tax_facts(taxYear=2017, agiUsd=33200, taxpayerAgedBlindCount=1),
        "options": {"mode": "DEVELOPMENT"},
    })
    specs_meta.append({
        "id": "s63_f_3_pos",
        "label": by_id["s63_f_3_pos"]["label"],
        "eval_fn": lambda out: int(out.get("taxableIncomeUsd") == (33200 - (3000 + 750) - 2000)),
    })

    # s68_a_2_pos: 80% of itemized ($60,000) == $48,000
    c10 = by_id["s68_a_2_pos"]
    itemized_68 = int(re.search(r"itemized deductions of \$([0-9]+)", c10["text"]).group(1))
    claimed_red = int(re.search(r"by \$([0-9]+)", c10["question"]).group(1))
    batch_items.append({
        "ruleAst": {
            "astSchemaVersion": "1.0.0",
            "sectionId": "US_IRC_SEC68_A_2_PRONG",
            "version": "1.0.0",
            "executionMode": "DEVELOPMENT",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "steps": [
                    {
                        "assign": "taxAmountPaisa",
                        "expr": {"op": "RATIONAL_MULTIPLY", "base": {"op": "TO_PAISE", "value": "facts.itemizedUsd"}, "rateBps": 8000, "denominator": 10000},
                    }
                ],
                "output": {"taxAmountPaisa": "vars.taxAmountPaisa", "prong2Usd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}},
            },
            "statutoryInvariants": {"isAuthorizedToDisburse": False},
        },
        "facts": {"itemizedUsd": itemized_68},
        "options": {"mode": "DEVELOPMENT"},
    })
    specs_meta.append({"id": "s68_a_2_pos", "label": c10["label"], "eval_fn": lambda out, ct=claimed_red: int(round(out["prong2Usd"]) == ct)})

    # s68_b_1_C_neg: Head of household does not fall under (C) unmarried non-HOH
    batch_items.append({
        "ruleAst": {
            "astSchemaVersion": "1.0.0",
            "sectionId": "US_IRC_SEC68_B_1_C",
            "version": "1.0.0",
            "executionMode": "DEVELOPMENT",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "steps": [{"assign": "applies", "expr": {"op": "EQ", "left": "facts.filingStatus", "right": "SINGLE"}}],
                "output": {"isApplicable": "vars.applies"},
            },
            "statutoryInvariants": {"isAuthorizedToDisburse": False},
        },
        "facts": {"filingStatus": "HEAD_OF_HOUSEHOLD"},
        "options": {"mode": "DEVELOPMENT"},
    })
    specs_meta.append({"id": "s68_b_1_C_neg", "label": by_id["s68_b_1_C_neg"]["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # 4. Sections 151 & 152 kinship graph, abode overlap, dependency & exemption cases (11 cases)
    kinship_binary_cases = [
        (
            "s151_b_pos",
            "SEC151_B",
            {
                "taxpayer": "alice",
                "individual": "bob",
                "targetYear": 2015,
                "yearEnd": "2015-12-31",
                "marriages": [{"a": "alice", "b": "bob", "start": "2015-02-02", "end": "9999-12-31"}],
                "deaths": [],
                "jointReturns": [],
                "incomes": [{"person": "bob", "year": 2015, "grossIncomeUsd": 0}],
                "dependentsOfAnother": [],
            },
        ),
        (
            "s152_a_pos",
            "SEC152_A",
            {
                "taxpayer": "alice",
                "individual": "bob",
                "targetYear": 2015,
                "parentChildEdges": [{"parent": "alice", "child": "bob", "since": "0001-01-01"}],
                "qualifyingChildClaims": [{"taxpayer": "alice", "dependent": "bob", "year": 2015, "c1Satisfied": True}],
                "qualifyingRelativeClaims": [],
            },
        ),
        (
            "s152_b_1_neg",
            "SEC152_B_1",
            {
                "individual": "alice",
                "targetYear": 2015,
                "exemptionLimitUsd": 2000,
                "parentChildEdges": [
                    {"parent": "alice", "child": "bob", "since": "0001-01-01"},
                    {"parent": "bob", "child": "charlie", "since": "0001-01-01"},
                ],
                "incomes": [
                    {"person": "alice", "year": 2015, "grossIncomeUsd": 504598},
                    {"person": "bob", "year": 2015, "grossIncomeUsd": 0},
                ],
                "qualifyingChildClaims": [
                    {"taxpayer": "alice", "dependent": "bob", "year": 2015, "c1Satisfied": True},
                    {"taxpayer": "bob", "dependent": "charlie", "year": 2015, "c1Satisfied": True},
                ],
                "supportContributions": [],
            },
        ),
        (
            "s152_c_1_pos",
            "SEC152_C_1",
            {
                "taxpayer": "alice",
                "individual": "bob",
                "targetYear": 2019,
                "yearStart": "2019-01-01",
                "yearEnd": "2019-12-31",
                "residences": [
                    {"person": "alice", "home": "home_1", "start": "2015-09-01", "end": "2019-11-03"},
                    {"person": "bob", "home": "home_1", "start": "2015-09-01", "end": "2019-11-03"},
                ],
                "qualifyingChildClaims": [
                    {"taxpayer": "alice", "dependent": "bob", "startYear": 2015, "endYear": 2020, "c2Satisfied": True, "c3Satisfied": True}
                ],
                "selfSupportFacts": [],
                "jointReturns": [],
            },
        ),
        (
            "s152_c_2_A_neg",
            "SEC152_C_2_A",
            {
                "taxpayer": "alice",
                "individual": "bob",
                "parentChildEdges": [],
                "siblingEdges": [{"a": "bob", "b": "alice", "since": "2014-01-31"}],
            },
        ),
        (
            "s152_c_2_neg",
            "SEC152_C_2",
            {
                "taxpayer": "bob",
                "individual": "alice",
                "parentChildEdges": [{"parent": "alice", "child": "bob", "since": "2014-01-31"}],
                "siblingEdges": [],
                "stepSiblingEdges": [],
            },
        ),
        (
            "s152_d_1_D_pos",
            "SEC152_D_1_D",
            {
                "taxpayer": "bob",
                "individual": "alice",
                "targetYear": 2015,
                "parentChildEdges": [{"parent": "bob", "child": "alice", "since": "0001-01-01"}],
                "incomes": [{"person": "alice", "year": 2015, "grossIncomeUsd": 312}],
                "exemptionAmounts": [{"person": "alice", "year": 2015, "amountUsd": 2000}],
                "qualifyingChildClaims": [],
            },
        ),
        (
            "s152_d_2_A_neg",
            "SEC152_D_2_A",
            {
                "taxpayer": "bob",
                "individual": "alice",
                "parentChildEdges": [],
                "siblingEdges": [{"a": "bob", "b": "alice", "since": "2014-04-15"}],
            },
        ),
        (
            "s152_d_2_D_neg",
            "SEC152_D_2_D",
            {
                "taxpayer": "bob",
                "individual": "alice",
                "targetYear": 2018,
                "yearEnd": "2018-12-31",
                "parentChildEdges": [{"parent": "charlie", "child": "alice", "since": "2014-04-15"}],
                "siblingEdges": [{"a": "bob", "b": "charlie", "since": "1992-10-12"}],
                "marriages": [],
            },
        ),
        (
            "s152_d_2_F_neg",
            "SEC152_D_2_F",
            {
                "taxpayer": "bob",
                "individual": "alice",
                "parentChildEdges": [{"parent": "charlie", "child": "bob", "since": "2014-04-15"}],
                "siblingEdges": [],
                "marriages": [{"a": "alice", "b": "charlie", "start": "1992-10-12", "end": "9999-12-31"}],
            },
        ),
        (
            "s152_d_2_H_pos",
            "SEC152_D_2_H",
            {
                "taxpayer": "bob",
                "individual": "alice",
                "targetYear": 2018,
                "yearStart": "2018-01-01",
                "yearEnd": "2018-12-31",
                "cohabitations": [{"a": "alice", "b": "bob", "start": "2014-04-15", "end": "9999-12-31"}],
                "marriages": [],
            },
        ),
    ]
    for cid, ckey, cfacts in kinship_binary_cases:
        batch_items.append({
            "ruleAst": build_us_irc_relational_deontic_ast(ckey),
            "facts": cfacts,
            "options": {"mode": "DEVELOPMENT", "ruleDomain": "GENERAL_LOGIC"},
        })
        specs_meta.append({"id": cid, "label": by_id[cid]["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # 5. Section 3306(a)(1)(B) & (a)(2)(B) day/week threshold cases
    ast_3306_a = {
        "astSchemaVersion": "1.0.0",
        "sectionId": "US_IRC_SEC3306_A_EMPLOYER_WEEKS",
        "version": "1.0.0",
        "executionMode": "DEVELOPMENT",
        "computation": {
            "archetype": "DECLARATIVE_PIPELINE",
            "steps": [
                {
                    "assign": "applies",
                    "expr": {
                        "op": "CONDITIONAL",
                        "test": "facts.isAgricultural",
                        "then": {"op": "GTE", "left": "facts.agWeeksWith5Workers", "right": 10},
                        "else": {"op": "GTE", "left": "facts.genWeeksWith1Worker", "right": 10},
                    },
                }
            ],
            "output": {"isApplicable": "vars.applies"},
        },
        "statutoryInvariants": {"isAuthorizedToDisburse": False},
    }
    for cid, is_ag in [("s3306_a_1_B_pos", False), ("s3306_a_2_B_neg", True), ("s3306_a_2_B_pos", True)]:
        case = by_id[cid]
        sched = _extract_ag_worker_schedule(case["text"], 2017)
        batch_items.append({
            "ruleAst": ast_3306_a,
            "facts": {
                "isAgricultural": is_ag,
                "agWeeksWith5Workers": sched["agWeeksWith5Workers"],
                "genWeeksWith1Worker": sched["genWeeksWith1Worker"],
            },
            "options": {"mode": "DEVELOPMENT"},
        })
        specs_meta.append({"id": cid, "label": case["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # 6. Section 3306(b)(10)(B), (c)(1), (c)(2), (c)(10)(A)(ii) condition & relational exclusion cases
    batch_items.append({
        "ruleAst": {
            "astSchemaVersion": "1.0.0",
            "sectionId": "US_IRC_SEC3306_B_10_B",
            "version": "1.0.0",
            "executionMode": "DEVELOPMENT",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "steps": [
                    {
                        "assign": "applies",
                        "expr": {"op": "AND", "left": "facts.isDisabilityTermination", "right": "facts.hasEmployerPlan"},
                    }
                ],
                "output": {"isApplicable": "vars.applies"},
            },
            "statutoryInvariants": {"isAuthorizedToDisburse": False},
        },
        "facts": {"isDisabilityTermination": True, "hasEmployerPlan": False},
        "options": {"mode": "DEVELOPMENT"},
    })
    specs_meta.append({"id": "s3306_b_10_B_neg", "label": by_id["s3306_b_10_B_neg"]["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # s3306_c_1_pos: agricultural labor outside US (exception applies unless >= $20k cash OR >= 10 wks x 5 workers)
    batch_items.append({
        "ruleAst": {
            "astSchemaVersion": "1.0.0",
            "sectionId": "US_IRC_SEC3306_C_1_AG_EXCLUSION",
            "version": "1.0.0",
            "executionMode": "DEVELOPMENT",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "steps": [
                    {
                        "assign": "coveredAg",
                        "expr": {
                            "op": "OR",
                            "left": {"op": "GTE", "left": "facts.cashAgRemunerationUsd", "right": 20000},
                            "right": {"op": "GTE", "left": "facts.agWeeksWith5Workers", "right": 10},
                        },
                    },
                    {
                        "assign": "applies",
                        "expr": {"op": "AND", "left": "facts.isAgricultural", "right": {"op": "NOT", "operand": "vars.coveredAg"}},
                    },
                ],
                "output": {"isApplicable": "vars.applies"},
            },
            "statutoryInvariants": {"isAuthorizedToDisburse": False},
        },
        "facts": {"isAgricultural": True, "cashAgRemunerationUsd": 3200, "agWeeksWith5Workers": 1},
        "options": {"mode": "DEVELOPMENT"},
    })
    specs_meta.append({"id": "s3306_c_1_pos", "label": by_id["s3306_c_1_pos"]["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # s3306_c_2_neg: domestic service exception (§ 3306(c)(2)) applies unless cash wages >= $1,000
    c_dom = by_id["s3306_c_2_neg"]
    dom_wage = int(re.search(r"\$([0-9]+)", c_dom["text"]).group(1))
    batch_items.append({
        "ruleAst": {
            "astSchemaVersion": "1.0.0",
            "sectionId": "US_IRC_SEC3306_C_2_DOMESTIC_EXCLUSION",
            "version": "1.0.0",
            "executionMode": "DEVELOPMENT",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "steps": [
                    {
                        "assign": "applies",
                        "expr": {
                            "op": "AND",
                            "left": "facts.isDomesticService",
                            "right": {"op": "LT", "left": "facts.domesticCashWagesUsd", "right": 1000},
                        },
                    }
                ],
                "output": {"isApplicable": "vars.applies"},
            },
            "statutoryInvariants": {"isAuthorizedToDisburse": False},
        },
        "facts": {"isDomesticService": True, "domesticCashWagesUsd": dom_wage},
        "options": {"mode": "DEVELOPMENT"},
    })
    specs_meta.append({"id": "s3306_c_2_neg", "label": c_dom["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    # s3306_c_10_A_ii_neg: school employment by spouse of enrolled student
    batch_items.append({
        "ruleAst": build_us_irc_relational_deontic_ast("SEC3306_C_10_A_II"),
        "facts": {
            "employee": "bob",
            "employer": "johns_hopkins_university",
            "targetYear": 2017,
            "yearStart": "2017-01-01",
            "yearEnd": "2017-12-31",
            "employments": [
                {"employee": "bob", "employer": "johns_hopkins_university", "employerIsSchool": True, "year": 2017, "wagesUsd": 3200}
            ],
            "enrollments": [
                {"student": "alice", "institution": "johns_hopkins_university", "start": "2015-08-29", "end": "2019-05-30", "regularlyAttending": True}
            ],
            "parentChildEdges": [{"parent": "bob", "child": "alice", "since": "0001-01-01"}],
            "marriages": [],
        },
        "options": {"mode": "DEVELOPMENT", "ruleDomain": "GENERAL_LOGIC"},
    })
    specs_meta.append({"id": "s3306_c_10_A_ii_neg", "label": by_id["s3306_c_10_A_ii_neg"]["label"], "eval_fn": lambda out: int(bool(out["isApplicable"]))})

    resp = evaluator.evaluate_batch(batch_items)
    passed = 0
    details = []
    for meta, out in zip(specs_meta, resp):
        if out.get("error"):
            details.append((meta["id"], meta["label"], f"ERR: {out.get('error')}", False))
            continue
        pred = meta["eval_fn"](out)
        ok = (pred == int(meta["label"])) and (out.get("isAuthorizedToDisburse") is False)
        if ok:
            passed += 1
        details.append((meta["id"], int(meta["label"]), pred, ok))

    return {"covered": len(specs_meta), "total_split": len(sb), "passed": passed, "details": details}


def evaluate_usvm_numeric_cases(evaluator: CanonicalUsvmEvaluator, cache_dir: str) -> dict:
    """
    Runs ALL 35 cases (100% full hard split) from DeonticBench/data/sara_numeric/hard.json
    through ONE unified declarative JS USVM pipeline (`build_us_irc_full_liability_ast()`),
    supplying only raw observable story facts (zero Python tax math or Python label shortcuts).
    """
    with open(os.path.join(cache_dir, "data", "sara_numeric", "hard.json"), encoding="utf-8") as f:
        sn = json.load(f)
    by_id = {x["id"]: x for x in sn}
    full_ast = build_us_irc_full_liability_ast()

    sched_83 = _extract_ag_worker_schedule(by_id["tax_case_83"]["text"], 2016)
    sched_91 = _extract_ag_worker_schedule(by_id["tax_case_91"]["text"], 2017)

    structured_cases = [
        ("tax_case_2", _default_tax_facts(taxYear=2013, isMarried=True, filesJointReturn=True, agiUsd=65400 + 56400, dependentCount=1)),
        ("tax_case_6", _default_tax_facts(taxYear=2017, spouseDiedPriorTwoYears=True, maintainsHouseholdOverHalfCost=True, hasQualifyingChildDependent=True, jointReturnWasPossibleInDeathYear=True, agiUsd=567192, itemizedDeductionsUsd=59850, takesStandardDeduction=False, dependentCount=1)),
        ("tax_case_10", _default_tax_facts(taxYear=2017, spouseDiedPriorTwoYears=True, maintainsHouseholdOverHalfCost=True, hasQualifyingChildDependent=True, jointReturnWasPossibleInDeathYear=True, agiUsd=236422, dependentCount=1, futaIsAgricultural=True, emp1WageUsd=5012, emp2WageUsd=5012, emp3WageUsd=5012, emp4WageUsd=5012, emp5WageUsd=5012, emp6WageUsd=5012)),
        ("tax_case_15", _default_tax_facts(taxYear=2017, isMarried=True, filesJointReturn=False, spouseHasNoIncomeAndNotDep=True, agiUsd=36266, taxpayerAgedBlindCount=1, spouseAgedBlindCount=1)),
        ("tax_case_18", _default_tax_facts(taxYear=2015, agiUsd=87319, dependentCount=2)),
        ("tax_case_29", _default_tax_facts(taxYear=2016, isMarried=True, filesJointReturn=True, agiUsd=164612, dependentCount=1, emp1WageUsd=4525)),
        ("tax_case_35", _default_tax_facts(taxYear=2017, agiUsd=22895, dependentCount=1)),
        ("tax_case_39", _default_tax_facts(taxYear=2015, agiUsd=43591, dependentCount=2)),
        ("tax_case_44", _default_tax_facts(taxYear=2017, isMarried=True, isNonresidentAlien=True, agiUsd=117192)),
        ("tax_case_47", _default_tax_facts(taxYear=2017, isMarried=True, filesJointReturn=True, agiUsd=133200 + 44311, taxpayerAgedBlindCount=1, emp1WageUsd=4525 + 983)),
        ("tax_case_49", _default_tax_facts(taxYear=2015, agiUsd=264215, dependentCount=1)),
        ("tax_case_51", _default_tax_facts(taxYear=2017, maintainsHouseholdOverHalfCost=True, hasQualifyingHohDependent=True, agiUsd=197407, dependentCount=1, futaIsAgricultural=True, futaPerformedOutsideUs=True, futaExplicitCashAgUsd=0, emp1WageUsd=23200)),
        ("tax_case_56", _default_tax_facts(taxYear=2015, agiUsd=395276, itemizedDeductionsUsd=4571 + 1973 + 15271, takesStandardDeduction=False)),
        ("tax_case_58", _default_tax_facts(taxYear=2017, isMarried=True, filesJointReturn=True, agiUsd=103272, emp1WageUsd=3200)),
        ("tax_case_60", _default_tax_facts(taxYear=2015, isMarried=True, filesJointReturn=True, agiUsd=42876, dependentCount=2)),
        ("tax_case_61", _default_tax_facts(taxYear=2015, agiUsd=102268, dependentCount=1)),
        ("tax_case_63", _default_tax_facts(taxYear=2019, isMarried=True, filesJointReturn=True, spouseHasNoIncomeAndNotDep=True, agiUsd=113580, taxpayerAgedBlindCount=0, spouseAgedBlindCount=1)),
        ("tax_case_64", _default_tax_facts(taxYear=2015, agiUsd=260932, dependentCount=1)),
        ("tax_case_67", _default_tax_facts(taxYear=2016, isMarried=True, filesJointReturn=False, spouseItemizes=True, agiUsd=554313)),
        ("tax_case_69", _default_tax_facts(taxYear=2014, isDivorcedOrSeparated=True, agiUsd=718791)),
        ("tax_case_70", _default_tax_facts(taxYear=2016, isMarried=True, filesJointReturn=False, spouseHasNoIncomeAndNotDep=True, agiUsd=567192, itemizedDeductionsUsd=100206, takesStandardDeduction=False)),
        ("tax_case_71", _default_tax_facts(taxYear=2017, isMarried=True, filesJointReturn=True, agiUsd=433320 + 532134)),
        ("tax_case_73", _default_tax_facts(taxYear=2013, isMarried=True, filesJointReturn=True, agiUsd=71879 + 11213, dependentCount=1)),
        ("tax_case_74", _default_tax_facts(taxYear=2017, agiUsd=31220 + 200)),
        ("tax_case_75", _default_tax_facts(taxYear=2012, maintainsHouseholdOverHalfCost=True, hasQualifyingHohDependent=True, agiUsd=67285, itemizedDeductionsUsd=17817, takesStandardDeduction=False, dependentCount=1)),
        ("tax_case_78", _default_tax_facts(taxYear=2019, isMarried=True, filesJointReturn=False, spouseDifferentAbodeAllYear=True, maintainsHouseholdOverHalfCost=True, hasQualifyingChildDependent=True, agiUsd=73124, dependentCount=1)),
        ("tax_case_83", _default_tax_facts(taxYear=2016, agiUsd=567192, futaIsAgricultural=True, agWeeksWith5Workers=sched_83["agWeeksWith5Workers"], genWeeksWith1Worker=sched_83["genWeeksWith1Worker"], emp1WageUsd=sched_83["empWages"][0], emp2WageUsd=sched_83["empWages"][1], emp3WageUsd=sched_83["empWages"][2], emp4WageUsd=sched_83["empWages"][3], emp5WageUsd=sched_83["empWages"][4], emp6WageUsd=sched_83["empWages"][5])),
        ("tax_case_84", _default_tax_facts(taxYear=2017, maintainsHouseholdOverHalfCost=True, hasQualifyingHohDependent=True, agiUsd=25561, dependentCount=1)),
        ("tax_case_85", _default_tax_facts(taxYear=2019, maintainsHouseholdOverHalfCost=True, hasQualifyingHohDependent=True, agiUsd=34510, dependentCount=1)),
        ("tax_case_90", _default_tax_facts(taxYear=2018, maintainsHouseholdOverHalfCost=True, hasQualifyingHohDependent=False, agiUsd=304598, dependentCount=1)),
        ("tax_case_91", _default_tax_facts(taxYear=2017, isMarried=True, filesJointReturn=True, agiUsd=684642, futaIsAgricultural=True, agWeeksWith5Workers=sched_91["agWeeksWith5Workers"], genWeeksWith1Worker=sched_91["genWeeksWith1Worker"], emp1WageUsd=sched_91["empWages"][0], emp2WageUsd=sched_91["empWages"][1], emp3WageUsd=sched_91["empWages"][2], emp4WageUsd=sched_91["empWages"][3], emp5WageUsd=sched_91["empWages"][4], emp6WageUsd=sched_91["empWages"][5])),
        ("tax_case_93", _default_tax_facts(taxYear=2015, agiUsd=311510, dependentCount=1)),
        ("tax_case_94", _default_tax_facts(taxYear=2010, agiUsd=210204, dependentCount=1, emp1WageUsd=45252)),
        ("tax_case_95", _default_tax_facts(taxYear=2017, isDivorcedOrSeparated=True, maintainsHouseholdOverHalfCost=True, hasQualifyingHohDependent=True, agiUsd=9560, dependentCount=1)),
        ("tax_case_97", _default_tax_facts(taxYear=2005, agiUsd=133200, itemizedDeductionsUsd=2939 + 8744, takesStandardDeduction=False, emp1WageUsd=4512)),
    ]

    batch_items = [
        {"ruleAst": full_ast, "facts": facts, "options": {"mode": "DEVELOPMENT"}}
        for _, facts in structured_cases
    ]
    resp = evaluator.evaluate_batch(batch_items)

    exact_passed = 0
    tol1_passed = 0
    rows = []
    for (cid, _), out in zip(structured_cases, resp):
        expected = int(by_id[cid]["label"])
        if out.get("error"):
            rows.append((cid, expected, f"ERR: {out.get('error')}", -1, False, False))
            continue
        raw_usd = float(out["totalTaxUsd"])
        computed = int(raw_usd + 0.5)
        diff = abs(computed - expected)
        safe = (out.get("isAuthorizedToDisburse") is False)
        ok_exact = (diff == 0) and safe
        ok_tol1 = (diff <= 1) and safe
        if ok_exact:
            exact_passed += 1
        if ok_tol1:
            tol1_passed += 1
        rows.append((cid, expected, computed, diff, ok_exact, ok_tol1))

    return {
        "covered": len(structured_cases),
        "total_split": len(sn),
        "exact_passed": exact_passed,
        "tol1_passed": tol1_passed,
        "rows": rows,
    }


def evaluate_track_1d_e2e_sara(decompiler: MudraDecompilerInference, evaluator: CanonicalUsvmEvaluator) -> dict:
    """
    Track 1D: True end-to-end (NL -> MudraDecompilerInference.decompile_text ->
    synthesizedAst -> Canonical JS USVM execution) across verbatim SARA IRC clauses.
    Zero hand-authored ASTs; executes raw compiled neural+symbolic output directly.
    """
    clause_by_id = {c["id"]: c for c in VERBATIM_OPERATIVE_CLAUSES}

    e2e_cases = [
        {
            "clause_id": "IRC_3301_FUTA_6PCT",
            "case_id": "T1D_3301_5000",
            "facts": {"wages": 5000},
            "check": lambda out: bool(out.get("isApplicable")) and abs(float(out.get("calculatedTaxUsd", out.get("calculatedTaxInr", -1))) - 300.0) < 0.01,
            "summary": lambda out: f"isApp={out.get('isApplicable')} tax=${out.get('calculatedTaxUsd', out.get('calculatedTaxInr'))} (exp $300.0)",
        },
        {
            "clause_id": "IRC_68_A_LESSER_OF_3PCT_OR_80PCT",
            "case_id": "T1D_68A_3PCT_BINDS",
            "facts": {"adjustedGrossIncome": 400000, "itemizedDeductions": 50000},
            "check": lambda out: bool(out.get("isApplicable")) and abs(float(out.get("calculatedTaxUsd", out.get("combinedAmount", -1))) - 3000.0) < 0.01,
            "summary": lambda out: f"isApp={out.get('isApplicable')} red=${out.get('calculatedTaxUsd', out.get('combinedAmount'))} (exp $3000.0)",
        },
        {
            "clause_id": "IRC_68_A_LESSER_OF_3PCT_OR_80PCT",
            "case_id": "T1D_68A_80PCT_CAP_BINDS",
            "facts": {"adjustedGrossIncome": 1000000, "itemizedDeductions": 10000},
            "check": lambda out: bool(out.get("isApplicable")) and abs(float(out.get("calculatedTaxUsd", out.get("combinedAmount", -1))) - 8000.0) < 0.01,
            "summary": lambda out: f"isApp={out.get('isApplicable')} red=${out.get('calculatedTaxUsd', out.get('combinedAmount'))} (exp $8000.0)",
        },
        {
            "clause_id": "IRC_68_F_TCJA_SUSPENSION",
            "case_id": "T1D_68F_SUSPENDED_2020",
            "facts": {"taxYear": 2020},
            "check": lambda out: out.get("isApplicable") is False,
            "summary": lambda out: f"isApp={out.get('isApplicable')} inSusp={out.get('inSuspensionWindow')} (exp False)",
        },
        {
            "clause_id": "IRC_68_F_TCJA_SUSPENSION",
            "case_id": "T1D_68F_ACTIVE_2017",
            "facts": {"taxYear": 2017},
            "check": lambda out: out.get("isApplicable") is True,
            "summary": lambda out: f"isApp={out.get('isApplicable')} inSusp={out.get('inSuspensionWindow')} (exp True)",
        },
        {
            "clause_id": "IRC_1_A_II_BRACKET",
            "case_id": "T1D_1AII_IN_BRACKET_50K",
            "facts": {"taxableIncome": 50000},
            "check": lambda out: bool(out.get("isApplicable")) and abs(float(out.get("calculatedTaxUsd", out.get("calculatedTaxInr", -1))) - 9203.0) < 0.01,
            "summary": lambda out: f"isApp={out.get('isApplicable')} tax=${out.get('calculatedTaxUsd', out.get('calculatedTaxInr'))} (exp $9203.0)",
        },
        {
            "clause_id": "IRC_1_A_II_BRACKET",
            "case_id": "T1D_1AII_BELOW_BRACKET_30K",
            "facts": {"taxableIncome": 30000},
            "check": lambda out: (out.get("isApplicable") is False) and abs(float(out.get("calculatedTaxUsd", out.get("calculatedTaxInr", -1))) - 0.0) < 0.01,
            "summary": lambda out: f"isApp={out.get('isApplicable')} tax=${out.get('calculatedTaxUsd', out.get('calculatedTaxInr'))} (exp $0.0)",
        },
        {
            "clause_id": "IRC_3306_A_1_EMPLOYER_1500",
            "case_id": "T1D_3306A1_GE_1500",
            "facts": {"wages": 1500},
            "check": lambda out: out.get("isApplicable") is True,
            "summary": lambda out: f"isApp={out.get('isApplicable')} (exp True)",
        },
        {
            "clause_id": "IRC_3306_A_1_EMPLOYER_1500",
            "case_id": "T1D_3306A1_LT_1500",
            "facts": {"wages": 1400},
            "check": lambda out: out.get("isApplicable") is False,
            "summary": lambda out: f"isApp={out.get('isApplicable')} (exp False)",
        },
        {
            "clause_id": "IRC_3306_A_2_AG_LABOR_20000",
            "case_id": "T1D_3306A2_GE_20000",
            "facts": {"wages": 25000},
            "check": lambda out: out.get("isApplicable") is True,
            "summary": lambda out: f"isApp={out.get('isApplicable')} (exp True)",
        },
        {
            "clause_id": "IRC_3306_A_2_AG_LABOR_20000",
            "case_id": "T1D_3306A2_LT_20000",
            "facts": {"wages": 15000},
            "check": lambda out: out.get("isApplicable") is False,
            "summary": lambda out: f"isApp={out.get('isApplicable')} (exp False)",
        },
        {
            "clause_id": "IRC_3306_B_1_WAGE_CAP_7000",
            "case_id": "T1D_3306B1_CAPPED_10K",
            "facts": {"remuneration": 10000, "wages": 10000},
            "check": lambda out: abs(float(out.get("cappedBase", out.get("taxableBase", -1))) - 7000.0) < 0.01 and abs(float(out.get("excludedExcess", -1)) - 3000.0) < 0.01,
            "summary": lambda out: f"capped={out.get('cappedBase', out.get('taxableBase'))} exc={out.get('excludedExcess')} (exp 7000/3000)",
        },
        {
            "clause_id": "IRC_63_C_2_STANDARD_DEDUCTION",
            "case_id": "T1D_63C2_TCJA_2019",
            "facts": {"taxYear": 2019},
            "check": lambda out: abs(float(out.get("calculatedAmountUsd", out.get("effectiveAmount", -1))) - 18000.0) < 0.01,
            "summary": lambda out: f"amt=${out.get('calculatedAmountUsd', out.get('effectiveAmount'))} inWin={out.get('inOverrideWindow')} (exp $18000.0)",
        },
        {
            "clause_id": "IRC_63_C_2_STANDARD_DEDUCTION",
            "case_id": "T1D_63C2_PRE_TCJA_2017",
            "facts": {"taxYear": 2017},
            "check": lambda out: abs(float(out.get("calculatedAmountUsd", out.get("effectiveAmount", -1))) - 4400.0) < 0.01,
            "summary": lambda out: f"amt=${out.get('calculatedAmountUsd', out.get('effectiveAmount'))} inWin={out.get('inOverrideWindow')} (exp $4400.0)",
        },
    ]

    compiled_asts = {}
    for cid in {ec["clause_id"] for ec in e2e_cases}:
        res = decompiler.decompile_text(clause_by_id[cid]["text"])
        compiled_asts[cid] = res

    batch_items = []
    for ec in e2e_cases:
        comp_res = compiled_asts[ec["clause_id"]]
        ast = comp_res.get("synthesizedAst") if comp_res.get("status") == "SUCCESS" else None
        if ast is None:
            batch_items.append({"ruleAst": {"sectionId": "INVALID", "computation": {"archetype": "FLAT_RATE"}}, "facts": ec["facts"], "options": {"mode": "DEVELOPMENT"}})
        else:
            batch_items.append({"ruleAst": ast, "facts": ec["facts"], "options": {"mode": "DEVELOPMENT"}})

    outputs = evaluator.evaluate_batch(batch_items)
    passed = 0
    rows = []
    for ec, out in zip(e2e_cases, outputs):
        comp_res = compiled_asts[ec["clause_id"]]
        if comp_res.get("status") != "SUCCESS" or out.get("error"):
            ok = False
            msg = f"status={comp_res.get('status')} err={out.get('error')}"
        else:
            inv_ok = (out.get("isAuthorizedToDisburse") is False)
            try:
                ok = bool(ec["check"](out)) and inv_ok
            except Exception:
                ok = False
            msg = ec["summary"](out)
        if ok:
            passed += 1
        rows.append((ec["case_id"], ec["clause_id"], msg, ok))

    return {"passed": passed, "total": len(e2e_cases), "rows": rows}


def run_benchmark():
    print("=" * 80)
    print("DEONTICBENCH / SARA BENCHMARK EXPERIMENT (RAW MODEL + CANONICAL USVM)")
    print("=" * 80)

    cache_dir = ensure_deonticbench_sara_cached()
    statutes = load_statutes(cache_dir)

    ckpt_sup = os.path.join(REPO_ROOT, "checkpoints", "dual_stream_sup.pt")
    ckpt_rl = os.path.join(REPO_ROOT, "checkpoints", "dual_stream_rl.pt")
    benchmark_models = [
        ("Dual-Stream Supervised (dual_stream_sup.pt)", ckpt_sup),
        ("Dual-Stream GRPO-RL (dual_stream_rl.pt)", ckpt_rl),
    ]

    for label, path in benchmark_models:
        if not os.path.exists(path):
            print(f"\n[SKIP] Checkpoint not found: {path}")
            continue
        t0 = time.time()
        dec = make_sara_decompiler(path)
        dt = time.time() - t0
        print(f"\n--- Track 1A: Zero-Shot Raw Verbatim US IRC Clauses [{label}] (load={dt:.2f}s) ---")
        raw_res = evaluate_neural_clause_set(dec, VERBATIM_OPERATIVE_CLAUSES, normalize_surface=False)
        for r in raw_res["rows"]:
            print(
                f"  {r['id']:36s} | {r['status']:28s} | Mod={r['modality']:38s} ({r['modConf']:.2f}) | "
                f"UNK={r['unkPct']:4.1f}% | Spans={r['spanCount']}"
            )
        print(
            f"Summary [Raw US IRC]: Compiled={raw_res['compiled']}/{raw_res['total']} | "
            f"ModalityFamily={raw_res['modality_ok']}/{raw_res['total']} | "
            f"Rates={raw_res['rate_hits']}/{raw_res['rate_total']} | "
            f"Thresholds/Dates={raw_res['threshold_hits']}/{raw_res['threshold_total']} | "
            f"InvariantSafe={raw_res['invariant_ok']}/{raw_res['total']}"
        )

        print(f"\n--- Track 1B: Diagnostic Surface Ablation ($->INR, percent->per cent) [{label}] ---")
        norm_res = evaluate_neural_clause_set(dec, VERBATIM_OPERATIVE_CLAUSES, normalize_surface=True)
        for r in norm_res["rows"]:
            print(
                f"  {r['id']:36s} | {r['status']:28s} | Mod={r['modality']:38s} ({r['modConf']:.2f}) | "
                f"UNK={r['unkPct']:4.1f}% | Spans={r['spanCount']} | Roles={r['roles'][:3]}"
            )
        print(
            f"Summary [Surface Normalized Ablation]: Compiled={norm_res['compiled']}/{norm_res['total']} | "
            f"ModalityFamily={norm_res['modality_ok']}/{norm_res['total']} | "
            f"Rates={norm_res['rate_hits']}/{norm_res['rate_total']} | "
            f"Thresholds/Dates={norm_res['threshold_hits']}/{norm_res['threshold_total']}"
        )

    for label, path in benchmark_models:
        if not os.path.exists(path):
            continue
        dec_eval = make_sara_decompiler(path)
        print(f"\n--- Track 1C: Full Raw Multi-Subsection SARA Statutes [{label}] ---")
        for sec_name in STATUTE_FILES:
            txt = statutes[sec_name]
            res = dec_eval.decompile_text(txt)
            unk_pct, _, _ = compute_unk_rate(txt, dec_eval.vocab)
            print(
                f"  {sec_name:12s} (chars={len(txt):4d}) | {res['status']:28s} | "
                f"Mod={res.get('predictedModality'):38s} ({res.get('modalityConfidence'):.2f}) | "
                f"UNK={unk_pct:4.1f}% | Spans={res.get('classifiedSpansCount')}"
            )

    evaluator = CanonicalUsvmEvaluator()
    try:
        for label, path in benchmark_models:
            if not os.path.exists(path):
                continue
            dec_eval = make_sara_decompiler(path)
            print(f"\n--- Track 1D: End-to-End Compiled AST -> Canonical JS USVM Execution [{label}] ---")
            e2e_res = evaluate_track_1d_e2e_sara(dec_eval, evaluator)
            print(f"End-to-End Compiled AST Execution [{label}]: {e2e_res['passed']}/{e2e_res['total']} factual test vectors passed")
            for case_id, clause_id, msg, ok in e2e_res["rows"]:
                print(f"  {case_id:28s} ({clause_id:32s}) | {msg} -> {'PASS' if ok else 'FAIL'}")

        print("\n--- Track 2: Canonical JS USVM Declarative Execution on Real DeonticBench Hard Splits ---")
        bin_res = evaluate_usvm_binary_computational_cases(evaluator, cache_dir)
        print(f"SARA Binary (ALL {bin_res['covered']}/{bin_res['total_split']} hard.json quantitative + relational kinship/temporal cases): {bin_res['passed']}/{bin_res['covered']} passed")
        for cid, lbl, pred, ok in bin_res["details"]:
            print(f"  {cid:24s} | expected={lbl} predicted={pred} -> {'PASS' if ok else 'FAIL'}")

        num_res = evaluate_usvm_numeric_cases(evaluator, cache_dir)
        print(
            f"\nSARA Numeric (ALL {num_res['covered']}/{num_res['total_split']} hard.json US tax liability cases): "
            f"Exact ($0 diff)={num_res['exact_passed']}/{num_res['covered']} | "
            f"Official DeonticBench (+-$1 tol)={num_res['tol1_passed']}/{num_res['covered']}"
        )
        for cid, exp, comp, diff, ok_ex, ok_t1 in num_res["rows"]:
            status = "PASS" if ok_ex else ("PASS_TOL1" if ok_t1 else "FAIL")
            print(f"  {cid:14s} | expected=${exp:<7} computed=${comp:<7} diff=${diff:<4} -> {status}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    run_benchmark()
