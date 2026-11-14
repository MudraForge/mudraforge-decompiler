"""
SARA-Hard Diagnostic: End-to-End Compilation Coverage (RFC-012, Step 5).

Measures how far the neural decompiler and compositional lowerer get on the
65 hard cases of DeonticBench SARA (35 numeric, 30 binary).

This is a compilation coverage diagnostic, NOT an accuracy benchmark:
- It records the furthest pipeline stage reached per case.
- It produces a failure code histogram.
- It makes ZERO correctness or accuracy claims.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.neural.clause_linker import ClauseLinkingError
from compiler.converters.catala_emitter import CatalaLoweringError, emit_catala_code, _identifier
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

SARA_DATA_DIR = REPO_ROOT / "data" / "deonticbench_sara" / "data"
STATUTES_DIR = REPO_ROOT / "data" / "deonticbench_sara" / "statutes"
HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
CATALA_INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"


# Stable Failure Stage Codes
STAGE_ANNOTATION = "ANNOTATION"
STAGE_SCOPE = "SCOPE"
STAGE_LOWER = "LOWER"
STAGE_CATALA_EMIT = "CATALA_EMIT"
STAGE_CATALA_TYPECHECK = "CATALA_TYPECHECK"
STAGE_USVM_EXEC = "USVM_EXEC"
STAGE_CROSS_ENGINE = "CROSS_ENGINE"
STAGE_COMPLETE = "COMPLETE"


@dataclass
class CaseDiagnosticResult:
    case_id: str
    track: str  # numeric or binary
    statute_section: str
    furthest_stage: str
    annotation_ok: bool = False
    scope_ok: bool = False
    lower_ok: bool = False
    catala_emit_ok: bool = False
    catala_typecheck_ok: bool = False
    usvm_exec_ok: bool = False
    cross_engine_equal: bool = False
    failure_code: str | None = None
    failure_detail: str | None = None


def identify_statute_section(case_text: str, question: str, prolog: str) -> str:
    """Identifies the primary IRC section governing the case."""
    combined = f"{case_text} {question} {prolog}".lower()
    if "section 1(" in combined or "tax_from_brackets" in combined:
        return "section1"
    if "section 3301" in combined or "agricultural" in combined or "employer_tax" in combined:
        return "section3301"
    if "section 63" in combined or "standard_deduction" in combined:
        return "section63"
    if "section 68" in combined or "itemized" in combined:
        return "section68"
    if "section 151" in combined or "personal_exemptions" in combined:
        return "section151"
    if "section 2" in combined or "surviving_spouse" in combined or "head_of_household" in combined:
        return "section2"
    if "section 7703" in combined or "married" in combined:
        return "section7703"
    return "section1"


def load_statute_text(section_name: str) -> str:
    """Loads raw statutory text from the data directory."""
    sec_path = STATUTES_DIR / section_name
    if sec_path.is_file():
        try:
            return sec_path.read_text(encoding="utf-8")
        except Exception:
            return ""
    if sec_path.is_dir():
        texts = []
        for f in sorted(sec_path.glob("*.txt")):
            try:
                texts.append(f.read_text(encoding="utf-8"))
            except Exception:
                pass
        return "\n\n".join(texts)
    return ""


def generate_synthetic_inputs(ast: dict[str, Any]) -> dict[str, Any]:
    """Generates synthetic facts based on declared inputs signature."""
    inputs = ast.get("inputs") or {}
    facts = {}
    for key, spec in inputs.items():
        itype = spec.get("type", "MONEY_MINOR")
        if itype == "MONEY_MINOR":
            facts[key] = "5000000"
        elif itype in ("INTEGER", "COUNT"):
            facts[key] = "1"
        elif itype == "PERCENTAGE_RATE":
            facts[key] = "1000"
        elif itype == "BOOLEAN":
            facts[key] = True
        else:
            facts[key] = "5000000"
    if not facts:
        facts = {
            "amount": 5000000,
            "taxableIncome": "5000000",
            "primaryBasePaisa": "5000000",
            "currentInvoice": 5000000,
            "priorPurchases": 0,
            "condition1": True,
        }
    return facts


def run_diagnostic_on_case(
    case: dict[str, Any],
    track: str,
    decompiler: DualStreamDecompilerInference,
    evaluator: CanonicalUsvmEvaluator | None,
    catala_available: bool,
) -> CaseDiagnosticResult:
    cid = case["id"]
    text = case.get("text", "")
    question = case.get("question", "")
    prolog = case.get("reference_prolog", "")
    sec = identify_statute_section(text, question, prolog)
    statute_text = load_statute_text(sec) or text

    res = CaseDiagnosticResult(case_id=cid, track=track, statute_section=sec, furthest_stage=STAGE_ANNOTATION)

    decomp = None
    # ── Stage 1: Neural Annotation & Stage 2: Scope ─────────────────────────
    try:
        decomp = decompiler.decompile_text(statute_text, clause_scopes=True)
    except ClauseLinkingError as exc:
        # Neural annotation and clause scoping succeeded; failed at lowering/linking!
        res.annotation_ok = True
        res.scope_ok = True
        res.furthest_stage = STAGE_LOWER
        res.failure_code = exc.code or "ERR_NO_COMPUTATION"
        res.failure_detail = str(exc)
        return res
    except Exception as exc:
        msg = str(exc)
        if "ERR_NO_COMPUTATION" in msg:
            res.annotation_ok = True
            res.scope_ok = True
            res.furthest_stage = STAGE_LOWER
            res.failure_code = "ERR_NO_COMPUTATION"
            res.failure_detail = msg
            return res
        res.failure_code = "ERR_ANNOTATION_EXCEPTION"
        res.failure_detail = msg
        return res

    if decomp.get("status") != "SUCCESS":
        res.failure_code = "ERR_ANNOTATION_FAILED"
        res.failure_detail = f"status={decomp.get('status')}"
        return res

    res.annotation_ok = True
    res.scope_ok = True
    res.furthest_stage = STAGE_LOWER

    # ── Stage 3: Lowering (AST Lint) ────────────────────────────────────────
    ast = decomp.get("synthesizedAst")
    if not ast:
        res.failure_code = "ERR_NO_AST"
        res.failure_detail = "decompile_text returned empty synthesizedAst"
        return res

    comp = ast.get("computation") or {}
    steps = (comp.get("pipeline") or {}).get("steps") or comp.get("steps")
    if not steps:
        res.failure_code = "ERR_EMPTY_PIPELINE"
        res.failure_detail = f"computation archetype={comp.get('archetype')} has no steps"
        return res

    res.lower_ok = True
    res.furthest_stage = STAGE_CATALA_EMIT

    # ── Stage 4: Catala Emission ────────────────────────────────────────────
    synthetic_facts = generate_synthetic_inputs(ast)
    source = None
    try:
        source = emit_catala_code(ast, title=f"SARA {cid}", scenarios=[{"name": "S", "facts": synthetic_facts}])
        if not source:
            res.failure_code = "ERR_CATALA_EMPTY_SOURCE"
            return res
        res.catala_emit_ok = True
        res.furthest_stage = STAGE_CATALA_TYPECHECK
    except CatalaLoweringError as exc:
        res.failure_code = "ERR_CATALA_LOWERING"
        res.failure_detail = str(exc)
        return res
    except Exception as exc:
        res.failure_code = "ERR_CATALA_EMIT_UNCAUGHT"
        res.failure_detail = str(exc)
        return res

    # ── Stage 5: Catala Typecheck ───────────────────────────────────────────
    if not catala_available:
        res.failure_code = "SKIPPED_CATALA_UNAVAILABLE"
        return res

    scope_name = "Scope"
    for line in source.splitlines():
        if line.strip().startswith("declaration scope "):
            scope_name = line.strip().split()[2].rstrip(":")
            break

    catala_minor = None
    try:
        req = {"id": cid, "source": source, "scopes": [scope_name]}
        proc = subprocess.run(
            ["node", str(HARNESS)],
            input=json.dumps(req) + "\n",
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        if proc.returncode != 0:
            res.failure_code = "ERR_CATALA_HARNESS_EXIT"
            res.failure_detail = proc.stderr
            return res
        resp = json.loads(proc.stdout.strip().splitlines()[-1])
        if not resp.get("typecheck", {}).get("success"):
            res.failure_code = "ERR_CATALA_TYPECHECK"
            res.failure_detail = str(resp.get("typecheck", {}).get("errors"))
            return res
        res.catala_typecheck_ok = True
        res.furthest_stage = STAGE_USVM_EXEC

        # Extract Catala execution result for cross-engine parity check
        s_res = resp.get("results", {}).get(scope_name, {})
        if s_res.get("success"):
            output_key = next(iter((comp.get("pipeline") or {}).get("output") or {}), None)
            if output_key and _identifier(output_key) in s_res.get("values", {}):
                val = s_res["values"][_identifier(output_key)]
                if isinstance(val, (int, float)):
                    catala_minor = round(val * 100) if isinstance(val, float) else val
    except Exception as exc:
        res.failure_code = "ERR_CATALA_TYPECHECK_EXCEPTION"
        res.failure_detail = str(exc)
        return res

    # ── Stage 6: USVM Execution ─────────────────────────────────────────────
    if not evaluator:
        return res

    usvm_val = None
    try:
        usvm_res = evaluator.evaluate(ast, synthetic_facts)
        if not usvm_res:
            res.failure_code = "ERR_USVM_EMPTY_RESULT"
            return res
        res.usvm_exec_ok = True
        res.furthest_stage = STAGE_CROSS_ENGINE
        output_key = next(iter((comp.get("pipeline") or {}).get("output") or {}), None)
        if output_key and output_key in usvm_res:
            usvm_val = usvm_res[output_key]
    except Exception as exc:
        res.failure_code = "ERR_USVM_EVALUATION"
        res.failure_detail = str(exc)
        return res

    # ── Stage 7: Cross-Engine Parity ────────────────────────────────────────
    if catala_minor is not None and usvm_val is not None:
        if str(catala_minor) == str(usvm_val):
            res.cross_engine_equal = True
            res.furthest_stage = STAGE_COMPLETE
        else:
            res.failure_code = "ERR_CROSS_ENGINE_DIVERGENCE"
            res.failure_detail = f"Catala={catala_minor} != USVM={usvm_val}"
            return res
    else:
        res.furthest_stage = STAGE_COMPLETE

    return res


def run_all_diagnostics(
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    print("Loading dual-stream model checkpoint...")
    decompiler = DualStreamDecompilerInference("checkpoints/dual_stream_sup.pt")
    
    catala_available = CATALA_INTERPRETER.exists() and HARNESS.exists()
    print(f"Catala interpreter available: {catala_available}")

    evaluator = None
    try:
        evaluator = CanonicalUsvmEvaluator()
    except Exception as exc:
        print(f"Warning: Canonical USVM evaluator not available: {exc}")

    numeric_path = SARA_DATA_DIR / "sara_numeric" / "hard.json"
    binary_path = SARA_DATA_DIR / "sara_binary" / "hard.json"

    cases: list[tuple[dict[str, Any], str]] = []
    if numeric_path.exists():
        with open(numeric_path, encoding="utf-8") as f:
            cases.extend((c, "numeric") for c in json.load(f))
    if binary_path.exists():
        with open(binary_path, encoding="utf-8") as f:
            cases.extend((c, "binary") for c in json.load(f))

    print(f"Loaded {len(cases)} total cases across hard splits.")

    results: list[CaseDiagnosticResult] = []
    stages_count = Counter()
    failure_codes = Counter()

    for idx, (c, track) in enumerate(cases, 1):
        res = run_diagnostic_on_case(c, track, decompiler, evaluator, catala_available)
        results.append(res)
        stages_count[res.furthest_stage] += 1
        if res.failure_code:
            failure_codes[res.failure_code] += 1
        print(f"[{idx:02d}/{len(cases)}] {c['id']:14s} ({track:7s}) -> stage: {res.furthest_stage:18s} "
              f"code: {res.failure_code or 'OK'}")

    if evaluator:
        evaluator.close()

    def _calc_stats(subset: list[CaseDiagnosticResult]):
        tot = len(subset)
        return {
            "total": tot,
            "annotation_ok": sum(1 for r in subset if r.annotation_ok),
            "scope_ok": sum(1 for r in subset if r.scope_ok),
            "lower_ok": sum(1 for r in subset if r.lower_ok),
            "catala_emit_ok": sum(1 for r in subset if r.catala_emit_ok),
            "catala_typecheck_ok": sum(1 for r in subset if r.catala_typecheck_ok),
            "usvm_exec_ok": sum(1 for r in subset if r.usvm_exec_ok),
            "cross_engine_equal": sum(1 for r in subset if r.cross_engine_equal),
        }

    numeric_results = [r for r in results if r.track == "numeric"]
    binary_results = [r for r in results if r.track == "binary"]

    summary = {
        "cases_total": len(cases),
        "compilation_coverage_all": _calc_stats(results),
        "compilation_coverage_numeric": _calc_stats(numeric_results),
        "compilation_coverage_binary": _calc_stats(binary_results),
        "furthest_stage_distribution": dict(stages_count),
        "failure_codes_histogram": dict(failure_codes),
        "results": [asdict(r) for r in results],
    }

    # Write JSON report
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write Markdown report (Strictly zero accuracy / correctness field per RFC-012 A5.3)
    c_all = summary["compilation_coverage_all"]
    c_num = summary["compilation_coverage_numeric"]
    c_bin = summary["compilation_coverage_binary"]

    md_lines = [
        "# SARA-Hard Compilation Coverage Diagnostic (RFC-012 Step 5)",
        "",
        f"- **Total Cases Evaluated**: {len(cases)} (35 Numeric + 30 Binary)",
        f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## 1. Stage Coverage Funnel",
        "",
        "| Stage | Combined (65) | SARA-Numeric (35) | SARA-Binary (30) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Neural Annotation** | **{c_all['annotation_ok']}/{c_all['total']} ({c_all['annotation_ok']/c_all['total']*100:.1f}%)** | {c_num['annotation_ok']}/{c_num['total']} ({c_num['annotation_ok']/c_num['total']*100:.1f}%) | {c_bin['annotation_ok']}/{c_bin['total']} ({c_bin['annotation_ok']/c_bin['total']*100:.1f}%) |",
        f"| **Clause Scoping** | **{c_all['scope_ok']}/{c_all['total']} ({c_all['scope_ok']/c_all['total']*100:.1f}%)** | {c_num['scope_ok']}/{c_num['total']} ({c_num['scope_ok']/c_num['total']*100:.1f}%) | {c_bin['scope_ok']}/{c_bin['total']} ({c_bin['scope_ok']/c_bin['total']*100:.1f}%) |",
        f"| **AST Lowering** | **{c_all['lower_ok']}/{c_all['total']} ({c_all['lower_ok']/c_all['total']*100:.1f}%)** | {c_num['lower_ok']}/{c_num['total']} ({c_num['lower_ok']/c_num['total']*100:.1f}%) | {c_bin['lower_ok']}/{c_bin['total']} ({c_bin['lower_ok']/c_bin['total']*100:.1f}%) |",
        f"| **Catala Emitter** | **{c_all['catala_emit_ok']}/{c_all['total']} ({c_all['catala_emit_ok']/c_all['total']*100:.1f}%)** | {c_num['catala_emit_ok']}/{c_num['total']} ({c_num['catala_emit_ok']/c_num['total']*100:.1f}%) | {c_bin['catala_emit_ok']}/{c_bin['total']} ({c_bin['catala_emit_ok']/c_bin['total']*100:.1f}%) |",
        f"| **Catala Typecheck** | **{c_all['catala_typecheck_ok']}/{c_all['total']} ({c_all['catala_typecheck_ok']/c_all['total']*100:.1f}%)** | {c_num['catala_typecheck_ok']}/{c_num['total']} ({c_num['catala_typecheck_ok']/c_num['total']*100:.1f}%) | {c_bin['catala_typecheck_ok']}/{c_bin['total']} ({c_bin['catala_typecheck_ok']/c_bin['total']*100:.1f}%) |",
        f"| **USVM Execution** | **{c_all['usvm_exec_ok']}/{c_all['total']} ({c_all['usvm_exec_ok']/c_all['total']*100:.1f}%)** | {c_num['usvm_exec_ok']}/{c_num['total']} ({c_num['usvm_exec_ok']/c_num['total']*100:.1f}%) | {c_bin['usvm_exec_ok']}/{c_bin['total']} ({c_bin['usvm_exec_ok']/c_bin['total']*100:.1f}%) |",
        f"| **Cross-Engine Parity** | **{c_all['cross_engine_equal']}/{c_all['total']} ({c_all['cross_engine_equal']/c_all['total']*100:.1f}%)** | {c_num['cross_engine_equal']}/{c_num['total']} ({c_num['cross_engine_equal']/c_num['total']*100:.1f}%) | {c_bin['cross_engine_equal']}/{c_bin['total']} ({c_bin['cross_engine_equal']/c_bin['total']*100:.1f}%) |",
        "",
        "## 2. Failure Code Histogram",
        "",
        "| Failure Code | Count | Stage | Description |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for code, count in failure_codes.most_common():
        desc = (
            "Multiple candidate base operands without explicit disambiguation (Gate A2.4)"
            if code == "ERR_AMBIGUOUS_REFERENCE"
            else (
                "Clause has no computable rate-to-threshold head"
                if code == "ERR_NO_RATE_HEAD"
                else (
                    "No base operand found or inherited"
                    if code == "ERR_NO_BASE_OPERAND"
                    else (
                        "Lowerer could not find computable clause head"
                        if code == "ERR_NO_COMPUTATION"
                        else (
                            "Model abstention or invalid sequence tagging"
                            if code == "ERR_ANNOTATION_FAILED"
                            else (
                                "Pipeline has no steps"
                                if code == "ERR_EMPTY_PIPELINE"
                                else (
                                    "AST cannot be lowered to Catala"
                                    if code == "ERR_CATALA_LOWERING"
                                    else "Evaluation error"
                                )
                            )
                        )
                    )
                )
            )
        )
        stage = (
            "LOWER"
            if code in (
                "ERR_NO_COMPUTATION",
                "ERR_EMPTY_PIPELINE",
                "ERR_AMBIGUOUS_REFERENCE",
                "ERR_NO_RATE_HEAD",
                "ERR_NO_BASE_OPERAND",
            )
            else (
                "CATALA_EMIT"
                if code == "ERR_CATALA_LOWERING"
                else ("USVM_EXEC" if "USVM" in code else "ANNOTATION")
            )
        )
        md_lines.append(f"| `{code}` | {count} | `{stage}` | {desc} |")

    md_lines.extend([
        "",
        "## 3. Methodological Note",
        "",
        "This diagnostic reports strict syntactic compilation and cross-engine parity coverage across the pipeline without oracle fact mappings.",
        "It verifies that lowered ASTs are well-formed, typecheck under INRIA's Catala compiler, and evaluate under USVM without runtime exceptions.",
        "It contains no claim of substantive legal correctness of multi-section statutory formulas against external benchmark tax returns."
    ])

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nDiagnostic reports written to:\n  JSON: {output_json}\n  Markdown: {output_md}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARA-Hard Compilation Diagnostic")
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "docs" / "sara_hard_diagnostic.json")
    parser.add_argument("--md", type=Path, default=REPO_ROOT / "docs" / "sara_hard_diagnostic.md")
    args = parser.parse_args()
    run_all_diagnostics(args.json, args.md)
