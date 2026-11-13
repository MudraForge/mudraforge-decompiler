#!/usr/bin/env python3
"""
=============================================================================
MUDRA DECOMPILER: COMMAND LINE INTERFACE (CLI)
=============================================================================

Compiles natural language legal statutes, tax provisions, and contracts into
executable Declarative ASTs and evaluates them against factual vectors.

Examples:
  python cli.py "In the case of a head of household, the basic standard deduction is $4,400..."
  python cli.py --file examples/statute.txt --format json
  python cli.py --interactive
  python cli.py "Any person responsible for paying..." --evaluate facts.json
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator
from compiler.converters.catala_emitter import emit_catala_code, save_catala_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mudra Decompiler: Zero-Shot Neuro-Symbolic Statutory Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Statutory or regulatory legal text to decompile",
    )
    parser.add_argument(
        "-f", "--file",
        dest="file_path",
        help="Path to text or markdown file containing statutory text",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        default=os.path.join(REPO_ROOT, "checkpoints", "dual_stream_sup.pt"),
        help="Path to model weights checkpoint (.pt)",
    )
    parser.add_argument(
        "-d", "--domain",
        choices=["AUTO", "STATUTORY_FINANCE", "GENERAL_LOGIC"],
        default="AUTO",
        help="Target statutory domain rule parser",
    )
    parser.add_argument(
        "--format",
        choices=["pretty", "json", "catala"],
        default="pretty",
        help="Output format (pretty, raw json, or literate catala)",
    )
    parser.add_argument(
        "--export-catala",
        dest="catala_path",
        help="Optional destination path to export decompiled Catala source (.catala_en)",
    )
    parser.add_argument(
        "-e", "--evaluate",
        dest="facts_path",
        help="Optional path to JSON facts vector to execute compiled AST in USVM",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run interactive statutory decompiler REPL session",
    )
    return parser.parse_args()


def print_pretty_result(res: dict, eval_res: dict = None):
    print("=" * 80)
    print("MUDRA NEURO-SYMBOLIC DECOMPILER RESULT")
    print("=" * 80)
    print(f"Status:               {res.get('status')}")
    print(f"Compiler:             {res.get('compiler')}")
    print(f"Rule Domain:          {res.get('ruleDomain')}")
    print(f"Predicted Modality:   {res.get('predictedModality')}")
    print(f"Modality Confidence:  {res.get('modalityConfidence')}")
    print(f"Calibrated Confidence:{res.get('calibratedConfidence')}")
    print(f"Statutory Invariants: {json.dumps(res.get('statutoryInvariants', {}))}")
    print(f"Authorized to Disburse: {res.get('isAuthorizedToDisburse')}")

    spans = res.get("classifiedSpans", [])
    print(f"\nClassified Spans ({len(spans)}):")
    print("-" * 80)
    for s in spans:
        role = s.get("role", "ROLE_IGNORE")
        conf = s.get("confidence", 0.0)
        span_txt = s.get("span", "")
        print(f"  [{role:<34s}] (conf={conf:.2f}): {span_txt}")

    edges = res.get("relationalEdges", [])
    if edges:
        print(f"\nRelational Edges ({len(edges)}):")
        print("-" * 80)
        for e in edges:
            print(f"  ({e.get('sourceSpan')}) --[{e.get('relationType')}]--> ({e.get('targetSpan')})")

    ast = res.get("synthesizedAst")
    if ast:
        print("\nSynthesized Declarative USVM AST (Summary):")
        print("-" * 80)
        comp = ast.get("computation", {})
        print(f"  Archetype: {comp.get('archetype')}")
        steps = comp.get("steps") or (comp.get("pipeline") or {}).get("steps", [])
        print(f"  Steps:     {len(steps)} pipeline operations")
        for idx, step in enumerate(steps[:5]):
            print(f"    Step {idx + 1}: assign [{step.get('assign')}] = {json.dumps(step.get('expr'))}")
        if len(steps) > 5:
            print(f"    ... ({len(steps) - 5} more steps)")

    if eval_res is not None:
        print("\n" + "=" * 80)
        print("CANONICAL USVM EXECUTION OUTCOME")
        print("=" * 80)
        for k, v in eval_res.items():
            if k not in ("executionTrace", "voucherDigest", "citations"):
                print(f"  {k:<32s}: {v}")
        trace = eval_res.get("executionTrace", [])
        if trace:
            print("\nExecution Trace:")
            for t in trace:
                print(f"  > {t}")
    print("=" * 80)


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        alt_ckpt = os.path.join(REPO_ROOT, "checkpoints", "dual_stream_rl.pt")
        if os.path.exists(alt_ckpt):
            args.checkpoint = alt_ckpt
        else:
            sys.stderr.write(f"Error: Model checkpoint not found at {args.checkpoint}\n")
            sys.exit(1)

    print(f"Loading Mudra Decompiler model from: {args.checkpoint}...")
    decompiler = DualStreamDecompilerInference(args.checkpoint)

    evaluator = None
    if args.facts_path or args.interactive:
        evaluator = CanonicalUsvmEvaluator()

    if args.interactive:
        print("\n--- Mudra Decompiler Interactive REPL ---")
        print("Enter statutory text below (or 'exit' / 'quit' to terminate):\n")
        try:
            while True:
                line = input("statute> ").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "q"):
                    break
                res = decompiler.decompile_text(line, rule_domain=args.domain)
                if args.format == "json":
                    print(json.dumps(res, indent=2))
                elif args.format == "catala":
                    print(emit_catala_code(res))
                else:
                    print_pretty_result(res)
        finally:
            if evaluator:
                evaluator.close()
        return

    text = args.text
    if args.file_path:
        with open(args.file_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        sys.stderr.write("Error: No statutory text provided. Pass text as argument or via --file / --interactive.\n")
        sys.exit(1)

    res = decompiler.decompile_text(text, rule_domain=args.domain)

    eval_res = None
    if args.facts_path and res.get("status") == "SUCCESS":
        with open(args.facts_path, "r", encoding="utf-8") as f:
            facts = json.load(f)
        if evaluator is None:
            evaluator = CanonicalUsvmEvaluator()
        ast = res.get("synthesizedAst")
        batch_out = evaluator.evaluate_batch([
            {"ruleAst": ast, "facts": facts, "options": {"mode": "DEVELOPMENT"}}
        ])
        eval_res = batch_out[0] if batch_out else None

    if evaluator:
        evaluator.close()

    if args.catala_path:
        out_catala = save_catala_file(res, args.catala_path)
        print(f"[CATALA] Decompiled Catala source exported to: {out_catala}")

    if args.format == "json":
        payload = {"decompilation": res}
        if eval_res is not None:
            payload["evaluation"] = eval_res
        print(json.dumps(payload, indent=2))
    elif args.format == "catala":
        print(emit_catala_code(res))
    else:
        print_pretty_result(res, eval_res)


if __name__ == "__main__":
    main()
