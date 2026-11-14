"""Commit 3 tests for Model v2 dataset generation, RL environment scoring, and Stage 1/2 smoke training."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.converters.catala_emitter import emit_catala_code
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.neural.inference_v2 import synthesize_usvm_ast_v2
from compiler.neural.statutory_neural_parser import (
    SPAN_UNITS,
    UNIT2ID,
    V2_BIO_TAGS,
    V2_ID2REL,
    V2_RELATION_TYPES,
    V2_SEMANTIC_ROLES,
)
from compiler.training.dataset_generator import generate_synthetic_dataset
from compiler.training.train_dual_stream import (
    train_grpo_logic_only,
    train_supervised_dual_stream,
)
from compiler.training.usvm_rl_environment import StatutoryRlEnvironment
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator


class TestModelV2Commit3(unittest.TestCase):
    def test_v2_dataset_emits_unit_labels_and_v2_opcodes(self):
        payload = generate_synthetic_dataset(num_samples=96, seed=7, v2=True)
        self.assertEqual(payload.get("architecture_version"), "v2")
        self.assertEqual(payload.get("span_units"), SPAN_UNITS)
        self.assertEqual(payload.get("roles"), V2_SEMANTIC_ROLES)
        self.assertEqual(payload.get("relations"), V2_RELATION_TYPES)
        self.assertEqual(payload.get("bio_tags"), V2_BIO_TAGS)

        all_samples = payload["train"] + payload["val"]
        self.assertGreater(len(all_samples), 40)

        seen_units = set()
        seen_rels = set()
        for s in all_samples:
            self.assertIn("unit_labels", s)
            self.assertEqual(len(s["unit_labels"]), len(s["span_boundaries"]))
            seen_units.update(s["unit_labels"])
            for row in s["rel_matrix"]:
                for rid in row:
                    if rid != 0:
                        seen_rels.add(V2_ID2REL[rid])

        self.assertIn(UNIT2ID["UNIT_MONEY"], seen_units)
        self.assertIn(UNIT2ID["UNIT_DURATION"], seen_units)
        self.assertIn(UNIT2ID["UNIT_COUNT"], seen_units)
        self.assertIn(UNIT2ID["UNIT_RATE"], seen_units)
        self.assertIn("REL_BASE_TAX_ADD", seen_rels)
        self.assertIn("REL_DEFINED_TERM_OF", seen_rels)
        self.assertTrue(
            "REL_SUB_CREDIT" in seen_rels or "REL_MIN_CAP" in seen_rels,
            f"Expected REL_SUB_CREDIT or REL_MIN_CAP in {sorted(seen_rels)}",
        )

    def test_rl_environment_scores_v2_asts_and_handles_fail_closed_candidates(self):
        text = "Tax equals 28% on excess of gross income over $36,900 minus credit of $500."
        spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "28%", "charRange": [11, 14], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "unit": "UNIT_SYMBOL_VAR", "span": "gross income", "charRange": [28, 40], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 0, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "over $36,900", "charRange": [41, 53], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 0, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$500", "charRange": [70, 74], "confidence": 0.99},
        ]
        gold_edges = [
            {"sourceIdx": 0, "targetIdx": 2, "relationType": "REL_RATE_TO_THRESHOLD"},
            {"sourceIdx": 3, "targetIdx": 0, "relationType": "REL_SUB_CREDIT"},
        ]
        wrong_edges = [
            {"sourceIdx": 0, "targetIdx": 2, "relationType": "REL_RATE_TO_THRESHOLD"},
            {"sourceIdx": 3, "targetIdx": 0, "relationType": "REL_MIN_CAP"},
        ]
        cm = [{"clauseIdx": 0, "charRange": [0, len(text)]}]
        gold_ast = synthesize_usvm_ast_v2(text=text, spans=spans, edges=gold_edges, clause_modalities=cm)
        wrong_ast = synthesize_usvm_ast_v2(text=text, spans=spans, edges=wrong_edges, clause_modalities=cm)

        env = StatutoryRlEnvironment()
        try:
            vectors = env.build_sample_vectors(gold_ast, stype="CREDIT_VS_CAP_COMPOSED_STYLE")
            self.assertGreaterEqual(len(vectors), 2)
            scored = env.compute_reward_group([gold_ast, wrong_ast, {}], vectors=vectors)
            r_gold, bd_gold = scored[0]
            r_wrong, bd_wrong = scored[1]
            r_fail, _ = scored[2]
            self.assertEqual(bd_gold["vector_passes"], bd_gold["vector_total"])
            self.assertGreater(r_gold, r_wrong)
            self.assertGreater(r_wrong, r_fail)
            self.assertLess(r_fail, 0.0)
        finally:
            env.evaluator.close()

    def test_v2_supervised_and_grpo_smoke_pipeline_with_inference_loading(self):
        from compiler.training.dataset_generator import generate_synthetic_dataset

        with tempfile.TemporaryDirectory() as td:
            sup_path = Path(td) / "v2_sup.pt"
            rl_path = Path(td) / "v2_rl.pt"
            ds_path = Path(td) / "v2_ds.pt"

            ds = generate_synthetic_dataset(num_samples=120, seed=42, v2=True)
            torch.save(ds, ds_path)

            # Train long enough (~2s) that the smoke checkpoint produces valid non-abstained v2 ASTs
            train_supervised_dual_stream(
                dataset_path=ds_path,
                checkpoint_out=sup_path,
                epochs=8,
                lr=1e-3,
                batch_size=16,
                smoke_test=False,
                v2=True,
            )
            self.assertTrue(sup_path.exists())
            sup_ckpt = torch.load(sup_path, map_location="cpu", weights_only=True)
            self.assertEqual(sup_ckpt["config"]["architecture_version"], "v2")
            self.assertEqual(sup_ckpt["config"]["num_span_units"], len(SPAN_UNITS))
            self.assertEqual(sup_ckpt["span_units"], SPAN_UNITS)

            # Verify DualStreamDecompilerInference auto-routes loaded v2 checkpoint and succeeds strictly
            engine = DualStreamDecompilerInference(checkpoint_path=sup_path)
            self.assertEqual(engine.architecture_version, "v2")
            self.assertEqual(engine.model.num_span_units, len(SPAN_UNITS))
            out = engine.decompile_text(
                "A charge of 12% applies to qualified revenue exceeding $50,000.",
            )
            # An 8-epoch smoke checkpoint has low RelF1 and may fail strict affine checks;
            # either way it must either emit valid v2 AST or fail closed with loweringError (never silent wrong AST).
            self.assertIn(out.get("status"), ("SUCCESS", "LOWERING_FAILED"), f"Unexpected status: {out}")
            if out["status"] == "SUCCESS":
                self.assertEqual(out.get("synthesizedAst", {}).get("astVersion"), "v2")
            else:
                self.assertIsNone(out.get("synthesizedAst"))
                self.assertTrue(str(out.get("loweringError", "")).startswith("ERR_"))
            self.assertFalse(out.get("isAuthorizedToDisburse", False))

            # Run 1-epoch GRPO smoke test on v2 supervised checkpoint
            train_grpo_logic_only(
                checkpoint_in=sup_path,
                checkpoint_out=rl_path,
                smoke_test=True,
            )
            self.assertTrue(rl_path.exists())
            rl_ckpt = torch.load(rl_path, map_location="cpu", weights_only=True)
            self.assertEqual(rl_ckpt["config"]["architecture_version"], "v2")
            self.assertEqual(rl_ckpt["span_units"], SPAN_UNITS)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_section_c_rate_bps_count_date_and_missing_required_fail_closed(self):
        ast = {
            "ruleId": "section_c_alignment",
            "version": "2.0",
            "astVersion": "v2",
            "inputs": {
                "baseAmount": {"type": "MONEY_MINOR"},
                "dynamicRate": {"type": "RATE_BPS"},
                "dependentCount": {"type": "COUNT"},
                "filingDate": {"type": "DATE"},
            },
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "pipeline": {
                    "steps": [
                        {
                            "assign": "rawTax",
                            "expr": {
                                "op": "RATIONAL_MULTIPLY",
                                "base": "inputs.baseAmount",
                                "rateBps": "inputs.dynamicRate",
                                "denominator": 10000,
                            },
                        },
                        {
                            "assign": "scaledTax",
                            "expr": {
                                "op": "CONDITIONAL",
                                "test": {"op": "GT", "left": "inputs.dependentCount", "right": 0},
                                "then": "vars.rawTax",
                                "else": {"op": "TO_PAISE", "value": "0"},
                            },
                        },
                    ],
                    "output": {"scaledTax": "vars.scaledTax", "recordedDate": "inputs.filingDate"},
                },
            },
        }
        catala_src = emit_catala_code(
            ast,
            scenarios=[
                {
                    "name": "OneDependent",
                    "facts": {
                        "baseAmount": 100000,
                        "dynamicRate": 1500,
                        "dependentCount": 1,
                        "filingDate": "2026-04-01",
                    },
                }
            ],
        )
        self.assertIn("data dynamic_rate content integer", catala_src)
        self.assertIn("data dependent_count content integer", catala_src)
        self.assertIn("data filing_date content date", catala_src)
        self.assertIn("-- filing_date: |2026-04-01|", catala_src)

        ev = CanonicalUsvmEvaluator()
        try:
            res = ev.evaluate(
                ast,
                {
                    "baseAmount": 100000,
                    "dynamicRate": 1500,
                    "dependentCount": 1,
                    "filingDate": "2026-04-01",
                },
            )
            self.assertEqual(res["scaledTax"], 15000)

            with self.assertRaises(RuntimeError) as ctx:
                ev.evaluate(
                    ast,
                    {
                        "baseAmount": None,
                        "dynamicRate": 1500,
                        "dependentCount": 1,
                        "filingDate": "2026-04-01",
                    },
                )
            self.assertIn("ERR_MISSING_INPUT", str(ctx.exception))
        finally:
            ev.close()

    def test_v2_combinatorial_ast_dataset_split_and_zero_section_hacks(self):
        from compiler.training.dataset_generator import generate_synthetic_dataset

        ds = generate_synthetic_dataset(num_samples=200, seed=42, v2=True)
        train_items = ds["train"]
        val_items = ds["val"]
        self.assertEqual(len(train_items), 160)
        self.assertEqual(len(val_items), 40)
        self.assertGreater(len(train_items), len(val_items) * 2)

        all_items = train_items + val_items
        stypes = {s["stype"] for s in all_items}
        for bad_kw in ("194Q", "194C", "206C", "IRC", "MSMED", "GST", "IBC"):
            self.assertTrue(all(bad_kw not in st for st in stypes), f"Found leaked section hack stype: {stypes}")

        expected_families = {
            "V2_AST_COMPOSED_ARITHMETIC",
            "V2_AST_BOUNDED_BRACKET",
            "V2_AST_BOOLEAN_DAG",
            "V2_AST_COLLECTION_FOLD",
            "V2_AST_SCOPED_SUBSTITUTION",
            "V2_AST_DEONTIC_GATE",
            "V2_AST_FOLD_TO_ARITHMETIC",
            "V2_AST_GATED_ARITHMETIC",
            "V2_AST_MULTI_TIER_BRACKET",
            "V2_AST_BRANCHED_RATES",
        }
        self.assertTrue(expected_families.issubset(stypes), f"Missing families: {expected_families - stypes}")

        for item in all_items:
            if item["modality_name"] != "MODALITY_ABSTAIN_UNRESOLVED":
                self.assertEqual(len(item["unit_labels"]), len(item["span_boundaries"]))

    def test_combinatorial_ast_composer_lowers_and_executes_in_usvm(self):
        import re
        from compiler.neural.inference_v2 import synthesize_usvm_ast_v2
        from compiler.training.ast_composer import CombinatorialAstComposer
        from compiler.training.usvm_rl_environment import StatutoryRlEnvironment
        from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

        composer = CombinatorialAstComposer(seed=123)
        env = StatutoryRlEnvironment()
        ev = CanonicalUsvmEvaluator()
        try:
            seen_families = set()
            saw_conditional_gate = False
            saw_fold_to_arith = False
            saw_branched_conditional = False
            saw_branched_additive = False
            for i in range(100):
                sample = composer.sample_rule(formulation=i % 4)
                seen_families.add(sample["stype"])

                # Verify single consistent currency per rule
                txt_l = sample["text"].lower()
                ccy_hits = set()
                if "$" in txt_l or re.search(r"\b(us\s+dollars|dollars|usd)\b", txt_l):
                    ccy_hits.add("USD")
                if "\u20b9" in txt_l or re.search(r"\b(rupees|rs\.?|inr|lakh|crore)\b", txt_l):
                    ccy_hits.add("INR")
                if "\u20ac" in txt_l or re.search(r"\b(euros|eur)\b", txt_l):
                    ccy_hits.add("EUR")
                if "\u00a3" in txt_l or re.search(r"\b(pounds|gbp)\b", txt_l):
                    ccy_hits.add("GBP")
                if re.search(r"\bsgd\b", txt_l):
                    ccy_hits.add("SGD")
                if re.search(r"\bchf\b", txt_l):
                    ccy_hits.add("CHF")
                self.assertLessEqual(
                    len(ccy_hits),
                    1,
                    f"Mixed currencies {ccy_hits} in single rule: {sample['text']!r}",
                )

                clauses = sample["clauses"]
                cmods = []
                pos = 0
                for c_idx, c_txt in enumerate(clauses):
                    ci = sample["text"].find(c_txt, pos)
                    cmods.append({"clauseIdx": c_idx, "modality": sample["modality_name"], "charRange": [ci, ci + len(c_txt)]})
                    pos = ci + len(c_txt)

                occupied = []
                order = sorted(range(len(sample["spans_with_units"])), key=lambda k: (-len(sample["spans_with_units"][k][0]), k))
                placed = [None] * len(sample["spans_with_units"])
                for oi in order:
                    txt, role, unit = sample["spans_with_units"][oi]
                    sp = 0
                    chosen = None
                    while True:
                        k = sample["text"].find(txt, sp)
                        if k == -1:
                            break
                        if all(k + len(txt) <= a or b <= k for a, b in occupied):
                            chosen = (k, k + len(txt))
                            occupied.append(chosen)
                            break
                        sp = k + 1
                    self.assertIsNotNone(chosen, f"Could not place span {txt!r} in {sample['text']!r}")
                    c_idx = 0
                    for cm in cmods:
                        cs, ce = cm["charRange"]
                        if chosen[0] >= cs and chosen[1] <= ce:
                            c_idx = cm["clauseIdx"]
                            break
                    placed[oi] = {
                        "spanIdx": oi,
                        "span": txt,
                        "role": role,
                        "unit": unit,
                        "charRange": list(chosen),
                        "clauseIdx": c_idx,
                    }

                edges = [{"sourceIdx": a, "targetIdx": b, "relation": r} for a, b, r in sample["relations"]]
                ast = synthesize_usvm_ast_v2(
                    text=sample["text"],
                    spans=placed,
                    edges=edges,
                    clause_modalities=cmods,
                    dominant_modality=sample["modality_name"],
                )
                self.assertEqual(ast.get("astVersion"), "v2")
                self.assertFalse(ast["executionDiagnostics"]["regexHeuristicsUsed"])

                # Verify boolean outputs never carry a 'Paisa' suffix
                for out_name, out_spec in ast.get("outputs", {}).items():
                    if isinstance(out_spec, dict) and out_spec.get("type") == "BOOLEAN":
                        self.assertFalse(out_name.endswith("Paisa"), f"Boolean output has Paisa suffix: {out_name}")

                # Track cross-primitive AST compositions
                ast_json = str(ast)
                if sample["stype"] == "V2_AST_GATED_ARITHMETIC" and "CONDITIONAL" in ast_json:
                    saw_conditional_gate = True
                if sample["stype"] == "V2_AST_FOLD_TO_ARITHMETIC" and ("SUM_OVER" in ast_json or "COUNT" in ast_json):
                    saw_fold_to_arith = True
                if sample["stype"] == "V2_AST_BRANCHED_RATES":
                    if "conditional-rate-branch" in ast_json and "'op': 'CONDITIONAL'" in ast_json:
                        saw_branched_conditional = True
                    if "additive-rate-sum" in ast_json and "'op': 'ADD'" in ast_json:
                        saw_branched_additive = True

                cands = env._build_v2_fact_candidates(ast)
                self.assertGreaterEqual(len(cands), 3)
                for _, cvec in cands:
                    out = ev.evaluate(ast, cvec)
                    self.assertIsNotNone(out)
            self.assertEqual(len(seen_families), 10)
            self.assertTrue(saw_conditional_gate)
            self.assertTrue(saw_fold_to_arith)
            self.assertTrue(saw_branched_conditional or saw_branched_additive)
        finally:
            ev.close()

    def test_stage_e_literals_currencies_segmentation_and_logic_vocab_pos(self):
        from compiler.neural.literals import (
            infer_rule_currency_info,
            parse_comparison_operator,
            segment_clauses,
        )
        from compiler.neural.logic_reasoner import LOGIC_VOCAB, tokens_to_logic_ids

        self.assertEqual(parse_comparison_operator("does not exceed"), "LTE")
        self.assertEqual(parse_comparison_operator("fewer than"), "LT")
        self.assertEqual(parse_comparison_operator("at or above"), "GTE")
        self.assertEqual(parse_comparison_operator("strictly greater than"), "GT")

        self.assertEqual(infer_rule_currency_info([{"span": "SGD 25,000"}], "SGD 25,000")[0], "SGD")
        self.assertEqual(infer_rule_currency_info([{"span": "CHF 10,000"}], "CHF 10,000")[0], "CHF")
        self.assertEqual(infer_rule_currency_info([{"span": "AED 50,000"}], "AED 50,000")[0], "AED")

        # Abbreviation & inline list references must not falsely split into tiny clauses
        text = "Under Sec. 12 and Reg. 4 (a) and (b), an entity shall file notice."
        toks = text.split()
        offs = []
        cursor = 0
        for tk in toks:
            idx = text.find(tk, cursor)
            offs.append((idx, idx + len(tk)))
            cursor = idx + len(tk)
        segs = segment_clauses(text, toks, offs)
        self.assertEqual(len(segs), 1)

        # Hyphenated compound legal terms must map to <LEGAL_NOUN>/<LEGAL_VERB>, never <UNK>
        ids = tokens_to_logic_ids(["cross-border", "shall", "re-verify", "taxpayer's", "allocation"])
        self.assertNotIn(LOGIC_VOCAB["<UNK>"], ids)
        self.assertEqual(ids[0], LOGIC_VOCAB["<LEGAL_NOUN>"])
        self.assertEqual(ids[2], LOGIC_VOCAB["<LEGAL_VERB>"])
        self.assertEqual(ids[3], LOGIC_VOCAB["<LEGAL_NOUN>"])

    def test_batched_amp_v2_train_smoke(self):
        import tempfile
        from pathlib import Path
        import torch
        from compiler.training.dataset_generator import generate_synthetic_dataset
        from compiler.training.train_dual_stream import train_supervised_dual_stream

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds_path = tmp_path / "v2_smoke_ds.pt"
            ckpt_path = tmp_path / "v2_smoke_ckpt.pt"
            ds = generate_synthetic_dataset(num_samples=40, seed=7, v2=True)
            torch.save(ds, ds_path)

            out_ckpt = train_supervised_dual_stream(
                dataset_path=ds_path,
                checkpoint_out=ckpt_path,
                epochs=1,
                lr=5e-4,
                batch_size=8,
                smoke_test=True,
                v2=True,
            )
            self.assertTrue(ckpt_path.exists())
            self.assertEqual(Path(out_ckpt), ckpt_path)
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            metrics = ckpt_data.get("metrics", {})
            self.assertIn("rel_f1", metrics)
            self.assertIn("unit_acc", metrics)
            self.assertIn("macro_f1", metrics)
            self.assertIn("sem_pass_pct", metrics)
            self.assertIn("disc_count", metrics)

            disc_log = tmp_path / "v2_smoke_ckpt_discrepancies.jsonl"
            self.assertTrue(disc_log.exists())
            log_lines = [ln for ln in disc_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertTrue(any('"EPOCH_SEMANTIC_AUDIT"' in ln for ln in log_lines))

    def test_discrepancy_monitor_catches_intended_vs_actual_drift(self):
        import copy
        from compiler.neural.inference_v2 import synthesize_usvm_ast_v2
        from compiler.neural.statutory_neural_parser import ID2UNIT, V2_ID2REL, V2_ID2ROLE
        from compiler.training.dataset_generator import generate_synthetic_dataset, tokenize_with_offsets
        from compiler.training.discrepancy_monitor import (
            check_ast_semantic_invariants,
            compare_ast_counterfactual_outputs,
            verify_v2_dataset_contract,
        )
        from compiler.training.usvm_rl_environment import StatutoryRlEnvironment
        from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

        ds = generate_synthetic_dataset(num_samples=50, seed=11, v2=True)
        summary = verify_v2_dataset_contract(ds, max_samples=40, strict=True)
        self.assertGreater(summary["checked"], 10)
        self.assertEqual(summary["errorCount"], 0)
        self.assertEqual(summary["passed"], summary["checked"])

        # 1. Fault injection: mask-blocked gold relation edge
        bad_ds = copy.deepcopy(ds)
        corrupted = False
        for it in bad_ds["train"]:
            if it.get("modality_name") != "MODALITY_ABSTAIN_UNRESOLVED" and any(
                cell != 0 for row in it.get("rel_matrix", []) for cell in row
            ):
                it["unit_labels"] = [0] * len(it.get("unit_labels", []))
                corrupted = True
                break
        self.assertTrue(corrupted)
        with self.assertRaises(RuntimeError) as ctx:
            verify_v2_dataset_contract(bad_ds, max_samples=50, strict=True)
        self.assertIn("MASK_BLOCKED_GOLD_EDGE", str(ctx.exception))

        # 2. Fault injection: semantic invariant checker flags Boolean Paisa suffix & wrong branch op
        invs = check_ast_semantic_invariants(
            {
                "astVersion": "v2",
                "executionDiagnostics": {"regexHeuristicsUsed": False},
                "outputs": {"thresholdMetPaisa": {"type": "BOOLEAN"}},
                "computation": {"pipeline": {"steps": [{"op": "MUL"}]}},
            },
            dag_meta={"kind": "BRANCHED_RATES", "mode": "conditional"},
        )
        self.assertIn("BOOLEAN_PAISA_SUFFIX:thresholdMetPaisa", invs)
        self.assertIn("BRANCHED_RATES_CONDITIONAL_MISSING_CONDITIONAL_OP", invs)

        # 3. Fault injection: counterfactual evaluator catches semantic op drift (MUL/SUB/CONDITIONAL -> constant wrong value)
        arith_sample = next(
            it for it in (ds["val"] + ds["train"])
            if it.get("stype") in ("V2_AST_BRANCHED_RATES", "V2_AST_COMPOSED_ARITHMETIC")
            and it.get("modality_name") != "MODALITY_ABSTAIN_UNRESOLVED"
        )
        _, offs = tokenize_with_offsets(arith_sample["text"])
        c_spans = arith_sample.get("clause_spans") or [(0, len(offs) - 1)]
        cmods = [
            {"clauseIdx": ci, "modality": arith_sample["modality_name"], "charRange": [offs[c0][0], offs[c1][1]]}
            for ci, (c0, c1) in enumerate(c_spans)
        ]
        spans = [
            {
                "spanIdx": i,
                "span": arith_sample["text"][offs[ts][0]:offs[te][1]],
                "role": V2_ID2ROLE[arith_sample["role_labels"][i]],
                "unit": ID2UNIT[arith_sample["unit_labels"][i]],
                "charRange": [offs[ts][0], offs[te][1]],
                "clauseIdx": next((ci for ci, (ccs, cce) in enumerate(c_spans) if ts >= ccs and te <= cce), 0),
            }
            for i, (ts, te) in enumerate(arith_sample["span_boundaries"])
        ]
        edges = [
            {"sourceIdx": r_i, "targetIdx": c_i, "relation": V2_ID2REL[rid]}
            for r_i, row in enumerate(arith_sample["rel_matrix"])
            for c_i, rid in enumerate(row)
            if r_i != c_i and rid != 0
        ]
        gold_ast = synthesize_usvm_ast_v2(
            text=arith_sample["text"],
            spans=spans,
            edges=edges,
            clause_modalities=cmods,
            dominant_modality=arith_sample["modality_name"],
        )
        drifted_ast = copy.deepcopy(gold_ast)
        for st in drifted_ast["computation"]["pipeline"]["steps"]:
            expr = st.get("expr")
            if isinstance(expr, dict) and expr.get("op") == "RATIONAL_MULTIPLY":
                expr["rateBps"] = int(expr.get("rateBps", 100)) + 5000
            elif isinstance(expr, dict) and expr.get("op") in ("ADD", "SUB"):
                expr["op"] = "SUB" if expr["op"] == "ADD" else "ADD"
        env = StatutoryRlEnvironment()
        try:
            matched, diffs = compare_ast_counterfactual_outputs(
                gold_ast, drifted_ast, env, env.evaluator, stype=arith_sample["stype"]
            )
            self.assertFalse(matched)
            self.assertGreater(len(diffs), 0)
        finally:
            env.evaluator.close()

    def test_v2_fail_closed_affine_and_literal_checks(self):
        from compiler.neural.ssa_walker import V2LoweringError

        bad_cases = [
            (
                "ignore_edge",
                "10% on ignored $99",
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "10%"},
                    {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_IGNORE", "unit": "UNIT_MONEY", "span": "$99"},
                ],
                [{"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_RATE_TO_THRESHOLD"}],
                None,
                "ERR_EDGE_TO_IGNORED_SPAN",
            ),
            (
                "unparseable_rate",
                "some percent over $500",
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "some percent"},
                    {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "$500"},
                ],
                [{"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_RATE_TO_THRESHOLD"}],
                None,
                "ERR_UNPARSEABLE_RATE",
            ),
            (
                "unparseable_money",
                "10% over unknown threshold",
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "10%"},
                    {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "unknown threshold"},
                ],
                [{"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_RATE_TO_THRESHOLD"}],
                None,
                "ERR_UNPARSEABLE_MONEY",
            ),
            (
                "unconsumed_money",
                "Charge 12% exceeding $50,000.",
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "12%"},
                    {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "$50,000"},
                ],
                [],
                None,
                "ERR_UNCONSUMED_SPAN",
            ),
            (
                "excess_without_floor",
                "10% excess",
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "10%"},
                ],
                [],
                [{"clauseIdx": 0, "modality": "MODALITY_COMPUTATIONAL_EXCESS"}],
                "ERR_NO_THRESHOLD",
            ),
        ]
        for label, text, spans, edges, cmods, expected_prefix in bad_cases:
            with self.subTest(case=label):
                with self.assertRaises(V2LoweringError) as ctx:
                    synthesize_usvm_ast_v2(text, spans, edges, clause_modalities=cmods)
                self.assertIn(expected_prefix, str(ctx.exception))

    def test_unseen_compound_rules_node_scoped_surcharge_and_deontic_ratio(self):
        env = StatutoryRlEnvironment()
        try:
            # Rule A: node-scoped conditional surcharge over duration window
            rule_a_text = "Tax is 1.5% on excess over EUR 250,000 plus EUR 5,000 if settlement occurs after 30 days."
            ast_a = synthesize_usvm_ast_v2(
                rule_a_text,
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "1.5%", "charRange": [7, 11]},
                    {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "over EUR 250,000", "charRange": [22, 38]},
                    {"spanIdx": 2, "clauseIdx": 0, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "EUR 5,000", "charRange": [44, 53]},
                    {"spanIdx": 3, "clauseIdx": 0, "role": "ROLE_DEADLINE_WINDOW", "unit": "UNIT_DURATION", "span": "settlement occurs after 30 days", "charRange": [57, 88]},
                ],
                [
                    {"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_RATE_TO_THRESHOLD"},
                    {"sourceIdx": 2, "targetIdx": 0, "relationType": "REL_BASE_TAX_ADD"},
                    {"sourceIdx": 3, "targetIdx": 2, "relationType": "REL_CONDITION_TO_GATE"},
                ],
            )
            self.assertEqual(env.evaluator.evaluate(ast_a, {"amount": 35000000, "settlementOccursDays": 20})["amountTaxPaisa"], 150000)
            self.assertEqual(env.evaluator.evaluate(ast_a, {"amount": 35000000, "settlementOccursDays": 31})["amountTaxPaisa"], 650000)
            self.assertEqual(env.evaluator.evaluate(ast_a, {"amount": 20000000, "settlementOccursDays": 31})["amountTaxPaisa"], 500000)

            # Rule B: deontic quantitative ratio + multi-year duration + boolean predicate
            rule_b_text = "Distribution prohibited unless reserve is at least 15% of net profits and holding period under 3 years and verified by auditor."
            ast_b = synthesize_usvm_ast_v2(
                rule_b_text,
                [
                    {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "unit": "UNIT_SYMBOL_VAR", "span": "reserve", "charRange": [31, 38]},
                    {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "at least 15%", "charRange": [42, 54]},
                    {"spanIdx": 2, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "unit": "UNIT_SYMBOL_VAR", "span": "net profits", "charRange": [58, 69]},
                    {"spanIdx": 3, "clauseIdx": 0, "role": "ROLE_DEADLINE_WINDOW", "unit": "UNIT_DURATION", "span": "holding period under 3 years", "charRange": [74, 102]},
                    {"spanIdx": 4, "clauseIdx": 0, "role": "ROLE_AGENT_STATE_PRECONDITION", "unit": "UNIT_BOOL_PREDICATE", "span": "verified by auditor", "charRange": [107, 126]},
                ],
                [
                    {"sourceIdx": 1, "targetIdx": 2, "relationType": "REL_RATE_TO_THRESHOLD"},
                    {"sourceIdx": 1, "targetIdx": 3, "relationType": "REL_LOGICAL_AND"},
                    {"sourceIdx": 3, "targetIdx": 4, "relationType": "REL_LOGICAL_AND"},
                ],
                clause_modalities=[{"clauseIdx": 0, "modality": "MODALITY_DEONTIC_PROHIBITION"}],
            )
            out_key = list(ast_b["computation"]["pipeline"]["output"].keys())[0]
            self.assertIs(env.evaluator.evaluate(ast_b, {"reserve": 160000, "netProfits": 1000000, "holdingPeriodDays": 500, "verifiedAuditor": True})[out_key], True)
            self.assertIs(env.evaluator.evaluate(ast_b, {"reserve": 100000, "netProfits": 1000000, "holdingPeriodDays": 500, "verifiedAuditor": True})[out_key], False)
            self.assertIs(env.evaluator.evaluate(ast_b, {"reserve": 160000, "netProfits": 1000000, "holdingPeriodDays": 1200, "verifiedAuditor": True})[out_key], False)
            vecs_b = env.build_sample_vectors(ast_b, stype="V2_AST_DEONTIC_GATE")
            self.assertGreaterEqual(len(vecs_b), 2)
        finally:
            env.evaluator.close()

    def test_catala_emitter_dynamic_enum_domain_without_hardcoded_table(self):
        from tests.test_model_v2_commit2 import TestModelV2Commit2

        enum_ast = {
            "ruleId": "dyn_enum_tier",
            "version": "2.0",
            "astVersion": "v2",
            "inputs": {"tier": {"type": "ENUM:TierCategory"}},
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "pipeline": {
                    "steps": [{"assign": "eligible", "expr": {"op": "EQ", "left": "inputs.tier", "right": "GOLD"}}],
                    "output": {"eligible": "vars.eligible"},
                },
            },
        }
        src = emit_catala_code(
            enum_ast,
            title="Dynamic Enum Tier",
            scenarios=[
                {"name": "Gold", "facts": {"tier": "GOLD"}},
                {"name": "Silver", "facts": {"tier": "SILVER"}},
            ],
        )
        self.assertIn("declaration enumeration TierCategory:", src)
        self.assertIn("-- Gold", src)
        self.assertIn("-- Silver", src)
        TestModelV2Commit2.setUpClass()
        cat_helper = TestModelV2Commit2()
        if cat_helper.can_run_catala:
            res = cat_helper._run_catala(
                enum_ast,
                [
                    {"name": "Gold", "facts": {"tier": "GOLD"}},
                    {"name": "Silver", "facts": {"tier": "SILVER"}},
                ],
            )
            self.assertTrue(res["results"]["Gold"]["success"], res["results"]["Gold"]["errors"])
            self.assertTrue(res["results"]["Silver"]["success"], res["results"]["Silver"]["errors"])
            self.assertEqual(res["results"]["Gold"]["values"]["eligible"], True)
            self.assertEqual(res["results"]["Silver"]["values"]["eligible"], False)

    def test_v2_abstain_with_fiscal_keyword_and_relational_edges_key(self):
        from unittest.mock import patch
        from compiler.neural.statutory_neural_parser import (
            RULE_MODALITIES,
            V2_BIO2ID,
            V2_RELATION_TYPES,
        )

        # Build an isolated v2 engine instance in-memory with a deterministic stub model
        engine = object.__new__(DualStreamDecompilerInference)
        engine.device = torch.device("cpu")
        engine.architecture_version = "v2"
        engine.vocab = {"<PAD>": 0, "<UNK>": 1, "tax": 2, "invoice": 3}
        engine.id2bio = {i: b for i, b in enumerate(V2_BIO_TAGS)}
        engine.id2role = dict(enumerate(V2_SEMANTIC_ROLES))
        engine.id2rel = {i: r for i, r in enumerate(V2_RELATION_TYPES)}
        engine.id2mod = {i: m for i, m in enumerate(RULE_MODALITIES)}
        engine.id2unit = dict(enumerate(SPAN_UNITS))
        engine.id2dom = {0: "DT_OPERATIVE_ENTITY"}
        engine.transition_mask = torch.zeros((len(V2_BIO_TAGS), len(V2_BIO_TAGS)), dtype=torch.float32)

        class _StubModel(torch.nn.Module):
            def __init__(self, abstain_prob: float, mod_idx: int, unit_logits_vec: list[float]):
                super().__init__()
                self.abstain_prob = abstain_prob
                self.mod_idx = mod_idx
                self.unit_logits_vec = unit_logits_vec

            def forward(self, raw_token_ids, logic_token_ids, suffix_ids=None, clause_spans=None, span_indices=None):
                t = raw_token_ids.size(1)
                mod_l = torch.full((1, len(RULE_MODALITIES)), -10.0)
                mod_l[0, self.mod_idx] = 10.0
                dom_p = torch.zeros((1, t, 8))
                dom_p[0, :, 0] = 1.0
                dom_a = torch.tensor([[1.0 - self.abstain_prob, self.abstain_prob]], dtype=torch.float32)
                bio_l = torch.full((t, len(V2_BIO_TAGS)), -10.0)
                bio_l[0, V2_BIO2ID["B-RATE"]] = 10.0
                bio_l[1:, V2_BIO2ID["O"]] = 10.0
                role_l = torch.zeros((1, len(V2_SEMANTIC_ROLES)))
                role_l[0, 1] = 10.0
                rel_l = torch.zeros((1, 1, len(V2_RELATION_TYPES)))
                unit_l = torch.tensor([self.unit_logits_vec], dtype=torch.float32)
                return {
                    "modality_logits": mod_l,
                    "clause_mod_logits": None,
                    "domain_probs": dom_p,
                    "domain_abstain_probs": dom_a,
                    "bio_logits": bio_l,
                    "role_logits": role_l,
                    "rel_logits": rel_l,
                    "unit_logits": unit_l,
                }

        # Case 1: Fiscal keywords present ("tax invoice") + non-abstain modality, but dom_abstain_prob=0.95 -> clean abstain
        unit_vec = [-10.0] * len(SPAN_UNITS)
        unit_vec[UNIT2ID["UNIT_RATE"]] = 10.0
        engine.model = _StubModel(abstain_prob=0.95, mod_idx=0, unit_logits_vec=unit_vec)
        out_abstain = engine.decompile_text("General policy tax invoice note without rates.")
        self.assertEqual(out_abstain["status"], "UNRESOLVED_STATUTORY_AMBIGUITY")
        self.assertFalse(out_abstain["isAuthorizedToDisburse"])

        # Case 2: Defensive guard if synthesize_usvm_ast_v2 returns UNRESOLVED_STATUTORY_AMBIGUITY directly
        engine.model = _StubModel(abstain_prob=0.05, mod_idx=0, unit_logits_vec=unit_vec)
        with patch(
            "compiler.neural.dual_stream_inference.synthesize_usvm_ast_v2",
            return_value={"status": "UNRESOLVED_STATUTORY_AMBIGUITY", "isAuthorizedToDisburse": False},
        ):
            out_passthrough = engine.decompile_text("10% tax invoice note.")
        self.assertEqual(out_passthrough["status"], "UNRESOLVED_STATUTORY_AMBIGUITY")

        # Case 3: LOWERING_FAILED uses "relationalEdges" (matching SUCCESS schema)
        out_fail = engine.decompile_text("nonnumeric tax invoice.")
        self.assertEqual(out_fail["status"], "LOWERING_FAILED")
        self.assertIn("relationalEdges", out_fail)
        self.assertNotIn("relationEdges", out_fail)


if __name__ == "__main__":
    unittest.main()
