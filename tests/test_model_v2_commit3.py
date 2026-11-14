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
        with tempfile.TemporaryDirectory() as td:
            sup_path = Path(td) / "v2_sup.pt"
            rl_path = Path(td) / "v2_rl.pt"

            train_supervised_dual_stream(checkpoint_out=sup_path, smoke_test=True, v2=True)
            self.assertTrue(sup_path.exists())
            sup_ckpt = torch.load(sup_path, map_location="cpu", weights_only=True)
            self.assertEqual(sup_ckpt["config"]["architecture_version"], "v2")
            self.assertEqual(sup_ckpt["config"]["num_span_units"], len(SPAN_UNITS))
            self.assertEqual(sup_ckpt["span_units"], SPAN_UNITS)

            # Verify DualStreamDecompilerInference auto-routes loaded v2 checkpoint
            engine = DualStreamDecompilerInference(checkpoint_path=sup_path)
            self.assertEqual(engine.architecture_version, "v2")
            self.assertEqual(engine.model.num_span_units, len(SPAN_UNITS))
            out = engine.decompile_text(
                "A levy of 10% applies on gross income exceeding $5,000.",
            )
            self.assertIn("compiler", out)
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


if __name__ == "__main__":
    unittest.main()
