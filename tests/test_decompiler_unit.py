"""
=============================================================================
MUDRA DECOMPILER: UNIT AND INTEGRATION TESTS
=============================================================================

Validates the full standalone neuro-symbolic pipeline:
1. Dual-stream transformer weights and logic vocabulary loading
2. Constrained Viterbi BIO sequence decoding
3. Modality prediction and semantic span extraction
4. Declarative AST synthesis conforming to OKF v0.2
5. USVM Node worker bridge execution and invariant enforcement
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.neural.tokenizer import tokenize_with_offsets
from compiler.neural.logic_reasoner import LOGIC_VOCAB, tokens_to_logic_ids
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator


class TestMudraDecompilerUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "dual_stream_sup.pt")
        if not os.path.exists(cls.ckpt_path):
            raise unittest.SkipTest(f"Checkpoint not found at {cls.ckpt_path}")
        cls.decompiler = DualStreamDecompilerInference(cls.ckpt_path)
        cls.evaluator = CanonicalUsvmEvaluator()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    def test_01_tokenizer_and_logic_vocab(self):
        text = "Any person responsible for paying to a resident sum exceeding fifty thousand rupees"
        tokens, offsets = tokenize_with_offsets(text)
        self.assertGreater(len(tokens), 5)
        self.assertEqual(len(tokens), len(offsets))

        logic_ids = tokens_to_logic_ids(tokens)
        self.assertEqual(len(logic_ids), len(tokens))
        for lid in logic_ids:
            self.assertIsInstance(lid, int)
            self.assertLess(lid, len(LOGIC_VOCAB))

    def test_02_decompile_computational_statute(self):
        clause = (
            "In the case of a head of household, the basic standard deduction is $4,400, "
            "except that for taxable years beginning after December 31, 2017, and before "
            "January 1, 2026, $18,000 shall be substituted for $4,400."
        )
        res = self.decompiler.decompile_text(clause)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["predictedModality"].startswith("MODALITY_COMPUTATIONAL"))
        self.assertGreater(res["classifiedSpansCount"], 0)
        self.assertIn("synthesizedAst", res)
        self.assertFalse(res["isAuthorizedToDisburse"])
        self.assertFalse(res["statutoryInvariants"]["isAuthorizedToDisburse"])

    def test_03_decompile_deontic_prohibition(self):
        clause = (
            "An individual shall not be treated as a dependent of a taxpayer if such individual "
            "has made a joint return with the individual's spouse for the taxable year."
        )
        res = self.decompiler.decompile_text(clause)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(
            res["predictedModality"].startswith("MODALITY_DEONTIC")
            or "STATUS" in res["predictedModality"]
        )
        self.assertFalse(res["isAuthorizedToDisburse"])

    def test_04_usvm_node_bridge_execution(self):
        ast = {
            "astSchemaVersion": "1.0.0",
            "sectionId": "UNIT_TEST_THRESHOLD",
            "sectionCode": "TEST_001",
            "version": "1.0.0",
            "executionMode": "DEVELOPMENT",
            "title": "Unit Test Statutory Threshold",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "steps": [
                    {
                        "assign": "isExceeding",
                        "expr": {
                            "op": "GT",
                            "left": "facts.amount",
                            "right": 50000,
                        },
                    }
                ],
                "output": {
                    "isExceeding": "vars.isExceeding",
                    "taxAmountPaisa": 0,
                    "isApplicable": "vars.isExceeding",
                },
            },
            "statutoryInvariants": {"isAuthorizedToDisburse": False},
        }

        facts = {"amount": 75000}
        resp = self.evaluator.evaluate_batch([
            {"ruleAst": ast, "facts": facts, "options": {"mode": "DEVELOPMENT"}}
        ])
        self.assertEqual(len(resp), 1)
        out = resp[0]
        self.assertFalse(out.get("error"))
        self.assertTrue(out.get("isApplicable", False))
        self.assertTrue(out.get("isExceeding", False))
        self.assertFalse(out["isAuthorizedToDisburse"])


if __name__ == "__main__":
    unittest.main()
