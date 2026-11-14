"""Commit 2 tests for Model v2 regex-free lowerer, type-masked decoding, and auto-routing."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.converters.catala_emitter import _identifier, emit_catala_code
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.neural.inference_v2 import (
    apply_type_compatibility_mask,
    synthesize_usvm_ast_v2,
)
from compiler.neural.logic_reasoner import DualStreamStatutoryParser, LOGIC_VOCAB
from compiler.neural.ssa_walker import V2LoweringError
from compiler.neural.statutory_neural_parser import (
    RULE_MODALITIES,
    SPAN_UNITS,
    UNIT2ID,
    V2_BIO_TAGS,
    V2_REL2ID,
    V2_RELATION_TYPES,
    V2_SEMANTIC_ROLES,
)
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator


class TestModelV2Commit2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = CanonicalUsvmEvaluator()
        cls.harness = REPO_ROOT / "scripts" / "catala_verify.mjs"
        cls.interpreter = REPO_ROOT / "scripts" / "catala_interpreter.cjs"
        cls.can_run_catala = shutil.which("node") is not None and cls.harness.exists() and cls.interpreter.exists()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    def _run_catala(self, ast: dict, scenarios: list[dict]) -> dict | None:
        if not self.can_run_catala:
            return None
        src = emit_catala_code(ast, title="V2 Test Rule", scenarios=scenarios)
        req = {"id": "v2-c2", "source": src, "scopes": [s["name"] for s in scenarios]}
        proc = subprocess.run(
            ["node", str(self.harness)],
            input=json.dumps(req) + "\n",
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, f"Catala harness error: {proc.stderr}")
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(res["typecheck"]["success"], f"Catala typecheck failed:\n{res['typecheck']['errors']}")
        return res

    def test_type_compatibility_mask_blocks_temporal_and_count_from_money_edges(self):
        # Span 0: "28%" (UNIT_RATE), Span 1: "1 taxable year" (UNIT_DURATION),
        # Span 2: "2 children" (UNIT_COUNT), Span 3: "$36,900" (UNIT_MONEY)
        unit_preds = [
            UNIT2ID["UNIT_RATE"],
            UNIT2ID["UNIT_DURATION"],
            UNIT2ID["UNIT_COUNT"],
            UNIT2ID["UNIT_MONEY"],
        ]
        n_rel = len(V2_RELATION_TYPES)
        rel_logits = torch.zeros((4, 4, n_rel), dtype=torch.float32)
        # Force high raw scores on forbidden edges (rate -> duration, rate -> count)
        rel_logits[0, 1, V2_REL2ID["REL_RATE_TO_THRESHOLD"]] = 9.0
        rel_logits[0, 2, V2_REL2ID["REL_RATE_TO_THRESHOLD"]] = 8.0
        rel_logits[1, 2, V2_REL2ID["REL_MIN_CAP"]] = 7.0
        rel_logits[2, 3, V2_REL2ID["REL_BASE_TAX_ADD"]] = 6.0
        # Valid edge (rate -> money)
        rel_logits[0, 3, V2_REL2ID["REL_RATE_TO_THRESHOLD"]] = 5.0

        masked = apply_type_compatibility_mask(rel_logits, unit_preds)
        preds = masked.argmax(dim=-1).tolist()

        # Forbidden edges fall back to REL_NO_EDGE (0); valid rate -> money stays intact
        self.assertEqual(preds[0][1], 0)
        self.assertEqual(preds[0][2], 0)
        self.assertEqual(preds[1][2], 0)
        self.assertEqual(preds[2][3], 0)
        self.assertEqual(preds[0][3], V2_REL2ID["REL_RATE_TO_THRESHOLD"])

    def test_progressive_bracket_via_rel_base_tax_add_without_regex(self):
        # Bracket text rewritten so NO regex could guess baseTax unless REL_BASE_TAX_ADD is read!
        header = "Section 1. Tax on taxable income. "
        body = "Five thousand five hundred thirty-five dollars initially; rate twenty-eight percent on excess above $36,900 capped at ceiling $89,150."
        text = header + body
        off = len(header)
        spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "unit": "UNIT_SYMBOL_VAR", "span": "taxable income", "charRange": [18, 32], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 1, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$5,535", "charRange": [off, off + 6], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 1, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "28%", "charRange": [off + 50, off + 53], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 1, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "above $36,900", "charRange": [off + 64, off + 77], "confidence": 0.99},
            {"spanIdx": 4, "clauseIdx": 1, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$89,150", "charRange": [off + 95, off + 102], "confidence": 0.99},
        ]
        edges = [
            {"sourceIdx": 1, "targetIdx": 2, "relationType": "REL_BASE_TAX_ADD"},
            {"sourceIdx": 2, "targetIdx": 3, "relationType": "REL_RATE_TO_THRESHOLD"},
            {"sourceIdx": 3, "targetIdx": 4, "relationType": "REL_BRACKET_NEXT"},
        ]
        ast = synthesize_usvm_ast_v2(
            text=text,
            spans=spans,
            edges=edges,
            clause_modalities=[
                {"clauseIdx": 0, "charRange": [0, off - 1]},
                {"clauseIdx": 1, "charRange": [off, len(text)]},
            ],
        )
        self.assertEqual(ast["astVersion"], "v2")
        self.assertFalse(ast["executionDiagnostics"]["regexHeuristicsUsed"])

        scenarios = [
            ("Below", {"taxableIncome": 2000000}, 0),
            ("Middle", {"taxableIncome": 5000000}, 920300),
            ("Above", {"taxableIncome": 10000000}, 0),
        ]
        for name, facts, expected in scenarios:
            usvm = self.evaluator.evaluate(ast, facts)
            self.assertEqual(usvm["taxableIncomeTaxPaisa"], expected, f"{name} mismatch")

        catala_res = self._run_catala(ast, [{"name": n, "facts": f} for n, f, _ in scenarios])
        if catala_res is not None:
            ck = _identifier("taxableIncomeTaxPaisa")
            for name, _, expected in scenarios:
                run = catala_res["results"][name]
                self.assertTrue(run["success"], run["errors"])
                self.assertEqual(round(float(run["values"][ck]) * 100), expected)

    def test_composed_credit_and_cap_exact_opcodes_and_order_invariance(self):
        text = "Tax equals 28 percent of excess of gross income over $36,900, minus credit of $500, and ceiling of $10,000."
        spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_STANDARD_STATUTORY_RATE", "unit": "UNIT_RATE", "span": "28 percent", "charRange": [11, 21], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "unit": "UNIT_SYMBOL_VAR", "span": "gross income", "charRange": [35, 47], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 0, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "unit": "UNIT_MONEY", "span": "over $36,900", "charRange": [48, 60], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 0, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$500", "charRange": [78, 82], "confidence": 0.99},
            {"spanIdx": 4, "clauseIdx": 0, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$10,000", "charRange": [99, 106], "confidence": 0.99},
        ]
        edges = [
            {"sourceIdx": 0, "targetIdx": 2, "relationType": "REL_RATE_TO_THRESHOLD"},
            {"sourceIdx": 3, "targetIdx": 0, "relationType": "REL_SUB_CREDIT"},
            {"sourceIdx": 4, "targetIdx": 0, "relationType": "REL_MIN_CAP"},
        ]
        cm = [{"clauseIdx": 0, "charRange": [0, len(text)]}]
        ast1 = synthesize_usvm_ast_v2(text=text, spans=spans, edges=edges, clause_modalities=cm)
        ast2 = synthesize_usvm_ast_v2(text=text, spans=list(reversed(spans)), edges=list(reversed(edges)), clause_modalities=cm)
        self.assertEqual(ast1["computation"]["pipeline"], ast2["computation"]["pipeline"])

        for gross, exp in ((3700000, 0), (4000000, 36800), (5000000, 316800), (10000000, 1000000)):
            self.assertEqual(self.evaluator.evaluate(ast1, {"grossIncome": gross})["grossIncomeTaxPaisa"], exp)

        # Cycle fail-closed check
        cyclic_edges = edges + [{"sourceIdx": 0, "targetIdx": 0, "relationType": "REL_RATE_TO_THRESHOLD"}]
        with self.assertRaises(V2LoweringError) as ctx:
            synthesize_usvm_ast_v2(text=text, spans=spans, edges=cyclic_edges, clause_modalities=cm)
        self.assertIn("ERR_CYCLIC_DEPENDENCY", str(ctx.exception))

    def test_defined_term_substitution_and_ambiguous_scope_failure(self):
        c0 = "(a) The basic standard deduction is $4,400. "
        c1 = "(b) For taxable years after December 31, 2017, and before January 1, 2026, $18,000 shall be substituted for $4,400."
        text = c0 + c1
        off = len(c0)
        spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_DEFINED_TERM", "unit": "UNIT_SYMBOL_VAR", "span": "standard deduction", "charRange": [14, 32], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$4,400", "charRange": [36, 42], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 1, "role": "ROLE_PRESCRIPTION_DEADLINE", "unit": "UNIT_YEAR_DATE", "span": "December 31, 2017", "charRange": [off + 28, off + 45], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 1, "role": "ROLE_PRESCRIPTION_DEADLINE", "unit": "UNIT_YEAR_DATE", "span": "January 1, 2026", "charRange": [off + 58, off + 73], "confidence": 0.99},
            {"spanIdx": 4, "clauseIdx": 1, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$18,000", "charRange": [off + 75, off + 82], "confidence": 0.99},
            {"spanIdx": 5, "clauseIdx": 1, "role": "ROLE_EXCEPTION_CARVEOUT", "unit": "UNIT_MONEY", "span": "$4,400", "charRange": [off + 109, off + 115], "confidence": 0.99},
        ]
        edges = [
            {"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_DEFINED_TERM_OF"},
            {"sourceIdx": 4, "targetIdx": 0, "relationType": "REL_SUBSTITUTE_WHEN"},
        ]
        ast = synthesize_usvm_ast_v2(
            text=text,
            spans=spans,
            edges=edges,
            clause_modalities=[
                {"clauseIdx": 0, "charRange": [0, off - 1]},
                {"clauseIdx": 1, "charRange": [off, len(text)]},
            ],
        )
        for yr, expected in ((2017, 440000), (2018, 1800000), (2025, 1800000), (2026, 440000)):
            res = self.evaluator.evaluate(ast, {"taxYear": yr})
            self.assertEqual(res["standardDeductionPaisa"], expected)

        # Sibling conflicting definitions without substitution fail closed
        conflict_text = "(a) Standard deduction is $4,400. (b) Standard deduction is $6,000."
        c_spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_DEFINED_TERM", "unit": "UNIT_SYMBOL_VAR", "span": "standard deduction", "charRange": [4, 22], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$4,400", "charRange": [26, 32], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 1, "role": "ROLE_DEFINED_TERM", "unit": "UNIT_SYMBOL_VAR", "span": "standard deduction", "charRange": [38, 56], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 1, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "unit": "UNIT_MONEY", "span": "$6,000", "charRange": [60, 66], "confidence": 0.99},
        ]
        c_edges = [
            {"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_DEFINED_TERM_OF"},
            {"sourceIdx": 2, "targetIdx": 3, "relationType": "REL_DEFINED_TERM_OF"},
        ]
        with self.assertRaises(V2LoweringError) as ctx:
            synthesize_usvm_ast_v2(
                text=conflict_text,
                spans=c_spans,
                edges=c_edges,
                clause_modalities=[
                    {"clauseIdx": 0, "charRange": [0, 33]},
                    {"clauseIdx": 1, "charRange": [34, len(conflict_text)]},
                ],
            )
        self.assertIn("ERR_AMBIGUOUS_DEFINITION", str(ctx.exception))

    def test_boolean_kinship_and_collection_folds(self):
        # Pure boolean kinship rule never emits TO_PAISE("100")
        b_text = "Individual is not married and maintains household for 1 taxable year."
        b_spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_PROHIBITION_GATE", "unit": "UNIT_BOOL_PREDICATE", "span": "married", "charRange": [18, 25], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_PROHIBITION_GATE", "unit": "UNIT_NONE", "span": "not", "charRange": [14, 17], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 0, "role": "ROLE_AGENT_STATE_PRECONDITION", "unit": "UNIT_BOOL_PREDICATE", "span": "maintains household", "charRange": [30, 49], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 0, "role": "ROLE_COUNT_QUANTIFIER", "unit": "UNIT_DURATION", "span": "1 taxable year", "charRange": [54, 68], "confidence": 0.99},
        ]
        b_edges = [
            {"sourceIdx": 1, "targetIdx": 0, "relationType": "REL_LOGICAL_NOT"},
            {"sourceIdx": 0, "targetIdx": 2, "relationType": "REL_LOGICAL_AND"},
        ]
        b_ast = synthesize_usvm_ast_v2(text=b_text, spans=b_spans, edges=b_edges, clause_modalities=[{"clauseIdx": 0, "charRange": [0, len(b_text)]}])
        self.assertNotIn("TO_PAISE", json.dumps(b_ast["computation"]))
        self.assertTrue(self.evaluator.evaluate(b_ast, {"married": False, "maintainsHousehold": True})["predicateResult"])
        self.assertFalse(self.evaluator.evaluate(b_ast, {"married": True, "maintainsHousehold": True})["predicateResult"])

        # SUM_OVER and COUNT folds
        sum_text = "Total sum of qualifying expenses paid across invoices."
        s_spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "unit": "UNIT_SYMBOL_VAR", "span": "qualifying expenses", "charRange": [13, 32], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 0, "role": "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", "unit": "UNIT_SYMBOL_VAR", "span": "invoice", "charRange": [45, 52], "confidence": 0.99},
        ]
        sum_ast = synthesize_usvm_ast_v2(
            text=sum_text,
            spans=s_spans,
            edges=[{"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_AGGREGATE_SUM"}],
            clause_modalities=[{"clauseIdx": 0, "charRange": [0, len(sum_text)]}],
        )
        self.assertIn("SUM_OVER", json.dumps(sum_ast["computation"]))

        cnt_ast = synthesize_usvm_ast_v2(
            text=sum_text,
            spans=s_spans,
            edges=[{"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_AGGREGATE_COUNT"}],
            clause_modalities=[{"clauseIdx": 0, "charRange": [0, len(sum_text)]}],
        )
        self.assertIn("COUNT", json.dumps(cnt_ast["computation"]))

    def test_dual_stream_inference_auto_routes_v2_and_preserves_v1(self):
        model = DualStreamStatutoryParser(
            raw_vocab_size=64,
            logic_vocab_size=len(LOGIC_VOCAB),
            num_roles=len(V2_SEMANTIC_ROLES),
            num_relations=len(V2_RELATION_TYPES),
            num_bio_tags=len(V2_BIO_TAGS),
            num_modalities=len(RULE_MODALITIES),
            num_span_units=len(SPAN_UNITS),
        )
        ckpt = {
            "config": {
                "architecture_version": "v2",
                "raw_vocab_size": 64,
                "logic_vocab_size": len(LOGIC_VOCAB),
                "num_span_units": len(SPAN_UNITS),
            },
            "raw_vocab": {"<PAD>": 0, "<UNK>": 1},
            "logic_vocab": LOGIC_VOCAB,
            "roles": V2_SEMANTIC_ROLES,
            "relations": V2_RELATION_TYPES,
            "bio_tags": V2_BIO_TAGS,
            "modalities": RULE_MODALITIES,
            "span_units": SPAN_UNITS,
            "model_state_dict": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v2_test.pt")
            torch.save(ckpt, p)
            inf = DualStreamDecompilerInference(p)
            self.assertEqual(inf.architecture_version, "v2")
            self.assertEqual(inf.num_span_units, len(SPAN_UNITS))
            self.assertIsNotNone(inf.model.logic_reasoner.unit_classifier)

        v1_ckpt = REPO_ROOT / "checkpoints" / "dual_stream_sup.pt"
        if v1_ckpt.exists():
            inf_v1 = DualStreamDecompilerInference(str(v1_ckpt))
            self.assertEqual(inf_v1.architecture_version, "v1")
            self.assertEqual(inf_v1.num_span_units, 0)


if __name__ == "__main__":
    unittest.main()
