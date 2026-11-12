"""
Unit tests for the canonical USVM -> Catala lowering.

Fixtures in tests/fixtures/financial_pipeline_asts.json are genuine ASTs
captured from DualStreamDecompilerInference (not hand-written schemas), so
these tests exercise the same DECLARATIVE_PIPELINE shape the neural
decompiler actually emits. Semantic equivalence against the official Catala
compiler is verified in tests/test_financial_pipeline_integration.py.
"""

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from compiler.converters.catala_emitter import (
    CatalaEmitter,
    CatalaLoweringError,
    _money_literal,
    _to_pascal_case,
    _to_snake_case,
    emit_catala_code,
    save_catala_file,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "financial_pipeline_asts.json").read_text(encoding="utf-8")
)


def _facts(invoice, prior=0):
    return {"current_invoice": invoice, "primary_base": invoice, "prior_purchases": prior}


class TestNamingAndLiterals(unittest.TestCase):
    def test_naming_helpers(self):
        self.assertEqual(_to_snake_case("purchaseAmount"), "purchase_amount")
        self.assertEqual(_to_snake_case("Section 194Q Excess"), "section_194q_excess")
        self.assertEqual(_to_pascal_case("sec_194q_excess"), "Sec194qExcess")
        self.assertEqual(_to_pascal_case("irc_section_3301"), "IrcSection3301")

    def test_money_literals_are_exact_paise(self):
        self.assertEqual(_money_literal(0), "$0.00")
        self.assertEqual(_money_literal(5000000), "$5000000.00")
        self.assertEqual(_money_literal(Fraction(1, 2)), "$0.50")
        with self.assertRaises(CatalaLoweringError):
            _money_literal(Fraction(1, 3))  # sub-paisa precision must fail closed
        with self.assertRaises(CatalaLoweringError):
            _money_literal(Fraction(1, 300))


class TestCanonicalPipelineLowering(unittest.TestCase):
    def setUp(self):
        self.emitter = CatalaEmitter(indent_size=2)

    def test_flat_rate_payroll(self):
        ast = FIXTURES["futa"]["ast"]
        code = self.emitter.emit(ast, scenarios=[{"name": "Wages100k", "facts": _facts(100000)}])
        self.assertIn("declaration scope Draft94dea82c:", code)
        self.assertIn("data current_invoice content money", code)
        self.assertIn("definition standard_rate_bps equals 600", code)
        self.assertIn("definition threshold_paisa equals $0.00", code)
        # 6% must be lowered from 600 bps, never fabricated as 0%.
        self.assertIn("((decimal of standard_rate_bps) / 10000.0)", code)
        self.assertNotIn("0.0%", code)
        self.assertIn("declaration scope Wages100k:", code)
        self.assertIn("output result scope Draft94dea82c", code)
        self.assertIn("-- current_invoice: $100000.00", code)

    def test_excess_threshold_194q(self):
        ast = FIXTURES["s194q"]["ast"]
        code = self.emitter.emit(ast)
        self.assertIn("definition threshold_paisa equals $500000000.00", code)
        self.assertIn("definition standard_rate_bps equals 10", code)
        # Excess-over-threshold semantics: cumulative base minus threshold, floored at zero.
        self.assertIn("cum_after", code)
        self.assertIn("raw_excess", code)
        self.assertIn("already_excess", code)
        self.assertIn("taxable_base_paisa", code)

    def test_cliff_threshold_gst51_compound_amount(self):
        ast = FIXTURES["gst51"]["ast"]
        code = self.emitter.emit(ast)
        # "two lakh and fifty thousand rupees" = 2,50,000 -> 25,000,000 paise.
        self.assertIn("definition threshold_paisa equals $25000000.00", code)
        self.assertIn("definition standard_rate_bps equals 200", code)
        self.assertIn("breached", code)

    def test_full_decompiler_result_is_unwrapped(self):
        ast = FIXTURES["futa"]["ast"]
        code = emit_catala_code({"synthesizedAst": ast, "text": "provision text"})
        self.assertIn("declaration scope Draft94dea82c:", code)
        self.assertIn('> "provision text"', code)

    def test_save_catala_file(self):
        ast = FIXTURES["futa"]["ast"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "futa.catala_en"
            saved = save_catala_file(ast, out_file, title="IRC Section 3301 FUTA")
            self.assertTrue(saved.exists())
            content = saved.read_text(encoding="utf-8")
            self.assertIn("# IRC Section 3301 FUTA", content)
            self.assertIn("definition standard_rate_bps equals 600", content)


class TestFailClosedLowering(unittest.TestCase):
    def test_legacy_invented_schema_is_rejected(self):
        legacy_ast = {
            "ruleId": "sec_194q_withholding",
            "predictedModality": "MODALITY_COMPUTATIONAL_EXCESS",
            "computation": {
                "archetype": "AST_EXCESS_THRESHOLD",
                "base": "purchase_value",
                "rate": 0.001,
                "excessThreshold": 50000,
            },
        }
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code(legacy_ast)

    def test_missing_computation_is_rejected(self):
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code({"ruleId": "x", "predictedModality": "MODALITY_DEONTIC_PROHIBITION"})

    def test_unsupported_operator_is_rejected(self):
        ast = json.loads(json.dumps(FIXTURES["futa"]["ast"]))
        ast["computation"]["pipeline"]["steps"][0]["expr"] = {"op": "COMPOUND_INTEREST", "principal": 1}
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code(ast)

    def test_unknown_fact_key_is_rejected(self):
        ast = json.loads(json.dumps(FIXTURES["futa"]["ast"]))
        ast["computation"]["pipeline"]["steps"][0]["expr"]["then"] = "facts.mysteryAmount"
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code(ast)

    def test_unsupported_rounding_policy_is_rejected(self):
        ast = json.loads(json.dumps(FIXTURES["futa"]["ast"]))
        ast["computation"]["pipeline"]["steps"].append({
            "assign": "roundedTax",
            "expr": {"op": "ROUND", "value": "vars.tdsAmountPaisa", "policy": "SECTION_288B"},
        })
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code(ast)

    def test_scenario_missing_fact_is_rejected(self):
        ast = FIXTURES["futa"]["ast"]
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code(ast, scenarios=[{"name": "Incomplete", "facts": {"current_invoice": 1}}])


if __name__ == "__main__":
    unittest.main()
