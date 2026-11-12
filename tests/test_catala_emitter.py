"""
Unit Tests for MudraForge AST-to-Catala Emitter Bridge
======================================================
Validates translation of Declarative OKF v0.2 ASTs into literate, idiomatic
Catala (.catala_en) source code according to the ACM ICFP 2021 Catala specification.
"""

import tempfile
import unittest
from pathlib import Path

from compiler.converters.catala_emitter import (
    CatalaEmitter,
    emit_catala_code,
    save_catala_file,
    _format_currency,
    _to_pascal_case,
    _to_snake_case,
)


class TestCatalaEmitter(unittest.TestCase):
    def setUp(self):
        self.emitter = CatalaEmitter(indent_size=2)

    def test_naming_helpers(self):
        self.assertEqual(_to_snake_case("purchaseAmount"), "purchase_amount")
        self.assertEqual(_to_snake_case("Section 194Q Excess"), "section_194q_excess")
        self.assertEqual(_to_pascal_case("sec_194q_excess"), "Sec194qExcess")
        self.assertEqual(_to_pascal_case("irc_section_3301"), "IrcSection3301")

    def test_format_currency(self):
        self.assertEqual(_format_currency(50000, "USD"), "$50,000")
        self.assertEqual(_format_currency(1000000, "INR"), "INR 1,000,000")
        self.assertEqual(_format_currency(2500.50, "EUR"), "2,500.50 EUR")
        self.assertEqual(_format_currency(0, "USD"), "$0")

    def test_emit_excess_threshold_rule(self):
        """
        Tests Section 194Q archetype: tax imposed on purchase value exceeding 50,000.
        """
        ast = {
            "ruleId": "sec_194q_withholding",
            "predictedModality": "MODALITY_COMPUTATIONAL_EXCESS",
            "rawText": "Any person paying to a resident sum for purchase of goods exceeding 50 Lakh rupees shall deduct 0.1 percent.",
            "computation": {
                "archetype": "AST_EXCESS_THRESHOLD",
                "base": "purchase_value",
                "rate": 0.001,
                "excessThreshold": 50000,
                "currency": "USD",
            },
            "temporalScope": {
                "cutoverDate": "2021-07-01",
                "sunsetDate": "2026-03-31",
            },
            "symbolLegend": {
                "buyer_turnover": {"type": "MONETARY_THRESHOLD"},
                "is_resident_seller": {"type": "BOOLEAN_FLAG"},
            },
        }

        catala_code = self.emitter.emit(ast, title="Section 194Q TDS Formalization")

        self.assertIn("# Section 194Q TDS Formalization", catala_code)
        self.assertIn("declaration structure Sec194qWithholdingInput:", catala_code)
        self.assertIn("data purchase_value content money", catala_code)
        self.assertIn("data transaction_date content date", catala_code)
        self.assertIn("data buyer_turnover content money", catala_code)
        self.assertIn("data is_resident_seller content boolean", catala_code)

        self.assertIn("declaration scope Sec194qWithholding:", catala_code)
        self.assertIn("input input_data content Sec194qWithholdingInput", catala_code)
        self.assertIn("output calculated_assessment content money", catala_code)

        self.assertIn("scope Sec194qWithholding:", catala_code)
        self.assertIn("under condition input_data.transaction_date >= |2021-07-01| and input_data.transaction_date <= |2026-03-31|", catala_code)
        self.assertIn("definition calculated_assessment equals", catala_code)
        self.assertIn("if input_data.purchase_value > $50,000 then", catala_code)
        self.assertIn("(input_data.purchase_value - $50,000) * 0.10%", catala_code)
        self.assertIn("else", catala_code)
        self.assertIn("$0", catala_code)

    def test_emit_cliff_threshold_rule(self):
        """
        Tests cliff threshold archetype: full amount subject to rate once cliff is breached.
        """
        ast = {
            "ruleId": "sec_194c_contractor",
            "predictedModality": "MODALITY_COMPUTATIONAL_CLIFF",
            "computation": {
                "archetype": "AST_CLIFF_THRESHOLD",
                "base": "single_contract_payment",
                "rate": 0.01,
                "cliffThreshold": 30000,
                "currency": "INR",
            },
            "temporalScope": {
                "taxYear": 2026,
            },
        }

        catala_code = emit_catala_code(ast)

        self.assertIn("declaration structure Sec194cContractorInput:", catala_code)
        self.assertIn("data single_contract_payment content money", catala_code)
        self.assertIn("data taxable_year content integer", catala_code)
        self.assertIn("under condition input_data.taxable_year = 2026", catala_code)
        self.assertIn("if input_data.single_contract_payment >= INR 30,000 then", catala_code)
        self.assertIn("input_data.single_contract_payment * 1.00%", catala_code)
        self.assertIn("INR 0", catala_code)

    def test_emit_progressive_slab_rule(self):
        """
        Tests multi-bracket progressive rate table.
        """
        ast = {
            "ruleId": "progressive_tax_schedule",
            "predictedModality": "MODALITY_COMPUTATIONAL_SLAB",
            "computation": {
                "archetype": "AST_PROGRESSIVE_SLAB",
                "base": "taxable_income",
                "currency": "USD",
                "brackets": [
                    {"cap": 11000, "rate": 0.10, "fixed": 0},
                    {"cap": 44725, "rate": 0.12, "fixed": 1100},
                    {"cap": None, "rate": 0.22, "fixed": 5147},
                ],
            },
        }

        catala_code = emit_catala_code(ast)

        self.assertIn("if input_data.taxable_income <= $11,000 then", catala_code)
        self.assertIn("input_data.taxable_income * 10.00%", catala_code)
        self.assertIn("else if input_data.taxable_income <= $44,725 then", catala_code)
        self.assertIn("$1,100 + (input_data.taxable_income - $11,000) * 12.00%", catala_code)
        self.assertIn("else", catala_code)
        self.assertIn("$5,147 + (input_data.taxable_income - $44,725) * 22.00%", catala_code)

    def test_emit_deontic_permission_and_prohibition(self):
        """
        Tests boolean deontic verdicts (permission and prohibition).
        """
        perm_ast = {
            "ruleId": "foreign_tax_credit_permission",
            "predictedModality": "MODALITY_DEONTIC_PERMISSION",
        }
        perm_code = emit_catala_code(perm_ast)
        self.assertIn("output is_permitted content boolean", perm_code)
        self.assertIn("definition is_permitted equals", perm_code)
        self.assertIn("input_data.is_resident", perm_code)

        prohib_ast = {
            "ruleId": "unauthorized_disbursement_prohibition",
            "predictedModality": "MODALITY_DEONTIC_PROHIBITION",
        }
        prohib_code = emit_catala_code(prohib_ast)
        self.assertIn("output is_prohibited content boolean", prohib_code)
        self.assertIn("definition is_prohibited equals", prohib_code)
        self.assertIn("true", prohib_code)

    def test_save_catala_file(self):
        """
        Verifies exporting Catala source to a file path.
        """
        ast = {
            "ruleId": "irc_section_3301_futa",
            "predictedModality": "MODALITY_COMPUTATIONAL_SLAB",
            "computation": {
                "base": "total_wages",
                "rate": 0.06,
                "threshold": 7000,
                "currency": "USD",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "futa_3301.catala_en"
            saved_p = save_catala_file(ast, out_file, title="IRC Section 3301 FUTA")
            self.assertTrue(saved_p.exists())
            content = saved_p.read_text(encoding="utf-8")
            self.assertIn("# IRC Section 3301 FUTA", content)
            self.assertIn("let capped_base = min input_data.total_wages $7,000 in", content)
            self.assertIn("capped_base * 6.00%", content)


if __name__ == "__main__":
    unittest.main()
