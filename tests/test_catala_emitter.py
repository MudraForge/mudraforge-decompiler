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
    _money_literal_minor,
    _scope_identifier,
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

    def test_scope_identifier_always_valid(self):
        # Scope names must start with a capital letter; digit-leading or
        # punctuation-only names would otherwise emit invalid Catala.
        self.assertEqual(_scope_identifier("Wages100k"), "Wages100k")
        self.assertEqual(_scope_identifier("100k bonus"), "S100kBonus")
        self.assertEqual(_scope_identifier("fr-vat-tva"), "FrVatTva")
        self.assertEqual(_scope_identifier("---"), "StatutoryScope")

    def test_money_literals_are_exact_paise(self):
        # _money_literal takes major units; _money_literal_minor takes paise.
        self.assertEqual(_money_literal(0), "$0.00")
        self.assertEqual(_money_literal(5000000), "$5000000.00")
        self.assertEqual(_money_literal(Fraction(1, 2)), "$0.50")
        self.assertEqual(_money_literal_minor(0), "$0.00")
        self.assertEqual(_money_literal_minor(25000000), "$250000.00")
        self.assertEqual(_money_literal_minor(4250000), "$42500.00")
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
        self.assertIn("definition threshold_paisa equals $5000000.00", code)
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
        self.assertIn("definition threshold_paisa equals $250000.00", code)
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


class TestFrenchTaxFixtures(unittest.TestCase):
    """French tax 2026 fixture archetypes lower to valid Catala source.

    These fixtures were authored with a flat schema (top-level archetype,
    computationSteps, output) and non-canonical operators (TO_CENTS, FROM_CENTS,
    VAR, MULTIPLY, LE, PROGRESSIVE_SLAB). The archetype normalizers convert them
    to DECLARATIVE_PIPELINE before the emitter renders Catala.
    """

    FIXTURE_DIR = Path(__file__).parent / "fixtures" / "french_tax_2026"

    def _load_ast(self, suite: str) -> dict:
        with open(self.FIXTURE_DIR / suite / "ast.json", encoding="utf-8") as f:
            return json.load(f)

    def test_vat_rate_lookup_lowers(self):
        ast = self._load_ast("fr_vat_tva")
        code = emit_catala_code(ast, title="FR-TVA-2026")
        self.assertIn("declaration scope FrTva2026:", code)
        self.assertIn("declaration enumeration CustomerType:", code)
        self.assertIn("declaration enumeration GoodsCategory:", code)
        self.assertIn("definition vat_rate_bps equals", code)
        self.assertIn("= Standard", code)  # Catala enum case, not a string literal
        self.assertIn("data customer_type content CustomerType", code)
        self.assertIn("definition vat_amount_cents equals", code)
        self.assertIn("output transaction_amount_eur content money", code)
        self.assertIn("output is_reverse_charge content boolean", code)

    def test_tiered_corporate_tax_lowers(self):
        ast = self._load_ast("fr_corporate_tax_is")
        code = emit_catala_code(ast, title="FR-IS-2026")
        self.assertIn("declaration scope FrIs2026:", code)
        # A fact used only inside a condition must still be an input field.
        self.assertIn("data is_sme_eligible content boolean", code)
        self.assertIn("definition applicable_rate_bps equals", code)
        self.assertIn("definition corporate_tax_cents equals", code)
        self.assertIn("output corporate_tax_eur content money", code)

    def test_progressive_slab_tax_lowers(self):
        ast = self._load_ast("fr_direct_tax_irpp")
        code = emit_catala_code(ast, title="FR-IRPP-2026")
        self.assertIn("declaration scope FrIrpp2026:", code)
        self.assertIn("definition income_tax_cents equals", code)
        self.assertIn("slab_", code)  # progressive slab expansion creates slab_N vars
        self.assertIn("output income_tax_eur content money", code)

    def test_payroll_contributions_lowers(self):
        ast = self._load_ast("fr_social_contributions")
        code = emit_catala_code(ast, title="FR-SOCIAL-2026")
        self.assertIn("declaration scope FrSocial2026:", code)
        self.assertIn("definition employer_contributions_cents equals", code)
        self.assertIn("definition employee_contributions_cents equals", code)
        self.assertIn("output net_salary_eur content money", code)
        self.assertIn("output total_employer_cost_eur content money", code)

    def test_property_tax_lowers(self):
        ast = self._load_ast("fr_property_tax")
        code = emit_catala_code(ast, title="FR-PROPERTY-2026")
        self.assertIn("declaration scope FrProperty2026:", code)
        # The municipal rate is a required input fact, never a hardcoded range.
        self.assertIn("data municipal_rate_bps content integer", code)
        self.assertIn("definition taxable_base_cents equals", code)
        self.assertIn("definition property_tax_cents equals", code)
        self.assertIn("output property_tax_eur content money", code)

    def test_unknown_archetype_is_rejected(self):
        unknown_ast = {
            "sectionId": "UNKNOWN-2026",
            "archetype": "NON_EXISTENT_ARCHETYPE",
            "computationSteps": [],
            "output": {},
        }
        with self.assertRaises(CatalaLoweringError):
            emit_catala_code(unknown_ast)


class TestArchetypeStrategies(unittest.TestCase):
    """Unit tests for archetype normalizers (table-driven dispatch per AGENTS.md Rule 20)."""

    def test_registry_has_all_five_french_archetypes(self):
        from compiler.converters.archetype_strategies import ARCHETYPE_NORMALIZERS
        expected = {
            "VAT_RATE_LOOKUP_AND_COMPUTE",
            "TIERED_CORPORATE_TAX",
            "PROGRESSIVE_SLAB_TAX",
            "PAYROLL_CONTRIBUTIONS",
            "AD_VALOREM_PROPERTY_TAX",
        }
        self.assertEqual(set(ARCHETYPE_NORMALIZERS.keys()), expected)

    def test_canonical_ast_passes_through_unchanged(self):
        from compiler.converters.archetype_strategies import normalize_ast
        canonical = {
            "sectionId": "CANON-1",
            "computation": {
                "archetype": "DECLARATIVE_PIPELINE",
                "thresholdPaisa": "0",
                "standardRateBps": 100,
                "pipeline": {
                    "steps": [{"assign": "x", "expr": {"op": "LITERAL", "value": 1}}],
                    "output": {"x": "vars.x"},
                },
            },
        }
        result = normalize_ast(canonical)
        self.assertIs(result, canonical)  # same object, no copy

    def test_operator_aliases_are_mapped(self):
        from compiler.converters.archetype_strategies import _map_op
        expr = {"op": "TO_CENTS", "value": "facts.transactionAmountEur"}
        result = _map_op(expr)
        self.assertEqual(result["op"], "TO_PAISE")

        expr = {"op": "VAR", "value": "facts.customerType"}
        result = _map_op(expr)
        self.assertEqual(result["op"], "VARIABLE")
        self.assertEqual(result["var"], "facts.customerType")
        self.assertNotIn("value", result)

        expr = {"op": "LE", "left": "a", "right": "b"}
        result = _map_op(expr)
        self.assertEqual(result["op"], "LTE")

    def test_progressive_slab_expansion(self):
        from compiler.converters.archetype_strategies import _expand_progressive_slab
        steps = [
            {"assign": "base", "expr": {"op": "VARIABLE", "var": "vars.income"}},
            {"assign": "tax", "expr": {
                "op": "PROGRESSIVE_SLAB",
                "base": "vars.income",
                "slabs": [
                    {"lowerCents": 0, "upperCents": 10000, "rateBps": 0},
                    {"lowerCents": 10001, "upperCents": 20000, "rateBps": 1000},
                    {"lowerCents": 20001, "upperCents": None, "rateBps": 2000},
                ],
            }},
        ]
        _expand_progressive_slab(steps, 1)
        # PROGRESSIVE_SLAB step replaced by slab steps + sum
        self.assertEqual(steps[1]["assign"], "slab_1_portion_cents")
        self.assertEqual(steps[2]["assign"], "slab_1_tax_cents")
        self.assertEqual(steps[5]["assign"], "tax")  # sum step at end
        self.assertTrue(all(s["expr"].get("op") != "PROGRESSIVE_SLAB" for s in steps))


if __name__ == "__main__":
    unittest.main()
