"""
Cross-engine conformance for the French tax 2026 fixtures.

The Catala-verified suite proves that emitted Catala reproduces statutory
ground truth. This suite proves a different, equally important property: the
production USVM and the Catala verification backend agree on *the same
executable AST* for *the same facts*, field by field.

Both engines consume ``tests/fixtures/french_tax_2026/*/ast.json``. Catala is
reached through the official compiler/interpreter harness; USVM is reached
through the canonical Node bridge. A drift in either direction (unit handling,
rounding, a mistyped constant) fails here before it can become a legal error.

Fact names differ by design: USVM uses camelCase fact keys, Catala uses the
snake_case fields declared by the shared ``FACT_BINDINGS`` map. The test
translates between them with that single map instead of maintaining a second
hand-written table.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.converters.archetype_strategies import normalize_ast
from compiler.converters.catala_emitter import (
    _identifier,
    _scope_identifier,
    emit_catala_code,
)
from compiler.converters.fact_bindings import FACT_BINDINGS
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "french_tax_2026"
HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"


# Catala field name -> canonical USVM fact key. Money facts have aliases in
# FACT_BINDINGS (Paisa/Inr/Eur); the first alias is stable and the French
# fixtures never use the INR aliases, so a plain setdefault is deterministic.
FIELD_TO_FACT: dict[str, str] = {}
for _fact, (_field, _typ) in FACT_BINDINGS.items():
    FIELD_TO_FACT.setdefault(_field, _fact)


def _usvm_facts(catala_facts: dict) -> dict:
    """Translate a snake_case Catala scenario into USVM camelCase facts."""
    try:
        return {FIELD_TO_FACT[key]: value for key, value in catala_facts.items()}
    except KeyError as exc:  # pragma: no cover - guards scenario typos
        raise AssertionError(f"Scenario fact {exc.args[0]!r} has no USVM binding") from exc


def _vat_facts(**overrides) -> dict:
    facts = {
        "transaction_amount": 1000,
        "customer_type": "B2C",
        "customer_country": "FRANCE",
        "goods_category": "standard",
        "eu_b2c_sales": 0,
        "customer_country_vat_rate_bps": 2000,
        "has_valid_vat_id": False,
    }
    facts.update(overrides)
    return facts


# Every scenario that exercises a legal branch of B1-B5. The expected values
# themselves live in test_french_tax_catala_verified.py; this table only needs
# facts because the assertion is engine equality, not ground truth.
SCENARIOS: dict[str, list[tuple[str, dict]]] = {
    "fr_direct_tax_irpp": [
        ("IrppSingle", {"taxable_income": 20000, "quotient_familial_parts": 1}),
        ("IrppCouple", {"taxable_income": 100000, "quotient_familial_parts": 2}),
        ("IrppCircularExample", {"taxable_income": 30000, "quotient_familial_parts": 1}),
    ],
    "fr_corporate_tax_is": [
        ("IsSme", {"taxable_profit": 40000, "is_sme_eligible": True}),
        ("IsAboveThreshold", {"taxable_profit": 60000, "is_sme_eligible": True}),
        ("IsNotEligible", {"taxable_profit": 40000, "is_sme_eligible": False}),
    ],
    "fr_social_contributions": [
        ("SocialBelowCeiling", {"gross_salary": 30000, "rgdu_relief": 0}),
        ("SocialAboveCeiling", {"gross_salary": 60000, "rgdu_relief": 0}),
        ("SocialWithRgdu", {"gross_salary": 60000, "rgdu_relief": 1000}),
        ("SocialHighAboveCsgCap", {"gross_salary": 200000, "rgdu_relief": 0}),
    ],
    "fr_vat_tva": [
        ("VatDomesticB2C", _vat_facts()),
        ("VatDomesticB2B", _vat_facts(customer_type="B2B", has_valid_vat_id=True)),
        (
            "VatReverseCharge",
            _vat_facts(customer_type="B2B", customer_country="EU", has_valid_vat_id=True),
        ),
        (
            "VatEuB2cBelow",
            _vat_facts(
                customer_country="EU",
                eu_b2c_sales=5000,
                customer_country_vat_rate_bps=1700,
            ),
        ),
        (
            "VatEuB2cAbove",
            _vat_facts(
                customer_country="EU",
                eu_b2c_sales=20000,
                customer_country_vat_rate_bps=1700,
            ),
        ),
        (
            "VatExport",
            _vat_facts(customer_country="NON_EU", customer_type="B2B", has_valid_vat_id=True),
        ),
        ("VatReduced", _vat_facts(goods_category="reduced")),
    ],
    "fr_property_tax": [
        ("Property20pct", {"cadastral_value": 20000, "municipal_rate_bps": 2000}),
        ("Property30pct", {"cadastral_value": 20000, "municipal_rate_bps": 3000}),
    ],
}


def _as_float(value) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean is not a numeric output")
    return float(value)


class TestFrenchTaxCrossEngine(unittest.TestCase):
    """USVM and Catala must agree on every declared output of every fixture."""

    @classmethod
    def setUpClass(cls):
        problems = []
        if not HARNESS.exists():
            problems.append(f"missing harness {HARNESS}")
        if not INTERPRETER.exists():
            problems.append("missing interpreter (run: node scripts/fetch_catala_interpreter.mjs)")
        if shutil.which("node") is None:
            problems.append("node is not on PATH")
        if problems:
            message = "Catala verification unavailable: " + "; ".join(problems)
            if os.environ.get("REQUIRE_CATALA") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        cls.evaluator = CanonicalUsvmEvaluator()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    def _load_canonical_ast(self, suite: str) -> dict:
        with open(FIXTURE_DIR / suite / "ast.json", encoding="utf-8") as f:
            return normalize_ast(json.load(f))

    def _run_catala(self, suite: str, ast: dict, scenarios: list[dict]) -> dict:
        source = emit_catala_code(ast, title=ast["sectionId"], scenarios=scenarios)
        scope_names = [_scope_identifier(s["name"]) for s in scenarios]
        request = {"id": suite, "source": source, "scopes": scope_names}
        proc = subprocess.run(
            ["node", str(HARNESS)],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, f"{suite}: harness failed: {proc.stderr}")
        response = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(
            response["typecheck"]["success"],
            f"{suite}: Catala typecheck failed:\n{response['typecheck']['errors']}",
        )
        results = {}
        for name in scope_names:
            run = response["results"][name]
            self.assertTrue(
                run["success"], f"{suite}/{name}: Catala interpret failed:\n{run['errors']}"
            )
            results[name] = run["values"]
        return results

    def _assert_cross_engine(self, suite: str) -> None:
        ast = self._load_canonical_ast(suite)
        scenarios = SCENARIOS[suite]
        outputs = ast["computation"]["pipeline"]["output"]
        expected_catala_outputs = {_identifier(key) for key in outputs}

        catala = self._run_catala(
            suite, ast, [{"name": name, "facts": facts} for name, facts in scenarios]
        )

        for (name, catala_facts), scope in zip(scenarios, catala):
            usvm_result = self.evaluator.evaluate(ast, _usvm_facts(catala_facts))
            self.assertFalse(usvm_result.get("error"), f"{suite}/{name}: {usvm_result.get('error')}")
            values = catala[scope]

            self.assertEqual(
                set(values),
                expected_catala_outputs,
                f"{suite}/{name}: Catala outputs {sorted(values)} != AST outputs "
                f"{sorted(expected_catala_outputs)}",
            )

            for key in outputs:
                catala_name = _identifier(key)
                self.assertIn(key, usvm_result, f"{suite}/{name}: USVM missing output {key!r}")
                usvm_value = usvm_result[key]
                catala_value = values[catala_name]

                if isinstance(catala_value, bool):
                    self.assertIs(
                        bool(usvm_value),
                        catala_value,
                        f"{suite}/{name}: {key} USVM={usvm_value!r} != Catala={catala_value!r}",
                    )
                    continue
                if isinstance(catala_value, str):
                    self.assertEqual(
                        str(usvm_value),
                        catala_value,
                        f"{suite}/{name}: {key} USVM={usvm_value!r} != Catala={catala_value!r}",
                    )
                    continue

                # USVM names money in minor units when the output key says
                # Cents/Paisa; Catala money is always major units.
                expected = _as_float(usvm_value)
                if key.endswith(("Cents", "Paisa")):
                    expected /= 100.0
                self.assertAlmostEqual(
                    _as_float(catala_value),
                    expected,
                    places=2,
                    msg=f"{suite}/{name}: {key} Catala={catala_value!r} != USVM={usvm_value!r}",
                )

    def test_irpp_cross_engine(self):
        self._assert_cross_engine("fr_direct_tax_irpp")

    def test_corporate_tax_cross_engine(self):
        self._assert_cross_engine("fr_corporate_tax_is")

    def test_payroll_cross_engine(self):
        self._assert_cross_engine("fr_social_contributions")

    def test_vat_cross_engine(self):
        self._assert_cross_engine("fr_vat_tva")

    def test_property_tax_cross_engine(self):
        self._assert_cross_engine("fr_property_tax")

    def test_integer_fact_must_not_be_fractional(self):
        """A count or bps fact is an integer; sub-integer input fails closed."""
        ast = self._load_canonical_ast("fr_direct_tax_irpp")
        with self.assertRaises(RuntimeError):
            self.evaluator.evaluate(
                ast, {"taxableIncomeEur": 30000, "quotientFamilialParts": 1.5}
            )


if __name__ == "__main__":
    unittest.main()
