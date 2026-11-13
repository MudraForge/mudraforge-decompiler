"""
Compiler-verified tests for the French tax 2026 fixtures.

Unlike the string assertions in test_catala_emitter.py, these tests run the
emitted Catala through the official Catala toolchain and assert numeric
results. The interpreter is a generated ~10 MB artifact and is not committed:
tools/catala.lock.json pins its URL, mirror list, SHA-256, byte size, Catala
version and js_of_ocaml version. Fetch and verify it with
`node scripts/fetch_catala_interpreter.mjs`.

Ground truth: expected values are cross-checked against each fixture's
circular. All five emitted fixtures now reproduce the statutory computation
(including the 2026 LFSS payroll reform and the progressive IS tranches), so
there are no expected-value exceptions left. Set REQUIRE_CATALA=1 in CI to
fail instead of skip when the interpreter is unavailable.

Unit contract: Catala `money` is denominated in major units ($1.00 == 1
euro/rupee), so a `..._cents`/`..._paisa` money value of $924.00 means 92,400
minor units. Tests assert the major-unit amount and, where the minor-unit view
matters, its x100 relationship explicitly.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compiler.converters.catala_emitter import (
    CatalaLoweringError,
    _scope_identifier,
    emit_catala_code,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "french_tax_2026"
HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_catala_interpreter.mjs"
LOCKFILE = REPO_ROOT / "tools" / "catala.lock.json"
GOLDEN_SCRIPT = REPO_ROOT / "scripts" / "emit_catala_fixtures.py"


class TestCatalaInterpreterIntegrity(unittest.TestCase):
    """The interpreter the tests trust must match the pinned official artifact."""

    def test_lockfile_records_both_toolchain_versions(self):
        with open(LOCKFILE, encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertRegex(manifest["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(manifest["bytes"], 0)
        self.assertTrue(manifest["url"].startswith("https://"))
        # 6.4.1 is the js_of_ocaml version stamped in the bundle header, not
        # the Catala version. Both must be recorded separately.
        self.assertRegex(manifest["jsoo_version"], r"^\d+\.\d+\.\d+$")
        self.assertTrue(manifest["catala_version"])
        self.assertTrue(manifest.get("catala_version_source"))
        self.assertIsInstance(manifest.get("mirrors", []), list)

    def test_bundle_matches_pinned_hash_when_present(self):
        if shutil.which("node") is None:
            self.skipTest("node is not on PATH")
        if not INTERPRETER.exists():
            self.skipTest("interpreter absent; run: node scripts/fetch_catala_interpreter.mjs")
        proc = subprocess.run(
            ["node", str(FETCH_SCRIPT), "--check"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(
            proc.returncode, 0, f"interpreter integrity check failed: {proc.stderr or proc.stdout}"
        )

    def test_check_reads_pin_from_lockfile(self):
        if shutil.which("node") is None or not INTERPRETER.exists():
            self.skipTest("interpreter unavailable")
        with open(LOCKFILE, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["bytes"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            sabotaged = Path(tmp) / "catala.lock.json"
            sabotaged.write_text(json.dumps(manifest), encoding="utf-8")
            env = os.environ.copy()
            env["CATALA_LOCKFILE"] = str(sabotaged)
            proc = subprocess.run(
                ["node", str(FETCH_SCRIPT), "--check"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(REPO_ROOT),
                env=env,
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("expected", proc.stderr + proc.stdout)


class TestFrenchTaxCatalaVerified(unittest.TestCase):
    """Emit -> typecheck -> interpret with the official Catala toolchain."""

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

    def _load_ast(self, suite: str) -> dict:
        with open(FIXTURE_DIR / suite / "ast.json", encoding="utf-8") as f:
            return json.load(f)

    def _verify(self, suite: str, scope: str, scenarios: list[dict]) -> dict:
        ast = self._load_ast(suite)
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
        self.assertEqual(proc.returncode, 0, f"harness failed: {proc.stderr}")
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

    def _assert_values(self, actual: dict, expected: dict) -> None:
        for key, want in expected.items():
            self.assertIn(key, actual, f"missing output {key}")
            got = actual[key]
            if isinstance(want, bool):
                self.assertIs(got, want, f"{key}: {got!r} != {want!r}")
            elif isinstance(want, (int, float)):
                self.assertAlmostEqual(got, want, places=2, msg=f"{key}: {got} != {want}")
            else:
                self.assertEqual(got, want, f"{key}: {got!r} != {want!r}")

    # ── VAT (CGI art. 278; taux 2026) ────────────────────────────────────────

    def _vat_facts(self, **overrides) -> dict:
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

    def test_vat_domestic_and_reverse_charge(self):
        results = self._verify(
            "fr_vat_tva",
            "FrTva2026",
            [
                {"name": "VatDomesticB2C", "facts": self._vat_facts()},
                {
                    "name": "VatDomesticB2B",
                    "facts": self._vat_facts(customer_type="B2B", has_valid_vat_id=True),
                },
                {
                    "name": "VatReverseCharge",
                    "facts": self._vat_facts(
                        customer_type="B2B",
                        customer_country="EU",
                        has_valid_vat_id=True,
                    ),
                },
            ],
        )
        # Domestic French B2C: standard 20%.
        self._assert_values(
            results["Vatdomesticb2c"],
            {
                "transaction_amount_eur": 1000,
                "vat_rate_percent": 20,
                "vat_amount_eur": 200,
                "total_amount_eur": 1200,
                "is_reverse_charge": False,
                "is_export": False,
            },
        )
        # Domestic French B2B is not a reverse charge: French VAT still applies.
        self._assert_values(
            results["Vatdomesticb2b"],
            {
                "vat_rate_percent": 20,
                "vat_amount_eur": 200,
                "total_amount_eur": 1200,
                "is_reverse_charge": False,
            },
        )
        # EU B2B with a valid VAT id: reverse charge, no VAT invoiced.
        self._assert_values(
            results["Vatreversecharge"],
            {
                "vat_rate_percent": 0,
                "vat_amount_eur": 0,
                "total_amount_eur": 1000,
                "is_reverse_charge": True,
            },
        )

    def test_vat_eu_distance_selling_threshold(self):
        results = self._verify(
            "fr_vat_tva",
            "FrTva2026",
            [
                {
                    "name": "VatEuB2cBelow",
                    "facts": self._vat_facts(
                        customer_country="EU",
                        eu_b2c_sales=5000,
                        customer_country_vat_rate_bps=1700,
                    ),
                },
                {
                    "name": "VatEuB2cAbove",
                    "facts": self._vat_facts(
                        customer_country="EU",
                        eu_b2c_sales=20000,
                        customer_country_vat_rate_bps=1700,
                    ),
                },
            ],
        )
        # Below EUR 10,000 cumulative EU B2C sales: French rate applies.
        self._assert_values(
            results["Vateub2cbelow"],
            {"vat_rate_percent": 20, "vat_amount_eur": 200, "total_amount_eur": 1200},
        )
        # Above the threshold: destination-country rate (17% in this scenario).
        self._assert_values(
            results["Vateub2cabove"],
            {"vat_rate_percent": 17, "vat_amount_eur": 170},
        )

    def test_vat_export_is_zero_rated(self):
        results = self._verify(
            "fr_vat_tva",
            "FrTva2026",
            [
                {
                    "name": "VatExport",
                    "facts": self._vat_facts(
                        customer_country="NON_EU",
                        customer_type="B2B",
                        has_valid_vat_id=True,
                    ),
                }
            ],
        )
        # Exports outside the EU are VAT-free (CGI art. 262-I).
        self._assert_values(
            results["Vatexport"],
            {
                "vat_rate_percent": 0,
                "vat_amount_eur": 0,
                "total_amount_eur": 1000,
                "is_reverse_charge": False,
                "is_export": True,
            },
        )

    def test_vat_reduced_rate(self):
        results = self._verify(
            "fr_vat_tva",
            "FrTva2026",
            [
                {
                    "name": "VatReduced",
                    "facts": self._vat_facts(goods_category="reduced"),
                }
            ],
        )
        self._assert_values(
            results["Vatreduced"],
            {
                "vat_rate_percent": 5.5,
                "vat_amount_eur": 55,
                "total_amount_eur": 1055,
                "is_reverse_charge": False,
            },
        )

    def test_unknown_enum_case_fails_closed(self):
        with self.assertRaises(CatalaLoweringError):
            self._verify(
                "fr_vat_tva",
                "FrTva2026",
                [
                    {
                        "name": "VatBadCategory",
                        "facts": self._vat_facts(goods_category="nonexistent"),
                    }
                ],
            )

    # ── Corporate tax (IS 2026: 15% SME / 25% standard) ─────────────────────

    def test_corporate_tax_sme_and_standard_rates(self):
        results = self._verify(
            "fr_corporate_tax_is",
            "FrIs2026",
            [
                {"name": "IsSme", "facts": {"taxable_profit": 40000, "is_sme_eligible": True}},
                {
                    "name": "IsAboveThreshold",
                    "facts": {"taxable_profit": 60000, "is_sme_eligible": True},
                },
                {
                    "name": "IsNotEligible",
                    "facts": {"taxable_profit": 40000, "is_sme_eligible": False},
                },
            ],
        )
        # 40,000 <= 42,500 SME threshold and eligible: 15% -> 6,000.
        self._assert_values(
            results["Issme"],
            {
                "taxable_profit_eur": 40000,
                "corporate_tax_eur": 6000,
                "applicable_rate_percent": 15,
            },
        )
        # Eligible SME with EUR 60,000 profit: 15% on the first EUR 42,500
        # (6,375) plus 25% on the remaining EUR 17,500 (4,375) = 10,750.
        self._assert_values(
            results["Isabovethreshold"],
            {"corporate_tax_eur": 10750, "applicable_rate_percent": 25},
        )
        # Below threshold but ineligible: standard 25% -> 10,000.
        self._assert_values(
            results["Isnoteligible"],
            {"corporate_tax_eur": 10000, "applicable_rate_percent": 25},
        )

    # ── Personal income tax (IRPP 2026 progressive scale) ───────────────────

    def test_income_tax_progressive_scale_and_quotient_bareme_only(self):
        results = self._verify(
            "fr_direct_tax_irpp",
            "FrIrpp2026",
            [
                {
                    "name": "IrppSingle",
                    "facts": {"taxable_income": 20000, "quotient_familial_parts": 1},
                },
                {
                    "name": "IrppCouple",
                    "facts": {"taxable_income": 100000, "quotient_familial_parts": 2},
                },
                {
                    # Worked example from the fixture circular itself.
                    "name": "IrppCircularExample",
                    "facts": {"taxable_income": 30000, "quotient_familial_parts": 1},
                },
            ],
        )
        # (20,000 - 11,600) * 11% = 924.
        self._assert_values(
            results["Irppsingle"],
            {"taxable_income_eur": 20000, "income_tax_eur": 924},
        )
        # Per part 50,000: 17,979*11% + 20,421*30% = 8,103.99; x2 parts.
        self._assert_values(results["Irppcouple"], {"income_tax_eur": 16207.98})
        # Circular example: 17,979*11% + 421*30% = 2,103.99.
        self._assert_values(results["Irppcircularexample"], {"income_tax_eur": 2103.99})
        # Catala money is major units; the minor-unit view is x100.
        self.assertAlmostEqual(
            results["Irppcouple"]["income_tax_cents"] * 100, 1620798, places=0
        )

    # ── Social contributions (payroll 2026) ─────────────────────────────────

    def test_payroll_below_and_above_pmss_ceiling(self):
        results = self._verify(
            "fr_social_contributions",
            "FrSocial2026",
            [
                {
                    "name": "SocialBelowCeiling",
                    "facts": {"gross_salary": 30000, "rgdu_relief": 0},
                },
                {
                    "name": "SocialAboveCeiling",
                    "facts": {"gross_salary": 60000, "rgdu_relief": 0},
                },
                {
                    # The RGDU amount is an explicit input; this scenario proves
                    # the relief is subtracted, without inventing a coefficient.
                    "name": "SocialWithRgdu",
                    "facts": {"gross_salary": 60000, "rgdu_relief": 1000},
                },
                {
                    # Above 4x PASS: the 98.25% CSG/CRDS abatement stops and
                    # the excess is included at 100%.
                    "name": "SocialHighAboveCsgCap",
                    "facts": {"gross_salary": 200000, "rgdu_relief": 0},
                },
            ],
        )
        # 2026 LFSS full rates (13% health, 5.25% family) before RGDU relief.
        self._assert_values(
            results["Socialbelowceiling"],
            {
                "gross_salary_eur": 30000,
                "employer_contributions_before_rgdu_eur": 10068,
                "rgdu_relief_eur": 0,
                "employer_contributions_eur": 10068,
                "employee_contributions_eur": 5049.08,
                "net_salary_eur": 24950.92,
                "total_employer_cost_eur": 40068,
            },
        )
        # Above PMSS: 8.55% / 6.90% apply to MIN(gross, 48,060), not zero.
        self._assert_values(
            results["Socialaboveceiling"],
            {
                "employer_contributions_before_rgdu_eur": 19115.13,
                "employee_contributions_eur": 9274.29,
                "net_salary_eur": 50725.71,
                "total_employer_cost_eur": 79115.13,
            },
        )
        self._assert_values(
            results["Socialwithrgdu"],
            {
                "employer_contributions_before_rgdu_eur": 19115.13,
                "rgdu_relief_eur": 1000,
                "employer_contributions_eur": 18115.13,
                "total_employer_cost_eur": 78115.13,
            },
        )
        # EUR 200,000: CSG/CRDS base = 192,240 x 98.25% + 7,760 at 100%.
        # Unemployment (4%) and AGS (0.25%) also stop at 4x PASS.
        self._assert_values(
            results["Socialhighabovecsgcap"],
            {
                "employer_contributions_before_rgdu_eur": 53799.33,
                "employer_contributions_eur": 53799.33,
                "employee_contributions_eur": 23189.81,
                "net_salary_eur": 176810.19,
                "total_employer_cost_eur": 253799.33,
            },
        )

    # ── Property tax (taxe fonciere: municipal rate is an input) ────────────

    def test_property_tax_uses_municipal_rate_input(self):
        results = self._verify(
            "fr_property_tax",
            "FrProperty2026",
            [
                {
                    "name": "Property20pct",
                    "facts": {"cadastral_value": 20000, "municipal_rate_bps": 2000},
                },
                {
                    "name": "Property30pct",
                    "facts": {"cadastral_value": 20000, "municipal_rate_bps": 3000},
                },
            ],
        )
        # Built property: 50% abatement, then the municipal rate.
        self._assert_values(
            results["Property20pct"],
            {
                "cadastral_value_eur": 20000,
                "taxable_base_eur": 10000,
                "property_tax_eur": 2000,
                "municipal_rate_percent": 20,
            },
        )
        self._assert_values(
            results["Property30pct"],
            {"taxable_base_eur": 10000, "property_tax_eur": 3000},
        )

    def test_property_tax_fails_closed_without_municipal_rate(self):
        # No national rate exists; omitting the municipal rate must fail rather
        # than silently substituting a range from the circular.
        with self.assertRaises(CatalaLoweringError):
            self._verify(
                "fr_property_tax",
                "FrProperty2026",
                [{"name": "MissingRate", "facts": {"cadastral_value": 20000}}],
            )


class TestGoldenCatalaArtifacts(unittest.TestCase):
    """The committed .catala_en files must be exactly what the emitter produces."""

    def test_golden_files_are_current(self):
        proc = subprocess.run(
            [sys.executable, str(GOLDEN_SCRIPT), "--check"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(
            proc.returncode,
            0,
            "committed Catala artifacts are stale; run "
            f"`python {GOLDEN_SCRIPT.relative_to(REPO_ROOT)}`:\n"
            f"{proc.stderr or proc.stdout}",
        )


class TestGoldenCatalaTypechecks(unittest.TestCase):
    """Every committed .catala_en artifact must typecheck with the official compiler."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("node") is None:
            message = "node is not on PATH"
        elif not INTERPRETER.exists():
            message = "missing interpreter (run: node scripts/fetch_catala_interpreter.mjs)"
        else:
            message = None
        if message:
            if os.environ.get("REQUIRE_CATALA") == "1":
                raise AssertionError(f"Catala verification unavailable: {message}")
            raise unittest.SkipTest(f"Catala verification unavailable: {message}")

    def test_golden_files_typecheck(self):
        artifacts = sorted(FIXTURE_DIR.glob("*/rule.catala_en"))
        self.assertEqual(len(artifacts), 5, f"expected five artifacts, found {artifacts}")
        for path in artifacts:
            source = path.read_text(encoding="utf-8")
            request = {"id": path.parent.name, "source": source, "scopes": []}
            proc = subprocess.run(
                ["node", str(HARNESS)],
                input=json.dumps(request) + "\n",
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(REPO_ROOT),
            )
            self.assertEqual(proc.returncode, 0, f"{path.name}: harness failed: {proc.stderr}")
            response = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertTrue(
                response["typecheck"]["success"],
                f"{path.parent.name}/{path.name}: Catala typecheck failed:\n"
                f"{response['typecheck']['errors']}",
            )


if __name__ == "__main__":
    unittest.main()
