"""
RFC-012 Step 1 acceptance gates (A1.1 - A1.7).

Step 1 makes one neural-decompiled clause lower end-to-end: the IRC
Section 1(a)(ii) taxable-income bracket is synthesized by the dual-stream
decompiler, lowered to Catala, and executed in both engines.

The gates below are the ones written down in
``docs/rfcs/RFC-012-lowerer-composition.md`` section 3.7:

* A1.1  the v2 AST contains no null expression;
* A1.2  the emitted Catala typechecks with the official toolchain;
* A1.3  minor-unit constants are not scaled by 100;
* A1.4  USVM and Catala agree on every scenario;
* A1.5  both engines reproduce the statutory values exactly;
* A1.6  the AST declares exactly one input and probes no aliases;
* A1.7  the legacy profile, the French golden artifacts, the 19 French
        cross-engine scenarios and the CLI quickstart stay green.

Money is integer minor units everywhere: the USVM reports ``taxAmountPaisa``
in paise and Catala reports ``tax_amount_paisa`` in major units, so the two
are compared after multiplying Catala by 100.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.converters.catala_emitter import emit_catala_code
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

SECTION_1_TEXT = (
    "If taxable income is over $36,900 but not over $89,150, tax shall be "
    "$5,535, plus 28% of the excess over $36,900."
)

# (id, taxable income in minor units, expected tax in minor units)
SCENARIOS = (
    ("S1", 3000000, 0),
    ("S2", 5000000, 920300),
    ("S3", 8915000, 2016500),
    ("S4", 3690100, 553528),
    ("S5", 9000000, 0),
)

HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"
GOLDEN_SCRIPT = REPO_ROOT / "scripts" / "emit_catala_fixtures.py"
CHECKPOINT = REPO_ROOT / "checkpoints" / "dual_stream_sup.pt"

QUICKSTART_TEXT = (
    "Any person paying to a resident sum for purchase of goods exceeding "
    "50 Lakh rupees shall deduct 0.1 percent."
)


def _walk(node):
    """Yields every (key, value) pair of a nested JSON structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield None, value
            yield from _walk(value)


class TestRFC012Step1(unittest.TestCase):
    """Step 1 gates for the IRC Section 1(a)(ii) bracket."""

    @classmethod
    def setUpClass(cls):
        problems = []
        if shutil.which("node") is None:
            problems.append("node is not on PATH")
        if not HARNESS.exists():
            problems.append(f"missing harness {HARNESS}")
        if not INTERPRETER.exists():
            problems.append("missing interpreter (run: node scripts/fetch_catala_interpreter.mjs)")
        if not CHECKPOINT.exists():
            problems.append(f"missing checkpoint {CHECKPOINT}")
        if problems:
            message = "RFC-012 Step 1 verification unavailable: " + "; ".join(problems)
            if os.environ.get("REQUIRE_CATALA") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

        cls.decompiler = DualStreamDecompilerInference(str(CHECKPOINT))
        cls.result = cls.decompiler.decompile_text(SECTION_1_TEXT, typed_inputs=True)
        cls.ast = cls.result["synthesizedAst"]
        cls.code = emit_catala_code(
            cls.ast,
            title="IRC Section 1(a)(ii) taxable income bracket",
            scenarios=[
                {"name": name, "facts": {"taxableIncome": income}}
                for name, income, _ in SCENARIOS
            ],
        )
        cls.evaluator = CanonicalUsvmEvaluator()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _catala_response(self) -> dict:
        request = {
            "id": "irc-1-a-ii",
            "source": self.code,
            "scopes": [name for name, _, _ in SCENARIOS],
        }
        proc = subprocess.run(
            ["node", str(HARNESS)],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, f"harness failed: {proc.stderr}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def _usvm(self, income: int) -> dict:
        return self.evaluator.evaluate(self.ast, {"taxableIncome": income})

    # ── A1.1 ────────────────────────────────────────────────────────────────

    def test_a1_1_v2_ast_has_no_null_expression(self):
        computation = self.ast["computation"]
        for key, value in _walk(computation):
            self.assertIsNotNone(
                value,
                f"null expression under computation at {key!r}: {json.dumps(computation)[:400]}",
            )
        # The presence probe that broke Catala lowering is gone entirely.
        self.assertNotIn("null", json.dumps(computation))

    # ── A1.2 ────────────────────────────────────────────────────────────────

    def test_a1_2_emitted_catala_typechecks(self):
        response = self._catala_response()
        self.assertTrue(
            response["typecheck"]["success"],
            f"Catala typecheck failed:\n{response['typecheck']['errors']}",
        )
        for name, _, _ in SCENARIOS:
            run = response["results"][name]
            self.assertTrue(run["success"], f"{name}: Catala interpret failed:\n{run['errors']}")

    # ── A1.3 ────────────────────────────────────────────────────────────────

    def test_a1_3_minor_units_are_not_scaled_by_100(self):
        # $36,900 and $5,535 are minor-unit constants: 3,690,000 and 553,500
        # paise. The legacy rendering multiplied them by 100.
        self.assertIn("$36900.00", self.code)
        self.assertIn("$5535.00", self.code)
        self.assertNotIn("$3690000.00", self.code)
        self.assertNotIn("$553500.00", self.code)

    # ── A1.4 / A1.5 ────────────────────────────────────────────────────────

    def test_a1_4_and_a1_5_engines_agree_and_match_expected_values(self):
        response = self._catala_response()
        for name, income, expected in SCENARIOS:
            run = response["results"][name]
            self.assertTrue(run["success"], f"{name}: Catala interpret failed:\n{run['errors']}")
            values = run["values"]
            self.assertIn("tax_amount_paisa", values, f"{name}: missing Catala output")
            catala_minor = round(float(values["tax_amount_paisa"]) * 100)
            usvm = self._usvm(income)

            self.assertIn("taxAmountPaisa", usvm, f"{name}: USVM missing output")
            self.assertEqual(
                usvm["taxAmountPaisa"],
                catala_minor,
                f"{name}: USVM={usvm['taxAmountPaisa']!r} != Catala={catala_minor!r} (minor units)",
            )
            self.assertEqual(
                catala_minor,
                expected,
                f"{name}: expected {expected} minor units, got {catala_minor}",
            )
            # The major-unit view must agree too.
            self.assertAlmostEqual(
                float(usvm["calculatedTaxUsd"]),
                float(values["calculated_tax_usd"]),
                places=2,
                msg=f"{name}: calculatedTaxUsd disagrees between engines",
            )

    # ── A1.6 ────────────────────────────────────────────────────────────────

    def test_a1_6_single_declared_input_without_alias_probing(self):
        inputs = self.ast.get("inputs")
        self.assertIsInstance(inputs, dict, "v2 AST must declare an inputs signature")
        self.assertEqual(list(inputs), ["taxableIncome"])
        self.assertEqual(inputs["taxableIncome"]["type"], "MONEY_MINOR")
        self.assertEqual(inputs["taxableIncome"]["currency"], "USD")
        self.assertTrue(inputs["taxableIncome"]["required"])

        blob = json.dumps(self.ast["computation"])
        self.assertNotIn("facts.", blob, "a v2 AST must not reference the legacy fact namespace")
        self.assertNotIn("primaryBasePaisa", blob, "alias probing must not survive")
        self.assertNotIn("taxableIncomeUsd", blob, "candidate key lists must not survive")

        # No step may carry a candidate list of fact keys.
        for step in self.ast["computation"]["pipeline"]["steps"]:
            self.assertNotIsInstance(step["expr"], list)

    def test_a1_6_v2_ast_never_reaches_the_legacy_fact_registry(self):
        """RFC-012 3.6.5: the legacy registry is authoritative for legacy ASTs only."""
        import compiler.converters.catala_emitter as emitter_module

        class _Tripwire(dict):
            def __contains__(self, key):
                raise AssertionError(f"legacy fact registry consulted for {key!r}")

            def __getitem__(self, key):
                raise AssertionError(f"legacy fact registry consulted for {key!r}")

        original = emitter_module._FACT_BINDINGS
        emitter_module._FACT_BINDINGS = _Tripwire()
        try:
            code = emit_catala_code(self.ast)
        finally:
            emitter_module._FACT_BINDINGS = original
        self.assertIn("data taxable_income content money", code)

    # ── A1.7 ────────────────────────────────────────────────────────────────

    def test_a1_7_legacy_profile_is_unchanged(self):
        legacy = self.decompiler.decompile_text(SECTION_1_TEXT)["synthesizedAst"]
        self.assertNotIn("inputs", legacy, "the legacy profile must not gain an inputs signature")
        legacy_blob = json.dumps(legacy["computation"])
        self.assertIn("facts.primaryBasePaisa", legacy_blob)
        self.assertIn("null", legacy_blob)

    def test_a1_7_french_golden_artifacts_are_current(self):
        proc = subprocess.run(
            [sys.executable, str(GOLDEN_SCRIPT), "--check"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(
            proc.returncode,
            0,
            "committed Catala artifacts are stale; run "
            f"`python {GOLDEN_SCRIPT.relative_to(REPO_ROOT)}`:\n{proc.stderr or proc.stdout}",
        )

    def test_a1_7_french_cross_engine_scenarios_remain_green(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_french_tax_cross_engine.py",
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(
            proc.returncode,
            0,
            "French cross-engine scenarios regressed:\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}",
        )

    def test_a1_7_cli_quickstart_reports_pipeline_steps(self):
        proc = subprocess.run(
            [sys.executable, "cli.py", QUICKSTART_TEXT],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, f"CLI quickstart failed: {proc.stderr[-2000:]}")
        match = re.search(r"Steps:\s+(\d+) pipeline operations", proc.stdout)
        self.assertIsNotNone(match, f"CLI did not report pipeline steps:\n{proc.stdout[-2000:]}")
        self.assertGreater(
            int(match.group(1)),
            0,
            "CLI must report the synthesized pipeline steps, not an empty list",
        )


if __name__ == "__main__":
    unittest.main()
