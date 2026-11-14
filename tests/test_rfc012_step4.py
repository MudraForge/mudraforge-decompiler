"""
RFC-012 Step 4 acceptance gates (A4.1 - A4.5).

Step 4 defines how the regimes of a multi-regime provision combine. The gates
below are the ones written down in
``docs/rfcs/RFC-012-lowerer-composition.md`` section 6.2:

* A4.1  BENEFICIAL_MIN picks the more favourable regime;
* A4.2  PENAL_MAX picks the heavier regime;
* A4.3  EXCLUSIVE_PREEMPT picks the highest declared priority;
* A4.4  INDEPENDENT_ACCUMULATE adds the regimes, each at most once;
* A4.5  the four combinators are order-independent and exact at the paise.

Every combinator lowers to the SSA primitives that already exist in both
engines, so no new USVM opcode is required and the Catala and USVM results
must agree to the paisa.
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

from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.neural.regime_combinators import (
    BENEFICIAL_MIN,
    ERR_DUPLICATE_REGIME,
    ERR_PREEMPTION_PRIORITY_MISSING,
    ERR_PREEMPTION_TIE,
    EXCLUSIVE_PREEMPT,
    INDEPENDENT_ACCUMULATE,
    PENAL_MAX,
    Regime,
    combine,
    rate_liability,
)
from compiler.converters.catala_emitter import _identifier, emit_catala_code
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

# A multi-regime provision the neural annotator really produces.
LESSER_TEXT = "A person shall pay the lesser of 10 percent of the amount or $5,000."
GREATER_TEXT = "A person shall pay the greater of $1,000 or 5 percent of the amount."

HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"
CHECKPOINT = REPO_ROOT / "checkpoints" / "dual_stream_sup.pt"


class TestRFC012Step4(unittest.TestCase):
    """Step 4 gates for the multi-regime combinators."""

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
            message = "RFC-012 Step 4 verification unavailable: " + "; ".join(problems)
            if os.environ.get("REQUIRE_CATALA") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

        cls.decompiler = DualStreamDecompilerInference(str(CHECKPOINT))
        cls.evaluator = CanonicalUsvmEvaluator()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    # ── A4.1 / A4.2 ─────────────────────────────────────────────────────────

    def test_a4_1_beneficial_min_picks_the_smaller_liability(self):
        result = combine(
            BENEFICIAL_MIN,
            [Regime(id="a", value=100000), Regime(id="b", value=80000)],
        )
        self.assertTrue(result.ok, [d.code for d in result.diagnostics])
        self.assertEqual(result.value, 80000)
        self.assertEqual(result.selected_regime, "b")
        self.assertEqual(result.expression["op"], "MIN")

    def test_a4_2_penal_max_picks_the_heavier_liability(self):
        result = combine(
            PENAL_MAX,
            [Regime(id="a", value=100000), Regime(id="b", value=180000)],
        )
        self.assertTrue(result.ok, [d.code for d in result.diagnostics])
        self.assertEqual(result.value, 180000)
        self.assertEqual(result.selected_regime, "b")
        self.assertEqual(result.expression["op"], "MAX")

    # ── A4.3 ────────────────────────────────────────────────────────────────

    def test_a4_3_exclusive_preempt_picks_the_highest_priority(self):
        base = 1000000
        general = Regime(
            id="general", value=rate_liability(base, 1000), priority=1,
            value_ref="vars.clause_0_generalTax_v1",
        )
        special = Regime(
            id="special", value=rate_liability(base, 600), priority=2,
            value_ref="vars.clause_1_specialTax_v1",
        )
        result = combine(EXCLUSIVE_PREEMPT, [general, special])
        self.assertTrue(result.ok, [d.code for d in result.diagnostics])
        # The special 6 percent liability wins over the general 10 percent.
        self.assertEqual(result.value, 60000)
        self.assertEqual(result.selected_regime, "special")
        self.assertEqual(result.priority_source, "declared")
        # The decision is auditable.
        self.assertEqual(result.audit()["selectedRegime"], "special")

    def test_a4_3_exclusive_preempt_fails_closed_on_a_tie(self):
        regimes = [
            Regime(id="a", value=100000, priority=1),
            Regime(id="b", value=90000, priority=1),
        ]
        result = combine(EXCLUSIVE_PREEMPT, regimes)
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, ERR_PREEMPTION_TIE)
        self.assertIsNone(result.expression)

    def test_a4_3_exclusive_preempt_fails_closed_without_a_priority(self):
        regimes = [
            Regime(id="a", value=100000, priority=1),
            Regime(id="b", value=90000),
        ]
        result = combine(EXCLUSIVE_PREEMPT, regimes)
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, ERR_PREEMPTION_PRIORITY_MISSING)

    def test_a4_3_notwithstanding_resolves_an_otherwise_tied_priority(self):
        regimes = [
            Regime(id="a", value=100000, priority=1),
            Regime(id="b", value=90000, priority=1, notwithstanding=True),
        ]
        result = combine(EXCLUSIVE_PREEMPT, regimes)
        self.assertTrue(result.ok, [d.code for d in result.diagnostics])
        self.assertEqual(result.selected_regime, "b")
        self.assertEqual(result.priority_source, "notwithstanding")

    # ── A4.4 ────────────────────────────────────────────────────────────────

    def test_a4_4_independent_accumulate_adds_every_regime(self):
        result = combine(
            INDEPENDENT_ACCUMULATE,
            [Regime(id="a", value=10000), Regime(id="b", value=25000)],
        )
        self.assertTrue(result.ok, [d.code for d in result.diagnostics])
        self.assertEqual(result.value, 35000)
        self.assertEqual(result.expression["op"], "ADD")

    def test_a4_4_a_regime_contributes_at_most_once(self):
        regimes = [
            Regime(id="a", value=10000),
            Regime(id="b", value=25000),
            Regime(id="a", value=10000),
        ]
        result = combine(INDEPENDENT_ACCUMULATE, regimes)
        self.assertTrue(result.ok, [d.code for d in result.diagnostics])
        # The same regime applied again contributes once: 35,000, not 45,000.
        self.assertEqual(result.value, 35000)
        self.assertEqual(result.selected, ["a", "b"])

    def test_a4_4_a_regime_with_two_values_fails_closed(self):
        regimes = [
            Regime(id="a", value=10000),
            Regime(id="a", value=25000),
        ]
        result = combine(INDEPENDENT_ACCUMULATE, regimes)
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, ERR_DUPLICATE_REGIME)

    # ── A4.5 ────────────────────────────────────────────────────────────────

    def test_a4_5_the_combinators_are_order_independent(self):
        cases = [
            (BENEFICIAL_MIN, [Regime(id="a", value=100000), Regime(id="b", value=80000)]),
            (PENAL_MAX, [Regime(id="a", value=100000), Regime(id="b", value=180000)]),
            (
                EXCLUSIVE_PREEMPT,
                [
                    Regime(id="a", value=100000, priority=1),
                    Regime(id="b", value=90000, priority=2),
                ],
            ),
            (INDEPENDENT_ACCUMULATE, [Regime(id="a", value=10000), Regime(id="b", value=25000)]),
        ]
        for combinator, regimes in cases:
            forward = combine(combinator, regimes)
            backward = combine(combinator, list(reversed(regimes)))
            self.assertTrue(forward.ok, combinator)
            self.assertTrue(backward.ok, combinator)
            self.assertEqual(forward.value, backward.value, combinator)
            self.assertEqual(forward.selected_regime, backward.selected_regime, combinator)
            self.assertEqual(forward.audit(), backward.audit(), combinator)

    def test_a4_5_the_rate_liability_is_exact_at_the_paise(self):
        # 10 percent of 3,333 paise is 333.3 paise: half-up gives 333.
        self.assertEqual(rate_liability(3333, 1000), 333)
        # 10 percent of 3,335 paise is 333.5 paise: half-up gives 334.
        self.assertEqual(rate_liability(3335, 1000), 334)
        self.assertEqual(rate_liability(1000000, 600), 60000)
        self.assertEqual(rate_liability(0, 1000), 0)

    # ── Cross-engine: the combinator lowers to existing primitives ──────────

    def _assert_cross_engine(self, text: str, facts: dict, amount: int) -> None:
        ast = self.decompiler.decompile_text(text, clause_scopes=True)["synthesizedAst"]
        output_key = next(iter(ast["computation"]["pipeline"]["output"]))
        source = emit_catala_code(ast, title="Multi-regime rule", scenarios=[{"name": "S", "facts": facts}])
        request = {"id": "step4", "source": source, "scopes": ["S"]}
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
            f"Catala typecheck failed:\n{response['typecheck']['errors']}",
        )
        run = response["results"]["S"]
        self.assertTrue(run["success"], f"Catala interpret failed:\n{run['errors']}")
        catala_minor = round(float(run["values"][_identifier(output_key)]) * 100)
        usvm = self.evaluator.evaluate(ast, facts)
        self.assertEqual(
            usvm[output_key],
            catala_minor,
            f"USVM={usvm[output_key]!r} != Catala={catala_minor!r}",
        )
        self.assertEqual(catala_minor, amount)

    def test_a4_5_the_beneficial_minimum_agrees_in_both_engines(self):
        # 10 percent of $100,000 is $10,000; the lesser of that and $5,000.
        self._assert_cross_engine(
            LESSER_TEXT, {"amount": 10000000}, 500000
        )

    def test_a4_5_the_penal_maximum_agrees_in_both_engines(self):
        # 5 percent of $100,000 is $5,000; the greater of that and $1,000.
        self._assert_cross_engine(
            GREATER_TEXT, {"amount": 10000000}, 500000
        )


if __name__ == "__main__":
    unittest.main()
