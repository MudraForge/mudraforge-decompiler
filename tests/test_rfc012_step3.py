"""
RFC-012 Step 3 acceptance gates (A3.1 - A3.4).

Step 3 replaces the "first matching archetype wins" chain with one
topological walk over the resolved relation graph. The gates below are the
ones written down in ``docs/rfcs/RFC-012-lowerer-composition.md`` section 5.4:

* A3.1  bracket, subtraction and cap compose in one clause;
* A3.2  the result does not depend on the order of the edges;
* A3.3  a cyclic dependency fails closed;
* A3.4  one step list, more than one handler, no truncation.

Money is integer minor units everywhere: the USVM reports ``...Paisa`` in
paise and Catala reports ``..._paisa`` in major units, so the two are compared
after multiplying Catala by 100.
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
from compiler.neural.inference import synthesize_usvm_ast
from compiler.converters.catala_emitter import _identifier, emit_catala_code
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

# Bracket, reduction and cap in a single clause.
COMPOSITION_TEXT = (
    "Tax equals 28 percent of the excess of gross income over $36,900, "
    "reduced by a $500 credit, and capped at $10,000."
)

# (gross income in minor units, expected tax in minor units)
COMPOSITION_SCENARIOS = (
    ("G37000", 3700000, 0),
    ("G40000", 4000000, 36800),
    ("G50000", 5000000, 316800),
    ("G100000", 10000000, 1000000),
)

HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"
CHECKPOINT = REPO_ROOT / "checkpoints" / "dual_stream_sup.pt"


class TestRFC012Step3(unittest.TestCase):
    """Step 3 gates for the topological SSA walk."""

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
            message = "RFC-012 Step 3 verification unavailable: " + "; ".join(problems)
            if os.environ.get("REQUIRE_CATALA") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

        cls.decompiler = DualStreamDecompilerInference(str(CHECKPOINT))
        cls.evaluator = CanonicalUsvmEvaluator()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _decompile(self, text: str) -> dict:
        return self.decompiler.decompile_text(text, clause_scopes=True)["synthesizedAst"]

    def _catala(self, ast: dict, scenarios: list[dict]) -> dict:
        source = emit_catala_code(ast, title="Composed rule", scenarios=scenarios)
        request = {
            "id": "rfc012-step3",
            "source": source,
            "scopes": [s["name"] for s in scenarios],
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
        response = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(
            response["typecheck"]["success"],
            f"Catala typecheck failed:\n{response['typecheck']['errors']}",
        )
        return response

    # ── A3.1 ────────────────────────────────────────────────────────────────

    def test_a3_1_bracket_subtraction_and_cap_compose(self):
        ast = self._decompile(COMPOSITION_TEXT)
        outputs = ast["computation"]["pipeline"]["output"]
        self.assertIn("grossIncomeTaxPaisa", outputs)

        scenarios = [
            {"name": name, "facts": {"grossIncome": gross}}
            for name, gross, _ in COMPOSITION_SCENARIOS
        ]
        response = self._catala(ast, scenarios)
        catala_key = _identifier("grossIncomeTaxPaisa")
        for (name, gross, expected), scenario in zip(COMPOSITION_SCENARIOS, scenarios):
            run = response["results"][name]
            self.assertTrue(run["success"], f"{name}: Catala interpret failed:\n{run['errors']}")
            catala_minor = round(float(run["values"][catala_key]) * 100)
            usvm = self.evaluator.evaluate(ast, scenario["facts"])
            self.assertEqual(
                usvm["grossIncomeTaxPaisa"],
                catala_minor,
                f"{name}: USVM={usvm['grossIncomeTaxPaisa']!r} != Catala={catala_minor!r}",
            )
            self.assertEqual(
                catala_minor,
                expected,
                f"{name}: expected {expected} minor units, got {catala_minor}",
            )

    # ── A3.2 ────────────────────────────────────────────────────────────────

    def test_a3_2_result_does_not_depend_on_edge_order(self):
        result = self.decompiler.decompile_text(COMPOSITION_TEXT, clause_scopes=True)
        edges = list(result["relationalEdges"])
        baseline = result["synthesizedAst"]["computation"]

        reordered = synthesize_usvm_ast(
            text=COMPOSITION_TEXT,
            spans=result["classifiedSpans"],
            edges=list(reversed(edges)),
            clause_modalities=result["clauseModalities"],
            clause_scopes=True,
        )
        self.assertEqual(
            reordered["computation"]["pipeline"],
            baseline["pipeline"],
            "the pipeline must not depend on the order of the edges",
        )

    # ── A3.3 ────────────────────────────────────────────────────────────────

    def test_a3_3_cyclic_dependency_fails_closed(self):
        # A rate that depends on its own result cannot be walked. The neural
        # annotator cannot produce such a graph, so the gate is exercised at
        # the synthesis boundary with the edges it would produce.
        text = "A tax of 28 percent applies to income over $36,900, reduced by a $500 credit."
        spans = [
            {
                "spanIdx": 0,
                "clauseIdx": 0,
                "role": "ROLE_STANDARD_STATUTORY_RATE",
                "span": "28 percent",
                "charRange": [10, 19],
                "confidence": 0.9,
            },
            {
                "spanIdx": 1,
                "clauseIdx": 0,
                "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD",
                "span": "over $36,900",
                "charRange": [30, 42],
                "confidence": 0.9,
            },
            {
                "spanIdx": 2,
                "clauseIdx": 0,
                "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD",
                "span": "$500",
                "charRange": [57, 61],
                "confidence": 0.9,
            },
        ]
        edges = [
            # The rate depends on its own result.
            {"sourceIdx": 0, "targetIdx": 0, "relationType": "REL_RATE_TO_THRESHOLD"},
            {"sourceIdx": 2, "targetIdx": 0, "relationType": "REL_SUBTRACTION_BASE"},
        ]
        with self.assertRaises(Exception) as ctx:
            synthesize_usvm_ast(
                text=text,
                spans=spans,
                edges=edges,
                clause_modalities=[{"clauseIdx": 0, "charRange": [0, len(text)]}],
                clause_scopes=True,
            )
        self.assertIn("ERR_CYCLIC_DEPENDENCY", str(ctx.exception))

    # ── A3.4 ────────────────────────────────────────────────────────────────

    def test_a3_4_one_step_list_uses_more_than_one_handler(self):
        ast = self._decompile(COMPOSITION_TEXT)
        scopes = ast["computation"]["clauseScopes"]
        handlers = scopes[0]["handlers"]
        self.assertGreaterEqual(
            len(set(handlers)),
            3,
            f"the composed clause must use more than one handler, got {handlers}",
        )
        for expected in ("base-arithmetic", "rate-on-base", "reduction", "cap"):
            self.assertIn(expected, handlers)

        steps = ast["computation"]["pipeline"]["steps"]
        self.assertEqual(
            len(steps),
            len(handlers),
            "every step of the composed clause must come from a handler",
        )
        # The law is not truncated: the cap and the credit are both present.
        blob = json.dumps(steps)
        self.assertIn("50000", blob, "the $500 credit must be applied")
        self.assertIn("1000000", blob, "the $10,000 cap must be applied")

    # ── Hardened Regression Guards ───────────────────────────────────────────

    def test_a3_5_excess_clause_without_threshold_fails_closed(self):
        """Excess-over-threshold clauses must NEVER silently degrade into flat rates."""
        text = "A tax of 3 percent of the excess of adjusted gross income over the applicable amount is imposed."
        spans = [
            {"span": "3 percent", "role": "ROLE_STANDARD_STATUTORY_RATE", "charRange": [9, 18], "clauseIdx": 0, "confidence": 0.99},
            {"span": "adjusted gross income", "role": "ROLE_COMPUTATION_OPERAND", "charRange": [37, 58], "clauseIdx": 0, "confidence": 0.99},
        ]
        edges = [{"sourceIdx": 0, "targetIdx": 1, "relationType": "REL_SUBTRACTION_BASE"}]
        # Rate node present, but no threshold node. Text contains 'excess of ... over'.
        with self.assertRaises(Exception) as ctx:
            synthesize_usvm_ast(
                text=text,
                spans=spans,
                edges=edges,
                clause_modalities=[{"clauseIdx": 0, "charRange": [0, len(text)]}],
                clause_scopes=True,
            )
        self.assertIn("ERR_NO_THRESHOLD", str(ctx.exception))

    def test_a3_6_emitted_inputs_never_leak_placeholder_currency(self):
        """Input schemas must never declare placeholder 'CURRENCY'; only valid ISO codes."""
        text = "An excise tax of 6.0 percent of total wages paid by an employer is imposed."
        ast = self._decompile(text)
        inputs = ast.get("inputs", {})
        self.assertTrue(len(inputs) > 0, "at least one input must be declared")
        for in_id, in_def in inputs.items():
            if in_def.get("type") == "MONEY_MINOR":
                ccy = in_def.get("currency")
                self.assertIn(ccy, ("USD", "INR", "EUR", "GBP"), f"input {in_id} has invalid currency {ccy!r}")
                self.assertNotEqual(ccy, "CURRENCY", f"input {in_id} leaked internal placeholder 'CURRENCY'")

    def test_a3_7_non_monetary_number_never_becomes_paise_threshold(self):
        """Bare non-monetary numbers ('1 taxable year', 'section 1', 'one') must never become TO_PAISE('100')."""
        from compiler.neural.clause_linker import is_monetary_quantum_text, parse_monetary_minor_units
        from compiler.neural.ssa_walker import SsaNode, _static_money

        for non_money in ("1", "one", "1 taxable year", "section 1", "(2)", "35,928.50"):
            self.assertFalse(is_monetary_quantum_text(non_money), f"{non_money!r} must not be MONETARY_QUANTUM")
            self.assertIsNone(parse_monetary_minor_units(non_money), f"{non_money!r} must not parse to minor units")
            node = SsaNode(index=0, role="ROLE_TRANSACTION_CLIFF_THRESHOLD", text=non_money, char_range=(0, len(non_money)))
            self.assertIsNone(_static_money(node), f"{non_money!r} must not emit TO_PAISE")

        for real_money, expected_minor in (("$1,000", 100000), ("fifty lakh rupees", 500000000), ("over $36,900", 3690000)):
            self.assertTrue(is_monetary_quantum_text(real_money), f"{real_money!r} must be MONETARY_QUANTUM")
            self.assertEqual(parse_monetary_minor_units(real_money), expected_minor)
            node = SsaNode(index=0, role="ROLE_TRANSACTION_EXCESS_THRESHOLD", text=real_money, char_range=(0, len(real_money)))
            self.assertEqual(_static_money(node), {"op": "TO_PAISE", "value": str(expected_minor)})

    def test_a3_8_boolean_kinship_status_clause_never_emits_fake_money_threshold(self):
        """Boolean marriage/kinship status clauses (e.g. IRC Section 2) must fail closed, never emitting $1.00 thresholds."""
        text = (
            "(b) Definition of head of household. "
            "An individual shall be considered a head of a household if, and only if, "
            "such individual is not married at the close of his taxable year, and "
            "maintains as his home a household for more than one-half of such taxable year."
        )
        spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_REGULATED_OBJECT_OR_STATUS_SCOPE", "span": "household", "charRange": [26, 35], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 1, "role": "ROLE_PROHIBITION_GATE", "span": "not", "charRange": [129, 132], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 1, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "span": "one", "charRange": [228, 231], "confidence": 0.99},
        ]
        with self.assertRaises(Exception) as ctx:
            synthesize_usvm_ast(
                text=text,
                spans=spans,
                edges=[],
                clause_modalities=[
                    {"clauseIdx": 0, "charRange": [0, 36]},
                    {"clauseIdx": 1, "charRange": [37, len(text)]},
                ],
                clause_scopes=True,
            )
        self.assertIn("ERR_NO_COMPUTATION", str(ctx.exception))

    def test_a3_9_bounded_bracket_enforces_ceiling_and_base_tax(self):
        """REL_BRACKET_NEXT must cap the slice at the ceiling, add baseTax, and gate inside (floor, ceiling]."""
        header = "§1. Tax imposed on taxable income. "
        bracket = "$5,535, plus 28% of the excess over $36,900 if the taxable income is over $36,900 but not over $89,150."
        text = header + bracket
        offset = len(header)
        spans = [
            {"spanIdx": 0, "clauseIdx": 0, "role": "ROLE_COMPUTATION_OPERAND", "span": "taxable income", "charRange": [19, 33], "confidence": 0.99},
            {"spanIdx": 1, "clauseIdx": 1, "role": "ROLE_STANDARD_STATUTORY_RATE", "span": "28%", "charRange": [offset + 13, offset + 16], "confidence": 0.99},
            {"spanIdx": 2, "clauseIdx": 1, "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "span": "over $36,900", "charRange": [offset + 31, offset + 43], "confidence": 0.99},
            {"spanIdx": 3, "clauseIdx": 1, "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD", "span": "not over $89,150", "charRange": [offset + 86, offset + 102], "confidence": 0.99},
        ]
        edges = [
            {"sourceIdx": 1, "targetIdx": 2, "relationType": "REL_RATE_TO_THRESHOLD"},
            {"sourceIdx": 2, "targetIdx": 3, "relationType": "REL_BRACKET_NEXT"},
        ]
        ast = synthesize_usvm_ast(
            text=text,
            spans=spans,
            edges=edges,
            clause_modalities=[
                {"clauseIdx": 0, "charRange": [0, offset - 1]},
                {"clauseIdx": 1, "charRange": [offset, len(text)]},
            ],
            clause_scopes=True,
        )
        steps = ast["computation"]["pipeline"]["steps"]
        handlers = ast["computation"]["clauseScopes"][1]["handlers"]
        self.assertIn("bracket-base-tax", handlers)
        self.assertIn("bracket-next", handlers)
        blob = json.dumps(steps)
        self.assertIn("553500", blob, "baseTax $5,535 (553500 minor units) must be emitted")
        self.assertIn("8915000", blob, "ceiling $89,150 (8915000 minor units) must be emitted")

        # Verify Catala == USVM across below-floor ($20k -> $0), in-bracket ($50k -> $9,203), and above-ceiling ($100k -> $0)
        bracket_scenarios = [
            ("G20000", {"taxableIncome": 2000000}, 0),
            ("G50000", {"taxableIncome": 5000000}, 920300),
            ("G100000", {"taxableIncome": 10000000}, 0),
        ]
        response = self._catala(
            ast,
            [{"name": name, "facts": facts} for name, facts, _ in bracket_scenarios],
        )
        catala_key = _identifier("taxableIncomeTaxPaisa")
        for name, facts, expected_minor in bracket_scenarios:
            run = response["results"][name]
            self.assertTrue(run["success"], f"{name}: Catala interpret failed:\n{run['errors']}")
            catala_minor = round(float(run["values"][catala_key]) * 100)
            usvm = self.evaluator.evaluate(ast, facts)
            self.assertEqual(usvm["taxableIncomeTaxPaisa"], expected_minor)
            self.assertEqual(catala_minor, expected_minor)


if __name__ == "__main__":
    unittest.main()
