"""
RFC-012 Step 2 acceptance gates (A2.1 - A2.7).

Step 2 gives the lowerer per-clause scopes and a def-use symbol linker. The
gates below are the ones written down in
``docs/rfcs/RFC-012-lowerer-composition.md`` section 4.7:

* A2.1  two clauses produce two computations; neither overwrites the other;
* A2.2  a value defined in one clause is used by another;
* A2.3  a statutory substitution is an SSA version bump (IRC Section 63);
* A2.4  an ambiguous reference fails closed;
* A2.5  a cyclic definition fails closed;
* A2.6  a single-clause statute is unchanged;
* A2.7  the result does not depend on the order of the spans.

Money is integer minor units everywhere: the USVM reports ``...Paisa`` in
paise and Catala reports ``..._paisa`` in major units, so the two are compared
after multiplying Catala by 100.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.neural.clause_linker import ClauseLinkingError
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.neural.inference import synthesize_usvm_ast
from compiler.converters.catala_emitter import _identifier, emit_catala_code
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

# Two independent clauses: 6 percent on wages, 2 percent on interest.
SECTION_A21 = (
    "A wages tax of 6 percent applies to the wages paid by an employer. "
    "An interest tax of 2 percent applies to the interest paid by a lender."
)

# Clause 0 defines the eligible amount; clause 1 taxes it.
SECTION_A22 = (
    "The eligible amount is the amount exceeding fifty lakh rupees. "
    "A person who pays such eligible amount shall deduct 0.1 percent of "
    "the eligible amount."
)

# IRC Section 63: a statutory substitution of a standard deduction.
SECTION_A23 = (
    "The basic standard deduction is $4,400. "
    "For taxable years beginning after December 31, 2017, and before "
    "January 1, 2026, $18,000 shall be substituted for $4,400."
)

# Two definitions normalize to the same term, and a later clause uses it.
SECTION_A24 = (
    "The eligible amount is fifty lakh rupees. "
    "The eligible amount is sixty lakh rupees. "
    "A person shall deduct 0.1 percent of the eligible amount."
)

# A single-clause statute (Section 194Q).
SECTION_A26 = (
    "Any person paying to a resident sum for purchase of goods exceeding "
    "50 Lakh rupees shall deduct 0.1 percent."
)

HARNESS = REPO_ROOT / "scripts" / "catala_verify.mjs"
INTERPRETER = REPO_ROOT / "scripts" / "catala_interpreter.cjs"
CHECKPOINT = REPO_ROOT / "checkpoints" / "dual_stream_sup.pt"


class TestRFC012Step2(unittest.TestCase):
    """Step 2 gates for clause scopes and def-use linking."""

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
            message = "RFC-012 Step 2 verification unavailable: " + "; ".join(problems)
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
        source = emit_catala_code(ast, title="Clause scoped rule", scenarios=scenarios)
        request = {
            "id": "rfc012-step2",
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

    def _assert_engines_agree(self, ast: dict, scenarios: list[dict], output_key: str) -> None:
        response = self._catala(ast, scenarios)
        catala_key = _identifier(output_key)
        for scenario in scenarios:
            name = scenario["name"]
            run = response["results"][name]
            self.assertTrue(run["success"], f"{name}: Catala interpret failed:\n{run['errors']}")
            catala_minor = round(float(run["values"][catala_key]) * 100)
            usvm = self.evaluator.evaluate(ast, scenario["facts"])
            self.assertIn(output_key, usvm, f"{name}: USVM missing output {output_key!r}")
            self.assertEqual(
                usvm[output_key],
                catala_minor,
                f"{name}: USVM={usvm[output_key]!r} != Catala={catala_minor!r} (minor units)",
            )

    # ── A2.1 ────────────────────────────────────────────────────────────────

    def test_a2_1_two_clauses_produce_two_computations(self):
        ast = self._decompile(SECTION_A21)
        scopes = ast["computation"]["clauseScopes"]
        self.assertEqual(len(scopes), 2)
        self.assertEqual([s["standardRateBps"] for s in scopes], [600, 200])

        outputs = ast["computation"]["pipeline"]["output"]
        self.assertIn("wagesTaxPaisa", outputs)
        self.assertIn("interestTaxPaisa", outputs)
        # Neither clause overwrites the other.
        self.assertNotEqual(outputs["wagesTaxPaisa"], outputs["interestTaxPaisa"])

        self._assert_engines_agree(
            ast,
            [
                {"name": "Wages", "facts": {"wages": 1000000, "interest": 2000000}},
                {"name": "Interest", "facts": {"wages": 5000000, "interest": 8000000}},
            ],
            "wagesTaxPaisa",
        )
        self._assert_engines_agree(
            ast,
            [{"name": "Interest", "facts": {"wages": 1000000, "interest": 2000000}}],
            "interestTaxPaisa",
        )

    # ── A2.2 ────────────────────────────────────────────────────────────────

    def test_a2_2_cross_clause_definition_is_linked(self):
        ast = self._decompile(SECTION_A22)
        blob = json.dumps(ast["computation"])
        self.assertNotIn("facts.", blob, "a clause-scoped AST must not use the legacy namespace")
        self.assertNotIn("facts.eligibleAmount", blob)

        steps = ast["computation"]["pipeline"]["steps"]
        referencing = [
            step for step in steps
            if "vars.clause_0_" in json.dumps(step["expr"])
        ]
        self.assertTrue(referencing, "clause 1 must reference the clause 0 output")
        self.assertIn("clause_0_eligibleAmountPaisa_v1", json.dumps(referencing))

        self._assert_engines_agree(
            ast,
            [{"name": "Eligible", "facts": {"amount": 600000000}}],
            "eligibleAmountTaxPaisa",
        )

    # ── A2.3 ────────────────────────────────────────────────────────────────

    def test_a2_3_substitution_is_an_ssa_version_bump(self):
        ast = self._decompile(SECTION_A23)
        steps = {step["assign"]: step["expr"] for step in ast["computation"]["pipeline"]["steps"]}

        self.assertIn("clause_0_standardDeductionPaisa_v1", steps)
        self.assertIn("clause_1_inWindow_v1", steps)
        self.assertIn("clause_1_standardDeductionPaisa_v2", steps)
        # The override is a conditional over the previous version.
        self.assertEqual(
            steps["clause_1_standardDeductionPaisa_v2"]["else"],
            "vars.clause_0_standardDeductionPaisa_v1",
        )
        # The exported value is the highest version.
        self.assertEqual(
            ast["computation"]["pipeline"]["output"]["standardDeductionPaisa"],
            "vars.clause_1_standardDeductionPaisa_v2",
        )

        scenarios = [
            {"name": f"Year{y}", "facts": {"taxYear": y}}
            for y in (2017, 2018, 2025, 2026)
        ]
        expected = {2017: 440000, 2018: 1800000, 2025: 1800000, 2026: 440000}
        response = self._catala(ast, scenarios)
        for scenario in scenarios:
            year = scenario["facts"]["taxYear"]
            run = response["results"][scenario["name"]]
            self.assertTrue(run["success"], f"{year}: Catala failed:\n{run['errors']}")
            catala_minor = round(float(run["values"]["standard_deduction_paisa"]) * 100)
            usvm = self.evaluator.evaluate(ast, scenario["facts"])
            self.assertEqual(
                usvm["standardDeductionPaisa"],
                catala_minor,
                f"{year}: USVM != Catala",
            )
            self.assertEqual(
                catala_minor,
                expected[year],
                f"{year}: expected {expected[year]} minor units, got {catala_minor}",
            )

    # ── A2.4 ────────────────────────────────────────────────────────────────

    def test_a2_4_ambiguous_reference_fails_closed(self):
        with self.assertRaises(ClauseLinkingError) as ctx:
            self._decompile(SECTION_A24)
        self.assertEqual(ctx.exception.code, "ERR_AMBIGUOUS_REFERENCE")

    # ── A2.5 ────────────────────────────────────────────────────────────────

    def test_a2_5_cyclic_definition_fails_closed(self):
        # The neural annotator abstains on self-referential text, so the gate
        # is exercised at the linker boundary with the spans it would produce.
        text = "The first amount is the second amount. The second amount is the first amount."
        spans = [
            {
                "spanIdx": 0,
                "clauseIdx": 0,
                "role": "ROLE_COMPUTATION_OPERAND",
                "span": "the second amount",
                "charRange": [20, 36],
                "confidence": 0.9,
            },
            {
                "spanIdx": 1,
                "clauseIdx": 1,
                "role": "ROLE_COMPUTATION_OPERAND",
                "span": "the first amount",
                "charRange": [60, 75],
                "confidence": 0.9,
            },
        ]
        clause_modalities = [
            {"clauseIdx": 0, "charRange": [0, 38]},
            {"clauseIdx": 1, "charRange": [39, 77]},
        ]
        with self.assertRaises(ClauseLinkingError) as ctx:
            synthesize_usvm_ast(
                text=text,
                spans=spans,
                edges=[],
                clause_modalities=clause_modalities,
                clause_scopes=True,
            )
        self.assertEqual(ctx.exception.code, "ERR_CYCLIC_DEFINITION")

    # ── A2.6 ────────────────────────────────────────────────────────────────

    def test_a2_6_single_clause_statute_is_unchanged(self):
        legacy = self.decompiler.decompile_text(SECTION_A26)["synthesizedAst"]
        scoped = self.decompiler.decompile_text(SECTION_A26, clause_scopes=True)["synthesizedAst"]
        self.assertEqual(
            scoped["computation"],
            legacy["computation"],
            "a single-clause statute must keep the current computation",
        )

    # ── A2.7 ────────────────────────────────────────────────────────────────

    def test_a2_7_result_does_not_depend_on_span_order(self):
        result = self.decompiler.decompile_text(SECTION_A23, clause_scopes=True)
        spans = result["classifiedSpans"]
        edges = result["relationalEdges"]
        baseline = result["synthesizedAst"]["computation"]

        shuffled = list(spans)
        random.Random(20261114).shuffle(shuffled)
        reordered = synthesize_usvm_ast(
            text=SECTION_A23,
            spans=shuffled,
            edges=edges,
            clause_modalities=result["clauseModalities"],
            clause_scopes=True,
        )
        self.assertEqual(
            reordered["computation"]["pipeline"],
            baseline["pipeline"],
            "the pipeline must not depend on the order of the spans",
        )
        self.assertEqual(
            [scope["symbols"] for scope in reordered["computation"]["clauseScopes"]],
            [scope["symbols"] for scope in baseline["clauseScopes"]],
            "the symbol table must not depend on the order of the spans",
        )


if __name__ == "__main__":
    unittest.main()
