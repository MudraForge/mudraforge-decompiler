"""
End-to-end financial provision verification.

Runs genuine statutory text through the dual-stream neural decompiler, then
checks three independent layers against statutory ground truth:

1. The synthesized canonical AST (threshold in paise, rate in basis points).
2. Numeric execution by the canonical Node USVM (@mudraforge/statutory).
3. The emitted Catala source, compiled and interpreted by the official Catala
   toolchain when CATALA_INTERPRETER_JS points at the official
   catala_web_interpreter.js bundle (https://assets.catala-lang.org/catala_web_interpreter.js).
   Without the bundle, layer 3 is skipped; layers 1-2 always run.

Ground truth is derived from the statutes themselves, not from model output:
- IRC 3301-style FUTA: 6% of total wages.
- Section 194Q: 0.1% TDS on the aggregate purchase value exceeding Rs 50 lakh.
- GST Section 51: 2% TDS where contract value exceeds Rs 2.5 lakh (cliff).
"""

import json
import os
import re
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

FUTA_CLAUSE = (
    "Every employer shall pay for each calendar year an excise tax equal to 6 percent "
    "of the total wages paid by such employer during the calendar year."
)
S194Q_CLAUSE = (
    "Any person, being a buyer, who is responsible for paying any sum to any resident "
    "for purchase of goods of the value or aggregate of such value exceeding fifty lakh "
    "rupees in any previous year, shall deduct an amount equal to 0.1 per cent of such "
    "sum exceeding fifty lakh rupees as income-tax."
)
GST51_CLAUSE = (
    "The Government shall deduct tax at the rate of two per cent from the payment made "
    "or credited to the supplier of taxable goods or services or both, where the total "
    "value of such supply, under a contract, exceeds two lakh and fifty thousand rupees."
)

# (scenario name, USVM facts, expected taxable base in paise, expected tax in paise, applicable)
PROVISIONS = {
    "futa": {
        "clause": FUTA_CLAUSE,
        "threshold_paisa": 0,
        "rate_bps": 600,
        "cases": [
            ("Wages100k", {"invoiceAmount": 100000}, 10000000, 600000, True),
            ("Zerowages", {"invoiceAmount": 0}, 0, 0, False),
        ],
    },
    "s194q": {
        "clause": S194Q_CLAUSE,
        "threshold_paisa": 500000000,
        "rate_bps": 10,
        "cases": [
            # 49L prior + 3L current: only the 2L excess above the 50L threshold is taxed.
            ("Crossesthreshold", {"invoiceAmount": 300000, "priorPurchases": 4900000}, 20000000, 20000, True),
            ("Belowthreshold", {"invoiceAmount": 100000, "priorPurchases": 4000000}, 0, 0, False),
            # Prior already above threshold: the entire current invoice is excess.
            ("Allpriorabove", {"invoiceAmount": 200000, "priorPurchases": 6000000}, 20000000, 20000, True),
        ],
    },
    "gst51": {
        "clause": GST51_CLAUSE,
        "threshold_paisa": 25000000,
        "rate_bps": 200,
        "cases": [
            ("Contract3lakh", {"invoiceAmount": 300000}, 30000000, 600000, True),
            ("Smallcontract", {"invoiceAmount": 100000}, 0, 0, False),
            # "exceeds" is strict: exactly 2.5 lakh does not breach the cliff.
            ("Exactcliff", {"invoiceAmount": 250000}, 0, 0, False),
        ],
    },
}


def _snake(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    return s.lower()


class TestFinancialPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = REPO_ROOT / "checkpoints" / "dual_stream_sup.pt"
        if not cls.ckpt_path.exists():
            raise unittest.SkipTest(f"Checkpoint not found at {cls.ckpt_path}")
        cls.decompiler = DualStreamDecompilerInference(str(cls.ckpt_path))
        cls.evaluator = CanonicalUsvmEvaluator()
        cls.catala_js = os.environ.get("CATALA_INTERPRETER_JS")
        if cls.catala_js and not Path(cls.catala_js).exists():
            cls.catala_js = None

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "evaluator"):
            cls.evaluator.close()

    def _verify_provision(self, key):
        spec = PROVISIONS[key]
        res = self.decompiler.decompile_text(spec["clause"], rule_domain="STATUTORY_FINANCE")
        self.assertEqual(res["status"], "SUCCESS")
        ast = res["synthesizedAst"]
        comp = ast["computation"]
        self.assertEqual(comp["archetype"], "DECLARATIVE_PIPELINE")
        self.assertEqual(int(comp["thresholdPaisa"]), spec["threshold_paisa"])
        self.assertEqual(int(comp["standardRateBps"]), spec["rate_bps"])

        cases = spec["cases"]
        outputs = self.evaluator.evaluate_batch([
            {"ruleAst": ast, "facts": facts, "options": {"mode": "DEVELOPMENT"}}
            for _, facts, _, _, _ in cases
        ])
        self.assertEqual(len(outputs), len(cases))
        for (name, _, base_p, tax_p, applicable), out in zip(cases, outputs):
            self.assertFalse(out.get("error"), f"{key}/{name}: {out.get('error')}")
            self.assertEqual(int(out["taxableBasePaisa"]), base_p, f"{key}/{name} taxable base")
            self.assertEqual(int(out["taxAmountPaisa"]), tax_p, f"{key}/{name} tax")
            self.assertEqual(bool(out["isApplicable"]), applicable, f"{key}/{name} applicability")
            self.assertFalse(out["isAuthorizedToDisburse"], f"{key}/{name} disbursement invariant")

        scenarios = [
            {
                "name": name,
                "facts": {
                    "current_invoice": facts["invoiceAmount"],
                    "primary_base": facts["invoiceAmount"],
                    "prior_purchases": facts.get("priorPurchases", 0),
                },
            }
            for name, facts, _, _, _ in cases
        ]
        code = emit_catala_code(ast, scenarios=scenarios)

        if not self.catala_js:
            return
        harness = REPO_ROOT / "scripts" / "catala_verify.mjs"
        request = {"id": key, "source": code, "scopes": [name for name, _, _, _, _ in cases]}
        proc = subprocess.run(
            ["node", str(harness)],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, f"Catala harness failed: {proc.stderr}")
        response = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(
            response["typecheck"]["success"],
            f"{key}: Catala typecheck failed:\n{response['typecheck']['errors']}",
        )
        for (name, _, base_p, tax_p, applicable), out in zip(cases, outputs):
            run = response["results"][name]
            self.assertTrue(run["success"], f"{key}/{name}: Catala interpret failed:\n{run['errors']}")
            values = run["values"]
            for usvm_key, catala_value in out.items():
                ck = _snake(usvm_key)
                if ck not in values:
                    continue
                if usvm_key.endswith("Paisa"):
                    expected = catala_value  # Catala now emits paise directly; no /100 needed
                else:
                    expected = catala_value
                self.assertEqual(
                    values[ck], expected,
                    f"{key}/{name}: Catala {ck}={values[ck]} != USVM {usvm_key}={expected}",
                )
            self.assertEqual(values["tax_amount_paisa"], tax_p, f"{key}/{name} Catala ground truth")
            self.assertEqual(values["taxable_base_paisa"], base_p, f"{key}/{name} Catala base")
            self.assertEqual(values["is_applicable"], applicable, f"{key}/{name} Catala applicability")

    def test_futa_flat_rate(self):
        self._verify_provision("futa")

    def test_section_194q_excess_threshold(self):
        self._verify_provision("s194q")

    def test_gst_section_51_cliff(self):
        self._verify_provision("gst51")

    def test_compound_amount_span_stitching(self):
        res = self.decompiler.decompile_text(GST51_CLAUSE, rule_domain="STATUTORY_FINANCE")
        threshold_spans = [
            s["span"] for s in res["classifiedSpans"]
            if s["role"] == "ROLE_TRANSACTION_CLIFF_THRESHOLD"
        ]
        self.assertEqual(len(threshold_spans), 1)
        self.assertIn("two lakh and fifty thousand", threshold_spans[0])


if __name__ == "__main__":
    unittest.main()
