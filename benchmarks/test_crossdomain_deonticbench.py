import sys
import os

# Add root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compiler.neural.dual_stream_inference import DualStreamDecompilerInference

CKPT_DEFAULT = os.environ.get("MUDRA_CHECKPOINT") or os.path.join(REPO_ROOT, "checkpoints", "dual_stream_sup.pt")
ckpt = os.environ.get("MUDRA_CHECKPOINT", CKPT_DEFAULT)

print(f"Loading Dual-Stream Production Checkpoint: {ckpt}")
model = DualStreamDecompilerInference(ckpt)

benchmark_clauses = [
    # 1. Tax (SARA)
    {
        "domain": "U.S. Federal Tax (SARA § 3301)",
        "clause": "There is hereby imposed on every employer for each calendar year an excise tax equal to 6.0 percent of the total wages paid by him during the calendar year.",
        "expected_modality": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB"
    },
    {
        "domain": "U.S. Federal Tax (SARA § 1(a))",
        "clause": "If taxable income is not over $36,900, the tax is 15% of taxable income.",
        "expected_modality": "MODALITY_COMPUTATIONAL_CLIFF"
    },
    # 2. Airline Baggage
    {
        "domain": "Airline Policies (Baggage Dimensions)",
        "clause": "The total size of your carry-on, including handles and wheels, cannot exceed 22 x 14 x 9 inches and must fit in the overhead bin.",
        "expected_modality": "MODALITY_COMPUTATIONAL_CLIFF"
    },
    {
        "domain": "Airline Policies (Excess Fees)",
        "clause": "A passenger shall pay a fee of $100 for any checked bag exceeding 50 pounds but not exceeding 70 pounds.",
        "expected_modality": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB"
    },
    # 3. State Housing Law
    {
        "domain": "State Housing Law (Michigan § 554.615)",
        "clause": "Any affected individual may bring an action to enforce this act in a court of competent jurisdiction in the county where the defendant resides.",
        "expected_modality": "MODALITY_DEONTIC_PERMISSION"
    },
    {
        "domain": "State Housing Law (Eviction Notice § 600.5714)",
        "clause": "A person entitled to premises may recover possession by summary proceedings when a tenant holds over after seven days notice in writing.",
        "expected_modality": "MODALITY_DEONTIC_PERMISSION"
    },
    # 4. USCIS Immigration Administration
    {
        "domain": "USCIS Immigration (INA § 203(b)(2))",
        "clause": "To establish eligibility for a national interest waiver, a petitioner must first demonstrate qualification for the underlying EB-2 visa classification.",
        "expected_modality": "MODALITY_ABSTAIN_UNRESOLVED"
    },
    {
        "domain": "USCIS Immigration (Matter of Dhanasar)",
        "clause": "The petitioner must demonstrate that the foreign national's proposed endeavor has both substantial merit and national importance.",
        "expected_modality": "MODALITY_ABSTAIN_UNRESOLVED"
    }
]

def run_crossdomain_eval():
    print("\n" + "=" * 80)
    print("DEONTICBENCH MULTI-DOMAIN STATUTORY COMPILATION EVALUATION")
    print("=" * 80)

    results = []
    for item in benchmark_clauses:
        clause = item["clause"]
        domain = item["domain"]
        res = model.decompile_text(clause)
        
        pred_mod = res.get("predictedModality")
        spans = res.get("classifiedSpans", [])
        ast = res.get("synthesizedAst", {})
        status = res.get("status")
        conf = res.get("calibratedConfidence", 0.0)
        has_ast = bool(ast.get("computation") or ast.get("guards") or ast.get("rootCondition"))
        
        print(f"\nDomain:   {domain}")
        print(f"Clause:   {clause}")
        print(f"Status:   {status} (Confidence: {conf:.4f})")
        print(f"Modality: {pred_mod}")
        print(f"Spans:    {len(spans)} extracted -> {[(s.get('span'), s.get('bioType'), s.get('role')) for s in spans[:4]]}")
        print(f"AST Valid: {has_ast}")
        
        results.append({
            "domain": domain,
            "status": status,
            "modality": pred_mod,
            "spans_count": len(spans),
            "ast_valid": has_ast
        })

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS ACROSS ALL DEONTICBENCH DOMAINS")
    print("=" * 80)
    for r in results:
        print(f"{r['domain']:<46} | status={r['status']} | modality={r['modality']} | AST={r['ast_valid']}")

if __name__ == "__main__":
    run_crossdomain_eval()
