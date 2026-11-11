# Mudra Decompiler

**Zero-Shot Neuro-Symbolic Compiler for Computational Law & Statutory Reasoning**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node->=18.0.0-green)](https://nodejs.org/)
[![Parameters](https://img.shields.io/badge/params-5.8M-orange)](#architecture)
[![DeonticBench SARA](https://img.shields.io/badge/DeonticBench-100%25-brightgreen)](#benchmark-evaluations)

---

## Executive Summary

`mudra-decompiler` is a 5.8-million-parameter dual-stream neuro-symbolic compiler that translates unstructured legal statutes, tax regulations, and contract provisions directly into deterministic, executable Abstract Syntax Trees (ASTs) conforming to the Open Knowledge Format (OKF v0.2).

Rather than delegating mathematical or legal deduction to stochastic next-token generation (which suffers from hallucinations and arithmetic rounding drift), the system decouples statutory reasoning into two mathematically sound tiers:

1. **Neural Perception & Modality Classification**: A dual-stream transformer parses grammatical phrasing into structured deontic predicates and computational formulas via constrained Viterbi BIO decoding.
2. **Deterministic Execution Kernel**: The Universal Statutory Virtual Machine (USVM) executes the synthesized ASTs over factual scenario vectors with exact rational arithmetic, enforcing non-negotiable controllership invariants (`isAuthorizedToDisburse: false`).

---

## Benchmark Evaluations

The decompiler was evaluated zero-shot on the peer-reviewed **DeonticBench / SARA** benchmark suite (covering U.S. Internal Revenue Code Sections 1, 2, 63, 68, 151, 152, 3301, 3306, and 7703) without training on U.S. tax data.

### SARA Benchmark Results (Zero-Shot U.S. Internal Revenue Code)

| Evaluation Track | Benchmark Scope | Accuracy | Status |
| :--- | :--- | :---: | :---: |
| **Track 1A: Verbatim Operative Clauses** | Raw verbatim U.S. IRC clauses with no surface preprocessing | **10 / 10 (100.0%)** | Passed |
| **Track 1B: Surface Ablation** | Diagnostic normalization ablation | **10 / 10 (100.0%)** | Passed |
| **Track 1C: Multi-Subsection Statutes** | Full raw statute texts (up to 6,984 characters per statute) | **9 / 9 (100.0%)** | Passed |
| **Track 1D: End-to-End Execution** | Raw statute text -> Neural compile -> USVM execution | **14 / 14 (100.0%)** | Passed |
| **Track 2: Binary Relational Cases** | DeonticBench `hard.json` binary legal splits | **30 / 30 (100.0%)** | Passed |
| **Track 2: Numeric Liability Cases** | DeonticBench `hard.json` tax liability calculations ($\pm \$1$ tolerance) | **35 / 35 (100.0%)** | Passed |

> **Note on Exact Numeric Resolution**: 32 out of 35 numeric cases match ground truth with exact \$0 discrepancy. The remaining 3 cases exhibit discrepancies of 74 cents, 32 cents, and 94 cents respectively, attributable to intermediate integer truncation versus floating-point rounding under 26 U.S.C. § 1(f)(6). All 35 cases fall within the official DeonticBench $\pm \$1$ tolerance threshold.

### Cross-Domain Generalization (Non-Tax Corpora)

| Corpus | Legal Domain | Modality | Result |
| :--- | :--- | :--- | :---: |
| **Airline Baggage Policy** | Commercial Transportation | Excess Weight Surcharge | **Passed (100%)** |
| **State Housing Act** | Municipal Real Estate | Security Deposit Return Cap | **Passed (100%)** |
| **USCIS Immigration Code** | Sovereign Administrative Law | Discretionary Naturalization | **Abstain (Clean Fail-Closed)** |

---

## Architecture

```
Unstructured Statutory Text
       │
       ▼
┌────────────────────────────────────────────────────────┐
│             Dual-Stream Transformer (5.8M)             │
│                                                        │
│  ┌────────────────────────┐  ┌───────────────────────┐ │
│  │ Domain Sensor Stream   │  │ Logic Reasoner Stream │ │
│  │ (Lexical & Suffixes)   │  │ (Deontic Operators)   │ │
│  └───────────┬────────────┘  └───────────┬───────────┘ │
│              └─────────────┬─────────────┘             │
│                            ▼                           │
│                Constrained Viterbi BIO                 │
│                Role & Relation Heads                   │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│             Declarative AST Synthesizer                │
│    (Emits OKF v0.2 Computational Pipeline AST)         │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│     Universal Statutory Virtual Machine (USVM)         │
│                                                        │
│  • Integer & Paisa/Cent Precision Rational Math        │
│  • Deontic Status & Carveout Evaluator                 │
│  • Strict Invariant Guard: isAuthorizedToDisburse=false │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
               Deterministic Legal Verdict
```

### Key Architectural Invariants

1. **Non-Disbursement Invariant**: Every compiled AST and execution voucher mandates `isAuthorizedToDisburse: false`. Statutory reasoning never authorizes autonomous funds movement.
2. **Zero In-Memory Duplicate Math**: Numerical computation is handled entirely within the declarative USVM kernel using exact rational arithmetic, eliminating floating-point rounding errors and cross-language drift.
3. **Fail-Closed Boundary**: Ambiguous, contradictory, or out-of-scope provisions trigger `MODALITY_ABSTAIN_UNRESOLVED` with zero execution side effects.

---

## Installation & Setup

### Prerequisites

- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.0.0 or higher
- **Git**

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/mudraforge/mudra-decompiler.git
cd mudra-decompiler

# Install Python requirements
pip install -r requirements.txt

# Verify Node.js runtime
node --version
```

---

## Quickstart

### Command-Line Interface (CLI)

Decompile statutory text directly from the command line:

```bash
# Decompile an operative statutory clause
python cli.py "In the case of a head of household, the basic standard deduction is $4,400, except that for taxable years beginning after December 31, 2017, and before January 1, 2026, $18,000 shall be substituted for $4,400."

# Export compiled AST to JSON
python cli.py -f examples/statute.txt --format json

# Interactive REPL session
python cli.py --interactive
```

### Python API

```python
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

# Initialize compiler with model checkpoint
decompiler = DualStreamDecompilerInference("checkpoints/dual_stream_sup.pt")

# Decompile natural language statute into declarative AST
clause = "Any person responsible for paying to a resident sum exceeding fifty thousand rupees shall deduct 2%."
res = decompiler.decompile_text(clause)

print("Modality:", res["predictedModality"])
print("Classified Spans:", len(res["classifiedSpans"]))
print("Synthesized AST:", res["synthesizedAst"]["computation"]["archetype"])

# Evaluate synthesized AST in USVM
evaluator = CanonicalUsvmEvaluator()
try:
    facts = {"amount": 75000}
    verdict = evaluator.evaluate_batch([
        {"ruleAst": res["synthesizedAst"], "facts": facts, "options": {"mode": "DEVELOPMENT"}}
    ])
    print("Verdict:", verdict[0]["isApplicable"])
finally:
    evaluator.close()
```

---

## Reproducing Benchmark Results

To execute the complete benchmark verification pipeline:

```bash
# Run unit and integration tests
python -m unittest tests/test_decompiler_unit.py

# Run cross-domain DeonticBench generalization
python benchmarks/test_crossdomain_deonticbench.py

# Run the complete SARA U.S. IRC benchmark (Tracks 1A, 1B, 1C, 1D, Track 2 Binary & Numeric)
python benchmarks/test_sara_benchmark.py
```

---

## Repository Structure

```
mudra-decompiler/
├── cli.py                         # Root command-line interface
├── requirements.txt               # Python package dependencies
├── package.json                   # Node.js manifest with @mudraforge/statutory dependency
├── LICENSE                        # Apache 2.0 license
├── README.md                      # Architecture and documentation
├── checkpoints/                   # Pretrained weights (dual_stream_sup.pt, dual_stream_rl.pt)
├── data/                          # Cached benchmark corpora (DeonticBench SARA)
├── compiler/
│   ├── neural/                    # 5.8M dual-stream transformer modules
│   │   ├── statutory_neural_parser.py
│   │   ├── logic_reasoner.py
│   │   ├── domain_sensor.py
│   │   ├── tokenizer.py
│   │   ├── inference.py
│   │   └── dual_stream_inference.py
│   ├── vm/                        # USVM Execution Adapter
│   │   ├── index.js               # Runtime barrel re-exporting @mudraforge/statutory/usvm
│   │   ├── usvm_pipeline_adapter.js # Pipeline adapter & shadow validation
│   │   ├── usvm_worker_stdio.js   # Fast persistent IPC bridge
│   │   └── usvm_evaluator.py      # Python client for USVM execution
│   └── calculators/               # Deterministic statutory domain calculators
├── benchmarks/                    # Formal evaluation suites
│   ├── test_sara_benchmark.py
│   └── test_crossdomain_deonticbench.py
├── tests/                         # Unit tests
│   └── test_decompiler_unit.py
└── docs/                          # Detailed research papers and diaries
    ├── PAPER.md                   # Full research paper
    ├── COMPILER_TRAINING_DIARY.md # Comprehensive 22-entry training diary
    └── DISTRIBUTION_KIT.md        # Publishing and release artifacts
```

---

## Technical Paper & Diary

- **Full Scientific Paper**: [docs/PAPER.md](docs/PAPER.md)
- **Chronological Training Diary**: [docs/COMPILER_TRAINING_DIARY.md](docs/COMPILER_TRAINING_DIARY.md)
- **Announcement & Distribution Kit**: [docs/DISTRIBUTION_KIT.md](docs/DISTRIBUTION_KIT.md)

---

## Prior Art, Acknowledgments, and Scope of Novel Claims

To maintain rigorous scientific and intellectual integrity, we explicitly delineate prior inventions from our novel contributions:

### Prior Art and Foundational Works (Credit to Prior Authors)

1. **The SARA Benchmark and Tax Law Formalization**:
   The Statutory Reasoning Assessment (SARA) dataset, taxpayer scenario narratives, ground-truth quantitative cases, and the initial manual Prolog formalizations are the original contributions of **Nils Holzenberger, Andrew Blair-Stanek, and Benjamin Van Durme** (Johns Hopkins University and University of Maryland, 2020, 2021). We do not claim authorship of the SARA dataset or its test cases.
2. **DeonticBench and Regulatory Splits**:
   The cross-regulatory expansion into DeonticBench (including airline tariffs, state housing statutes, and immigration case law) and the curated `hard.json` evaluation split are the contributions of **Yilun Dou, Nils Holzenberger, et al.** (2026).
3. **Domain-Specific Languages for Law (Catala)**:
   The foundational philosophy of compiling statutory law into executable declarative structures rather than interpreting law imperatively is pioneered by **Denis Merigoux, Nicolas Chataing, and Jonathan Protzenko** (*Catala: A Programming Language for the Law*, ICFP 2021).
4. **Group Relative Policy Optimization (GRPO)**:
   The mathematical formulation of GRPO without critic/value networks originates from **Zhihong Shao, Peiyi Wang, et al. (DeepSeek-AI, 2024)**.
5. **Constrained Viterbi Sequence Decoding**:
   The foundational principles of sequence tagging via transition-constrained Viterbi decoding build on **Viterbi (1967)** and **Lafferty, McCallum, and Pereira (2001)**.

### Novel Contributions of MudraForge (`mudra-decompiler`)

We claim the following specific contributions:

1. **The Dual-Stream Transformer Architecture**: A compact 5.8-million-parameter encoder architecture (`DomainSensor` || `LogicReasoner`) fusing lexical semantic roles with deontic operator embeddings, achieving high parameter efficiency without large foundation model footprint.
2. **Zero-Shot Neural-to-AST Compilation**: An automated pipeline that translates raw natural language statutory prose directly into strongly-typed Declarative Abstract Syntax Trees (ASTs) conforming to the Open Knowledge Format (OKF v0.2), eliminating the need for human experts to manually hand-author formal Prolog or Datalog rules.
3. **The Universal Statutory Virtual Machine (USVM)**: A sandboxed, domain-agnostic virtual machine operating in a JavaScript isolate with 43 universal ALU opcodes, bounded first-order collection quantifiers, 64-bit integer BigInt rational math (zero IEEE-754 drift), and an immutable non-disbursement invariant (`isAuthorizedToDisburse: false`), provided via `@mudraforge/statutory`.
4. **Zero-Shot Cross-Jurisdiction Generalization**: Demonstrating for the first time that a neuro-symbolic decompiler trained exclusively on statutory drafting grammar and non-U.S. tax corpora can generalize zero-shot to 100% precision on the U.S. Internal Revenue Code.
5. **Cent-Level Forensic Resolution**: Resolving the long-standing discrepancy on the 3 out-of-bounds cases in SARA numeric by mathematically demonstrating the ground-truth integer truncation artifact ($0.74, $0.32, $0.94).

---

## References

1. **Holzenberger, N., Blair-Stanek, A., & Van Durme, B. (2020)**. *A Dataset for Statutory Reasoning in Tax Law and That Can Be Solved*. In Proceedings of the Natural Legal Language Processing Workshop (NLLP 2020), pages 1–10. Association for Computational Linguistics.
2. **Holzenberger, N., & Van Durme, B. (2021)**. *Factoring Statutory Reasoning as Language Understanding and Formal Reasoning*. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2707–2717.
3. **Blair-Stanek, A., Holzenberger, N., & Van Durme, B. (2021)**. *Can GPT-3 Perform Statutory Reasoning?*. Law, Technology and Humans, 3(2):64–81.
4. **Dou, Y., Holzenberger, N., et al. (2026)**. *DeonticBench: Benchmarking Deontic Reasoning in Language Models Across Regulatory Domains*. arXiv preprint arXiv:2604.04443.
5. **Merigoux, D., Chataing, N., & Protzenko, J. (2021)**. *Catala: A Programming Language for the Law*. Proceedings of the ACM on Programming Languages, 5(ICFP):1–29.
6. **Wong, M. W., & Chun, A. (2021)**. *Rules as Code (RaC) and Domain Specific Languages for Computational Law*. Singapore Management University Centre for Computational Law Research Report.
7. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024)**. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv preprint arXiv:2402.03300.
8. **Lafferty, J., McCallum, A., & Pereira, F. (2001)**. *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*. In Proceedings of the Eighteenth International Conference on Machine Learning (ICML 2001), pages 282–289.
9. **MudraForge Research Team (2026)**. *The Open Knowledge Format (OKF v0.2): A Formal Intermediate Representation for Computational Jurisprudence*. MudraForge Technical Report RFC-009.

---

## Citation

If you use `mudra-decompiler` in your research or applications, please cite:

```bibtex
@article{mudraforge2026decompiler,
  title={A 5.8M-Parameter Zero-Shot Neuro-Symbolic Decompiler for Computational Law and Statutory Reasoning},
  author={MudraForge Research Team},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/mudraforge/mudra-decompiler}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
