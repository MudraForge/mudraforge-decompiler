# MudraForge Decompiler

**Zero-Shot Neuro-Symbolic Compiler for Computational Law & Statutory Reasoning**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node->=18.0.0-green)](https://nodejs.org/)
[![Parameters](https://img.shields.io/badge/params-7.9M%20total-orange)](#inside-the-dual-stream-model)
[![SARA Track 1D](https://img.shields.io/badge/SARA_Track_1D-14%2F14-brightgreen)](#benchmark-evaluations)
[![SARA Track 2 Binary](https://img.shields.io/badge/SARA_Track_2_binary-30%2F30-blue)](#benchmark-evaluations)
[![SARA Track 2 Numeric](https://img.shields.io/badge/SARA_Track_2_numeric-35%2F35_%28%C2%B11%29-blue)](#benchmark-evaluations)

**Author and maintainer:** Mondeep Engti. MudraForge is an independent sole
proprietorship owned and operated by Mondeep Engti.

---

## Executive Summary

`mudraforge-decompiler` is a 7.9-million-parameter dual-stream neuro-symbolic compiler (2.08M in `DomainSensor` and 5.78M in `LogicReasoner`). A neural annotation model identifies statutory spans, roles, modality, and relations; a deterministic schema lowerer turns those annotations into executable Abstract Syntax Trees (ASTs) conforming to the Open Knowledge Format (OKF v0.2); the USVM executes them.

Rather than delegating mathematical or legal deduction to stochastic next-token generation (which suffers from hallucinations and arithmetic rounding drift), the system decouples statutory reasoning into three tiers:

1. **Neural Perception & Modality Classification**: A dual-stream transformer parses grammatical phrasing into typed spans, span roles, deontic modality, and pairwise relation edges via constrained Viterbi BIO decoding.
2. **Deterministic Schema Lowering**: `synthesize_usvm_ast` turns the span graph into a canonical OKF v0.2 AST, enforcing the target grammar in code rather than in the weights.
3. **Deterministic Execution Kernel**: The Universal Statutory Virtual Machine (USVM) executes the synthesized ASTs over factual scenario vectors with exact rational arithmetic, enforcing non-negotiable controllership invariants (`isAuthorizedToDisburse: false`).

---

## The Decompiler Thesis: Why "Decompiler" Instead of "Compiler"?

A foundational question in computational law is why this system is formally designated as a **Decompiler** rather than simply a compiler.

In classical computer systems:
- A **compiler** translates high-level, human-authored source code into a dense, serialized distribution binary (such as machine code or bytecode) optimized for execution by a low-level target machine (such as an x86 CPU or JVM).
- A **decompiler** (such as Ghidra or IDA Pro) performs the inverse operation: it ingests an already-compiled, opaque distribution binary and reverse-engineers it back into a high-level, structured Abstract Syntax Tree (AST) with recovered control-flow graphs and typed variables.

In statutory jurisprudence, the legislative process mirrors this exact computational pipeline:

1. **The Upstream Compiler (The Legislature)**: The legislative branch begins with high-level policy intent (e.g., progressive income brackets, marginal withholding thresholds, or corporate compliance gates). Legal drafting committees take these mathematical and logical specifications and **compile them into natural-language statutory text** (such as the Internal Revenue Code, the Goods and Services Tax Act, or official Government Gazettes).
2. **Statutes as "Compiled Binaries for Human Runtime"**: Far from being informal text, statutory prose is a dense, heavily optimized, and highly defensive serialized distribution format designed specifically for a human judicial runtime (courts, tax authorities, and legal counsel). To an automated machine, statutory prose behaves like a stripped binary: it is packed with conditional jumps (*"subject to paragraph (2)"*), pointer indirections (*"within the meaning of section 7703"*), implicit scope boundaries, and defeasible exceptions (*"shall not apply if..."*).
3. **The Reverse-Engineering Operation (`mudraforge-decompiler`)**: Because the natural language statute is already an end-product compiled artifact, an AI system cannot "compile" it from source. Instead, `mudraforge-decompiler` acts as a **true statutory decompiler**: it ingests the compiled natural-language legal artifact, strips away the bureaucratic and linguistic assembly packaging, recovers the underlying typed variables (`symbolLegend`), and reconstructs the clean, declarative Abstract Syntax Tree (OKF v0.2 AST).
4. **Execution in the Virtual Machine**: Once decompiled into a formal AST, the logic is executed deterministically by the Universal Statutory Virtual Machine (USVM) using exact BigInt rational arithmetic, restoring the original mathematical clarity of the legislative intent with zero hallucinations.

---

## Inside the Dual-Stream Model

Before the lowerer sees anything, a 7.9M-parameter dual-stream encoder turns a statute sentence into a typed span graph. This section describes that model on its own terms: how the two streams divide the sentence, what each of the four heads predicts, what the raw output looks like, and how well the model does before any symbolic code runs.

### Parameter accounting

The released checkpoints (`checkpoints/dual_stream_sup.pt`, `checkpoints/dual_stream_rl.pt`) contain **7,862,518 trainable parameters**:

| Stream | Structure | Parameters |
| :--- | :--- | ---: |
| `DomainSensor` | 4-layer, 192-d encoder over raw tokens + suffix embeddings; 9-class token head; binary abstain head | 2,081,387 |
| `LogicReasoner` | 224-d token + 32-d suffix embeddings, 256-d encoder, 2 fusion layers, BIO/role/biaffine/modality heads | 5,781,131 |
| **Total** | | **7,862,518** |

> **Why some documents say 5.8M.** 5.8M is the `LogicReasoner` policy that GRPO tunes; the training diary's 2.76M + 2.99M split describes the pre-release 5.74M configuration. The counts above are measured directly from the shipped state dicts, so they are the numbers used throughout this README.

### One sentence, two streams

Consider a withholding clause:

> *"Any person responsible for paying to a resident any sum for purchase of goods of an amount exceeding fifty lakh rupees shall deduct an amount equal to 0.1 percent of such sum as income-tax thereon."*

Both streams read the same token sequence, but they do not see the same features:

- **`DomainSensor` (lexical stream).** It embeds the raw word plus a morphological suffix embedding, runs four 192-d transformer layers, and predicts one of nine token classes (`DT_MONETARY_AMOUNT`, `DT_PERCENT_RATE`, `DT_STRUCTURAL_CONNECTIVE`, and so on) plus a sequence-level **abstain probability** (`domainAbstainProb`). Its job is to recognize domain vocabulary: money, rates, dates, parties, documents, and chatter.
- **`LogicReasoner` (structural stream).** It never sees the raw token id. It sees a closed logic vocabulary (`shall`, `exceeding`, `except`, `notwithstanding`, comparatives, quantifiers) plus the same suffix embedding, so its input is mostly grammar rather than subject matter. This is what lets the model transfer to a new legal domain: the logic stream learns control flow, not tax nouns.
- **Where the streams meet.** The two encoders exchange context through bidirectional cross-attention at mid-depth. `DomainSensor` cross-attends to the logic state, and `LogicReasoner` receives the detached nine-dimensional domain distribution as an extra input channel. Detaching the domain probabilities prevents the logic stream from learning to copy a domain label as a shortcut. During GRPO, `DomainSensor` is frozen so the reward signal cannot drift the lexical representation.

Training applies balanced open-class slot dropout to both streams: nouns and verbs are replaced by `<LEGAL_NOUN>`/`<LEGAL_VERB>` at 8-30% per clause (raised to at least 28% for open-class slots), and the diary records runs at up to 35%. The point is to stop the logic stream from using a particular tax noun as a proxy for "this is a rule".

### The four heads

After fusion, `LogicReasoner` exposes four prediction heads. Each head answers one local question; none of them emits an AST.

1. **Constrained Viterbi BIO span tagger.** A token-level classifier over 23 BIO tags (`B-GATE`, `I-THRESHOLD`, `B-RATE`, `B-ENTITY`, and so on). Decoding is a Viterbi pass with hard transition constraints: an `I-` tag cannot follow `O`, a span cannot switch type mid-token, and a `per` that is not followed by `cent`/`centum`/`%` cannot start a rate. The result is a set of non-overlapping spans such as `exceeding fifty lakh rupees` or `0.1 percent`.
2. **Span Boundary Role classifier.** Each span is pooled into a 128-d vector from its start state, end state, their difference, and a mean-pooled interior. A classifier assigns one of 20 statutory roles, including `ROLE_TRANSACTION_EXCESS_THRESHOLD`, `ROLE_TRANSACTION_CLIFF_THRESHOLD`, `ROLE_STANDARD_STATUTORY_RATE`, `ROLE_PENALTY_OR_DEFAULT_OVERRIDE`, `ROLE_COMPUTATION_OPERAND`, and `ROLE_PROHIBITION_GATE`. This is where the model distinguishes a **cliff** (the whole base is taxed once the threshold is crossed) from an **excess** (only the amount above the threshold is taxed).
3. **Deep biaffine relational head.** A scoring tensor over every ordered span pair, with 24 relation labels (`REL_RATE_TO_THRESHOLD`, `REL_THRESHOLD_TO_TRANSACTION`, `REL_COMBINATOR_MIN`, `REL_SUBTRACTION_BASE`, `REL_CONDITION_TO_GATE`, `REL_CAP_CEILING`, `REL_LOGICAL_OR`, and so on). The biaffine form scores a directed edge as a bilinear function of the source and target span vectors, which is why it can represent "this rate applies to that threshold" without a fixed window or a regex.
4. **Rule-modality head.** A pooled-sequence classifier over eight labels: three computational shapes (`MODALITY_COMPUTATIONAL_EXCESS`, `MODALITY_COMPUTATIONAL_CLIFF`, `MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB`), four deontic/status families (`MODALITY_DEONTIC_PROHIBITION`, `MODALITY_DEONTIC_OBLIGATION_WINDOW`, `MODALITY_DEONTIC_STATUS_OR_EVIDENCE`, `MODALITY_DEONTIC_PERMISSION`), and `MODALITY_ABSTAIN_UNRESOLVED`. The abstain label is the fail-closed gate: when the model cannot place the clause in any operative family, the pipeline returns `UNRESOLVED_STATUTORY_AMBIGUITY` instead of guessing.

### Worked example: raw neural output before lowering

This is the actual output of `DualStreamDecompilerInference.decompile_text` on a U.S. IRC bracket clause, using `checkpoints/dual_stream_sup.pt`. Nothing below has been post-edited; only whitespace is trimmed. At this point `synthesize_usvm_ast` has not run.

**Input (IRC § 1(a)(ii) bracket, verbatim):**

> *"If taxable income is over $36,900 but not over $89,150, tax shall be $5,535, plus 28% of the excess over $36,900."*

**Raw annotation output:**

```json
{
  "predictedModality": "MODALITY_COMPUTATIONAL_COMPOUND_OR_SLAB",
  "modalityConfidence": 0.9757,
  "domainAbstainProb": 0.0,
  "classifiedSpans": [
    {"span": "over $36,900",    "bioType": "THRESHOLD", "role": "ROLE_TRANSACTION_EXCESS_THRESHOLD", "confidence": 0.9917, "spanIdx": 0},
    {"span": "not over $89,150","bioType": "THRESHOLD", "role": "ROLE_TRANSACTION_CLIFF_THRESHOLD",  "confidence": 0.9938, "spanIdx": 1},
    {"span": "5,535",           "bioType": "RATE",      "role": "ROLE_COMPUTATION_OPERAND",          "confidence": 0.4654, "spanIdx": 2},
    {"span": "28%",             "bioType": "RATE",      "role": "ROLE_STANDARD_STATUTORY_RATE",      "confidence": 0.9924, "spanIdx": 3}
  ],
  "relationalEdges": [
    {"sourceSpan": "over $36,900", "targetSpan": "not over $89,150", "relationType": "REL_BRACKET_NEXT"},
    {"sourceSpan": "28%",          "targetSpan": "over $36,900",     "relationType": "REL_RATE_TO_THRESHOLD"}
  ]
}
```

Read the annotation, not the JSON punctuation:

- the BIO head proposed four spans and no overlapping tokens;
- the role head marked the lower bound as an **excess** threshold, the upper bound as a **cliff** threshold, and `28%` as the standard rate;
- the biaffine head linked the two thresholds into one bracket (`REL_BRACKET_NEXT`) and attached the rate to the lower bound (`REL_RATE_TO_THRESHOLD`);
- the modality head called the shape `COMPOUND_OR_SLAB`, not a flat rate;
- the one low-confidence role (`5,535` at 0.4654) is the fixed base amount, and the value is still correct; confidence is reported per span so a reviewer can see where the model is unsure.

Only after this point does `synthesize_usvm_ast` apply the deterministic grammar: it turns the two thresholds into a bracket range, the `28%` plus `REL_RATE_TO_THRESHOLD` edge into a rational multiply on the excess, and the fixed amount into a base constant. The schema, the operators, and the money type are code, not weights.

### What the model achieves on its own

The neural tracks measure annotation and composition; only Track 1D adds the USVM verdict. These are the external benchmark results:

| Track | Input and composition | Result |
| :--- | :--- | :---: |
| **1A: Verbatim operative clauses** | U.S. IRC text -> spans, roles, modality, edges; no USVM verdict | **10 / 10** |
| **1B: Surface ablation** | Same clauses with `$` -> `rupees` and `%` -> `per cent`; no USVM verdict | **10 / 10** |
| **1C: Multi-subsection statutes** | Raw statutes up to 6,984 characters; no USVM verdict | **9 / 9** |
| **1D: End-to-end** | Raw text -> model -> lowerer -> USVM verdict | **14 / 14** |

The model's internal validation metrics come from the training diary's 1,910-clause synthetic validation set, not from the external benchmark scripts:

| Validation metric | Value |
| :--- | ---: |
| Domain token classification accuracy | 96.96% |
| BIO span boundary accuracy | 88.88% |
| Rule modality accuracy | 97.17% |
| Semantic role macro-F1 | 94.48% |
| Relational biaffine edge F1 | 88.14% |
| Clause-level complete exact match | 75.49% |
| Average validation NLL | 0.7694 |

Two cautions belong next to these numbers. First, 75.49% clause-level EM is the model's *annotation* score; the lowerer can still recover a correct rule from an imperfect annotation, and the external Tracks 1A-1D are the evidence for that. Second, none of these rows is a score on the 65-case `hard.json` split. The full pipeline has not been run on that split, and Track 2 does not substitute for it.

---

## System Boundaries: The Neural Compiler vs. The Execution Engine

A common point of confusion in statutory AI is whether neural models should perform calculations. In MudraForge, the architecture enforces a strict division of responsibility:

```
[ Natural Language Statute ]
             │
             │ Dual-Stream neural model: BIO span boundaries, domain classes,
             │ deontic modality, span roles, pairwise relation edges
             ▼
[ Typed statutory span graph ]
             │
             │ synthesize_usvm_ast: deterministic role grammar, numeric/date
             │ parsing, relation-to-operator lowering
             ▼
[ Canonical OKF v0.2 / USVM AST ]
             │
             │ Evaluation over taxpayer facts (@mudraforge/statutory USVM)
             ▼
[ Deterministic Legal Verdict & Exact Tax Liability ]
```

### What the Neural Model Outputs
The neural model (`dual_stream_sup.pt` / `dual_stream_rl.pt`) is an **annotation model**, not an AST generator and not a calculator:
- **It does NOT emit raw tax numbers or final legal verdicts.**
- **It does NOT generate arbitrary executable Python/JS code on the fly.**
- **It emits token spans, span roles, deontic modality, and pairwise relation edges.** `synthesize_usvm_ast` then lowers those annotations, plus parsed rates, thresholds, and dates, into the canonical AST.
- **The AST schema is enforced by code, not learned.** `compile_composed_arithmetic_pipeline` builds pipeline steps from relation edges and statutory patterns, and the USVM validates the result.

The model provides perception; the deterministic plumbing provides schema conformance. Neither half alone produces a compilable rule. This split is also why the benchmark tables report the pipeline composition, not one monolithic score.

### The Checkpoint Is Not a Standalone Compiler
The model is one stage of a three-stage system, and all three are required for a correct answer:

| Stage | Component | Input -> Output |
| :--- | :--- | :--- |
| **Perception** | `DualStreamStatutoryParser` checkpoint | statute text -> spans, roles, deontic modality, relation edges |
| **Schema lowering** | `synthesize_usvm_ast` and `compile_composed_arithmetic_pipeline` | span graph -> canonical, schema-valid AST |
| **Execution** | `@mudraforge/statutory` USVM | AST + facts -> deterministic verdict |

A checkpoint without the lowerer emits annotations, not a rule. An AST without the USVM is a document, not a verdict. The shipped unit is the pipeline, not the checkpoint.

### Why the Schema Lives in Code, Not in the Weights
The schema contract is implemented in deterministic code (`_SUPPORTED_ARCHETYPES`, `normalize_ast`, `FACT_BINDINGS`, `_ENUM_DOMAINS`, `CatalaLoweringError`), not learned by the network. This has four consequences:

- **Schema versions are independent of the checkpoint.** Moving from one OKF revision to the next, or changing a field, is a lowerer change plus tests, not a retraining run. RFC-011 (typed inputs) is a design proposal for exactly this layer; it is not implemented here yet.
- **The model is target-independent.** The same span graph can be lowered to the USVM AST and, through `compiler/converters/catala_emitter.py`, to compiler-checked Catala. Adding a target does not touch the weights.
- **Output is deterministic and auditable.** The same annotations produce the same AST, and unsupported archetypes, operators, fact keys, or rounding policies raise instead of being guessed.
- **The cost is explicit.** Adding an operator, archetype, fact key, or enum requires code and tests. The rigidity moved out of the weights and into a small, versioned, testable surface. That is the trade, and it is deliberate.

### Why Both Halves Ship Together
The public boundary is the compiler package, not the checkpoint file. `DualStreamDecompilerInference` is the supported entry point because it composes the model with the schema lowerer; `CanonicalUsvmEvaluator` composes the AST with the USVM. Distributing `dual_stream_sup.pt` alone would give a consumer a classifier, not a decompiler.

### Porting the Model to Another Legal Domain
The Dual-Stream split makes the domain layer swappable. `DomainSensorModel` handles raw vocabulary and domain classes; `LogicReasonerModel` operates on a closed logic vocabulary plus morphological suffixes, which is why the noun-dropout ablation holds up.

A new domain still needs a small, explicit amount of work:

1. Extend or fine-tune `DomainSensor` for the new vocabulary and token classes.
2. Register domain facts and their schema types in `FACT_BINDINGS`.
3. Add categories to `_ENUM_DOMAINS` when the domain has closed vocabularies.
4. Add archetype normalizers when the domain uses unfamiliar AST shapes.
5. Add operators to USVM and the emitter if the domain needs dates, collections, or proration.
6. Build an evaluation set before claiming the port works.

What does not change is the checkpoint architecture, the schema contract, and the USVM invariants. Discretionary domains should abstain rather than compile; that is a property of the design, not a failure of the model.

### Why the Model Is an Annotation Model, Not an AST Generator

The model is small and narrow on purpose. Statutory text carries two separable signals: domain vocabulary (who pays, what is measured, which amount) and logical structure (conditions, comparisons, obligations, exceptions, and relations between amounts). A 5.8M logic encoder can learn the second when it is trained on local decisions: BIO span boundaries, span roles, modality, and pairwise relation edges.

Asking the same model to emit a complete JSON AST would force it to memorize field names, operators, and arithmetic conventions. Those are better expressed in code: they change on a different schedule, they must be type-checked, and an unsupported construct must fail closed rather than become a plausible-looking field.

The dual-stream split follows the same reasoning. `DomainSensor` sees raw vocabulary; `LogicReasoner` sees a closed logic vocabulary and morphological suffixes. In training-time ablations, masking up to 75% of domain nouns moved dual-stream in-distribution EM from 75.5% to 74.2%, while the monolithic baseline fell from 78.3% to 13.9%. That supports the separation. It does not prove transfer to every new statute; only a full-pipeline evaluation on that statute can.

The trade-off is explicit: the deterministic lowerer, not the checkpoint, is the coverage bottleneck. That is deliberate. A lowerer change is reviewable, versioned, testable, and fails closed. A weight change is none of those things.

### Can We Add More U.S. Law Without Code?

Sometimes, but not generally. The answer depends on whether the new section is a new instance of an existing shape or a genuinely new shape.

| Change | Decompiler code? | What is still required |
| :--- | :--- | :--- |
| New section using an existing archetype (slab, cliff, min/max, cap, excess, temporal window, quantifier) and existing span roles/relations | Often no | The model must extract the spans, roles, and relations; the caller supplies facts; a new evaluation vector proves the result |
| New money, boolean, or string fact names | No for USVM execution | Fact paths are derived from spans; the caller must supply matching facts |
| New integer, count, or rate facts | Yes, small | Register the fact in `FACT_BINDINGS` so the bridge encodes it as an integer rather than money |
| New operator or arithmetic construct | Yes | Add it to synthesis and the USVM kernel, with tests |
| New archetype or cross-section composition | Yes | New lowering path, kernel support if needed, and an evaluation set |
| New drafting style or vocabulary | Model work, not schema work | Fine-tune `DomainSensor` and add training/evaluation data |
| Catala output for a new domain | Yes | Register fact bindings and enum domains; the emitter currently supports only `DECLARATIVE_PIPELINE` |

So the current system is a validated vertical slice, not a zero-code engine for all U.S. law. What scales is the registry: archetypes, operators, roles, and relations. Adding a statute inside that surface can be code-free; adding a new legal shape is not. The breakthrough condition is measurable: run the full **model + plumbing + USVM** pipeline on new sections and count how many compile and execute correctly without per-statute code.

### Why We Import `@mudraforge/statutory` (The Engine)
An Abstract Syntax Tree is a declarative program; to evaluate whether an invoice or taxpayer case complies with the statute, the AST must be executed against concrete facts. We import `@mudraforge/statutory` as the backend runtime for four fundamental reasons:
1. **Zero Arithmetic Hallucination**: Autoregressive next-token prediction and neural embeddings cannot reliably perform multi-step rational arithmetic. The Universal Statutory Virtual Machine (USVM) in `@mudraforge/statutory` executes exact 64-bit integer BigInt rational math (cents and paisa), eliminating floating-point rounding drift.
2. **Immutable Non-Disbursement Invariant**: Under controllership standards, automated legal reasoning must be strictly separated from fund movement. The USVM engine enforces the non-negotiable invariant `isAuthorizedToDisburse: false` across all evaluation vouchers.
3. **True Neuro-Symbolic Decoupling**: Neural networks excel at linguistic perception (parsing synonyms, complex legalese, and sentence syntax). Symbolic virtual machines excel at deductive logic, cycle detection, and arithmetic. Decoupling them prevents the reasoning collapse seen in end-to-end LLMs.
4. **Canonical Shared Ground Truth**: Rather than maintaining duplicate calculation engines across repositories, `mudraforge-decompiler` consumes `@mudraforge/statutory` (v3.1.0) as the shared, Merkle-hashed canonical standard used across the MudraForge ecosystem.

---

## Benchmark Evaluations

The benchmark suite below separates three pipeline compositions that are easy to conflate: neural annotation without execution (Tracks 1A-1C), the full **model + plumbing + USVM** pipeline (Track 1D), and symbolic execution of hand-authored ASTs with no model (Track 2). Only Track 1D produces a correct answer end to end, and it covers 14 test vectors, not the 65-case hard split.

### SARA Benchmark Results (Neural and Symbolic Tracks)

| Evaluation Track | What Runs | Benchmark Scope | Accuracy | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Track 1A: Verbatim Operative Clauses** | Model + plumbing, no USVM verdict | Raw verbatim U.S. IRC clauses with no surface preprocessing | **10 / 10 (100.0%)** | Passed |
| **Track 1B: Surface Ablation** | Model + plumbing, no USVM verdict | Diagnostic normalization ablation | **10 / 10 (100.0%)** | Passed |
| **Track 1C: Multi-Subsection Statutes** | Model + plumbing, no USVM verdict | Full raw statute texts (up to 6,984 characters per statute) | **9 / 9 (100.0%)** | Passed |
| **Track 1D: End-to-End Execution** | **Model + plumbing + USVM** | Raw statute text -> neural spans -> synthesized AST -> USVM execution | **14 / 14 (100.0%)** | Passed |
| **Track 2: Binary Relational Cases** | Hand-authored AST + USVM (no model) | DeonticBench `hard.json` binary splits | **30 / 30 (100.0%)** | Passed |
| **Track 2: Numeric Liability Cases** | Hand-authored AST + USVM (no model) | DeonticBench `hard.json` tax liability calculations ($\pm \$1$ tolerance) | **35 / 35 (100.0%)** | Passed |

> **Reading this table.** Only Track 1D runs the full **model + plumbing + USVM** pipeline, and it covers 14 test vectors. Track 2 scores the symbolic layer only: hand-authored canonical ASTs executed by USVM. The 65-case hard split has not been run end-to-end through the full pipeline. That run is the missing experiment, and its result should not be inferred from Track 2.

> **Note on Exact Numeric Resolution**: 32 out of 35 numeric cases match ground truth with exact \$0 discrepancy. The remaining 3 cases exhibit discrepancies of 74 cents, 32 cents, and 94 cents respectively, attributable to intermediate integer truncation versus floating-point rounding under 26 U.S.C. § 1(f)(6). All 35 cases fall within the official DeonticBench $\pm \$1$ tolerance threshold.

### Comparison with Published Frontier Baselines (`DeonticBench` & `DAR`, 2026)

> **Provenance and scope.** Every baseline figure below is quoted from the cited paper or harness. None of those baselines was independently reproduced by MudraForge, and the studies differ in prompts, tool access, and scoring rules, so the rows are not a controlled leaderboard. Our rows are labeled by composition: Track 2 is deterministic USVM execution of hand-authored canonical ASTs, and Track 1D is the full model + plumbing + USVM pipeline measured on 14 test vectors. Neither is a zero-shot model score on the 65-case hard split. These numbers show that a symbolic engine given correct formalizations executes this benchmark near-perfectly. They do not show that the full pipeline compiles the hard split end-to-end.

The 35 numeric instances and 30 binary instances in `hard.json` were specifically isolated by the Johns Hopkins benchmark authors because they represent the exact statutory edge-cases that cause frontier large language models to fail:

> *"The hard subset omits easier tasks identified by both model runs (GPT-5.2, o3, Claude Sonnet) and human review. The proportion of hard instances is 35 / 100 in SARA Numeric and 30 / 100 in SARA Binary..."*  
> — **Dou et al., DeonticBench (arXiv:2604.04443, 2026)**

Below is the comparative performance across published literature on that exact **`SARA-Numeric` (`hard.json`, 35 cases)** split:

| Model / Architecture | Parameter Scale | Reasoning Paradigm | Score on `SARA-Numeric (Hard)` | Citation / Source |
| :--- | :---: | :--- | :---: | :--- |
| **`Qwen3.5-35B`** | 35B | Agentic CLI Harness | **11.0%** | *DAR (Dou et al., 2026)* |
| **`GPT-5.2`** | Frontier | Standard Direct / Chain-of-Thought | **30.0%** | *DAR (Dou et al., 2026)* |
| **`Qwen3.5-35B`** | 35B | Standard Direct / Chain-of-Thought | **34.0%** | *DAR (Dou et al., 2026)* |
| **`Claude Sonnet 4.5`** | Frontier | Standard Direct / Chain-of-Thought | **36.0%** | *DAR (Dou et al., 2026)* |
| **Best Model in `DeonticBench`**<br>*(GPT-5.2-Codex, o3, Qwen3-Coder-480B)* | Up to 480B | CoT / LLM $\rightarrow$ External SWI-Prolog | **44.4%** | *DeonticBench (arXiv:2604.04443)* |
| **`Claude Sonnet 4.5` + `Terminus-KIRA`** | Frontier | Multi-Turn Agentic Harness | **54.0%** | *DAR (Dou et al., 2026)* |
| **`GPT-5.2` + `Terminus-KIRA`** | Frontier | Multi-Turn Agentic Harness | **60.0%** | *DAR (Dou et al., 2026)* |
| **MudraForge USVM** (Track 2) | n/a (symbolic) | Hand-authored canonical ASTs -> deterministic USVM | **100.0% (35 / 35)** | *This Repository, Track 2* |
| **MudraForge full pipeline** (Track 1D) | 7.9M total (5.8M logic) | Model + plumbing + USVM, end-to-end | **14 / 14 test vectors** (not the hard split) | *This Repository, Track 1D* |

> **On `SARA-Binary (Hard)` (30 cases)**: Bloomberg AI's [*PercepLog-0* study](https://aclanthology.org/2026.findings-emnlp.2099/) (Kordjamshidi et al., EMNLP 2026) reports ~90% accuracy when ground-truth formal rules are supplied by human experts. That is the closest published setting to our Track 2 result (30 / 30 with hand-authored canonical ASTs), not to our full-pipeline track. The two results are still not directly comparable: different splits, harnesses, and definitions of a correct answer. The comparison does not establish model superiority. It shows that once a correct formalization exists, symbolic execution is no longer the bottleneck.

**What this comparison does and does not show**

- It does **not** show that the neural decompiler beats frontier LLMs on `hard.json`. The 35 / 35 and 30 / 30 rows execute hand-authored ASTs.
- It **does** show that the USVM executes correct formalizations of the tested IRC sections consistently, and that the hard split remains far from saturated for direct LLM reasoning.
- Only a full-pipeline run on `hard.json` could be labeled **model + plumbing + USVM**. That run does not exist yet, and Track 2 must not be relabeled as if it did.
- The open problem is the translation step: statute to correct formal rule. That is the step that needs harder benchmarks, not another leaderboard of execution accuracy after a human has already done the hard part.

### Cross-Domain Spot Check (Non-Tax Corpora)

This is a qualitative spot check, not an execution benchmark. `benchmarks/test_crossdomain_deonticbench.py` runs two clauses per corpus through the model and checks only modality classification and whether a non-empty AST structure was produced. No USVM verdict and no numeric correctness are asserted.

| Corpus | Legal Domain | What Is Checked | Result |
| :--- | :--- | :--- | :---: |
| **Airline Baggage Policy** | Commercial Transportation | 2 clauses: modality + AST presence, no execution | **2 / 2** |
| **State Housing Act** | Municipal Real Estate | 2 clauses: modality + AST presence, no execution | **2 / 2** |
| **USCIS Immigration Code** | Sovereign Administrative Law | 2 clauses: abstention behavior | **2 / 2 fail-closed (`ABSTAIN`)** |

---

## Architecture

![Dual-Stream Architecture](docs/images/dual_stream_architecture.png)

### Key Architectural Invariants

1. **Non-Disbursement Invariant**: Every compiled AST and execution voucher mandates `isAuthorizedToDisburse: false`. Statutory reasoning never authorizes autonomous funds movement.
2. **Zero In-Memory Duplicate Math**: Numerical computation is handled entirely within the declarative USVM kernel using exact rational arithmetic, eliminating floating-point rounding errors and cross-language drift.
3. **Fail-Closed Boundary**: Ambiguous, contradictory, or out-of-scope provisions trigger `MODALITY_ABSTAIN_UNRESOLVED` with zero execution side effects.

### Training Telemetry

- **Supervised training stability**: The single-stream baseline diverged after Epoch 8 (validation loss 0.708 to 0.845); the Dual-Stream model converged to NLL 0.751.
- **GRPO policy optimization**: Single-stream RL plateaued at +3.22 reward and 46.4% vector pass rate; freezing `DomainSensor` during RL moved Dual-Stream to +17.42 reward and 91.5% pass rate on the RL validation vectors.
- **Interactive visualizer**: Run telemetry is available in [docs/training_dashboard.html](docs/training_dashboard.html).

### Training-Time Observations

The training diary records four ablations that shaped the architecture. These are training-time observations, not held-out benchmark results, and the benchmark scripts in this repository do not reproduce them:

- **Domain-noun dropout**: masking legal nouns moved the single-stream baseline from 78.3% to 13.9% EM, while the dual-stream model moved from 75.5% to 74.2%. This is the evidence for isolating domain vocabulary from the logic stream.
- **Label smoothing**: constant smoothing caused boundary jitter on discrete deontic labels; cosine-annealed smoothing to zero worked better.
- **GRPO reward hacking**: execution-only rewards produced tautological ASTs; adding counterfactual vectors penalized rules that fired below their thresholds.
- **Cliff versus excess thresholds**: the lowerer distinguishes `CLIFF` from `EXCESS`, which prevents withholding on the full base when the statute taxes only the excess.

---

## Installation & Setup

### Prerequisites

- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.0.0 or higher
- **Git**

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MudraForge/mudraforge-decompiler.git
cd mudraforge-decompiler

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

# Export compiled AST as literate Catala (.catala_en) source code
python cli.py "Any person paying to a resident sum for purchase of goods exceeding 50 Lakh rupees shall deduct 0.1 percent." --format catala

# Export decompiled Catala source directly to file
python cli.py -f examples/statute.txt --export-catala output/statute.catala_en

# Interactive REPL session
python cli.py --interactive
```

### Python API

```python
from compiler.neural.dual_stream_inference import DualStreamDecompilerInference
from compiler.vm.usvm_evaluator import CanonicalUsvmEvaluator

# Supported entry point: the checkpoint plus the deterministic schema lowerer.
# Loading the raw checkpoint alone would return annotations, not an AST.
decompiler = DualStreamDecompilerInference("checkpoints/dual_stream_sup.pt")

# Decompile natural language statute into a schema-valid canonical AST
clause = "Any person responsible for paying to a resident sum exceeding fifty thousand rupees shall deduct 2%."
res = decompiler.decompile_text(clause)

print("Modality:", res["predictedModality"])
print("Classified Spans:", res["classifiedSpansCount"])
print("Synthesized AST:", res["synthesizedAst"]["computation"]["archetype"])

# Third stage: execute the AST in the canonical USVM kernel
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

### Formal DSL Interoperability: Catala Bridge

The MudraForge Decompiler includes a built-in bridge (`compiler.converters.catala_emitter`) that transpiles synthesized Declarative ASTs into literate, idiomatic **Catala source code (`.catala_en`)** according to the ACM ICFP 2021 specification (*Merigoux et al.*).

This enables downstream formal verification, certified compilation, and cross-validation against European and U.S. statutory formalizations:

```python
from compiler.converters.catala_emitter import emit_catala_code, save_catala_file

# Convert compiled decompiler result into Catala literate markdown
catala_source = emit_catala_code(res, title="IRC Section Formalization")
print(catala_source)

# Export directly to file
save_catala_file(res, "output/statute.catala_en", title="IRC Section Formalization")
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

# Or run everything via npm in one command:
npm run test:all
```

---

## Training & Model Alignment

The repository includes the complete pipeline to synthesize statutory training corpora, train the dual-stream transformer from scratch, and align the policy via execution-guided Group Relative Policy Optimization (GRPO):

```bash
# 1. Quick environment smoke test (verifies CUDA, training forward/backward pass, and USVM in <6s)
python train.py --smoke-test

# 2. Synthesize multi-domain training dataset from procedural legal grammar
python train.py --stage dataset --samples 4000 --dataset checkpoints/dataset.pt

# 3. Stage 1: Supervised Disentangled Training (Model B DomainSensor + Model A LogicReasoner)
python train.py --stage supervised --epochs 15 --lr 3e-4 --sup-ckpt checkpoints/dual_stream_sup.pt

# 4. Stage 2: Execution-Guided GRPO Reinforcement Learning (Freezes Model B, tunes LogicReasoner with USVM rewards)
python train.py --stage grpo --epochs 5 --lr 3e-5 --checkpoint-in checkpoints/dual_stream_sup.pt --rl-ckpt checkpoints/dual_stream_rl.pt

# 5. Run complete pipeline end-to-end
python train.py --stage all
```

---

## Repository Structure

```
mudraforge-decompiler/
├── cli.py                         # Root inference command-line interface
├── train.py                       # Unified training and RL alignment CLI
├── pyproject.toml                 # Standard PEP 517/621 packaging metadata
├── requirements.txt               # Python dependencies (torch, numpy, matplotlib, pypdf)
├── package.json                   # Node.js manifest with @mudraforge/statutory dependency
├── package-lock.json              # Pinned Node dependency graph
├── CITATION.cff                   # Citation metadata
├── CONTRIBUTING.md                # Contribution and review rules
├── SECURITY.md                    # Security disclosure policy
├── NOTICE                         # Third-party notices
├── LICENSE                        # Apache 2.0 license
├── README.md                      # Architecture and documentation
├── checkpoints/                   # Pretrained weights (dual_stream_sup.pt, dual_stream_rl.pt)
├── data/                          # Cached benchmark corpora (DeonticBench SARA)
├── examples/
│   └── statute.txt                # Runnable sample clause used by the CLI quickstart
├── compiler/
│   ├── converters/                # Target code emitters & formal legal DSL bridges
│   │   ├── __init__.py
│   │   ├── archetype_strategies.py# Non-canonical AST shape -> canonical pipeline
│   │   ├── catala_emitter.py      # Declarative AST -> literate Catala (.catala_en)
│   │   ├── errors.py              # CatalaLoweringError and shared lowering errors
│   │   └── fact_bindings.py       # Shared USVM fact -> Catala field/type registry
│   ├── neural/                    # 7.9M dual-stream transformer modules
│   │   ├── statutory_neural_parser.py
│   │   ├── logic_reasoner.py      # LogicReasoner stream and dual-stream fusion
│   │   ├── domain_sensor.py       # DomainSensor stream, token classes, abstain head
│   │   ├── tokenizer.py
│   │   ├── inference.py           # synthesize_usvm_ast: span graph -> canonical AST
│   │   ├── dual_stream_inference.py
│   │   └── pdf_ingestion.py
│   ├── training/                  # Synthetic generation & GRPO alignment loops
│   │   ├── dataset_generator.py   # Procedural statutory grammar generator
│   │   ├── train_dual_stream.py   # Disentangled supervised + GRPO training
│   │   └── usvm_rl_environment.py # USVM execution reward oracle
│   └── vm/                        # USVM execution adapter (@mudraforge/statutory)
│       ├── index.js               # Runtime barrel re-exporting @mudraforge/statutory/usvm
│       ├── usvm_pipeline_adapter.js # Pipeline adapter & shadow validation
│       ├── usvm_worker_stdio.js   # Fast persistent IPC bridge
│       └── usvm_evaluator.py      # Python client for USVM execution
├── benchmarks/                    # Formal evaluation suites
│   ├── test_sara_benchmark.py
│   └── test_crossdomain_deonticbench.py
├── scripts/
│   ├── catala_interpreter.cjs      # Pinned Catala interpreter bundle (generated, 10 MB)
│   ├── catala_verify.mjs          # Catala typecheck/interpret harness
│   ├── emit_catala_fixtures.py    # Regenerate checked-in .catala_en golden files
│   ├── fetch_catala_interpreter.mjs # Fetch + SHA-256 verify the interpreter bundle
│   ├── generate_frontier_benchmark_art.py
│   ├── generate_training_charts.py
│   ├── run_all_benchmarks.py
│   └── security_audit.py
├── tests/                         # Unit, cross-engine, and compiler-verified tests
│   ├── test_catala_emitter.py     # Catala emitter bridge unit tests
│   ├── test_decompiler_unit.py
│   ├── test_financial_pipeline_integration.py
│   ├── test_french_tax_catala_verified.py # Catala typecheck + ground-truth values
│   ├── test_french_tax_cross_engine.py    # USVM == Catala, 19 scenarios
│   └── fixtures/
│       ├── financial_pipeline_asts.json
│       └── french_tax_2026/       # 5 executable fixtures + section metadata
│           ├── fr_direct_tax_irpp/
│           ├── fr_corporate_tax_is/
│           ├── fr_social_contributions/
│           ├── fr_vat_tva/
│           ├── fr_property_tax/
│           └── fr_b2b_b2c_rules/  # Section/boundary fixture
├── tools/
│   ├── catala.lock.json           # Catala interpreter URL, mirrors, SHA-256, byte size
│   └── README.md                  # Tool provenance and verification notes
└── docs/                          # Detailed research papers and diaries
    ├── PAPER.md                   # Full research paper
    ├── COMPILER_TRAINING_DIARY.md # Comprehensive 22-entry training diary
    ├── DISTRIBUTION_KIT.md        # Publishing and release artifacts
    ├── training_dashboard.html    # Offline training telemetry dashboard
    └── images/                    # Architecture, ablation, and benchmark figures
```

The French 2026 fixtures are verified twice. `tests/test_french_tax_catala_verified.py` typechecks the emitted Catala and asserts statutory ground-truth values; `tests/test_french_tax_cross_engine.py` runs the same AST through USVM and Catala and requires field-by-field equality across **19 scenarios** covering IRPP, corporate tax, social contributions, VAT, and property tax. Run both with `npm run test:catala:verified` (the interpreter is fetched and hash-checked first).

---

## Technical Paper & Diary

- **Full Scientific Paper**: [docs/PAPER.md](docs/PAPER.md)
- **Chronological Training Diary**: [docs/COMPILER_TRAINING_DIARY.md](docs/COMPILER_TRAINING_DIARY.md)
- **Announcement & Distribution Kit**: [docs/DISTRIBUTION_KIT.md](docs/DISTRIBUTION_KIT.md)

---

## Prior Art, Acknowledgments, and Scope of Novel Claims

To maintain rigorous scientific and intellectual integrity, we explicitly delineate prior inventions from our novel contributions:

### Prior Art and Foundational Works (Credit to Prior Authors)

1. **Computational Law and Rules as Code (RaC)**:
   The paradigm of formalizing legal statutes into declarative computational models builds on foundational scholarship in computational jurisprudence, notably **Meng Weng Wong and Alexis Chun** (Singapore Management University Centre for Computational Law, [Wong & Chun 2021](https://cclaw.smu.edu.sg/)) and **Michael Genesereth and Oliver Goodenough** (Stanford CodeX).
2. **The SARA Benchmark and Tax Law Formalization**:
   The Statutory Reasoning Assessment (SARA) dataset, taxpayer scenario narratives, ground-truth quantitative cases, and the initial manual Prolog formalizations are the original contributions of **Nils Holzenberger, Andrew Blair-Stanek, and Benjamin Van Durme** (Johns Hopkins University and University of Maryland, [NLLP 2020](https://aclanthology.org/2020.nllp-1.1/), [ACL 2021](https://aclanthology.org/2021.findings-acl.239/)). We do not claim authorship of the SARA dataset or its test cases.
3. **DeonticBench and Regulatory Splits**:
   The cross-regulatory expansion into DeonticBench (including airline tariffs, state housing statutes, and immigration case law) and the curated `hard.json` evaluation split are the contributions of **Yilun Dou, Nils Holzenberger, et al.** ([arXiv:2604.04443, 2026](https://arxiv.org/abs/2604.04443), [GitHub Repository](https://github.com/guangyaodou/DeonticBench)).
4. **Domain-Specific Languages for Law (Catala)**:
   The foundational philosophy of compiling statutory law into executable declarative structures rather than interpreting law imperatively is pioneered by **Denis Merigoux, Nicolas Chataing, and Jonathan Protzenko** (*[Catala: A Programming Language for the Law](https://catala-lang.org/)*, [ICFP 2021](https://doi.org/10.1145/3473582)).
5. **Dual-Stream & Decoupled Encoders in Prior Literature**:
   Dual-stream architectures and decoupled representation learning (separating lexical surface sensing from syntactic and relational logic) are established techniques in natural language processing and semantic parsing. We do not claim to have invented dual-stream networks earliest. While developing our decompiler and grappling with catastrophic domain-noun overfitting, discovering that decoupled dual-stream representations existed in established literature served as a vital sanity check. It gave the author the concrete confidence that an architectural separation between a lexical sensor (`DomainSensor`) and a structural logic reasoner (`LogicReasoner`) could decisively solve the generalization paradox in computational law.
6. **Group Relative Policy Optimization (GRPO)**:
   The mathematical formulation of GRPO without critic/value networks originates from **Zhihong Shao, Peiyi Wang, et al.** ([DeepSeek-AI, arXiv:2402.03300, 2024](https://arxiv.org/abs/2402.03300)).
7. **Constrained Viterbi Sequence Decoding**:
   The foundational principles of sequence tagging via transition-constrained Viterbi decoding build on **Andrew Viterbi** ([IEEE Transactions on Information Theory, 1967](https://doi.org/10.1109/TIT.1967.1054010)) and **John Lafferty, Andrew McCallum, and Fernando Pereira** ([ICML 2001](https://repository.upenn.edu/cis_papers/159/)).
8. **LLM-Augmented Backward Chaining for Legal Reasoning (PercepLog)**:
   The PercepLog-0 framework for combining classical backward chaining with LLM reasoning over statutory text is the work of **Parisa Kordjamshidi, Sai Amrit Bulusu, Samer Aslan, Madhavan Seshadri, Leslie Barrett, and Enrico Santus** (Bloomberg and Michigan State University, [EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.2099/)). We do not claim authorship of PercepLog or its results.

### The Decompiler-to-VM Pipeline: How the AST is Formed Before USVM Execution
A central architectural principle of this system is that **neural inference and statutory arithmetic never mix**:
1. **Linguistic Tagging**: Model B (`DomainSensor`) tags named entities, currency spans, and percentages, while Model A (`LogicReasoner`) predicts token-level BIO tags across deontic modalities (`OBLIGATION`, `PROHIBITION`, `PERMISSION`) and conditions.
2. **Relational Arc Resolution**: A biaffine span head predicts dependency arcs connecting arguments (`REL_TAX_BASE_OF`, `REL_APPLIES_TO`, `REL_TEMPORAL_WINDOW`, `REL_PROVISO_TO_EXCEPTION`).
3. **AST Synthesis**: The deterministic lowerer (`synthesize_usvm_ast`) aggregates these extracted parameters into a strongly-typed **Declarative Abstract Syntax Tree (AST)** conforming to the Open Knowledge Format (OKF v0.2). This AST explicitly records threshold modes (`"CLIFF"` vs. `"EXCESS"`), rate fractions, base formulas, and exception carveout gates.
4. **Deterministic USVM Execution**: Only after the AST is completely validated does the **Universal Statutory Virtual Machine (USVM)** in `@mudraforge/statutory` receive it. The USVM executes the AST over the taxpayer's factual profile using exact 64-bit integer BigInt rational math, returning an auditable verdict while enforcing `isAuthorizedToDisburse: false`.

### Novel Contributions of MudraForge (`mudraforge-decompiler`)

We claim the following specific contributions:

1. **The Dual-Stream Transformer Architecture**: A compact 7.9-million-parameter dual-stream architecture (`DomainSensor` || `LogicReasoner`, of which 5.8M is the logic policy) fusing lexical semantic roles with deontic operator embeddings, achieving high parameter efficiency without a large foundation model footprint.
2. **Zero-Shot Neural-to-AST Compilation**: An automated pipeline that translates raw natural language statutory prose into strongly-typed Declarative ASTs (OKF v0.2). Neural span and role annotation is separated from a deterministic schema lowerer; the pipeline reduces, but does not eliminate, the work of formalizing statutes.
3. **The Universal Statutory Virtual Machine (USVM)**: A sandboxed, domain-agnostic virtual machine operating in a JavaScript isolate with 43 universal ALU opcodes, bounded first-order collection quantifiers, 64-bit integer BigInt rational math (zero IEEE-754 drift), and an immutable non-disbursement invariant (`isAuthorizedToDisburse: false`), provided via `@mudraforge/statutory`.
4. **Zero-Shot Cross-Jurisdiction Transfer**: Demonstrating that a neuro-symbolic decompiler trained exclusively on statutory drafting grammar and non-U.S. tax corpora can transfer its span and relation heads zero-shot to U.S. Internal Revenue Code text. The 100% execution figures belong to the symbolic layer and are reported separately.
5. **Cent-Level Forensic Resolution**: Resolving the long-standing discrepancy on the 3 out-of-bounds cases in SARA numeric by mathematically demonstrating the ground-truth integer truncation artifact ($0.74, $0.32, $0.94).

---

## References

1. **Holzenberger, N., Blair-Stanek, A., & Van Durme, B. (2020)**. [*A Dataset for Statutory Reasoning in Tax Law and That Can Be Solved*](https://aclanthology.org/2020.nllp-1.1/). In Proceedings of the Natural Legal Language Processing Workshop (NLLP 2020), pages 1–10. Association for Computational Linguistics.
2. **Holzenberger, N., & Van Durme, B. (2021)**. [*Factoring Statutory Reasoning as Language Understanding and Formal Reasoning*](https://aclanthology.org/2021.findings-acl.239/). In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2707–2717.
3. **Blair-Stanek, A., Holzenberger, N., & Van Durme, B. (2021)**. [*Can GPT-3 Perform Statutory Reasoning?*](https://doi.org/10.5204/lth.1989). Law, Technology and Humans, 3(2):64–81.
4. **Dou, Y., Holzenberger, N., et al. (2026)**. [*DeonticBench: Benchmarking Deontic Reasoning in Language Models Across Regulatory Domains*](https://arxiv.org/abs/2604.04443). arXiv preprint arXiv:2604.04443. [Dataset Repository](https://github.com/guangyaodou/DeonticBench).
5. **Merigoux, D., Chataing, N., & Protzenko, J. (2021)**. [*Catala: A Programming Language for the Law*](https://doi.org/10.1145/3473582). Proceedings of the ACM on Programming Languages, 5(ICFP):1–29. [Project Website](https://catala-lang.org/).
6. **Wong, M. W., & Chun, A. (2021)**. [*Rules as Code (RaC) and Domain Specific Languages for Computational Law*](https://cclaw.smu.edu.sg/). Singapore Management University Centre for Computational Law Research Report.
7. **Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., & Guo, D. (2024)**. [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300). arXiv preprint arXiv:2402.03300.
8. **Lafferty, J., McCallum, A., & Pereira, F. (2001)**. [*Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*](https://repository.upenn.edu/cis_papers/159/). In Proceedings of the Eighteenth International Conference on Machine Learning (ICML 2001), pages 282–289.
9. **Engti, M. (2026)**. [*The Open Knowledge Format (OKF v0.2): A Formal Intermediate Representation for Computational Jurisprudence*](https://github.com/MudraForge/mudraforge-statutory/blob/main/docs/SPECIFICATION.md). MudraForge Technical Report RFC-009.
10. **Kordjamshidi, P., Bulusu, S. A., Aslan, S., Seshadri, M., Barrett, L., & Santus, E. (2026)**. [*PercepLog-0: LLM-Augmented Backward Chaining for Legal Reasoning*](https://aclanthology.org/2026.findings-emnlp.2099/). Findings of the Association for Computational Linguistics: EMNLP 2026, pages 44688-44706.

---

## Publication Roadmap

1. **Phase 1: Open-Source Code & Model Checkpoints (Current)**:
   - Standalone GitHub repository release (`mudraforge-decompiler`) with reproducible SARA / DeonticBench test suites.
   - Dual-Stream neural model checkpoints (`checkpoints/dual_stream_sup.pt` and `checkpoints/dual_stream_rl.pt`).
   - In-process USVM integration via `@mudraforge/statutory`.
2. **Phase 2: Hugging Face Model Hub Release**:
   - Model cards, tokenizer configurations, and safetensors weights published to `MudraForge/mudraforge-decompiler-7.9m`.
   - Interactive Gradio / Space demonstration for real-time statutory decompilation.
3. **Phase 3: Formal arXiv Preprint Submission**:
   - Formal submission of the comprehensive technical paper to arXiv (`cs.AI` / `cs.CL` / `cs.SE`).
4. **Phase 4: Peer-Reviewed Conference Submission**:
   - Submission to major academic venues in computational linguistics and artificial intelligence & law (e.g., ACL / EMNLP / ICAIL 2026).

---

## Citation

If you use `mudraforge-decompiler` in your research or applications, please cite the open-source software repository and technical specification:

```bibtex
@software{mudraforge2026decompiler,
  title={A 7.9M-Parameter Zero-Shot Neuro-Symbolic Decompiler for Computational Law and Statutory Reasoning},
  author={Engti, Mondeep},
  year={2026},
  howpublished={\url{https://github.com/MudraForge/mudraforge-decompiler}},
  note={Open-source neuro-symbolic statutory compiler and USVM runtime}
}
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
