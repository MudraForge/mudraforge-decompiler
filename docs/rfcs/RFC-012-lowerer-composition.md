# RFC-012: Compositional Lowering for Multi-Clause Statutes

- **Status:** Draft. Design only; this change contains no implementation.
- **Date:** 2026-11-14
- **Depends on:** RFC-011 typed-input semantics. RFC-011 is a separate draft and
  is not part of this change. The field name and gate for that contract are
  fixed by this RFC (OQ-1, closed: `inputs`, gated by `typed_inputs`). This
  RFC does not assign an OKF version.
- **Does not change:** the OKF version (remains 0.2), USVM opcodes, checkpoints,
  or the behavior of legacy ASTs.
- **Normative language:** MUST, MUST NOT, SHOULD, MAY (RFC 2119).

---

## 0. Scope

This RFC specifies five gated steps toward a compositional statutory lowerer.
Steps 1 and 2 are implementation-ready. Steps 3 through 5 define the contracts
and acceptance tests; their internal algorithms may be refined without changing
the gates.

The five gates are:

| Step | Capability | Gate |
| --- | --- | --- |
| 1 | One neural AST lowers end-to-end | IRC Section 1 bracket emits Catala and equals USVM |
| 2 | Per-clause scopes and def-use linking | Two-clause text produces two linked computations |
| 3 | One topological SSA walk | Bracket + subtraction + cap compose in one clause |
| 4 | Multi-regime combinators | MIN, MAX, PREEMPT, ACCUMULATE have exact integer semantics |
| 5 | SARA-Hard diagnostic | Compilation coverage is measured without a correctness claim |

No step may be declared complete on the basis of a single passing example. Each
gate names the exact regression set that must also stay green.

---

## 1. Why the current lowerer stops at one rule

The current pipeline is not missing a single feature. It has one architectural
mismatch: the synthesizer flattens a document into a single-rule AST, while the
emitter expects a hand-authored single-rule AST. The symptoms below are
observable in the current tree.

| ID | Symptom | Location | Consequence |
| --- | --- | --- | --- |
| E1 | A presence test is encoded as `NEQ` with `right: null` | `compiler/neural/inference.py:690` (`finite_number_test`); rejected at `compiler/converters/catala_emitter.py:393` | Section 1 cannot reach Catala lowering |
| E2 | Static `TO_PAISE` renders minor units as major units | `catala_emitter.py:620` calls `_money_literal`; `_money_literal` at 121 expects major units; `_render_number` at 691 uses `_money_literal_minor` | Every threshold is 100x too large in Catala |
| E3 | Operand names are guessed at runtime | `build_operand_fact_keys` at `inference.py:650`, `emit_operand_resolution_steps` at 702, registry check at `catala_emitter.py:487` | New fact names fail late or bind to the wrong input |
| E4 | Spans are flattened into document-wide scalars | `inference.py:2607` (excess threshold, longest wins), 2617 (standard rate, first wins), 2628 (cutover, last wins) | A second clause overwrites or loses the first |
| E5 | Only one fallback archetype can run | `if computation is None` at `inference.py:3269, 3390, 3554, 3614, 3668, 3731, 3754, 3782, 3921, 3958, 3996` | Compound rules are cut to the first matching shape |
| E6 | Single-edge logic is duplicated outside the composed path | `_lower_branch_node` at `inference.py:1227` versus branches at 3390, 3554, 3614 | Fixes must be made twice and drift |
| E7 | Clause identity is discarded after annotation | `segment_clauses` at `inference.py:399`; `clauseIdx` is per span only | Cross-clause definitions cannot be linked |

### 1.1 The Section 1 failure chain, in order

The synthesized AST for the bracket clause is:

```json
{
  "steps": [
    {
      "assign": "primaryBasePaisa",
      "expr": {
        "op": "CONDITIONAL",
        "test": {
          "op": "AND",
          "left": {"op": "NEQ", "left": "facts.primaryBasePaisa", "right": null},
          "right": {
            "op": "OR",
            "left": {"op": "GTE", "left": "facts.primaryBasePaisa", "right": 0},
            "right": {"op": "LT", "left": "facts.primaryBasePaisa", "right": 0}
          }
        },
        "then": {"op": "TO_PAISE", "value": "facts.primaryBasePaisa"},
        "else": {"op": "TO_PAISE", "value": "0"}
      }
    },
    {
      "assign": "rawSlabTaxPaisa",
      "expr": {
        "op": "PROGRESSIVE_SLABS",
        "base": "vars.primaryBasePaisa",
        "slabs": [
          {
            "rateBps": 2800,
            "floor": {"op": "TO_PAISE", "value": "3690000"},
            "upTo": {"op": "TO_PAISE", "value": "8915000"},
            "baseTax": {"op": "TO_PAISE", "value": "553500"}
          }
        ]
      }
    },
    {
      "assign": "inBracketRange",
      "expr": {
        "op": "AND",
        "left": {"op": "GT", "left": "vars.primaryBasePaisa", "right": {"op": "TO_PAISE", "value": "3690000"}},
        "right": {"op": "LTE", "left": "vars.primaryBasePaisa", "right": {"op": "TO_PAISE", "value": "8915000"}}
      }
    },
    {
      "assign": "taxAmountPaisa",
      "expr": {
        "op": "CONDITIONAL",
        "test": "vars.inBracketRange",
        "then": "vars.rawSlabTaxPaisa",
        "else": {"op": "TO_PAISE", "value": "0"}
      }
    }
  ]
}
```

Lowering fails in four places, in this order:

1. `NEQ` with `right: null` raises `CatalaLoweringError: Null expression in pipeline`.
2. If the null is removed, `PROGRESSIVE_SLABS` has no branch in the emitter. The
   emitter currently supports `LITERAL`, `NOT`, `MUL`, `DIV`, `ABS`,
   `TO_PAISE`, `FROM_PAISE`, `RATIONAL_MULTIPLY`, `CONDITIONAL`, `ROUND`,
   comparisons, `MIN`, and `MAX`. It has no `PROGRESSIVE_SLABS`, `SUM_OVER`,
   `COUNT`, or `LEN`.
3. If the operator is added naively, `TO_PAISE("3690000")` renders as
   `$3690000.00` instead of `$36900.00`. Verified:

   ```
   _money_literal(3690000)       -> $3690000.00
   _money_literal_minor(3690000) -> $36900.00
   ```

4. If the above are fixed, `facts.primaryBasePaisa` happens to be in the
   20-key registry, so Section 1 binds. The next statute with
   `taxableIncomeUsd` does not: the registry has `taxableIncomeEur` only.

Point 4 matters for the gate. Section 1 must not pass by accidentally using a
legacy key. It must declare its input.

---

## 2. Invariants

These hold for every step. A change that violates one is out of scope, even if
it passes a local test.

- **I1. Exact money.** Canonical money is an integer count of minor units
  (paise or cents). No floating-point arithmetic is permitted in the canonical
  IR. Rounding occurs only at the operator that quantizes to paise, and it uses
  the existing half-up rule.
- **I2. One canonical input per semantic operand.** A v2 AST MUST NOT contain
  alias-probe chains, candidate lists, or fallback tests for the same value.
  Exactly one input id is bound to one value.
- **I3. No null expressions.** The producer MUST NOT emit `null` as an
  expression. If a value is absent, the input signature declares it optional or
  supplies a default. The emitter MUST reject null with a precise diagnostic
  rather than guessing.
- **I4. Legacy is frozen.** ASTs that do not opt into the v2 contract MUST
  produce byte-identical behavior to the current implementation. The legacy
  path remains available until the removal gate in Section 9.
- **I5. Determinism.** Given the same text, spans, and edges, synthesis MUST
  produce the same AST, independent of input order. Set iteration MUST NOT leak
  into output order.
- **I6. Fail closed.** Unknown types, unknown references, ambiguous definitions,
  and cycles MUST return a stable error code. The lowerer MUST NOT choose a
  plausible candidate silently.
- **I7. OKF stays at 0.2.** OKF is Google's specification and its version is not
  ours to advance. The typed-input contract is an internal profile. Its field
  name and gate are OQ-1 and MUST NOT be presented as an OKF version.
- **I8. Cross-engine equality.** Every acceptance scenario that executes MUST
  produce the same canonical result in USVM and Catala, field by field.

---

## 3. Step 1: one neural AST end-to-end

### 3.1 Goal and gate

Given the verbatim IRC Section 1 bracket clause, the pipeline MUST:

1. synthesize an AST with no null expression and one declared input;
2. lower that AST to Catala;
3. typecheck the emitted Catala;
4. execute the same AST in USVM and Catala;
5. return identical results on the scenarios in 3.3.

Gate: A1.1 through A1.7 all pass, and the 194Q quickstart, the 19 French
cross-engine scenarios, and the existing 53-test suite remain green.

### 3.2 Input text and required law

Input, verbatim:

> If taxable income is over $36,900 but not over $89,150, tax shall be
> $5,535, plus 28% of the excess over $36,900.

Canonical paise values:

| Name | Value |
| --- | ---: |
| Lower bound | 3,690,000 paise |
| Upper bound | 8,915,000 paise |
| Base tax | 553,500 paise |
| Rate | 2,800 bps |

Law implemented by this clause alone:

```
raw(income) =
    553500 + quantize((min(income, 8915000) - 3690000) * 2800 / 10000)
    if income > 3690000
    0
    otherwise

tax(income) =
    raw(income)  if 3690000 < income <= 8915000
    0            otherwise
```

The upper "otherwise" is this clause's gate, not a claim about the next bracket.
The scenario table pins that behavior explicitly.

### 3.3 Acceptance scenarios

All amounts are integer paise. The expected tax is the output of `taxAmountPaisa`.

| ID | Taxable income | Raw bracket tax | Expected tax | Note |
| --- | ---: | ---: | ---: | --- |
| S1 | 3,000,000 | 0 | 0 | Below lower bound |
| S2 | 5,000,000 | 920,300 | 920,300 | Mid-bracket |
| S3 | 8,915,000 | 2,016,500 | 2,016,500 | Upper bound inclusive |
| S4 | 3,690,100 | 553,528 | 553,528 | One paise over lower bound |
| S5 | 9,000,000 | 2,016,500 | 0 | Gate excludes above-bracket |

Arithmetic for S2: `(5,000,000 - 3,690,000) * 2800 / 10000 = 366,800`;
`366,800 + 553,500 = 920,300`.

Arithmetic for S4: `100 * 2800 / 10000 = 28`; `553,500 + 28 = 553,528`.

### 3.4 Required AST shape

The input map is named `inputs` and the opt-in gate is the `typed_inputs`
flag (OQ-1, closed). The gate is the presence of a typed, declared input.

The enclosing OKF bundle, if any, continues to declare `okf_version: "0.2"`.
The JSON AST is an internal executable profile, not an OKF schema; its opt-in
gate is OQ-1.

```json
{
  "inputs": {
    "taxableIncome": {
      "type": "MONEY_MINOR",
      "currency": "USD",
      "scale": 100,
      "required": true
    }
  },
  "computation": {
    "pipeline": {
      "steps": [
        {"assign": "basePaisa", "expr": "inputs.taxableIncome"},
        {
          "assign": "rawSlabTaxPaisa",
          "expr": {
            "op": "PROGRESSIVE_SLABS",
            "base": "vars.basePaisa",
            "slabs": [
              {
                "rateBps": 2800,
                "floor": {"op": "TO_PAISE", "value": "3690000"},
                "upTo": {"op": "TO_PAISE", "value": "8915000"},
                "baseTax": {"op": "TO_PAISE", "value": "553500"}
              }
            ]
          }
        },
        {
          "assign": "inBracketRange",
          "expr": {
            "op": "AND",
            "left": {"op": "GT", "left": "vars.basePaisa", "right": {"op": "TO_PAISE", "value": "3690000"}},
            "right": {"op": "LTE", "left": "vars.basePaisa", "right": {"op": "TO_PAISE", "value": "8915000"}}
          }
        },
        {
          "assign": "taxAmountPaisa",
          "expr": {
            "op": "CONDITIONAL",
            "test": "vars.inBracketRange",
            "then": "vars.rawSlabTaxPaisa",
            "else": {"op": "TO_PAISE", "value": "0"}
          }
        }
      ],
      "output": {
        "taxAmountPaisa": "vars.taxAmountPaisa",
        "calculatedTaxUsd": {"op": "FROM_PAISE", "value": "vars.taxAmountPaisa"}
      }
    }
  }
}
```

Requirements:

- The AST MUST NOT contain `facts.` references.
- The AST MUST NOT contain alias candidate lists or `finite_number_test`.
- The AST MUST NOT contain `null` anywhere under `computation`.
- `TO_PAISE` with a digit-string operand is a **typed cast of an already-minor
  value**, not a conversion. This is the USVM `toPaise` contract
  (`universal_evaluator.js:184`): a bare digit string is already minor; a
  JavaScript number is major.
- `PROGRESSIVE_SLABS` floor, ceiling, and base tax are minor-unit digit strings
  wrapped in `TO_PAISE`. This keeps one representation and one cast site.

### 3.5 Producer changes

1. **Remove the presence test from v2 synthesis.** `finite_number_test` and the
   `CONDITIONAL` wrapper around `primaryBasePaisa` MUST NOT be emitted for a v2
   AST. Presence is a boundary concern: the marshaler validates required inputs
   before execution. If a default is needed, it is declared on the input.
2. **Bind one symbol.** The linker (Step 2) or, for Step 1, the single-clause
   resolver MUST produce one canonical input id. `build_operand_fact_keys` is a
   legacy function and MUST NOT run for v2 ASTs.
3. **Keep the slab semantics.** The synthesized `PROGRESSIVE_SLABS` node MUST
   preserve the USVM behavior, including the `baseTax`-only-if-`slabBase > 0`
   rule. Step 1 does not "fix" the bracket law; it makes the existing law
   executable in both engines.
4. **Emit stable ordering.** Steps MUST be emitted in dependency order, with a
   deterministic tie-break by semantic name.

### 3.6 Emitter changes

1. **Static `TO_PAISE`.** If the operand is a digit string (optionally signed),
   render with `_money_literal_minor`. If the operand is a JSON number, retain
   the legacy major-unit rendering for legacy ASTs. The rule is:

   ```text
   TO_PAISE("-?[0-9]+")  -> _money_literal_minor(value)
   TO_PAISE(number)      -> _money_literal(value)      # legacy only
   TO_PAISE(dynamic)     -> identity for money; error otherwise
   ```

   This fixes E2 without changing any legacy fixture, because legacy fixtures
   use numbers or dynamic references for `TO_PAISE`.

2. **Add `PROGRESSIVE_SLABS`.** `_infer` returns `money`. `_render` MUST emit a
   piecewise expression equivalent to the USVM loop:

   ```text
   total = 0
   previous_ceiling = 0
   for each slab:
       floor = max(slab.floor, previous_ceiling) if slab.floor else previous_ceiling
       ceiling = slab.upTo if present else base
       slab_base = min(base, ceiling) - floor
       if slab_base > 0:
           total += slab.baseTax + slab_base * rateBps / 10000
       if slab.upTo is absent: break
       previous_ceiling = slab.upTo
   ```

   Catala may express this with nested `let` bindings and conditionals. The
   emitted source MUST typecheck. It MUST NOT approximate with a single
   multiplication when `baseTax` is present.

3. **Declared inputs.** `_infer_reference` and `_render_reference` MUST accept
   the v2 input namespace and resolve its type from the signature. An undeclared
   input is `ERR_UNDECLARED_INPUT`. A v2 AST that mixes `inputs.*` and `facts.*`
   is rejected.

4. **Null diagnostics.** `ERR_NULL_EXPRESSION` MUST include the node path and
   the enclosing step id. The emitter continues to reject null; it does not
   learn to accept it.

5. **Legacy isolation.** `_FACT_BINDINGS` remains authoritative for legacy
   ASTs only. A v2 AST MUST NOT consult it. A CI assertion fails if a v2 AST
   reaches the legacy registry.

### 3.7 Acceptance tests

- **A1.1 No null.** Walk `computation` and assert no value is `None` and no
  `right`/`left`/`value` is null.
- **A1.2 Emits and typechecks.** `emit_catala_code` succeeds and
  `scripts/catala_verify.mjs` accepts the file.
- **A1.3 No 100x.** The emitted source contains `$36900.00` and `$5535.00`.
  It MUST NOT contain `$3690000.00` or `$553500.00`.
- **A1.4 Engine equality.** For S1 through S5, `taxAmountPaisa` from USVM equals
  `taxAmountPaisa` from Catala.
- **A1.5 Expected values.** For S1 through S5, both engines equal the table in
  3.3.
- **A1.6 No alias probing.** The v2 AST contains exactly one declared input,
  zero `facts.` references, and no candidate-key list.
- **A1.7 Regression.** The 194Q CLI quickstart, the 19 French cross-engine
  scenarios, and the existing 53 tests remain green.

### 3.8 Rollback

The v2 path is flag-gated and default-off. Rollback is setting the flag to
`legacy`; no data migration is required. The legacy path remains the default
until A1.1 through A1.7 pass in CI.

---

## 4. Step 2: per-clause scopes and def-use linking

### 4.1 Goal and gate

A document with multiple clauses MUST compile to multiple scoped computations,
not one flattened rule. A term defined in one clause and used in another MUST
resolve to the defining clause's output, not to a guessed external fact.

Gate: A2.1 through A2.7 pass, and the single-clause 194Q output is unchanged.

### 4.2 Data model

The implementation MAY use dataclasses or equivalent. The required fields are:

```text
SpanRef
  clause_idx: int
  span_idx: int
  role: str
  char_range: tuple[int, int]
  text: str
  confidence: float

EdgeRef
  source: SpanRef
  target: SpanRef
  relation: str
  confidence: float

ClauseScope
  index: int
  char_range: tuple[int, int]
  spans: list[SpanRef]
  edges: list[EdgeRef]
  definitions: list[Symbol]
  uses: list[Use]
  outputs: list[str]

Symbol
  id: str                  # stable, derived from normalized surface text
  kind: INPUT | OUTPUT | TERM | PARAM
  semantic_type: str
  defined_in: int          # clause index
  name: str
  aliases: list[str]

Use
  symbol_id: str | None    # None when unresolved
  used_in: int
  step_id: str
  expected_type: str

LinkResult
  scopes: list[ClauseScope]
  symbols: dict[str, Symbol]
  cross_refs: list[tuple[int, str, int, str]]
  diagnostics: list[Diagnostic]
```

`Diagnostic` carries a stable `code`, a severity, the clause index, and the
offending span or symbol.

### 4.3 Partitioning

1. If every span carries a contiguous `clauseIdx`, use it as authoritative.
2. Otherwise, assign spans to the ranges returned by `segment_clauses` by
   character offset.
3. Reject a span whose range crosses a clause boundary with
   `ERR_SPAN_CROSSES_CLAUSE`.
4. An edge whose endpoints fall in different clauses becomes a candidate
   cross-reference; it is not silently dropped.

### 4.4 Definition extraction

Definitions are extracted in this precedence order. The first matching pattern
wins; lower-precedence patterns do not create duplicate symbols.

1. Explicit term definition: `the term <X> means <expr>`,
   `<X> is defined as <expr>`, `for the purposes of ..., <X> means ...`.
2. Explicit assignment: `the <X> is <amount>` or `the <X> equals <expr>`.
3. Role-derived output: a span whose role is a threshold, standard rate,
   cap, or computation operand AND which is the target of at least one edge in
   the same clause.
4. No definition. The span remains a local operand.

Normalization for matching: lowercase, strip leading determiners, collapse
whitespace, remove trailing punctuation, and singularize a trailing plural `s`
only when the resulting token is not itself a defined term.

### 4.5 Use extraction and resolution

Uses come from three sources:

1. A repeated normalized noun phrase in a later clause.
2. Anaphora: `such <noun>`, `the said <noun>`, `that amount`, `the excess`,
   `the same`.
3. An explicit reference: `the <X>` where `<X>` is already a symbol.

Resolution precedence:

1. Exact normalized-phrase match to one symbol.
2. Unique role-compatible match (same semantic type, compatible role family).
3. External input. The symbol becomes a declared input with an inferred type.

If more than one candidate remains at the same precedence, return
`ERR_AMBIGUOUS_REFERENCE`. If resolution is transitive and returns to the
origin, return `ERR_CYCLIC_DEFINITION`. No candidate list is embedded in the
AST.

### 4.6 SSA naming and overrides

- Every assigned value has one SSA name:
  `clause_<i>.<semanticName>_v<k>`.
- `k` starts at 1 and increments on every rebind. A name MUST NOT be assigned
  twice.
- An exported output is `clause_<i>.<semanticName>` and refers to the highest
  version.
- A substitution (`$18,000 shall be substituted for $4,400`) emits:
  ```text
  clause_1.inWindow = AND(GT(year, 2017), LT(year, 2026))
  clause_1.standardDeductionPaisa_v2 =
      CONDITIONAL(vars.clause_1.inWindow,
                  TO_PAISE("1800000"),
                  vars.clause_0.standardDeductionPaisa_v1)
  ```

- An exception emits `CONDITIONAL(exception, exception_value, prior)`.
- A `notwithstanding` clause is a preemption, not an arithmetic override; it is
  resolved by Step 4.
- Clause order MUST NOT be inferred from span order. It comes from `clauseIdx`
  or character ranges.

### 4.6.1 Encoding rules (as built)

The dotted form above is the *logical* symbol id. The AST MUST encode it so
that both engines can resolve it:

- The USVM resolves `vars.<name>` through a dotted property lookup, so a dot
  inside a variable name would be read as a path segment. The AST therefore
  writes the same identity with an underscore separator:
  `clause_<i>_<semanticName>_v<k>`. The dotted id stays the logical symbol
  id, recorded in the clause scopes.
- The exported output key is the semantic name of the highest version
  (`standardDeductionPaisa`), and its value is the SSA reference of that
  version. A statutory override replaces the export of the term it overrides.
- The clause-scoped profile is opt-in through `clause_scopes=True` on
  `synthesize_usvm_ast` and `decompile_text`. A single-clause statute keeps
  the current computation unchanged.

### 4.7 Acceptance tests

- **A2.1 Independent clauses.** Input: clause 0 imposes 6 percent on wages;
  clause 1 imposes 2 percent on interest. The AST contains two computations
  with rates 600 and 200 bps. Neither overwrites the other.
- **A2.2 Cross-clause definition.** Input: clause 0 defines `eligible amount`
  as the amount exceeding fifty lakh rupees; clause 1 taxes 0.1 percent of the
  eligible amount. Clause 1 references the clause 0 output. There are zero
  `facts.` references and no `facts.eligibleAmount`.
- **A2.3 Substitution.** The Section 63 example in 4.8 produces the expected
  values in and out of the window.
- **A2.4 Ambiguity.** Two definitions normalize to the same symbol and both are
  candidates for one use. Synthesis returns `ERR_AMBIGUOUS_REFERENCE` and emits
  no AST.
- **A2.5 Cycle.** Clause 0 uses a term defined in clause 1, and clause 1 uses a
  term defined in clause 0, with no base case. Synthesis returns
  `ERR_CYCLIC_DEFINITION`.
- **A2.6 Single clause unchanged.** The 194Q example produces the same
  computation as the current path, apart from the input/alias contract.
- **A2.7 Order independence.** Shuffling the span list before synthesis
  produces the same symbol table and the same output values.

### 4.8 Worked example: Section 63 substitution

Clause 0:

> The basic standard deduction is $4,400.

Clause 1:

> For taxable years beginning after December 31, 2017, and before January 1,
> 2026, $18,000 shall be substituted for $4,400.

Expected scoped AST:

```text
clause_0.standardDeductionPaisa_v1 = TO_PAISE("440000")

clause_1.inWindow =
    AND(GT(inputs.taxYear, 2017), LT(inputs.taxYear, 2026))

clause_1.standardDeductionPaisa_v2 =
    CONDITIONAL(
        vars.clause_1.inWindow,
        TO_PAISE("1800000"),
        vars.clause_0.standardDeductionPaisa_v1
    )
```

Expected outputs:

| Tax year | Deduction (paise) |
| ---: | ---: |
| 2017 | 440,000 |
| 2018 | 1,800,000 |
| 2025 | 1,800,000 |
| 2026 | 440,000 |

---

## 5. Step 3: one topological SSA walk

### 5.1 Goal and gate

Replace the `if computation is None` fallback chain with one walk over the
resolved relation DAG. Existing archetype logic becomes node handlers that
return value references; no handler returns a complete AST.

Gate: A3.1 through A3.4 pass, and Steps 1 and 2 remain green.

### 5.2 Handler registry

Each handler receives a node, its resolved neighbors, and an `SSABuilder`, and
returns one value reference.

| Relation or role | Handler | Result |
| --- | --- | --- |
| `REL_RATE_TO_THRESHOLD` | rate-on-base | `RATIONAL_MULTIPLY(base, rateBps)` |
| `REL_SUBTRACTION_BASE` | subtraction | `SUB(left, right)` |
| `REL_CAP_CEILING` | cap | `MIN(value, cap)` |
| `REL_COMBINATOR_MIN` | beneficial min | `MIN(a, b)` |
| `REL_COMBINATOR_MAX` | penal max | `MAX(a, b)` |
| `REL_THRESHOLD_TO_TRANSACTION` | gate | `CONDITIONAL(compare(base, threshold), value, 0)` |
| `REL_BRACKET_NEXT` | slab chain | `PROGRESSIVE_SLABS(...)` |
| `REL_AGGREGATE_SUM` | bounded aggregation | `SUM_OVER(collection, value)` |
| `REL_LOGICAL_AND` / `REL_LOGICAL_OR` | boolean composition | `AND` / `OR` |
| `REL_LOGICAL_NOT` | negation | `NOT` |

Unhandled relations produce a diagnostic and a fail-closed synthesis error. They
MUST NOT be silently dropped.

### 5.3 Algorithm

1. Build a graph of spans and resolved edges within one clause scope.
2. Topologically sort by data dependency, not by span order. Ties break by
   stable semantic name.
3. Visit each node once. Emit one SSA assignment.
4. Cache the value reference for each node so a diamond dependency is emitted
   once.
5. Detect a cycle with `ERR_CYCLIC_DEPENDENCY`.
6. Apply explicit statutory ordering only where the text states it. The default
   order is: base arithmetic, reductions, caps, floor at zero.
7. Return one step list. Delete the `if computation is None` chain only after
   every branch's behavior is represented by a handler.

### 5.3.1 Encoding rules (as built)

- The walk lives in `compiler/neural/ssa_walker.py`. The relation registry is
  a table: `REL_RATE_TO_THRESHOLD`, `REL_CAP_CEILING` and
  `REL_SUBTRACTION_BASE` are registered; a relation without a handler is a
  fail-closed `ERR_UNHANDLED_RELATION`, never a silent drop.
- **A relation label does not decide the sense of an adjustment.** The neural
  annotator labels both `capped at $10,000` and `reduced by a $500 credit` as
  `REL_CAP_CEILING`. The statutory text decides: the handler reads the words
  around the span and lowers a ceiling to `MIN` and a credit to `SUB`. This is
  rule 6 of 5.3 in practice.
- The default order of the composed pipeline is base arithmetic (the amount
  above the threshold), the rate, the reductions, the caps, and the floor at
  zero. Reductions come before caps because a credit reduces the liability
  that the cap then bounds.
- The walk takes over when a clause composes an adjustment
  (`REL_CAP_CEILING` or `REL_SUBTRACTION_BASE`) next to its rate. A clause with
  only a rate and a threshold keeps the current archetype, so the single
  clause regression set is unchanged.
- The `if computation is None` chain is **not** deleted yet. Rule 7 of 5.3
  allows its removal only once every branch of the chain is represented by a
  handler, and the chain still owns archetypes the registry does not cover
  (slab schedules, combinators, aggregates, deontic gates). Until then the
  chain stays and the walk owns the composed clauses of the v2 profile.

### 5.4 Acceptance tests

- **A3.1 Composition.** Input: "Tax equals 28 percent of the excess of gross
  income over $36,900, reduced by a $500 credit, and capped at $10,000."
  Expected:

  | Gross income | Tax |
  | ---: | ---: |
  | 37,000 | 0 |
  | 40,000 | 368 |
  | 50,000 | 3,168 |
  | 100,000 | 10,000 |

- **A3.2 Order independence.** Reversing the edge list produces the same steps
  and values.
- **A3.3 Cycle.** A rate that depends on its own result returns
  `ERR_CYCLIC_DEPENDENCY`.
- **A3.4 One path.** A compound input invokes more than one handler in one step
  list; the AST does not stop at the first matching archetype.

---

## 6. Step 4: multi-regime combinators

### 6.1 Semantics

All combinators operate on integer minor units and return one integer. No
intermediate rounding is permitted.

| Combinator | Semantics | Tie rule |
| --- | --- | --- |
| `BENEFICIAL_MIN` | `min(a, b)` | Ties are equal; pick the lower regime id for determinism |
| `PENAL_MAX` | `max(a, b)` | Ties are equal; pick the lower regime id for determinism |
| `EXCLUSIVE_PREEMPT` | Select the regime with the highest priority | Equal priority is `ERR_PREEMPTION_TIE` unless one regime has an explicit `notwithstanding` |
| `INDEPENDENT_ACCUMULATE` | `a + b` | A regime id may contribute at most once |

Priority for `EXCLUSIVE_PREEMPT` comes from, in order: an explicit
`notwithstanding` or `subject to` relation; the specificity of the regulated
object; the later section number. The chosen source MUST be recorded in the AST
so the decision is auditable.

### 6.2 Acceptance tests

- **A4.1** `BENEFICIAL_MIN(100000, 80000) = 80000`.
- **A4.2** `PENAL_MAX(100000, 180000) = 180000`.
- **A4.3** General 10 percent (priority 1) versus special 6 percent (priority 2)
  under `EXCLUSIVE_PREEMPT` returns the 6 percent liability. Equal priority with
  no `notwithstanding` returns `ERR_PREEMPTION_TIE`.
- **A4.4** `INDEPENDENT_ACCUMULATE(10000, 25000) = 35000`. Applying the same
  regime id twice still returns 35000.
- **A4.5** All four combinators are order-independent and exact at the paise.

---

## 7. Step 5: SARA-Hard diagnostic

### 7.1 Purpose

Measure how far the pipeline gets on the hard splits. This is a compilation
coverage diagnostic, not an accuracy benchmark. The hard splits provide
narrative text and reference Prolog, but no oracle fact mapping for the neural
pipeline. Without that mapping, a right answer cannot be distinguished from a
lucky one, so this step makes no correctness claim.

### 7.2 Metrics

Per case, record the furthest stage reached:

```text
cases_total
annotation_ok        # model returned SUCCESS
scope_ok             # clauses partitioned
lower_ok             # AST non-null and passes the v2 lint
catala_emit_ok       # emit_catala_code succeeded
catala_typecheck_ok  # interpreter accepted the source
usvm_exec_ok         # USVM executed without error
cross_engine_equal   # both engines produced the same canonical output
failure_code         # stable code for the first failed stage
```

Report counts and a histogram of failure codes. Label the summary
`compilation_coverage`. Do not emit a field named `accuracy` or `correctness`.

### 7.3 Acceptance tests

- **A5.1** The diagnostic runs on `sara_numeric/hard.json` (35 cases) and
  `sara_binary/hard.json` (30 cases) without crashing.
- **A5.2** It writes a machine-readable JSON report and a Markdown summary.
- **A5.3** The summary contains no accuracy or correctness field.
- **A5.4** Every failed case has a stable failure code and a stage.

---

## 8. Test matrix

| Level | What it proves | Where |
| --- | --- | --- |
| 0 | SSA builder, linker, and combinators are correct in isolation | New unit tests |
| 1 | The v2 AST lint holds: no null, no aliases, one input | New unit tests |
| 2 | Emitted Catala typechecks | `scripts/catala_verify.mjs` |
| 3 | USVM and Catala agree on every acceptance scenario | New cross-engine tests |
| 4 | Legacy fixtures are unchanged | Existing 53 tests and 19 French scenarios |
| 5 | SARA-Hard coverage is measured and labeled correctly | New diagnostic |

The 19 French scenarios are the regression anchor for I4. The Section 1 bracket
is the anchor for I8. The 194Q clause is the anchor for the single-clause path.

---

## 9. Rollout and gates

| Phase | Change | Default | Exit gate |
| --- | --- | --- | --- |
| 0 | This RFC | n/a | Review accepted |
| 1 | Step 1 | v2 off | A1.1-A1.7 |
| 2 | Step 2 | v2 off for multi-clause | A2.1-A2.7 |
| 3 | Step 3 | v2 on for new ASTs | A3.1-A3.4 |
| 4 | Step 4 | enabled for multi-regime documents | A4.1-A4.5 |
| 5 | Step 5 | diagnostic only | A5.1-A5.4 |

Rules:

- A phase ships only when its gate and the full regression set pass.
- The legacy path stays available until Phase 3 is green and one release has
  shipped with v2 as the default.
- No phase may retrain the model. The checkpoints are frozen.
- No phase may introduce a USVM opcode. If an opcode is required, it needs its
  own RFC and version bump.

---

## 10. Open questions

- **OQ-1. Typed-input field name and gate. CLOSED.** The field name is
  `inputs` and the opt-in gate is the `typed_inputs` flag (Step 1) and the
  `clause_scopes` flag (Step 2). This is an internal compiler contract, not an
  OKF version. OKF remains 0.2.
- **OQ-2. `TO_PAISE` in v2.** Step 1 keeps it as a typed cast for compatibility.
  A later RFC may replace it with a dedicated minor-money literal.
- **OQ-3. Preemption priority.** Whether priority is textual
  (`notwithstanding`), specificity-based, or section-order based needs a
  reviewer decision before Step 4.
- **OQ-4. Non-2-decimal currencies.** Catala `money` is fixed at two decimals.
  JPY and KWD remain out of scope; they fail closed.
- **OQ-5. SARA oracle spans.** If an oracle span mapping is published later, a
  separate accuracy track can be defined. It is not part of Step 5.

---

## 11. Non-goals

- Advancing or renaming the OKF version. OKF stays at 0.2.
- Retraining or replacing the neural checkpoints.
- Adding USVM opcodes.
- Claiming SARA-Hard accuracy without an oracle fact mapping.
- Multi-currency FX conversion.
- Replacing Catala with a different execution target.

---

## 12. References

1. `compiler/neural/inference.py`: `segment_clauses` (399),
   `build_operand_fact_keys` (650), `finite_number_test` (690),
   `emit_operand_resolution_steps` (702),
   `compile_composed_arithmetic_pipeline` (963), `_lower_branch_node` (1227),
   `compile_general_logic_pipeline` (1417), `synthesize_usvm_ast` (2485).
2. `compiler/converters/catala_emitter.py`: `_money_literal` (121),
   `_money_literal_minor` (141), null rejection (393), `TO_PAISE` inference
   (450), `TO_PAISE` rendering (620), legacy registry check (487).
3. `compiler/converters/fact_bindings.py`: the 20-key legacy registry.
4. `node_modules/@mudraforge/statutory/src/usvm/universal_evaluator.js`:
   `toPaise` (184), `PROGRESSIVE_SLABS` (765).
5. `docs/COMPILER_TRAINING_DIARY.md`: checkpoint metrics and validation set.
6. Google Cloud, *Open Knowledge Format (OKF) v0.2*,
   `GoogleCloudPlatform/knowledge-catalog/okf/SPEC.md`.
7. RFC-011: typed-input semantics (field naming deferred).
