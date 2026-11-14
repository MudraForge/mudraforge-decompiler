# SARA-Hard Compilation Coverage Diagnostic (RFC-012 Step 5)

- **Total Cases Evaluated**: 65 (35 Numeric + 30 Binary)
- **Date**: 2026-11-14 03:53:06 UTC

## 1. Stage Coverage Funnel

| Stage | Combined (65) | SARA-Numeric (35) | SARA-Binary (30) |
| :--- | :---: | :---: | :---: |
| **Neural Annotation** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **Clause Scoping** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **AST Lowering** | **53/65 (81.5%)** | 31/35 (88.6%) | 22/30 (73.3%) |
| **Catala Emitter** | **53/65 (81.5%)** | 31/35 (88.6%) | 22/30 (73.3%) |
| **Catala Typecheck** | **53/65 (81.5%)** | 31/35 (88.6%) | 22/30 (73.3%) |
| **USVM Execution** | **53/65 (81.5%)** | 31/35 (88.6%) | 22/30 (73.3%) |
| **Cross-Engine Parity** | **0/65 (0.0%)** | 0/35 (0.0%) | 0/30 (0.0%) |

## 2. Failure Code Histogram

| Failure Code | Count | Stage | Description |
| :--- | :---: | :---: | :--- |
| `ERR_NO_THRESHOLD` | 5 | `ANNOTATION` | Evaluation error |
| `ERR_NO_RATE_HEAD` | 4 | `LOWER` | Clause has no computable rate-to-threshold head |
| `ERR_NO_COMPUTATION` | 3 | `LOWER` | Lowerer could not find computable clause head |

## 3. Methodological Note

This diagnostic reports strict syntactic compilation and cross-engine parity coverage across the pipeline without oracle fact mappings.
It verifies that lowered ASTs are well-formed, typecheck under INRIA's Catala compiler, and evaluate under USVM without runtime exceptions.
It contains no claim of substantive legal correctness of multi-section statutory formulas against external benchmark tax returns.
