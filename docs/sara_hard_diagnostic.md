# SARA-Hard Compilation Coverage Diagnostic (RFC-012 Step 5)

- **Total Cases Evaluated**: 65 (35 Numeric + 30 Binary)
- **Date**: 2026-11-14 02:28:25 UTC

## 1. Stage Coverage Funnel

| Stage | Combined (65) | SARA-Numeric (35) | SARA-Binary (30) |
| :--- | :---: | :---: | :---: |
| **Neural Annotation** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **Clause Scoping** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **AST Lowering** | **27/65 (41.5%)** | 7/35 (20.0%) | 20/30 (66.7%) |
| **Catala Emitter** | **27/65 (41.5%)** | 7/35 (20.0%) | 20/30 (66.7%) |
| **Catala Typecheck** | **27/65 (41.5%)** | 7/35 (20.0%) | 20/30 (66.7%) |
| **USVM Execution** | **27/65 (41.5%)** | 7/35 (20.0%) | 20/30 (66.7%) |
| **Cross-Engine Parity** | **0/65 (0.0%)** | 0/35 (0.0%) | 0/30 (0.0%) |

## 2. Failure Code Histogram

| Failure Code | Count | Stage | Description |
| :--- | :---: | :---: | :--- |
| `ERR_AMBIGUOUS_REFERENCE` | 26 | `LOWER` | Multiple candidate base operands without explicit disambiguation (Gate A2.4) |
| `ERR_NO_RATE_HEAD` | 9 | `LOWER` | Clause has no computable rate-to-threshold head |
| `ERR_NO_COMPUTATION` | 3 | `LOWER` | Lowerer could not find computable clause head |

## 3. Methodological Note

This diagnostic reports strict compilation coverage across the pipeline without oracle fact mappings.
It contains no accuracy or correctness claim.
