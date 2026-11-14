# SARA-Hard Compilation Coverage Diagnostic (RFC-012 Step 5)

- **Total Cases Evaluated**: 65 (35 Numeric + 30 Binary)
- **Date**: 2026-11-14 01:58:13 UTC

## 1. Stage Coverage Funnel

| Stage | Combined (65) | SARA-Numeric (35) | SARA-Binary (30) |
| :--- | :---: | :---: | :---: |
| **Neural Annotation** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **Clause Scoping** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **AST Lowering** | **10/65 (15.4%)** | 5/35 (14.3%) | 5/30 (16.7%) |
| **Catala Emitter** | **10/65 (15.4%)** | 5/35 (14.3%) | 5/30 (16.7%) |
| **Catala Typecheck** | **10/65 (15.4%)** | 5/35 (14.3%) | 5/30 (16.7%) |
| **USVM Execution** | **10/65 (15.4%)** | 5/35 (14.3%) | 5/30 (16.7%) |
| **Cross-Engine Parity** | **0/65 (0.0%)** | 0/35 (0.0%) | 0/30 (0.0%) |

## 2. Failure Code Histogram

| Failure Code | Count | Stage | Description |
| :--- | :---: | :---: | :--- |
| `ERR_NO_BASE_OPERAND` | 43 | `ANNOTATION` | Evaluation error |
| `ERR_NO_RATE_HEAD` | 9 | `ANNOTATION` | Evaluation error |
| `ERR_NO_COMPUTATION` | 3 | `LOWER` | Lowerer could not find computable clause head |

## 3. Methodological Note

This diagnostic reports strict compilation coverage across the pipeline without oracle fact mappings.
It contains no accuracy or correctness claim.
