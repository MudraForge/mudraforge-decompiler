# SARA-Hard Compilation Coverage Diagnostic (RFC-012 Step 5)

- **Total Cases Evaluated**: 65 (35 Numeric + 30 Binary)
- **Date**: 2026-11-14 05:21:05 UTC

## 1. Stage Coverage Funnel

| Stage | Combined (65) | SARA-Numeric (35) | SARA-Binary (30) |
| :--- | :---: | :---: | :---: |
| **Neural Annotation** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **Clause Scoping** | **65/65 (100.0%)** | 35/35 (100.0%) | 30/30 (100.0%) |
| **AST Lowering** | **36/65 (55.4%)** | 30/35 (85.7%) | 6/30 (20.0%) |
| **Clause Coverage (Lowered / Total)** | **176/954 (18.4%)** | 92/442 (20.8%) | 84/512 (16.4%) |
| **Catala Emitter** | **36/65 (55.4%)** | 30/35 (85.7%) | 6/30 (20.0%) |
| **Catala Typecheck** | **36/65 (55.4%)** | 30/35 (85.7%) | 6/30 (20.0%) |
| **USVM Execution** | **36/65 (55.4%)** | 30/35 (85.7%) | 6/30 (20.0%) |
| **Cross-Engine Parity** | **36/65 (55.4%)** | 30/35 (85.7%) | 6/30 (20.0%) |

## 2. Statute Clause Coverage (Lowered vs. Skipped Clauses)

| Statute Section | Cases | Lowered / Total Clauses | Skipped Clauses | Status |
| :--- | :---: | :---: | :---: | :--- |
| `section1` | 6 | **20/21 (95.2%)** | 1 | `LOWERED` |
| `section151` | 1 | **0/12 (0.0%)** | 12 | `ERR_NO_RATE_HEAD` |
| `section152` | 10 | **0/16 (0.0%)** | 16 | `ERR_NO_COMPUTATION` |
| `section2` | 5 | **0/12 (0.0%)** | 12 | `ERR_NO_COMPUTATION` |
| `section3301` | 4 | **1/2 (50.0%)** | 1 | `LOWERED` |
| `section3306` | 7 | **0/24 (0.0%)** | 24 | `ERR_NO_COMPUTATION` |
| `section63` | 26 | **2/15 (13.3%)** | 13 | `LOWERED` |
| `section68` | 6 | **0/5 (0.0%)** | 5 | `ERR_NO_THRESHOLD` |

## 3. Failure Code Histogram

| Failure Code | Count | Stage | Description |
| :--- | :---: | :---: | :--- |
| `ERR_NO_COMPUTATION` | 22 | `LOWER` | Lowerer could not find computable clause head |
| `ERR_NO_THRESHOLD` | 6 | `LOWER` | Clause specifies excess over threshold without a resolved monetary threshold |
| `ERR_NO_RATE_HEAD` | 1 | `LOWER` | Clause has no computable rate-to-threshold head |

## 4. Methodological Note

This diagnostic reports strict syntactic compilation and cross-engine parity coverage across the pipeline without oracle fact mappings.
It verifies that lowered ASTs are well-formed, typecheck under INRIA's Catala compiler, and evaluate under USVM without runtime exceptions.
It contains no claim of substantive legal correctness of multi-section statutory formulas against external benchmark tax returns.
