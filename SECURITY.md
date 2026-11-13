# Security Policy

MudraForge (Mondeep Engti) is committed to maintaining the highest security and formal correctness standards across the `mudraforge-decompiler` neuro-symbolic compiler pipeline.

Because this compiler generates intermediate Abstract Syntax Trees (ASTs) executed by deterministic legal and financial virtual machines, an AST generation flaw or parser exploit is treated with the highest severity.

---

## Supported Versions

| Version | Supported | Description |
| :--- | :--- | :--- |
| `1.0.x` | Yes | Active production release with dual-stream transformer & USVM integration |

---

## Architectural Invariants & Security Boundaries

`mudraforge-decompiler` enforces non-negotiable security boundaries:

1. **Non-Disbursement Invariant**: Every synthesized AST and execution voucher mandates `isAuthorizedToDisburse: false`. Statutory reasoning never authorizes autonomous funds movement.
2. **Deterministic Rational Precision**: ASTs are evaluated in exact 64-bit BigInt rational math in the Universal Statutory Virtual Machine (`@mudraforge/statutory`), eliminating IEEE-754 floating-point drift.
3. **Execution Sandboxing**: Neural weights do not execute arbitrary code; the model parses natural language strictly into strongly-typed Declarative Abstract Syntax Trees (ASTs).

---

## Reporting a Vulnerability

Please do **NOT** report security vulnerabilities via public GitHub issues.

Send security reports to:
- **Security contact**: [security@mudraforge.in](mailto:security@mudraforge.in)
- **Subject**: `[SECURITY VULNERABILITY] mudraforge-decompiler: <brief description>`

### What to Include:
1. Affected module or file path (e.g. `compiler/neural/inference.py` or `compiler/vm/usvm_pipeline_adapter.js`).
2. Minimal reproducible statutory clause or payload demonstrating the issue.
3. Impact assessment (e.g. DoS, malformed AST emission, invariant escape).

We acknowledge vulnerability reports within 48 hours and provide detailed remediation updates through coordinated disclosure.
