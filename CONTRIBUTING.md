# Contributing to MudraForge Decompiler

Thank you for your interest in contributing to `mudraforge-decompiler`.

Our mission is to establish a mathematically verifiable, zero-shot neuro-symbolic compiler for computational law, statutory reasoning, and regulatory compliance.

---

## 1. Core Engineering Principles

1. **Rule 18: Branch Isolation**:
   - Zero direct commits to `main`.
   - All functional work, bug fixes, and documentation must occur on short-lived branches (`feat/<name>`, `fix/<name>`, `refactor/<name>`, `docs/<name>`).
2. **Rule 20: Strictly Zero Emojis**:
   - Emojis are strictly forbidden in all codebase files, comments, docstrings, and commit messages.
   - Verified automatically via `python scripts/security_audit.py`.
3. **Deterministic Verification Over Probabilistic Claims**:
   - Neural parser output must synthesize valid Declarative ASTs conforming to the Open Knowledge Format (OKF v0.2).
   - All executions must enforce `isAuthorizedToDisburse: false`.

---

## 2. Development Setup

### Prerequisites
- Python 3.10 or higher
- Node.js 18 or higher
- Git

### Installation
```bash
git clone https://github.com/MudraForge/mudraforge-decompiler.git
cd mudraforge-decompiler

# Install Python requirements
pip install -r requirements.txt

# Install Node.js USVM runtime dependency
npm install
```

---

## 3. Running Verification Suites

Before opening a pull request, ensure all verification suites pass:

```bash
# 1. Run static security & Rule 20 audit
python scripts/security_audit.py

# 2. Run unit tests
python -m unittest tests/test_decompiler_unit.py

# 3. Run cross-domain DeonticBench generalization
python benchmarks/test_crossdomain_deonticbench.py

# 4. Run full SARA benchmark evaluation
python benchmarks/test_sara_benchmark.py
```

---

## 4. Commit Message Standard

We enforce [Conventional Commits](https://www.conventionalcommits.org/):
- `feat: <description>`
- `fix: <description>`
- `docs: <description>`
- `refactor: <description>`
- `test: <description>`
- `chore: <description>`

Zero emojis allowed in commit titles or descriptions.
