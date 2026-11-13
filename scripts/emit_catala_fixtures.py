#!/usr/bin/env python3
"""
Regenerate the checked-in Catala artifacts for the French 2026 fixtures.

Every executable fixture keeps its canonical ``ast.json`` next to the exact
``rule.catala_en`` source emitted from it. The golden file is what a reviewer
reads and what CI typechecks; it is never hand-edited.

Usage:
    python scripts/emit_catala_fixtures.py           # rewrite golden files
    python scripts/emit_catala_fixtures.py --check   # fail if any file is stale
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compiler.converters.catala_emitter import emit_catala_code

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "french_tax_2026"


def render(ast: dict) -> str:
    return emit_catala_code(ast, title=ast["sectionId"]).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if a golden file differs from the emitter",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for ast_path in sorted(FIXTURE_DIR.glob("*/ast.json")):
        ast = json.loads(ast_path.read_text(encoding="utf-8"))
        target = ast_path.with_name("rule.catala_en")
        source = render(ast)

        if args.check:
            current = target.read_text(encoding="utf-8") if target.exists() else None
            if current != source:
                stale.append(target)
            continue

        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(source)
        print(f"wrote {target.relative_to(REPO_ROOT)}")

    if stale:
        for path in stale:
            print(f"stale: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        print("run: python scripts/emit_catala_fixtures.py", file=sys.stderr)
        return 1

    if args.check:
        print("all Catala fixture artifacts are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
