# French tax 2026 fixtures

Each directory is one executable statutory rule. Four artifact kinds live here:

| File | Role |
| --- | --- |
| `ast.json` | The canonical, executable AST consumed by the USVM and lowered to Catala. |
| `rule.catala_en` | The exact Catala source emitted from `ast.json`. Generated, never hand-edited. |
| `section.json` | Source metadata, `symbolLegend`, and the statutory scope note. |
| `circulars/*.md` | The legal source text the fixture is derived from. Expected values are traceable to it. |

`fr_b2b_b2c_rules` intentionally has no `ast.json`: it documents the VAT
classification rules that the executable `fr_vat_tva` fixture implements.

## Unit convention

Money is integer minor units (cents) inside the AST; fractional minor units
are invalid.

* A `...Eur` **fact** is a major-unit amount (for example `60000` means
  EUR 60,000.00). The pipeline converts it with `TO_PAISE`/`TO_CENTS`.
* A money **literal** in the pipeline (a slab bound, a ceiling, a threshold) is
  an integer number of cents. It is written as a **digit string**, for example
  `"4250000"`, not `4250000`. The USVM treats a JavaScript number as a
  major-unit amount and multiplies it by 100 when an operator needs paise,
  while a bare digit string is read as an already-minor-unit integer. Catala
  infers the same literal as `money` from its arithmetic context and renders it
  as `$42500.00`.
* Basis-point rates (`rateBps`, `municipalRateBps`, `customerCountryVatRateBps`)
  and counts (`quotientFamilialParts`) are plain scalar integers, never money.
  The USVM bridge encodes them as digit strings for the same reason; the shared
  `compiler/converters/fact_bindings.py` map is the single source of truth for
  which facts are scalar.

This convention is what makes the same `ast.json` produce identical results in
both engines. A mismatch fails `tests/test_french_tax_cross_engine.py`.

## Regenerate and verify

```bash
python scripts/emit_catala_fixtures.py          # rewrite rule.catala_en
python scripts/emit_catala_fixtures.py --check  # fail if a golden file is stale
npm run test:catala:verified                    # typecheck + interpret + cross-engine
```

The Catala interpreter is a pinned generated artifact. Fetch and SHA-256
verify it with `node scripts/fetch_catala_interpreter.mjs`; `tools/catala.lock.json`
records the pin.
