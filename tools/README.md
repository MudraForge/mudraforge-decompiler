# Catala interpreter pin

The Catala web interpreter is a ~10 MB generated js_of_ocaml bundle. It is not
committed; `catala.lock.json` is the single source of truth for the artifact.

## Lockfile fields

| Field | Meaning |
|-------|---------|
| `catala_version` | Version reported by the bundle's embedded `Catala_utils.Version`. The upstream web build currently reports `dev`. |
| `jsoo_version` | js_of_ocaml version stamped in the bundle header (`version=6.4.1`). This is **not** the Catala version. |
| `url` | Primary upstream artifact URL. |
| `mirrors` | MudraForge-controlled immutable copies, tried after `url`. |
| `sha256` | Expected SHA-256 of the exact artifact. |
| `bytes` | Expected byte length. |
| `node` | Node runtime range used by CI. |

## Local usage

```sh
npm run fetch:catala
npm run check:catala
```

`--check` fails closed on a missing file, wrong size, wrong hash, or invalid
manifest.

## Bumping Catala

1. Prefer a version-tagged release URL over a mutable "latest" URL.
2. Download the new bundle and compute `sha256` and byte length.
3. Update `tools/catala.lock.json` (including both `catala_version` and
   `jsoo_version`).
4. Run `npm run fetch:catala`, then `npm run check:catala`.
5. Run the verified suite with `REQUIRE_CATALA=1`.

## Mirroring

`assets.catala-lang.org` serves a mutable filename. To keep builds working if
upstream rebuilds or removes it, upload the **exact verified bytes** to a
MudraForge-controlled immutable location (GitHub Release asset, S3/R2 object,
or internal artifact store) and add that URL to `mirrors`. The SHA-256 check is
applied to every candidate, so a mirror cannot substitute different bytes.

## CI

`.github/workflows/ci.yml`:

- pins Node 22 LTS;
- caches `scripts/catala_interpreter.cjs` with a key derived from this lockfile;
- runs fetch + `--check` before the test suite;
- sets `REQUIRE_CATALA=1`, so a missing or broken interpreter fails instead of
  silently skipping Catala verification.

Do not add this artifact to `requirements.txt`: it is a Node/JavaScript tool
artifact, not an importable Python package. Do not commit the raw bundle
either; fetch and verify it.
