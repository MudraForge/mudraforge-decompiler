/**
 * Catala verification harness.
 *
 * Executes emitted .catala_en sources with the official Catala compiler and
 * interpreter (js_of_ocaml build published by the Catala team at
 * https://assets.catala-lang.org/catala_web_interpreter.js).
 *
 * The interpreter bundle is not committed (it is a ~10 MB generated
 * artifact). tools/catala.lock.json pins its URL, mirrors, size, SHA-256 and
 * both toolchain versions. Fetch and verify it with:
 *   node scripts/fetch_catala_interpreter.mjs
 * Override the location with CATALA_INTERPRETER_JS if needed.
 *
 * Protocol: one JSON request per stdin line:
 *   { "id": ..., "source": "<catala_en text>", "scopes": ["ScopeA", ...] }
 * Response per line:
 *   { "id": ..., "typecheck": { "success", "errors" },
 *     "results": { "ScopeA": { "success", "values", "errors" } }
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import readline from 'node:readline';
import { createRequire } from 'node:module';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_INTERPRETER = path.join(__dirname, 'catala_interpreter.cjs');
const interpreterPath =
    process.env.CATALA_INTERPRETER_JS ??
    (fs.existsSync(DEFAULT_INTERPRETER) ? DEFAULT_INTERPRETER : null);

if (!interpreterPath || !fs.existsSync(interpreterPath)) {
    process.stderr.write(
        'No Catala interpreter found. Run: node scripts/fetch_catala_interpreter.mjs\n' +
            'or set CATALA_INTERPRETER_JS to the official catala_web_interpreter.js bundle.\n',
    );
    process.exit(2);
}
const require = createRequire(import.meta.url);
const catala = require(interpreterPath);

function parseValue(raw) {
    const trimmed = raw.trim();
    if (trimmed === 'true') return true;
    if (trimmed === 'false') return false;
    const money = /^(-?)\$([\d,]+(?:\.\d+)?)$/.exec(trimmed);
    if (money) return Number(`${money[1]}${money[2].replace(/,/g, '')}`);
    if (/^-?[\d,]+(?:\.\d+)?$/.test(trimmed)) return Number(trimmed.replace(/,/g, ''));
    return trimmed;
}

function parseOutputs(output) {
    const values = {};
    // Struct-valued results: "-- field: value".
    for (const m of output.matchAll(/--\s+([a-z_][a-z0-9_]*)\s*:\s*([^\n]+)/g)) {
        values[m[1]] = parseValue(m[2]);
    }
    // Plain scope results: "name = value" (optionally prefixed by a box char).
    for (const line of output.split('\n')) {
        const m = /^[│|]?\s*([a-z_][a-z0-9_]*)\s*=\s*(.+)$/.exec(line);
        if (m) values[m[1]] = parseValue(m[2]);
    }
    return values;
}

function errorText(diagnostics) {
    return (diagnostics || [])
        .filter((d) => d.level === 'error')
        .map((d) => d.message)
        .join('\n');
}

const rl = readline.createInterface({ input: process.stdin });
for await (const line of rl) {
    if (!line.trim()) continue;
    const req = JSON.parse(line);
    const options = {
        files: { 'main.catala_en': req.source },
        main: 'main.catala_en',
        language: 'en',
        outputFormat: 'plain',
    };
    const response = { id: req.id ?? null, typecheck: null, results: {} };
    const tc = catala.typecheck(options);
    response.typecheck = { success: Boolean(tc.success), errors: errorText(tc.diagnostics) };
    if (tc.success) {
        for (const scope of req.scopes || []) {
            const run = catala.interpret({ ...options, scope });
            const values = run.success ? parseOutputs(run.output) : {};
            const parsedNothing = run.success && Object.keys(values).length === 0;
            response.results[scope] = {
                success: Boolean(run.success) && !parsedNothing,
                values,
                errors: parsedNothing
                    ? 'interpreter succeeded but produced no parseable output values'
                    : errorText(run.diagnostics),
            };
        }
    }
    process.stdout.write(`${JSON.stringify(response)}\n`);
}
