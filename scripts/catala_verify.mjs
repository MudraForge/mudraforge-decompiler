/**
 * Catala verification harness.
 *
 * Executes emitted .catala_en sources with the official Catala compiler and
 * interpreter (js_of_ocaml build published by the Catala team at
 * https://assets.catala-lang.org/catala_web_interpreter.js).
 *
 * The interpreter bundle is not vendored. Point CATALA_INTERPRETER_JS at a
 * local copy of the official bundle to enable compiler-verified tests.
 *
 * Protocol: one JSON request per stdin line:
 *   { "id": ..., "source": "<catala_en text>", "scopes": ["ScopeA", ...] }
 * Response per line:
 *   { "id": ..., "typecheck": { "success", "errors" },
 *     "results": { "ScopeA": { "success", "values", "errors" } } }
 */

import fs from 'node:fs';
import readline from 'node:readline';
import { createRequire } from 'node:module';

const interpreterPath = process.env.CATALA_INTERPRETER_JS;
if (!interpreterPath || !fs.existsSync(interpreterPath)) {
    process.stderr.write('CATALA_INTERPRETER_JS must point to the official catala_web_interpreter.js bundle\n');
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
    for (const m of output.matchAll(/--\s+([a-z_][a-z0-9_]*)\s*:\s*([^\n]+)/g)) {
        values[m[1]] = parseValue(m[2]);
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
            response.results[scope] = {
                success: Boolean(run.success),
                values: run.success ? parseOutputs(run.output) : {},
                errors: errorText(run.diagnostics),
            };
        }
    }
    process.stdout.write(`${JSON.stringify(response)}\n`);
}
