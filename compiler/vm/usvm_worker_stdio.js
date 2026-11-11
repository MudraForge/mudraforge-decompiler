/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE USVM - PERSISTENT NDJSON STDIO WORKER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Keeps ONE single V8/USVM isolate warm for Python bridges (train_grpo.py,
 * usvm_evaluator.py, cli.py --verify) communicating over newline-delimited JSON
 * on stdin/stdout. Eliminates Windows subprocess spawn overhead during RL loops.
 *
 * Supported operations:
 *   - ping
 *   - validate      { ruleAst, mode }
 *   - lifecycle     { ruleAst, options }
 *   - evaluate      { ruleAst, facts, options }
 *   - evaluate_batch { items: [{ ruleAst, facts, options }] }
 */

import readline from 'node:readline';
import {
    validateStatutoryAst,
    evaluateStatutoryAst,
    classifyOkfRuleLifecycle
} from './index.js';

function jsonSafeReplacer(_key, value) {
    if (typeof value === 'bigint') {
        if (value <= BigInt(Number.MAX_SAFE_INTEGER) && value >= BigInt(Number.MIN_SAFE_INTEGER)) {
            return Number(value);
        }
        return value.toString();
    }
    return value;
}

function serializeLine(payload) {
    return JSON.stringify(payload, jsonSafeReplacer);
}

function handleSingleEvaluate(item = {}) {
    const ruleAst = item.ruleAst || item.ast;
    const facts = item.facts || {};
    const options = item.options || { mode: item.mode || 'DEVELOPMENT' };
    return evaluateStatutoryAst(ruleAst, facts, options);
}

function processMessage(req) {
    const id = req?.id ?? null;
    const op = String(req?.op || 'evaluate').trim().toLowerCase();

    try {
        if (op === 'ping') {
            return { id, ok: true, result: { pong: true, runtime: 'USVM_CANONICAL_JS' } };
        }

        if (op === 'validate') {
            const valid = validateStatutoryAst(req.ruleAst || req.ast, req.options || { mode: req.mode });
            return { id, ok: true, result: { valid } };
        }

        if (op === 'lifecycle') {
            const res = classifyOkfRuleLifecycle(req.ruleAst || req.ast, req.options || {});
            return { id, ok: true, result: res };
        }

        if (op === 'evaluate') {
            const voucher = handleSingleEvaluate(req);
            return { id, ok: true, result: voucher };
        }

        if (op === 'evaluate_batch') {
            const items = Array.isArray(req.items) ? req.items : [];
            const results = items.map((item) => {
                try {
                    return { ok: true, result: handleSingleEvaluate(item) };
                } catch (itemErr) {
                    return {
                        ok: false,
                        error: itemErr.message,
                        code: itemErr.details?.code || 'ERR_USVM_EVALUATION_FAILED',
                        isAuthorizedToDisburse: false
                    };
                }
            });
            return { id, ok: true, results };
        }

        return {
            id,
            ok: false,
            error: `Unsupported USVM worker operation '${op}'`,
            code: 'ERR_UNSUPPORTED_OP'
        };
    } catch (err) {
        return {
            id,
            ok: false,
            error: err.message,
            code: err.details?.code || 'ERR_USVM_EXECUTION_FAILED',
            isAuthorizedToDisburse: false
        };
    }
}

const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity
});

rl.on('line', (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    let parsed;
    try {
        parsed = JSON.parse(trimmed);
    } catch (parseErr) {
        process.stdout.write(`${serializeLine({ id: null, ok: false, error: `Invalid JSON: ${parseErr.message}`, code: 'ERR_INVALID_JSON' })}\n`);
        return;
    }

    const response = processMessage(parsed);
    process.stdout.write(`${serializeLine(response)}\n`);
});

rl.on('close', () => {
    process.exit(0);
});
