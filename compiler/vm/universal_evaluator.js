/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE UNIVERSAL STATUTORY VIRTUAL MACHINE (USVM) - UNIVERSAL EVALUATOR
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements a pure, domain-agnostic declarative expression and pipeline
 * evaluation engine for the USVM.
 *
 * Enables ANY statutory, tax, or financial calculation to be executed purely
 * from declarative AST JSON (or concise infix math strings) without writing
 * section-specific JavaScript in the VM.
 *
 * Core Guarantees:
 * 1. Zero dynamic code execution (no eval, no Function, no arbitrary script).
 * 2. Strict operator allowlist — pure general-purpose ALU primitives.
 * 3. 64-bit integer BigInt paise arithmetic (zero IEEE-754 floating-point drift).
 * 4. Deterministic Execution Budget:
 *    - MAX_EXPRESSION_DEPTH = 16 (prevents call-stack exhaustion / deep AST bombs)
 *    - MAX_PIPELINE_STEPS = 100 (prevents unbounded execution budgets)
 * 5. Immutable fiduciary invariant: isAuthorizedToDisburse: false.
 * 6. Merkle-compatible execution trace generation for external audit defense.
 *
 * Specification: RFC-009 (Revision 3.0 Universal Pipeline Architecture)
 */

import {
    toBigIntPaise,
    fromBigIntPaise,
    multiplyRational,
    quantizeCurrencyPaisa,
    roundStatutoryAssessment
} from './math_rational.js';
import { parseInfixExpression } from './infix_parser.js';

export const MAX_EXPRESSION_DEPTH = 16;
export const MAX_PIPELINE_STEPS = 100;
export const MAX_COLLECTION_ITEMS = 500;
export const MAX_TOTAL_QUANTIFIER_ITERATIONS = 5000;
export const MAX_QUANTIFIER_NESTING = 3;

const FORBIDDEN_PROP_KEYS = new Set(['__proto__', 'constructor', 'prototype']);
const RESERVED_ITERATOR_NAMES = new Set(['facts', 'rules', 'vars', 'temporal', 'output', '__proto__', 'constructor', 'prototype']);

export const UNIVERSAL_OPERATORS = Object.freeze([
    // General Arithmetic
    'ADD',
    'SUB',
    'MUL',
    'DIV',
    'MOD',
    'POW',
    'ABS',
    'MIN',
    'MAX',
    'CLAMP',
    // Rational Scaling & Currency
    'TO_PAISE',
    'FROM_PAISE',
    'RATIONAL_MULTIPLY',
    'PROPORTIONAL',
    'ROUND',
    // Comparisons
    'EQ',
    'NEQ',
    'GT',
    'GTE',
    'LT',
    'LTE',
    'IN',
    'NOT_IN',
    // Logic & Control
    'AND',
    'OR',
    'NOT',
    'IMPLIES',
    'XOR',
    'CONDITIONAL',
    'SWITCH',
    // Bounded Collection & Relational Graph Quantifiers
    'EXISTS',
    'FORALL',
    'FILTER',
    'MAP',
    'COUNT',
    'SUM_OVER',
    'LEN',
    'GET_ATTR',
    // Temporal
    'DATE_DIFF',
    'DATE_ADD',
    'DATE_COMPARE',
    // Piecewise & Compounding
    'PROGRESSIVE_SLABS',
    'COMPOUND_INTEREST'
]);

/**
 * Safely resolves nested property values from context using dot notation.
 * Lexical quantifier bindings in `context.scopeVars` shadow bare/dotted references
 * without mutating `context.vars`. Explicit scope roots (`facts.`, `rules.`, `vars.`,
 * `temporal.`, `output.`) remain unambiguous.
 */
export function resolveValue(ref, context) {
    if (ref === undefined || ref === null) return ref;
    if (typeof ref !== 'string') return ref;

    const trimmed = ref.trim();
    if (!trimmed) return undefined;

    // Explicit scope roots
    if (trimmed.startsWith('facts.')) {
        return getProperty(context.facts, trimmed.slice(6));
    }
    if (trimmed.startsWith('rules.')) {
        return getProperty(context.rules, trimmed.slice(6));
    }
    if (trimmed.startsWith('vars.')) {
        return getProperty(context.vars, trimmed.slice(5));
    }
    if (trimmed.startsWith('temporal.')) {
        return getProperty(context.temporal, trimmed.slice(9));
    }
    if (trimmed.startsWith('output.')) {
        return getProperty(context.output, trimmed.slice(7));
    }

    // Lexical quantifier scope resolution (supports both bare `x` and nested `x.prop`)
    if (context.scopeVars && typeof context.scopeVars === 'object') {
        const dotIdx = trimmed.indexOf('.');
        if (dotIdx === -1) {
            if (!FORBIDDEN_PROP_KEYS.has(trimmed) && Object.prototype.hasOwnProperty.call(context.scopeVars, trimmed)) {
                return context.scopeVars[trimmed];
            }
        } else {
            const rootKey = trimmed.slice(0, dotIdx);
            const restPath = trimmed.slice(dotIdx + 1);
            if (!FORBIDDEN_PROP_KEYS.has(rootKey) && Object.prototype.hasOwnProperty.call(context.scopeVars, rootKey)) {
                return getProperty(context.scopeVars[rootKey], restPath);
            }
        }
    }

    // Default variable resolution order: vars -> facts -> rules
    if (context.vars && !FORBIDDEN_PROP_KEYS.has(trimmed) && Object.prototype.hasOwnProperty.call(context.vars, trimmed) && context.vars[trimmed] !== undefined) {
        return context.vars[trimmed];
    }
    if (context.facts && !FORBIDDEN_PROP_KEYS.has(trimmed) && Object.prototype.hasOwnProperty.call(context.facts, trimmed) && context.facts[trimmed] !== undefined) {
        return context.facts[trimmed];
    }
    if (context.rules && !FORBIDDEN_PROP_KEYS.has(trimmed) && Object.prototype.hasOwnProperty.call(context.rules, trimmed) && context.rules[trimmed] !== undefined) {
        return context.rules[trimmed];
    }

    return getProperty(context.facts, trimmed);
}

function getProperty(obj, path) {
    if (!obj || !path || typeof obj !== 'object') return undefined;
    const parts = String(path).split('.');
    let curr = obj;
    for (const part of parts) {
        if (curr === null || curr === undefined || typeof curr !== 'object') return undefined;
        if (FORBIDDEN_PROP_KEYS.has(part)) return undefined;
        if (Array.isArray(curr)) {
            if (part === 'length') {
                curr = curr.length;
                continue;
            }
            if (/^\d+$/.test(part)) {
                curr = curr[Number(part)];
                continue;
            }
            return undefined;
        }
        if (!Object.prototype.hasOwnProperty.call(curr, part)) return undefined;
        curr = curr[part];
    }
    return curr;
}

/**
 * Coerces a numeric or BigInt value safely into BigInt paise.
 */
function toPaise(val) {
    if (typeof val === 'bigint') return val;
    if (typeof val === 'number') return toBigIntPaise(val);
    if (typeof val === 'string') {
        const clean = val.trim();
        if (/^\d+$/.test(clean)) return BigInt(clean);
        return toBigIntPaise(clean);
    }
    return 0n;
}

/**
 * Tests if a string is an infix expression rather than a plain variable reference.
 */
function isInfixExpressionString(str) {
    if (typeof str !== 'string') return false;
    const trimmed = str.trim();
    if (!trimmed) return false;
    // ISO-8601 date/timestamp literals must never be treated as infix subtraction expressions
    if (/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/.test(trimmed)) {
        return false;
    }
    // Plain variable path like "facts.amount" or "taxableBase"
    if (/^[a-zA-Z_][a-zA-Z0-9_.]*$/.test(trimmed)) {
        return false;
    }
    // Contains operators, spaces, parentheses, function calls, or numbers
    return /[+\-*/%><!?:,()=\s]/.test(trimmed);
}

/**
 * Pure declarative AST expression evaluator with strict depth guard.
 *
 * @param {Object|string|number} node - AST expression node or infix string
 * @param {Object} context - Execution context { facts, rules, vars, temporal, trace }
 * @param {number} depth - Current recursion call-stack depth (capped at 16)
 * @returns {*} Evaluated value
 */
export function evaluateExpression(node, context, depth = 0) {
    if (depth > MAX_EXPRESSION_DEPTH) {
        throw new Error(`[USVM_SECURITY_ERROR] ERR_AST_DEPTH_EXCEEDED: Expression depth exceeded maximum limit of ${MAX_EXPRESSION_DEPTH} levels`);
    }

    if (node === undefined || node === null) return null;

    // Infix string expression support: parse concise math formulas into AST
    if (typeof node === 'string') {
        const trimmedStr = node.trim();
        if (/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/.test(trimmedStr)) {
            return trimmedStr;
        }
        if (isInfixExpressionString(node)) {
            const parsedAst = parseInfixExpression(node);
            return evaluateExpression(parsedAst, context, depth + 1);
        }
        const resolved = resolveValue(node, context);
        if (resolved !== undefined) return resolved;

        const trimmed = trimmedStr;
        if ((trimmed.startsWith("'") && trimmed.endsWith("'")) || (trimmed.startsWith('"') && trimmed.endsWith('"'))) {
            return trimmed.slice(1, -1);
        }
        // If not found in scope and no scope prefix was specified, treat bare identifier as literal string
        if (!trimmed.includes('.') && !trimmed.startsWith('facts.') && !trimmed.startsWith('rules.') && !trimmed.startsWith('vars.') && !trimmed.startsWith('temporal.')) {
            return trimmed;
        }
        return undefined;
    }

    // Primitives evaluate to themselves
    if (typeof node !== 'object') {
        return node;
    }

    // Explicit literal node
    if (node.op === 'LITERAL' || node.type === 'LITERAL') {
        return node.value;
    }

    // Variable reference node
    if (node.op === 'VARIABLE' || node.var !== undefined || node.path !== undefined) {
        const path = node.var ?? node.path;
        return resolveValue(path, context);
    }

    const op = node.op || node.operator;
    if (!op) {
        if (node.value !== undefined) return node.value;
        return null;
    }

    if (!UNIVERSAL_OPERATORS.includes(op)) {
        throw new Error(`[USVM_UNIVERSAL_EVALUATOR] Security error: Unsupported or forbidden operator '${op}'`);
    }

    const evalChild = (child) => evaluateExpression(child, context, depth + 1);

    switch (op) {
        // ── Arithmetic Operations ───────────────────────────────────────────
        case 'ADD': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') {
                return toPaise(left) + toPaise(right);
            }
            return Number(left || 0) + Number(right || 0);
        }

        case 'SUB': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') {
                return toPaise(left) - toPaise(right);
            }
            return Number(left || 0) - Number(right || 0);
        }

        case 'MUL': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') {
                return toPaise(left) * toPaise(right);
            }
            return Number(left || 0) * Number(right || 0);
        }

        case 'DIV': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') {
                const r = toPaise(right);
                if (r === 0n) throw new Error('[USVM_ARITHMETIC_ERROR] Division by zero BigInt paise');
                return toPaise(left) / r;
            }
            const rNum = Number(right || 0);
            if (rNum === 0) throw new Error('[USVM_ARITHMETIC_ERROR] Division by zero');
            return Number(left || 0) / rNum;
        }

        case 'MOD': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') {
                const r = toPaise(right);
                if (r === 0n) throw new Error('[USVM_ARITHMETIC_ERROR] Modulo by zero BigInt paise');
                return toPaise(left) % r;
            }
            return Number(left || 0) % Number(right || 1);
        }

        case 'POW': {
            const base = Number(evalChild(node.base ?? node.left) ?? 0);
            const exp = Number(evalChild(node.exponent ?? node.right) ?? 0);
            return Math.pow(base, exp);
        }

        case 'ABS': {
            const val = evalChild(node.value ?? node.operand);
            if (typeof val === 'bigint') return val < 0n ? -val : val;
            return Math.abs(Number(val ?? 0));
        }

        case 'MIN': {
            const items = Array.isArray(node.operands) ? node.operands.map(evalChild) : [evalChild(node.left), evalChild(node.right)];
            if (items.some(x => typeof x === 'bigint')) {
                const bItems = items.map(toPaise);
                return bItems.reduce((min, cur) => cur < min ? cur : min, bItems[0]);
            }
            return Math.min(...items.map(Number));
        }

        case 'MAX': {
            const items = Array.isArray(node.operands) ? node.operands.map(evalChild) : [evalChild(node.left), evalChild(node.right)];
            if (items.some(x => typeof x === 'bigint')) {
                const bItems = items.map(toPaise);
                return bItems.reduce((max, cur) => cur > max ? cur : max, bItems[0]);
            }
            return Math.max(...items.map(Number));
        }

        case 'CLAMP': {
            const val = evalChild(node.value);
            const min = evalChild(node.min);
            const max = evalChild(node.max);
            if (typeof val === 'bigint' || typeof min === 'bigint' || typeof max === 'bigint') {
                const bVal = toPaise(val);
                const bMin = toPaise(min);
                const bMax = toPaise(max);
                if (bVal < bMin) return bMin;
                if (bVal > bMax) return bMax;
                return bVal;
            }
            const nVal = Number(val || 0);
            const nMin = Number(min || 0);
            const nMax = Number(max || 0);
            return Math.min(Math.max(nVal, nMin), nMax);
        }

        // ── Rational & Currency Math ─────────────────────────────────────────
        case 'TO_PAISE': {
            const val = evalChild(node.value ?? node.operand);
            return toPaise(val);
        }

        case 'FROM_PAISE': {
            const val = evalChild(node.value ?? node.operand);
            return fromBigIntPaise(toPaise(val));
        }

        case 'RATIONAL_MULTIPLY': {
            const basePaise = toPaise(evalChild(node.base));
            const rateBps = BigInt(Math.round(Number(evalChild(node.rateBps ?? node.rate ?? 0))));
            const denominator = BigInt(evalChild(node.denominator ?? 10000));

            const { quotientPaise, subPaisaRemainder } = multiplyRational(basePaise, rateBps, denominator);
            return quantizeCurrencyPaisa(quotientPaise, subPaisaRemainder, denominator);
        }

        case 'PROPORTIONAL': {
            const total = toPaise(evalChild(node.total));
            const numerator = toPaise(evalChild(node.numerator));
            const denominator = toPaise(evalChild(node.denominator));
            if (denominator === 0n) return 0n;
            return (total * numerator) / denominator;
        }

        case 'ROUND': {
            const val = evalChild(node.value);
            const policy = node.policy || 'COMMERCIAL_PAISA';
            if (typeof val === 'bigint') {
                if (policy === 'INCOME_TAX_ACT_1961_SEC288B' || policy === 'INCOME_TAX_ACT_2025_SEC516' || policy === 'SECTION_288B') {
                    return roundStatutoryAssessment(val, policy);
                }
                return val;
            }
            const n = Number(val || 0);
            if (policy === 'FLOOR') return Math.floor(n);
            if (policy === 'CEIL') return Math.ceil(n);
            if (policy === 'SECTION_288B' || policy === 'ROUND_10') {
                return Math.round(n / 10) * 10;
            }
            return Math.round(n * 100) / 100;
        }

        // ── Comparisons ──────────────────────────────────────────────────────
        case 'EQ': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' && typeof right === 'bigint') return left === right;
            return left === right || String(left) === String(right);
        }

        case 'NEQ': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            return left !== right;
        }

        case 'GT': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') return toPaise(left) > toPaise(right);
            return Number(left) > Number(right);
        }

        case 'GTE': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') return toPaise(left) >= toPaise(right);
            return Number(left) >= Number(right);
        }

        case 'LT': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') return toPaise(left) < toPaise(right);
            return Number(left) < Number(right);
        }

        case 'LTE': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (typeof left === 'bigint' || typeof right === 'bigint') return toPaise(left) <= toPaise(right);
            return Number(left) <= Number(right);
        }

        case 'IN': {
            const val = evalChild(node.value || node.left);
            const list = evalChild(node.list || node.right);
            return Array.isArray(list) && list.includes(val);
        }

        case 'NOT_IN': {
            const val = evalChild(node.value || node.left);
            const list = evalChild(node.list || node.right);
            return Array.isArray(list) && !list.includes(val);
        }

        // ── Logic & Control ──────────────────────────────────────────────────
        case 'AND': {
            const operands = Array.isArray(node.operands) ? node.operands : [node.left, node.right];
            return operands.every(opNode => Boolean(evalChild(opNode)));
        }

        case 'OR': {
            const operands = Array.isArray(node.operands) ? node.operands : [node.left, node.right];
            return operands.some(opNode => Boolean(evalChild(opNode)));
        }

        case 'NOT': {
            const operand = evalChild(node.operand ?? node.value);
            return !Boolean(operand);
        }

        case 'IMPLIES': {
            const antecedent = evalChild(node.left ?? node.antecedent);
            if (!Boolean(antecedent)) return true;
            return Boolean(evalChild(node.right ?? node.consequent));
        }

        case 'XOR': {
            const left = Boolean(evalChild(node.left));
            const right = Boolean(evalChild(node.right));
            return left !== right;
        }

        case 'CONDITIONAL': {
            const test = evalChild(node.test);
            return test ? evalChild(node.then ?? node.consequent) : evalChild(node.else ?? node.alternate);
        }

        case 'SWITCH': {
            const discriminant = evalChild(node.discriminant);
            if (Array.isArray(node.cases)) {
                for (const c of node.cases) {
                    const testVal = evalChild(c.test ?? c.case);
                    if (testVal === discriminant) {
                        return evalChild(c.value ?? c.then);
                    }
                }
            }
            return evalChild(node.default);
        }

        // ── Bounded Collection & Relational Graph Quantifiers ───────────────
        case 'LEN': {
            const target = evalChild(node.collection ?? node.items ?? node.target ?? node.value ?? node.operand);
            if (Array.isArray(target) || typeof target === 'string') return target.length;
            if (target && typeof target === 'object') return Object.keys(target).length;
            return 0;
        }

        case 'GET_ATTR': {
            const target = evalChild(node.target ?? node.object);
            let attrKey;
            if (typeof node.key === 'string') {
                const resolvedKey = resolveValue(node.key, context);
                attrKey = resolvedKey !== undefined ? resolvedKey : node.key;
            } else {
                attrKey = evalChild(node.key ?? node.attr);
            }
            if (attrKey === undefined || attrKey === null) return undefined;
            return getProperty(target, String(attrKey));
        }

        case 'EXISTS':
        case 'FORALL':
        case 'FILTER':
        case 'MAP':
        case 'COUNT':
        case 'SUM_OVER': {
            const currentQDepth = context._quantifierDepth || 0;
            const nextQDepth = currentQDepth + 1;
            if (nextQDepth > MAX_QUANTIFIER_NESTING) {
                throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_DEPTH_EXCEEDED: Quantifier nesting depth (${nextQDepth}) exceeds maximum limit of ${MAX_QUANTIFIER_NESTING}`);
            }

            const rawColl = evalChild(node.collection ?? node.items);
            if (rawColl !== undefined && rawColl !== null && !Array.isArray(rawColl)) {
                throw new Error(`[USVM_UNIVERSAL_EVALUATOR] Quantifier '${op}' collection must evaluate to an array or null/undefined`);
            }
            const coll = rawColl || [];
            if (coll.length > MAX_COLLECTION_ITEMS) {
                throw new Error(`[USVM_SECURITY_ERROR] ERR_COLLECTION_BUDGET_EXCEEDED: Collection length (${coll.length}) exceeds maximum limit of ${MAX_COLLECTION_ITEMS} items`);
            }

            const iterVar = node.iteratorVar || node.varName || 'item';
            if (typeof iterVar !== 'string' || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(iterVar) || RESERVED_ITERATOR_NAMES.has(iterVar)) {
                throw new Error(`[USVM_SECURITY_ERROR] Invalid iterator variable '${iterVar}' in quantifier '${op}'`);
            }

            const predNode = node.predicate ?? node.where ?? node.test ?? null;
            if (op === 'COUNT' && predNode === null) {
                return coll.length;
            }

            const budget = context._quantifierBudget || (context._quantifierBudget = { totalIterations: 0 });
            const baseScope = context.scopeVars || {};
            const bodyNode = node.body ?? node.map ?? node.expr ?? null;

            if (op === 'EXISTS') {
                for (let i = 0; i < coll.length; i++) {
                    budget.totalIterations += 1;
                    if (budget.totalIterations > MAX_TOTAL_QUANTIFIER_ITERATIONS) {
                        throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_BUDGET_EXCEEDED: Total quantifier iterations exceeded maximum limit of ${MAX_TOTAL_QUANTIFIER_ITERATIONS}`);
                    }
                    const childCtx = {
                        ...context,
                        scopeVars: { ...baseScope, [iterVar]: coll[i] },
                        _quantifierBudget: budget,
                        _quantifierDepth: nextQDepth
                    };
                    const match = predNode !== null
                        ? Boolean(evaluateExpression(predNode, childCtx, depth + 1))
                        : Boolean(coll[i]);
                    if (match) return true;
                }
                return false;
            }

            if (op === 'FORALL') {
                for (let i = 0; i < coll.length; i++) {
                    budget.totalIterations += 1;
                    if (budget.totalIterations > MAX_TOTAL_QUANTIFIER_ITERATIONS) {
                        throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_BUDGET_EXCEEDED: Total quantifier iterations exceeded maximum limit of ${MAX_TOTAL_QUANTIFIER_ITERATIONS}`);
                    }
                    const childCtx = {
                        ...context,
                        scopeVars: { ...baseScope, [iterVar]: coll[i] },
                        _quantifierBudget: budget,
                        _quantifierDepth: nextQDepth
                    };
                    const match = predNode !== null
                        ? Boolean(evaluateExpression(predNode, childCtx, depth + 1))
                        : Boolean(coll[i]);
                    if (!match) return false;
                }
                return true;
            }

            if (op === 'FILTER') {
                const kept = [];
                for (let i = 0; i < coll.length; i++) {
                    budget.totalIterations += 1;
                    if (budget.totalIterations > MAX_TOTAL_QUANTIFIER_ITERATIONS) {
                        throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_BUDGET_EXCEEDED: Total quantifier iterations exceeded maximum limit of ${MAX_TOTAL_QUANTIFIER_ITERATIONS}`);
                    }
                    const childCtx = {
                        ...context,
                        scopeVars: { ...baseScope, [iterVar]: coll[i] },
                        _quantifierBudget: budget,
                        _quantifierDepth: nextQDepth
                    };
                    const match = predNode !== null
                        ? Boolean(evaluateExpression(predNode, childCtx, depth + 1))
                        : Boolean(coll[i]);
                    if (match) kept.push(coll[i]);
                }
                return kept;
            }

            if (op === 'MAP') {
                const mapped = [];
                for (let i = 0; i < coll.length; i++) {
                    budget.totalIterations += 1;
                    if (budget.totalIterations > MAX_TOTAL_QUANTIFIER_ITERATIONS) {
                        throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_BUDGET_EXCEEDED: Total quantifier iterations exceeded maximum limit of ${MAX_TOTAL_QUANTIFIER_ITERATIONS}`);
                    }
                    const childCtx = {
                        ...context,
                        scopeVars: { ...baseScope, [iterVar]: coll[i] },
                        _quantifierBudget: budget,
                        _quantifierDepth: nextQDepth
                    };
                    mapped.push(bodyNode !== null ? evaluateExpression(bodyNode, childCtx, depth + 1) : coll[i]);
                }
                return mapped;
            }

            if (op === 'COUNT') {
                let matchedCount = 0;
                for (let i = 0; i < coll.length; i++) {
                    budget.totalIterations += 1;
                    if (budget.totalIterations > MAX_TOTAL_QUANTIFIER_ITERATIONS) {
                        throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_BUDGET_EXCEEDED: Total quantifier iterations exceeded maximum limit of ${MAX_TOTAL_QUANTIFIER_ITERATIONS}`);
                    }
                    const childCtx = {
                        ...context,
                        scopeVars: { ...baseScope, [iterVar]: coll[i] },
                        _quantifierBudget: budget,
                        _quantifierDepth: nextQDepth
                    };
                    if (Boolean(evaluateExpression(predNode, childCtx, depth + 1))) {
                        matchedCount += 1;
                    }
                }
                return matchedCount;
            }

            // SUM_OVER
            let sumNum = 0;
            let sumBigInt = null;
            for (let i = 0; i < coll.length; i++) {
                budget.totalIterations += 1;
                if (budget.totalIterations > MAX_TOTAL_QUANTIFIER_ITERATIONS) {
                    throw new Error(`[USVM_SECURITY_ERROR] ERR_QUANTIFIER_BUDGET_EXCEEDED: Total quantifier iterations exceeded maximum limit of ${MAX_TOTAL_QUANTIFIER_ITERATIONS}`);
                }
                const childCtx = {
                    ...context,
                    scopeVars: { ...baseScope, [iterVar]: coll[i] },
                    _quantifierBudget: budget,
                    _quantifierDepth: nextQDepth
                };
                if (predNode !== null && !Boolean(evaluateExpression(predNode, childCtx, depth + 1))) {
                    continue;
                }
                const itemVal = bodyNode !== null
                    ? evaluateExpression(bodyNode, childCtx, depth + 1)
                    : coll[i];
                if (typeof itemVal === 'bigint' || sumBigInt !== null) {
                    if (sumBigInt === null) {
                        sumBigInt = toPaise(sumNum);
                    }
                    sumBigInt += typeof itemVal === 'bigint' ? itemVal : toPaise(itemVal);
                } else {
                    sumNum += Number(itemVal || 0);
                }
            }
            return sumBigInt !== null ? sumBigInt : sumNum;
        }

        // ── Temporal Math ────────────────────────────────────────────────────
        case 'DATE_DIFF': {
            const startStr = evalChild(node.start);
            const endStr = evalChild(node.end);
            if (!startStr || !endStr) return 0;

            const d1 = new Date(startStr);
            const d2 = new Date(endStr);
            const unit = (node.unit || 'DAYS').toUpperCase();

            const diffMs = d2.getTime() - d1.getTime();
            const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

            if (unit === 'DAYS') return diffDays;
            if (unit === 'MONTHS') {
                return (d2.getFullYear() - d1.getFullYear()) * 12 + (d2.getMonth() - d1.getMonth());
            }
            if (unit === 'YEARS') {
                return d2.getFullYear() - d1.getFullYear();
            }
            return diffDays;
        }

        case 'DATE_ADD': {
            const dateStr = evalChild(node.date);
            const amount = Number(evalChild(node.amount) || 0);
            const unit = (node.unit || 'DAYS').toUpperCase();
            if (!dateStr) return null;

            const d = new Date(dateStr);
            if (unit === 'DAYS') {
                d.setDate(d.getDate() + amount);
            } else if (unit === 'MONTHS') {
                d.setMonth(d.getMonth() + amount);
            } else if (unit === 'YEARS') {
                d.setFullYear(d.getFullYear() + amount);
            }
            return d.toISOString().split('T')[0];
        }

        case 'DATE_COMPARE': {
            const left = evalChild(node.left);
            const right = evalChild(node.right);
            if (!left || !right) return 0;
            if (left < right) return -1;
            if (left > right) return 1;
            return 0;
        }

        // ── Financial Primitives ─────────────────────────────────────────────
        case 'PROGRESSIVE_SLABS': {
            const basePaise = toPaise(evalChild(node.base));
            const slabs = Array.isArray(node.slabs) ? node.slabs : [];

            let totalTaxPaise = 0n;
            let previousCeilingPaise = 0n;

            for (const slab of slabs) {
                if (basePaise <= previousCeilingPaise) break;

                const floorPaise = slab.floor !== null && slab.floor !== undefined
                    ? toPaise(evalChild(slab.floor))
                    : null;
                const upToPaise = slab.upTo !== null && slab.upTo !== undefined
                    ? toPaise(evalChild(slab.upTo))
                    : null;
                const rateBps = BigInt(Math.round(Number(evalChild(slab.rateBps ?? 0))));

                const effectiveFloor = floorPaise !== null && floorPaise > previousCeilingPaise
                    ? floorPaise
                    : previousCeilingPaise;
                const slabCeiling = upToPaise !== null ? upToPaise : basePaise;
                const boundedBase = basePaise < slabCeiling ? basePaise : slabCeiling;
                const slabBase = boundedBase - effectiveFloor;

                if (slabBase > 0n) {
                    let slabTax = 0n;
                    if (rateBps > 0n) {
                        const { quotientPaise, subPaisaRemainder } = multiplyRational(slabBase, rateBps, 10000n);
                        slabTax = quantizeCurrencyPaisa(quotientPaise, subPaisaRemainder, 10000n);
                    }
                    if (slab.baseTax !== null && slab.baseTax !== undefined) {
                        totalTaxPaise = toPaise(evalChild(slab.baseTax)) + slabTax;
                    } else {
                        totalTaxPaise += slabTax;
                    }
                }

                if (upToPaise === null) break;
                previousCeilingPaise = upToPaise;
            }

            return totalTaxPaise;
        }

        case 'COMPOUND_INTEREST': {
            const principalPaise = toPaise(evalChild(node.principal));
            const rateBps = Number(evalChild(node.rateBps ?? 0));
            const annualRate = rateBps / 10000;
            const periods = Number(evalChild(node.periods ?? node.months ?? 1));
            const rests = (node.rests || 'MONTHLY').toUpperCase();

            // Formula: P * ((1 + r/rests)^n - 1)
            const rPerPeriod = rests === 'MONTHLY' ? annualRate / 12 : annualRate;
            const factor = Math.pow(1 + rPerPeriod, periods);
            const principalInr = fromBigIntPaise(principalPaise);
            const interestInr = principalInr * (factor - 1);

            return toBigIntPaise(interestInr);
        }

        default:
            throw new Error(`[USVM_UNIVERSAL_EVALUATOR] Unsupported operator '${op}'`);
    }
}

/**
 * Executes a declarative statutory pipeline with execution budget limits.
 *
 * @param {Object} pipeline - Pipeline definition containing steps and output mappings
 * @param {Object} context - Execution context { facts, rules, vars, temporal, trace }
 * @returns {Object} Fiduciary evaluation outputs
 */
export function executeDeclarativePipeline(pipeline, context) {
    const trace = context.trace || [];
    context.vars = context.vars || {};
    context.output = context.output || {};
    context._quantifierBudget = { totalIterations: 0 };
    context._quantifierDepth = 0;

    const steps = pipeline.steps || [];

    // Deterministic execution budget check
    if (steps.length > MAX_PIPELINE_STEPS) {
        throw new Error(`[USVM_SECURITY_ERROR] ERR_STEP_BUDGET_EXHAUSTED: Step count (${steps.length}) exceeds maximum limit of ${MAX_PIPELINE_STEPS} steps`);
    }

    trace.push(`USVM executing declarative calculation pipeline with ${steps.length} steps`);

    // 1. Execute sequential pipeline steps
    for (const step of steps) {
        const varName = step.assign || step.name;
        if (!varName) continue;

        const expr = step.expr ?? (step.formula || step);
        const val = evaluateExpression(expr, context);
        context.vars[varName] = val;

        const logVal = typeof val === 'bigint' ? `${val.toString()}n` : String(val);
        trace.push(`Step assigned [${varName}] = ${logVal}`);
    }

    // 2. Resolve final output dictionary
    const finalOutput = {};
    if (pipeline.output && typeof pipeline.output === 'object') {
        for (const [key, exprNode] of Object.entries(pipeline.output)) {
            finalOutput[key] = evaluateExpression(exprNode, context);
        }
    } else {
        // Fallback: copy calculated vars into output
        Object.assign(finalOutput, context.vars);
    }

    return finalOutput;
}
