/**
 * ═══════════════════════════════════════════════════════════════════════════
 * INFIX EXPRESSION PARSER & AST LOWERER (ZERO DEPENDENCY / ZERO EVAL)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements a safe, recursive-descent precedence-climbing parser that converts
 * human-readable statutory math strings into declarative AST expression DAGs.
 *
 * Solves Greenspun's Tenth Rule: Humans & CAs write concise formulas like:
 *   "max(0, min(currentInvoicePaisa, priorPurchasesPaisa + currentInvoicePaisa - thresholdPaisa))"
 * and this parser compiles it to validated AST nodes with zero eval().
 */

const PRECEDENCE = {
    '||': 1,
    '&&': 2,
    '==': 3,
    '!=': 3,
    '<': 4,
    '<=': 4,
    '>': 4,
    '>=': 4,
    '+': 5,
    '-': 5,
    '*': 6,
    '/': 6,
    '%': 6
};

const OP_MAP = {
    '+': 'ADD',
    '-': 'SUB',
    '*': 'MUL',
    '/': 'DIV',
    '%': 'MOD',
    '>': 'GT',
    '>=': 'GTE',
    '<': 'LT',
    '<=': 'LTE',
    '==': 'EQ',
    '!=': 'NEQ',
    '&&': 'AND',
    '||': 'OR'
};

export function tokenize(str) {
    const tokens = [];
    let i = 0;
    const len = str.length;

    while (i < len) {
        const ch = str[i];

        // Whitespace
        if (/\s/.test(ch)) {
            i++;
            continue;
        }

        // Three-character strict equality operators (lowered cleanly to USVM EQ/NEQ)
        const three = str.slice(i, i + 3);
        if (three === '===' || three === '!==') {
            tokens.push({ type: 'OP', value: three.slice(0, 2) });
            i += 3;
            continue;
        }

        // Two-character operators
        const two = str.slice(i, i + 2);
        if (['>=', '<=', '==', '!=', '&&', '||'].includes(two)) {
            tokens.push({ type: 'OP', value: two });
            i += 2;
            continue;
        }

        // Single-character operators & punctuation
        if ('+-*/%><!?:,()'.includes(ch)) {
            tokens.push({ type: 'OP', value: ch });
            i++;
            continue;
        }

        // Numeric literals
        if (/\d/.test(ch)) {
            let numStr = '';
            while (i < len && /[\d.]/.test(str[i])) {
                numStr += str[i++];
            }
            if (str[i] === 'n') {
                tokens.push({ type: 'BIGINT', value: BigInt(numStr) });
                i++;
            } else {
                tokens.push({ type: 'NUMBER', value: Number(numStr) });
            }
            continue;
        }

        // String literals
        if (ch === "'" || ch === '"') {
            const quote = ch;
            i++;
            let s = '';
            while (i < len && str[i] !== quote) {
                if (str[i] === '\\' && i + 1 < len) i++;
                s += str[i++];
            }
            i++; // consume closing quote
            tokens.push({ type: 'STRING', value: s });
            continue;
        }

        // Identifiers (variables or function names)
        if (/[a-zA-Z_]/.test(ch)) {
            let ident = '';
            while (i < len && /[a-zA-Z0-9_.]/.test(str[i])) {
                ident += str[i++];
            }
            if (ident === 'true') tokens.push({ type: 'BOOLEAN', value: true });
            else if (ident === 'false') tokens.push({ type: 'BOOLEAN', value: false });
            else if (ident === 'null') tokens.push({ type: 'NULL', value: null });
            else tokens.push({ type: 'IDENT', value: ident });
            continue;
        }

        throw new Error(`[USVM_INFIX_PARSER] Unexpected character '${ch}' at position ${i}`);
    }

    tokens.push({ type: 'EOF', value: null });
    return tokens;
}

const RESERVED_ITERATOR_NAMES = new Set([
    'facts', 'rules', 'vars', 'temporal', 'output', '__proto__', 'constructor', 'prototype'
]);

function extractIteratorVar(argNode, fnName) {
    if (!argNode || typeof argNode !== 'object' || typeof argNode.var !== 'string' || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(argNode.var)) {
        throw new Error(`[USVM_INFIX_PARSER] ${fnName}() first argument must be a bare iterator identifier`);
    }
    if (RESERVED_ITERATOR_NAMES.has(argNode.var)) {
        throw new Error(`[USVM_INFIX_PARSER] ${fnName}() iterator variable '${argNode.var}' is reserved`);
    }
    return argNode.var;
}

export function parseInfixExpression(input) {
    if (typeof input !== 'string') return input;
    const trimmed = input.trim();
    if (!trimmed) return null;

    const tokens = tokenize(trimmed);
    let pos = 0;

    function peek() {
        return tokens[pos] || { type: 'EOF', value: null };
    }

    function consume(expectedValue) {
        const tok = peek();
        if (expectedValue && tok.value !== expectedValue) {
            throw new Error(`[USVM_INFIX_PARSER] Expected '${expectedValue}', found '${tok.value}'`);
        }
        pos++;
        return tok;
    }

    function parsePrimary() {
        const tok = peek();

        if (tok.type === 'NUMBER' || tok.type === 'BIGINT' || tok.type === 'BOOLEAN' || tok.type === 'STRING' || tok.type === 'NULL') {
            consume();
            return { op: 'LITERAL', value: tok.value };
        }

        if (tok.type === 'OP' && tok.value === '(') {
            consume('(');
            const expr = parseTernary();
            consume(')');
            return expr;
        }

        if (tok.type === 'OP' && (tok.value === '!' || tok.value === '-')) {
            consume();
            const operand = parsePrimary();
            if (tok.value === '!') {
                return { op: 'NOT', operand };
            }
            return { op: 'SUB', left: { op: 'LITERAL', value: 0 }, right: operand };
        }

        if (tok.type === 'IDENT') {
            const ident = consume().value;

            // Check if function call
            if (peek().type === 'OP' && peek().value === '(') {
                consume('(');
                const args = [];
                if (peek().value !== ')') {
                    while (true) {
                        args.push(parseTernary());
                        if (peek().value === ',') {
                            consume(',');
                        } else {
                            break;
                        }
                    }
                }
                consume(')');

                const fn = ident.toUpperCase();
                if (fn === 'MAX') {
                    if (args.length === 2) return { op: 'MAX', left: args[0], right: args[1] };
                    return { op: 'MAX', operands: args };
                }
                if (fn === 'MIN') {
                    if (args.length === 2) return { op: 'MIN', left: args[0], right: args[1] };
                    return { op: 'MIN', operands: args };
                }
                if (fn === 'CLAMP') {
                    return { op: 'CLAMP', value: args[0], min: args[1], max: args[2] };
                }
                if (fn === 'ABS') {
                    return { op: 'ABS', value: args[0] };
                }
                if (fn === 'TO_PAISE') {
                    return { op: 'TO_PAISE', value: args[0] };
                }
                if (fn === 'FROM_PAISE') {
                    return { op: 'FROM_PAISE', value: args[0] };
                }
                if (fn === 'RATIONAL_MULT' || fn === 'RATIONAL_MULTIPLY') {
                    return { op: 'RATIONAL_MULTIPLY', base: args[0], rateBps: args[1], denominator: args[2] || 10000 };
                }
                if (fn === 'PROPORTIONAL') {
                    return { op: 'PROPORTIONAL', total: args[0], numerator: args[1], denominator: args[2] };
                }
                if (fn === 'DATE_DIFF') {
                    return { op: 'DATE_DIFF', start: args[0], end: args[1], unit: args[2]?.value || 'DAYS' };
                }
                if (fn === 'DATE_ADD') {
                    return { op: 'DATE_ADD', date: args[0], amount: args[1], unit: args[2]?.value || 'DAYS' };
                }
                if (fn === 'ROUND') {
                    return { op: 'ROUND', value: args[0], policy: args[1]?.value || 'COMMERCIAL_PAISA' };
                }
                if (fn === 'IMPLIES') {
                    if (args.length !== 2) throw new Error('[USVM_INFIX_PARSER] implies() expects exactly 2 arguments');
                    return { op: 'IMPLIES', left: args[0], right: args[1] };
                }
                if (fn === 'XOR') {
                    if (args.length !== 2) throw new Error('[USVM_INFIX_PARSER] xor() expects exactly 2 arguments');
                    return { op: 'XOR', left: args[0], right: args[1] };
                }
                if (fn === 'LEN') {
                    if (args.length !== 1) throw new Error('[USVM_INFIX_PARSER] len() expects exactly 1 argument');
                    return { op: 'LEN', collection: args[0] };
                }
                if (fn === 'GET_ATTR') {
                    if (args.length !== 2) throw new Error('[USVM_INFIX_PARSER] get_attr() expects exactly 2 arguments');
                    return { op: 'GET_ATTR', target: args[0], key: args[1] };
                }
                if (fn === 'ANY' || fn === 'EXISTS') {
                    if (args.length !== 3) throw new Error(`[USVM_INFIX_PARSER] ${ident}() expects 3 arguments (var, collection, predicate)`);
                    return { op: 'EXISTS', iteratorVar: extractIteratorVar(args[0], ident), collection: args[1], predicate: args[2] };
                }
                if (fn === 'ALL' || fn === 'FORALL') {
                    if (args.length !== 3) throw new Error(`[USVM_INFIX_PARSER] ${ident}() expects 3 arguments (var, collection, predicate)`);
                    return { op: 'FORALL', iteratorVar: extractIteratorVar(args[0], ident), collection: args[1], predicate: args[2] };
                }
                if (fn === 'FILTER') {
                    if (args.length !== 3) throw new Error('[USVM_INFIX_PARSER] filter() expects 3 arguments (var, collection, predicate)');
                    return { op: 'FILTER', iteratorVar: extractIteratorVar(args[0], ident), collection: args[1], predicate: args[2] };
                }
                if (fn === 'MAP') {
                    if (args.length !== 3) throw new Error('[USVM_INFIX_PARSER] map() expects 3 arguments (var, collection, body)');
                    return { op: 'MAP', iteratorVar: extractIteratorVar(args[0], ident), collection: args[1], body: args[2] };
                }
                if (fn === 'COUNT') {
                    if (args.length === 1) return { op: 'LEN', collection: args[0] };
                    if (args.length === 3) {
                        return { op: 'COUNT', iteratorVar: extractIteratorVar(args[0], ident), collection: args[1], predicate: args[2] };
                    }
                    throw new Error('[USVM_INFIX_PARSER] count() expects 1 argument (collection) or 3 arguments (var, collection, predicate)');
                }
                if (fn === 'SUM_OVER') {
                    if (args.length !== 3 && args.length !== 4) {
                        throw new Error('[USVM_INFIX_PARSER] sum_over() expects 3 or 4 arguments (var, collection, body[, predicate])');
                    }
                    return {
                        op: 'SUM_OVER',
                        iteratorVar: extractIteratorVar(args[0], ident),
                        collection: args[1],
                        body: args[2],
                        ...(args[3] ? { predicate: args[3] } : {})
                    };
                }

                throw new Error(`[USVM_INFIX_PARSER] Unsupported function '${ident}'`);
            }

            // Variable reference
            return { var: ident };
        }

        throw new Error(`[USVM_INFIX_PARSER] Unexpected token '${tok.value}' (type ${tok.type})`);
    }

    function parseBinary(minPrec = 0) {
        let left = parsePrimary();

        while (true) {
            const tok = peek();
            if (tok.type !== 'OP' || !PRECEDENCE[tok.value]) {
                break;
            }

            const prec = PRECEDENCE[tok.value];
            if (prec < minPrec) {
                break;
            }

            const op = consume().value;
            const right = parseBinary(prec + 1);

            const mappedOp = OP_MAP[op];
            left = { op: mappedOp, left, right };
        }

        return left;
    }

    function parseTernary() {
        const condition = parseBinary(0);

        if (peek().type === 'OP' && peek().value === '?') {
            consume('?');
            const thenBranch = parseTernary();
            consume(':');
            const elseBranch = parseTernary();

            return {
                op: 'CONDITIONAL',
                test: condition,
                then: thenBranch,
                else: elseBranch
            };
        }

        return condition;
    }

    const ast = parseTernary();
    if (pos < tokens.length - 1) {
        throw new Error(`[USVM_INFIX_PARSER] Unexpected trailing token '${peek().value}'`);
    }

    return ast;
}
