/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE UNIVERSAL STATUTORY VIRTUAL MACHINE (USVM) - AST SCHEMA VALIDATOR
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements strict, typed schema validation and operator allowlists for
 * declarative statutory AST packages. Rejects dynamic code execution, unknown
 * operators, and unconstrained arithmetic.
 *
 * Specification: RFC-009 (Revision 2.1)
 */

import {
    UNIVERSAL_OPERATORS,
    MAX_EXPRESSION_DEPTH as EVAL_MAX_EXPRESSION_DEPTH,
    MAX_PIPELINE_STEPS as EVAL_MAX_PIPELINE_STEPS,
    MAX_QUANTIFIER_NESTING as EVAL_MAX_QUANTIFIER_NESTING
} from '../universal_evaluator.js';

export const EXECUTION_MODES = Object.freeze([
    'DEVELOPMENT',
    'PRODUCTION'
]);

export const OKF_LIFECYCLE_STATUSES = Object.freeze([
    'OKF_VERIFIED_ACTIVE',
    'OKF_AMENDMENT_STALE_DETECTED',
    'OKF_UNREGISTERED_NEW'
]);

export const ALLOWED_CITATION_DOMAINS = Object.freeze([
    '.gov.in',
    '.rbi.org.in',
    '.mca.gov.in',
    '.icai.org',
    'indiankanoon.org',
    '.irs.gov'
]);

export const SUPPORTED_NODE_TYPES = Object.freeze([
    'PREDICATE',
    'TEMPORAL_VALIDITY',
    'SNAPSHOT_VERIFY',
    'THRESHOLD_TEST',
    'DELTA',
    'SELECT',
    'OVERRIDE',
    'PRECEDENCE',
    'ARITHMETIC',
    'QUANTIZATION',
    'ASSESSMENT_ROUND',
    'DECLARATIVE_PIPELINE',
    'PIPELINE_STEP',
    'DEONTIC_STATE_GATE',
    'STATUS_DISQUALIFICATION_GATE',
    'CORPORATE_APPROVAL_GATE',
    'PROCEDURAL_NOTICE_GATE',
    'TEMPORAL_WINDOW_GATE',
    'BOOLEAN_PROHIBITION_GATE',
    'SUBJECTIVE_EVIDENCE_GATE',
    'ACCOUNTING_RECOGNITION_GATE',
    'ACCOUNTING_ELECTION_CARVEOUT',
    'MEASUREMENT_VALUATION_RULE',
    'DISCLOSURE_PRESENTATION_OBLIGATION',
    'TRANSACTIONAL_WITHHOLDING',
    'GENERAL_RULE_GATE'
]);

export const TYPED_OPERATORS = Object.freeze([
    'EQ',
    'NEQ',
    'IN',
    'NOT_IN',
    'GT',
    'GTE',
    'LT',
    'LTE',
    'IS_TRUE',
    'IS_FALSE',
    'EXISTS',
    'NOT_EXISTS'
]);

export const SYMBOL_TYPES = Object.freeze([
    'MONETARY_QUANTUM',
    'PERCENTAGE_RATE',
    'TEMPORAL_DURATION',
    'EXEMPTION_CONDITION',
    'DOCUMENT_VERIFICATION',
    'SUBJECTIVE_CRITERION_EVALUATION',
    'ENTITY_CLASSIFICATION',
    'PREREQUISITE_STATUS',
    'BOOLEAN_FLAG'
]);

export const SYMBOL_DATA_TYPES = Object.freeze([
    'CURRENCY_PAISA',
    'CURRENCY_CENTS',
    'CURRENCY_MINOR_UNITS',
    'PERCENT',
    'NUMBER',
    'BOOLEAN',
    'DOCUMENT_ID',
    'CATEGORICAL_STRING'
]);

export const TRIGGER_POLICIES = Object.freeze([
    'EARLIER_OF',
    'PAYMENT_ONLY',
    'DEBIT_OR_RECEIPT',
    'TRANSACTION_DATE',
    'INVOICE_DATE',
    'DUE_DATE',
    'FISCAL_YEAR_END'
]);

export const MATH_ARCHETYPES = Object.freeze([
    'EXCESS_OVER_THRESHOLD',
    'CLIFF_THRESHOLD',
    'FLAT_RATE',
    'PROGRESSIVE_SLAB',
    'GROSS_UP_FACTOR',
    'COMPOUND_INTEREST_MONTHLY_RESTS',
    'MSME_FISCAL_BOUNDARY_43BH',
    'ENTERPRISE_CLASSIFICATION_MSME',
    'PROPORTIONAL_REVERSAL_180_DAYS',
    'SUPPLIER_NON_FILING_RULE37A',
    'COMMON_CREDIT_APPORTIONMENT_RULE42',
    'CAPITAL_GOODS_60_MONTH_RULE43',
    'DTAA_TREATY_TAX_HAVEN_MATRIX',
    'TRANSFER_PRICING_PERCENTILE_ALP',
    'VOTING_CONTINUITY_PERCENTAGE',
    'TAX_CLEARANCE_VOID_TRANSFER_SEC281',
    'CORPORATE_TAX_MAT_VS_CONCESSIONAL',
    'SALARY_SLAB_PROGRESSIVE_TDS',
    'NON_FILER_PENAL_RATE_206AB',
    'DEFERRED_TAX_TIMING_DIFFERENCE',
    'BLOCKED_CREDIT_GATE_17_5',
    'ITC_ELIGIBILITY_GATE_16_2',
    'RCM_LIABILITY_EVALUATION_9_3',
    'COMPOSITE_MIXED_SUPPLY_SEC8',
    'COMPOSITION_LEVY_SEC10',
    'CREDIT_NOTE_TIMELINES_SEC34',
    'EXPORT_LUT_ZERO_RATED_SEC16',
    'EINVOICING_MANDATE_RULE48_4',
    'EWAYBILL_CONSIGNMENT_RULE138',
    'GST_TDS_DEDUCTION_SEC51',
    'MCA_MSME_1_AGING_BIFURCATION',
    'MSME_DISCLOSURES_SEC22',
    'LEASE_AMORTIZATION_INDAS116',
    'REVENUE_ALLOCATION_INDAS115',
    'PROVISIONS_CONTINGENCY_INDAS37',
    'FX_SPOT_REVALUATION_INDAS21_109',
    'EXPECTED_CREDIT_LOSS_INDAS109',
    'FOREIGN_TAX_CREDIT_RELIEF',
    'SLUMP_SALE_NET_WORTH_50B',
    'BUSINESS_COMBINATIONS_INDAS103',
    'IMPAIRMENT_TESTING_INDAS36',
    'SHARE_BASED_PAYMENTS_INDAS102',
    'CASH_FLOWS_INDAS7',
    'PPE_DEPRECIATION_INDAS16',
    'EMPLOYEE_BENEFITS_INDAS19',
    'INVENTORY_VALUATION_INDAS2',
    'INVESTMENTS_ASSOCIATES_INDAS28',
    'EARNINGS_PER_SHARE_INDAS33',
    'INTANGIBLES_CAPITALIZATION_INDAS38',
    'CONSOLIDATION_NCI_INDAS110',
    'FAIR_VALUE_HIERARCHY_INDAS113',
    'TDS_VDA_CRYPTO_194S',
    'TDS_ONLINE_GAMING_194BA',
    'TDS_CASH_WITHDRAWAL_194N',
    'TDS_LIFE_INSURANCE_194DA',
    'TDS_MUTUAL_FUNDS_194K',
    'TCS_REMITTANCE_LRS_206C1G',
    'GST_TCS_ECOMMERCE_SEC52',
    'DECLARATIVE_PIPELINE',
    'UNIVERSAL_PIPELINE',
    'UNIVERSAL_EXPRESSION',
    'DEONTIC_STATE_GATE',
    'STATUS_DISQUALIFICATION_GATE',
    'CORPORATE_APPROVAL_GATE',
    'PROCEDURAL_NOTICE_GATE',
    'TEMPORAL_WINDOW_GATE',
    'BOOLEAN_PROHIBITION_GATE',
    'SUBJECTIVE_EVIDENCE_GATE',
    'ACCOUNTING_RECOGNITION_GATE',
    'ACCOUNTING_ELECTION_CARVEOUT',
    'MEASUREMENT_VALUATION_RULE',
    'DISCLOSURE_PRESENTATION_OBLIGATION',
    'FRACTIONAL_RATIO_ACCRUAL',
    'BOOLEAN_BINARY_GATE',
    'EXEMPTION_THRESHOLD_GATE',
    'STATUTORY_PREDICATE_EVALUATION',
    'PURE_LOGIC_PIPELINE',
    'GENERAL_RULE_GATE'
]);

export const ROUNDING_POLICIES = Object.freeze([
    'COMMERCIAL_PAISA',
    'INCOME_TAX_ACT_1961_SEC288B',
    'INCOME_TAX_ACT_2025_SEC516',
    'NONE'
]);

export class StatutoryAstValidationError extends Error {
    constructor(message, details = {}) {
        super(`[USVM_AST_VALIDATION_ERROR] ${message}`);
        this.name = 'StatutoryAstValidationError';
        this.details = details;
    }
}

export const MAX_EXPRESSION_DEPTH = EVAL_MAX_EXPRESSION_DEPTH ?? 16;
export const MAX_PIPELINE_STEPS = EVAL_MAX_PIPELINE_STEPS ?? 100;
export const MAX_QUANTIFIER_NESTING = EVAL_MAX_QUANTIFIER_NESTING ?? 3;

const QUANTIFIER_OPS = new Set(['EXISTS', 'FORALL', 'FILTER', 'MAP', 'COUNT', 'SUM_OVER']);
const RESERVED_ITERATOR_NAMES = new Set(['facts', 'rules', 'vars', 'temporal', 'output', '__proto__', 'constructor', 'prototype']);

const EXPRESSION_CHILD_KEYS = Object.freeze([
    'left', 'right', 'operand', 'value', 'base', 'min', 'max',
    'principal', 'rateBps', 'rate', 'denominator', 'total', 'numerator',
    'periods', 'months', 'discriminant', 'default', 'start', 'end', 'date', 'amount',
    'test', 'then', 'else', 'consequent', 'alternate',
    'collection', 'items', 'predicate', 'where', 'body', 'map', 'expr',
    'target', 'object', 'key', 'attr', 'list', 'antecedent'
]);

function forEachExpressionChild(expr, visitChild) {
    if (!expr || typeof expr !== 'object') return;

    for (const key of EXPRESSION_CHILD_KEYS) {
        const child = expr[key];
        if (child && typeof child === 'object') {
            visitChild(child);
        }
    }

    if (Array.isArray(expr.operands)) {
        for (const op of expr.operands) {
            if (op && typeof op === 'object') visitChild(op);
        }
    }
    if (Array.isArray(expr.cases)) {
        for (const c of expr.cases) {
            if (c && typeof c === 'object') {
                if (c.test && typeof c.test === 'object') visitChild(c.test);
                if (c.case && typeof c.case === 'object') visitChild(c.case);
                if (c.value && typeof c.value === 'object') visitChild(c.value);
                if (c.then && typeof c.then === 'object') visitChild(c.then);
            }
        }
    }
    if (Array.isArray(expr.slabs)) {
        for (const slab of expr.slabs) {
            if (slab && typeof slab === 'object') {
                if (slab.upTo && typeof slab.upTo === 'object') visitChild(slab.upTo);
                if (slab.floor && typeof slab.floor === 'object') visitChild(slab.floor);
                if (slab.rateBps && typeof slab.rateBps === 'object') visitChild(slab.rateBps);
                if (slab.baseTax && typeof slab.baseTax === 'object') visitChild(slab.baseTax);
            }
        }
    }
}

export function validateNodeDepth(expr, depth = 0) {
    if (depth > MAX_EXPRESSION_DEPTH) {
        throw new StatutoryAstValidationError(
            `Expression depth exceeded maximum limit of ${MAX_EXPRESSION_DEPTH} levels (ERR_AST_DEPTH_EXCEEDED)`,
            { depth, limit: MAX_EXPRESSION_DEPTH }
        );
    }
    forEachExpressionChild(expr, (child) => validateNodeDepth(child, depth + 1));
}

export function validateDeepOfficialCitations(officialCitations) {
    if (!Array.isArray(officialCitations) || officialCitations.length === 0) {
        return {
            valid: false,
            normalizedUrls: [],
            reason: 'PRODUCTION mode requires at least one verified officialCitations entry.'
        };
    }

    const normalizedUrls = [];
    for (const cit of officialCitations) {
        const rawUrl = typeof cit === 'string'
            ? cit.trim()
            : String(cit?.canonicalDocumentUrl || cit?.officialUrl || cit?.url || '').trim();

        if (!rawUrl) {
            return {
                valid: false,
                normalizedUrls,
                reason: 'Citation entry is missing canonicalDocumentUrl / url.'
            };
        }

        if (!rawUrl.startsWith('https://')) {
            return {
                valid: false,
                normalizedUrls,
                reason: `Insecure citation protocol rejected (HTTPS strictly required): ${rawUrl}`
            };
        }

        let parsed;
        try {
            parsed = new URL(rawUrl);
        } catch {
            return {
                valid: false,
                normalizedUrls,
                reason: `Malformed citation URL: ${rawUrl}`
            };
        }

        const host = parsed.hostname.toLowerCase();
        const isDomainAllowed = ALLOWED_CITATION_DOMAINS.some((allowed) => {
            const cleanAllowed = allowed.startsWith('.') ? allowed.slice(1) : allowed;
            return host === cleanAllowed || host.endsWith(`.${cleanAllowed}`);
        });

        if (!isDomainAllowed) {
            return {
                valid: false,
                normalizedUrls,
                reason: `Citation domain '${host}' is not on the whitelisted government/regulatory domains.`
            };
        }

        const cleanPath = parsed.pathname.replace(/^\/+|\/+$/g, '');
        if (!cleanPath && !parsed.search) {
            return {
                valid: false,
                normalizedUrls,
                reason: `Generic portal root URL is forbidden in PRODUCTION mode (deep canonical document path required): ${rawUrl}`
            };
        }

        normalizedUrls.push(rawUrl);
    }

    return {
        valid: true,
        normalizedUrls
    };
}

function validateUniversalExpressionOperators(expr, contextLabel = 'expression', depth = 0, qDepth = 0) {
    if (depth > MAX_EXPRESSION_DEPTH) {
        throw new StatutoryAstValidationError(
            `Expression depth exceeded maximum limit of ${MAX_EXPRESSION_DEPTH} levels (ERR_AST_DEPTH_EXCEEDED)`,
            { depth, limit: MAX_EXPRESSION_DEPTH, contextLabel }
        );
    }
    if (!expr || typeof expr !== 'object') return;
    if (expr.op && expr.op !== 'LITERAL' && expr.op !== 'VARIABLE' && !UNIVERSAL_OPERATORS.includes(expr.op)) {
        throw new StatutoryAstValidationError(
            `Operator '${expr.op}' in step '${contextLabel}' not in universal allowlist`,
            { expr, allowed: UNIVERSAL_OPERATORS }
        );
    }

    let nextQDepth = qDepth;
    if (expr.op && QUANTIFIER_OPS.has(expr.op)) {
        nextQDepth = qDepth + 1;
        if (nextQDepth > MAX_QUANTIFIER_NESTING) {
            throw new StatutoryAstValidationError(
                `Quantifier nesting depth exceeded maximum limit of ${MAX_QUANTIFIER_NESTING} levels (ERR_QUANTIFIER_DEPTH_EXCEEDED)`,
                { code: 'ERR_QUANTIFIER_DEPTH_EXCEEDED', quantifierDepth: nextQDepth, limit: MAX_QUANTIFIER_NESTING, contextLabel }
            );
        }
        const iterVar = expr.iteratorVar ?? expr.varName;
        if (iterVar !== undefined) {
            if (typeof iterVar !== 'string' || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(iterVar) || RESERVED_ITERATOR_NAMES.has(iterVar)) {
                throw new StatutoryAstValidationError(
                    `Invalid quantifier iteratorVar '${iterVar}' in '${contextLabel}'`,
                    { code: 'ERR_INVALID_ITERATOR_VAR', iteratorVar, contextLabel }
                );
            }
        }
    }

    forEachExpressionChild(expr, (child) => validateUniversalExpressionOperators(child, contextLabel, depth + 1, nextQDepth));
}

/**
 * Validates a declarative statutory AST rule package against RFC-009 Revision 3.1 contracts.
 * Supports explicit execution modes:
 * - DEVELOPMENT: allows draft/unregistered rules without sectionId, version, or officialCitations.
 * - PRODUCTION: fail-closed enforcement of verified sectionId, version, and deep officialCitations.
 * - Omitted (legacy default): validates identity fields without requiring officialCitations.
 *
 * Throws StatutoryAstValidationError if any contract is violated.
 */
export function validateStatutoryAst(ruleAst, options = {}) {
    if (!ruleAst || typeof ruleAst !== 'object') {
        throw new StatutoryAstValidationError('Rule AST must be a non-null object', { ruleAst });
    }

    // Absolute Fiduciary Invariant Check
    if (ruleAst.isAuthorizedToDisburse === true || ruleAst.statutoryInvariants?.isAuthorizedToDisburse === true) {
        throw new StatutoryAstValidationError(
            'Fiduciary invariant violation: rule AST cannot declare isAuthorizedToDisburse = true (ERR_FIDUCIARY_INVARIANT_VIOLATION)',
            { code: 'ERR_FIDUCIARY_INVARIANT_VIOLATION', sectionId: ruleAst.sectionId }
        );
    }

    const rawRequestedMode = options?.mode ?? ruleAst.executionMode ?? null;
    const requestedMode = rawRequestedMode ? String(rawRequestedMode).trim().toUpperCase() : null;

    if (requestedMode !== null && !EXECUTION_MODES.includes(requestedMode)) {
        throw new StatutoryAstValidationError(
            `Unsupported executionMode '${requestedMode}'. Must be one of: ${EXECUTION_MODES.join(', ')}`,
            { code: 'ERR_INVALID_EXECUTION_MODE', mode: requestedMode }
        );
    }

    // 1. Mode-Aware Identity & Provenance Validation
    if (requestedMode === 'PRODUCTION') {
        if (!ruleAst.sectionId || typeof ruleAst.sectionId !== 'string' || (!ruleAst.isVerifiedInternalPolicy && ruleAst.sectionId.startsWith('DRAFT_'))) {
            throw new StatutoryAstValidationError(
                'ERR_UNVERIFIED_PRODUCTION_RULE: PRODUCTION mode requires a verified non-draft sectionId',
                { code: 'ERR_UNVERIFIED_PRODUCTION_RULE', sectionId: ruleAst.sectionId }
            );
        }
        if (!ruleAst.version || typeof ruleAst.version !== 'string') {
            throw new StatutoryAstValidationError(
                'ERR_UNVERIFIED_PRODUCTION_RULE: PRODUCTION mode requires a valid semantic version string',
                { code: 'ERR_UNVERIFIED_PRODUCTION_RULE', sectionId: ruleAst.sectionId }
            );
        }
        if (!ruleAst.isVerifiedInternalPolicy && !options?.allowInternalPolicy) {
            const citCheck = validateDeepOfficialCitations(ruleAst.officialCitations);
            if (!citCheck.valid) {
                throw new StatutoryAstValidationError(
                    `ERR_UNVERIFIED_PRODUCTION_RULE: ${citCheck.reason}`,
                    { code: 'ERR_UNVERIFIED_PRODUCTION_RULE', sectionId: ruleAst.sectionId, reason: citCheck.reason }
                );
            }
        }
        if (ruleAst.okfVerificationStatus === 'UNREGISTERED' || ruleAst.okfVerificationStatus === 'STALE') {
            throw new StatutoryAstValidationError(
                `ERR_UNVERIFIED_PRODUCTION_RULE: Rule okfVerificationStatus is '${ruleAst.okfVerificationStatus}'`,
                { code: 'ERR_UNVERIFIED_PRODUCTION_RULE', sectionId: ruleAst.sectionId, okfVerificationStatus: ruleAst.okfVerificationStatus }
            );
        }
    } else if (requestedMode === 'DEVELOPMENT') {
        if (ruleAst.sectionId !== undefined && typeof ruleAst.sectionId !== 'string') {
            throw new StatutoryAstValidationError('sectionId, when provided in DEVELOPMENT mode, must be a string', { sectionId: ruleAst.sectionId });
        }
        if (ruleAst.version !== undefined && typeof ruleAst.version !== 'string') {
            throw new StatutoryAstValidationError('version, when provided in DEVELOPMENT mode, must be a string', { version: ruleAst.version });
        }
    } else {
        // Legacy / omitted mode contract
        if (!ruleAst.sectionId || typeof ruleAst.sectionId !== 'string') {
            throw new StatutoryAstValidationError('Rule AST must contain a valid string sectionId', { ruleAst });
        }
        if (!ruleAst.version || typeof ruleAst.version !== 'string') {
            throw new StatutoryAstValidationError('Rule AST must contain a valid string version', { sectionId: ruleAst.sectionId });
        }
    }

    // 2. Trigger Policy Validation
    if (ruleAst.triggerPolicy) {
        if (!TRIGGER_POLICIES.includes(ruleAst.triggerPolicy.type)) {
            throw new StatutoryAstValidationError(
                `Unsupported triggerPolicy type: '${ruleAst.triggerPolicy.type}'. Must be one of: ${TRIGGER_POLICIES.join(', ')}`,
                { triggerPolicy: ruleAst.triggerPolicy }
            );
        }
        if (ruleAst.triggerPolicy.type === 'EARLIER_OF' && (!Array.isArray(ruleAst.triggerPolicy.events) || ruleAst.triggerPolicy.events.length < 2)) {
            throw new StatutoryAstValidationError(
                "Trigger policy 'EARLIER_OF' requires at least 2 events to compare",
                { triggerPolicy: ruleAst.triggerPolicy }
            );
        }
    }

    // 3. Guards Allowlist Validation
    if (ruleAst.guards) {
        if (!Array.isArray(ruleAst.guards)) {
            throw new StatutoryAstValidationError('guards must be an array', { guards: ruleAst.guards });
        }
        for (const guard of ruleAst.guards) {
            if (!guard.field || typeof guard.field !== 'string') {
                throw new StatutoryAstValidationError('Guard must define a string field path', { guard });
            }
            if (!TYPED_OPERATORS.includes(guard.operator)) {
                throw new StatutoryAstValidationError(
                    `Operator '${guard.operator}' in guard '${guard.id || guard.field}' not in typed allowlist: ${TYPED_OPERATORS.join(', ')}`,
                    { guard }
                );
            }
        }
    }

    // 4. Threshold, Math Archetype & Declarative Pipeline Validation
    if (ruleAst.computation) {
        const comp = ruleAst.computation;
        if (comp.archetype && !MATH_ARCHETYPES.includes(comp.archetype)) {
            throw new StatutoryAstValidationError(
                `Unsupported math archetype '${comp.archetype}'. Must be one of: ${MATH_ARCHETYPES.join(', ')}`,
                { computation: comp }
            );
        }
        if (comp.thresholdPaisa !== undefined && comp.thresholdPaisa !== null && typeof comp.thresholdPaisa !== 'string' && typeof comp.thresholdPaisa !== 'number' && typeof comp.thresholdPaisa !== 'bigint') {
            throw new StatutoryAstValidationError(
                'thresholdPaisa must be a valid decimal string or number',
                { thresholdPaisa: comp.thresholdPaisa }
            );
        }
        if (comp.standardRateBps !== undefined && comp.standardRateBps !== null && (typeof comp.standardRateBps !== 'number' || comp.standardRateBps < 0)) {
            throw new StatutoryAstValidationError(
                'standardRateBps must be a non-negative number',
                { standardRateBps: comp.standardRateBps }
            );
        }

        const steps = comp.steps || comp.pipeline?.steps;
        if (steps) {
            if (!Array.isArray(steps)) {
                throw new StatutoryAstValidationError(
                    'Declarative pipeline steps must be an array of step objects',
                    { steps }
                );
            }
            if (steps.length > MAX_PIPELINE_STEPS) {
                throw new StatutoryAstValidationError(
                    `Step count (${steps.length}) exceeds maximum limit of ${MAX_PIPELINE_STEPS} steps (ERR_STEP_BUDGET_EXHAUSTED)`,
                    { stepCount: steps.length, limit: MAX_PIPELINE_STEPS }
                );
            }
            for (const step of steps) {
                if (!step || typeof step !== 'object') {
                    throw new StatutoryAstValidationError(
                        'Each step in declarative pipeline must be a valid object',
                        { step }
                    );
                }
                const assignVar = step.assign || step.name;
                if (!assignVar || typeof assignVar !== 'string') {
                    throw new StatutoryAstValidationError(
                        'Each step in declarative pipeline must define an assignment variable (assign or name)',
                        { step }
                    );
                }
                const expr = step.expr ?? step;
                if (expr && typeof expr === 'object') {
                    validateUniversalExpressionOperators(expr, assignVar, 0, 0);
                    validateNodeDepth(expr, 0);
                }
            }
        }

        const pipelineOutput = comp.pipeline?.output || comp.output;
        if (pipelineOutput && typeof pipelineOutput === 'object' && !Array.isArray(pipelineOutput)) {
            for (const [outKey, outExpr] of Object.entries(pipelineOutput)) {
                if (outExpr && typeof outExpr === 'object') {
                    validateUniversalExpressionOperators(outExpr, `output.${outKey}`, 0, 0);
                    validateNodeDepth(outExpr, 0);
                }
            }
        }

        // Validate optional declarative Deontic Gate structure
        const deonticGate = comp.deonticGate || ruleAst.deonticGate;
        if (deonticGate && typeof deonticGate === 'object') {
            if (deonticGate.prohibitionPredicate && typeof deonticGate.prohibitionPredicate === 'object') {
                validateUniversalExpressionOperators(deonticGate.prohibitionPredicate, 'deonticGate.prohibitionPredicate', 0);
                validateNodeDepth(deonticGate.prohibitionPredicate, 0);
            }
            if (deonticGate.obligationPredicate && typeof deonticGate.obligationPredicate === 'object') {
                validateUniversalExpressionOperators(deonticGate.obligationPredicate, 'deonticGate.obligationPredicate', 0);
                validateNodeDepth(deonticGate.obligationPredicate, 0);
            }
            for (const arrKey of ['exceptionCarveouts', 'disqualificationTriggers', 'mandatoryApprovals', 'gates']) {
                if (Array.isArray(deonticGate[arrKey])) {
                    for (const item of deonticGate[arrKey]) {
                        if (item && typeof item === 'object') {
                            const exprNode = item.expr || item.predicate || item.condition;
                            if (exprNode && typeof exprNode === 'object') {
                                validateUniversalExpressionOperators(exprNode, `deonticGate.${arrKey}`, 0);
                                validateNodeDepth(exprNode, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Rounding Policy Validation
    if (ruleAst.rounding) {
        if (ruleAst.rounding.transactionLine && !ROUNDING_POLICIES.includes(ruleAst.rounding.transactionLine)) {
            throw new StatutoryAstValidationError(
                `Unsupported transactionLine rounding policy: '${ruleAst.rounding.transactionLine}'`,
                { rounding: ruleAst.rounding }
            );
        }
        if (ruleAst.rounding.assessmentTotal && !ROUNDING_POLICIES.includes(ruleAst.rounding.assessmentTotal)) {
            throw new StatutoryAstValidationError(
                `Unsupported assessmentTotal rounding policy: '${ruleAst.rounding.assessmentTotal}'`,
                { rounding: ruleAst.rounding }
            );
        }
    }

    // 6. Symbol Legend Validation
    if (ruleAst.symbolLegend) {
        if (typeof ruleAst.symbolLegend !== 'object' || Array.isArray(ruleAst.symbolLegend)) {
            throw new StatutoryAstValidationError(
                'symbolLegend must be an object map of symbolId -> symbol definition',
                { symbolLegend: ruleAst.symbolLegend }
            );
        }
        for (const [symId, sym] of Object.entries(ruleAst.symbolLegend)) {
            if (!sym || typeof sym !== 'object') {
                throw new StatutoryAstValidationError(
                    `Symbol definition for '${symId}' must be a valid non-null object`,
                    { symId }
                );
            }
            if (sym.comparisonOperator && !TYPED_OPERATORS.includes(sym.comparisonOperator)) {
                throw new StatutoryAstValidationError(
                    `Operator '${sym.comparisonOperator}' in symbol '${symId}' not in typed allowlist: ${TYPED_OPERATORS.join(', ')}`,
                    { symId, comparisonOperator: sym.comparisonOperator }
                );
            }
        }
    }

    return true;
}
