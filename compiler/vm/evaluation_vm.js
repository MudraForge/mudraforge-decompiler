/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE UNIVERSAL STATUTORY VIRTUAL MACHINE (USVM) - EVALUATION KERNEL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements the stateless evaluation kernel for declarative statutory AST packages.
 * Evaluates predicates, triggers, thresholds, rational math, dual-Act regimes,
 * execution modes (DEVELOPMENT vs PRODUCTION), OKF staleness epoch splits,
 * and universal Deontic State Gates with zero external dependencies.
 *
 * Specification: RFC-009 (Revision 3.1)
 */

import crypto from 'node:crypto';
import {
    validateStatutoryAst,
    StatutoryAstValidationError
} from './schemas/statutory_ast_schema.js';
import { cleanNumericAmount } from '../calculators/numeric_utils.js';
import { executeDeclarativePipeline, evaluateExpression } from './universal_evaluator.js';
import { classifyOkfRuleLifecycle } from './okf_lifecycle_monitor.js';
import {
    toBigIntPaise,
    fromBigIntPaise,
    multiplyRational,
    quantizeCurrencyPaisa,
    roundStatutoryAssessment
} from './math_rational.js';
import {
    evaluateMsmeDelayedPaymentCompoundInterest,
    evaluateSection43bHFiscalBoundary,
    evaluateMsmeEnterpriseClassification,
    evaluateGstRule37ProportionalReversal,
    evaluateGstRule37ASupplierNonFiling,
    evaluateGstRule42CommonCreditApportionment,
    evaluateCrossBorderDtaaMatrix,
    evaluateTransferPricingPercentileAlp,
    evaluateSection79VotingContinuity,
    evaluateSection281VoidTransfer,
    evaluateCorporateTaxMatVsConcessionalUsvm,
    evaluateSalaryProgressiveTds,
    evaluateNonFilerPenalRate206AB,
    evaluateDeferredTaxIndAs12Usvm,
    evaluateBlockedCreditSection17_5,
    evaluateItcEligibilitySection16_2,
    evaluateRcmLiabilitySection9_3,
    evaluateCompositeMixedSuppliesSection8,
    evaluateCompositionLevySection10,
    evaluateCreditDebitNotesSection34,
    evaluateExportLutSection16,
    evaluateEInvoicingRule48_4,
    evaluateEWayBillRule138,
    evaluateGstTdsSection51,
    evaluateMcaMsme1Aging,
    evaluateMsmeDisclosuresSection22,
    evaluateLeaseAmortizationIndAs116Usvm,
    evaluateRevenueAllocationIndAs115Usvm,
    evaluateProvisionsAndContingenciesIndAs37Usvm,
    evaluateForeignExchangeRevaluationUsvm,
    evaluateExpectedCreditLossIndAs109Usvm,
    evaluateForeignTaxCreditRelief,
    evaluateSlumpSaleSection50B,
    evaluateBusinessCombinationsIndAs103Usvm,
    evaluateImpairmentIndAs36Usvm,
    evaluateShareBasedPaymentsIndAs102Usvm,
    evaluateCashFlowsIndAs7Usvm,
    evaluatePropertyPlantEquipmentIndAs16Usvm,
    evaluateEmployeeBenefitsIndAs19Usvm,
    evaluateInventoryValuationIndAs2Usvm,
    evaluateEquityMethodIndAs28Usvm,
    evaluateEarningsPerShareIndAs33Usvm,
    evaluateIntangiblesCapitalizationIndAs38Usvm,
    evaluateConsolidationNciIndAs110Usvm,
    evaluateFairValueHierarchyIndAs113Usvm,
    evaluateSection194SUsvm,
    evaluateSection194BAUsvm,
    evaluateSection194NUsvm,
    evaluateSection194DAUsvm,
    evaluateSection194KUsvm,
    evaluateSection206C1GUsvm,
    evaluateSection52GstTcsUsvm
} from './usvm_domain_evaluators.js';

const NUMBER_WORDS = Object.freeze({
    zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5,
    six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
    eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
    sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20,
    thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90
});

const DEONTIC_ARCHETYPES = new Set([
    'DEONTIC_STATE_GATE',
    'STATUS_DISQUALIFICATION_GATE',
    'CORPORATE_APPROVAL_GATE',
    'PROCEDURAL_NOTICE_GATE',
    'SUBJECTIVE_EVIDENCE_GATE',
    'BOOLEAN_PROHIBITION_GATE',
    'BOOLEAN_BINARY_GATE',
    'GENERAL_RULE_GATE'
]);

const DEONTIC_NODE_FALLBACKS = new Set([
    'DEONTIC_STATE_GATE',
    'STATUS_DISQUALIFICATION_GATE',
    'CORPORATE_APPROVAL_GATE',
    'PROCEDURAL_NOTICE_GATE',
    'SUBJECTIVE_EVIDENCE_GATE',
    'BOOLEAN_PROHIBITION_GATE',
    'ACCOUNTING_RECOGNITION_GATE',
    'ACCOUNTING_ELECTION_CARVEOUT',
    'GENERAL_RULE_GATE'
]);

const FORBIDDEN_PROP_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

/**
 * Safely parses natural Indian monetary text or numbers into BigInt paise string.
 */
export function parseMonetaryTextToPaisa(raw) {
    if (raw === undefined || raw === null) return null;
    if (typeof raw === 'bigint') return raw.toString();
    if (typeof raw === 'number' && Number.isFinite(raw)) {
        return String(Math.round(raw * 100));
    }

    const str = String(raw).trim().toLowerCase();
    if (!str) return null;
    if (/^\d+$/.test(str)) {
        return BigInt(str).toString();
    }

    const hasPerCentPhrase = /\bper\s+cent(?:um)?\b/i.test(str);
    const isExplicitMinorUnit = !hasPerCentPhrase &&
        /\b(?:cents?|paise|paisa)\b/i.test(str) &&
        !/\b(?:dollars?|usd|rupees?|inr|euros?|eur|pounds?|gbp)\b/i.test(str);
    const clean = str.replace(/[,₹$€£]/g, '').replace(/\b(?:rs\.?|inr|usd|eur|gbp)\b/gi, '').trim();

    const croreMatch = /(\d+(?:\.\d+)?)\s*(?:crore|cr)\b/i.exec(clean);
    if (croreMatch) {
        return String(Math.round(Number(croreMatch[1]) * 10000000 * 100));
    }
    const lakhMatch = /(\d+(?:\.\d+)?)\s*(?:lakh|lac|lacs|lakhs)\b/i.exec(clean);
    if (lakhMatch) {
        return String(Math.round(Number(lakhMatch[1]) * 100000 * 100));
    }
    const billionMatch = /(\d+(?:\.\d+)?)\s*(?:billion|bn)\b/i.exec(clean);
    if (billionMatch) {
        return String(Math.round(Number(billionMatch[1]) * 1000000000 * 100));
    }
    const millionMatch = /(\d+(?:\.\d+)?)\s*(?:million|mn)\b/i.exec(clean);
    if (millionMatch) {
        return String(Math.round(Number(millionMatch[1]) * 1000000 * 100));
    }
    const thousandMatch = /(\d+(?:\.\d+)?)\s*thousand\b/i.exec(clean);
    if (thousandMatch) {
        return String(Math.round(Number(thousandMatch[1]) * 1000 * 100));
    }

    const wordTokens = clean.split(/\s+/);
    let baseWordVal = null;
    for (const tok of wordTokens) {
        if (NUMBER_WORDS[tok] !== undefined) {
            baseWordVal = (baseWordVal || 0) + NUMBER_WORDS[tok];
        }
    }
    if (baseWordVal !== null) {
        if (clean.includes('crore')) return String(baseWordVal * 10000000 * 100);
        if (clean.includes('lakh') || clean.includes('lac')) return String(baseWordVal * 100000 * 100);
        if (clean.includes('billion')) return String(baseWordVal * 1000000000 * 100);
        if (clean.includes('million')) return String(baseWordVal * 1000000 * 100);
        if (clean.includes('thousand')) return String(baseWordVal * 1000 * 100);
        if (isExplicitMinorUnit) return String(Math.round(baseWordVal));
    }

    const strippedNum = clean.replace(/\b(?:rupees?|inr|dollars?|euros?|pounds?|cents?|paise|paisa)\b/gi, '').trim();
    const plainNumMatch = /^(\d+(?:\.\d+)?)$/.exec(strippedNum);
    if (plainNumMatch) {
        const val = Number(plainNumMatch[1]);
        return isExplicitMinorUnit ? String(Math.round(val)) : String(Math.round(val * 100));
    }

    return null;
}

/**
 * Safely parses percentage text or numeric rate into integer basis points (bps).
 */
export function parseRateTextToBps(raw) {
    if (raw === undefined || raw === null) return null;
    if (typeof raw === 'number' && Number.isFinite(raw)) {
        return Math.round(raw * 100);
    }
    const str = String(raw).trim().toLowerCase();
    const m = /(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)/i.exec(str);
    if (m) {
        return Math.round(Number(m[1]) * 100);
    }
    const plain = /^(\d+(?:\.\d+)?)$/.exec(str);
    if (plain) {
        return Math.round(Number(plain[1]) * 100);
    }
    return null;
}

/**
 * Safely extracts a nested property using dot-separated notation.
 */
function getNestedValue(obj, path) {
    if (!obj || !path || typeof obj !== 'object') return undefined;
    const parts = String(path).split('.');
    let curr = obj;
    for (const part of parts) {
        if (curr === null || curr === undefined || typeof curr !== 'object') return undefined;
        if (FORBIDDEN_PROP_KEYS.has(part)) return undefined;
        if (!Object.prototype.hasOwnProperty.call(curr, part) && !Array.isArray(curr)) return undefined;
        curr = curr[part];
    }
    return curr;
}

export function isPureLogicRule(ruleAst, options = {}) {
    const optDomain = options?.ruleDomain ? String(options.ruleDomain).trim().toUpperCase() : null;
    if (optDomain === 'STATUTORY_FINANCE') return false;
    if (optDomain === 'GENERAL_LOGIC') return true;
    const astDomain = ruleAst?.ruleDomain ? String(ruleAst.ruleDomain).trim().toUpperCase() : null;
    if (astDomain === 'STATUTORY_FINANCE') return false;
    if (astDomain === 'GENERAL_LOGIC') return true;
    const archetype = ruleAst?.computation?.archetype;
    const outputMode = ruleAst?.computation?.outputMode;
    const nodeType = ruleAst?.rootCondition?.nodeType;
    return archetype === 'PURE_LOGIC_PIPELINE' || outputMode === 'PURE_LOGIC' || nodeType === 'GENERAL_RULE_GATE';
}

/**
 * Evaluates a typed comparison operator against fact value and target value.
 */
function evaluateOperator(factValue, operator, targetValue) {
    let fv = factValue;
    let tv = targetValue;
    if (typeof fv === 'string' && /^\d+$/.test(fv)) {
        fv = BigInt(fv);
    }
    if (typeof tv === 'string' && /^\d+$/.test(tv)) {
        tv = BigInt(tv);
    }
    if (typeof fv === 'bigint' && typeof tv === 'number') {
        tv = BigInt(tv);
    } else if (typeof fv === 'number' && typeof tv === 'bigint') {
        fv = BigInt(fv);
    }

    switch (operator) {
        case 'EQ':
            return fv === tv;
        case 'NEQ':
            return fv !== tv;
        case 'GT':
            return fv > tv;
        case 'GTE':
            return fv >= tv;
        case 'LT':
            return fv < tv;
        case 'LTE':
            return fv <= tv;
        case 'IN':
            return Array.isArray(targetValue) && targetValue.includes(factValue);
        case 'NOT_IN':
            return Array.isArray(targetValue) && !targetValue.includes(factValue);
        case 'IS_TRUE':
            return Boolean(factValue) === true;
        case 'IS_FALSE':
            return Boolean(factValue) === false;
        case 'EXISTS':
            return factValue !== undefined && factValue !== null;
        case 'NOT_EXISTS':
            return factValue === undefined || factValue === null;
        default:
            return false;
    }
}

/**
 * Normalizes a raw Mudra-Decompiler-0 output AST (`rootCondition`, `compositionalBuildingBlocks`,
 * optional `symbolLegend`) directly into an executable USVM AST without external wrappers.
 */
export function normalizeDecompilerAstEnvelope(ruleAst, requestedMode = null) {
    if (!ruleAst || typeof ruleAst !== 'object') return ruleAst;

    const normalized = { ...ruleAst };
    const blocks = normalized.compositionalBuildingBlocks || {};
    const rootCond = normalized.rootCondition || {};
    const compExpr = blocks.computationalExpression || {};
    const deonticMod = blocks.deonticModality || {};
    const tempBlock = blocks.temporalCrystallization || {};

    // In explicit DEVELOPMENT mode, synthesize deterministic DRAFT identity if missing
    const modeUpper = requestedMode ? String(requestedMode).trim().toUpperCase() : null;
    if (modeUpper === 'DEVELOPMENT') {
        if (!normalized.sectionId || typeof normalized.sectionId !== 'string') {
            const payload = JSON.stringify(ruleAst, (_, v) => (typeof v === 'bigint' ? v.toString() : v));
            const digest = crypto.createHash('sha256').update(payload).digest('hex').slice(0, 8).toUpperCase();
            normalized.sectionId = `DRAFT_${digest}`;
        }
        if (!normalized.version || typeof normalized.version !== 'string') {
            normalized.version = '0.1.0-draft';
        }
    }

    // Normalize triggerPolicy from decompiler string forms
    if (!normalized.triggerPolicy && (rootCond.triggerPolicy || tempBlock.triggerPolicy)) {
        const rawTrig = String(rootCond.triggerPolicy || tempBlock.triggerPolicy).toUpperCase();
        if (rawTrig.includes('EARLIER')) {
            normalized.triggerPolicy = { type: 'EARLIER_OF', events: ['BOOK_CREDIT', 'PAYMENT_DISBURSEMENT'] };
        } else if (rawTrig.includes('PAYMENT')) {
            normalized.triggerPolicy = { type: 'PAYMENT_ONLY' };
        } else if (rawTrig.includes('INVOICE')) {
            normalized.triggerPolicy = { type: 'INVOICE_DATE' };
        } else {
            normalized.triggerPolicy = { type: 'TRANSACTION_DATE' };
        }
    }

    // Populate computation parameters from rootCondition / compositionalBuildingBlocks if missing
    if (!normalized.computation && (rootCond.nodeType || compExpr.expressionType)) {
        const comp = {};
        if (compExpr.expressionType) {
            comp.archetype = compExpr.expressionType;
        } else if (rootCond.nodeType && DEONTIC_NODE_FALLBACKS.has(rootCond.nodeType)) {
            comp.archetype = rootCond.nodeType;
        }

        const rawThresh = compExpr.excessThreshold ?? rootCond.applicableExcessThreshold ?? rootCond.applicableCliffThreshold;
        const parsedPaisa = parseMonetaryTextToPaisa(rawThresh);
        if (parsedPaisa !== null) {
            comp.thresholdPaisa = parsedPaisa;
        }

        const rawRate = compExpr.standardRate ?? rootCond.standardTaxRate;
        const parsedBps = parseRateTextToBps(rawRate);
        if (parsedBps !== null) {
            comp.standardRateBps = parsedBps;
        }

        if (comp.archetype || comp.thresholdPaisa !== undefined || comp.standardRateBps !== undefined) {
            normalized.computation = comp;
        }
    }

    // Synthesize deonticGate automatically for non-computational decompiler ASTs when not already set
    const gateType = normalized.deonticGate?.gateType
        || normalized.computation?.deonticGate?.gateType
        || (DEONTIC_NODE_FALLBACKS.has(rootCond.nodeType) ? rootCond.nodeType : null);

    if (!normalized.deonticGate && !normalized.computation?.deonticGate && gateType && !normalized.computation?.standardRateBps && !normalized.computation?.thresholdPaisa && !normalized.computation?.steps) {
        const carveoutsRaw = rootCond.exceptionCarveouts || blocks.carveoutsAndExceptions?.safeHarbors || [];
        const exceptionCarveouts = Array.isArray(carveoutsRaw)
            ? carveoutsRaw.map((c, idx) => (typeof c === 'string'
                ? { id: `CARVEOUT_${idx + 1}`, rawText: c, factKey: `carveout_${idx + 1}` }
                : c))
            : [];

        normalized.deonticGate = {
            gateType,
            modalityType: deonticMod.modalityType || (gateType === 'BOOLEAN_PROHIBITION_GATE' ? 'PROHIBITION' : 'OBLIGATION'),
            statutoryDuty: deonticMod.statutoryDuty || null,
            exceptionCarveouts,
            disqualificationTriggers: Array.isArray(rootCond.disqualificationTriggers) ? rootCond.disqualificationTriggers : [],
            mandatoryApprovals: Array.isArray(rootCond.mandatoryApprovals) ? rootCond.mandatoryApprovals : [],
            noticeWindowDays: rootCond.noticeWindowDays ?? null,
            requireAttestationIfChecklist: gateType === 'SUBJECTIVE_EVIDENCE_GATE'
        };
    }

    return normalized;
}

/**
 * Computes RFC 8785 compatible canonical key-sorted digest for sealed production vouchers.
 */
function computeVoucherDigest(voucherBody) {
    const normalizeForHash = (val) => {
        if (typeof val === 'bigint') return val.toString();
        if (Array.isArray(val)) return val.map(normalizeForHash);
        if (val && typeof val === 'object') {
            const sorted = {};
            for (const key of Object.keys(val).sort()) {
                if (key === 'voucherDigest' || key === 'voucherSealStatus') continue;
                sorted[key] = normalizeForHash(val[key]);
            }
            return sorted;
        }
        return val;
    };
    const canonical = JSON.stringify(normalizeForHash(voucherBody));
    return crypto.createHash('sha256').update(canonical).digest('hex');
}

/**
 * Stamps immutable execution mode provenance, citations policy, and fiduciary invariant on voucher.
 */
function stampVoucherProvenance(voucher, normalizedAst, provenance = {}) {
    const mode = provenance.mode || 'DEVELOPMENT';
    const isHalted = Boolean(voucher.isBlocked || voucher.disposition === 'HOLD_STALE_OKF_PENDING_AMENDMENT');
    const isProductionCertified = mode === 'PRODUCTION' && !isHalted;

    voucher.isAuthorizedToDisburse = false;
    voucher.executionMode = mode;
    voucher.isProductionCertified = isProductionCertified;
    voucher.okfLifecycleStatus = provenance.lifecycle?.lifecycleStatus
        || (mode === 'PRODUCTION' ? 'OKF_VERIFIED_ACTIVE' : 'OKF_UNREGISTERED_NEW');
    voucher.citations = mode === 'PRODUCTION'
        ? (Array.isArray(normalizedAst?.officialCitations) ? normalizedAst.officialCitations : [])
        : [];
    voucher.voucherSealStatus = isProductionCertified ? 'RFC8785_SHA256_SEALED' : 'UNSEALED_DRAFT';
    voucher.voucherDigest = isProductionCertified ? computeVoucherDigest(voucher) : null;

    return voucher;
}

/**
 * Evaluates a single declarative gate predicate against execution context and facts.
 */
function evaluateGateCondition(item, context, facts) {
    if (!item || typeof item !== 'object') return false;
    const exprNode = item.expr ?? item.predicate ?? item.condition ?? null;
    if (exprNode !== null && exprNode !== undefined) {
        return Boolean(evaluateExpression(exprNode, context));
    }
    if (item.field || item.factKey) {
        const key = item.field || item.factKey;
        const val = getNestedValue(facts, key) ?? facts[key];
        if (item.operator) {
            return evaluateOperator(val, item.operator, item.value);
        }
        return Boolean(val) === true;
    }
    if (item.id && facts[item.id] !== undefined) {
        return Boolean(facts[item.id]) === true;
    }
    return false;
}

/**
 * Universal Declarative Deontic State Gate Evaluator:
 * Evaluates disqualifications, corporate approvals, procedural notice windows,
 * prohibitions + safe-harbor carveouts (legal precedence first!), and controllership attestations
 * without any section-specific hardcoded JavaScript math.
 */
export function evaluateDeonticStateGate(ruleAst, facts = {}, temporal = {}, trace = [], baseResult = {}) {
    const deonticGate = ruleAst.deonticGate || ruleAst.computation?.deonticGate || {};
    const gateType = deonticGate.gateType
        || ruleAst.computation?.archetype
        || ruleAst.rootCondition?.nodeType
        || 'DEONTIC_STATE_GATE';

    const evalContext = {
        facts: { ...facts },
        rules: { ...(ruleAst.computation || {}), ...deonticGate },
        vars: {},
        temporal,
        trace
    };

    const result = {
        ...baseResult,
        isAuthorizedToDisburse: false,
        isApplicable: true,
        gateType,
        modalityType: deonticGate.modalityType || ruleAst.compositionalBuildingBlocks?.deonticModality?.modalityType || 'OBLIGATION',
        isBlocked: false,
        isCompliant: true,
        isAdmissible: true,
        matchedCarveouts: [],
        triggeredBars: [],
        missingApprovals: []
    };

    const finishGate = (res) => {
        if (res.modalityType === 'PERMISSION' || res.modalityType === 'PERMISSION_ELECTION') {
            res.isPermitted = !res.isBlocked;
        }
        return res;
    };

    trace.push(`USVM evaluating Deontic State Gate [${gateType}]`);

    // Precedence 1: Exception Carveouts & Safe Harbors evaluate FIRST (legal override)
    const carveouts = Array.isArray(deonticGate.exceptionCarveouts) ? deonticGate.exceptionCarveouts : [];
    for (const c of carveouts) {
        if (evaluateGateCondition(c, evalContext, facts)) {
            const label = c.id || c.factKey || c.rawText || 'STATUTORY_CARVEOUT';
            result.matchedCarveouts.push(label);
        }
    }

    // Support top-level urgent interim relief flag for procedural notice gates
    if (gateType === 'PROCEDURAL_NOTICE_GATE' && deonticGate.allowUrgentInterimRelief !== false) {
        const urgentKey = deonticGate.urgentReliefCarveoutKey || 'hasUrgentInterimRelief';
        if (Boolean(facts[urgentKey] || facts.urgentInterimRelief) === true) {
            result.matchedCarveouts.push('URGENT_INTERIM_RELIEF');
        }
    }

    if (result.matchedCarveouts.length > 0) {
        trace.push(`Deontic exception carveout established: ${result.matchedCarveouts.join(', ')}`);
        result.disposition = gateType === 'PROCEDURAL_NOTICE_GATE'
            ? 'ADMISSIBLE_URGENT_CARVEOUT'
            : 'EXEMPT_STATUTORY_CARVEOUT';
        result.statutoryDisposition = result.disposition;
        result.statutoryReason = `Statutory exception carveout applied (${result.matchedCarveouts.join(', ')})`;
        result.isBlocked = false;
        result.isCompliant = true;
        result.isAdmissible = true;
        return finishGate(result);
    }

    // Precedence 2: Disqualification Triggers (Companies Act Sec 164, IBC Sec 29A)
    const triggers = Array.isArray(deonticGate.disqualificationTriggers) ? deonticGate.disqualificationTriggers : [];
    for (const trig of triggers) {
        if (evaluateGateCondition(trig, evalContext, facts)) {
            const label = trig.id || trig.factKey || trig.reason || 'DISQUALIFICATION_TRIGGER';
            result.triggeredBars.push(label);
        }
    }
    if (result.triggeredBars.length > 0) {
        trace.push(`Disqualification triggers tripped: ${result.triggeredBars.join(', ')}`);
        result.disposition = 'BLOCKED_STATUTORY_DISQUALIFICATION';
        result.statutoryDisposition = 'BLOCKED_STATUTORY_DISQUALIFICATION';
        result.statutoryReason = `Statutory disqualification triggered: ${result.triggeredBars.join(', ')}`;
        result.isBlocked = true;
        result.isCompliant = false;
        result.isAdmissible = false;
        return finishGate(result);
    }

    // Precedence 3: Prohibition Predicate (SEBI PIT Reg 3, CGST Sec 17(5), Sec 269ST)
    if (deonticGate.prohibitionPredicate !== undefined && deonticGate.prohibitionPredicate !== null) {
        const isProhibited = typeof deonticGate.prohibitionPredicate === 'object' || typeof deonticGate.prohibitionPredicate === 'string'
            ? Boolean(evaluateExpression(deonticGate.prohibitionPredicate, evalContext))
            : Boolean(deonticGate.prohibitionPredicate);

        if (isProhibited) {
            trace.push('Prohibition predicate evaluated TRUE with no active carveout');
            result.disposition = 'BLOCKED_STATUTORY_PROHIBITION';
            result.statutoryDisposition = 'BLOCKED_STATUTORY_PROHIBITION';
            result.statutoryReason = deonticGate.prohibitionReason || 'Conduct barred by statutory prohibition without safe-harbor carveout.';
            result.isBlocked = true;
            result.isCompliant = false;
            result.isAdmissible = false;
            return finishGate(result);
        }
    }

    // Precedence 4: Mandatory Approvals / Corporate Governance Preconditions (Companies Act Sec 188, Sec 177)
    const approvals = Array.isArray(deonticGate.mandatoryApprovals) ? deonticGate.mandatoryApprovals : [];
    for (const app of approvals) {
        if (!evaluateGateCondition(app, evalContext, facts)) {
            const label = app.id || app.factKey || app.description || 'MANDATORY_APPROVAL';
            result.missingApprovals.push(label);
        }
    }
    if (result.missingApprovals.length > 0) {
        trace.push(`Missing mandatory corporate approvals: ${result.missingApprovals.join(', ')}`);
        result.disposition = 'BLOCKED_MISSING_CORPORATE_APPROVAL';
        result.statutoryDisposition = 'BLOCKED_MISSING_CORPORATE_APPROVAL';
        result.statutoryReason = `Required statutory approval(s) missing: ${result.missingApprovals.join(', ')}`;
        result.isBlocked = true;
        result.isCompliant = false;
        result.isAdmissible = false;
        return finishGate(result);
    }

    // Precedence 5: Procedural Notice / Wait Window (CPC Sec 80, Commercial Courts Sec 12A)
    const requiredNoticeDays = deonticGate.noticeWindowDays !== undefined && deonticGate.noticeWindowDays !== null
        ? Number(deonticGate.noticeWindowDays)
        : null;
    if (requiredNoticeDays !== null && requiredNoticeDays > 0) {
        const elapsedField = deonticGate.elapsedDaysField || 'noticeElapsedDays';
        let elapsedDays = facts[elapsedField] ?? getNestedValue(facts, elapsedField);
        if ((elapsedDays === undefined || elapsedDays === null) && facts.noticeServedDate && (facts.filingDate || facts.actionDate || facts.transactionDate)) {
            elapsedDays = evaluateExpression({
                op: 'DATE_DIFF',
                start: { var: 'facts.noticeServedDate' },
                end: facts.filingDate ? { var: 'facts.filingDate' } : (facts.actionDate ? { var: 'facts.actionDate' } : { var: 'facts.transactionDate' }),
                unit: 'DAYS'
            }, evalContext);
        }
        const numericElapsed = Number(elapsedDays ?? 0);
        const noticeServed = facts.noticeServed !== undefined ? Boolean(facts.noticeServed) : (numericElapsed > 0 || Boolean(facts.noticeServedDate));

        result.requiredNoticeDays = requiredNoticeDays;
        result.elapsedNoticeDays = numericElapsed;

        if (!noticeServed || numericElapsed < requiredNoticeDays) {
            const remainingDays = Math.max(0, requiredNoticeDays - (noticeServed ? numericElapsed : 0));
            result.remainingWindowDays = remainingDays;
            trace.push(`Procedural notice wait window unsatisfied (${numericElapsed}/${requiredNoticeDays} days)`);
            result.disposition = 'BLOCKED_MANDATORY_WAIT_WINDOW';
            result.statutoryDisposition = 'BLOCKED_MANDATORY_WAIT_WINDOW';
            result.statutoryReason = `Mandatory statutory wait period of ${requiredNoticeDays} days not completed (${remainingDays} days remaining).`;
            result.isBlocked = true;
            result.isCompliant = false;
            result.isAdmissible = false;
            return finishGate(result);
        }
    }

    // Precedence 6: Generic Gates Array (arbitrary custom deontic rules)
    if (Array.isArray(deonticGate.gates)) {
        for (const g of deonticGate.gates) {
            const ok = evaluateGateCondition(g, evalContext, facts);
            if (g.blockOn === 'TRUE' && ok) {
                result.disposition = g.disposition || 'BLOCKED_STATUTORY_BAR';
                result.statutoryDisposition = result.disposition;
                result.statutoryReason = g.reason || `Deontic gate '${g.id || 'GATE'}' blocked transaction`;
                result.isBlocked = true;
                result.isCompliant = false;
                result.isAdmissible = false;
                return finishGate(result);
            }
            if (g.blockOn !== 'TRUE' && !ok) {
                result.disposition = g.disposition || 'BLOCKED_STATUTORY_BAR';
                result.statutoryDisposition = result.disposition;
                result.statutoryReason = g.reason || `Mandatory deontic gate '${g.id || 'GATE'}' unsatisfied`;
                result.isBlocked = true;
                result.isCompliant = false;
                result.isAdmissible = false;
                return finishGate(result);
            }
        }
    }

    // Precedence 7: Controllership Attestation & Evidentiary Checklist
    const checklist = Array.isArray(result.attestationChecklist) ? result.attestationChecklist : [];
    const enforcesChecklist = Boolean(
        checklist.length > 0 &&
        (deonticGate.requireAttestationIfChecklist === true || gateType === 'SUBJECTIVE_EVIDENCE_GATE')
    );

    if (enforcesChecklist) {
        const hasGlobalAttestation = Boolean(facts.hasControllerAttestation || facts.isAttested);
        const attestedMap = (facts.attestations && typeof facts.attestations === 'object') ? facts.attestations : {};
        const missingAttestations = checklist
            .map((item) => (typeof item === 'string' ? item : item.symbolId || item.id))
            .filter((symKey) => symKey && !hasGlobalAttestation && !Boolean(attestedMap[symKey]) && !Boolean(facts[symKey]));

        if (missingAttestations.length > 0) {
            result.missingAttestations = missingAttestations;
            result.disposition = 'BLOCKED_PENDING_ATTESTATION';
            result.statutoryDisposition = 'BLOCKED_PENDING_ATTESTATION';
            result.statutoryReason = `Subjective or evidentiary controllership attestation required for: ${missingAttestations.join(', ')}`;
            result.isBlocked = true;
            result.isCompliant = false;
            result.isAdmissible = false;
            result.requiresControllershipAttestation = true;
            return finishGate(result);
        }
    }

    // Compliant / Admissible Deontic State
    result.disposition = gateType === 'PROCEDURAL_NOTICE_GATE'
        ? 'COMPLIANT_WAIT_WINDOW_SATISFIED'
        : (gateType === 'CORPORATE_APPROVAL_GATE' ? 'COMPLIANT_APPROVED' : 'COMPLIANT_ELIGIBLE');
    result.statutoryDisposition = result.disposition;
    result.statutoryReason = 'All deontic gates and preconditions satisfied.';
    result.isBlocked = false;
    result.isCompliant = true;
    result.isAdmissible = true;
    return finishGate(result);
}

/**
 * Resolves the statutory crystallization date and governing regime (1961 vs 2025 Act)
 * based on the provision's trigger policy and transaction facts.
 */
export function resolveTemporalCrystallization(triggerPolicy, facts, cutoverDate = '2026-04-01') {
    const creditDate = facts.creditDate || facts.temporalFacts?.bookCreditDate || null;
    const paymentDate = facts.paymentDate || facts.temporalFacts?.paymentDate || null;
    const transactionDate = facts.transactionDate || facts.temporalFacts?.transactionDate || null;

    let crystallizedDate = null;
    let crystallizationEvent = 'UNDETERMINED';
    let isStraddling = false;

    const policyType = triggerPolicy?.type || 'EARLIER_OF';

    if (policyType === 'EARLIER_OF') {
        if (creditDate && paymentDate) {
            if (creditDate <= paymentDate) {
                crystallizedDate = creditDate;
                crystallizationEvent = 'BOOK_CREDIT';
            } else {
                crystallizedDate = paymentDate;
                crystallizationEvent = 'PAYMENT_DISBURSEMENT';
            }

            const creditIsPre = creditDate < cutoverDate;
            const paymentIsPre = paymentDate < cutoverDate;
            if (creditIsPre !== paymentIsPre) {
                isStraddling = true;
            }
        } else if (creditDate) {
            crystallizedDate = creditDate;
            crystallizationEvent = 'BOOK_CREDIT';
        } else if (paymentDate) {
            crystallizedDate = paymentDate;
            crystallizationEvent = 'PAYMENT_DISBURSEMENT';
        } else if (transactionDate) {
            crystallizedDate = transactionDate;
            crystallizationEvent = 'TRANSACTION_DATE';
        }
    } else if (policyType === 'PAYMENT_ONLY') {
        crystallizedDate = paymentDate || transactionDate || null;
        crystallizationEvent = 'PAYMENT_DISBURSEMENT';
    } else {
        crystallizedDate = transactionDate || creditDate || paymentDate || null;
        crystallizationEvent = 'TRANSACTION_DATE';
    }

    const hasDate = Boolean(crystallizedDate);
    const isAct2025 = hasDate ? crystallizedDate >= cutoverDate : false;
    const governingRegime = hasDate ? (isAct2025 ? 'ACT_2025' : 'ACT_1961') : 'INDETERMINATE_PENDING_TRANSACTION_DATE';

    return {
        crystallizedDate,
        crystallizationEvent,
        isAct2025,
        governingRegime,
        isStraddling,
        creditDate,
        paymentDate
    };
}

/**
 * Pure stateless evaluation kernel for the Universal Statutory Virtual Machine (USVM).
 * Evaluates a declarative AST rule package against transaction facts under explicit mode bounds.
 *
 * @param {Object} ruleAst - Declarative AST package (or raw Mudra-Decompiler-0 output)
 * @param {Object} [facts={}] - Ingress transaction facts
 * @param {Object} [options={}] - Execution options ({ mode: 'DEVELOPMENT' | 'PRODUCTION', checkOkfStaleness, okfRegistry, customSnapshots, cutoverDate })
 * @returns {Object} Fiduciary statutory evaluation voucher
 */
export function evaluateStatutoryAst(ruleAst, facts = {}, options = {}) {
    facts = (facts && typeof facts === 'object') ? facts : {};
    options = (options && typeof options === 'object') ? options : {};

    const explicitMode = options.mode
        ? String(options.mode).trim().toUpperCase()
        : (ruleAst?.executionMode ? String(ruleAst.executionMode).trim().toUpperCase() : null);

    let activeAst = normalizeDecompilerAstEnvelope(ruleAst, explicitMode);

    // 1. Strict Mode-Aware Schema Validation
    if (explicitMode) {
        validateStatutoryAst(activeAst, {
            mode: explicitMode,
            allowInternalPolicy: options.allowInternalPolicy
        });
    } else {
        validateStatutoryAst(activeAst);
    }

    const effectiveMode = explicitMode
        || ((Array.isArray(activeAst.officialCitations) && activeAst.officialCitations.length > 0) ? 'PRODUCTION' : 'DEVELOPMENT');

    const trace = [];
    let lifecycle = null;
    let historicalEpochMeta = null;

    // 2. OKF Registry & Staleness Verification Gate (PRODUCTION Mode)
    const cutoverDate = activeAst.transitionalCutoverDate || '2026-04-01';
    let temporal = resolveTemporalCrystallization(activeAst.triggerPolicy, facts, cutoverDate);

    if (explicitMode === 'PRODUCTION' && options.checkOkfStaleness !== false && options.okfRegistry !== false) {
        lifecycle = classifyOkfRuleLifecycle(activeAst, {
            okfRegistry: options.okfRegistry,
            customSnapshots: options.customSnapshots,
            okfRootDir: options.okfRootDir,
            rulesDir: options.rulesDir,
            cutoverDate: options.cutoverDate,
            forceStale: options.okfStatus === 'OKF_AMENDMENT_STALE_DETECTED'
        });

        if (
            lifecycle.lifecycleStatus === 'OKF_UNREGISTERED_NEW' &&
            !activeAst.isVerifiedInternalPolicy &&
            !options.allowUnregisteredWithValidCitations
        ) {
            throw new StatutoryAstValidationError(
                `ERR_UNVERIFIED_PRODUCTION_RULE: Rule '${activeAst.sectionId}' is not registered in the verified OKF corpus. Execute under mode='DEVELOPMENT' or ratify into OKF first.`,
                {
                    code: 'ERR_UNVERIFIED_PRODUCTION_RULE',
                    sectionId: activeAst.sectionId,
                    lifecycleStatus: lifecycle.lifecycleStatus
                }
            );
        }

        if (lifecycle.lifecycleStatus === 'OKF_AMENDMENT_STALE_DETECTED') {
            const amendmentCutover = lifecycle.cutoverDate;
            if (!temporal.crystallizedDate || temporal.crystallizedDate >= amendmentCutover) {
                trace.push(`HOLD_STALE_OKF_PENDING_AMENDMENT: crystallizedDate=${temporal.crystallizedDate || 'NONE'} >= amendmentCutover=${amendmentCutover}`);
                return stampVoucherProvenance({
                    isAuthorizedToDisburse: false,
                    status: 'HALTED',
                    code: 'HOLD_STALE_OKF_PENDING_AMENDMENT',
                    disposition: 'HOLD_STALE_OKF_PENDING_AMENDMENT',
                    statutoryDisposition: 'HOLD_STALE_OKF_PENDING_AMENDMENT',
                    statutoryReason: `Active OKF rule (${lifecycle.matchedSectionId}) differs from incoming amendment effective ${amendmentCutover}. Halted fail-closed pending Temporal Epoch ratification.`,
                    isBlocked: true,
                    ruleId: activeAst.sectionId,
                    ruleVersion: activeAst.version,
                    effectiveDate: temporal.crystallizedDate,
                    cutoverDate: amendmentCutover,
                    staleDiff: lifecycle.fieldDiffs,
                    epochSplitProposed: lifecycle.epochSplitProposed,
                    executionTrace: trace
                }, activeAst, { mode: 'PRODUCTION', lifecycle, blocked: true });
            }

            // Retrospective audit strictly prior to amendment cutover executes against closed historical epoch
            if (lifecycle.epochSplitProposed?.historicalEpoch) {
                const preservedCitations = activeAst.officialCitations || lifecycle.matchedSnapshot?.officialCitations || [];
                const mergedHistorical = {
                    ...lifecycle.epochSplitProposed.historicalEpoch,
                    officialCitations: preservedCitations,
                    triggerPolicy: activeAst.triggerPolicy || lifecycle.epochSplitProposed.historicalEpoch.triggerPolicy
                };
                activeAst = normalizeDecompilerAstEnvelope(mergedHistorical, 'PRODUCTION');
                temporal = resolveTemporalCrystallization(activeAst.triggerPolicy, facts, activeAst.transitionalCutoverDate || '2026-04-01');
                historicalEpochMeta = {
                    historicalEpochApplied: true,
                    historicalValidTo: lifecycle.epochSplitProposed.historicalValidTo,
                    amendmentCutoverDate: amendmentCutover,
                    pendingPostCutoverEpoch: lifecycle.epochSplitProposed.pendingEpoch
                };
                trace.push(`Retrospective audit pre-${amendmentCutover}: applied historical OKF epoch closed at ${lifecycle.epochSplitProposed.historicalValidTo}`);
            }
        }
    }

    const provenance = { mode: effectiveMode, lifecycle };
    const finalize = (voucherObj) => {
        if (historicalEpochMeta) {
            Object.assign(voucherObj, historicalEpochMeta);
        }
        return stampVoucherProvenance(voucherObj, activeAst, provenance);
    };

    const isPureLogic = isPureLogicRule(activeAst, options);

    // 3. Fiduciary Base Result Initialization
    const result = isPureLogic
        ? {
            isAuthorizedToDisburse: false,
            ruleId: activeAst.sectionId,
            ruleVersion: activeAst.version,
            ruleDomain: 'GENERAL_LOGIC',
            disposition: 'EVALUATING',
            isApplicable: false,
            attestationChecklist: Array.isArray(activeAst.attestationChecklist) ? activeAst.attestationChecklist : [],
            requiresControllershipAttestation: Boolean(Array.isArray(activeAst.attestationChecklist) && activeAst.attestationChecklist.length > 0),
            executionTrace: trace
        }
        : {
            isAuthorizedToDisburse: false,
            ruleId: activeAst.sectionId,
            ruleVersion: activeAst.version,
            disposition: 'EVALUATING',
            isApplicable: false,
            isTdsApplicable: false,
            isSectionApplicable: false,
            invoiceAmountInr: 0,
            taxableBase: 0,
            tdsRatePercent: 0,
            effectiveRateBps: 0,
            calculatedTdsInr: 0,
            calculatedTaxInr: 0,
            netWireInr: 0,
            attestationChecklist: Array.isArray(activeAst.attestationChecklist) ? activeAst.attestationChecklist : [],
            requiresControllershipAttestation: Boolean(Array.isArray(activeAst.attestationChecklist) && activeAst.attestationChecklist.length > 0),
            executionTrace: trace
        };

    if (!isPureLogic) {
        result.isAct2025 = temporal.isAct2025;
        result.effectiveDate = temporal.crystallizedDate;
        result.regimeStatus = temporal.governingRegime === 'INDETERMINATE_PENDING_TRANSACTION_DATE' ? 'INDETERMINATE_PENDING_TRANSACTION_DATE' : 'DETERMINED';

        if (temporal.isStraddling) {
            result.temporalStraddlingAnalysis = {
                crystallizedDate: temporal.crystallizedDate,
                crystallizationEvent: temporal.crystallizationEvent,
                governingRegime: temporal.governingRegime,
                creditDate: temporal.creditDate,
                paymentDate: temporal.paymentDate,
                legalRule: 'Under Indian direct tax jurisprudence, tax deduction attaches at the earlier of credit or payment. Pre-1 April 2026 crystallization event cements Income-tax Act, 1961 regime.'
            };
        }

        const governingSection = temporal.isAct2025
            ? `${activeAst.successorStatute || 'Income-tax Act, 2025'} ${activeAst.successorSection || ''}`.trim()
            : `${activeAst.governingStatute || 'Income-tax Act, 1961'} ${activeAst.statutorySection || ''}`.trim();

        result.governingWithholdingSection = governingSection;
        result.governingSection = governingSection;
        result.governingStatute = temporal.governingRegime;
    } else if (temporal.crystallizedDate) {
        result.effectiveDate = temporal.crystallizedDate;
    }

    // 4. Evaluate Guards (Pre-conditions & Penalties)
    let overrideRateBps = null;
    let panPenaltySection = null;

    if (Array.isArray(activeAst.guards)) {
        for (const guard of activeAst.guards) {
            let factValue = getNestedValue(facts, guard.field);
            if (factValue === undefined && guard.field.startsWith('payee.')) {
                const flatKey = guard.field.replace('payee.', '');
                if (flatKey === 'isPanValid' && facts.hasPan !== undefined) {
                    factValue = Boolean(facts.hasPan) && !Boolean(facts.isPanInoperative);
                } else if (facts[flatKey] !== undefined) {
                    factValue = facts[flatKey];
                }
            }
            if (factValue === undefined && guard.field === 'payee.isResident' && facts.sellerIsNonResident !== undefined) {
                factValue = !Boolean(facts.sellerIsNonResident);
            }

            if (factValue === undefined) {
                if (guard.field === 'buyerPriorTurnoverPaisa') {
                    if (facts.buyerPriorTurnoverPaisa !== undefined) factValue = facts.buyerPriorTurnoverPaisa;
                    else if (facts.priorFyTurnoverPaisa !== undefined) factValue = facts.priorFyTurnoverPaisa;
                    else if (facts.buyerTurnoverInr !== undefined) factValue = BigInt(Math.round(Number(facts.buyerTurnoverInr) * 100));
                    else if (facts.priorTurnoverInr !== undefined) factValue = BigInt(Math.round(Number(facts.priorTurnoverInr) * 100));
                } else if (guard.field === 'continuousServiceYears') {
                    factValue = facts.continuousServiceYears ?? facts.serviceYears ?? facts.tenureYears;
                } else if (guard.field === 'leaseTermMonths') {
                    factValue = facts.leaseTermMonths ?? facts.termMonths ?? facts.leaseDurationMonths;
                } else {
                    const minorAlias = facts[`${guard.field}Paisa`] ?? facts[`${guard.field}Cents`] ?? facts[`${guard.field}MinorUnits`];
                    if (minorAlias !== undefined) {
                        factValue = typeof minorAlias === 'bigint' ? minorAlias : BigInt(Math.round(Number(minorAlias)));
                    }
                }
            }

            if (factValue === undefined && guard.skipIfMissingFact) {
                continue;
            }

            const isMonetaryStringGuard = typeof guard.value === 'string' && /^\d+$/.test(guard.value) && (
                guard.valueType === 'CURRENCY_MINOR_UNITS' ||
                /(?:precondition|turnover|threshold)/i.test(guard.id || '') ||
                /(?:paisa|cents|minorunits)/i.test(guard.field || '')
            );
            if (typeof factValue === 'number' && (typeof guard.value === 'bigint' || isMonetaryStringGuard) && !/(?:paisa|cents|minorunits)$/i.test(guard.field)) {
                factValue = BigInt(Math.round(factValue * 100));
            }

            const isMatch = evaluateOperator(factValue, guard.operator, guard.value);

            if (isMatch) {
                trace.push(`Guard matched: ${guard.id}`);
                const onMatch = guard.onMatch;

                if (onMatch?.action === 'HALT_EXEMPT') {
                    const rawInvoice = Number(facts.invoiceAmount || facts.grossAmountInr || facts.lineItem?.grossAmountPaisa || 0);
                    result.disposition = onMatch.disposition || 'EXEMPT';
                    result.statutoryDisposition = onMatch.disposition || 'EXEMPT';
                    result.statutoryReason = onMatch.reason || 'Guard halted computation as exempt';
                    result.isExemptForeignImport = Boolean(facts.isImportOfGoods || facts.sellerIsNonResident);
                    result.invoiceAmountInr = rawInvoice;
                    result.netWireInr = rawInvoice;
                    if (onMatch.governingSection) {
                        result.governingSection = onMatch.governingSection;
                    } else if (guard.id === 'RESIDENT_STATUS_GUARD' || guard.field === 'payee.isResident' || facts.isImportOfGoods || facts.sellerIsNonResident) {
                        result.governingSection = temporal.isAct2025
                            ? 'CBDT Circular No. 13 of 2021 Question 3 (Read with Section 393(2))'
                            : 'CBDT Circular No. 13 of 2021 Question 3 (Read with Section 195)';
                    }
                    return finalize(result);
                }

                if (onMatch?.action === 'OVERRIDE_RATE') {
                    overrideRateBps = onMatch.overrideRateBps;
                    panPenaltySection = temporal.isAct2025
                        ? onMatch.governingSection2025
                        : onMatch.governingSection1961;
                    trace.push(`Override applied: rate set to ${overrideRateBps} bps (${overrideRateBps / 100}%)`);
                }
            } else if (guard.onMismatch) {
                trace.push(`Guard mismatch: ${guard.id}`);
                const onMismatch = guard.onMismatch;

                if (onMismatch.action === 'HALT_EXEMPT') {
                    const rawInvoice = Number(facts.invoiceAmount || facts.grossAmountInr || facts.lineItem?.grossAmountPaisa || 0);
                    result.disposition = onMismatch.disposition || 'EXEMPT';
                    result.statutoryDisposition = onMismatch.disposition || 'EXEMPT';
                    result.statutoryReason = onMismatch.reason || 'Guard failed precondition as exempt';
                    result.invoiceAmountInr = rawInvoice;
                    result.netWireInr = rawInvoice;
                    return finalize(result);
                }

                if (onMismatch.action === 'BLOCK_COMPLIANCE') {
                    result.disposition = 'BLOCKED_NON_COMPLIANT';
                    result.statutoryDisposition = onMismatch.disposition || 'BLOCKED_NON_COMPLIANT';
                    result.statutoryReason = onMismatch.reason || 'Mandatory statutory guard not satisfied';
                    result.isBlocked = true;
                    return finalize(result);
                }
            }
        }
    }

    if (!isPureLogic) {
        result.governingPanPenaltySection = panPenaltySection || (temporal.isAct2025 ? 'Section 397 (PAN compliant)' : 'Section 206AA (PAN compliant)');
    }

    const archetype = activeAst.computation?.archetype || 'EXCESS_OVER_THRESHOLD';

    // ── 5A. Universal Declarative Pipeline Execution ─────────────────────────
    const pipeline = activeAst.computation?.pipeline || (Array.isArray(activeAst.computation?.steps) ? activeAst.computation : null);
    if (pipeline || archetype === 'DECLARATIVE_PIPELINE' || archetype === 'UNIVERSAL_PIPELINE' || archetype === 'UNIVERSAL_EXPRESSION' || archetype === 'PURE_LOGIC_PIPELINE') {
        const safeNum = (val) => {
            if (val === undefined || val === null || typeof val === 'object' || typeof val === 'boolean') return 0;
            try {
                const cleaned = cleanNumericAmount(val);
                return Number.isFinite(cleaned) ? cleaned : 0;
            } catch {
                return 0;
            }
        };
        const rawInvoiceInr = safeNum(facts.invoiceAmount ?? facts.grossAmountInr ?? facts.principal ?? 0);
        const rawPrimaryBase = safeNum(
            facts.invoiceAmount ??
            facts.grossAmountInr ??
            facts.principal ??
            facts.taxableIncome ??
            facts.taxableIncomeUsd ??
            facts.annualTaxableIncome ??
            facts.adjustedGrossIncome ??
            facts.adjustedGrossIncomeUsd ??
            facts.agiUsd ??
            facts.grossIncome ??
            facts.wages ??
            facts.wagesUsd ??
            facts.grossReceipts ??
            facts.annualIncome ??
            facts.compensation ??
            facts.baseAmount ??
            facts.amount ??
            0
        );
        const pipelineContext = {
            facts: {
                ...facts,
                invoiceAmountInr: rawInvoiceInr,
                invoiceAmountPaisa: toBigIntPaise(rawInvoiceInr),
                currentInvoicePaisa: toBigIntPaise(rawInvoiceInr),
                primaryBasePaisa: facts.primaryBasePaisa !== undefined
                    ? BigInt(facts.primaryBasePaisa)
                    : toBigIntPaise(rawPrimaryBase),
                priorPurchasesPaisa: facts.priorPurchasesPaisa !== undefined
                    ? BigInt(facts.priorPurchasesPaisa)
                    : toBigIntPaise(safeNum(facts.priorPurchasesInCurrentFyInr ?? facts.priorPurchases ?? 0))
            },
            rules: {
                ...activeAst.computation,
                standardRateBps: activeAst.computation?.standardRateBps ?? 0,
                thresholdPaisa: activeAst.computation?.thresholdPaisa !== undefined ? BigInt(activeAst.computation.thresholdPaisa) : 0n,
                effectiveRateBps: overrideRateBps !== null ? overrideRateBps : (activeAst.computation?.standardRateBps ?? 0)
            },
            vars: {
                overrideRateBps,
                effectiveRateBps: overrideRateBps !== null ? overrideRateBps : (activeAst.computation?.standardRateBps ?? 0)
            },
            temporal,
            trace
        };

        const pipeOutputs = executeDeclarativePipeline(pipeline || activeAst.computation, pipelineContext);

        Object.assign(result, pipeOutputs);

        if (!isPureLogic) {
            if (result.taxableBasePaisa !== undefined) {
                result.taxableBase = fromBigIntPaise(result.taxableBasePaisa);
            }
            if (result.tdsAmountPaisa !== undefined) {
                result.calculatedTdsInr = fromBigIntPaise(result.tdsAmountPaisa);
            }
            if (result.taxAmountPaisa !== undefined) {
                result.calculatedTaxInr = fromBigIntPaise(result.taxAmountPaisa);
            } else if (result.calculatedTaxInr === 0 && result.calculatedTdsInr > 0) {
                result.calculatedTaxInr = result.calculatedTdsInr;
            }
            if ((result.invoiceAmountInr === 0 || result.invoiceAmountInr === undefined) && rawInvoiceInr > 0) {
                result.invoiceAmountInr = rawInvoiceInr;
            }
            if ((result.netWireInr === 0 || result.netWireInr === undefined) && rawInvoiceInr > 0) {
                const tax = result.calculatedTdsInr || result.calculatedTaxInr || 0;
                result.netWireInr = Math.max(0, rawInvoiceInr - tax);
            }
            if (result.isTdsApplicable === false && (result.calculatedTdsInr > 0 || result.calculatedTaxInr > 0)) {
                result.isTdsApplicable = true;
            }
            if (
                pipeOutputs.effectiveRateBps === undefined &&
                (result.isApplicable || result.isTdsApplicable || result.calculatedTdsInr > 0 || result.calculatedTaxInr > 0) &&
                (result.tdsAmountPaisa !== undefined || result.taxAmountPaisa !== undefined || result.calculatedTdsInr > 0 || result.calculatedTaxInr > 0)
            ) {
                const rateBps = pipelineContext.rules.effectiveRateBps ?? 0;
                if (rateBps > 0) {
                    result.effectiveRateBps = rateBps;
                    result.tdsRatePercent = rateBps / 100;
                }
            }
            if (result.effectiveRateBps > 0 && pipeOutputs.tdsRatePercent === undefined) {
                result.tdsRatePercent = result.effectiveRateBps / 100;
            }
            if (result.disposition === 'EVALUATING') {
                result.disposition = (result.calculatedTdsInr > 0 || result.calculatedTaxInr > 0 || result.isApplicable)
                    ? 'TAX_DEDUCTIBLE'
                    : 'EXEMPT_BELOW_STATUTORY_THRESHOLD';
            }
            if (!result.statutoryDisposition) {
                result.statutoryDisposition = result.disposition;
            }
        } else {
            if (pipeOutputs.isProhibited !== undefined) {
                result.isProhibited = Boolean(pipeOutputs.isProhibited);
                if (result.isBlocked === undefined) {
                    result.isBlocked = result.isProhibited;
                }
            }
            if (pipeOutputs.isPermitted !== undefined) {
                result.isPermitted = Boolean(pipeOutputs.isPermitted);
                if (result.isCompliant === undefined) {
                    result.isCompliant = result.isPermitted;
                }
            }
            if (result.disposition === 'EVALUATING') {
                if (result.isProhibited === true) {
                    result.disposition = 'PROHIBITED';
                } else if (pipeOutputs.isPermitted !== undefined) {
                    result.disposition = result.isPermitted ? 'PERMITTED' : 'NOT_PERMITTED';
                } else if (result.isApplicable || result.isCompliant || pipeOutputs.isSatisfied) {
                    result.disposition = 'RULE_SATISFIED';
                } else {
                    result.disposition = 'RULE_NOT_SATISFIED';
                }
            }
            if (!result.statutoryDisposition) {
                result.statutoryDisposition = result.disposition;
            }
        }

        return finalize(result);
    }

    // ── 5B. Universal Declarative Deontic State Gate Execution ───────────────
    const hasExplicitDeonticGate = Boolean(activeAst.deonticGate || activeAst.computation?.deonticGate);
    const hasNoWithholdingRatesOrThresholds = activeAst.computation?.standardRateBps === undefined && activeAst.computation?.thresholdPaisa === undefined;
    const isDeonticArchetype = DEONTIC_ARCHETYPES.has(archetype) && (hasExplicitDeonticGate || hasNoWithholdingRatesOrThresholds);
    const isDeonticRootOnly = hasNoWithholdingRatesOrThresholds && DEONTIC_NODE_FALLBACKS.has(activeAst.rootCondition?.nodeType);

    if (hasExplicitDeonticGate || isDeonticArchetype || isDeonticRootOnly) {
        const deonticResult = evaluateDeonticStateGate(activeAst, facts, temporal, trace, result);
        return finalize(deonticResult);
    }

    // ── 5C. Domain Archetype Dispatches ──────────────────────────────────────
    if (archetype === 'COMPOUND_INTEREST_MONTHLY_RESTS') {
        return finalize(evaluateMsmeDelayedPaymentCompoundInterest(activeAst, facts, trace));
    }
    if (archetype === 'MSME_FISCAL_BOUNDARY_43BH') {
        return finalize(evaluateSection43bHFiscalBoundary(activeAst, facts, trace));
    }
    if (archetype === 'ENTERPRISE_CLASSIFICATION_MSME') {
        return finalize(evaluateMsmeEnterpriseClassification(activeAst, facts, trace));
    }
    if (archetype === 'PROPORTIONAL_REVERSAL_180_DAYS') {
        return finalize(evaluateGstRule37ProportionalReversal(activeAst, facts, trace));
    }
    if (archetype === 'SUPPLIER_NON_FILING_RULE37A') {
        return finalize(evaluateGstRule37ASupplierNonFiling(activeAst, facts, trace));
    }
    if (archetype === 'COMMON_CREDIT_APPORTIONMENT_RULE42') {
        return finalize(evaluateGstRule42CommonCreditApportionment(activeAst, facts, trace));
    }
    if (archetype === 'DTAA_TREATY_TAX_HAVEN_MATRIX') {
        return finalize(evaluateCrossBorderDtaaMatrix(activeAst, facts, temporal, trace));
    }
    if (archetype === 'TRANSFER_PRICING_PERCENTILE_ALP') {
        return finalize(evaluateTransferPricingPercentileAlp(activeAst, facts, trace));
    }
    if (archetype === 'VOTING_CONTINUITY_PERCENTAGE') {
        return finalize(evaluateSection79VotingContinuity(activeAst, facts, trace));
    }
    if (archetype === 'TAX_CLEARANCE_VOID_TRANSFER_SEC281') {
        return finalize(evaluateSection281VoidTransfer(activeAst, facts, trace));
    }
    if (archetype === 'CORPORATE_TAX_MAT_VS_CONCESSIONAL') {
        return finalize(evaluateCorporateTaxMatVsConcessionalUsvm(activeAst, facts, trace));
    }
    if (archetype === 'SALARY_SLAB_PROGRESSIVE_TDS') {
        return finalize(evaluateSalaryProgressiveTds(activeAst, facts, trace));
    }
    if (archetype === 'NON_FILER_PENAL_RATE_206AB') {
        return finalize(evaluateNonFilerPenalRate206AB(activeAst, facts, trace));
    }
    if (archetype === 'DEFERRED_TAX_TIMING_DIFFERENCE') {
        return finalize(evaluateDeferredTaxIndAs12Usvm(activeAst, facts, trace));
    }
    if (archetype === 'BLOCKED_CREDIT_GATE_17_5') {
        return finalize(evaluateBlockedCreditSection17_5(activeAst, facts, trace));
    }
    if (archetype === 'ITC_ELIGIBILITY_GATE_16_2') {
        return finalize(evaluateItcEligibilitySection16_2(activeAst, facts, trace));
    }
    if (archetype === 'RCM_LIABILITY_EVALUATION_9_3') {
        return finalize(evaluateRcmLiabilitySection9_3(activeAst, facts, trace));
    }
    if (archetype === 'COMPOSITE_MIXED_SUPPLY_SEC8') {
        return finalize(evaluateCompositeMixedSuppliesSection8(activeAst, facts, trace));
    }
    if (archetype === 'COMPOSITION_LEVY_SEC10') {
        return finalize(evaluateCompositionLevySection10(activeAst, facts, trace));
    }
    if (archetype === 'CREDIT_NOTE_TIMELINES_SEC34') {
        return finalize(evaluateCreditDebitNotesSection34(activeAst, facts, trace));
    }
    if (archetype === 'EXPORT_LUT_ZERO_RATED_SEC16') {
        return finalize(evaluateExportLutSection16(activeAst, facts, trace));
    }
    if (archetype === 'EINVOICING_MANDATE_RULE48_4') {
        return finalize(evaluateEInvoicingRule48_4(activeAst, facts, trace));
    }
    if (archetype === 'EWAYBILL_CONSIGNMENT_RULE138') {
        return finalize(evaluateEWayBillRule138(activeAst, facts, trace));
    }
    if (archetype === 'GST_TDS_DEDUCTION_SEC51') {
        return finalize(evaluateGstTdsSection51(activeAst, facts, trace));
    }
    if (archetype === 'MCA_MSME_1_AGING_BIFURCATION') {
        return finalize(evaluateMcaMsme1Aging(activeAst, facts, trace));
    }
    if (archetype === 'MSME_DISCLOSURES_SEC22') {
        return finalize(evaluateMsmeDisclosuresSection22(activeAst, facts, trace));
    }
    if (archetype === 'LEASE_AMORTIZATION_INDAS116') {
        return finalize(evaluateLeaseAmortizationIndAs116Usvm(activeAst, facts, trace));
    }
    if (archetype === 'REVENUE_ALLOCATION_INDAS115') {
        return finalize(evaluateRevenueAllocationIndAs115Usvm(activeAst, facts, trace));
    }
    if (archetype === 'PROVISIONS_CONTINGENCY_INDAS37') {
        return finalize(evaluateProvisionsAndContingenciesIndAs37Usvm(activeAst, facts, trace));
    }
    if (archetype === 'FX_SPOT_REVALUATION_INDAS21_109') {
        return finalize(evaluateForeignExchangeRevaluationUsvm(activeAst, facts, trace));
    }
    if (archetype === 'EXPECTED_CREDIT_LOSS_INDAS109') {
        return finalize(evaluateExpectedCreditLossIndAs109Usvm(activeAst, facts, trace));
    }
    if (archetype === 'FOREIGN_TAX_CREDIT_RELIEF') {
        return finalize(evaluateForeignTaxCreditRelief(activeAst, facts, trace));
    }
    if (archetype === 'SLUMP_SALE_NET_WORTH_50B') {
        return finalize(evaluateSlumpSaleSection50B(activeAst, facts, trace));
    }
    if (archetype === 'BUSINESS_COMBINATIONS_INDAS103') {
        return finalize(evaluateBusinessCombinationsIndAs103Usvm(activeAst, facts, trace));
    }
    if (archetype === 'IMPAIRMENT_TESTING_INDAS36') {
        return finalize(evaluateImpairmentIndAs36Usvm(activeAst, facts, trace));
    }
    if (archetype === 'SHARE_BASED_PAYMENTS_INDAS102') {
        return finalize(evaluateShareBasedPaymentsIndAs102Usvm(activeAst, facts, trace));
    }
    if (archetype === 'CASH_FLOWS_INDAS7') {
        return finalize(evaluateCashFlowsIndAs7Usvm(activeAst, facts, trace));
    }
    if (archetype === 'PPE_DEPRECIATION_INDAS16') {
        return finalize(evaluatePropertyPlantEquipmentIndAs16Usvm(activeAst, facts, trace));
    }
    if (archetype === 'EMPLOYEE_BENEFITS_INDAS19') {
        return finalize(evaluateEmployeeBenefitsIndAs19Usvm(activeAst, facts, trace));
    }
    if (archetype === 'INVENTORY_VALUATION_INDAS2') {
        return finalize(evaluateInventoryValuationIndAs2Usvm(activeAst, facts, trace));
    }
    if (archetype === 'INVESTMENTS_ASSOCIATES_INDAS28') {
        return finalize(evaluateEquityMethodIndAs28Usvm(activeAst, facts, trace));
    }
    if (archetype === 'EARNINGS_PER_SHARE_INDAS33') {
        return finalize(evaluateEarningsPerShareIndAs33Usvm(activeAst, facts, trace));
    }
    if (archetype === 'INTANGIBLES_CAPITALIZATION_INDAS38') {
        return finalize(evaluateIntangiblesCapitalizationIndAs38Usvm(activeAst, facts, trace));
    }
    if (archetype === 'CONSOLIDATION_NCI_INDAS110') {
        return finalize(evaluateConsolidationNciIndAs110Usvm(activeAst, facts, trace));
    }
    if (archetype === 'FAIR_VALUE_HIERARCHY_INDAS113') {
        return finalize(evaluateFairValueHierarchyIndAs113Usvm(activeAst, facts, trace));
    }
    if (archetype === 'TDS_VDA_CRYPTO_194S') {
        return finalize(evaluateSection194SUsvm(activeAst, facts, trace));
    }
    if (archetype === 'TDS_ONLINE_GAMING_194BA') {
        return finalize(evaluateSection194BAUsvm(activeAst, facts, trace));
    }
    if (archetype === 'TDS_CASH_WITHDRAWAL_194N') {
        return finalize(evaluateSection194NUsvm(activeAst, facts, trace));
    }
    if (archetype === 'TDS_LIFE_INSURANCE_194DA') {
        return finalize(evaluateSection194DAUsvm(activeAst, facts, trace));
    }
    if (archetype === 'TDS_MUTUAL_FUNDS_194K') {
        return finalize(evaluateSection194KUsvm(activeAst, facts, trace));
    }
    if (archetype === 'TCS_REMITTANCE_LRS_206C1G') {
        return finalize(evaluateSection206C1GUsvm(activeAst, facts, trace));
    }
    if (archetype === 'GST_TCS_ECOMMERCE_SEC52') {
        return finalize(evaluateSection52GstTcsUsvm(activeAst, facts, trace));
    }

    // ── 6. Ingress Invoice Consideration Quantization ────────────────────────
    let currentInvoicePaise = 0n;
    let rawInvoiceInr = 0;

    if (facts.lineItem?.grossAmountPaisa !== undefined) {
        currentInvoicePaise = BigInt(facts.lineItem.grossAmountPaisa);
        rawInvoiceInr = fromBigIntPaise(currentInvoicePaise);
    } else {
        rawInvoiceInr = cleanNumericAmount(facts.invoiceAmount ?? facts.grossAmountInr ?? 0);
        const gstAmountInr = cleanNumericAmount(facts.gstAmountInr || 0);
        const isAdvancePayment = Boolean(facts.isAdvancePayment);
        const creditNoteInr = cleanNumericAmount(facts.creditNoteInr || 0);
        const creditNoteTiming = facts.creditNoteTiming || 'BEFORE_TAX_DEDUCTION';

        const effectivePurchaseInr = isAdvancePayment
            ? rawInvoiceInr
            : Math.max(0, rawInvoiceInr - gstAmountInr);

        const isCreditNoteBefore = creditNoteTiming !== 'AFTER_TAX_DEDUCTION';
        const amountInr = isCreditNoteBefore
            ? Math.max(0, effectivePurchaseInr - creditNoteInr)
            : effectivePurchaseInr;

        currentInvoicePaise = toBigIntPaise(amountInr);
    }

    // ── 7. Cumulative Threshold Consideration Evaluation ─────────────────────
    let priorPurchasesPaise = 0n;
    if (facts.aggregateSnapshot?.aggregatePaisa !== undefined) {
        priorPurchasesPaise = BigInt(facts.aggregateSnapshot.aggregatePaisa);
    } else {
        const priorPurchasesInr = cleanNumericAmount(
            facts.priorPurchasesInCurrentFyInr ??
            facts.currentFyPriorPurchasesInr ??
            facts.priorFyPurchases ??
            0
        );
        priorPurchasesPaise = toBigIntPaise(priorPurchasesInr);
    }

    const thresholdPaise = activeAst.computation?.thresholdPaisa !== undefined
        ? BigInt(activeAst.computation.thresholdPaisa)
        : 500000000n;

    const cumulativeBeforePaise = priorPurchasesPaise;
    const cumulativeAfterPaise = priorPurchasesPaise + currentInvoicePaise;

    let taxableBasePaise = 0n;
    let isThresholdBreached = false;

    if (archetype === 'EXCESS_OVER_THRESHOLD') {
        if (cumulativeAfterPaise <= thresholdPaise) {
            taxableBasePaise = 0n;
            isThresholdBreached = false;
        } else if (cumulativeBeforePaise >= thresholdPaise) {
            taxableBasePaise = currentInvoicePaise;
            isThresholdBreached = true;
        } else {
            taxableBasePaise = cumulativeAfterPaise - thresholdPaise;
            isThresholdBreached = true;
        }
    } else if (archetype === 'CLIFF_THRESHOLD') {
        if (cumulativeAfterPaise > thresholdPaise) {
            taxableBasePaise = currentInvoicePaise;
            isThresholdBreached = true;
        }
    } else if (archetype === 'FLAT_RATE') {
        taxableBasePaise = currentInvoicePaise;
        isThresholdBreached = true;
    }

    // ── 8. Rate Resolution & 5-Stage Rational Arithmetic ─────────────────────
    const standardRateBps = activeAst.computation?.standardRateBps ?? 10;
    const effectiveRateBps = overrideRateBps !== null ? overrideRateBps : standardRateBps;

    let lineItemTdsPaise = 0n;

    if (taxableBasePaise > 0n && effectiveRateBps > 0) {
        const { quotientPaise, subPaisaRemainder } = multiplyRational(taxableBasePaise, effectiveRateBps, 10000n);
        lineItemTdsPaise = quantizeCurrencyPaisa(quotientPaise, subPaisaRemainder, 10000n);
    }

    const assessmentRoundingPolicy = temporal.isAct2025 ? 'INCOME_TAX_ACT_2025_SEC516' : 'INCOME_TAX_ACT_1961_SEC288B';
    const assessmentTdsPaise = roundStatutoryAssessment(lineItemTdsPaise, assessmentRoundingPolicy);

    // ── 9. Final Fiduciary Return Formulation ────────────────────────────────
    const taxableBaseInr = fromBigIntPaise(taxableBasePaise);
    const calculatedTdsInr = fromBigIntPaise(lineItemTdsPaise);
    const netWireInr = Math.max(0, rawInvoiceInr - calculatedTdsInr);

    result.invoiceAmountInr = rawInvoiceInr;
    result.taxableBase = taxableBaseInr;
    result.tdsRatePercent = effectiveRateBps / 100;
    result.calculatedTdsInr = calculatedTdsInr;
    result.calculatedTaxInr = calculatedTdsInr;
    result.assessmentRoundedTdsInr = fromBigIntPaise(assessmentTdsPaise);
    result.netWireInr = netWireInr;
    result.isTdsApplicable = calculatedTdsInr > 0;
    result.isApplicable = isThresholdBreached;
    result.isSection194QApplicable = isThresholdBreached;
    result.effectiveRateBps = effectiveRateBps;
    result.disposition = isThresholdBreached ? 'TAX_DEDUCTIBLE' : 'THRESHOLD_NOT_BREACHED';
    result.statutoryDisposition = isThresholdBreached ? 'SECTION_194Q_DEDUCTION_MANDATORY' : 'EXEMPT_BELOW_STATUTORY_THRESHOLD';

    return finalize(result);
}
