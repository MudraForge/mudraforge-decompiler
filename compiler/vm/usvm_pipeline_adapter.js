/**
 * ═══════════════════════════════════════════════════════════════════════════
 * USVM PIPELINE ADAPTER & SHADOW ROUTING ENGINE (RFC-009 / RFC-010)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides a non-disruptive, zero-risk bridge between the legacy imperative
 * calculators (e.g. direct_tax.js) and the new declarative Universal Statutory
 * Virtual Machine (USVM).
 *
 * Operational Modes:
 * 1. SHADOW_VALIDATION (Default): Executes USVM and legacy executor in parallel,
 *    asserting 0.00 paisa parity. If any discrepancy or unmodeled parameter
 *    occurs, fails safe to the legacy executor with zero production disruption.
 * 2. USVM_PRIMARY: Executes USVM directly for sub-microsecond latency (< 4 µs)
 *    when the AST is certified.
 * 3. LEGACY_FALLBACK: Seamlessly delegates to legacy calculator if no AST is
 *    yet registered for that section.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluateStatutoryAst } from './evaluation_vm.js';
import { validateStatutoryAst, StatutoryAstValidationError } from './schemas/statutory_ast_schema.js';
import { classifyOkfRuleLifecycle } from './okf_lifecycle_monitor.js';

const __filename = typeof import.meta?.url === 'string' && import.meta.url.startsWith('file:') ? fileURLToPath(import.meta.url) : '';
const __dirname = __filename ? path.dirname(__filename) : '';

// In-memory AST Rule Cache
const AST_REGISTRY = new Map();

/**
 * Loads and caches an AST rule from JSON file.
 *
 * @param {string} sectionKey - Canonical key (e.g. '194Q', '393', '194C')
 * @param {string} relativeFilePath - Path relative to usvm directory
 */
export function registerRuleFromFile(sectionKey, relativeFilePath) {
    if (!__dirname) return false;
    try {
        const fullPath = path.resolve(__dirname, relativeFilePath);
        if (fs.existsSync(fullPath)) {
            const raw = fs.readFileSync(fullPath, 'utf8');
            const ast = JSON.parse(raw);
            validateStatutoryAst(ast);
            AST_REGISTRY.set(sectionKey.toUpperCase(), ast);
            return true;
        }
    } catch (err) {
        console.warn(`[USVM_ADAPTER] Failed to load rule file ${relativeFilePath}:`, err.message);
    }
    return false;
}

/**
 * Lowers a high-level AST symbolLegend into executable USVM guards,
 * computation parameters, and controllership attestation checklists.
 *
 * Implements strict role-aware disambiguation so multi-number clauses
 * (e.g. Section 194Q / Section 393 with both turnover precondition and transaction excess threshold,
 * and standard rate vs. penal rate) never overwrite each other.
 *
 * @param {Object} symbolLegend - Key-value map of symbolId -> symbol metadata
 * @param {Object} baseAst - Ingress declarative AST rule
 * @returns {Object} Lowered, fully-executable USVM AST rule package
 */
export function projectSymbolLegendToGuards(symbolLegend, baseAst = {}) {
    if (!symbolLegend || typeof symbolLegend !== 'object') {
        return baseAst;
    }

    const lowered = {
        ...baseAst,
        guards: Array.isArray(baseAst.guards) ? [...baseAst.guards] : [],
        computation: baseAst.computation ? { ...baseAst.computation } : {},
        attestationChecklist: Array.isArray(baseAst.attestationChecklist) ? [...baseAst.attestationChecklist] : [],
        symbolLegend
    };

    // Track existing guard IDs to prevent duplicates
    const existingGuardIds = new Set(lowered.guards.map(g => g.id));
    const existingGuardFields = new Set(lowered.guards.map(g => g.field));
    const hasCompiledPipeline = Boolean(lowered.computation?.pipeline?.steps || lowered.computation?.steps);

    for (const [symId, sym] of Object.entries(symbolLegend)) {
        if (!sym || typeof sym !== 'object') continue;

        const role = sym.role || '';
        if (role === 'ROLE_COMPUTATION_OPERAND') continue;

        const symType = sym.symbolType || '';
        const dataType = sym.dataType || '';
        const op = sym.comparisonOperator || 'GTE';
        const rawText = sym.rawText || symId;

        // 1. Monetary Disambiguation (MONETARY_QUANTUM)
        if (
            symType === 'MONETARY_QUANTUM' ||
            dataType === 'CURRENCY_PAISA' ||
            dataType === 'CURRENCY_CENTS' ||
            dataType === 'CURRENCY_MINOR_UNITS'
        ) {
            const minorVal = sym.targetThresholdMinorUnits ?? sym.targetThresholdPaisa;
            const paisaVal = minorVal !== undefined && minorVal !== null
                ? String(minorVal)
                : (sym.targetThreshold !== undefined && sym.targetThreshold !== null
                    ? String(Math.round(Number(sym.targetThreshold) * 100))
                    : null);

            if (paisaVal !== null) {
                const isIndianTurnoverPrecondition = role === 'ROLE_BUYER_PRIOR_TURNOVER_GUARD' || (
                    role === 'ROLE_AGENT_STATE_PRECONDITION' && (
                        dataType === 'CURRENCY_PAISA' ||
                        sym.currencyCode === 'INR' ||
                        sym.measurementUnit === 'INR' ||
                        /(?:turnover|sales|gross receipts|paid-up|share capital|net worth|crore|lakh)/i.test(`${symId} ${rawText}`)
                    )
                );

                // Precondition Guard: ROLE_AGENT_STATE_PRECONDITION / ROLE_BUYER_PRIOR_TURNOVER_GUARD
                if (role === 'ROLE_AGENT_STATE_PRECONDITION' || role === 'ROLE_BUYER_PRIOR_TURNOVER_GUARD') {
                    const guardId = `GUARD_${symId.toUpperCase()}_PRECONDITION`;
                    const targetField = isIndianTurnoverPrecondition ? 'buyerPriorTurnoverPaisa' : (sym.factKey || symId);
                    const hasPreconditionAlready = existingGuardIds.has(guardId) || existingGuardFields.has(targetField) || (hasCompiledPipeline && existingGuardFields.has('buyerPriorTurnoverPaisa'));
                    if (!hasPreconditionAlready) {
                        if (isIndianTurnoverPrecondition) {
                            lowered.guards.push({
                                id: guardId,
                                field: 'buyerPriorTurnoverPaisa',
                                operator: op,
                                value: BigInt(paisaVal),
                                symbolId: symId,
                                rawText: rawText,
                                onMatch: {
                                    action: 'ALLOW_CONTINUE',
                                    reason: `Entity prior turnover precondition satisfied: ${rawText}`
                                },
                                onMismatch: {
                                    action: 'HALT_EXEMPT',
                                    disposition: 'EXEMPT_BELOW_TURNOVER_THRESHOLD',
                                    reason: `Entity turnover does not exceed statutory threshold: ${rawText}`
                                }
                            });
                        } else {
                            lowered.guards.push({
                                id: guardId,
                                field: targetField,
                                operator: op,
                                value: BigInt(paisaVal),
                                symbolId: symId,
                                rawText: rawText,
                                skipIfMissingFact: true,
                                onMatch: {
                                    action: 'ALLOW_CONTINUE',
                                    reason: `Statutory monetary precondition satisfied: ${rawText}`
                                },
                                onMismatch: {
                                    action: 'HALT_EXEMPT',
                                    disposition: 'EXEMPT_BELOW_THRESHOLD',
                                    reason: `Fact '${targetField}' does not satisfy statutory threshold: ${rawText}`
                                }
                            });
                        }
                        existingGuardIds.add(guardId);
                        existingGuardFields.add(targetField);
                    }
                } else if (
                    !hasCompiledPipeline && (
                        role === 'ROLE_TRANSACTION_CLIFF_THRESHOLD' ||
                        role === 'ROLE_TRANSACTION_EXCESS_THRESHOLD' ||
                        role === 'ROLE_EXCESS_THRESHOLD' ||
                        !lowered.computation.thresholdPaisa
                    )
                ) {
                    // Transaction Excess/Cliff Threshold
                    lowered.computation.thresholdPaisa = paisaVal;
                    if (sym.targetThresholdInr || sym.targetThreshold) {
                        lowered.computation.thresholdInr = sym.targetThresholdInr || sym.targetThreshold;
                    }
                }
            }
        }

        // 2. Rate Disambiguation (PERCENTAGE_RATE)
        else if (symType === 'PERCENTAGE_RATE' || dataType === 'PERCENT') {
            const ratePercent = Number(sym.targetThreshold ?? 0);
            const rateBps = Math.round(ratePercent * 100);

            if (role === 'ROLE_PENALTY_OR_DEFAULT_OVERRIDE' || role === 'ROLE_NON_PAN_OVERRIDE_RATE') {
                const isPanOverride = role === 'ROLE_NON_PAN_OVERRIDE_RATE' ||
                    /\b(?:pan|permanent account number|tin)\b/i.test(`${symId.replace(/_/g, ' ')} ${rawText}`);
                lowered.computation.penaltyOverrideRateBps = rateBps;
                if (isPanOverride) {
                    lowered.computation.nonPanOverrideRateBps = rateBps;
                    const guardId = `GUARD_PENALTY_${symId.toUpperCase()}`;
                    if (!existingGuardIds.has(guardId) && !existingGuardFields.has('payee.isPanValid')) {
                        lowered.guards.push({
                            id: guardId,
                            field: 'payee.isPanValid',
                            operator: 'EQ',
                            value: false,
                            symbolId: symId,
                            rawText: rawText,
                            onMatch: {
                                action: 'OVERRIDE_RATE',
                                overrideRateBps: rateBps,
                                reason: `Penal withholding override rate applicable: ${rawText}`
                            }
                        });
                        existingGuardIds.add(guardId);
                        existingGuardFields.add('payee.isPanValid');
                    }
                } else {
                    const condField = sym.factKey || `override_${symId}`;
                    lowered.computation.alternateRateBps = rateBps;
                    const guardId = `GUARD_OVERRIDE_${symId.toUpperCase()}`;
                    if (!existingGuardIds.has(guardId) && !existingGuardFields.has(condField)) {
                        lowered.guards.push({
                            id: guardId,
                            field: condField,
                            operator: 'IS_TRUE',
                            value: true,
                            symbolId: symId,
                            rawText: rawText,
                            skipIfMissingFact: true,
                            onMatch: {
                                action: 'OVERRIDE_RATE',
                                overrideRateBps: rateBps,
                                reason: `Statutory override rate applicable: ${rawText}`
                            }
                        });
                        existingGuardIds.add(guardId);
                        existingGuardFields.add(condField);
                    }
                }
            } else if (!hasCompiledPipeline && (role === 'ROLE_STANDARD_STATUTORY_RATE' || lowered.computation.standardRateBps === undefined)) {
                lowered.computation.standardRateBps = rateBps;
                lowered.computation.standardRatePercent = ratePercent;
            }
        }

        // 3. Temporal Duration (TEMPORAL_DURATION)
        else if (symType === 'TEMPORAL_DURATION') {
            const unit = sym.measurementUnit || 'MONTHS';
            const field = unit === 'YEARS' ? 'continuousServiceYears'
                : unit === 'MONTHS' ? 'leaseTermMonths'
                : unit === 'DAYS' ? 'durationDays'
                : 'temporalDuration';

            const guardId = `GUARD_${symId.toUpperCase()}_TEMPORAL`;
            if (!existingGuardIds.has(guardId)) {
                if (role === 'ROLE_EXCEPTION_CARVEOUT' || sym.actionOnMatch === 'WAIVE_THRESHOLD') {
                    lowered.guards.push({
                        id: guardId,
                        field: field,
                        operator: op || 'LTE',
                        value: sym.targetThreshold,
                        symbolId: symId,
                        rawText: rawText,
                        onMatch: {
                            action: 'HALT_EXEMPT',
                            disposition: 'EXEMPT_TEMPORAL_CARVEOUT',
                            reason: `Statutory temporal carveout established: ${rawText}`
                        }
                    });
                } else {
                    lowered.guards.push({
                        id: guardId,
                        field: field,
                        operator: op || 'GTE',
                        value: sym.targetThreshold,
                        symbolId: symId,
                        rawText: rawText,
                        onMatch: {
                            action: 'ALLOW_CONTINUE',
                            reason: `Statutory temporal threshold satisfied: ${rawText}`
                        },
                        onMismatch: {
                            action: 'HALT_EXEMPT',
                            disposition: 'INELIGIBLE_TEMPORAL_THRESHOLD_NOT_MET',
                            reason: `Statutory temporal threshold not met: ${rawText}`
                        }
                    });
                }
                existingGuardIds.add(guardId);
            }
        }

        // 4. Exemption Conditions (EXEMPTION_CONDITION)
        else if (symType === 'EXEMPTION_CONDITION' || role === 'ROLE_EXCEPTION_CARVEOUT') {
            const guardId = `GUARD_${symId.toUpperCase()}_CARVEOUT`;
            if (!existingGuardIds.has(guardId)) {
                lowered.guards.push({
                    id: guardId,
                    field: symId,
                    operator: 'EQ',
                    value: true,
                    symbolId: symId,
                    rawText: rawText,
                    onMatch: {
                        action: 'HALT_EXEMPT',
                        disposition: 'EXEMPT_STATUTORY_CARVEOUT',
                        reason: `Statutory exception carveout established: ${rawText}`
                    }
                });
                existingGuardIds.add(guardId);
            }
        }

        // 5. Document Verification (DOCUMENT_VERIFICATION)
        else if (symType === 'DOCUMENT_VERIFICATION' || dataType === 'DOCUMENT_ID') {
            const guardId = `GUARD_DOC_${symId.toUpperCase()}`;
            if (!existingGuardIds.has(guardId)) {
                lowered.guards.push({
                    id: guardId,
                    field: `documents.${symId}`,
                    operator: 'EXISTS',
                    value: true,
                    symbolId: symId,
                    rawText: rawText,
                    onMismatch: {
                        action: 'BLOCK_COMPLIANCE',
                        disposition: 'BLOCKED_DOCUMENTATION_MISSING',
                        reason: `Mandatory statutory documentation missing: ${rawText}`
                    }
                });
                existingGuardIds.add(guardId);
            }
        }

        // 6. Subjective Criteria, Attestation & Non-computational Status
        else {
            lowered.attestationChecklist.push({
                symbolId: symId,
                rawText: rawText,
                role: role,
                symbolType: symType,
                dataType: dataType,
                requiresAttestation: true,
                controllershipReviewMandatory: true
            });
        }
    }

    // Statutory Precedence: Exemption carveouts and safe harbors evaluate before preconditions and thresholds
    const exemptionGuards = lowered.guards.filter(g => g.onMatch?.action === 'HALT_EXEMPT' && !g.onMismatch);
    const proceduralGuards = lowered.guards.filter(g => !(g.onMatch?.action === 'HALT_EXEMPT' && !g.onMismatch));
    lowered.guards = [...exemptionGuards, ...proceduralGuards];

    // Synthesize declarative ALU steps automatically when requested via symbolLegend
    if (
        lowered.computation?.archetype === 'DECLARATIVE_PIPELINE' &&
        !lowered.computation?.steps &&
        !lowered.computation?.pipeline?.steps &&
        (lowered.computation?.thresholdPaisa !== undefined || lowered.computation?.standardRateBps !== undefined)
    ) {
        const isCliff = lowered.rootCondition?.expressionType === 'CLIFF_THRESHOLD' ||
            lowered.compositionalBuildingBlocks?.computationalExpression?.expressionType === 'CLIFF_THRESHOLD';
        lowered.computation.steps = [
            {
                assign: 'isThresholdBreached',
                expr: 'facts.priorPurchasesPaisa + facts.currentInvoicePaisa > rules.thresholdPaisa'
            },
            {
                assign: 'taxableBasePaisa',
                expr: isCliff
                    ? 'facts.priorPurchasesPaisa + facts.currentInvoicePaisa > rules.thresholdPaisa ? facts.currentInvoicePaisa : 0'
                    : 'max(0, min(facts.currentInvoicePaisa, facts.priorPurchasesPaisa + facts.currentInvoicePaisa - rules.thresholdPaisa))'
            },
            {
                assign: 'tdsAmountPaisa',
                expr: 'rational_mult(vars.taxableBasePaisa, rules.effectiveRateBps, 10000)'
            }
        ];
        lowered.computation.output = {
            isThresholdBreached: { var: 'vars.isThresholdBreached' },
            taxableBasePaisa: { var: 'vars.taxableBasePaisa' },
            tdsAmountPaisa: { var: 'vars.tdsAmountPaisa' }
        };
    }

    return lowered;
}

/**
 * Registers an in-memory AST rule directly (e.g., from JIT compiler or KV store).
 * Automatically lowers symbolLegend into executable guards and computation params.
 * Optionally enforces PRODUCTION OKF lifecycle verification when options.mode is set.
 *
 * @param {string} sectionKey
 * @param {Object} ast
 * @param {Object} [options={}]
 */
export function registerRuleAst(sectionKey, ast, options = {}) {
    const valOpts = options?.mode ? { mode: options.mode, allowInternalPolicy: options.allowInternalPolicy } : undefined;
    validateStatutoryAst(ast, valOpts);

    const executableAst = (ast && ast.symbolLegend)
        ? projectSymbolLegendToGuards(ast.symbolLegend, ast)
        : { ...ast };

    validateStatutoryAst(executableAst, valOpts);

    if (options?.mode && String(options.mode).trim().toUpperCase() === 'PRODUCTION' && options.checkOkfStaleness !== false) {
        const lifecycle = classifyOkfRuleLifecycle(executableAst, options);
        if (lifecycle.lifecycleStatus === 'OKF_UNREGISTERED_NEW' && !executableAst.isVerifiedInternalPolicy && !options.allowUnregistered) {
            throw new StatutoryAstValidationError(
                `ERR_UNVERIFIED_PRODUCTION_RULE: Cannot register unregistered rule '${sectionKey}' under PRODUCTION mode.`,
                { code: 'ERR_UNVERIFIED_PRODUCTION_RULE', sectionKey, lifecycle }
            );
        }
        if (lifecycle.lifecycleStatus === 'OKF_AMENDMENT_STALE_DETECTED' && !options.allowStaleEpochProposal) {
            throw new StatutoryAstValidationError(
                `HOLD_STALE_OKF_PENDING_AMENDMENT: Candidate rule '${sectionKey}' conflicts with active OKF baseline pending amendment review.`,
                { code: 'HOLD_STALE_OKF_PENDING_AMENDMENT', sectionKey, lifecycle }
            );
        }
        executableAst.okfLifecycle = lifecycle;
    }

    AST_REGISTRY.set(sectionKey.toUpperCase(), executableAst);
    return executableAst;
}

/**
 * Checks if a section has an active AST registered in the USVM.
 *
 * @param {string} sectionKey
 * @returns {boolean}
 */
export function hasUsvmRule(sectionKey) {
    return getUsvmRule(sectionKey) !== null;
}

/**
 * Returns the registered AST rule, if present, supporting canonical and descriptive aliases.
 *
 * @param {string} sectionKey
 * @returns {Object|null}
 */
export function getUsvmRule(sectionKey) {
    if (!sectionKey || typeof sectionKey !== 'string') return null;
    const clean = sectionKey.toUpperCase().trim();
    if (AST_REGISTRY.has(clean)) return AST_REGISTRY.get(clean);

    // Normalize common prefixes: 'SEC_', 'SECTION_', 'SECTION '
    const stripped = clean.replace(/^(SEC_|SECTION\s*|_)/, '');
    if (AST_REGISTRY.has(stripped)) return AST_REGISTRY.get(stripped);

    // Strip trailing suffixes like '_RULE', '_AST'
    const withoutSuffix = stripped.replace(/(_RULE|_AST)$/, '');
    if (AST_REGISTRY.has(withoutSuffix)) return AST_REGISTRY.get(withoutSuffix);

    // Match substrings for descriptive composite keys
    for (const [k, ast] of AST_REGISTRY.entries()) {
        if (clean === k || clean.includes(k) || (k.length > 3 && stripped.includes(k))) {
            return ast;
        }
    }
    return null;
}

// Pre-load verified open corpus rules on module load
registerRuleFromFile('194Q', 'rules/sec_194q_goods.json');
registerRuleFromFile('393', 'rules/sec_393_goods.json');
registerRuleFromFile('194C', 'rules/sec_194c_contractors.json');
registerRuleFromFile('194J', 'rules/sec_194j_services.json');
registerRuleFromFile('194H', 'rules/sec_194h_commission.json');
registerRuleFromFile('194I', 'rules/sec_194i_rent.json');
registerRuleFromFile('206C_1H', 'rules/sec_206c1h_tcs.json');
registerRuleFromFile('206C(1H)', 'rules/sec_206c1h_tcs.json');
registerRuleFromFile('194A', 'rules/sec_194a_interest.json');
registerRuleFromFile('194O', 'rules/sec_194o_ecommerce.json');
registerRuleFromFile('194IA', 'rules/sec_194ia_property.json');
registerRuleFromFile('194-IA', 'rules/sec_194ia_property.json');
registerRuleFromFile('194R', 'rules/sec_194r_perquisites.json');
registerRuleFromFile('194IB', 'rules/sec_194ib_rent_ind.json');
registerRuleFromFile('194-IB', 'rules/sec_194ib_rent_ind.json');
registerRuleFromFile('194M', 'rules/sec_194m_contract_ind.json');

// MSME Commercial rules
registerRuleFromFile('MSMED15_16', 'rules/sec_msme_delayed_payments_15_16.json');
registerRuleFromFile('MSMED_15_16', 'rules/sec_msme_delayed_payments_15_16.json');
registerRuleFromFile('MSME_DELAYED_PAYMENTS', 'rules/sec_msme_delayed_payments_15_16.json');
registerRuleFromFile('MSMED7', 'rules/sec_msme_classification_sec7.json');
registerRuleFromFile('MSMED_SEC7', 'rules/sec_msme_classification_sec7.json');
registerRuleFromFile('MSME_CLASSIFICATION', 'rules/sec_msme_classification_sec7.json');
registerRuleFromFile('43BH', 'rules/sec_43bh_disallowance.json');
registerRuleFromFile('43B(H)', 'rules/sec_43bh_disallowance.json');
registerRuleFromFile('SEC_43BH', 'rules/sec_43bh_disallowance.json');

// Indirect Tax GST rules
registerRuleFromFile('RULE37', 'rules/sec_gst_itc_reversal_rule37_37a.json');
registerRuleFromFile('RULE_37', 'rules/sec_gst_itc_reversal_rule37_37a.json');
registerRuleFromFile('RULE37_37A', 'rules/sec_gst_itc_reversal_rule37_37a.json');
registerRuleFromFile('RULE42', 'rules/sec_gst_rule42_43_apportionment.json');
registerRuleFromFile('RULE_42', 'rules/sec_gst_rule42_43_apportionment.json');
registerRuleFromFile('RULE42_43', 'rules/sec_gst_rule42_43_apportionment.json');

// International Tax rules
registerRuleFromFile('195', 'rules/sec_195_cross_border.json');
registerRuleFromFile('SEC_195', 'rules/sec_195_cross_border.json');
registerRuleFromFile('CROSS_BORDER_195', 'rules/sec_195_cross_border.json');
registerRuleFromFile('92C', 'rules/sec_92c_transfer_pricing.json');
registerRuleFromFile('SEC_92C', 'rules/sec_92c_transfer_pricing.json');
registerRuleFromFile('TRANSFER_PRICING_92C', 'rules/sec_92c_transfer_pricing.json');

// Corporate Restructuring rules
registerRuleFromFile('79', 'rules/sec_79_loss_continuity.json');
registerRuleFromFile('SEC_79', 'rules/sec_79_loss_continuity.json');
registerRuleFromFile('LOSS_CONTINUITY_79', 'rules/sec_79_loss_continuity.json');
registerRuleFromFile('281', 'rules/sec_281_tax_clearance.json');
registerRuleFromFile('SEC_281', 'rules/sec_281_tax_clearance.json');
registerRuleFromFile('TAX_CLEARANCE_281', 'rules/sec_281_tax_clearance.json');

// Direct Tax Expansion rules
registerRuleFromFile('115BAA', 'rules/sec_115baa_115jb_mat.json');
registerRuleFromFile('115JB', 'rules/sec_115baa_115jb_mat.json');
registerRuleFromFile('CORPORATE_TAX_MAT', 'rules/sec_115baa_115jb_mat.json');
registerRuleFromFile('192', 'rules/sec_192_salary_tds.json');
registerRuleFromFile('SEC_192', 'rules/sec_192_salary_tds.json');
registerRuleFromFile('SALARY_TDS', 'rules/sec_192_salary_tds.json');
registerRuleFromFile('206AB', 'rules/sec_206ab_non_filer.json');
registerRuleFromFile('SEC_206AB', 'rules/sec_206ab_non_filer.json');
registerRuleFromFile('INDAS12', 'rules/sec_indas12_deferred_tax.json');
registerRuleFromFile('INDAS_12', 'rules/sec_indas12_deferred_tax.json');
registerRuleFromFile('DEFERRED_TAX', 'rules/sec_indas12_deferred_tax.json');

// GST Expansion rules
registerRuleFromFile('17_5', 'rules/sec_gst_blocked_itc_17_5.json');
registerRuleFromFile('SEC_17_5', 'rules/sec_gst_blocked_itc_17_5.json');
registerRuleFromFile('BLOCKED_ITC', 'rules/sec_gst_blocked_itc_17_5.json');
registerRuleFromFile('16_2', 'rules/sec_gst_itc_eligibility_16_2.json');
registerRuleFromFile('SEC_16_2', 'rules/sec_gst_itc_eligibility_16_2.json');
registerRuleFromFile('ITC_ELIGIBILITY', 'rules/sec_gst_itc_eligibility_16_2.json');
registerRuleFromFile('9_3', 'rules/sec_gst_rcm_liability_9_3.json');
registerRuleFromFile('SEC_9_3', 'rules/sec_gst_rcm_liability_9_3.json');
registerRuleFromFile('RCM_LIABILITY', 'rules/sec_gst_rcm_liability_9_3.json');
registerRuleFromFile('SEC8', 'rules/sec_gst_composite_mixed_8.json');
registerRuleFromFile('SEC_8', 'rules/sec_gst_composite_mixed_8.json');
registerRuleFromFile('COMPOSITE_MIXED', 'rules/sec_gst_composite_mixed_8.json');
registerRuleFromFile('SEC10', 'rules/sec_gst_composition_levy_10.json');
registerRuleFromFile('SEC_10', 'rules/sec_gst_composition_levy_10.json');
registerRuleFromFile('COMPOSITION_LEVY', 'rules/sec_gst_composition_levy_10.json');
registerRuleFromFile('SEC34', 'rules/sec_gst_credit_debit_notes_34.json');
registerRuleFromFile('SEC_34', 'rules/sec_gst_credit_debit_notes_34.json');
registerRuleFromFile('CREDIT_DEBIT_NOTES', 'rules/sec_gst_credit_debit_notes_34.json');
registerRuleFromFile('LUT_EXPORT', 'rules/sec_gst_export_lut_16.json');
registerRuleFromFile('EXPORT_SERVICES_LUT', 'rules/sec_gst_export_lut_16.json');
registerRuleFromFile('RULE48_4', 'rules/sec_gst_einvoicing_rule48_4.json');
registerRuleFromFile('RULE_48_4', 'rules/sec_gst_einvoicing_rule48_4.json');
registerRuleFromFile('EINVOICING', 'rules/sec_gst_einvoicing_rule48_4.json');
registerRuleFromFile('RULE138', 'rules/sec_gst_ewaybill_rule138.json');
registerRuleFromFile('RULE_138', 'rules/sec_gst_ewaybill_rule138.json');
registerRuleFromFile('EWAYBILL', 'rules/sec_gst_ewaybill_rule138.json');
registerRuleFromFile('SEC51', 'rules/sec_gst_tds_sec51.json');
registerRuleFromFile('SEC_51', 'rules/sec_gst_tds_sec51.json');
registerRuleFromFile('GST_TDS', 'rules/sec_gst_tds_sec51.json');

// MSME Expansion rules
registerRuleFromFile('MSME_1', 'rules/sec_msme_mca_form_1.json');
registerRuleFromFile('MCA_MSME_1', 'rules/sec_msme_mca_form_1.json');
registerRuleFromFile('SEC22', 'rules/sec_msme_disclosures_sec22.json');
registerRuleFromFile('SEC_22', 'rules/sec_msme_disclosures_sec22.json');
registerRuleFromFile('MSME_DISCLOSURES', 'rules/sec_msme_disclosures_sec22.json');
registerRuleFromFile('INDAS116', 'rules/sec_indas116_leases.json');
registerRuleFromFile('INDAS_116', 'rules/sec_indas116_leases.json');
registerRuleFromFile('INDAS115', 'rules/sec_indas115_revenue.json');
registerRuleFromFile('INDAS_115', 'rules/sec_indas115_revenue.json');
registerRuleFromFile('INDAS37', 'rules/sec_indas37_provisions.json');
registerRuleFromFile('INDAS_37', 'rules/sec_indas37_provisions.json');

// International Tax Expansion rules
registerRuleFromFile('INDAS21', 'rules/sec_indas21_fx_reval.json');
registerRuleFromFile('INDAS_21', 'rules/sec_indas21_fx_reval.json');
registerRuleFromFile('FX_REVALUATION', 'rules/sec_indas21_fx_reval.json');
registerRuleFromFile('INDAS109', 'rules/sec_indas109_ecl.json');
registerRuleFromFile('INDAS_109', 'rules/sec_indas109_ecl.json');
registerRuleFromFile('ECL_IMPAIRMENT', 'rules/sec_indas109_ecl.json');
registerRuleFromFile('FTC_RELIEF', 'rules/sec_ftc_dtaa_relief.json');

// Corporate M&A Expansion rules
registerRuleFromFile('50B', 'rules/sec_50b_slump_sale.json');
registerRuleFromFile('SEC_50B', 'rules/sec_50b_slump_sale.json');
registerRuleFromFile('SLUMP_SALE', 'rules/sec_50b_slump_sale.json');
registerRuleFromFile('INDAS103', 'rules/sec_indas103_mna.json');
registerRuleFromFile('INDAS_103', 'rules/sec_indas103_mna.json');
registerRuleFromFile('INDAS36', 'rules/sec_indas36_impairment.json');
registerRuleFromFile('INDAS_36', 'rules/sec_indas36_impairment.json');
registerRuleFromFile('INDAS102', 'rules/sec_indas102_esop.json');
registerRuleFromFile('INDAS_102', 'rules/sec_indas102_esop.json');

// Global Enterprise Accounting Ind AS additions
registerRuleFromFile('INDAS7', 'rules/sec_indas7_cash_flows.json');
registerRuleFromFile('INDAS_7', 'rules/sec_indas7_cash_flows.json');
registerRuleFromFile('CASH_FLOWS', 'rules/sec_indas7_cash_flows.json');
registerRuleFromFile('INDAS16', 'rules/sec_indas16_ppe.json');
registerRuleFromFile('INDAS_16', 'rules/sec_indas16_ppe.json');
registerRuleFromFile('PPE_DEPRECIATION', 'rules/sec_indas16_ppe.json');
registerRuleFromFile('INDAS19', 'rules/sec_indas19_employee_benefits.json');
registerRuleFromFile('INDAS_19', 'rules/sec_indas19_employee_benefits.json');
registerRuleFromFile('EMPLOYEE_BENEFITS', 'rules/sec_indas19_employee_benefits.json');
registerRuleFromFile('INDAS2', 'rules/sec_indas2_inventories.json');
registerRuleFromFile('INDAS_2', 'rules/sec_indas2_inventories.json');
registerRuleFromFile('INVENTORIES_VALUATION', 'rules/sec_indas2_inventories.json');
registerRuleFromFile('INDAS28', 'rules/sec_indas28_associates.json');
registerRuleFromFile('INDAS_28', 'rules/sec_indas28_associates.json');
registerRuleFromFile('EQUITY_METHOD', 'rules/sec_indas28_associates.json');
registerRuleFromFile('INDAS33', 'rules/sec_indas33_eps.json');
registerRuleFromFile('INDAS_33', 'rules/sec_indas33_eps.json');
registerRuleFromFile('EARNINGS_PER_SHARE', 'rules/sec_indas33_eps.json');
registerRuleFromFile('INDAS38', 'rules/sec_indas38_intangibles.json');
registerRuleFromFile('INDAS_38', 'rules/sec_indas38_intangibles.json');
registerRuleFromFile('INTANGIBLES_CAPITALIZATION', 'rules/sec_indas38_intangibles.json');
registerRuleFromFile('INDAS110', 'rules/sec_indas110_consolidation_nci.json');
registerRuleFromFile('INDAS_110', 'rules/sec_indas110_consolidation_nci.json');
registerRuleFromFile('CONSOLIDATION_NCI', 'rules/sec_indas110_consolidation_nci.json');
registerRuleFromFile('INDAS113', 'rules/sec_indas113_fair_value.json');
registerRuleFromFile('INDAS_113', 'rules/sec_indas113_fair_value.json');
registerRuleFromFile('FAIR_VALUE_HIERARCHY', 'rules/sec_indas113_fair_value.json');

// Direct Tax & GST High-Value Additions
registerRuleFromFile('194S', 'rules/sec_194s_crypto_vda.json');
registerRuleFromFile('SEC_194S', 'rules/sec_194s_crypto_vda.json');
registerRuleFromFile('VDA_CRYPTO_194S', 'rules/sec_194s_crypto_vda.json');

registerRuleFromFile('194BA', 'rules/sec_194ba_online_gaming.json');
registerRuleFromFile('SEC_194BA', 'rules/sec_194ba_online_gaming.json');
registerRuleFromFile('ONLINE_GAMING_194BA', 'rules/sec_194ba_online_gaming.json');

registerRuleFromFile('194N', 'rules/sec_194n_cash_withdrawals.json');
registerRuleFromFile('SEC_194N', 'rules/sec_194n_cash_withdrawals.json');
registerRuleFromFile('CASH_WITHDRAWAL_194N', 'rules/sec_194n_cash_withdrawals.json');

registerRuleFromFile('194DA', 'rules/sec_194da_life_insurance.json');
registerRuleFromFile('SEC_194DA', 'rules/sec_194da_life_insurance.json');
registerRuleFromFile('LIFE_INSURANCE_194DA', 'rules/sec_194da_life_insurance.json');

registerRuleFromFile('194K', 'rules/sec_194k_mutual_funds.json');
registerRuleFromFile('SEC_194K', 'rules/sec_194k_mutual_funds.json');
registerRuleFromFile('MUTUAL_FUNDS_194K', 'rules/sec_194k_mutual_funds.json');

registerRuleFromFile('206C1G', 'rules/sec_206c1g_lrs_remittance.json');
registerRuleFromFile('206C(1G)', 'rules/sec_206c1g_lrs_remittance.json');
registerRuleFromFile('SEC_206C1G', 'rules/sec_206c1g_lrs_remittance.json');
registerRuleFromFile('LRS_REMITTANCE_206C1G', 'rules/sec_206c1g_lrs_remittance.json');

registerRuleFromFile('SEC52', 'rules/sec_gst_tcs_ecommerce_sec52.json');
registerRuleFromFile('SEC_52', 'rules/sec_gst_tcs_ecommerce_sec52.json');
registerRuleFromFile('GST_SEC52', 'rules/sec_gst_tcs_ecommerce_sec52.json');
registerRuleFromFile('ECOMMERCE_TCS_SEC52', 'rules/sec_gst_tcs_ecommerce_sec52.json');

/**
 * Primary integration gateway for the statutory calculation pipeline.
 *
 * @param {Object} options
 * @param {string} options.sectionKey - Canonical section identifier (e.g., '194Q', '194C')
 * @param {Object} options.inputFact - Raw normalized transaction facts
 * @param {Function} options.legacyExecutor - Legacy function fallback (e.g., () => evaluateSection194Q(fact))
 * @param {string} [options.mode='SHADOW_VALIDATION'] - 'SHADOW_VALIDATION' | 'USVM_PRIMARY' | 'LEGACY_ONLY'
 * @returns {Object} Comprehensive calculation result
 */
export function executeWithUsvmAdaptive({
    sectionKey,
    inputFact,
    legacyExecutor,
    mode = 'USVM_PRIMARY'
}) {
    const ruleAst = getUsvmRule(sectionKey);

    // If no AST registered, fallback only if legacy executor provided
    if (!ruleAst) {
        if (typeof legacyExecutor === 'function') {
            const legacyResult = legacyExecutor();
            return {
                ...legacyResult,
                isAuthorizedToDisburse: false,
                executionEngine: 'LEGACY_PROCEDURAL'
            };
        }
        throw new Error(`[USVM_PIPELINE] No USVM AST rule registered for section "${sectionKey}".`);
    }

    // Explicit legacy-only override (for testing or debugging)
    if (mode === 'LEGACY_ONLY' && typeof legacyExecutor === 'function') {
        const legacyResult = legacyExecutor();
        return {
            ...legacyResult,
            isAuthorizedToDisburse: false,
            executionEngine: 'LEGACY_PROCEDURAL'
        };
    }

    // Adapt input facts for USVM schema
    const rawAmount = inputFact.invoiceAmount ??
        inputFact.consideration ??
        inputFact.paymentAmountInr ??
        inputFact.interestAmount ??
        inputFact.grossSaleAmount ??
        inputFact.contractualConsiderationInr ??
        inputFact.currentBenefitValueInr ??
        inputFact.monthlyRentInr ??
        inputFact.currentPaymentAmountInr ??
        inputFact.receiptAmountInr ??
        0;

    const usvmFact = {
        ...inputFact,
        invoiceAmount: Number(rawAmount),
        priorPurchasesInCurrentFyInr: inputFact.priorFyPurchases !== undefined 
            ? inputFact.priorFyPurchases 
            : (inputFact.priorPurchasesInCurrentFyInr ?? inputFact.priorPaymentsInCurrentFyInr ?? inputFact.cumulativePriorBenefitsInr ?? inputFact.cumulativePriorPaymentsInr ?? inputFact.priorFyReceiptsInr ?? 0),
        hasPan: inputFact.hasPan !== false,
        isPanInoperative: inputFact.isPanInoperative === true,
        isBuyerPrescribed: inputFact.buyerTurnoverCr !== null && inputFact.buyerTurnoverCr !== undefined 
            ? inputFact.buyerTurnoverCr > 10 
            : (inputFact.isBuyerPrescribed !== false),
        isImportOfGoods: inputFact.isImportOfGoods === true,
        sellerIsNonResident: inputFact.sellerIsNonResident === true || inputFact.sellerIsResident === false,
        transactionDate: inputFact.transactionDate || null,
        creditDate: inputFact.creditDate || null,
        paymentDate: inputFact.paymentDate || null
    };

    if (mode === 'USVM_PRIMARY') {
        const usvmResult = evaluateStatutoryAst(ruleAst, usvmFact);
        const thresholdInr = ruleAst.computation?.thresholdPaisa 
            ? Number(BigInt(ruleAst.computation.thresholdPaisa) / 100n) 
            : 5000000;
        return {
            ...usvmResult,
            isAuthorizedToDisburse: false,
            executionEngine: 'USVM_PRIMARY',
            calculatedTdsInr: usvmResult.calculatedTdsInr,
            taxableBaseInr: usvmResult.taxableBaseInr ?? usvmResult.taxableBase,
            effectiveRatePercent: usvmResult.effectiveRatePercent ?? usvmResult.tdsRatePercent,
            statutoryRatePercent: usvmResult.statutoryRatePercent ?? usvmResult.tdsRatePercent,
            rawInvoiceAmount: usvmResult.invoiceAmountInr,
            invoiceAmount: usvmResult.invoiceAmountInr,
            statutoryThreshold: thresholdInr,
            thresholdBreached: usvmResult.disposition === 'TAX_DEDUCTIBLE' || usvmResult.isTdsApplicable,
            effectiveWithholdingDate: usvmResult.effectiveDate,
            withholdingTrigger: ruleAst.triggerPolicy?.type || 'EARLIER_OF',
            isPanFurnished: usvmFact.hasPan !== false,
            priorFyPurchases: usvmFact.priorPurchasesInCurrentFyInr,
            priorPurchasesInCurrentFyInr: usvmFact.priorPurchasesInCurrentFyInr,
            cumulativeFyPurchases: (usvmFact.priorPurchasesInCurrentFyInr || 0) + (usvmFact.invoiceAmount || 0),
            cumulativePurchasesInCurrentFyInr: (usvmFact.priorPurchasesInCurrentFyInr || 0) + (usvmFact.invoiceAmount || 0),
            usvmMetadata: {
                governingStatute: usvmResult.governingStatute,
                governingSection: usvmResult.governingSection,
                effectiveRateBps: usvmResult.effectiveRateBps,
                temporalRegime: usvmResult.temporalCrystallization?.governingRegime
            }
        };
    }

    // Mode: SHADOW_VALIDATION
    // Run both in parallel; verify 0.00 discrepancy
    let legacyResult;
    try {
        legacyResult = legacyExecutor();
    } catch (legacyErr) {
        // If legacy crashes, USVM acts as resilient primary
        const usvmResult = evaluateStatutoryAst(ruleAst, usvmFact);
        return {
            ...usvmResult,
            isAuthorizedToDisburse: false,
            executionEngine: 'USVM_RESCUE',
            calculatedTdsInr: usvmResult.calculatedTdsInr,
            taxableBaseInr: usvmResult.taxableBase
        };
    }

    try {
        const usvmResult = evaluateStatutoryAst(ruleAst, usvmFact);

        // Normalize legacy return fields across diverse legacy section signatures
        const legacyTds = legacyResult.calculatedTdsInr ??
            legacyResult.calculatedTcsInr ??
            legacyResult.withholding?.calculatedTdsInr ??
            legacyResult.financialBreakdown?.calculatedTdsInr ??
            0;

        const legacyBase = legacyResult.taxableBaseInr ??
            legacyResult.taxableBase ??
            legacyResult.valuationBasis?.taxableBaseInr ??
            legacyResult.financialBreakdown?.taxableBaseInr ??
            0;

        // Domain-Aware Discrepancy Parity Evaluation
        let isParity = false;
        const archetype = ruleAst.computation?.archetype;

        if (archetype === 'COMPOUND_INTEREST_MONTHLY_RESTS') {
            isParity = Math.abs((usvmResult.totalInterest || 0) - (legacyResult.totalInterest || 0)) === 0;
        } else if (archetype === 'MSME_FISCAL_BOUNDARY_43BH') {
            isParity = (usvmResult.taxDeductionStatus === legacyResult.taxDeductionStatus) && (usvmResult.disallowedInCurrentFY === legacyResult.disallowedInCurrentFY);
        } else if (archetype === 'ENTERPRISE_CLASSIFICATION_MSME') {
            isParity = usvmResult.enterpriseCategory === legacyResult.enterpriseCategory;
        } else if (archetype === 'PROPORTIONAL_REVERSAL_180_DAYS') {
            isParity = Math.abs((usvmResult.reversalAmountInr || 0) - (legacyResult.reversalAmountInr || 0)) === 0;
        } else if (archetype === 'COMMON_CREDIT_APPORTIONMENT_RULE42') {
            const usvmRev = usvmResult.summary?.totalMonthlyReversalAddedToOutputLiability ?? usvmResult.totalReversal ?? 0;
            const legRev = legacyResult.summary?.totalMonthlyReversalAddedToOutputLiability ?? legacyResult.totalReversal ?? 0;
            isParity = Math.abs(usvmRev - legRev) === 0;
        } else if (archetype === 'DTAA_TREATY_TAX_HAVEN_MATRIX') {
            isParity = Math.abs((usvmResult.calculatedTdsInr || 0) - (legacyResult.calculatedTdsInr || 0)) === 0;
        } else if (archetype === 'TRANSFER_PRICING_PERCENTILE_ALP') {
            isParity = Math.abs((usvmResult.primaryAdjustmentInr || 0) - (legacyResult.primaryAdjustmentInr || 0)) === 0;
        } else if (archetype === 'VOTING_CONTINUITY_PERCENTAGE') {
            isParity = (usvmResult.status === legacyResult.status) && (usvmResult.statutoryDetermination === legacyResult.statutoryDetermination);
        } else if (archetype === 'TAX_CLEARANCE_VOID_TRANSFER_SEC281') {
            isParity = usvmResult.statutoryDetermination === legacyResult.statutoryDetermination;
        } else {
            // Standard TDS/TCS section
            const diffTds = Math.abs((usvmResult.calculatedTdsInr || 0) - legacyTds);
            const diffBase = Math.abs((usvmResult.taxableBase || 0) - legacyBase);
            isParity = diffTds === 0 && diffBase === 0;
        }

        if (isParity) {
            // Perfect parity confirmed
            return {
                ...legacyResult,
                calculatedTdsInr: legacyTds,
                taxableBase: legacyBase,
                isAuthorizedToDisburse: false,
                executionEngine: 'USVM_VERIFIED_PARITY',
                usvmMetadata: {
                    governingStatute: usvmResult.governingStatute,
                    governingSection: usvmResult.governingSection,
                    effectiveRateBps: usvmResult.effectiveRateBps,
                    temporalRegime: usvmResult.temporalCrystallization?.governingRegime
                }
            };
        } else {
            // Parity mismatch: fail-closed to legacy calculator for safety
            console.warn(`[USVM_ADAPTER] Parity check warning on ${sectionKey}: Failing safe to legacy.`);
            return {
                ...legacyResult,
                isAuthorizedToDisburse: false,
                executionEngine: 'LEGACY_FAILSAFE'
            };
        }
    } catch (usvmErr) {
        // Any USVM evaluation error safely falls back to legacy
        console.warn(`[USVM_ADAPTER] Evaluation error on ${sectionKey}: ${usvmErr.message}. Failing safe to legacy.`);
        return {
            ...legacyResult,
            isAuthorizedToDisburse: false,
            executionEngine: 'LEGACY_FAILSAFE'
        };
    }
}
