/**
 * GST DETERMINISTIC CALCULATORS
 * Rules 37, 37A, 42, 43 ITC apportionments & reversals, Place of Supply & LUT.
 */
import {
    roundToPaisa,
    assertMandatoryNumeric,
    cleanNumericAmount,
    toIntegerPaise,
    fromIntegerPaise,
    MalformedNumericInputError,
    MissingMandatoryParameterError
} from './numeric_utils.js';

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CORPORATE FINANCE: GST INPUT TAX CREDIT APPORTIONMENT & REVERSAL
 * CGST Rules 2017: Rule 42 (Inputs & Services) & Rule 43 (Capital Goods)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements monthly common credit isolation (C2), exempt supply ratio (E/F),
 * non-business 5% apportionment (D2), eligible common credit (C3),
 * 60-month capital goods depreciation apportionment (Te), and annual true-up.
 */
export function calculateGstItcApportionmentRule42_43({
    taxPeriod = null,
    financialYear = '2025-26',
    // Rule 42 Inputs (Inputs & Input Services):
    totalInputTaxCreditT,
    exclusiveNonBusinessT1 = 0,
    exclusiveExemptT2 = 0,
    blockedSection17_5_T3 = 0,
    exclusiveTaxableT4 = 0,
    exemptTurnoverE = 0,
    totalTurnoverF = 0,
    // Rule 43 Inputs (Capital Goods):
    capitalGoodsCommonCreditA = 0,
    capitalGoodsPriorQuartersExempt = 0,
    // Annual True-Up Reconciliation parameters:
    isAnnualTrueUp = false,
    annualExemptTurnover = null,
    annualTotalTurnover = null,
    sumOfMonthlyReversalsD1 = 0,
    isRealEstateProject = false
}) {
    if (totalInputTaxCreditT === undefined || totalInputTaxCreditT === null) {
        return {
            status: 'INDETERMINATE',
            code: 'MISSING_TOTAL_ITC',
            statutoryDetermination: 'INDETERMINATE',
            message: 'Field "totalInputTaxCreditT" is required for GST Rule 42/43 apportionment.',
            netEligibleItc: null,
            totalReversal: null,
            isAuthorizedToDisburse: false
        };
    }

    const T = Number(totalInputTaxCreditT);
    const T1 = Math.max(0, Number(exclusiveNonBusinessT1 || 0));
    const T2 = Math.max(0, Number(exclusiveExemptT2 || 0));
    const T3 = Math.max(0, Number(blockedSection17_5_T3 || 0));
    const T4 = Math.max(0, Number(exclusiveTaxableT4 || 0));
    const E = Math.max(0, Number(exemptTurnoverE || 0));
    const F = Math.max(0, Number(totalTurnoverF || 0));

    if (isNaN(T) || T < 0 || !Number.isFinite(T)) {
        throw new Error('[@mudraforge/statutory] Invalid totalInputTaxCreditT: expected non-negative finite number.');
    }

    // Step 1: Ineligible credits under Rule 42(1)(a)-(c)
    const ineligibleDirect = roundToPaisa(T1 + T2 + T3);
    if (ineligibleDirect > T) {
        throw new Error(`[@mudraforge/statutory] Sum of direct ineligible credits (T1 + T2 + T3 = ₹${ineligibleDirect}) cannot exceed total ITC (T = ₹${T}).`);
    }

    // Step 2: C1 - Amount credited to Electronic Credit Ledger
    const C1 = roundToPaisa(T - ineligibleDirect);

    // Step 3: T4 - Exclusive taxable supplies
    if (T4 > C1) {
        throw new Error(`[@mudraforge/statutory] Exclusive taxable credit T4 (₹${T4}) cannot exceed available credit C1 (₹${C1}).`);
    }

    // Step 4: C2 - Common Input Credit
    const C2 = roundToPaisa(C1 - T4);

    // Step 5: D1 - Ineligible common credit attributable to exempt supplies
    // Formula: D1 = (E / F) * C2
    let exemptRatio = 0;
    let D1 = 0;
    let turnoverBasis = 'TAX_PERIOD_TURNOVER';

    if (C2 > 0) {
        if (F > 0) {
            exemptRatio = E / F;
            if (exemptRatio > 1) exemptRatio = 1; // Defensive cap: exempt cannot exceed total turnover
            D1 = roundToPaisa(C2 * exemptRatio);
        } else if (E === 0 && F === 0) {
            // No turnover in tax period; Rule 42 proviso: use last tax period with turnover
            turnoverBasis = 'ZERO_TURNOVER_DEFERRED_TO_PRECEDING_PERIOD';
            exemptRatio = 0;
            D1 = 0;
        } else {
            // E > 0 but F <= 0 (data anomaly)
            exemptRatio = 1.0;
            D1 = C2;
        }
    }

    // Step 6: D2 - Ineligible common credit attributable to non-business use (5% of C2)
    const D2 = C2 > 0 ? roundToPaisa(C2 * 0.05) : 0;

    // Step 7: C3 - Eligible common credit
    // Formula: C3 = C2 - (D1 + D2)
    const C3 = roundToPaisa(Math.max(0, C2 - (D1 + D2)));

    // Total Rule 42 reversal added to output tax liability:
    const rule42Reversal = roundToPaisa(D1 + D2);

    // Total net eligible input credit for the period (exclusive taxable T4 + eligible common C3):
    const netEligibleInputsItc = roundToPaisa(T4 + C3);

    // ═══════════════════════════════════════════════════════════════════════
    // RULE 43: CAPITAL GOODS APPORTIONMENT
    // ═══════════════════════════════════════════════════════════════════════
    const rawA = Math.max(0, Number(capitalGoodsCommonCreditA || 0));
    const priorQ = Math.max(0, Number(capitalGoodsPriorQuartersExempt || 0));
    
    // If capital goods were used exclusively for exempt/non-business and converted to common,
    // reduce credit by 5% per quarter of prior usage
    const reductionFactor = Math.min(1.0, priorQ * 0.05);
    const effectiveA = roundToPaisa(rawA * (1 - reductionFactor));

    // Deemed useful life: 60 months (5 years)
    const DEEMED_USEFUL_LIFE_MONTHS = 60;
    const monthlyCapitalGoodsCreditTm = rawA > 0 ? roundToPaisa(effectiveA / DEEMED_USEFUL_LIFE_MONTHS) : 0;

    // Capital goods reversal for the month: Te = (E / F) * Tr
    let capitalGoodsReversalTe = 0;
    if (monthlyCapitalGoodsCreditTm > 0 && F > 0) {
        capitalGoodsReversalTe = roundToPaisa(monthlyCapitalGoodsCreditTm * exemptRatio);
    }

    const totalPeriodReversal = roundToPaisa(rule42Reversal + capitalGoodsReversalTe);
    const totalNetEligibleItc = roundToPaisa(netEligibleInputsItc + Math.max(0, monthlyCapitalGoodsCreditTm - capitalGoodsReversalTe));

    // ═══════════════════════════════════════════════════════════════════════
    // ANNUAL TRUE-UP RECONCILIATION (RULE 42(2) & 43(2))
    // ═══════════════════════════════════════════════════════════════════════
    let annualTrueUpDetails = null;
    if (isAnnualTrueUp && annualTotalTurnover !== null && annualExemptTurnover !== null) {
        const annualE = Math.max(0, Number(annualExemptTurnover || 0));
        const annualF = Math.max(0, Number(annualTotalTurnover || 0));
        const annualRatio = annualF > 0 ? (annualE / annualF) : 0;
        
        // Final annual D1 calculated on cumulative common credit C2
        const annualD1Final = roundToPaisa(C2 * annualRatio);
        const priorD1Claimed = Math.max(0, Number(sumOfMonthlyReversalsD1 || 0));
        const trueUpDifference = roundToPaisa(annualD1Final - priorD1Claimed);

        annualTrueUpDetails = {
            isAnnualTrueUp: true,
            financialYear,
            annualExemptTurnover: annualE,
            annualTotalTurnover: annualF,
            annualExemptRatio: Number(annualRatio.toFixed(6)),
            finalAnnualD1Inr: annualD1Final,
            sumOfMonthlyD1ReversalsClaimedInr: priorD1Claimed,
            differentialAdjustmentInr: trueUpDifference,
            adjustmentAction: trueUpDifference > 0 
                ? 'ADD_TO_OUTPUT_LIABILITY_WITH_SECTION_50_INTEREST' 
                : (trueUpDifference < 0 ? 'RECLAIM_AS_ELIGIBLE_ITC' : 'ZERO_DIFFERENTIAL_MATCHED'),
            statutoryFilingDeadlineNotice: isRealEstateProject
                ? 'Real estate project true-up governed by Annexure I/II of Notification 16/2019-CT.'
                : 'Annual adjustment due on or before filing return for month of September following FY end (or 30th November cutoff).'
        };
    }

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'APPLICABLE',
        governingRules: 'CGST Rules 2017: Rule 42 (Inputs & Services) & Rule 43 (Capital Goods)',
        taxPeriod: taxPeriod || 'CURRENT_MONTH',
        financialYear,
        rule42InputsBreakdown: {
            totalItcT: T,
            directIneligible: {
                exclusiveNonBusinessT1: T1,
                exclusiveExemptT2: T2,
                blockedSection17_5_T3: T3,
                sumIneligible: ineligibleDirect
            },
            creditedToElectronicLedgerC1: C1,
            exclusiveTaxableT4: T4,
            commonCreditC2: C2,
            exemptTurnoverE: E,
            totalTurnoverF: F,
            exemptRatio: Number(exemptRatio.toFixed(6)),
            turnoverBasis,
            reversals: {
                ineligibleExemptCommonD1: D1,
                nonBusinessCommonD2: D2,
                totalRule42Reversal: rule42Reversal
            },
            eligibleCommonCreditC3: C3,
            netEligibleInputsItc
        },
        rule43CapitalGoodsBreakdown: {
            capitalGoodsCommonCreditA: rawA,
            deemedUsefulLifeMonths: DEEMED_USEFUL_LIFE_MONTHS,
            priorQuartersExempt: priorQ,
            reductionFactorAppliedPercent: reductionFactor * 100,
            effectiveCommonCredit: effectiveA,
            monthlyCreditTm: monthlyCapitalGoodsCreditTm,
            monthlyReversalTe: capitalGoodsReversalTe,
            eligibleCapitalGoodsCreditMonthly: Math.max(0, monthlyCapitalGoodsCreditTm - capitalGoodsReversalTe)
        },
        summary: {
            totalGrossItcT: T,
            totalDirectIneligible: ineligibleDirect,
            totalCommonCreditApportioned: C2,
            totalMonthlyReversalAddedToOutputLiability: totalPeriodReversal,
            netEligibleItcTotal: totalNetEligibleItc
        },
        annualTrueUp: annualTrueUpDetails,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Rule 42 credited ₹${C1.toLocaleString('en-IN')} to ledger, identified ₹${C2.toLocaleString('en-IN')} common credit, and calculated ₹${rule42Reversal.toLocaleString('en-IN')} reversal (D1: ₹${D1}, D2: ₹${D2}). Rule 43 capital goods reversal: ₹${capitalGoodsReversalTe.toLocaleString('en-IN')}. Net eligible ITC: ₹${totalNetEligibleItc.toLocaleString('en-IN')}. Reversal must be added to GSTR-3B Table 4(B).`
    };
}

/**
 * -----------------------------------------------------------------------------
 * 29. PLACE OF SUPPLY (POS) & ZERO-RATED EXPORT UNDER LUT (IGST ACT SECTIONS 10, 12, 16)
 * -----------------------------------------------------------------------------
 * Evaluates:
 * 1. Domestic B2B / B2C POS (Sec 10 for Goods, Sec 12 for Services).
 * 2. Intra-State (CGST + SGST) vs Inter-State (IGST) classification.
 * 3. Export of Services under Section 16 & Section 2(6) cumulative 5 conditions.
 * 4. Letter of Undertaking (LUT) zero-rate export vs tax payment refund.
 */
export function evaluatePlaceOfSupplyAndLut({
    transactionType = 'DOMESTIC_SERVICES',
    supplierStateCode = null,
    supplierGstin = null,
    recipientStateCode = null,
    recipientGstin = null,
    recipientCountryCode = 'IN',
    invoiceAmountInr = 0,
    gstRate = 0.18,
    isB2B = true,
    placeOfSupplyStateCode = null,
    immovablePropertyLocationStateCode = null,
    isSupplierInIndia = true,
    isRecipientOutsideIndia = null,
    isPlaceOfSupplyOutsideIndia = null,
    isConvertibleForexReceivedOrRbiInr = null,
    isDistinctPersonEstablishment = false,
    isIntermediaryService = false,
    hasValidLut = false,
    lutArn = null
}) {
    const amount = Number(invoiceAmountInr) || 0;
    const rate = Number(gstRate) || 0.18;
    const isTransExport = transactionType === 'EXPORT_SERVICES' || transactionType === 'EXPORT_GOODS' || (recipientCountryCode && recipientCountryCode.toUpperCase() !== 'IN');

    // Extract state codes
    let supState = supplierStateCode;
    if (!supState && supplierGstin && typeof supplierGstin === 'string' && supplierGstin.length >= 2) {
        supState = supplierGstin.slice(0, 2);
    }
    let recState = recipientStateCode;
    if (!recState && recipientGstin && typeof recipientGstin === 'string' && recipientGstin.length >= 2) {
        recState = recipientGstin.slice(0, 2);
    }

    if (isTransExport) {
        // Evaluate Section 2(6) 5 Cumulative Conditions for Export of Services
        const cond1SupplierInIndia = isSupplierInIndia === true;
        const cond2RecipientOutsideIndia = recipientCountryCode.toUpperCase() !== 'IN' && (isRecipientOutsideIndia === null || isRecipientOutsideIndia === true);
        // Intermediary services under Section 13(8)(b) have POS at supplier location in India!
        const cond3PosOutsideIndia = !isIntermediaryService && (isPlaceOfSupplyOutsideIndia === null || isPlaceOfSupplyOutsideIndia === true);
        const cond4ForexReceived = isConvertibleForexReceivedOrRbiInr === true;
        const cond5NotDistinctPersons = isDistinctPersonEstablishment === false;

        const conditionAudit = [
            { condition: '1. Supplier located in India (Sec 2(6)(i))', status: cond1SupplierInIndia ? 'PASS' : 'FAIL' },
            { condition: '2. Recipient located outside India (Sec 2(6)(ii))', status: cond2RecipientOutsideIndia ? 'PASS' : 'FAIL' },
            { condition: '3. Place of supply outside India (Sec 2(6)(iii) & Sec 13)', status: cond3PosOutsideIndia ? 'PASS' : (isIntermediaryService ? 'FAIL_INTERMEDIARY_SEC_13_8_B' : 'FAIL') },
            { condition: '4. Convertible forex / RBI permitted INR received (Sec 2(6)(iv))', status: cond4ForexReceived ? 'PASS' : 'FAIL_NO_FIRC_EBRC' },
            { condition: '5. Not merely establishments of distinct persons (Sec 2(6)(v))', status: cond5NotDistinctPersons ? 'PASS' : 'FAIL_DISTINCT_PERSON_BRANCH' }
        ];

        const isFullyQualifiedExport = cond1SupplierInIndia && cond2RecipientOutsideIndia && cond3PosOutsideIndia && cond4ForexReceived && cond5NotDistinctPersons;

        if (isFullyQualifiedExport) {
            if (hasValidLut) {
                return {
                    status: 'SUCCESS',
                    statutoryDetermination: 'ZERO_RATED_EXPORT_UNDER_LUT',
                    governingLaw: 'IGST Act, 2017 Sections 2(6), 16(1)(a) & 16(3)(a); CGST Rules, 2017 Rule 96A',
                    transactionType: 'EXPORT_SERVICES',
                    invoiceAmountInr: amount,
                    gstRate: 0,
                    igstPayableInr: 0,
                    cgstPayableInr: 0,
                    sgstPayableInr: 0,
                    lutArn: lutArn || 'LUT_ARN_ACTIVE_ON_RECORD',
                    isZeroRated: true,
                    itcRefundEligible: true,
                    reportingTable: 'GSTR-1 Table 6A (Export without payment of tax)',
                    conditionAudit,
                    isAuthorizedToDisburse: false,
                    statutoryRationale: `Zero-Rated Export of Services under LUT: All 5 conditions of Section 2(6) IGST Act verified. Export proceeds in convertible foreign exchange/RBI INR confirmed. Supply is zero-rated at 0% IGST under valid LUT ARN ${lutArn || 'ACTIVE'}. Accumulated unutilized ITC is eligible for refund under Section 54(3).`
                };
            } else {
                const igstTax = roundToPaisa(amount * rate);
                return {
                    status: 'SUCCESS',
                    statutoryDetermination: 'ZERO_RATED_EXPORT_WITH_TAX_PAYMENT',
                    governingLaw: 'IGST Act, 2017 Sections 2(6), 16(1)(a) & 16(3)(b); CGST Act Section 54',
                    transactionType: 'EXPORT_SERVICES',
                    invoiceAmountInr: amount,
                    gstRate: rate,
                    igstPayableInr: igstTax,
                    cgstPayableInr: 0,
                    sgstPayableInr: 0,
                    isZeroRated: true,
                    itcRefundEligible: true,
                    reportingTable: 'GSTR-1 Table 6A (Export with payment of tax)',
                    conditionAudit,
                    isAuthorizedToDisburse: false,
                    statutoryRationale: `Zero-Rated Export with payment of tax: All 5 conditions of Section 2(6) satisfied, but no valid LUT on record. IGST of INR ${igstTax.toLocaleString('en-IN')} (at ${(rate * 100)}%) is charged and eligible for refund under Section 54(1).`
                };
            }
        } else {
            const igstTax = roundToPaisa(amount * rate);
            const failedReasons = conditionAudit.filter(c => c.status !== 'PASS').map(c => c.condition);
            return {
                status: 'SUCCESS',
                statutoryDetermination: 'DISQUALIFIED_FROM_ZERO_RATING',
                governingLaw: 'IGST Act, 2017 Sections 2(6), 7(5)(a) & 13',
                transactionType: 'DISQUALIFIED_EXPORT_SERVICES',
                invoiceAmountInr: amount,
                gstRate: rate,
                igstPayableInr: igstTax,
                cgstPayableInr: 0,
                sgstPayableInr: 0,
                isZeroRated: false,
                itcRefundEligible: false,
                disqualificationReasons: failedReasons,
                conditionAudit,
                isAuthorizedToDisburse: false,
                statutoryRationale: `Disqualified from Zero-Rated Export: Supply failed statutory condition(s): ${failedReasons.join('; ')}. Treated as standard domestic inter-state supply to foreign person subject to standard ${(rate * 100)}% IGST (INR ${igstTax.toLocaleString('en-IN')}).`
            };
        }
    }

    // Domestic Transaction Place of Supply Determination
    let posState = placeOfSupplyStateCode;
    let posLegalRule = '';

    if (immovablePropertyLocationStateCode) {
        posState = immovablePropertyLocationStateCode;
        posLegalRule = 'IGST Act Section 12(3) (Directly related to immovable property - location of property)';
    } else if (transactionType === 'DOMESTIC_GOODS') {
        posState = posState || recState || supState;
        posLegalRule = 'IGST Act Section 10(1)(a) (Location of goods at termination of delivery)';
    } else {
        if (isB2B && recState) {
            posState = recState;
            posLegalRule = 'IGST Act Section 12(2)(a) (B2B supply - location of registered recipient)';
        } else if (recState) {
            posState = recState;
            posLegalRule = 'IGST Act Section 12(2)(b) (B2C supply with address on record)';
        } else {
            posState = supState;
            posLegalRule = 'IGST Act Section 12(2)(b) proviso (B2C supply without address - location of supplier)';
        }
    }

    const isIntraState = supState && posState && (supState === posState);
    let cgstRate = 0, sgstRate = 0, igstRate = 0;
    let cgstAmountInr = 0, sgstAmountInr = 0, igstAmountInr = 0;

    if (isIntraState) {
        cgstRate = rate / 2;
        sgstRate = rate / 2;
        cgstAmountInr = roundToPaisa(amount * cgstRate);
        sgstAmountInr = roundToPaisa(amount * sgstRate);
    } else {
        igstRate = rate;
        igstAmountInr = roundToPaisa(amount * igstRate);
    }
    const totalGstInr = roundToPaisa(cgstAmountInr + sgstAmountInr + igstAmountInr);

    return {
        status: 'SUCCESS',
        statutoryDetermination: isIntraState ? 'INTRA_STATE_SUPPLY' : 'INTER_STATE_SUPPLY',
        governingLaw: isIntraState ? 'CGST Act Section 9 & SGST Act Section 9' : 'IGST Act Section 5 & Section 7',
        posLegalRule,
        supplierStateCode: supState,
        placeOfSupplyStateCode: posState,
        recipientStateCode: recState,
        invoiceAmountInr: amount,
        supplyNature: isIntraState ? 'INTRA_STATE' : 'INTER_STATE',
        taxBreakdown: {
            cgstRate,
            cgstAmountInr,
            sgstRate,
            sgstAmountInr,
            igstRate,
            igstAmountInr,
            totalGstAmountInr: totalGstInr
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: isIntraState
            ? `Intra-State Supply: Supplier state (${supState}) matches Place of Supply (${posState}). CGST (${(cgstRate * 100)}%: INR ${cgstAmountInr.toLocaleString('en-IN')}) and SGST (${(sgstRate * 100)}%: INR ${sgstAmountInr.toLocaleString('en-IN')}) levied under Section 9.`
            : `Inter-State Supply: Supplier state (${supState}) differs from Place of Supply (${posState}). IGST (${(igstRate * 100)}%: INR ${igstAmountInr.toLocaleString('en-IN')}) levied under Section 5 of IGST Act.`
    };
}

/**
 * GST RULE 42: INPUTS & INPUT SERVICES COMMON ITC REVERSAL CALCULATOR
 * CGST Act, 2017: Section 17(1) & 17(2) read with Rule 42
 *
 * Segregates total input tax (T) into:
 * - Specific ineligibles: T1 (non-business), T2 (exempt), T3 (blocked u/s 17(5))
 * - Credited to Electronic Credit Ledger: C1 = T - (T1 + T2 + T3)
 * - Specific eligible: T4 (taxable & zero-rated supplies)
 * - Common Credit: C2 = C1 - T4
 * - Monthly Ineligible Reversals: D1 = (E / F) * C2, D2 = 5% of C2 (non-business)
 * - Eligible Common Credit: C3 = C2 - (D1 + D2)
 * - Net Eligible Inputs Credit: G = T4 + C3 = C1 - (D1 + D2)
 */
export function calculateRule42InputServiceReversal({
    taxPeriod = 'CURRENT_PERIOD',
    totalItcInr,
    t1NonBusinessInr = 0,
    t2ExemptInr = 0,
    t3BlockedInr = 0,
    t4TaxableInr = 0,
    exemptTurnoverInr = 0,
    totalTurnoverInr = 0,
    hasNonBusinessUse = true,
    precedingExemptTurnoverInr = 0,
    precedingTotalTurnoverInr = 0
} = {}) {
    if (totalItcInr === undefined || totalItcInr === null) {
        throw new Error('[statutory_calculator] calculateRule42InputServiceReversal: totalItcInr is required.');
    }

    const T = roundToPaisa(Math.max(0, cleanNumericAmount(totalItcInr)));
    const T1 = roundToPaisa(Math.max(0, cleanNumericAmount(t1NonBusinessInr)));
    const T2 = roundToPaisa(Math.max(0, cleanNumericAmount(t2ExemptInr)));
    const T3 = roundToPaisa(Math.max(0, cleanNumericAmount(t3BlockedInr)));
    const T4 = roundToPaisa(Math.max(0, cleanNumericAmount(t4TaxableInr)));
    const E = Math.max(0, cleanNumericAmount(exemptTurnoverInr));
    const F = Math.max(0, cleanNumericAmount(totalTurnoverInr));

    const directIneligible = roundToPaisa(T1 + T2 + T3);
    if (directIneligible > T) {
        throw new Error(`[statutory_calculator] Sum of direct ineligible credits (T1 + T2 + T3 = ₹${directIneligible}) cannot exceed total ITC (T = ₹${T}).`);
    }

    // Step 2: C1 - Amount credited to Electronic Credit Ledger
    const C1 = roundToPaisa(T - directIneligible);

    // Step 3: T4 check
    if (T4 > C1) {
        throw new Error(`[statutory_calculator] Exclusive taxable credit T4 (₹${T4}) cannot exceed available credit C1 (₹${C1}).`);
    }

    // Step 4: C2 - Common Input Credit
    const C2 = roundToPaisa(C1 - T4);

    // Step 5: Turnover Apportionment (E / F)
    let exemptRatio = 0;
    let turnoverBasis = 'CURRENT_TAX_PERIOD';

    if (C2 > 0) {
        if (F > 0) {
            exemptRatio = Math.min(1.0, E / F);
        } else if (E === 0 && F === 0) {
            const precE = Math.max(0, cleanNumericAmount(precedingExemptTurnoverInr));
            const precF = Math.max(0, cleanNumericAmount(precedingTotalTurnoverInr));
            if (precF > 0) {
                exemptRatio = Math.min(1.0, precE / precF);
                turnoverBasis = 'PRECEDING_TAX_PERIOD_PROVISO';
            } else {
                exemptRatio = 0;
                turnoverBasis = 'ZERO_TURNOVER_NO_REVERSAL';
            }
        } else {
            // E > 0 but F <= 0 (defensive full exempt cap)
            exemptRatio = 1.0;
            turnoverBasis = 'DEFENSIVE_FULL_EXEMPT_CAP';
        }
    }

    const D1 = C2 > 0 ? roundToPaisa(C2 * exemptRatio) : 0;
    const D2 = (C2 > 0 && hasNonBusinessUse) ? roundToPaisa(C2 * 0.05) : 0;
    const rule42Reversal = roundToPaisa(D1 + D2);
    const C3 = roundToPaisa(Math.max(0, C2 - rule42Reversal));
    const netEligibleInputsItc = roundToPaisa(T4 + C3);

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'RULE_42_APPORTIONMENT_CALCULATED',
        governingRules: 'Central Goods and Services Tax Rules, 2017: Rule 42 (Inputs and Input Services)',
        taxPeriod,
        totalItcT: T,
        directIneligibleCredits: {
            exclusiveNonBusinessT1: T1,
            exclusiveExemptT2: T2,
            blockedSection17_5_T3: T3,
            sumIneligible: directIneligible
        },
        creditedToElectronicCreditLedgerC1: C1,
        exclusiveTaxableT4: T4,
        commonCreditC2: C2,
        exemptTurnoverE: E,
        totalTurnoverF: F,
        turnoverRatioE_F: Number(exemptRatio.toFixed(6)),
        turnoverBasis,
        reversals: {
            ineligibleExemptCommonD1: D1,
            nonBusinessCommonD2: D2,
            totalRule42Reversal: rule42Reversal
        },
        eligibleCommonCreditC3: C3,
        netEligibleInputsItc,
        gstr3bReporting: {
            table4A5GrossIngestion: T,
            table4B1PermanentReversal: roundToPaisa(directIneligible + rule42Reversal),
            table4B1Rule42OnlyReversal: rule42Reversal,
            table4CNetItcAvailable: netEligibleInputsItc
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Rule 42 credited ₹${C1.toLocaleString('en-IN')} to ledger (after deducting ₹${directIneligible.toLocaleString('en-IN')} direct ineligible). Identified ₹${C2.toLocaleString('en-IN')} common credit, requiring ₹${rule42Reversal.toLocaleString('en-IN')} reversal (D1 exempt: ₹${D1}, D2 non-business: ₹${D2}). Net eligible credit: ₹${netEligibleInputsItc.toLocaleString('en-IN')}. Permanent reversal must be reported in GSTR-3B Table 4(B)(1).`
    };
}

/**
 * GST RULE 42(2): ANNUAL COMMON CREDIT TRUE-UP & RECONCILIATION ENGINE
 * CGST Rules, 2017: Rule 42(2) read with Section 16(4) and Section 50(1)
 * Amended by Finance Act, 2022 & Notification No. 18/2022-Central Tax w.e.f. 01.10.2022
 *
 * Annual Cutoff: 30th November following the end of the financial year.
 * - Shortfall Reversal: Payable with 18% p.a. interest under Section 50(1) from 1st April of succeeding FY.
 * - Excess Reversal: Reclaimable as credit in Form GSTR-3B Table 4(A)(5) and Table 4(D)(1) before 30th November.
 */
export function reconcileRule42AnnualTrueUp({
    financialYear = '2024-25',
    annualCommonCreditC2Inr,
    annualExemptTurnoverInr = 0,
    annualTotalTurnoverInr = 0,
    cumulativeMonthlyReversalsInr,
    trueUpExecutionDate = null,
    hasNonBusinessUse = true,
    annualInterestRatePercent = 18.0,
    isRealEstateProject = false
} = {}) {
    if (annualCommonCreditC2Inr === undefined || annualCommonCreditC2Inr === null) {
        throw new Error('[statutory_calculator] reconcileRule42AnnualTrueUp: annualCommonCreditC2Inr is required.');
    }
    if (cumulativeMonthlyReversalsInr === undefined || cumulativeMonthlyReversalsInr === null) {
        throw new Error('[statutory_calculator] reconcileRule42AnnualTrueUp: cumulativeMonthlyReversalsInr is required.');
    }

    const C2 = roundToPaisa(Math.max(0, cleanNumericAmount(annualCommonCreditC2Inr)));
    const priorReversals = roundToPaisa(Math.max(0, cleanNumericAmount(cumulativeMonthlyReversalsInr)));
    const annualE = Math.max(0, cleanNumericAmount(annualExemptTurnoverInr));
    const annualF = Math.max(0, cleanNumericAmount(annualTotalTurnoverInr));

    // Derive succeeding FY start date & statutory cutoff (30th November)
    let endYear = 2025;
    const fyMatch = String(financialYear).match(/^(\d{4})-(\d{2,4})$/);
    if (fyMatch) {
        const y1 = parseInt(fyMatch[1], 10);
        endYear = y1 + 1;
    }
    const succeedingFyStartDate = `${endYear}-04-01`;
    const statutoryCutoffDate = `${endYear}-11-30`;

    const execDateIso = trueUpExecutionDate ? String(trueUpExecutionDate).slice(0, 10) : statutoryCutoffDate;
    const isFiledBeforeCutoff = execDateIso <= statutoryCutoffDate;

    const annualRatio = annualF > 0 ? Math.min(1.0, annualE / annualF) : 0;
    const finalAnnualD1 = roundToPaisa(C2 * annualRatio);
    const finalAnnualD2 = hasNonBusinessUse ? roundToPaisa(C2 * 0.05) : 0;
    const finalAnnualReversalTotal = roundToPaisa(finalAnnualD1 + finalAnnualD2);

    const differential = roundToPaisa(finalAnnualReversalTotal - priorReversals);

    let reconciliationOutcome = 'BALANCED_ZERO_VARIANCE';
    let shortfallAmountInr = 0;
    let excessReversedInr = 0;
    let interestPayableInr = 0;
    let daysElapsed = 0;
    let statutoryAction = 'Annual true-up precisely matches cumulative monthly reversals. No adjustment required.';

    if (differential > 0) {
        reconciliationOutcome = 'SHORT_REVERSAL_PAYMENT_DUE';
        shortfallAmountInr = differential;
        
        // Interest under Section 50(1) from 1st April of succeeding FY
        const dStart = new Date(succeedingFyStartDate + 'T00:00:00.000Z').getTime();
        const dExec = new Date(execDateIso + 'T00:00:00.000Z').getTime();
        daysElapsed = Math.max(0, Math.floor((dExec - dStart) / (1000 * 60 * 60 * 24)));
        const rateFrac = (Number(annualInterestRatePercent) || 18.0) / 100;
        interestPayableInr = roundToPaisa(shortfallAmountInr * rateFrac * (daysElapsed / 365));

        statutoryAction = `Shortfall of ₹${shortfallAmountInr.toLocaleString('en-IN')} must be added to output tax liability / reversed in GSTR-3B Table 4(B)(1) on or before ${statutoryCutoffDate} along with ₹${interestPayableInr.toLocaleString('en-IN')} interest under Section 50(1) (${daysElapsed} days @ ${annualInterestRatePercent}% p.a.).`;
    } else if (differential < 0) {
        reconciliationOutcome = 'EXCESS_REVERSAL_RECLAIM_PERMITTED';
        excessReversedInr = roundToPaisa(Math.abs(differential));
        statutoryAction = `Excess reversal of ₹${excessReversedInr.toLocaleString('en-IN')} may be reclaimed as eligible ITC in Form GSTR-3B Table 4(A)(5) and reported in Table 4(D)(1) on or before ${statutoryCutoffDate}.`;
    }

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'ANNUAL_TRUE_UP_RECONCILED',
        governingRules: 'Central Goods and Services Tax Rules, 2017: Rule 42(2) & Finance Act, 2022',
        financialYear,
        statutoryCutoffDate,
        succeedingFyStartDate,
        trueUpExecutionDate: execDateIso,
        isFiledBeforeCutoff,
        annualCommonCreditC2Inr: C2,
        annualExemptTurnoverInr: annualE,
        annualTotalTurnoverInr: annualF,
        annualExemptRatio: Number(annualRatio.toFixed(6)),
        finalAnnualReversals: {
            finalAnnualD1ExemptInr: finalAnnualD1,
            finalAnnualD2NonBusinessInr: finalAnnualD2,
            finalAnnualRequiredTotalInr: finalAnnualReversalTotal
        },
        cumulativeMonthlyReversalsMadeInr: priorReversals,
        differentialAdjustmentInr: differential,
        reconciliationOutcome,
        shortfallAmountInr,
        excessReversedInr,
        interestAssessment: {
            interestRatePercent: annualInterestRatePercent,
            interestCommencementDate: succeedingFyStartDate,
            interestEndDate: execDateIso,
            daysElapsed,
            interestPayableInr
        },
        isRealEstateProject: !!isRealEstateProject,
        statutoryAction,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Annual common credit ₹${C2.toLocaleString('en-IN')} with annual exempt ratio ${annualRatio.toFixed(4)} requires aggregate annual reversal of ₹${finalAnnualReversalTotal.toLocaleString('en-IN')}. Cumulative monthly reversals made: ₹${priorReversals.toLocaleString('en-IN')}. Outcome: ${reconciliationOutcome} (${differential > 0 ? `Shortfall ₹${shortfallAmountInr} + Sec 50(1) Interest ₹${interestPayableInr}` : (differential < 0 ? `Reclaimable ₹${excessReversedInr}` : 'Zero Variance')}).`
    };
}

/**
 * GST RULE 43: CAPITAL GOODS COMMON ITC AMORTISATION & REVERSAL CALCULATOR
 * CGST Act, 2017: Section 17(1) & 17(2) read with Rule 43
 *
 * Implements:
 * - 60-month statutory useful life tracking for each capital asset.
 * - Monthly credit slice: Tm = Tc / 60.
 * - Aggregate monthly common credit: Tr = sum(Tm).
 * - Monthly exempt reversal: Te = (E / F) * Tr (added to output tax liability in Table 4(B)(1)).
 * - Asset conversion: 5% reduction per quarter under Rule 43(1)(c) for exclusive-to-common transitions.
 */
export function calculateRule43CapitalGoodsReversal({
    taxPeriod = 'CURRENT_PERIOD',
    taxPeriodDate = new Date().toISOString().slice(0, 10),
    exemptTurnoverInr = 0,
    totalTurnoverInr = 0,
    capitalGoodsList = [],
    precedingExemptTurnoverInr = 0,
    precedingTotalTurnoverInr = 0
} = {}) {
    if (!Array.isArray(capitalGoodsList)) {
        throw new Error('[statutory_calculator] calculateRule43CapitalGoodsReversal: capitalGoodsList must be an array.');
    }

    const E = Math.max(0, cleanNumericAmount(exemptTurnoverInr));
    const F = Math.max(0, cleanNumericAmount(totalTurnoverInr));

    let exemptRatio = 0;
    let turnoverBasis = 'CURRENT_TAX_PERIOD';

    if (F > 0) {
        exemptRatio = Math.min(1.0, E / F);
    } else if (E === 0 && F === 0) {
        const precE = Math.max(0, cleanNumericAmount(precedingExemptTurnoverInr));
        const precF = Math.max(0, cleanNumericAmount(precedingTotalTurnoverInr));
        if (precF > 0) {
            exemptRatio = Math.min(1.0, precE / precF);
            turnoverBasis = 'PRECEDING_TAX_PERIOD_PROVISO';
        } else {
            exemptRatio = 0;
            turnoverBasis = 'ZERO_TURNOVER_NO_REVERSAL';
        }
    } else {
        exemptRatio = 1.0;
        turnoverBasis = 'DEFENSIVE_FULL_EXEMPT_CAP';
    }

    const DEEMED_USEFUL_LIFE_MONTHS = 60;
    const calcDate = new Date(String(taxPeriodDate).slice(0, 10) + 'T00:00:00.000Z');

    let totalCommonTc = 0;
    let totalMonthlySliceTr = 0;
    const assetSchedules = [];

    for (const asset of capitalGoodsList) {
        const assetId = asset.assetId || 'ASSET-UNKNOWN';
        const description = asset.description || 'Capital Good';
        const rawItc = roundToPaisa(Math.max(0, cleanNumericAmount(asset.itcAmountInr || asset.taxAmount)));
        const category = String(asset.category || asset.useCategory || 'COMMON').toUpperCase();
        const invoiceDateIso = asset.invoiceDate ? String(asset.invoiceDate).slice(0, 10) : taxPeriodDate;

        if (category === 'EXEMPT_ONLY') {
            assetSchedules.push({
                assetId,
                description,
                category,
                invoiceDate: invoiceDateIso,
                originalItcInr: rawItc,
                effectiveCommonCreditInr: 0,
                monthlyCreditTmInr: 0,
                monthlyReversalTeInr: 0,
                status: 'EXEMPT_EXCLUDED_FROM_LEDGER',
                statutoryNote: 'Under Rule 43(1)(a), capital goods used exclusively for exempt supplies are not credited to Electronic Credit Ledger.'
            });
            continue;
        }

        if (category === 'TAXABLE_ONLY') {
            assetSchedules.push({
                assetId,
                description,
                category,
                invoiceDate: invoiceDateIso,
                originalItcInr: rawItc,
                effectiveCommonCreditInr: 0,
                monthlyCreditTmInr: 0,
                monthlyReversalTeInr: 0,
                status: 'TAXABLE_FULL_CREDIT_NO_REVERSAL',
                statutoryNote: 'Under Rule 43(1)(b), capital goods used exclusively for taxable supplies are credited in full with zero monthly reversal.'
            });
            continue;
        }

        // Category is COMMON (or converted to COMMON)
        let effectiveItc = rawItc;
        let quartersElapsed = 0;
        let reductionPercent = 0;

        if (asset.priorCategory === 'EXEMPT_ONLY' && asset.conversionDate) {
            const convDateIso = String(asset.conversionDate).slice(0, 10);
            const dInv = new Date(invoiceDateIso + 'T00:00:00.000Z').getTime();
            const dConv = new Date(convDateIso + 'T00:00:00.000Z').getTime();
            const daysConv = Math.max(0, Math.floor((dConv - dInv) / (1000 * 60 * 60 * 24)));
            quartersElapsed = daysConv > 0 ? Math.ceil(daysConv / 91.25) : 0;
            reductionPercent = Math.min(100, quartersElapsed * 5);
            effectiveItc = roundToPaisa(rawItc * (1 - reductionPercent / 100));
        }

        // Calculate months elapsed from invoice date to calculation date
        const dInv = new Date(invoiceDateIso + 'T00:00:00.000Z');
        const monthsElapsed = (calcDate.getFullYear() - dInv.getFullYear()) * 12 + (calcDate.getMonth() - dInv.getMonth());

        let monthlyCreditTm = 0;
        let monthlyReversalTe = 0;
        let status = 'ACTIVE_IN_COMMON_POOL';

        if (monthsElapsed >= 0 && monthsElapsed < DEEMED_USEFUL_LIFE_MONTHS) {
            monthlyCreditTm = roundToPaisa(effectiveItc / DEEMED_USEFUL_LIFE_MONTHS);
            monthlyReversalTe = roundToPaisa(monthlyCreditTm * exemptRatio);
            totalCommonTc = roundToPaisa(totalCommonTc + effectiveItc);
            totalMonthlySliceTr = roundToPaisa(totalMonthlySliceTr + monthlyCreditTm);
        } else if (monthsElapsed >= DEEMED_USEFUL_LIFE_MONTHS) {
            status = 'EXPIRED_USEFUL_LIFE_60_MONTHS';
        } else {
            status = 'PRE_ACQUISITION_PERIOD';
        }

        assetSchedules.push({
            assetId,
            description,
            category: 'COMMON',
            invoiceDate: invoiceDateIso,
            originalItcInr: rawItc,
            effectiveCommonCreditInr: effectiveItc,
            reductionFactorAppliedPercent: reductionPercent,
            monthsElapsed,
            remainingUsefulLifeMonths: Math.max(0, DEEMED_USEFUL_LIFE_MONTHS - Math.max(0, monthsElapsed)),
            monthlyCreditTmInr: monthlyCreditTm,
            monthlyReversalTeInr: monthlyReversalTe,
            status,
            statutoryNote: status === 'ACTIVE_IN_COMMON_POOL'
                ? `Active in common pool (Month ${monthsElapsed + 1}/60). Tm = ₹${monthlyCreditTm}, Te reversal = ₹${monthlyReversalTe}.`
                : (status === 'EXPIRED_USEFUL_LIFE_60_MONTHS' ? 'Deemed useful life of 60 months has expired. No further reversal required.' : 'Asset acquired after tax period.')
        });
    }

    const totalPeriodReversalTe = roundToPaisa(totalMonthlySliceTr * exemptRatio);
    const netEligibleMonthlyCapitalItc = roundToPaisa(Math.max(0, totalMonthlySliceTr - totalPeriodReversalTe));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'RULE_43_CAPITAL_GOODS_APPORTIONED',
        governingRules: 'Central Goods and Services Tax Rules, 2017: Rule 43 (Capital Goods)',
        taxPeriod,
        taxPeriodDate: String(taxPeriodDate).slice(0, 10),
        exemptTurnoverE: E,
        totalTurnoverF: F,
        turnoverRatioE_F: Number(exemptRatio.toFixed(6)),
        turnoverBasis,
        deemedUsefulLifeMonths: DEEMED_USEFUL_LIFE_MONTHS,
        summary: {
            totalAssetsAssessed: capitalGoodsList.length,
            activeCommonAssetsCount: assetSchedules.filter(a => a.status === 'ACTIVE_IN_COMMON_POOL').length,
            totalCommonCreditTcInr: totalCommonTc,
            monthlyCommonCreditSliceTrInr: totalMonthlySliceTr,
            monthlyReversalTeInr: totalPeriodReversalTe,
            netEligibleMonthlyCapitalItcInr: netEligibleMonthlyCapitalItc
        },
        gstr3bReporting: {
            reportingTable: 'Table 4(B)(1) (Permanent Reversals)',
            monthlyReversalAmountInr: totalPeriodReversalTe
        },
        assetSchedules,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Assessed ${capitalGoodsList.length} capital goods. Common pool ITC (Tc): ₹${totalCommonTc.toLocaleString('en-IN')}, monthly slice (Tr): ₹${totalMonthlySliceTr.toLocaleString('en-IN')}. Monthly exempt reversal (Te = Tr * E/F): ₹${totalPeriodReversalTe.toLocaleString('en-IN')}. Reversal must be added to output tax liability / reported in Table 4(B)(1).`
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * GST RULE 37: 180-DAY VENDOR NON-PAYMENT ITC REVERSAL & INTEREST CALCULATOR
 * ─────────────────────────────────────────────────────────────────────────────
 * CGST Act, 2017: Section 16(2) second proviso read with Rule 37
 * Amended by Notification No. 19/2022-Central Tax w.e.f. 01.10.2022.
 *
 * Implements authoritative 180-day aging, proportional unpaid ITC computation,
 * Section 50(1) 18% annual interest, RCM/Schedule I exemptions, and GSTR-3B Table 4(B)(2) mapping.
 */
export function calculateItcReversalRule37({
    invoiceNumber = 'INV-UNKNOWN',
    invoiceDate,
    calculationDate = new Date().toISOString().slice(0, 10),
    invoiceAmount, // Total invoice value (taxable + tax)
    taxAmount,    // Total ITC availed
    paidAmount = 0, // Amount paid to supplier towards invoice value + tax
    itcAvailmentDate = null,
    isRcm = false, // Supplies under Section 9(3)/9(4) RCM are exempt
    isScheduleISupply = false, // Deemed supplies without consideration under Schedule I are exempt
    annualInterestRatePercent = 18.0
} = {}) {
    const totalInvoice = cleanNumericAmount(invoiceAmount);
    const totalItc = cleanNumericAmount(taxAmount);
    const totalPaid = Math.max(0, cleanNumericAmount(paidAmount));
    const unpaidAmount = roundToPaisa(Math.max(0, totalInvoice - totalPaid));

    if (!invoiceDate || !/^\d{4}-\d{2}-\d{2}$/.test(String(invoiceDate).slice(0, 10))) {
        throw new Error('[statutory_calculator] calculateItcReversalRule37: invoiceDate must be a valid ISO YYYY-MM-DD date.');
    }

    const calcDateIso = String(calculationDate).slice(0, 10);
    const invDateIso = String(invoiceDate).slice(0, 10);
    const availDateIso = itcAvailmentDate ? String(itcAvailmentDate).slice(0, 10) : invDateIso;

    const invTime = new Date(invDateIso).getTime();
    const calcTime = new Date(calcDateIso).getTime();
    const daysElapsed = Math.max(0, Math.floor((calcTime - invTime) / (1000 * 60 * 60 * 24)));

    // Statutory 180-day grace period under Section 16(2) second proviso
    const GRACE_PERIOD_DAYS = 180;
    const isOverdue = daysElapsed > GRACE_PERIOD_DAYS;
    const daysOverdueBeyond180 = isOverdue ? daysElapsed - GRACE_PERIOD_DAYS : 0;

    // Exemptions under Section 16(2) provisos & Rule 37(1) provisos
    if (isRcm) {
        return {
            invoiceNumber,
            invoiceDate: invDateIso,
            calculationDate: calcDateIso,
            invoiceAmount: totalInvoice,
            taxAmount: totalItc,
            paidAmount: totalPaid,
            unpaidAmount,
            daysElapsed,
            isOverdue: false,
            isExempt: true,
            exemptionReason: 'RCM_SUPPLY_SECTION_9_3_9_4',
            reversalAmountInr: 0,
            interestPayableInr: 0,
            gstr3bReportingTable: null,
            canReclaimOnPayment: false,
            statutoryDisposition: 'EXEMPT_FROM_180_DAY_REVERSAL_RCM',
            statutoryRationale: 'Supplies subject to reverse charge mechanism under Section 9(3) or 9(4) are expressly excluded from the second proviso to Section 16(2).',
            isAuthorizedToDisburse: false
        };
    }

    if (isScheduleISupply) {
        return {
            invoiceNumber,
            invoiceDate: invDateIso,
            calculationDate: calcDateIso,
            invoiceAmount: totalInvoice,
            taxAmount: totalItc,
            paidAmount: totalPaid,
            unpaidAmount,
            daysElapsed,
            isOverdue: false,
            isExempt: true,
            exemptionReason: 'SCHEDULE_I_SUPPLY_WITHOUT_CONSIDERATION',
            reversalAmountInr: 0,
            interestPayableInr: 0,
            gstr3bReportingTable: null,
            canReclaimOnPayment: false,
            statutoryDisposition: 'EXEMPT_FROM_180_DAY_REVERSAL_SCHEDULE_I',
            statutoryRationale: 'Deemed supplies without consideration under Schedule I of the CGST Act are deemed paid under first proviso to Rule 37(1).',
            isAuthorizedToDisburse: false
        };
    }

    if (!isOverdue) {
        return {
            invoiceNumber,
            invoiceDate: invDateIso,
            calculationDate: calcDateIso,
            invoiceAmount: totalInvoice,
            taxAmount: totalItc,
            paidAmount: totalPaid,
            unpaidAmount,
            daysElapsed,
            daysRemainingInGracePeriod: GRACE_PERIOD_DAYS - daysElapsed,
            isOverdue: false,
            isExempt: false,
            reversalAmountInr: 0,
            interestPayableInr: 0,
            gstr3bReportingTable: null,
            canReclaimOnPayment: false,
            statutoryDisposition: 'WITHIN_180_DAYS_GRACE_PERIOD',
            statutoryRationale: `Invoice age is ${daysElapsed} days, which is within the statutory 180-day grace period under Section 16(2).`,
            isAuthorizedToDisburse: false
        };
    }

    // Days elapsed > 180 days: Evaluate unpaid portion
    if (unpaidAmount === 0) {
        return {
            invoiceNumber,
            invoiceDate: invDateIso,
            calculationDate: calcDateIso,
            invoiceAmount: totalInvoice,
            taxAmount: totalItc,
            paidAmount: totalPaid,
            unpaidAmount: 0,
            daysElapsed,
            isOverdue: false,
            isExempt: false,
            reversalAmountInr: 0,
            interestPayableInr: 0,
            gstr3bReportingTable: null,
            canReclaimOnPayment: false,
            statutoryDisposition: 'FULLY_PAID_NO_REVERSAL',
            statutoryRationale: 'Supplier has been fully paid within or prior to evaluation. Zero ITC reversal required.',
            isAuthorizedToDisburse: false
        };
    }

    // Proportionate Reversal: ITC to reverse = (Unpaid Amount / Total Invoice Value) * Total ITC availed
    const proportionUnpaid = totalInvoice > 0 ? (unpaidAmount / totalInvoice) : 1;
    const reversalAmountInr = roundToPaisa(totalItc * proportionUnpaid);

    // Interest calculation under Section 50(1) at 18% per annum
    // Under substituted Rule 37(1) w.e.f. 01.10.2022: interest runs from date of availing ITC till date of reversal
    const availTime = new Date(availDateIso).getTime();
    const interestDays = Math.max(0, Math.floor((calcTime - availTime) / (1000 * 60 * 60 * 24)));
    const annualRate = Number(annualInterestRatePercent) / 100;
    const interestPayableInr = roundToPaisa(reversalAmountInr * annualRate * (interestDays / 365));

    return {
        invoiceNumber,
        invoiceDate: invDateIso,
        calculationDate: calcDateIso,
        itcAvailmentDate: availDateIso,
        invoiceAmount: totalInvoice,
        taxAmount: totalItc,
        paidAmount: totalPaid,
        unpaidAmount,
        daysElapsed,
        daysOverdueBeyond180,
        isOverdue: true,
        isExempt: false,
        reversalAmountInr,
        interestDays,
        annualInterestRatePercent,
        interestPayableInr,
        gstr3bReportingTable: 'TABLE_4_B_2_OTHERS_RECLAIMABLE',
        reAvailmentTable: 'TABLE_4_A_5_AND_4_D_1',
        canReclaimOnPayment: true,
        statutoryDisposition: 'ITC_REVERSAL_MANDATORY_RULE_37',
        statutoryRationale: `Payment of ₹${unpaidAmount.toLocaleString('en-IN')} outstanding beyond 180 days (${daysElapsed} days elapsed). Proportional ITC of ₹${reversalAmountInr.toLocaleString('en-IN')} must be reversed in Form GSTR-3B Table 4(B)(2) along with ₹${interestPayableInr.toLocaleString('en-IN')} interest under Section 50(1). Reclaimable upon payment.`,
        isAuthorizedToDisburse: false
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * GST RULE 37A: SUPPLIER GSTR-3B NON-FILING REVERSAL CALCULATOR
 * ─────────────────────────────────────────────────────────────────────────────
 * CGST Rules, 2017: Rule 37A (Inserted by Notification No. 26/2022-CT w.e.f. 26.12.2022)
 *
 * Where supplier reported invoice in GSTR-1/IFF but failed to file GSTR-3B by
 * 30th November following the end of the FY in which ITC was availed:
 * - Recipient must reverse ITC in GSTR-3B on or before 30th November.
 * - If delayed past 30th November: 18% annual interest applies u/s 50(1).
 * - Interest-free re-availment permitted once supplier subsequently furnishes GSTR-3B.
 */
export function calculateRule37ASupplierNonFilingReversal({
    financialYear, // e.g. '2024-25'
    invoiceNumber = 'INV-UNKNOWN',
    invoiceDate,
    itcAmount,
    supplierGstr1Filed = true,
    supplierGstr3bFiled = false,
    calculationDate = new Date().toISOString().slice(0, 10),
    annualInterestRatePercent = 18.0
} = {}) {
    const rawItc = cleanNumericAmount(itcAmount);
    const calcDateIso = String(calculationDate).slice(0, 10);

    // Compute November 30 cutoff date for the given financial year
    let cutoffYear = null;
    if (financialYear && typeof financialYear === 'string') {
        const parts = financialYear.split('-');
        if (parts.length === 2 && parts[0].length === 4) {
            cutoffYear = parseInt(parts[0], 10) + 1;
        }
    } else if (invoiceDate) {
        const d = new Date(invoiceDate);
        const invYear = d.getFullYear();
        const invMonth = d.getMonth() + 1;
        cutoffYear = invMonth >= 4 ? invYear + 1 : invYear;
    } else {
        cutoffYear = new Date().getFullYear();
    }

    const cutoffDate = `${cutoffYear}-11-30`;
    const isPastCutoff = calcDateIso > cutoffDate;

    if (supplierGstr3bFiled) {
        return {
            invoiceNumber,
            financialYear,
            cutoffDate,
            calculationDate: calcDateIso,
            itcAmount: rawItc,
            supplierGstr3bFiled: true,
            isReversalRequired: false,
            reversalAmountInr: 0,
            interestPayableInr: 0,
            statutoryDisposition: 'SUPPLIER_GSTR3B_FILED_NO_REVERSAL',
            statutoryRationale: 'Supplier has furnished Form GSTR-3B. Section 16(2)(c) tax payment requirement satisfied.',
            isAuthorizedToDisburse: false
        };
    }

    if (!isPastCutoff) {
        return {
            invoiceNumber,
            financialYear,
            cutoffDate,
            calculationDate: calcDateIso,
            itcAmount: rawItc,
            supplierGstr3bFiled: false,
            isReversalRequired: false,
            reversalAmountInr: 0,
            interestPayableInr: 0,
            statutoryDisposition: 'WITHIN_RULE_37A_GRACE_PERIOD',
            statutoryRationale: `Supplier has not yet filed GSTR-3B, but calculation date ${calcDateIso} is on or before statutory cutoff ${cutoffDate}. Reversal not yet mandatory.`,
            isAuthorizedToDisburse: false
        };
    }

    // Past 30th November and supplier has not filed GSTR-3B: Reversal is mandatory!
    const cutoffTime = new Date(cutoffDate).getTime();
    const calcTime = new Date(calcDateIso).getTime();
    const delayDays = Math.max(0, Math.floor((calcTime - cutoffTime) / (1000 * 60 * 60 * 24)));
    const annualRate = Number(annualInterestRatePercent) / 100;
    const interestPayableInr = roundToPaisa(rawItc * annualRate * (delayDays / 365));

    return {
        invoiceNumber,
        financialYear,
        cutoffDate,
        calculationDate: calcDateIso,
        itcAmount: rawItc,
        supplierGstr3bFiled: false,
        isReversalRequired: true,
        reversalAmountInr: rawItc,
        delayDaysBeyondNovember30: delayDays,
        interestPayableInr,
        gstr3bReportingTable: 'TABLE_4_B_2_OTHERS_RECLAIMABLE',
        canReclaimOnSupplierFiling: true,
        statutoryDisposition: 'RULE_37A_REVERSAL_MANDATORY',
        statutoryRationale: `Supplier failed to furnish Form GSTR-3B on or before ${cutoffDate}. Input tax credit of ₹${rawItc.toLocaleString('en-IN')} must be reversed under Rule 37A. ${delayDays > 0 ? `Delayed by ${delayDays} days: interest of ₹${interestPayableInr.toLocaleString('en-IN')} payable under Section 50.` : 'Reversal within November returns carries zero interest.'} May be re-availed interest-free upon subsequent supplier GSTR-3B filing.`,
        isAuthorizedToDisburse: false
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: GST Section 52 (TCS by E-Commerce Operators)
 * Central Goods and Services Tax Act, 2017 Section 52.
 * Applicable on every electronic commerce operator (ECO) collecting consideration on behalf of supplier.
 * Rate: 0.5% CGST + 0.5% SGST (1.0% total) for intra-state supplies, or 1.0% IGST for inter-state supplies.
 * Base: Net value of taxable supplies = Taxable supplies (other than Sec 9(5)) LESS returned supplies.
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection52GstTcs({
    grossTaxableSuppliesInr,
    returnedSuppliesInr = 0,
    isInterState = false,
    isSec9_5SpecifiedService = false,
    transactionDate = null
} = {}) {
    const grossSupplies = Math.max(0, assertMandatoryNumeric(grossTaxableSuppliesInr, 'grossTaxableSuppliesInr', 'evaluateSection52GstTcs'));
    const returns = Math.max(0, cleanNumericAmount(returnedSuppliesInr));

    if (isSec9_5SpecifiedService) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_SECTION_9_5_SUPPLY',
            applicableTcsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTcsInr: 0,
            isAuthorizedToDisburse: false,
            governingLaw: 'CGST Act, 2017 Section 52(1) Explanation (Section 9(5) excluded)',
            statutoryRationale: 'Services notified under Section 9(5) (passenger transport, accommodation, housekeeping, restaurant) are subject to direct reverse charge by ECO, and excluded from Section 52 TCS.'
        };
    }

    const netValue = Math.max(0, roundToPaisa(grossSupplies - returns));
    if (netValue <= 0) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_NET_VALUE_ZERO',
            applicableTcsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTcsInr: 0,
            netValueInr: netValue,
            isAuthorizedToDisburse: false,
            governingLaw: 'CGST Act, 2017 Section 52(1)',
            statutoryRationale: 'Net value of taxable supplies after adjusting sales returns is zero or negative. No TCS applicable.'
        };
    }

    const totalRate = 1.0;
    const calculatedTcs = roundToPaisa((netValue * totalRate) / 100);

    const cgstTcs = isInterState ? 0 : roundToPaisa(calculatedTcs / 2);
    const sgstTcs = isInterState ? 0 : roundToPaisa(calculatedTcs - cgstTcs);
    const igstTcs = isInterState ? calculatedTcs : 0;
    const netDisbursedToSupplier = roundToPaisa(Math.max(0, grossSupplies - calculatedTcs));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'COLLECTION_MANDATORY',
        applicableTcsRatePercent: totalRate,
        taxableBaseInr: netValue,
        calculatedTcsInr: calculatedTcs,
        bifurcation: {
            cgstInr: cgstTcs,
            sgstInr: sgstTcs,
            igstInr: igstTcs
        },
        grossSuppliesInr: grossSupplies,
        returnedSuppliesInr: returns,
        netDisbursedToSupplierInr: netDisbursedToSupplier,
        complianceRequirements: {
            monthlyReturnForm: 'GSTR-8',
            filingCutoff: '10th of subsequent calendar month'
        },
        governingLaw: 'CGST Act, 2017 Section 52 / SGST Act Section 52 / IGST Act Section 20',
        isAuthorizedToDisburse: false,
        statutoryRationale: `E-commerce net supplies of INR ${netValue.toLocaleString('en-IN')} subject to 1% TCS = INR ${calculatedTcs.toLocaleString('en-IN')} (${isInterState ? `IGST INR ${igstTcs}` : `CGST INR ${cgstTcs} + SGST INR ${sgstTcs}`}). Net disbursed to supplier: INR ${netDisbursedToSupplier.toLocaleString('en-IN')}. File Form GSTR-8 by 10th.`
    };
}
