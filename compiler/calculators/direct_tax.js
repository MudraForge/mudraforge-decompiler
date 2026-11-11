/**
 * DIRECT TAX DETERMINISTIC CALCULATORS
 * TDS (194Q, 194C, 194J, 194I, 194H, 194A, 194O, 194IA, 194IB, 194M, 194R), TCS (206C(1H)),
 * Non-filer penalty (206AB), Advance Tax, Section 43B statutory payables, Concessional Corporate Tax (115BAA/115JB).
 */
import {
    roundToPaisa,
    roundSection288B,
    decodeTaxIdentifier,
    assertMandatoryNumeric,
    cleanNumericAmount,
    toIntegerPaise,
    fromIntegerPaise,
    MalformedNumericInputError,
    MissingMandatoryParameterError
} from './numeric_utils.js';
import { evaluateLowerDeductionCertificateValidity } from './international.js';

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * AUTHORITATIVE STATUTORY RATE SCHEDULE TABLE
 * ─────────────────────────────────────────────────────────────────────────────
 * Maintains the verified, authoritative statutory rate schedules for Finance Act
 * provisions and Double Taxation Avoidance Agreements (DTAA).
 * Rather than trusting caller-supplied rate numbers, the engine validates or retrieves
 * the legally prescribed rate directly from this verified schedule.
 */
export const AUTHORITATIVE_STATUTORY_RATES = {
    // 1. Finance Act Rates
    'FINANCE_ACT:2025:SCHEDULE_I_PART_II:ITEM_2_B': {
        ratePercent: 10.0,
        source: 'FINANCE_ACT',
        citation: 'Finance Act, 2025 Schedule I Part II Item 2(b)',
        effectiveFrom: '2025-04-01',
        effectiveTo: '2026-03-31',
        applicableCategories: ['TECHNICAL_SERVICES', 'ROYALTY', 'SPECIFIED_FEES']
    },
    'FINANCE_ACT:2026:SCHEDULE_I_PART_II:ITEM_2_B': {
        ratePercent: 10.0,
        source: 'FINANCE_ACT',
        citation: 'Finance Act, 2026 Schedule I Part II Item 2(b)',
        effectiveFrom: '2026-04-01',
        effectiveTo: '2027-03-31',
        applicableCategories: ['TECHNICAL_SERVICES', 'ROYALTY', 'SPECIFIED_FEES']
    },
    // 2. DTAA Double Taxation Avoidance Treaty Rates
    'DTAA_TREATY:INDIA_US:ARTICLE_12_2': {
        ratePercent: 15.0,
        source: 'DTAA_TREATY',
        citation: 'India-US Double Taxation Avoidance Agreement Article 12(2)',
        effectiveFrom: '1991-01-01',
        effectiveTo: '2099-12-31',
        applicableCategories: ['FTS', 'ROYALTY', 'TECHNICAL_SERVICES']
    },
    'DTAA_TREATY:INDIA_SINGAPORE:ARTICLE_12_2': {
        ratePercent: 10.0,
        source: 'DTAA_TREATY',
        citation: 'India-Singapore Double Taxation Avoidance Agreement Article 12(2)',
        effectiveFrom: '1994-08-08',
        effectiveTo: '2099-12-31',
        applicableCategories: ['FTS', 'ROYALTY', 'TECHNICAL_SERVICES']
    },
    'DTAA_TREATY:INDIA_GERMANY:ARTICLE_12_2': {
        ratePercent: 10.0,
        source: 'DTAA_TREATY',
        citation: 'India-Germany Double Taxation Avoidance Agreement Article 12(2)',
        effectiveFrom: '1996-10-26',
        effectiveTo: '2099-12-31',
        applicableCategories: ['FTS', 'ROYALTY', 'TECHNICAL_SERVICES']
    }
};

/**
 * Resolves the statutory missing-PAN penalty floor:
 * - Income-tax Act, 1961 Section 206AA(1) First Proviso
 * - Income-tax Act, 2025 Section 397(2)
 *
 * Statutory Scoping:
 * - 5.0% applies strictly to:
 *   (a) Section 194Q / Section 393(1), Table 1, Sl. No. 8(ii) (Purchase of goods)
 *   (b) Section 194-O / Section 393(1), Table 1, Sl. No. 8(i) (E-commerce participants)
 * - 20.0% applies to all other categories:
 *   Section 194J / Table 1, Sl. No. 6(iii) (Fees for professional/technical services),
 *   Section 194C / Table 1, Sl. No. 6(i) (Contractors),
 *   Section 194I / Table 1, Sl. No. 2 (Rent),
 *   Section 194H / Table 1, Sl. No. 1(ii) (Commission),
 *   Section 195 / Section 394 (Non-residents).
 */
export function resolveMissingPanPenaltyFloor({
    sectionCode = '194Q',
    tableItem = null,
    isAct2025 = false
} = {}) {
    const sec = String(sectionCode || '').toUpperCase().trim();
    const item = String(tableItem || '').toLowerCase().trim();

    const isGoodsPurchase = sec === '194Q' || sec === '393' || item.includes('8_ii') || item.includes('8(ii)');
    const isEcommerce = sec === '194O' || sec === '194-O' || item.includes('8_i') || item.includes('8_v') || item.includes('8(i)');

    if (isGoodsPurchase || isEcommerce) {
        return {
            penaltyFloorPercent: 5.0,
            isConcessional5PercentApplicable: true,
            categoryScope: isGoodsPurchase ? 'SPECIFIED_GOODS_PURCHASE_5_PERCENT' : 'SPECIFIED_ECOMMERCE_PARTICIPANT_5_PERCENT',
            governingProviso: isAct2025
                ? `Income-tax Act, 2025 Section 397(2)(b)(i)(${isGoodsPurchase ? 'C' : 'D'}) read with Section 393(1), Table 1, Sl. No. ${isGoodsPurchase ? '8(ii)' : '8(i)'}`
                : `Income-tax Act, 1961 Section 206AA(1) First Proviso read with Section ${sec}`
        };
    }

    return {
        penaltyFloorPercent: 20.0,
        isConcessional5PercentApplicable: false,
        categoryScope: 'GENERAL_SERVICES_AND_FEES_20_PERCENT',
        governingProviso: isAct2025
            ? 'Income-tax Act, 2025 Section 397(2)(a)'
            : 'Income-tax Act, 1961 Section 206AA(1)(iii)'
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * STATUTORY RATE PROVENANCE & EVIDENCE VALIDATOR
 * ─────────────────────────────────────────────────────────────────────────────
 * Validates that an asserted rate in force is grounded in authoritative statutory evidence:
 * 1. Checks citation against Authoritative Statutory Rate Table.
 * 2. Rejects fabricated rate values that contradict the authoritative statutory schedule.
 * 3. Enforces temporal validity window [effectiveFrom, effectiveTo] against transactionDate.
 * 4. Disallows private commercial contracts from modifying statutory withholding rates.
 * 5. Rejects unverified or contradictory rate evidence (fails closed).
 */
export function validateRateEvidence({
    rateInForce = null,
    ratesInForceSource = 'STATUTORY_BASE',
    sourceReference = null,
    rateScheduleKey = null,
    effectiveFrom = null,
    effectiveTo = null,
    transactionDate = null
} = {}) {
    // 0. Disallow private commercial contracts from attempting to define statutory rate in force
    if (ratesInForceSource === 'COMMERCIAL_CONTRACT' || ratesInForceSource === 'PRIVATE_AGREEMENT') {
        return {
            isValid: false,
            status: 'INVALID_LEGAL_SOURCE',
            error: 'Private commercial contract cannot define or modify statutory "rates in force" under Income-tax Act, 1961 Section 2(37A) / Income-tax Act, 2025 Section 397(1)(b).'
        };
    }

    // 1. Explicit Schedule Key Resolution with Fail-Closed Rejection of Unknown Keys
    if (rateScheduleKey && !AUTHORITATIVE_STATUTORY_RATES[rateScheduleKey]) {
        return {
            isValid: false,
            status: 'UNKNOWN_RATE_SCHEDULE_KEY',
            error: `Unknown rateScheduleKey "${rateScheduleKey}". Key is not registered in authoritative statutory schedule registry.`
        };
    }

    let schedule = null;
    if (rateScheduleKey && AUTHORITATIVE_STATUTORY_RATES[rateScheduleKey]) {
        schedule = AUTHORITATIVE_STATUTORY_RATES[rateScheduleKey];
    } else if (sourceReference) {
        const refLower = String(sourceReference).toLowerCase();
        for (const [k, v] of Object.entries(AUTHORITATIVE_STATUTORY_RATES)) {
            if (v.citation.toLowerCase().includes(refLower) || refLower.includes(v.citation.toLowerCase())) {
                schedule = v;
                break;
            }
        }
    }

    if (rateInForce === null || rateInForce === undefined) {
        if (schedule) {
            const effStart = schedule.effectiveFrom;
            const effEnd = schedule.effectiveTo;
            if (transactionDate) {
                const tx = String(transactionDate).slice(0, 10);
                if (effStart && tx < String(effStart).slice(0, 10)) {
                    return {
                        isValid: false,
                        status: 'RATE_PRE_EFFECTIVE',
                        error: `Rate in force evidence is not yet effective on transaction date ${tx} (effective from ${effStart}).`
                    };
                }
                if (effEnd && tx > String(effEnd).slice(0, 10)) {
                    return {
                        isValid: false,
                        status: 'RATE_EXPIRED',
                        error: `Rate in force evidence expired on ${effEnd} prior to transaction date ${tx}.`
                    };
                }
            }
            return {
                isValid: true,
                status: 'RETRIEVED_FROM_AUTHORITATIVE_SCHEDULE',
                ratePercent: schedule.ratePercent,
                ratesInForceSource: schedule.source,
                sourceReference: schedule.citation,
                effectiveFrom: schedule.effectiveFrom,
                effectiveTo: schedule.effectiveTo,
                reason: `Rate of ${schedule.ratePercent}% retrieved directly from authoritative statutory schedule (${schedule.citation}).`
            };
        }
        return {
            isValid: true,
            status: 'DEFAULT_STATUTORY_BASE',
            ratePercent: 0.1,
            ratesInForceSource: 'STATUTORY_BASE',
            sourceReference: 'Statutory Base Provision',
            reason: 'No external rate in force asserted; defaulting to baseline statutory rate.'
        };
    }

    const numericRate = Number(rateInForce);
    if (!Number.isFinite(numericRate) || isNaN(numericRate) || numericRate < 0) {
        return {
            isValid: false,
            status: 'INVALID_NUMERIC_RATE',
            error: `Invalid rateInForce "${rateInForce}": must be a non-negative finite number.`
        };
    }

    const recognizedSources = ['FINANCE_ACT', 'DTAA_TREATY', 'CBDT_NOTIFIED', 'STATUTORY_BASE'];
    if (ratesInForceSource && !recognizedSources.includes(ratesInForceSource)) {
        return {
            isValid: false,
            status: 'UNRECOGNIZED_RATE_SOURCE',
            error: `Unrecognized ratesInForceSource "${ratesInForceSource}". Must be one of: ${recognizedSources.join(', ')}.`
        };
    }

    // If authoritative schedule is matched, enforce that supplied rate matches prescribed rate
    if (schedule && Math.abs(numericRate - schedule.ratePercent) > 0.001) {
        return {
            isValid: false,
            status: 'RATE_CONTRADICTS_AUTHORITATIVE_SCHEDULE',
            error: `Supplied rate in force (${numericRate}%) contradicts authoritative statutory schedule for "${schedule.citation}" (prescribed rate: ${schedule.ratePercent}%).`
        };
    }

    // Require citation reference for higher rates
    if (numericRate > 0.1 && ratesInForceSource && ratesInForceSource !== 'STATUTORY_BASE' && !sourceReference && !schedule) {
        return {
            isValid: false,
            status: 'UNAUTHENTICATED_RATE_EVIDENCE',
            error: `Rate in force (${numericRate}%) from source "${ratesInForceSource}" requires an authoritative legal citation in sourceReference (e.g. "Finance Act 2025 Schedule I Part II" or "India-US DTAA Article 12(2)").`
        };
    }

    // Temporal validity window check
    const effStart = schedule?.effectiveFrom || effectiveFrom;
    const effEnd = schedule?.effectiveTo || effectiveTo;

    if (transactionDate) {
        const tx = String(transactionDate).slice(0, 10);
        if (effStart && tx < String(effStart).slice(0, 10)) {
            return {
                isValid: false,
                status: 'RATE_PRE_EFFECTIVE',
                error: `Rate in force evidence is not yet effective on transaction date ${tx} (effective from ${effStart}).`
            };
        }
        if (effEnd && tx > String(effEnd).slice(0, 10)) {
            return {
                isValid: false,
                status: 'RATE_EXPIRED',
                error: `Rate in force evidence expired on ${effEnd} prior to transaction date ${tx}.`
            };
        }
    }

    return {
        isValid: true,
        status: schedule ? 'VERIFIED_AGAINST_AUTHORITATIVE_SCHEDULE' : 'AUTHENTICATED_STATUTORY_EVIDENCE',
        ratePercent: numericRate,
        ratesInForceSource: schedule?.source || ratesInForceSource || 'STATUTORY_BASE',
        sourceReference: schedule?.citation || sourceReference,
        effectiveFrom: effStart,
        effectiveTo: effEnd,
        reason: `Rate of ${numericRate}% verified against authoritative evidence (${schedule?.source || ratesInForceSource || 'STATUTORY_BASE'}: ${schedule?.citation || sourceReference || 'Statutory Baseline'}).`
    };
}

export function evaluateSection194Q({
    invoiceAmount,
    priorFyPurchases = 0,
    priorPurchasesInCurrentFyInr = undefined,
    currentFyPriorPurchasesInr = undefined,
    precedingFyPurchasesInr = undefined,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    buyerTurnoverCr = null,
    sellerTurnoverCr = null,
    isBuyerPrescribed = null,
    isSellerPrescribed = null,
    buyerName = null,
    sellerName = null,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null,
    gstAmountInr = 0,
    isAdvancePayment = false,
    creditNoteInr = 0,
    creditNoteTiming = 'BEFORE_TAX_DEDUCTION', // 'BEFORE_TAX_DEDUCTION' | 'AFTER_TAX_DEDUCTION'
    advanceNettedInr = 0,
    priorAdvanceTdsDeductedInr = 0,
    isImportOfGoods = false,
    sellerIsNonResident = false,
    rateInForce = null,
    ratesInForceSource = null, // 'FINANCE_ACT' | 'DTAA_TREATY' | 'CBDT_NOTIFIED' | 'STATUTORY_BASE'
    sourceReference = null,
    rateScheduleKey = null,
    rateEffectiveFrom = null,
    rateEffectiveTo = null
} = {}) {
    const rawInvoiceAmount = Number(invoiceAmount) || 0;
    const gstAmount = Number(gstAmountInr) || 0;
    const creditNote = Number(creditNoteInr) || 0;
    const advanceNetted = Number(advanceNettedInr) || 0;
    const priorAdvanceTds = Number(priorAdvanceTdsDeductedInr) || 0;

    // CBDT Circular No. 13 of 2021 Question 3:
    // Does Section 194Q apply to import of goods from non-residents?
    // Clarification: Section 194Q does not apply to an import of goods where the seller
    // is a non-resident without a permanent establishment in India. (Section 195 applies instead).
    const isForeignVendorImport = Boolean(isImportOfGoods) || Boolean(sellerIsNonResident);
    if (isForeignVendorImport) {
        const effDate = transactionDate || creditDate || paymentDate || null;
        const isAct2025 = effDate ? effDate >= '2026-04-01' : false;
        return {
            invoiceAmountInr: rawInvoiceAmount,
            taxableBase: 0,
            tdsRatePercent: 0,
            calculatedTdsInr: 0,
            netWireInr: rawInvoiceAmount,
            isTdsApplicable: false,
            isSection194QApplicable: false,
            isExemptForeignImport: true,
            effectiveDate: effDate,
            regimeStatus: effDate ? 'DETERMINED' : 'INDETERMINATE_PENDING_TRANSACTION_DATE',
            isAct2025,
            governingSection: isAct2025 ? 'CBDT Circular No. 13 of 2021 Question 3 (Read with Section 393(2))' : 'CBDT Circular No. 13 of 2021 Question 3 (Read with Section 195)',
            statutoryReason: 'CBDT Circular 13/2021 Question 3 explicitly clarifies Section 194Q does not apply to import of goods from non-resident sellers lacking a permanent establishment in India. Governed under cross-border provisions (Section 195 / Section 393(2)).',
            statutoryDisposition: 'EXEMPT_IMPORT_OF_GOODS_FOREIGN_VENDOR'
        };
    }

    // CBDT Circular No. 13 of 2021 Question 1:
    // If tax is separately indicated on invoice and transaction is not advance payment,
    // TDS under Section 194Q is deducted on basic goods value excluding GST component.
    // If advance payment is made before invoice issuance, TDS applies to gross amount.
    const effectivePurchaseAmount = isAdvancePayment 
        ? rawInvoiceAmount 
        : Math.max(0, rawInvoiceAmount - gstAmount);
    
    // CBDT Circular No. 13 of 2021 Question 4 (Purchase Returns & Credit Notes):
    // 1. Credit Note BEFORE withholding (agreed price discount, immediate defect return before ledger entry):
    //    Deduction applies on actual net consideration per Section 194Q(1).
    // 2. Purchase Return AFTER withholding:
    //    Tax already deducted cannot be refunded or adjusted against government remittance.
    //    Purchase value reduction is carried forward for adjustment against subsequent purchases from the same seller in the same FY.
    const isCreditNoteBeforeWithholding = creditNoteTiming !== 'AFTER_TAX_DEDUCTION';
    const amount = isCreditNoteBeforeWithholding 
        ? Math.max(0, effectivePurchaseAmount - creditNote)
        : effectivePurchaseAmount;
    const creditNoteCarryForwardInr = !isCreditNoteBeforeWithholding ? creditNote : 0;
    const cbdtCircular13Q4Status = creditNote > 0
        ? (isCreditNoteBeforeWithholding ? 'CREDIT_NOTE_REDUCED_PURCHASE_VALUE_BEFORE_WITHHOLDING' : 'CREDIT_NOTE_AFTER_WITHHOLDING_CARRIED_FORWARD')
        : 'N/A';
    const creditNoteStatutoryCitation = creditNote > 0
        ? (isCreditNoteBeforeWithholding
            ? 'Section 194Q(1): Deduction on sum for purchase of goods, reflecting net agreed consideration after pre-withholding credit note.'
            : 'CBDT Circular 13/2021 Question 4: Tax deducted prior to purchase return is preserved; credit note value is carried forward for subsequent purchases from same seller in same FY.')
        : null;

    // Disambiguate Current FY purchases vs Preceding FY purchases:
    // Section 194Q(1) statutory threshold evaluates cumulative purchases in the CURRENT previous year (current FY).
    // Purchases from the preceding financial year do NOT consume the current financial year threshold.
    const prior = Math.max(0, Number(
        currentFyPriorPurchasesInr !== undefined
            ? currentFyPriorPurchasesInr
            : (priorPurchasesInCurrentFyInr !== undefined ? priorPurchasesInCurrentFyInr : priorFyPurchases)
    ) || 0);
    const cumulative = prior + amount;
    const THRESHOLD = 5000000; // ₹50 Lakhs

    // Buyer turnover prescription under Section 194Q(1) Explanation:
    // Person whose total sales, gross receipts or turnover exceeds ₹10 Crore in preceding FY.
    let buyerPrescriptionStatus = 'INDETERMINATE';
    if (isBuyerPrescribed !== null && isBuyerPrescribed !== undefined) {
        buyerPrescriptionStatus = isBuyerPrescribed ? 'APPLICABLE' : 'NOT_APPLICABLE';
    } else if (buyerTurnoverCr !== null && buyerTurnoverCr !== undefined) {
        buyerPrescriptionStatus = Number(buyerTurnoverCr) > 10 ? 'APPLICABLE' : 'NOT_APPLICABLE';
    }

    let sellerPrescriptionStatus = 'INDETERMINATE';
    if (isSellerPrescribed !== null && isSellerPrescribed !== undefined) {
        sellerPrescriptionStatus = isSellerPrescribed ? 'APPLICABLE' : 'NOT_APPLICABLE';
    } else if (sellerTurnoverCr !== null && sellerTurnoverCr !== undefined) {
        sellerPrescriptionStatus = Number(sellerTurnoverCr) > 10 ? 'APPLICABLE' : 'NOT_APPLICABLE';
    }

    const isSection194QApplicable = buyerPrescriptionStatus === 'APPLICABLE';
    const isBuyerPrescriptionIndeterminate = buyerPrescriptionStatus === 'INDETERMINATE';

    // Incremental threshold absorption formula with Paisa integer precision:
    // Delta = max(0, cumulative - THRESHOLD) - max(0, prior - THRESHOLD)
    // Taxable base is min(amount, Delta)
    const incrementalExcess = roundToPaisa(Math.max(0, cumulative - THRESHOLD) - Math.max(0, prior - THRESHOLD));
    const conditionalTaxableBase = roundToPaisa(Math.max(0, Math.min(amount, incrementalExcess)));
    const taxableBase = isSection194QApplicable 
        ? conditionalTaxableBase 
        : (isBuyerPrescriptionIndeterminate ? conditionalTaxableBase : 0);

    // Temporal Resolution: "Credit or Payment, whichever is earlier"
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isTemporalStraddle = Boolean(
        creditDate && paymentDate &&
        ((creditDate < '2026-04-01' && paymentDate >= '2026-04-01') ||
         (paymentDate < '2026-04-01' && creditDate >= '2026-04-01'))
    );

    const temporalStraddlingAnalysis = isTemporalStraddle ? {
        isStraddlingCommencement: true,
        creditDate,
        paymentDate,
        crystallizedDate: effectiveDate,
        withholdingTrigger,
        governingRegime: effectiveDate >= '2026-04-01' ? 'ACT_2025' : 'ACT_1961',
        statutoryRationale: effectiveDate < '2026-04-01'
            ? `Tax withholding obligation crystallized on ${effectiveDate} (${withholdingTrigger}) under the Income-tax Act, 1961 prior to 2025 Act commencement. Subsequent settlement on ${effectiveDate === creditDate ? paymentDate : creditDate} creates no duplicate withholding under the 2025 Act.`
            : `Both credit and payment occurred on or after 1 April 2026. Governed exclusively by the Income-tax Act, 2025.`
    } : null;

    // Date & Regime analysis (preserves uncertainty when effectiveDate is null rather than coercing machine date)
    let isAct2025 = null;
    let isPost206cSunset = null;
    let governingWithholdingSection = "";
    let regimeStatus = "DETERMINED";

    if (effectiveDate) {
        isAct2025 = effectiveDate >= '2026-04-01';
        isPost206cSunset = effectiveDate >= '2025-04-01';
        governingWithholdingSection = isSection194QApplicable || isBuyerPrescriptionIndeterminate
            ? (isAct2025 ? "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 8(ii)" : "Income-tax Act, 1961 Section 194Q")
            : "N/A (Buyer turnover does not exceed ₹10 Cr threshold)";
    } else {
        // Date is missing: preserve uncertainty rather than coercing machine date
        regimeStatus = "INDETERMINATE_PENDING_TRANSACTION_DATE";
        isAct2025 = false;
        isPost206cSunset = true; // Section 206C(1H) sunset on 1 April 2025
        governingWithholdingSection = isSection194QApplicable || isBuyerPrescriptionIndeterminate
            ? "Income-tax Act, 1961 Section 194Q (or Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 8(ii) if credited/paid on or after 1 April 2026)"
            : "N/A (Buyer turnover does not exceed ₹10 Cr threshold)";
    }

    let governingPanPenaltySection = isAct2025 
        ? "Income-tax Act, 2025 Section 397(2)(b)(i)(C) read with Section 393(1), Table 1, Sl. No. 8(ii)" 
        : "Income-tax Act, 1961 Section 206AA(1) First Proviso read with Section 194Q";
    
    // Resolve certificate framework with Section 536(2) transitional savings and validity evaluation
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    let governingCertificateSection = certResolution.governingCertificateSection;

    // Rate determination per Section 194Q / Section 393(1), Table 1, Sl. No. 8(ii) read with
    // Section 206AA(1) First Proviso / Section 397(2) First Proviso:
    // Withholding for missing/inoperative PAN applies the HIGHEST of:
    // (i) rate specified in the Act (0.1%),
    // (ii) rate or rates in force (if specified and higher), or
    // (iii) statutory penalty floor (5.0% for goods purchase; 20.0% standard).
    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    const isCertificateValid = certResolution.isValid;
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && isCertificateValid;

    // Validate rate in force against authoritative evidence (Section 2(37A) / Section 397(1)(b))
    const rateEvidence = validateRateEvidence({
        rateInForce,
        ratesInForceSource: ratesInForceSource || (rateInForce > 0.1 ? 'FINANCE_ACT' : 'STATUTORY_BASE'),
        sourceReference: sourceReference || (rateInForce > 0.1 ? 'Finance Act Rate Schedule' : null),
        rateScheduleKey,
        effectiveFrom: rateEffectiveFrom,
        effectiveTo: rateEffectiveTo,
        transactionDate: effectiveDate
    });

    if (rateEvidence && !rateEvidence.isValid) {
        regimeStatus = 'INDETERMINATE_RATE_EVIDENCE_REJECTED';
    }

    const statutoryBaseRate = 0.1;
    // Authenticated rate is applied if evidence is valid; unauthenticated / invalid rates fail closed to baseline
    const effectiveRateInForce = rateEvidence.isValid ? rateEvidence.ratePercent : statutoryBaseRate;
    const panFloorResolution = resolveMissingPanPenaltyFloor({
        sectionCode: '194Q',
        tableItem: isAct2025 ? '393_8_ii' : null,
        isAct2025
    });
    const missingPanPenaltyFloor = panFloorResolution.penaltyFloorPercent;
    governingPanPenaltySection = panFloorResolution.governingProviso;
    const missingPanRate = Math.max(statutoryBaseRate, effectiveRateInForce, missingPanPenaltyFloor);
    const standardRatePercent = panAvailable ? statutoryBaseRate : missingPanRate;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    // Cumulative Certificate Ceiling Exhaustion Tracking
    const certificateCeiling = certificateMonetaryCeilingInr !== null && certificateMonetaryCeilingInr !== undefined
        ? Number(certificateMonetaryCeilingInr)
        : (lowerDeductionCertificate?.monetaryCeilingInr !== undefined ? Number(lowerDeductionCertificate.monetaryCeilingInr) : null);
    
    const certificatePriorUtilized = Number(certificatePriorUtilizedInr) || 
        (lowerDeductionCertificate?.consumedAmountInr !== undefined ? Number(lowerDeductionCertificate.consumedAmountInr) : 0);
    
    const availableCeilingHeadroom = certificateCeiling !== null
        ? Math.max(0, certificateCeiling - certificatePriorUtilized)
        : null;

    let certifiedEligibleBase = 0;
    let uncoveredExcessBase = taxableBase;
    let ceilingExhaustionStatus = 'NO_LOWER_DEDUCTION_CERTIFICATE';

    if (hasLowerDeduction) {
        if (certificateCeiling === null) {
            certifiedEligibleBase = taxableBase;
            uncoveredExcessBase = 0;
            ceilingExhaustionStatus = 'UNCONSTRAINED';
        } else if (availableCeilingHeadroom >= taxableBase) {
            certifiedEligibleBase = taxableBase;
            uncoveredExcessBase = 0;
            ceilingExhaustionStatus = 'WITHIN_CEILING';
        } else if (availableCeilingHeadroom > 0) {
            certifiedEligibleBase = availableCeilingHeadroom;
            uncoveredExcessBase = taxableBase - availableCeilingHeadroom;
            ceilingExhaustionStatus = 'PARTIALLY_EXHAUSTED';
        } else {
            certifiedEligibleBase = 0;
            uncoveredExcessBase = taxableBase;
            ceilingExhaustionStatus = 'FULLY_EXHAUSTED';
        }
    } else if (lowerDeductionRate !== null && !isCertificateValid) {
        ceilingExhaustionStatus = certResolution.status;
    }

    let certifiedTdsInr = 0;
    let uncoveredTdsInr = 0;

    if (isSection194QApplicable || isBuyerPrescriptionIndeterminate) {
        certifiedTdsInr = Number(((certifiedEligibleBase * certifiedRatePercent) / 100).toFixed(2));
        uncoveredTdsInr = Number(((uncoveredExcessBase * standardRatePercent) / 100).toFixed(2));
    }
    const calculatedTdsInr = Number((certifiedTdsInr + uncoveredTdsInr).toFixed(2));
    const effectiveRatePercent = taxableBase > 0 
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : 0.0;
    const statutoryRatePercent = isSection194QApplicable || isBuyerPrescriptionIndeterminate
        ? (hasLowerDeduction ? certifiedRatePercent : standardRatePercent)
        : 0.0;

    let sellerTcsStatus = "";

    if (isPost206cSunset) {
        sellerTcsStatus = "Section 206C(1H) was omitted/withdrawn effective 1 April 2025 by Finance Act, 2025. Sellers no longer collect TCS on goods. Withholding rests solely with the buyer.";
    } else if (sellerPrescriptionStatus === 'APPLICABLE' && buyerPrescriptionStatus === 'NOT_APPLICABLE') {
        sellerTcsStatus = "Buyer turnover does not exceed ₹10 Cr, so Section 194Q does not apply. Seller is prescribed (turnover > ₹10 Cr) and must collect TCS under Section 206C(1H) on receipts exceeding ₹50 Lakhs.";
    } else {
        sellerTcsStatus = "Section 194Q holds statutory precedence over Section 206C(1H). Because the buyer is prescribed (turnover > ₹10 Cr) and mandated to deduct TDS under Section 194Q, the seller's Section 206C(1H) collection duty is statutorily displaced/excused.";
    }

    const effectiveBuyerName = buyerName || "Buyer";
    const effectiveSellerName = sellerName || "Seller";

    let applicabilityDisposition = "DETERMINED_APPLICABLE";
    if (buyerPrescriptionStatus === 'NOT_APPLICABLE') {
        applicabilityDisposition = "EXEMPT_BELOW_BUYER_TURNOVER_THRESHOLD";
    } else if (buyerPrescriptionStatus === 'INDETERMINATE') {
        applicabilityDisposition = "INDETERMINATE_BUYER_TURNOVER_UNVERIFIED";
    } else if (rateEvidence && !rateEvidence.isValid) {
        applicabilityDisposition = "INDETERMINATE_RATE_EVIDENCE_REJECTED";
    }

    let formulaText = "";
    if (buyerPrescriptionStatus === 'NOT_APPLICABLE') {
        formulaText = "Section 194Q inapplicable: Buyer turnover does not exceed ₹10 Crore threshold.";
    } else {
        const qualifier = isBuyerPrescriptionIndeterminate ? " [Contingent on > ₹10 Cr Buyer Turnover]" : "";
        if (hasLowerDeduction) {
            if (ceilingExhaustionStatus === 'PARTIALLY_EXHAUSTED') {
                formulaText = `Partial Ceiling Split: ₹${certifiedEligibleBase.toLocaleString('en-IN')} @ ${certifiedRatePercent}% (Sanctioned Certificate Headroom) + ₹${uncoveredExcessBase.toLocaleString('en-IN')} @ ${standardRatePercent}% (Marginal Excess beyond Sanctioned Limit) = ₹${calculatedTdsInr.toLocaleString('en-IN')}${qualifier}`;
            } else if (ceilingExhaustionStatus === 'FULLY_EXHAUSTED') {
                formulaText = `Certificate Fully Exhausted: Sanctioned ceiling of ₹${certificateCeiling.toLocaleString('en-IN')} is 100% utilized. Taxable base (₹${taxableBase.toLocaleString('en-IN')}) deducted at standard statutory rate of ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}${qualifier}`;
            } else {
                formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Lower Deduction Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}${qualifier}`;
            }
        } else if (lowerDeductionRate !== null && !isCertificateValid) {
            formulaText = `Lower Deduction Certificate Disallowed (${certResolution.status}): ${certResolution.reason}. Deducted at standard statutory rate of ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}${qualifier}`;
        } else if (panAvailable) {
            formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 0.1% = ₹${calculatedTdsInr.toLocaleString('en-IN')}${qualifier}`;
        } else {
            const penaltyProvisoCitation = isAct2025 ? 'Section 397(2)(b)(i)(C) 5% cap for goods' : 'Section 206AA(1) First Proviso 5% cap for goods';
            formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% (${penaltyProvisoCitation}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}${qualifier}`;
        }
    }

    const remainingTdsInr = Math.max(0, roundToPaisa(calculatedTdsInr - priorAdvanceTds));
    const netWireInr = roundToPaisa(rawInvoiceAmount - advanceNetted - remainingTdsInr);

    return {
        invoiceAmount: amount,
        rawInvoiceAmount,
        gstAmountInr: gstAmount,
        isAdvancePayment,
        creditNoteInr: creditNote,
        creditNoteTiming,
        creditNoteCarryForwardInr,
        cbdtCircular13Q4Status,
        creditNoteStatutoryCitation,
        advanceNettedInr: advanceNetted,
        priorAdvanceTdsDeductedInr: priorAdvanceTds,
        totalOrderTdsInr: calculatedTdsInr,
        remainingTdsInr,
        netWireInr,
        effectivePurchaseAmount,
        cbdtCircular13NettingApplied: !isAdvancePayment && gstAmount > 0,
        priorFyPurchases: prior,
        priorPurchasesInCurrentFyInr: prior,
        currentFyPriorPurchasesInr: prior,
        precedingFyPurchasesInr: precedingFyPurchasesInr !== undefined ? Number(precedingFyPurchasesInr) : null,
        thresholdAccountingBasis: 'Section 194Q(1) evaluates cumulative purchases in the CURRENT previous year (current FY). Preceding financial year purchases do not consume this threshold.',
        thresholdRemainingInCurrentFyInr: Math.max(0, THRESHOLD - prior),
        cumulativeFyPurchases: cumulative,
        statutoryThreshold: THRESHOLD,
        thresholdBreached: cumulative > THRESHOLD,
        taxableBase,
        conditionalTaxableBase,
        isPanFurnished: panAvailable,
        statutoryRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        certifiedTdsInr,
        uncoveredTdsInr,
        certificateCeiling,
        certificatePriorUtilized,
        availableCeilingHeadroom,
        certifiedEligibleBase,
        uncoveredExcessBase,
        ceilingExhaustionStatus,
        cumulativePurchasesInCurrentFyInr: cumulative,
        buyerTurnoverPrecondition: {
            status: buyerPrescriptionStatus,
            thresholdCr: 10,
            rule: "Section 194Q(1) applies strictly if buyer's total turnover/gross receipts in preceding financial year exceeded ₹10 Crore."
        },
        steelCastingHsnGstAssumption: "Steel turbine castings (HSN Chapter 72/73) standardly attract 18% GST. Confirm whether invoice is inclusive of GST or if GST invoice is pending.",
        buyerTurnoverCr: buyerTurnoverCr !== null ? Number(buyerTurnoverCr) : null,
        sellerTurnoverCr: sellerTurnoverCr !== null ? Number(sellerTurnoverCr) : null,
        buyerPrescriptionStatus,
        sellerPrescriptionStatus,
        isSection194QApplicable: buyerPrescriptionStatus === 'APPLICABLE',
        isBuyerPrescribed: buyerPrescriptionStatus === 'APPLICABLE',
        isSellerPrescribed: sellerPrescriptionStatus === 'APPLICABLE',
        applicabilityDisposition,
        statutoryDetermination: (buyerPrescriptionStatus === 'APPLICABLE' && regimeStatus === 'DETERMINED' && (!rateEvidence || rateEvidence.isValid)) ? 'APPLICABLE' : (buyerPrescriptionStatus === 'NOT_APPLICABLE' && (!rateEvidence || rateEvidence.isValid) ? 'EXEMPT' : 'INDETERMINATE'),
        regimeStatus,
        isAct2025: Boolean(isAct2025),
        creditDate,
        paymentDate,
        effectiveWithholdingDate: effectiveDate,
        withholdingTrigger,
        temporalStraddlingAnalysis,
        lowerDeductionRate: hasLowerDeduction ? Number(lowerDeductionRate) : null,
        lowerDeductionCertificate: lowerDeductionCertificate || (hasLowerDeduction ? { ratePercent: Number(lowerDeductionRate), applied: true } : null),
        governingWithholdingSection,
        governingPanPenaltySection,
        governingCertificateSection,
        isSection536TransitionalSavingsApplied: certResolution.isSection536TransitionalSavingsApplied,
        transitionalSavingsClause: certResolution.transitionalSavingsClause,
        certificateValidity: certResolution,
        certificateDisallowanceReason: certResolution.isValid ? null : certResolution.reason,
        legalPrecedence: "Section 194Q (Buyer TDS) overrides Section 206C(1H) (Seller TCS)",
        sellerTcsStatus,
        requiredParty: isSection194QApplicable ? effectiveBuyerName : (sellerPrescriptionStatus === 'APPLICABLE' ? `${effectiveSellerName} (under Section 206C(1H))` : "None"),
        rateInForce: effectiveRateInForce,
        ratesInForceSource: ratesInForceSource || (effectiveRateInForce > 0.1 ? 'FINANCE_ACT' : 'STATUTORY_BASE'),
        statutoryRateHierarchy: {
            statutoryBaseRate,
            rateInForce: effectiveRateInForce,
            ratesInForceSource: ratesInForceSource || (effectiveRateInForce > 0.1 ? 'FINANCE_ACT' : 'STATUTORY_BASE'),
            penaltyFloorRate: missingPanPenaltyFloor,
            governingProviso: governingPanPenaltySection,
            rule: 'HIGHER_OF_BASE_RATE_IN_FORCE_OR_PENALTY_FLOOR'
        },
        rateEvidenceValidation: rateEvidence,
        statutoryFormula: formulaText
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 4B. SECTION 206C(1H): DETERMINISTIC SELLER TCS ON GOODS ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates tax collection at source by sellers on sale of goods:
 * - Preceding FY Seller Turnover precondition (> ₹10 Crore)
 * - Buyer Turnover precondition: If buyer turnover > ₹10 Cr, Section 194Q
 *   takes absolute statutory precedence under Section 206C(1H) Second Proviso.
 *   Seller is statutorily excused from collecting TCS.
 * - Threshold: Aggregate consideration received in FY > ₹50 Lakhs.
 * - Rate: 0.1% if PAN furnished and operative.
 * - Section 206CC Proviso Rate Cap: If PAN is missing or inoperative,
 *   rate is strictly capped at 1.0% (NOT 5% or 20%).
 * - Finance Act, 2025 Sunset: Omitted/withdrawn effective 1 April 2025.
 *   On or after 1 April 2025, TCS collection is statutorily barred.
 */
export function evaluateSection206C1H({
    receiptAmountInr = 0,
    priorFyReceiptsInr = 0,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    sellerTurnoverCr = null,
    buyerTurnoverCr = null,
    isBuyerLiableAndDeducting194Q = false,
    sellerName = null,
    buyerName = null
} = {}) {
    const receipt = Math.max(0, Number(receiptAmountInr) || 0);
    const prior = Math.max(0, Number(priorFyReceiptsInr) || 0);
    const cumulative = prior + receipt;
    const THRESHOLD = 5000000; // ₹50 Lakhs

    const effectiveDate = transactionDate || new Date().toISOString().split('T')[0];
    const isPostSunset = effectiveDate >= '2025-04-01';

    // 1. Finance Act 2025 Sunset Check (Effective 1 April 2025)
    if (isPostSunset) {
        return {
            receiptAmountInr: receipt,
            priorFyReceiptsInr: prior,
            cumulativeFyReceiptsInr: cumulative,
            taxableBase: 0,
            tcsRatePercent: 0,
            calculatedTcsInr: 0,
            isTcsApplicable: false,
            isOmittedByFinanceAct2025: true,
            effectiveDate,
            governingSection: 'Section 206C(1H) (Omitted w.e.f. 1 April 2025)',
            statutoryReason: 'Section 206C(1H) was omitted/withdrawn effective 1 April 2025 by Finance Act, 2025. Sellers no longer collect TCS on goods. Obligation rests solely with buyer withholding under the Income-tax Act.',
            statutoryDisposition: 'OMITTED_SUNSET_BY_FINANCE_ACT_2025'
        };
    }

    // 2. Seller Turnover Precondition (> ₹10 Crore in preceding FY)
    const isSellerPrescribed = sellerTurnoverCr !== null && Number(sellerTurnoverCr) > 10;
    if (sellerTurnoverCr !== null && !isSellerPrescribed) {
        return {
            receiptAmountInr: receipt,
            priorFyReceiptsInr: prior,
            cumulativeFyReceiptsInr: cumulative,
            taxableBase: 0,
            tcsRatePercent: 0,
            calculatedTcsInr: 0,
            isTcsApplicable: false,
            isSellerPrescribed: false,
            effectiveDate,
            governingSection: 'Income-tax Act, 1961 Section 206C(1H)',
            statutoryReason: `Seller turnover (${sellerTurnoverCr} Cr) in preceding financial year did not exceed ₹10 Crore threshold. Section 206C(1H) is inapplicable.`,
            statutoryDisposition: 'EXEMPT_SELLER_TURNOVER_BELOW_THRESHOLD'
        };
    }

    // 3. Section 194Q Absolute Statutory Precedence (Section 194Q(5)(b) read with Second Proviso to Section 206C(1H))
    const isBuyerLiable194Q = (buyerTurnoverCr !== null && Number(buyerTurnoverCr) > 10) || Boolean(isBuyerLiableAndDeducting194Q);
    if (isBuyerLiable194Q) {
        return {
            receiptAmountInr: receipt,
            priorFyReceiptsInr: prior,
            cumulativeFyReceiptsInr: cumulative,
            taxableBase: 0,
            tcsRatePercent: 0,
            calculatedTcsInr: 0,
            isTcsApplicable: false,
            isDisplacedBySection194Q: true,
            effectiveDate,
            governingSection: 'Income-tax Act, 1961 Section 194Q(5)(b) read with Section 206C(1H) Second Proviso',
            statutoryReason: 'Section 194Q(5)(b) and Second Proviso to Section 206C(1H) mandate that TCS under 206C(1H) shall not be collected if buyer is liable to deduct tax at source under Section 194Q and has deducted such amount. Section 194Q takes absolute statutory precedence.',
            statutoryDisposition: 'STATUTORILY_DISPLACED_BY_SECTION_194Q_TDS'
        };
    }

    // 4. Threshold & Taxable Base Calculation
    const incrementalExcess = Math.max(0, cumulative - THRESHOLD) - Math.max(0, prior - THRESHOLD);
    const taxableBase = Math.max(0, Math.min(receipt, incrementalExcess));

    // 5. Rate Determination (Section 206C(1H) read with Section 206CC Proviso)
    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    const tcsRatePercent = panAvailable ? 0.1 : 1.0; // Section 206CC proviso caps missing PAN at 1%, NOT 5% or 20%
    const calculatedTcsInr = Number(((taxableBase * tcsRatePercent) / 100).toFixed(2));

    return {
        receiptAmountInr: receipt,
        priorFyReceiptsInr: prior,
        cumulativeFyReceiptsInr: cumulative,
        statutoryThreshold: THRESHOLD,
        thresholdBreached: cumulative > THRESHOLD,
        taxableBase,
        isPanFurnished: panAvailable,
        tcsRatePercent,
        calculatedTcsInr,
        isTcsApplicable: taxableBase > 0,
        effectiveDate,
        governingSection: 'Income-tax Act, 1961 Section 206C(1H)',
        governingPenaltySection: panAvailable ? null : 'Income-tax Act, 1961 Section 206CC Proviso (Capped at 1%)',
        statutoryDisposition: taxableBase > 0 ? 'DETERMINED_TCS_COLLECTION_REQUIRED' : 'BELOW_THRESHOLD_RECEIPT',
        statutoryFormula: taxableBase > 0
            ? `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${tcsRatePercent}%${panAvailable ? '' : ' (Section 206CC Proviso Cap for 206C(1H))'} = ₹${calculatedTcsInr.toLocaleString('en-IN')}`
            : 'Consideration received does not exceed ₹50 Lakhs threshold in current FY.'
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 5. SECTION 194C: DETERMINISTIC CONTRACTOR & SUB-CONTRACTOR ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates payments to resident contractors and sub-contractors:
 * - Deductee entity classification:
 *   - Individual / HUF: 1.0%
 *   - Other entity (Company, LLP, Firm, AOP, BOI): 2.0%
 *   - Auto-inferred if PAN or GSTIN is supplied (4th character decoding).
 * - Dual statutory threshold under Section 194C(5):
 *   - Single invoice > ₹30,000 OR
 *   - Aggregate in Financial Year > ₹1,00,000
 *   - When aggregate crosses ₹1,00,000, catch-up withholding applies to untaxed prior base.
 * - Transporter Exemption under Section 194C(6):
 *   - Contractor in business of plying, hiring or leasing goods carriages who owns
 *     <= 10 goods carriages and furnishes declaration + PAN: 0.0% (NIL TDS).
 * - Section 206AA Penalty: 20.0% if PAN is absent or inoperative.
 * - Section 197 Lower Deduction Certificate with cumulative ceiling tracking.
 * - CBDT Circular No. 23/2017: Excludes separately stated GST.
 * - Income-tax Act, 2025 Transition: Section 393(1), Table 1, Sl. No. 6(i).
 */
export function evaluateSection194C({
    invoiceAmount,
    deducteeType = null, // 'INDIVIDUAL', 'HUF', 'COMPANY', 'FIRM', 'LLP', 'OTHER'
    pan = null,
    gstin = null,
    priorFyPayments = 0,
    priorFyPaymentsAlreadyTaxed = null,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    isTransporter = false,
    ownsTenOrLessCarriages = false,
    transporterDeclarationFurnished = false,
    gstAmountInr = 0,
    isAdvancePayment = false,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null
} = {}) {
    const rawInvoiceAmount = Number(invoiceAmount) || 0;
    const gstAmount = Number(gstAmountInr) || 0;

    // CBDT Circular No. 23/2017: If GST is separately stated, TDS applies to basic value
    const effectiveContractAmount = isAdvancePayment
        ? rawInvoiceAmount
        : Math.max(0, rawInvoiceAmount - gstAmount);

    const prior = Math.max(0, Number(priorFyPayments) || 0);
    const cumulative = prior + effectiveContractAmount;

    const SINGLE_THRESHOLD = 30000;
    const AGGREGATE_THRESHOLD = 100000;

    const panAvailable = Boolean(hasPan) && !isPanInoperative;

    // ── Transporter Exemption Check (Section 194C(6)) ──
    const isTransporterExempt = Boolean(
        isTransporter &&
        ownsTenOrLessCarriages &&
        transporterDeclarationFurnished &&
        panAvailable
    );

    // ── Threshold Evaluation (Section 194C(5)) ──
    const singleBreach = effectiveContractAmount > SINGLE_THRESHOLD;
    const aggregateBreach = cumulative > AGGREGATE_THRESHOLD;
    const isThresholdBreached = singleBreach || aggregateBreach;

    let taxableBase = 0;
    let thresholdDisposition = 'BELOW_THRESHOLD_EXEMPT';

    if (isTransporterExempt) {
        taxableBase = 0;
        thresholdDisposition = 'EXEMPT_TRANSPORTER_SECTION_194C_6';
    } else if (isThresholdBreached) {
        if (aggregateBreach) {
            // If aggregate crossed ₹1,00,000, determine previously taxed vs untaxed amount
            const previouslyTaxed = priorFyPaymentsAlreadyTaxed !== null && priorFyPaymentsAlreadyTaxed !== undefined
                ? Math.max(0, Number(priorFyPaymentsAlreadyTaxed))
                : (prior > AGGREGATE_THRESHOLD ? prior : 0);

            taxableBase = Math.max(0, cumulative - previouslyTaxed);
            thresholdDisposition = prior <= AGGREGATE_THRESHOLD && previouslyTaxed === 0
                ? 'AGGREGATE_THRESHOLD_BREACHED_WITH_CATCHUP'
                : 'AGGREGATE_THRESHOLD_ACTIVE';
        } else {
            // Single bill breached ₹30,000 while cumulative <= ₹1,00,000
            taxableBase = effectiveContractAmount;
            thresholdDisposition = 'SINGLE_BILL_THRESHOLD_BREACHED';
        }
    }

    // Temporal Resolution: "Credit or Payment, whichever is earlier"
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isTemporalStraddle = Boolean(
        creditDate && paymentDate &&
        ((creditDate < '2026-04-01' && paymentDate >= '2026-04-01') ||
         (paymentDate < '2026-04-01' && creditDate >= '2026-04-01'))
    );

    const temporalStraddlingAnalysis = isTemporalStraddle ? {
        isStraddlingCommencement: true,
        creditDate,
        paymentDate,
        crystallizedDate: effectiveDate,
        withholdingTrigger,
        governingRegime: effectiveDate >= '2026-04-01' ? 'ACT_2025' : 'ACT_1961',
        statutoryRationale: effectiveDate < '2026-04-01'
            ? `Tax withholding obligation crystallized on ${effectiveDate} (${withholdingTrigger}) under the Income-tax Act, 1961 prior to 2025 Act commencement. Subsequent settlement on ${effectiveDate === creditDate ? paymentDate : creditDate} creates no duplicate withholding under the 2025 Act.`
            : `Both credit and payment occurred on or after 1 April 2026. Governed exclusively by the Income-tax Act, 2025.`
    } : null;

    // ── Temporal Act Bridge ──
    const isAct2025 = effectiveDate ? effectiveDate >= '2026-04-01' : false;
    const governingSection = isAct2025
        ? "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 6(i)"
        : "Income-tax Act, 1961 Section 194C";
    
    // Resolve certificate framework with Section 536(2) transitional savings and validity evaluation
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    const governingCertificateSection = certResolution.governingCertificateSection;

    // ── Entity Rate Determination & Tax Identifier Decoding ──
    let decodedTaxId = null;
    const taxIdInput = gstin || pan;
    if (taxIdInput && typeof taxIdInput === 'string') {
        decodedTaxId = decodeTaxIdentifier(taxIdInput);
        if (decodedTaxId.isValid) {
            if (deducteeType === null || deducteeType === undefined) {
                deducteeType = decodedTaxId.isIndividualOrHuf ? 'INDIVIDUAL' : 'COMPANY';
            }
        }
    }
    const normalizedType = String(deducteeType || 'COMPANY').toUpperCase().trim();
    const isIndividualOrHuf = normalizedType === 'INDIVIDUAL' || normalizedType === 'HUF';
    const baseStatutoryRatePercent = isIndividualOrHuf ? 1.0 : 2.0;

    // Section 206AA: 20% if PAN missing/inoperative (no 5% goods cap for 194C)
    const standardRatePercent = panAvailable ? baseStatutoryRatePercent : 20.0;

    // ── Lower Deduction Certificate Validation ──
    const isCertificateValid = certResolution.isValid;
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && isCertificateValid;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    const certificateCeiling = certificateMonetaryCeilingInr !== null && certificateMonetaryCeilingInr !== undefined
        ? Number(certificateMonetaryCeilingInr)
        : (lowerDeductionCertificate?.monetaryCeilingInr !== undefined ? Number(lowerDeductionCertificate.monetaryCeilingInr) : null);

    const certificatePriorUtilized = Number(certificatePriorUtilizedInr) ||
        (lowerDeductionCertificate?.consumedAmountInr !== undefined ? Number(lowerDeductionCertificate.consumedAmountInr) : 0);

    const availableCeilingHeadroom = certificateCeiling !== null
        ? Math.max(0, certificateCeiling - certificatePriorUtilized)
        : null;

    let certifiedEligibleBase = 0;
    let uncoveredExcessBase = taxableBase;
    let ceilingExhaustionStatus = 'NO_LOWER_DEDUCTION_CERTIFICATE';

    if (hasLowerDeduction && taxableBase > 0) {
        if (certificateCeiling === null) {
            certifiedEligibleBase = taxableBase;
            uncoveredExcessBase = 0;
            ceilingExhaustionStatus = 'UNCONSTRAINED';
        } else if (availableCeilingHeadroom >= taxableBase) {
            certifiedEligibleBase = taxableBase;
            uncoveredExcessBase = 0;
            ceilingExhaustionStatus = 'WITHIN_CEILING';
        } else if (availableCeilingHeadroom > 0) {
            certifiedEligibleBase = availableCeilingHeadroom;
            uncoveredExcessBase = taxableBase - availableCeilingHeadroom;
            ceilingExhaustionStatus = 'PARTIALLY_EXHAUSTED';
        } else {
            certifiedEligibleBase = 0;
            uncoveredExcessBase = taxableBase;
            ceilingExhaustionStatus = 'FULLY_EXHAUSTED';
        }
    } else if (lowerDeductionRate !== null && !isCertificateValid) {
        ceilingExhaustionStatus = certResolution.status;
    }

    let certifiedTdsInr = 0;
    let uncoveredTdsInr = 0;

    if (taxableBase > 0) {
        certifiedTdsInr = Number(((certifiedEligibleBase * certifiedRatePercent) / 100).toFixed(2));
        uncoveredTdsInr = Number(((uncoveredExcessBase * standardRatePercent) / 100).toFixed(2));
    }
    const calculatedTdsInr = Number((certifiedTdsInr + uncoveredTdsInr).toFixed(2));
    const effectiveRatePercent = taxableBase > 0
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : (isTransporterExempt ? 0.0 : standardRatePercent);

    let formulaText = "";
    if (isTransporterExempt) {
        formulaText = "Section 194C(6) Nil withholding: Transporter furnished declaration of owning <= 10 goods carriages with valid PAN.";
    } else if (taxableBase === 0) {
        formulaText = `Section 194C inapplicable: Bill (₹${effectiveContractAmount.toLocaleString('en-IN')}) <= ₹30,000 single limit and cumulative (₹${cumulative.toLocaleString('en-IN')}) <= ₹1,00,000 aggregate limit.`;
    } else if (hasLowerDeduction) {
        if (ceilingExhaustionStatus === 'PARTIALLY_EXHAUSTED') {
            formulaText = `Partial Ceiling Split: ₹${certifiedEligibleBase.toLocaleString('en-IN')} @ ${certifiedRatePercent}% + ₹${uncoveredExcessBase.toLocaleString('en-IN')} @ ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
        } else {
            formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
        }
    } else if (lowerDeductionRate !== null && !isCertificateValid) {
        formulaText = `Lower Deduction Certificate Disallowed (${certResolution.status}): ${certResolution.reason}. Deducted at standard statutory rate of ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else if (panAvailable) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% (${isIndividualOrHuf ? 'Individual/HUF' : 'Company/LLP/Firm'}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 20% (Section 206AA Penal Rate for missing PAN) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    }

    return {
        section: '194C',
        governingSection,
        governingCertificateSection,
        isSection536TransitionalSavingsApplied: certResolution.isSection536TransitionalSavingsApplied,
        transitionalSavingsClause: certResolution.transitionalSavingsClause,
        certificateValidity: certResolution,
        certificateDisallowanceReason: certResolution.isValid ? null : certResolution.reason,
        creditDate,
        paymentDate,
        effectiveWithholdingDate: effectiveDate,
        withholdingTrigger,
        temporalStraddlingAnalysis,
        rawInvoiceAmount,
        gstAmountInr: gstAmount,
        isAdvancePayment,
        effectiveContractAmount,
        cbdtCircular23Applied: !isAdvancePayment && gstAmount > 0,
        priorFyPayments: prior,
        cumulativeFyPayments: cumulative,
        singleThreshold: SINGLE_THRESHOLD,
        aggregateThreshold: AGGREGATE_THRESHOLD,
        singleBreach,
        aggregateBreach,
        isThresholdBreached,
        thresholdDisposition,
        deducteeType: normalizedType,
        isIndividualOrHuf,
        decodedTaxIdentifier: decodedTaxId,
        isTransporter,
        isTransporterExempt,
        ownsTenOrLessCarriages,
        transporterDeclarationFurnished,
        isPanFurnished: panAvailable,
        taxableBase,
        baseStatutoryRatePercent,
        standardRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        certifiedTdsInr,
        uncoveredTdsInr,
        certificateCeiling,
        certificatePriorUtilized,
        availableCeilingHeadroom,
        certifiedEligibleBase,
        uncoveredExcessBase,
        ceilingExhaustionStatus,
        lowerDeductionRate: hasLowerDeduction ? Number(lowerDeductionRate) : null,
        statutoryFormula: formulaText
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 6. SECTION 194J: DETERMINISTIC PROFESSIONAL & TECHNICAL SERVICES ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates payments for professional, technical services, royalty, and director fees:
 * - Service Category Classification:
 *   - Fees for Technical Services (FTS) / BPO / Royalty for cinematographic films: 2.0%
 *   - Royalty (general, non-film): 2.0%
 *   - Professional Services (Legal, Medical, Engineering, Accountancy, etc.): 10.0%
 *   - Non-compete fees (Section 28(va)): 10.0%
 *   - Director Remuneration (other than salary covered by Sec 192): 10.0% (Zero threshold!)
 * - Statutory Threshold under Section 194J(1):
 *   - ₹30,000 aggregate per category in FY.
 *   - Director remuneration: ₹0 threshold (first rupee deductible).
 *   - When cumulative crosses ₹30,000, catch-up withholding applies to untaxed prior base.
 * - Section 206AA Penalty: 20.0% if PAN is missing or inoperative.
 * - Section 197 Lower Deduction Certificate with cumulative ceiling tracking.
 * - CBDT Circular No. 23/2017: Excludes separately stated GST.
 * - Income-tax Act, 2025 Transition: Section 393(1), Table 1, Sl. No. 6(iii).
 */
export const VALID_194J_CATEGORIES = [
    'PROFESSIONAL_SERVICES',
    'TECHNICAL_SERVICES',
    'DIRECTOR_REMUNERATION',
    'ROYALTY_GENERAL',
    'ROYALTY_FILM',
    'NON_COMPETE'
];

export function normalize194JCategory(rawCategory) {
    if (!rawCategory || typeof rawCategory !== 'string') return null;
    const clean = rawCategory.trim().toUpperCase().replace(/[-\s]+/g, '_');
    
    if (VALID_194J_CATEGORIES.includes(clean)) {
        return clean;
    }
    
    // Synonyms for PROFESSIONAL_SERVICES (10%)
    if (/^(PROFESSIONAL|PROF|CA|LEGAL|LAWYER|ADVOCATE|CONSULTING|CONSULTANT|AUDIT|AUDITOR|MEDICAL|DOCTOR|ARCHITECT)$/i.test(clean) ||
        clean.includes('PROFESSIONAL') || clean.includes('LEGAL_') || clean.includes('AUDIT_')) {
        return 'PROFESSIONAL_SERVICES';
    }

    // Synonyms for TECHNICAL_SERVICES (2%)
    if (/^(TECHNICAL|FTS|FEES_FOR_TECHNICAL_SERVICES|IT|SOFTWARE|HOSTING|TECH|DEVELOPMENT|ENGINEERING|CLOUD)$/i.test(clean) ||
        clean.includes('TECHNICAL') || clean.includes('FTS')) {
        return 'TECHNICAL_SERVICES';
    }

    // Synonyms for DIRECTOR_REMUNERATION (10%, ₹0 threshold)
    if (/^(DIRECTOR|DIRECTOR_FEE|DIRECTOR_FEES|SITTING_FEE|SITTING_FEES|BOARD_MEETING_FEE)$/i.test(clean) ||
        clean.includes('DIRECTOR') || clean.includes('SITTING_FEE')) {
        return 'DIRECTOR_REMUNERATION';
    }

    // Synonyms for ROYALTY_GENERAL (2%)
    if (/^(ROYALTY|PATENT|COPYRIGHT|TRADEMARK|IP_LICENSING)$/i.test(clean) ||
        clean === 'ROYALTY_GENERAL') {
        return 'ROYALTY_GENERAL';
    }

    // Synonyms for ROYALTY_FILM (2%)
    if (/^(ROYALTY_FILM|FILM_ROYALTY|CINEMATOGRAPHIC_FILM)$/i.test(clean)) {
        return 'ROYALTY_FILM';
    }

    // Synonyms for NON_COMPETE (10%)
    if (/^(NON_COMPETE|NONCOMPETE|NON_COMPETE_FEE|NON_COMPETE_FEES)$/i.test(clean)) {
        return 'NON_COMPETE';
    }

    return null;
}

export function evaluateSection194J({
    invoiceAmount,
    serviceCategory = 'TECHNICAL_SERVICES', // 'TECHNICAL_SERVICES', 'PROFESSIONAL_SERVICES', 'ROYALTY_FILM', 'ROYALTY_GENERAL', 'DIRECTOR_REMUNERATION', 'NON_COMPETE'
    priorFyCategoryPayments = 0,
    priorPaymentsInCurrentFyInr = null,
    priorFyCategoryPaymentsAlreadyTaxed = null,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    gstAmountInr = 0,
    isAdvancePayment = false,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null
} = {}) {
    const rawInvoiceAmount = Number(invoiceAmount) || 0;
    const gstAmount = Number(gstAmountInr) || 0;

    const normalizedCategory = normalize194JCategory(serviceCategory);
    if (!normalizedCategory) {
        throw new Error(`INVALID_SERVICE_CATEGORY: Unknown serviceCategory "${serviceCategory}". Allowed categories: ${VALID_194J_CATEGORIES.join(', ')}`);
    }

    // CBDT Circular No. 23/2017: If GST is separately stated, TDS applies to basic value
    const effectiveServiceAmount = isAdvancePayment
        ? rawInvoiceAmount
        : Math.max(0, rawInvoiceAmount - gstAmount);

    const effectivePriorInput = priorPaymentsInCurrentFyInr !== null && priorPaymentsInCurrentFyInr !== undefined
        ? priorPaymentsInCurrentFyInr
        : priorFyCategoryPayments;
    const prior = Math.max(0, Number(effectivePriorInput) || 0);
    const cumulative = prior + effectiveServiceAmount;

    const isDirectorFee = normalizedCategory === 'DIRECTOR_REMUNERATION';
    const isProfessionalOrNonCompete = normalizedCategory === 'PROFESSIONAL_SERVICES' || normalizedCategory === 'NON_COMPETE';

    // Temporal Resolution: "Credit or Payment, whichever is earlier"
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isTemporalStraddle = Boolean(
        creditDate && paymentDate &&
        ((creditDate < '2026-04-01' && paymentDate >= '2026-04-01') ||
         (paymentDate < '2026-04-01' && creditDate >= '2026-04-01'))
    );

    const temporalStraddlingAnalysis = isTemporalStraddle ? {
        isStraddlingCommencement: true,
        creditDate,
        paymentDate,
        crystallizedDate: effectiveDate,
        withholdingTrigger,
        governingRegime: effectiveDate >= '2026-04-01' ? 'ACT_2025' : 'ACT_1961',
        statutoryRationale: effectiveDate < '2026-04-01'
            ? `Tax withholding obligation crystallized on ${effectiveDate} (${withholdingTrigger}) under the Income-tax Act, 1961 prior to 2025 Act commencement. Subsequent settlement on ${effectiveDate === creditDate ? paymentDate : creditDate} creates no duplicate withholding under the 2025 Act.`
            : `Both credit and payment occurred on or after 1 April 2026. Governed exclusively by the Income-tax Act, 2025.`
    } : null;

    // ── Temporal Act Bridge ──
    const isAct2025 = Boolean(effectiveDate && String(effectiveDate).slice(0, 10) >= '2026-04-01');
    const governingSection = isAct2025
        ? "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 6(iii)"
        : "Income-tax Act, 1961 Section 194J";

    // ── Rate Determination ──
    // 2% for FTS, BPO, Royalty; 10% for Professional Services, Non-compete, and Director fees
    const baseStatutoryRatePercent = isProfessionalOrNonCompete || isDirectorFee ? 10.0 : 2.0;

    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    const panFloorResolution = resolveMissingPanPenaltyFloor({
        sectionCode: '194J',
        tableItem: isAct2025 ? '393_7' : null,
        isAct2025
    });
    // Section 206AA(1)(iii) / Section 397(2)(a): Strictly 20% floor for professional & technical services
    const standardRatePercent = panAvailable ? baseStatutoryRatePercent : Math.max(baseStatutoryRatePercent, panFloorResolution.penaltyFloorPercent);

    // ── Threshold Evaluation ──
    // Director remuneration has NO threshold under 3rd proviso to Section 194J(1) / Section 393(1) Table 1 Sl. No. 6(iii)
    // 1961 Act threshold: ₹30,000; 2025 Act threshold: ₹50,000 (Section 393(1), Table 1, Sl. No. 6(iii))
    const STATUTORY_THRESHOLD = isAct2025 ? 50000 : 30000;
    const CATEGORY_THRESHOLD = isDirectorFee ? 0 : STATUTORY_THRESHOLD;
    const isThresholdBreached = isDirectorFee || cumulative > CATEGORY_THRESHOLD;

    let taxableBase = 0;
    let thresholdDisposition = 'BELOW_THRESHOLD_EXEMPT';

    if (isThresholdBreached) {
        if (isDirectorFee) {
            taxableBase = effectiveServiceAmount;
            thresholdDisposition = 'DIRECTOR_FEES_ZERO_THRESHOLD';
        } else {
            const previouslyTaxed = priorFyCategoryPaymentsAlreadyTaxed !== null && priorFyCategoryPaymentsAlreadyTaxed !== undefined
                ? Math.max(0, Number(priorFyCategoryPaymentsAlreadyTaxed))
                : (prior > CATEGORY_THRESHOLD ? prior : 0);

            taxableBase = Math.max(0, cumulative - previouslyTaxed);
            thresholdDisposition = prior <= CATEGORY_THRESHOLD && previouslyTaxed === 0
                ? 'CATEGORY_THRESHOLD_BREACHED_WITH_CATCHUP'
                : 'CATEGORY_THRESHOLD_ACTIVE';
        }
    }
    
    // Resolve certificate framework with Section 536(2) transitional savings and validity evaluation
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    const governingCertificateSection = certResolution.governingCertificateSection;

    // ── Lower Deduction Certificate Validation ──
    const isCertificateValid = certResolution.isValid;
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && isCertificateValid;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    const certificateCeiling = certificateMonetaryCeilingInr !== null && certificateMonetaryCeilingInr !== undefined
        ? Number(certificateMonetaryCeilingInr)
        : (lowerDeductionCertificate?.monetaryCeilingInr !== undefined ? Number(lowerDeductionCertificate.monetaryCeilingInr) : null);

    const certificatePriorUtilized = Number(certificatePriorUtilizedInr) ||
        (lowerDeductionCertificate?.consumedAmountInr !== undefined ? Number(lowerDeductionCertificate.consumedAmountInr) : 0);

    const availableCeilingHeadroom = certificateCeiling !== null
        ? Math.max(0, certificateCeiling - certificatePriorUtilized)
        : null;

    let certifiedEligibleBase = 0;
    let uncoveredExcessBase = taxableBase;
    let ceilingExhaustionStatus = 'NO_LOWER_DEDUCTION_CERTIFICATE';

    if (hasLowerDeduction && taxableBase > 0) {
        if (certificateCeiling === null) {
            certifiedEligibleBase = taxableBase;
            uncoveredExcessBase = 0;
            ceilingExhaustionStatus = 'UNCONSTRAINED';
        } else if (availableCeilingHeadroom >= taxableBase) {
            certifiedEligibleBase = taxableBase;
            uncoveredExcessBase = 0;
            ceilingExhaustionStatus = 'WITHIN_CEILING';
        } else if (availableCeilingHeadroom > 0) {
            certifiedEligibleBase = availableCeilingHeadroom;
            uncoveredExcessBase = taxableBase - availableCeilingHeadroom;
            ceilingExhaustionStatus = 'PARTIALLY_EXHAUSTED';
        } else {
            certifiedEligibleBase = 0;
            uncoveredExcessBase = taxableBase;
            ceilingExhaustionStatus = 'FULLY_EXHAUSTED';
        }
    } else if (lowerDeductionRate !== null && !isCertificateValid) {
        ceilingExhaustionStatus = certResolution.status;
    }

    let certifiedTdsInr = 0;
    let uncoveredTdsInr = 0;

    if (taxableBase > 0) {
        certifiedTdsInr = Number(((certifiedEligibleBase * certifiedRatePercent) / 100).toFixed(2));
        uncoveredTdsInr = Number(((uncoveredExcessBase * standardRatePercent) / 100).toFixed(2));
    }
    const calculatedTdsInr = Number((certifiedTdsInr + uncoveredTdsInr).toFixed(2));
    const effectiveRatePercent = taxableBase > 0
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : standardRatePercent;

    let formulaText = "";
    if (taxableBase === 0) {
        formulaText = `Section 194J inapplicable: Cumulative fees (₹${cumulative.toLocaleString('en-IN')}) do not exceed ₹30,000 threshold for ${normalizedCategory}.`;
    } else if (hasLowerDeduction) {
        if (ceilingExhaustionStatus === 'PARTIALLY_EXHAUSTED') {
            formulaText = `Partial Ceiling Split: ₹${certifiedEligibleBase.toLocaleString('en-IN')} @ ${certifiedRatePercent}% + ₹${uncoveredExcessBase.toLocaleString('en-IN')} @ ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
        } else {
            formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
        }
    } else if (lowerDeductionRate !== null && !isCertificateValid) {
        formulaText = `Lower Deduction Certificate Disallowed (${certResolution.status}): ${certResolution.reason}. Deducted at standard statutory rate of ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else if (panAvailable) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% (${normalizedCategory}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 20% (${panFloorResolution.governingProviso}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    }

    return {
        section: '194J',
        isAct2025,
        governingSection,
        governingPanPenaltySection: panFloorResolution.governingProviso,
        governingCertificateSection,
        isSection536TransitionalSavingsApplied: certResolution.isSection536TransitionalSavingsApplied,
        transitionalSavingsClause: certResolution.transitionalSavingsClause,
        certificateValidity: certResolution,
        certificateDisallowanceReason: certResolution.isValid ? null : certResolution.reason,
        creditDate,
        paymentDate,
        effectiveWithholdingDate: effectiveDate,
        withholdingTrigger,
        temporalStraddlingAnalysis,
        rawInvoiceAmount,
        gstAmountInr: gstAmount,
        isAdvancePayment,
        effectiveServiceAmount,
        cbdtCircular23Applied: !isAdvancePayment && gstAmount > 0,
        serviceCategory: normalizedCategory,
        priorFyCategoryPayments: prior,
        priorPaymentsInCurrentFyInr: prior,
        cumulativeFyCategoryPayments: cumulative,
        cumulativePaymentsInCurrentFyInr: cumulative,
        categoryThreshold: CATEGORY_THRESHOLD,
        isDirectorFee,
        isThresholdBreached,
        thresholdDisposition,
        isPanFurnished: panAvailable,
        taxableBase,
        baseStatutoryRatePercent,
        standardRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        certifiedTdsInr,
        uncoveredTdsInr,
        certificateCeiling,
        certificatePriorUtilized,
        availableCeilingHeadroom,
        certifiedEligibleBase,
        uncoveredExcessBase,
        ceilingExhaustionStatus,
        lowerDeductionRate: hasLowerDeduction ? Number(lowerDeductionRate) : null,
        statutoryFormula: formulaText,
        classificationRiskAdvisory: normalizedCategory === 'TECHNICAL_SERVICES' ? {
            statutoryRatePercent: 2.0,
            reclassificationRiskRatePercent: 10.0,
            potentialExposure: "If Assessing Officer reclassifies ERP advisory/technical consultancy as professional services under Section 194J(1)(a), differential 8% withholding plus Section 201(1A) interest (1% per month) applies.",
            recommendation: "Review contract scope of work: pure technical/IT services qualify for 2% under Section 194J(1)(II); managerial/professional consulting attracts 10% under Section 194J(1)(a)."
        } : null,
        statutoryRectificationProcedure: {
            excessDeductionRemedy: "Deductor cannot claim a cash refund from the vendor for excess TDS deducted. The deductor must file a correction statement (Revised Form 26Q) via TRACES to adjust challan mapping, or the deductee/vendor claims credit/refund in their annual Income Tax Return (ITR) based on Form 26AS/AIS.",
            belatedCertificateRemedy: "If Section 197 certificate is authenticated post-disbursement, adjust against subsequent payments in the same financial year or advise vendor to claim refund in their ITR."
        }
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 7. SECTION 194I: DETERMINISTIC TDS ON RENT ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates payments for lease or rental of machinery, equipment, land, or buildings:
 * - Rates:
 *   - Plant, Machinery, or Equipment: 2.0% (Section 194I(a))
 *   - Land, Building, Furniture, or Fittings: 10.0% (Section 194I(b))
 * - Annual Threshold: ₹2,40,000 per financial year per payee.
 * - Missing PAN: 20.0% (Section 206AA(1)(iii)).
 * - CBDT Circular No. 23/2017: Separately stated GST excluded from TDS.
 * - Income-tax Act, 2025 Transition: Section 393(1), Table 1, Sl. No. 2.
 */
export function evaluateSection194I({
    invoiceAmount,
    monthlyRent = null,
    rentPeriodMonths = null,
    priorFyRent = 0,
    priorFyPayments = null,
    priorFyPaymentsAlreadyTaxed = null,
    rentType = 'LAND_BUILDING', // 'LAND_BUILDING', 'PLANT_MACHINERY'
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    gstAmountInr = 0,
    isAdvancePayment = false,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null
} = {}) {
    const rawInvoiceAmount = Number(invoiceAmount) || 0;
    const gstAmount = Number(gstAmountInr) || 0;

    const effectiveRentAmount = isAdvancePayment
        ? rawInvoiceAmount
        : Math.max(0, rawInvoiceAmount - gstAmount);

    const effectivePrior = priorFyPayments !== null && priorFyPayments !== undefined
        ? Number(priorFyPayments) || 0
        : Number(priorFyRent) || 0;
    const prior = Math.max(0, effectivePrior);
    const cumulative = prior + effectiveRentAmount;

    // Normalizing rentType:
    const cleanType = String(rentType || '').trim().toUpperCase().replace(/[-\s]+/g, '_');
    const isPlantOrMachinery = /^(PLANT|MACHINERY|EQUIPMENT|PLANT_MACHINERY|PLANT_AND_MACHINERY)$/i.test(cleanType) || cleanType.includes('MACHINERY') || cleanType.includes('EQUIPMENT');
    const normalizedRentType = isPlantOrMachinery ? 'PLANT_MACHINERY' : 'LAND_BUILDING';
    const baseStatutoryRatePercent = isPlantOrMachinery ? 2.0 : 10.0;

    // Temporal Resolution: Credit or Payment, whichever is earlier
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isAct2025 = Boolean(effectiveDate && String(effectiveDate).slice(0, 10) >= '2026-04-01');
    const governingSection = isAct2025
        ? "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 2"
        : "Income-tax Act, 1961 Section 194I";

    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    const panFloorResolution = resolveMissingPanPenaltyFloor({
        sectionCode: '194I',
        tableItem: isAct2025 ? '393_5' : null,
        isAct2025
    });
    const standardRatePercent = panAvailable ? baseStatutoryRatePercent : Math.max(baseStatutoryRatePercent, panFloorResolution.penaltyFloorPercent);

    // Threshold: Under 1961 Act: ₹2,40,000 per financial year; Under 2025 Act: ₹50,000 per month or part thereof (Section 393(1), Table 1, Sl. No. 2)
    const effectiveRentMonths = (rentPeriodMonths !== null && rentPeriodMonths !== undefined && Number(rentPeriodMonths) > 0)
        ? Number(rentPeriodMonths)
        : (isAct2025 && !priorFyPayments && !priorFyRent ? 1 : null);
    let RENT_THRESHOLD;
    if (isAct2025) {
        RENT_THRESHOLD = effectiveRentMonths ? (50000 * effectiveRentMonths) : 50000;
    } else {
        RENT_THRESHOLD = 240000;
    }
    const isThresholdBreached = cumulative > RENT_THRESHOLD;

    let taxableBase = 0;
    let thresholdDisposition = 'BELOW_THRESHOLD_EXEMPT';

    if (isThresholdBreached) {
        const previouslyTaxed = priorFyPaymentsAlreadyTaxed !== null && priorFyPaymentsAlreadyTaxed !== undefined
            ? Math.max(0, Number(priorFyPaymentsAlreadyTaxed))
            : (prior > RENT_THRESHOLD ? prior : 0);

        taxableBase = Math.max(0, cumulative - previouslyTaxed);
        thresholdDisposition = prior <= RENT_THRESHOLD && previouslyTaxed === 0
            ? 'CATEGORY_THRESHOLD_BREACHED_WITH_CATCHUP'
            : 'CATEGORY_THRESHOLD_ACTIVE';
    }

    // Lower Deduction Certificate resolution
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && certResolution.isValid;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    let calculatedTdsInr = 0;
    if (taxableBase > 0) {
        calculatedTdsInr = Number(((taxableBase * certifiedRatePercent) / 100).toFixed(2));
    }
    const effectiveRatePercent = taxableBase > 0
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : standardRatePercent;

    let formulaText = "";
    if (taxableBase === 0) {
        formulaText = `Section 194I inapplicable: Cumulative rent (₹${cumulative.toLocaleString('en-IN')}) does not exceed ₹2,40,000 annual threshold.`;
    } else if (hasLowerDeduction) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else if (panAvailable) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% (${normalizedRentType}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 20% (${panFloorResolution.governingProviso}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    }

    return {
        section: isAct2025 ? '393' : '194I',
        isAct2025,
        governingSection,
        governingPanPenaltySection: panFloorResolution.governingProviso,
        rawInvoiceAmount,
        gstAmountInr: gstAmount,
        effectiveRentAmount,
        cbdtCircular23Applied: !isAdvancePayment && gstAmount > 0,
        rentType: normalizedRentType,
        priorFyRent: prior,
        cumulativeFyRent: cumulative,
        rentThreshold: RENT_THRESHOLD,
        isThresholdBreached,
        thresholdDisposition,
        isPanFurnished: panAvailable,
        taxableBase,
        baseStatutoryRatePercent,
        standardRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        taxAmountRoundedSection288B: roundSection288B(calculatedTdsInr),
        netPayable: roundToPaisa(Math.max(0, rawInvoiceAmount - calculatedTdsInr)),
        statutoryFormula: formulaText
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 8. SECTION 194H: DETERMINISTIC TDS ON COMMISSION & BROKERAGE ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates payments for commission (other than insurance commission) or brokerage:
 * - Rates:
 *   - Prior to 1 October 2024: 5.0%
 *   - On or after 1 October 2024: 2.0% (Amended by Finance (No. 2) Act, 2024)
 * - Annual Threshold: ₹15,000 per financial year per payee.
 * - Missing PAN: 20.0% (Section 206AA(1)(iii)).
 * - CBDT Circular No. 23/2017: Separately stated GST excluded from TDS.
 * - Income-tax Act, 2025 Transition: Section 393(1), Table 1, Sl. No. 1(ii).
 */
export function evaluateSection194H({
    invoiceAmount,
    priorFyCommission = 0,
    priorFyPayments = null,
    priorFyPaymentsAlreadyTaxed = null,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    gstAmountInr = 0,
    isAdvancePayment = false,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null
} = {}) {
    const rawInvoiceAmount = Number(invoiceAmount) || 0;
    const gstAmount = Number(gstAmountInr) || 0;

    const effectiveCommissionAmount = isAdvancePayment
        ? rawInvoiceAmount
        : Math.max(0, rawInvoiceAmount - gstAmount);

    const effectivePrior = priorFyPayments !== null && priorFyPayments !== undefined
        ? Number(priorFyPayments) || 0
        : Number(priorFyCommission) || 0;
    const prior = Math.max(0, effectivePrior);
    const cumulative = prior + effectiveCommissionAmount;

    // Temporal Resolution: Credit or Payment, whichever is earlier
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const dateStr = effectiveDate ? String(effectiveDate).slice(0, 10) : new Date().toISOString().split('T')[0];
    const isAct2025 = Boolean(dateStr >= '2026-04-01');
    const isPostOctober2024 = dateStr >= '2024-10-01';

    // Base statutory rate: 2% from 1 Oct 2024 onwards; 5% prior
    const baseStatutoryRatePercent = isPostOctober2024 ? 2.0 : 5.0;

    const governingSection = isAct2025
        ? "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 1(ii)"
        : "Income-tax Act, 1961 Section 194H";

    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    const panFloorResolution = resolveMissingPanPenaltyFloor({
        sectionCode: '194H',
        tableItem: isAct2025 ? '393_4' : null,
        isAct2025
    });
    const standardRatePercent = panAvailable ? baseStatutoryRatePercent : Math.max(baseStatutoryRatePercent, panFloorResolution.penaltyFloorPercent);

    // Threshold: ₹15,000 under 1961 Act; ₹20,000 under 2025 Act Section 393(1), Table 1, Sl. No. 1(ii)
    const COMMISSION_THRESHOLD = isAct2025 ? 20000 : 15000;
    const isThresholdBreached = cumulative > COMMISSION_THRESHOLD;

    let taxableBase = 0;
    let thresholdDisposition = 'BELOW_THRESHOLD_EXEMPT';

    if (isThresholdBreached) {
        const previouslyTaxed = priorFyPaymentsAlreadyTaxed !== null && priorFyPaymentsAlreadyTaxed !== undefined
            ? Math.max(0, Number(priorFyPaymentsAlreadyTaxed))
            : (prior > COMMISSION_THRESHOLD ? prior : 0);

        taxableBase = Math.max(0, cumulative - previouslyTaxed);
        thresholdDisposition = prior <= COMMISSION_THRESHOLD && previouslyTaxed === 0
            ? 'CATEGORY_THRESHOLD_BREACHED_WITH_CATCHUP'
            : 'CATEGORY_THRESHOLD_ACTIVE';
    }

    // Lower Deduction Certificate resolution
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && certResolution.isValid;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    let calculatedTdsInr = 0;
    if (taxableBase > 0) {
        calculatedTdsInr = Number(((taxableBase * certifiedRatePercent) / 100).toFixed(2));
    }
    const effectiveRatePercent = taxableBase > 0
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : standardRatePercent;

    let formulaText = "";
    if (taxableBase === 0) {
        formulaText = `Section 194H inapplicable: Cumulative commission (₹${cumulative.toLocaleString('en-IN')}) does not exceed ₹15,000 annual threshold.`;
    } else if (hasLowerDeduction) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else if (panAvailable) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% (${isPostOctober2024 ? 'Finance (No. 2) Act 2024 rate 2%' : 'Pre-amendment rate 5%'}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 20% (${panFloorResolution.governingProviso}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    }

    return {
        section: isAct2025 ? '393' : '194H',
        isAct2025,
        governingSection,
        governingPanPenaltySection: panFloorResolution.governingProviso,
        rawInvoiceAmount,
        gstAmountInr: gstAmount,
        effectiveCommissionAmount,
        cbdtCircular23Applied: !isAdvancePayment && gstAmount > 0,
        priorFyCommission: prior,
        cumulativeFyCommission: cumulative,
        commissionThreshold: COMMISSION_THRESHOLD,
        isThresholdBreached,
        thresholdDisposition,
        isPanFurnished: panAvailable,
        taxableBase,
        baseStatutoryRatePercent,
        standardRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        taxAmountRoundedSection288B: roundSection288B(calculatedTdsInr),
        netPayable: roundToPaisa(Math.max(0, rawInvoiceAmount - calculatedTdsInr)),
        statutoryFormula: formulaText
    };
}

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CORPORATE FINANCE: ADVANCE TAX & INTEREST CALCULATOR (DUAL-ACT ENGINE)
 * Income-tax Act, 1961: Sections 208, 209, 211, 234B, 234C
 * Income-tax Act, 2025: Sections 405 (Computation & Threshold), 408 (Installments)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements authoritative corporate advance tax installment compliance,
 * 12% & 36% safe harbors for Q1 and Q2, Section 234C sudden income provisos
 * (capital gains, dividends, new business), and Section 234B default interest.
 */
export function calculateAdvanceTaxAndInterest({
    taxYear = '2025-26',
    assessmentYear = null,
    estimatedAssessedTaxInr,
    tdsTcsCreditInr = 0,
    paidQ1Inr = 0,
    paidQ2Inr = 0,
    paidQ3Inr = 0,
    paidQ4Inr = 0,
    suddenIncomeExceptions = null,
    assessmentDate = null,
    isAct2025 = null
}) {
    if (estimatedAssessedTaxInr === undefined || estimatedAssessedTaxInr === null) {
        return {
            status: 'INDETERMINATE',
            code: 'MISSING_ESTIMATED_TAX',
            statutoryDetermination: 'INDETERMINATE',
            message: 'Field "estimatedAssessedTaxInr" is required to determine advance tax liability.',
            isAdvanceTaxPayable: null,
            totalInterest: null,
            isAuthorizedToDisburse: false
        };
    }

    const estimatedTax = Number(estimatedAssessedTaxInr);
    const credits = Number(tdsTcsCreditInr || 0);
    if (isNaN(estimatedTax) || estimatedTax < 0 || !Number.isFinite(estimatedTax)) {
        throw new Error('[@mudraforge/statutory] Invalid estimatedAssessedTaxInr: expected non-negative finite number.');
    }
    if (isNaN(credits) || credits < 0 || !Number.isFinite(credits)) {
        throw new Error('[@mudraforge/statutory] Invalid tdsTcsCreditInr: expected non-negative finite number.');
    }

    // Regime resolution: Income-tax Act, 1961 vs Income-tax Act, 2025
    const resolvedIsAct2025 = isAct2025 !== null
        ? Boolean(isAct2025)
        : (taxYear >= '2026-27' || (assessmentYear && assessmentYear >= '2027-28'));

    const governingAct = resolvedIsAct2025 ? 'Income-tax Act, 2025' : 'Income-tax Act, 1961';
    const governingComputationSection = resolvedIsAct2025 ? 'Section 405' : 'Section 208 read with Section 209';
    const governingInstallmentSection = resolvedIsAct2025 ? 'Section 408' : 'Section 211';
    const governingInterestSection = resolvedIsAct2025 ? 'Sections 410 & 411' : 'Sections 234B & 234C';

    const netTaxPayable = roundToPaisa(Math.max(0, estimatedTax - credits));
    const ADVANCE_TAX_THRESHOLD = 10000; // ₹10,000 threshold under Section 208 / Section 405

    // Zero-liability below threshold
    if (netTaxPayable < ADVANCE_TAX_THRESHOLD) {
        return {
            status: 'EXEMPT_BELOW_THRESHOLD',
            statutoryDetermination: 'EXEMPT',
            governingAct,
            governingComputationSection,
            governingInstallmentSection,
            governingInterestSection,
            taxYear,
            assessmentYear: assessmentYear || (taxYear ? `${Number(taxYear.slice(0, 4)) + 1}-${Number(taxYear.slice(5, 7)) + 1}` : null),
            estimatedAssessedTaxInr: estimatedTax,
            tdsTcsCreditInr: credits,
            netTaxPayableInr: netTaxPayable,
            advanceTaxThresholdInr: ADVANCE_TAX_THRESHOLD,
            isAdvanceTaxPayable: false,
            installments: [],
            totalShortfallQ1toQ4: 0,
            interestSection234CInr: 0,
            interestSection234BInr: 0,
            totalInterestInr: 0,
            totalPayableInr: 0,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Assessed tax after TDS/TCS credits (₹${netTaxPayable.toLocaleString('en-IN')}) is below the ₹10,000 threshold under ${governingComputationSection}. No advance tax obligation arises.`
        };
    }

    // Cumulative installments calculation (15%, 45%, 75%, 100%)
    const pQ1 = Math.max(0, Number(paidQ1Inr || 0));
    const pQ2 = Math.max(0, Number(paidQ2Inr || 0));
    const pQ3 = Math.max(0, Number(paidQ3Inr || 0));
    const pQ4 = Math.max(0, Number(paidQ4Inr || 0));

    const cumPaidQ1 = pQ1;
    const cumPaidQ2 = cumPaidQ1 + pQ2;
    const cumPaidQ3 = cumPaidQ2 + pQ3;
    const cumPaidQ4 = cumPaidQ3 + pQ4;

    const reqQ1 = roundToPaisa(netTaxPayable * 0.15);
    const reqQ2 = roundToPaisa(netTaxPayable * 0.45);
    const reqQ3 = roundToPaisa(netTaxPayable * 0.75);
    const reqQ4 = roundToPaisa(netTaxPayable * 1.00);

    // Section 234C Safe Harbors (First Proviso to Section 234C(1)):
    // Q1: If paid on or before 15 June is not less than 12%, no 234C interest.
    // Q2: If paid on or before 15 September is not less than 36%, no 234C interest.
    const safeHarborQ1Threshold = roundToPaisa(netTaxPayable * 0.12);
    const safeHarborQ2Threshold = roundToPaisa(netTaxPayable * 0.36);

    const safeHarborQ1Applied = cumPaidQ1 >= safeHarborQ1Threshold;
    const safeHarborQ2Applied = cumPaidQ2 >= safeHarborQ2Threshold;

    // Sudden income exception relief under First Proviso to Section 234C(1):
    // Capital gains, casual income, new business, domestic dividends
    const suddenReliefTax = (suddenIncomeExceptions && typeof suddenIncomeExceptions === 'object')
        ? Math.max(0, Number(
            (suddenIncomeExceptions.capitalGainsTaxInr || 0) +
            (suddenIncomeExceptions.dividendTaxInr || 0) +
            (suddenIncomeExceptions.newBusinessTaxInr || 0) +
            (suddenIncomeExceptions.casualIncomeTaxInr || 0)
        ))
        : 0;

    // Q1 Interest: 1% per month for 3 months
    let shortfallQ1 = 0;
    let interestQ1 = 0;
    if (!safeHarborQ1Applied && cumPaidQ1 < reqQ1) {
        shortfallQ1 = Math.max(0, reqQ1 - cumPaidQ1);
        if (suddenReliefTax > 0 && suddenIncomeExceptions?.occurrenceQuarter > 1) {
            // Sudden income occurred in subsequent quarter; relief applied to earlier installment
            shortfallQ1 = Math.max(0, shortfallQ1 - roundToPaisa(suddenReliefTax * 0.15));
        }
        interestQ1 = roundSection288B(shortfallQ1) * 0.01 * 3;
    }

    // Q2 Interest: 1% per month for 3 months
    let shortfallQ2 = 0;
    let interestQ2 = 0;
    if (!safeHarborQ2Applied && cumPaidQ2 < reqQ2) {
        shortfallQ2 = Math.max(0, reqQ2 - cumPaidQ2);
        if (suddenReliefTax > 0 && suddenIncomeExceptions?.occurrenceQuarter > 2) {
            shortfallQ2 = Math.max(0, shortfallQ2 - roundToPaisa(suddenReliefTax * 0.45));
        }
        interestQ2 = roundSection288B(shortfallQ2) * 0.01 * 3;
    }

    // Q3 Interest: 1% per month for 3 months (No statutory safe harbor buffer)
    let shortfallQ3 = Math.max(0, reqQ3 - cumPaidQ3);
    let interestQ3 = 0;
    if (shortfallQ3 > 0) {
        if (suddenReliefTax > 0 && suddenIncomeExceptions?.occurrenceQuarter > 3) {
            shortfallQ3 = Math.max(0, shortfallQ3 - roundToPaisa(suddenReliefTax * 0.75));
        }
        interestQ3 = roundSection288B(shortfallQ3) * 0.01 * 3;
    }

    // Q4 Interest: 1% per month for 1 month
    let shortfallQ4 = Math.max(0, reqQ4 - cumPaidQ4);
    let interestQ4 = 0;
    if (shortfallQ4 > 0) {
        interestQ4 = roundSection288B(shortfallQ4) * 0.01 * 1;
    }

    const totalInterest234C = roundSection288B(interestQ1 + interestQ2 + interestQ3 + interestQ4);

    // Section 234B Default Interest:
    // Triggers if total advance tax paid on or before 31 March is < 90% of assessed tax.
    const isNinetyPercentSatisfied = cumPaidQ4 >= roundToPaisa(netTaxPayable * 0.90);
    let shortfall234B = 0;
    let interestSection234B = 0;
    let months234B = 0;

    if (!isNinetyPercentSatisfied) {
        shortfall234B = Math.max(0, netTaxPayable - cumPaidQ4);
        
        // Month count from 1 April of AY to date of determination/assessment
        if (assessmentDate) {
            const ayStart = new Date(`${(assessmentYear ? assessmentYear.slice(0, 4) : Number(taxYear.slice(0, 4)) + 1)}-04-01T00:00:00.000Z`);
            const dAssess = new Date(String(assessmentDate).slice(0, 10) + 'T00:00:00.000Z');
            if (!isNaN(dAssess.getTime()) && dAssess > ayStart) {
                const diffTime = dAssess.getTime() - ayStart.getTime();
                const rawMonths = diffTime / (1000 * 60 * 60 * 24 * 30.4375);
                months234B = Math.max(1, Math.ceil(rawMonths));
            } else {
                months234B = 1;
            }
        } else {
            months234B = 1; // Minimum statutory period: 1 month
        }

        interestSection234B = roundSection288B(shortfall234B) * 0.01 * months234B;
    }

    const totalInterest = roundSection288B(totalInterest234C + interestSection234B);
    const balanceTaxDue = Math.max(0, netTaxPayable - cumPaidQ4);
    const totalPayable = roundToPaisa(balanceTaxDue + totalInterest);

    const installments = [
        {
            quarter: 'Q1',
            statutoryDueDate: `${taxYear.slice(0, 4)}-06-15`,
            cumulativePercentTarget: 15,
            cumulativeRequiredInr: reqQ1,
            cumulativePaidInr: cumPaidQ1,
            quarterlyPaidInr: pQ1,
            shortfallInr: shortfallQ1,
            safeHarbor12PercentThresholdInr: safeHarborQ1Threshold,
            isSafeHarborApplied: safeHarborQ1Applied,
            ratePercent: 1.0,
            statutoryMonths: 3,
            interestInr: interestQ1,
            statutoryRule: safeHarborQ1Applied ? 'SAFE_HARBOR_12_PERCENT_SATISFIED' : (shortfallQ1 > 0 ? 'DEFERMENT_INTEREST_APPLIED' : 'PAID_IN_FULL')
        },
        {
            quarter: 'Q2',
            statutoryDueDate: `${taxYear.slice(0, 4)}-09-15`,
            cumulativePercentTarget: 45,
            cumulativeRequiredInr: reqQ2,
            cumulativePaidInr: cumPaidQ2,
            quarterlyPaidInr: pQ2,
            shortfallInr: shortfallQ2,
            safeHarbor36PercentThresholdInr: safeHarborQ2Threshold,
            isSafeHarborApplied: safeHarborQ2Applied,
            ratePercent: 1.0,
            statutoryMonths: 3,
            interestInr: interestQ2,
            statutoryRule: safeHarborQ2Applied ? 'SAFE_HARBOR_36_PERCENT_SATISFIED' : (shortfallQ2 > 0 ? 'DEFERMENT_INTEREST_APPLIED' : 'PAID_IN_FULL')
        },
        {
            quarter: 'Q3',
            statutoryDueDate: `${taxYear.slice(0, 4)}-12-15`,
            cumulativePercentTarget: 75,
            cumulativeRequiredInr: reqQ3,
            cumulativePaidInr: cumPaidQ3,
            quarterlyPaidInr: pQ3,
            shortfallInr: shortfallQ3,
            isSafeHarborApplied: false,
            ratePercent: 1.0,
            statutoryMonths: 3,
            interestInr: interestQ3,
            statutoryRule: shortfallQ3 > 0 ? 'DEFERMENT_INTEREST_APPLIED' : 'PAID_IN_FULL'
        },
        {
            quarter: 'Q4',
            statutoryDueDate: `${Number(taxYear.slice(0, 4)) + 1}-03-15`,
            cumulativePercentTarget: 100,
            cumulativeRequiredInr: reqQ4,
            cumulativePaidInr: cumPaidQ4,
            quarterlyPaidInr: pQ4,
            shortfallInr: shortfallQ4,
            isSafeHarborApplied: false,
            ratePercent: 1.0,
            statutoryMonths: 1,
            interestInr: interestQ4,
            statutoryRule: shortfallQ4 > 0 ? 'DEFERMENT_INTEREST_APPLIED' : 'PAID_IN_FULL'
        }
    ];

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'APPLICABLE',
        governingAct,
        governingComputationSection,
        governingInstallmentSection,
        governingInterestSection,
        isAct2025: resolvedIsAct2025,
        taxYear,
        assessmentYear: assessmentYear || `${Number(taxYear.slice(0, 4)) + 1}-${Number(taxYear.slice(5, 7)) + 1}`,
        estimatedAssessedTaxInr: estimatedTax,
        tdsTcsCreditInr: credits,
        netTaxPayableInr: netTaxPayable,
        advanceTaxThresholdInr: ADVANCE_TAX_THRESHOLD,
        isAdvanceTaxPayable: true,
        installments,
        totalAdvanceTaxPaidInr: cumPaidQ4,
        balanceTaxDueInr: balanceTaxDue,
        interestSection234CInr: totalInterest234C,
        section234BDetails: {
            ninetyPercentThresholdInr: roundToPaisa(netTaxPayable * 0.90),
            isNinetyPercentSatisfied,
            shortfall234BInr: shortfall234B,
            months234B,
            ratePerMonthPercent: 1.0,
            interestInr: interestSection234B,
            rule: isNinetyPercentSatisfied ? 'EXEMPT_PAID_AT_LEAST_90_PERCENT' : 'DEFAULT_INTEREST_APPLIED_BELOW_90_PERCENT'
        },
        interestSection234BInr: interestSection234B,
        totalInterestInr: totalInterest,
        totalPayableInr: totalPayable,
        isAuthorizedToDisburse: false,
        statutoryRationale: `${governingComputationSection} requires advance tax on net liability ₹${netTaxPayable.toLocaleString('en-IN')}. Section 234C interest ₹${totalInterest234C.toLocaleString('en-IN')} (deferment), Section 234B interest ₹${interestSection234B.toLocaleString('en-IN')} (${isNinetyPercentSatisfied ? 'satisfied >=90%' : 'default <90%'}). Total interest: ₹${totalInterest.toLocaleString('en-IN')}. Note: Payment authorization requires corporate controller sign-off.`
    };
}

/**
 * -----------------------------------------------------------------------------
 * 28. CORPORATE TAX: MAT (SECTION 115JB) VS CONCESSIONAL REGIME (SECTION 115BAA)
 * -----------------------------------------------------------------------------
 * Computes comparative corporate tax liability across:
 * 1. Normal Provisions: 25% (turnover <= 400Cr) or 30% (> 400Cr) + surcharge + cess.
 * 2. Section 115JB MAT: 15% on Adjusted Book Profit + surcharge + cess.
 * 3. Section 115JAA MAT Credit: Set-off against normal tax or newly generated.
 * 4. Section 115BAA Concessional Regime: Flat 22% + 10% surcharge + 4% cess = 25.168% effective.
 *    Exempt from Section 115JB MAT u/s 115JB(5A). Forfeits accumulated MAT credit.
 */
export function calculateCorporateTaxMatVsConcessional({
    financialYear = '2025-26',
    assessmentYear = null,
    turnoverPrecedingFyInr = 0,
    netProfitBeforeTaxInr = 0,
    normalTaxAdditionsInr = 0,
    normalTaxDeductionsInr = 0,
    accumulatedMatCreditInr = 0,
    bookProfitAdjustments = null,
    adjustedBookProfitInr = null,
    optedSection115BAA = null
}) {
    const fy = financialYear || '2025-26';
    const ay = assessmentYear || (fy === '2025-26' ? '2026-27' : '2027-28');
    const turnover = Number(turnoverPrecedingFyInr) || 0;
    const npbt = Number(netProfitBeforeTaxInr) || 0;
    const additions = Number(normalTaxAdditionsInr) || 0;
    const deductions = Number(normalTaxDeductionsInr) || 0;
    const priorMatCredit = Number(accumulatedMatCreditInr) || 0;

    // A. Normal Corporate Tax Regime
    const taxableIncomeNormal = Math.max(0, roundToPaisa(npbt + additions - deductions));
    const baseRateNormal = turnover <= 4000000000 ? 0.25 : 0.30;
    const baseTaxNormal = roundToPaisa(taxableIncomeNormal * baseRateNormal);

    let surchargeRateNormal = 0;
    if (taxableIncomeNormal > 100000000) {
        surchargeRateNormal = 0.12;
    } else if (taxableIncomeNormal > 10000000) {
        surchargeRateNormal = 0.07;
    }
    const surchargeNormal = roundToPaisa(baseTaxNormal * surchargeRateNormal);
    const cessNormal = roundToPaisa((baseTaxNormal + surchargeNormal) * 0.04);
    const grossNormalTax = roundToPaisa(baseTaxNormal + surchargeNormal + cessNormal);

    // B. Minimum Alternate Tax (Section 115JB)
    let adjBookProfit = 0;
    if (adjustedBookProfitInr !== null && adjustedBookProfitInr !== undefined) {
        adjBookProfit = Math.max(0, Number(adjustedBookProfitInr) || 0);
    } else if (bookProfitAdjustments && typeof bookProfitAdjustments === 'object') {
        const bpAdditions = (Number(bookProfitAdjustments.incomeTaxProvision) || 0) +
            (Number(bookProfitAdjustments.reservesTransfer) || 0) +
            (Number(bookProfitAdjustments.contingentLiabilities) || 0) +
            (Number(bookProfitAdjustments.diminutionInAssetValue) || 0) +
            (Number(bookProfitAdjustments.exemptIncomeExpenses) || 0);
        const bfLoss = Number(bookProfitAdjustments.broughtForwardLoss) || 0;
        const unabsDep = Number(bookProfitAdjustments.unabsorbedDepreciation) || 0;
        const lossDepDeduction = (bfLoss > 0 && unabsDep > 0) ? Math.min(bfLoss, unabsDep) : Math.max(bfLoss, unabsDep);
        const bpDeductions = lossDepDeduction + (Number(bookProfitAdjustments.exemptIncome) || 0);
        adjBookProfit = Math.max(0, roundToPaisa(npbt + bpAdditions - bpDeductions));
    } else {
        adjBookProfit = npbt;
    }

    const baseMatRate = 0.15;
    const baseMatTax = roundToPaisa(adjBookProfit * baseMatRate);
    let surchargeRateMat = 0;
    if (adjBookProfit > 100000000) {
        surchargeRateMat = 0.12;
    } else if (adjBookProfit > 10000000) {
        surchargeRateMat = 0.07;
    }
    const surchargeMat = roundToPaisa(baseMatTax * surchargeRateMat);
    const cessMat = roundToPaisa((baseMatTax + surchargeMat) * 0.04);
    const grossMatTax = roundToPaisa(baseMatTax + surchargeMat + cessMat);

    // C. Old Regime Set-off & MAT Credit Determination (Section 115JAA)
    let matCreditUtilized = 0;
    let newMatCreditGenerated = 0;
    let netNormalTaxPayable = grossNormalTax;
    let taxPayableOldRegime = 0;

    if (grossMatTax > grossNormalTax) {
        taxPayableOldRegime = grossMatTax;
        newMatCreditGenerated = roundToPaisa(grossMatTax - grossNormalTax);
    } else {
        const maxCreditAllowable = roundToPaisa(grossNormalTax - grossMatTax);
        matCreditUtilized = Math.min(priorMatCredit, maxCreditAllowable);
        netNormalTaxPayable = roundToPaisa(grossNormalTax - matCreditUtilized);
        taxPayableOldRegime = netNormalTaxPayable;
    }
    const closingMatCreditOldRegime = roundToPaisa(priorMatCredit - matCreditUtilized + newMatCreditGenerated);

    // D. Section 115BAA Concessional Regime
    const taxableIncome115BAA = Math.max(0, roundToPaisa(npbt + additions));
    const baseRate115BAA = 0.22;
    const baseTax115BAA = roundToPaisa(taxableIncome115BAA * baseRate115BAA);
    const surcharge115BAA = roundToPaisa(baseTax115BAA * 0.10); // Flat 10% surcharge u/s 115BAA(1) proviso
    const cess115BAA = roundToPaisa((baseTax115BAA + surcharge115BAA) * 0.04);
    const totalTax115BAA = roundToPaisa(baseTax115BAA + surcharge115BAA + cess115BAA);

    // Under Section 115JB(5A), MAT does not apply to 115BAA companies
    // Under Section 115JAA, accumulated MAT credit cannot be utilized and is forfeited
    const forfeitedMatCredit115BAA = priorMatCredit;

    // E. Strategic Comparative Recommendation
    let recommendedRegime = 'SECTION_115BAA';
    let recommendationRationale = '';
    let netCashSavingsInr = 0;

    if (optedSection115BAA === true) {
        recommendedRegime = 'SECTION_115BAA';
        netCashSavingsInr = roundToPaisa(taxPayableOldRegime - totalTax115BAA);
        recommendationRationale = `Assessee has explicitly opted for Section 115BAA (25.168% effective rate). Section 115JB MAT is inapplicable u/s 115JB(5A). Accumulated MAT credit of INR ${forfeitedMatCredit115BAA.toLocaleString('en-IN')} is forfeited.`;
    } else if (optedSection115BAA === false) {
        recommendedRegime = 'OLD_REGIME';
        netCashSavingsInr = roundToPaisa(totalTax115BAA - taxPayableOldRegime);
        recommendationRationale = `Assessee remains under Old Regime. Applicable tax is INR ${taxPayableOldRegime.toLocaleString('en-IN')} (Normal Tax: INR ${grossNormalTax.toLocaleString('en-IN')}, MAT: INR ${grossMatTax.toLocaleString('en-IN')}). Closing MAT credit balance: INR ${closingMatCreditOldRegime.toLocaleString('en-IN')}.`;
    } else {
        if (taxPayableOldRegime < totalTax115BAA) {
            recommendedRegime = 'OLD_REGIME';
            netCashSavingsInr = roundToPaisa(totalTax115BAA - taxPayableOldRegime);
            recommendationRationale = `Old Regime is optimal for FY ${fy}: Net cash outflow is INR ${taxPayableOldRegime.toLocaleString('en-IN')} vs INR ${totalTax115BAA.toLocaleString('en-IN')} under Section 115BAA (Net saving: INR ${netCashSavingsInr.toLocaleString('en-IN')}). Utilizing MAT credit or Chapter VI-A deductions yields lower current cash drain.`;
        } else {
            recommendedRegime = 'SECTION_115BAA';
            netCashSavingsInr = roundToPaisa(taxPayableOldRegime - totalTax115BAA);
            recommendationRationale = `Section 115BAA is optimal for FY ${fy}: Net tax payable is INR ${totalTax115BAA.toLocaleString('en-IN')} vs INR ${taxPayableOldRegime.toLocaleString('en-IN')} under Old Regime (Net saving: INR ${netCashSavingsInr.toLocaleString('en-IN')}). Form 10-IC must be filed before the due date u/s 139(1). Note: INR ${forfeitedMatCredit115BAA.toLocaleString('en-IN')} in accumulated MAT credit will lapse.`;
        }
    }

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'APPLICABLE',
        governingLaw: 'Income-tax Act, 1961 Sections 115BAA, 115JB & 115JAA',
        financialYear: fy,
        assessmentYear: ay,
        taxableIncomeNormalInr: taxableIncomeNormal,
        adjustedBookProfitInr: adjBookProfit,
        normalTaxRegime: {
            turnoverPrecedingFyInr: turnover,
            baseTaxRate: baseRateNormal,
            baseTaxInr: baseTaxNormal,
            surchargeRate: surchargeRateNormal,
            surchargeInr: surchargeNormal,
            cessInr: cessNormal,
            grossNormalTaxInr: grossNormalTax,
            effectiveRate: taxableIncomeNormal > 0 ? roundToPaisa((grossNormalTax / taxableIncomeNormal) * 100) : 0
        },
        minimumAlternateTaxSec115JB: {
            baseMatRate: baseMatRate,
            baseMatTaxInr: baseMatTax,
            surchargeRate: surchargeRateMat,
            surchargeInr: surchargeMat,
            cessInr: cessMat,
            grossMatTaxInr: grossMatTax,
            effectiveMatRate: adjBookProfit > 0 ? roundToPaisa((grossMatTax / adjBookProfit) * 100) : 0
        },
        matCreditScheduleSec115JAA: {
            openingMatCreditInr: priorMatCredit,
            matCreditUtilizedInr: matCreditUtilized,
            newMatCreditGeneratedInr: newMatCreditGenerated,
            closingMatCreditInr: closingMatCreditOldRegime,
            expiryYears: '15 Assessment Years'
        },
        concessionalRegimeSec115BAA: {
            taxableIncomeInr: taxableIncome115BAA,
            baseTaxRate: 0.22,
            baseTaxInr: baseTax115BAA,
            surchargeRate: 0.10,
            surchargeInr: surcharge115BAA,
            cessInr: cess115BAA,
            totalTaxInr: totalTax115BAA,
            effectiveRate: 25.168,
            matApplicable: false,
            matExemptionClause: 'Section 115JB(5A)',
            forfeitedMatCreditInr: forfeitedMatCredit115BAA
        },
        comparisonSummary: {
            cashOutflowOldRegimeInr: taxPayableOldRegime,
            cashOutflowSection115BAAInr: totalTax115BAA,
            recommendedRegime,
            netCashSavingsInr,
            form10IcRequired: recommendedRegime === 'SECTION_115BAA'
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: recommendationRationale,
        citations: [
            'Income-tax Act, 1961 Section 115BAA (Tax on income of certain domestic companies)',
            'Income-tax Act, 1961 Section 115JB (Special provision for payment of tax by certain companies)',
            'Income-tax Act, 1961 Section 115JB(5A) (Non-applicability of MAT to Section 115BAA)',
            'Income-tax Act, 1961 Section 115JAA (Tax credit in respect of tax paid on deemed income)'
        ]
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 2B. SECTION 43B STATUTORY PAYABLES & SECTION 36(1)(va) CHECKMATE ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates corporate statutory liabilities and payroll welfare deductions under:
 * - Section 43B(a): Taxes, duties, cess, fee (GST, Customs, Property Tax)
 * - Section 43B(b): Employer contribution to EPF, ESIC, gratuity, superannuation
 * - Section 43B(c): Bonus or commission to employees (Sec 36(1)(ii))
 * - Section 43B(d)/(da)/(e): Interest on loans from PFIs, NBFCs, scheduled banks
 * - Section 43B(f): Leave encashment provision
 * - Section 43B(g): Railway payables
 * - Section 36(1)(va) read with Section 2(24)(x): Employee welfare contributions
 *
 * Governing Legal Principles:
 * 1. First Proviso to Section 43B:
 *    - For 43B(a) through 43B(g), deduction is ALLOWED in the year of accrual
 *      if actually paid on or before the due date for filing return under Section 139(1).
 *    - If paid after Section 139(1) due date, deduction is disallowed in current FY
 *      and allowed in the subsequent FY when actually paid (timing difference).
 * 2. Supreme Court Checkmate Doctrine (Checkmate Services P. Ltd. v. CIT (2022) 448 ITR 518 (SC)):
 *    - First Proviso relaxation strictly DOES NOT APPLY to employee contributions (Sec 36(1)(va)).
 *    - Employee contributions withheld from salary constitute deemed income under Section 2(24)(x).
 *    - Allowable as deduction ONLY IF deposited on or before the statutory due date
 *      under the respective Act (e.g. 15th of following month for EPF/ESIC).
 *    - Late deposit results in PERMANENT DISALLOWANCE in that Assessment Year.
 * 3. Explanations 3C & 3D to Section 43B:
 *    - Conversion of unpaid interest into a loan, advance, debenture, or other instrument
 *      does NOT constitute actual payment and remains disallowed.
 */
export function evaluateSection43BStatutoryPayables({
    financialYear = '2025-26',
    assessmentYear = null,
    itrFilingDueDate = '2026-10-31', // Section 139(1) due date for assessees subject to tax audit
    payables = []
} = {}) {
    const defaultAy = assessmentYear || (() => {
        const parts = String(financialYear).split('-');
        if (parts.length === 2 && !isNaN(Number(parts[0]))) {
            const startYr = Number(parts[0]);
            return `${startYr + 1}-${String(startYr + 2).slice(-2)}`;
        }
        return '2026-27';
    })();

    if (!Array.isArray(payables) || payables.length === 0) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'NO_PAYABLES_EVALUATED',
            financialYear,
            assessmentYear: defaultAy,
            itrFilingDueDate,
            totalPayablesCount: 0,
            summary: {
                totalIncurredInr: 0,
                totalAllowedInr: 0,
                totalDisallowedInr: 0,
                totalTimingDifferencesInr: 0,
                totalPermanentDisallowancesInr: 0,
                form3CdClause26DisallowanceInr: 0,
                form3CdClause20bDisallowanceInr: 0
            },
            evaluatedPayables: [],
            isAuthorizedToDisburse: false,
            statutoryRationale: 'No statutory payable items submitted for Section 43B / 36(1)(va) evaluation.'
        };
    }

    const evaluatedPayables = [];
    let totalIncurredInr = 0;
    let totalAllowedInr = 0;
    let totalDisallowedInr = 0;
    let totalTimingDifferencesInr = 0;
    let totalPermanentDisallowancesInr = 0;
    let form3CdClause26DisallowanceInr = 0;
    let form3CdClause20bDisallowanceInr = 0;

    const clauseBreakdown = {};

    payables.forEach((item, idx) => {
        const payableId = item.payableId || `PAY-${String(idx + 1).padStart(3, '0')}`;
        const rawClause = String(item.clause || '43B(a)').trim();
        const normalizedClause = rawClause.toLowerCase().includes('36')
            ? '36(1)(va)'
            : rawClause.toUpperCase(); // e.g. '43B(A)', '43B(B)', '43B(C)', etc.
        const description = item.description || `Statutory Liability under ${normalizedClause}`;
        const amountInr = roundToPaisa(Math.max(0, Number(item.amountInr) || 0));
        const statutoryDueDate = item.statutoryDueDate ? String(item.statutoryDueDate).trim() : null;
        const paidDate = item.paidDate ? String(item.paidDate).trim() : null;
        const paidAmountInr = paidDate
            ? roundToPaisa(item.paidAmountInr !== undefined && item.paidAmountInr !== null
                ? Math.max(0, Number(item.paidAmountInr) || 0)
                : amountInr)
            : 0;
        const isInterestConvertedToLoan = Boolean(item.isInterestConvertedToLoan);

        totalIncurredInr = roundToPaisa(totalIncurredInr + amountInr);

        if (!clauseBreakdown[normalizedClause]) {
            clauseBreakdown[normalizedClause] = {
                incurredInr: 0,
                allowedInr: 0,
                disallowedInr: 0
            };
        }
        clauseBreakdown[normalizedClause].incurredInr = roundToPaisa(clauseBreakdown[normalizedClause].incurredInr + amountInr);

        // ── CATEGORY 1: EMPLOYEE WELFARE CONTRIBUTIONS (Section 36(1)(va)) ──
        if (normalizedClause === '36(1)(va)') {
            // Checkmate Doctrine: Must be deposited on or before statutory due date under EPF/ESIC regulations
            const effectiveDueDate = statutoryDueDate || '2026-04-15';
            const isOnTime = Boolean(paidDate && paidDate <= effectiveDueDate);

            if (isOnTime) {
                const allowedAmt = roundToPaisa(Math.min(paidAmountInr, amountInr));
                const disallowedAmt = roundToPaisa(amountInr - allowedAmt);

                totalAllowedInr = roundToPaisa(totalAllowedInr + allowedAmt);
                clauseBreakdown[normalizedClause].allowedInr = roundToPaisa(clauseBreakdown[normalizedClause].allowedInr + allowedAmt);

                if (disallowedAmt > 0) {
                    totalDisallowedInr = roundToPaisa(totalDisallowedInr + disallowedAmt);
                    totalPermanentDisallowancesInr = roundToPaisa(totalPermanentDisallowancesInr + disallowedAmt);
                    form3CdClause20bDisallowanceInr = roundToPaisa(form3CdClause20bDisallowanceInr + disallowedAmt);
                    clauseBreakdown[normalizedClause].disallowedInr = roundToPaisa(clauseBreakdown[normalizedClause].disallowedInr + disallowedAmt);
                }

                evaluatedPayables.push({
                    payableId,
                    clause: normalizedClause,
                    description,
                    amountInr,
                    statutoryDueDate: effectiveDueDate,
                    paidDate,
                    paidAmountInr,
                    isAllowed: disallowedAmt === 0,
                    allowedAmountInr: allowedAmt,
                    disallowedAmountInr: disallowedAmt,
                    disallowanceType: disallowedAmt > 0 ? 'PERMANENT_ADD_BACK' : 'NONE',
                    governingRule: 'Section 36(1)(va) read with Section 2(24)(x) & Checkmate Doctrine',
                    taxAuditReporting: {
                        form: 'Form 3CD',
                        clause: 'Clause 20(b)',
                        status: disallowedAmt > 0 ? 'PARTIAL_DISALLOWANCE_REPORTABLE' : 'COMPLIANT'
                    },
                    statutoryOutcome: disallowedAmt > 0 ? 'DISALLOWED_PERMANENT_CHECKMATE' : 'ALLOWED',
                    statutoryAction: disallowedAmt > 0
                        ? `Unpaid portion of INR ${disallowedAmt.toLocaleString('en-IN')} permanently disallowed under Section 36(1)(va).`
                        : `Employee contribution deposited on or before statutory due date (${effectiveDueDate}). Fully deductible under Section 36(1)(va).`
                });
            } else {
                // Late deposit or unpaid -> Permanent Disallowance under Checkmate Services (2022)
                totalDisallowedInr = roundToPaisa(totalDisallowedInr + amountInr);
                totalPermanentDisallowancesInr = roundToPaisa(totalPermanentDisallowancesInr + amountInr);
                form3CdClause20bDisallowanceInr = roundToPaisa(form3CdClause20bDisallowanceInr + amountInr);
                clauseBreakdown[normalizedClause].disallowedInr = roundToPaisa(clauseBreakdown[normalizedClause].disallowedInr + amountInr);

                evaluatedPayables.push({
                    payableId,
                    clause: normalizedClause,
                    description,
                    amountInr,
                    statutoryDueDate: effectiveDueDate,
                    paidDate,
                    paidAmountInr,
                    isAllowed: false,
                    allowedAmountInr: 0,
                    disallowedAmountInr: amountInr,
                    disallowanceType: 'PERMANENT_ADD_BACK',
                    governingRule: 'Section 36(1)(va) read with Section 2(24)(x) & Supreme Court ruling in Checkmate Services (2022) 448 ITR 518 (SC)',
                    taxAuditReporting: {
                        form: 'Form 3CD',
                        clause: 'Clause 20(b)',
                        status: 'MANDATORY_PERMANENT_DISALLOWANCE'
                    },
                    statutoryOutcome: 'DISALLOWED_PERMANENT_CHECKMATE',
                    statutoryAction: paidDate
                        ? `Deposited after statutory due date (${paidDate} > ${effectiveDueDate}). Under Supreme Court Checkmate Doctrine read with Explanation 5 to Section 36(1)(va) and Section 2(24)(x), late deposit of employee contribution is permanently disallowed. Cannot be claimed under Section 43B or in any subsequent year.`
                        : `Unpaid employee contribution past statutory due date (${effectiveDueDate}). Permanently disallowed under Section 36(1)(va) read with Section 2(24)(x).`
                });
            }
        }
        // ── CATEGORY 2: EMPLOYER STATUTORY PAYABLES (Section 43B(a) to 43B(g)) ──
        else {
            // Check Explanation 3C & 3D for loan conversion
            if (isInterestConvertedToLoan) {
                totalDisallowedInr = roundToPaisa(totalDisallowedInr + amountInr);
                totalTimingDifferencesInr = roundToPaisa(totalTimingDifferencesInr + amountInr);
                form3CdClause26DisallowanceInr = roundToPaisa(form3CdClause26DisallowanceInr + amountInr);
                clauseBreakdown[normalizedClause].disallowedInr = roundToPaisa(clauseBreakdown[normalizedClause].disallowedInr + amountInr);

                evaluatedPayables.push({
                    payableId,
                    clause: normalizedClause,
                    description,
                    amountInr,
                    itrFilingDueDate,
                    paidDate,
                    paidAmountInr,
                    isAllowed: false,
                    allowedAmountInr: 0,
                    disallowedAmountInr: amountInr,
                    disallowanceType: 'TIMING_DIFFERENCE_ADD_BACK',
                    governingRule: 'Section 43B Explanations 3C & 3D (No actual payment via debt restructuring)',
                    taxAuditReporting: {
                        form: 'Form 3CD',
                        clause: 'Clause 26',
                        status: 'DISALLOWANCE_REPORTABLE'
                    },
                    statutoryOutcome: 'DISALLOWED_INTEREST_CONVERSION',
                    statutoryAction: 'Conversion of unpaid interest into a loan, advance, debenture, or funded interest term loan does NOT constitute actual payment under Explanation 3C/3D to Section 43B. Disallowed until actual cash/bank outflow occurs.'
                });
            } else {
                // First Proviso evaluation: Paid on or before Section 139(1) due date
                const isPaidBeforeItrDue = Boolean(paidDate && paidDate <= itrFilingDueDate);

                if (isPaidBeforeItrDue) {
                    const allowedAmt = roundToPaisa(Math.min(paidAmountInr, amountInr));
                    const disallowedAmt = roundToPaisa(amountInr - allowedAmt);

                    totalAllowedInr = roundToPaisa(totalAllowedInr + allowedAmt);
                    clauseBreakdown[normalizedClause].allowedInr = roundToPaisa(clauseBreakdown[normalizedClause].allowedInr + allowedAmt);

                    if (disallowedAmt > 0) {
                        totalDisallowedInr = roundToPaisa(totalDisallowedInr + disallowedAmt);
                        totalTimingDifferencesInr = roundToPaisa(totalTimingDifferencesInr + disallowedAmt);
                        form3CdClause26DisallowanceInr = roundToPaisa(form3CdClause26DisallowanceInr + disallowedAmt);
                        clauseBreakdown[normalizedClause].disallowedInr = roundToPaisa(clauseBreakdown[normalizedClause].disallowedInr + disallowedAmt);
                    }

                    evaluatedPayables.push({
                        payableId,
                        clause: normalizedClause,
                        description,
                        amountInr,
                        itrFilingDueDate,
                        paidDate,
                        paidAmountInr,
                        isAllowed: disallowedAmt === 0,
                        allowedAmountInr: allowedAmt,
                        disallowedAmountInr: disallowedAmt,
                        disallowanceType: disallowedAmt > 0 ? 'TIMING_DIFFERENCE_ADD_BACK' : 'NONE',
                        governingRule: 'Section 43B First Proviso (Actual payment before return filing cutoff)',
                        taxAuditReporting: {
                            form: 'Form 3CD',
                            clause: 'Clause 26',
                            status: disallowedAmt > 0 ? 'PARTIAL_DISALLOWANCE_REPORTABLE' : 'COMPLIANT'
                        },
                        statutoryOutcome: disallowedAmt > 0 ? 'DISALLOWED_TIMING_DIFFERENCE' : 'ALLOWED_FIRST_PROVISO',
                        statutoryAction: disallowedAmt > 0
                            ? `Unpaid balance of INR ${disallowedAmt.toLocaleString('en-IN')} disallowed in current FY; allowable upon actual payment in subsequent FY.`
                            : `Paid on or before Section 139(1) return filing due date (${paidDate} <= ${itrFilingDueDate}). Fully deductible in current financial year (${financialYear}) under Section 43B First Proviso.`
                    });
                } else {
                    // Late payment or unpaid -> Timing Difference Disallowance under Section 43B
                    totalDisallowedInr = roundToPaisa(totalDisallowedInr + amountInr);
                    totalTimingDifferencesInr = roundToPaisa(totalTimingDifferencesInr + amountInr);
                    form3CdClause26DisallowanceInr = roundToPaisa(form3CdClause26DisallowanceInr + amountInr);
                    clauseBreakdown[normalizedClause].disallowedInr = roundToPaisa(clauseBreakdown[normalizedClause].disallowedInr + amountInr);

                    evaluatedPayables.push({
                        payableId,
                        clause: normalizedClause,
                        description,
                        amountInr,
                        itrFilingDueDate,
                        paidDate,
                        paidAmountInr,
                        isAllowed: false,
                        allowedAmountInr: 0,
                        disallowedAmountInr: amountInr,
                        disallowanceType: 'TIMING_DIFFERENCE_ADD_BACK',
                        governingRule: 'Section 43B (Payment not made on or before Section 139(1) due date)',
                        taxAuditReporting: {
                            form: 'Form 3CD',
                            clause: 'Clause 26',
                            status: 'MANDATORY_DISALLOWANCE_REPORTABLE'
                        },
                        statutoryOutcome: 'DISALLOWED_TIMING_DIFFERENCE',
                        statutoryAction: paidDate
                            ? `Paid after Section 139(1) return filing due date (${paidDate} > ${itrFilingDueDate}). Strictly DISALLOWED in current FY (${financialYear}) and carried forward as deductible in the subsequent FY of actual disbursement.`
                            : `Unpaid as of Section 139(1) return filing due date (${itrFilingDueDate}). Disallowed in current FY (${financialYear}); will become deductible upon actual payment in subsequent financial year.`
                    });
                }
            }
        }
    });

    let overallDetermination = 'ALL_PAYABLES_QUALIFIED';
    if (totalPermanentDisallowancesInr > 0 && totalTimingDifferencesInr > 0) {
        overallDetermination = 'DISALLOWED_MIXED_PERMANENT_AND_TIMING';
    } else if (totalPermanentDisallowancesInr > 0) {
        overallDetermination = 'DISALLOWED_PERMANENT_CHECKMATE';
    } else if (totalTimingDifferencesInr > 0) {
        overallDetermination = 'DISALLOWED_TIMING_DIFFERENCE';
    }

    return {
        status: 'SUCCESS',
        statutoryDetermination: overallDetermination,
        financialYear,
        assessmentYear: defaultAy,
        itrFilingDueDate,
        totalPayablesCount: evaluatedPayables.length,
        summary: {
            totalIncurredInr,
            totalAllowedInr,
            totalDisallowedInr,
            totalTimingDifferencesInr,
            totalPermanentDisallowancesInr,
            form3CdClause26DisallowanceInr,
            form3CdClause20bDisallowanceInr,
            clauseBreakdown
        },
        evaluatedPayables,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Section 43B / 36(1)(va) Evaluation: Evaluated ${evaluatedPayables.length} liabilities (Total INR ${totalIncurredInr.toLocaleString('en-IN')}). Allowed: INR ${totalAllowedInr.toLocaleString('en-IN')}; Total Disallowed: INR ${totalDisallowedInr.toLocaleString('en-IN')} (Permanent Clause 20(b) Checkmate: INR ${totalPermanentDisallowancesInr.toLocaleString('en-IN')}; Timing Difference Clause 26: INR ${totalTimingDifferencesInr.toLocaleString('en-IN')}). Statutory audit calculations do not authorize payment disbursement.`
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 9. SECTION 194A: DETERMINISTIC TDS ON INTEREST (OTHER THAN SECURITIES)
 * ─────────────────────────────────────────────────────────────────────────────
 * Income-tax Act, 1961: Section 194A
 * Income-tax Act, 2025: Section 393(1), Table 1, Sl. No. 5(ii) and 5(iii)
 *
 * Implements authoritative interest withholding logic:
 * - Rates:
 *   - Standard rate: 10.0%
 *   - Missing / Inoperative PAN: 20.0% (Section 206AA(1)(iii) / Section 397)
 *   - Valid Form 15G / Form 15H: 0.0%
 * - Annual Thresholds per Payee:
 *   - Banking Company, Co-operative Bank, or Post Office: ₹40,000 (₹50,000 for Senior Citizens u/s 80TTB)
 *   - Any Other Payer (Commercial, Vendor Delayed Interest, Inter-Corporate Loans): ₹5,000
 *   - Firm to Partner / Exempt Institutions: Always 0% (Section 194A(3) statutory exclusion)
 * - MSMED Act Section 16 Interaction:
 *   - Penal interest payable under Section 16 to Micro & Small enterprises is subject to 194A withholding if exceeding ₹5,000.
 *   - Section 23 of MSMED Act: Strictly disallowed as deduction in computing taxable income (tax add-back).
 * - Temporal Resolution: Earlier of credit in books or payment disbursement.
 * - Dual-Act Crystallization: Pre-1 April 2026 -> 1961 Act; Post-1 April 2026 -> 2025 Act.
 */
export function evaluateSection194A({
    interestAmount,
    priorPaymentsInCurrentFyInr = 0,
    priorFyCategoryPayments = null,
    priorFyPaymentsAlreadyTaxed = null,
    payeeCategory = 'COMMERCIAL_OTHER', // 'COMMERCIAL_OTHER' | 'BANKING_OR_COOPERATIVE' | 'SENIOR_CITIZEN_DEPOSIT' | 'FIRM_TO_PARTNER' | 'EXEMPT_INSTITUTION'
    hasPan = true,
    isPanInoperative = false,
    hasForm15G15H = false,
    isForm15Valid = true,
    isMsmeInterestSec16 = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null
} = {}) {
    const rawInterestAmount = assertMandatoryNumeric(interestAmount, 'interestAmount', 'evaluateSection194A');

    const effectivePriorInput = priorPaymentsInCurrentFyInr !== null && priorPaymentsInCurrentFyInr !== undefined
        ? priorPaymentsInCurrentFyInr
        : priorFyCategoryPayments;
    const prior = Math.max(0, cleanNumericAmount(effectivePriorInput));
    const cumulative = prior + rawInterestAmount;

    // Temporal Resolution: "Credit or Payment, whichever is earlier"
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isAct2025 = Boolean(effectiveDate && String(effectiveDate).slice(0, 10) >= '2026-04-01');
    // Payee Category & Threshold Resolution
    const isExemptInstitution = payeeCategory === 'EXEMPT_INSTITUTION';
    const isFirmToPartner = payeeCategory === 'FIRM_TO_PARTNER';
    const isStatutoryExclusion = isExemptInstitution || isFirmToPartner;

    let INTEREST_THRESHOLD;
    let governingSection;
    if (isAct2025) {
        if (payeeCategory === 'SENIOR_CITIZEN_DEPOSIT') {
            INTEREST_THRESHOLD = 100000;
            governingSection = "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 5(ii)";
        } else if (payeeCategory === 'BANKING_OR_COOPERATIVE') {
            INTEREST_THRESHOLD = 50000;
            governingSection = "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 5(ii)";
        } else {
            INTEREST_THRESHOLD = 10000;
            governingSection = "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 5(iii)";
        }
    } else {
        if (payeeCategory === 'SENIOR_CITIZEN_DEPOSIT') {
            INTEREST_THRESHOLD = 50000;
        } else if (payeeCategory === 'BANKING_OR_COOPERATIVE') {
            INTEREST_THRESHOLD = 40000;
        } else {
            INTEREST_THRESHOLD = 5000;
        }
        governingSection = "Income-tax Act, 1961 Section 194A";
    }

    const isThresholdBreached = !isStatutoryExclusion && (cumulative > INTEREST_THRESHOLD);

    let taxableBase = 0;
    let thresholdDisposition = 'BELOW_THRESHOLD_EXEMPT';

    if (isStatutoryExclusion) {
        taxableBase = 0;
        thresholdDisposition = isFirmToPartner
            ? 'EXEMPT_FIRM_TO_PARTNER_SEC_194A_3_IV'
            : 'EXEMPT_INSTITUTION_SEC_194A_3_III';
    } else if (isThresholdBreached) {
        const previouslyTaxed = priorFyPaymentsAlreadyTaxed !== null && priorFyPaymentsAlreadyTaxed !== undefined
            ? Math.max(0, Number(priorFyPaymentsAlreadyTaxed))
            : (prior > INTEREST_THRESHOLD ? prior : 0);

        taxableBase = Math.max(0, cumulative - previouslyTaxed);
        thresholdDisposition = prior <= INTEREST_THRESHOLD && previouslyTaxed === 0
            ? 'CATEGORY_THRESHOLD_BREACHED_WITH_CATCHUP'
            : 'CATEGORY_THRESHOLD_ACTIVE';
    }

    // Form 15G / 15H Exemption
    const isForm15Applied = Boolean(hasForm15G15H) && Boolean(isForm15Valid) && !isStatutoryExclusion;

    // Rate Determination
    const baseStatutoryRatePercent = (isStatutoryExclusion || isForm15Applied) ? 0.0 : 10.0;
    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    const panFloorResolution = resolveMissingPanPenaltyFloor({
        sectionCode: '194A',
        tableItem: isAct2025 ? '393_4' : null,
        isAct2025
    });

    const standardRatePercent = (isStatutoryExclusion || isForm15Applied)
        ? 0.0
        : (panAvailable ? baseStatutoryRatePercent : Math.max(baseStatutoryRatePercent, panFloorResolution.penaltyFloorPercent));

    // Lower Deduction Certificate resolution
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && certResolution.isValid && !isStatutoryExclusion && !isForm15Applied;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    let calculatedTdsInr = 0;
    if (taxableBase > 0 && !isForm15Applied && !isStatutoryExclusion) {
        calculatedTdsInr = Number(((taxableBase * certifiedRatePercent) / 100).toFixed(2));
    }

    const effectiveRatePercent = taxableBase > 0
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : standardRatePercent;

    // MSMED Act Section 16 & Section 23 Interaction
    const msmeAnalysis = isMsmeInterestSec16 ? {
        isMsmePenalInterest: true,
        msmedStatute: "Micro, Small and Medium Enterprises Development Act, 2006",
        section16Status: "SUBJECT_TO_SECTION_194A_WITHHOLDING",
        section23Disallowance: "MANDATORY_TAX_ADDBACK",
        statutoryRationale: "Interest paid under Section 16 of the MSMED Act constitutes interest under Section 2(28A) of the Income-tax Act, 1961 and is subject to Section 194A TDS upon crossing ₹5,000 threshold. Under Section 23 of the MSMED Act, this interest is strictly non-deductible as an expenditure in computing total taxable income."
    } : null;

    let formulaText = "";
    if (isStatutoryExclusion) {
        formulaText = `Section 194A exempt: Payee category ${payeeCategory} excluded under Section 194A(3).`;
    } else if (isForm15Applied) {
        formulaText = `Section 194A NIL deduction: Payee furnished valid Form 15G/15H declaration under Section 197A.`;
    } else if (taxableBase === 0) {
        formulaText = `Section 194A inapplicable: Cumulative interest (₹${cumulative.toLocaleString('en-IN')}) does not exceed ₹${INTEREST_THRESHOLD.toLocaleString('en-IN')} annual threshold.`;
    } else if (hasLowerDeduction) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else if (panAvailable) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 20% (${panFloorResolution.governingProviso}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    }

    const netPayable = roundToPaisa(Math.max(0, rawInterestAmount - calculatedTdsInr));

    return {
        section: isAct2025 ? '393' : '194A',
        isAct2025,
        governingSection,
        governingPanPenaltySection: panFloorResolution.governingProviso,
        rawInterestAmount,
        payeeCategory,
        priorFyInterest: prior,
        cumulativeFyInterest: cumulative,
        interestThreshold: INTEREST_THRESHOLD,
        isThresholdBreached,
        thresholdDisposition,
        hasForm15G15H: isForm15Applied,
        isPanFurnished: panAvailable,
        taxableBase,
        baseStatutoryRatePercent,
        standardRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        taxAmountRoundedSection288B: roundSection288B(calculatedTdsInr),
        netPayable,
        msmeSection16Analysis: msmeAnalysis,
        taxAddbackDisallowedMsmeSec23: Boolean(isMsmeInterestSec16),
        statutoryFormula: formulaText,
        isAuthorizedToDisburse: false
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 10. SECTION 194O: DETERMINISTIC TDS ON E-COMMERCE OPERATORS ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Income-tax Act, 1961: Section 194O
 * Income-tax Act, 2025: Section 393(1), Table 1, Sl. No. 8(i)
 *
 * Implements authoritative marketplace withholding logic:
 * - Rates:
 *   - Prior to 1 October 2024: 1.0%
 *   - On or after 1 October 2024: 0.1% (Finance (No. 2) Act, 2024 rate cut)
 *   - Missing / Inoperative PAN: 5.0% (Section 206AA(1) second proviso specific cap)
 * - De Minimis Annual Threshold:
 *   - Individuals and HUFs: ₹5,00,000 gross GMV per FY (provided PAN is furnished)
 *   - Companies, LLPs, Firms: ₹0 threshold (TDS applies from ₹1)
 * - Gross Amount Scope:
 *   - Includes direct payments made by purchaser to seller deemed paid by operator (Explanation to 194O(1)).
 * - Statutory Precedence:
 *   - Overrides Section 194Q (buyer TDS) and Section 206C(1H) (seller TCS).
 * - Temporal Resolution: Earlier of credit to account (including clearing) or payment.
 */
export function evaluateSection194O({
    grossSaleAmount,
    directPurchaserPaymentDeemedInr = 0,
    priorPaymentsInCurrentFyInr = 0,
    priorFyCategoryPayments = null,
    priorFyPaymentsAlreadyTaxed = null,
    participantEntityType = 'INDIVIDUAL_HUF', // 'INDIVIDUAL_HUF' | 'COMPANY' | 'PARTNERSHIP_FIRM' | 'LLP' | 'OTHER'
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null,
    creditDate = null,
    paymentDate = null,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    certificateMonetaryCeilingInr = null,
    certificatePriorUtilizedInr = 0,
    certificateSection = null
} = {}) {
    const rawGrossSale = assertMandatoryNumeric(grossSaleAmount, 'grossSaleAmount', 'evaluateSection194O');
    const deemedDirectPayment = Math.max(0, cleanNumericAmount(directPurchaserPaymentDeemedInr));
    const totalTransactionGmv = rawGrossSale + deemedDirectPayment;

    const effectivePriorInput = priorPaymentsInCurrentFyInr !== null && priorPaymentsInCurrentFyInr !== undefined
        ? priorPaymentsInCurrentFyInr
        : priorFyCategoryPayments;
    const prior = Math.max(0, cleanNumericAmount(effectivePriorInput));
    const cumulative = prior + totalTransactionGmv;

    // Temporal Resolution: "Credit or Payment, whichever is earlier"
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isAct2025 = Boolean(effectiveDate && String(effectiveDate).slice(0, 10) >= '2026-04-01');
    const governingSection = isAct2025
        ? "Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 8(i)"
        : "Income-tax Act, 1961 Section 194O";

    // Rate Determination: 1.0% pre-Oct 2024, 0.1% post-Oct 2024 (Finance (No. 2) Act 2024)
    const effectiveDateIso = effectiveDate ? String(effectiveDate).slice(0, 10) : '2025-04-01';
    const isPostOctober2024 = effectiveDateIso >= '2024-10-01';
    const baseStatutoryRatePercent = isPostOctober2024 ? 0.1 : 1.0;

    const panAvailable = Boolean(hasPan) && !isPanInoperative;
    // Section 206AA(1) second proviso specifies a capped 5% penalty rate for Section 194-O (not 20%)
    const penaltyRatePercent = 5.0;
    const standardRatePercent = panAvailable ? baseStatutoryRatePercent : penaltyRatePercent;

    // De Minimis Threshold Resolution
    const isIndividualOrHuf = participantEntityType === 'INDIVIDUAL_HUF';
    // Individual/HUF threshold is ₹5 Lakhs, but ONLY if PAN is provided! If PAN is missing, threshold is ₹0
    const THRESHOLD = (isIndividualOrHuf && panAvailable) ? 500000 : 0;
    const isThresholdBreached = cumulative > THRESHOLD;

    let taxableBase = 0;
    let thresholdDisposition = 'BELOW_THRESHOLD_EXEMPT';

    if (isThresholdBreached) {
        const previouslyTaxed = priorFyPaymentsAlreadyTaxed !== null && priorFyPaymentsAlreadyTaxed !== undefined
            ? Math.max(0, Number(priorFyPaymentsAlreadyTaxed))
            : (prior > THRESHOLD ? prior : 0);

        taxableBase = Math.max(0, cumulative - previouslyTaxed);
        thresholdDisposition = prior <= THRESHOLD && previouslyTaxed === 0
            ? 'CATEGORY_THRESHOLD_BREACHED_WITH_CATCHUP'
            : 'CATEGORY_THRESHOLD_ACTIVE';
    } else if (isIndividualOrHuf && panAvailable && cumulative <= THRESHOLD) {
        thresholdDisposition = 'INDIVIDUAL_HUF_BELOW_5L_EXEMPT';
    }

    // Lower Deduction Certificate resolution
    const certResolution = evaluateLowerDeductionCertificateValidity({
        effectiveDate,
        lowerDeductionCertificate,
        certificateSection,
        isAct2025
    });
    const hasLowerDeduction = (lowerDeductionRate !== null && lowerDeductionRate !== undefined) && certResolution.isValid;
    const certifiedRatePercent = hasLowerDeduction ? Number(lowerDeductionRate) : standardRatePercent;

    let calculatedTdsInr = 0;
    if (taxableBase > 0) {
        calculatedTdsInr = Number(((taxableBase * certifiedRatePercent) / 100).toFixed(2));
    }

    const effectiveRatePercent = taxableBase > 0
        ? Number(((calculatedTdsInr / taxableBase) * 100).toFixed(4))
        : standardRatePercent;

    let formulaText = "";
    if (taxableBase === 0) {
        formulaText = `Section 194O inapplicable: Cumulative GMV (₹${cumulative.toLocaleString('en-IN')}) does not exceed ₹${THRESHOLD.toLocaleString('en-IN')} annual threshold for ${participantEntityType}.`;
    } else if (hasLowerDeduction) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${certifiedRatePercent}% (${isAct2025 ? 'Section 395' : 'Section 197'} Certificate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else if (panAvailable) {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × ${standardRatePercent}% (${isPostOctober2024 ? 'Finance (No. 2) Act 2024 rate 0.1%' : 'Pre-amendment rate 1.0%'}) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    } else {
        formulaText = `Taxable Base (₹${taxableBase.toLocaleString('en-IN')}) × 5% (Section 206AA second proviso capped rate) = ₹${calculatedTdsInr.toLocaleString('en-IN')}`;
    }

    const netPayable = roundToPaisa(Math.max(0, rawGrossSale - calculatedTdsInr));

    return {
        section: isAct2025 ? '393' : '194O',
        isAct2025,
        governingSection,
        governingPanPenaltySection: panAvailable ? null : "Section 206AA(1) second proviso",
        rawGrossSale,
        directPurchaserPaymentDeemedInr: deemedDirectPayment,
        totalTransactionGmv,
        participantEntityType,
        priorFyGmv: prior,
        cumulativeFyGmv: cumulative,
        gmvThreshold: THRESHOLD,
        isThresholdBreached,
        thresholdDisposition,
        isPanFurnished: panAvailable,
        taxableBase,
        baseStatutoryRatePercent,
        standardRatePercent,
        effectiveRatePercent,
        calculatedTdsInr,
        taxAmountRoundedSection288B: roundSection288B(calculatedTdsInr),
        netPayable,
        statutoryPrecedence: {
            overridesSection194Q: true,
            overridesSection206C1H: true,
            statutoryRationale: "Under Section 194O(3) read with Section 194Q(5) and Section 206C(1H), transactions subject to Section 194O take absolute statutory precedence. No buyer TDS under 194Q or seller TCS under 206C(1H) shall be deducted or collected."
        },
        statutoryFormula: formulaText,
        isAuthorizedToDisburse: false
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * SECTION 194-IA: TDS ON TRANSFER OF CERTAIN IMMOVABLE PROPERTY
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates tax deduction at source on acquisition/transfer of immovable property
 * (land, commercial office, factory, residential premises other than agricultural land):
 * - Rate: 1.0% on HIGHER of contractual consideration or Stamp Duty Value (SDV)
 *   (Finance Act, 2022 amendment w.e.f. 1 April 2022).
 * - Threshold: INR 50,00,000 (Section 194-IA(2)).
 *   Applies if either consideration OR SDV >= INR 50,00,000.
 * - Consideration Scope (Finance Act, 2019 Explanation):
 *   Includes club membership fees, car parking charges, electricity/water facility charges,
 *   maintenance advance fees, and all other charges incidental to transfer.
 * - Missing / Inoperative PAN: 20.0% statutory penalty rate under Section 206AA.
 * - Exemptions: Agricultural land, non-resident sellers (governed by Section 195).
 * - Compliance: Form 26QB (30 days from month end), Form 16B (15 days from 26QB).
 * - Dual-Act 2025: Section 393(1), Table 1, Sl. No. 3(i).
 */
export function evaluateSection194IA({
    contractualConsiderationInr,
    stampDutyValueInr = null,
    incidentalChargesInr = 0,
    isAgriculturalLand = false,
    isSellerResident = true,
    hasPan = true,
    isPanInoperative = false,
    lowerDeductionRate = null,
    lowerDeductionCertificate = null,
    transactionDate = null,
    creditDate = null,
    paymentDate = null
} = {}) {
    const rawBaseConsideration = Math.max(0, assertMandatoryNumeric(contractualConsiderationInr, 'contractualConsiderationInr', 'evaluateSection194IA'));

    // Exclusion 1: Agricultural Land
    if (isAgriculturalLand) {
        const grossAmount = roundToPaisa(rawBaseConsideration + Math.max(0, cleanNumericAmount(incidentalChargesInr)));
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_AGRICULTURAL_LAND',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            netPayableToSellerInr: grossAmount,
            governingLaw: 'Income-tax Act, 1961 Section 194-IA Explanation (agricultural land excluded)',
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Section 194-IA specifically excludes agricultural land from tax deduction at source. No withholding applies.'
        };
    }

    // Exclusion 2: Non-Resident Seller
    if (!isSellerResident) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'GOVERNED_BY_SECTION_195',
            applicableTdsRatePercent: null,
            taxableBaseInr: null,
            calculatedTdsInr: null,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 195 (Cross-Border Non-Resident Remittance)',
            statutoryRationale: 'Section 194-IA applies exclusively to resident transferors. Remittances to non-resident sellers on property transfers are subject to Section 195 withholding on capital gains (with Form 15CA/15CB routing).'
        };
    }

    const rawIncidental = Math.max(0, cleanNumericAmount(incidentalChargesInr));
    const totalContractualConsideration = roundToPaisa(rawBaseConsideration + rawIncidental);

    const effectiveSdv = (stampDutyValueInr !== null && stampDutyValueInr !== undefined)
        ? roundToPaisa(Math.max(0, cleanNumericAmount(stampDutyValueInr)))
        : totalContractualConsideration;

    // Finance Act 2022 Amendment: Higher of consideration or Stamp Duty Value
    const taxableBaseInr = roundToPaisa(Math.max(totalContractualConsideration, effectiveSdv));

    // Threshold Check: De minimis INR 50 Lakh (Section 194-IA(2))
    // TDS triggered if EITHER consideration OR SDV >= INR 50,00,000
    const isThresholdMet = (totalContractualConsideration >= 5000000 || effectiveSdv >= 5000000);

    // Temporal Resolution: Earlier of credit or payment
    let effectiveDate = transactionDate;
    let withholdingTrigger = 'TRANSACTION_DATE';

    if (creditDate && paymentDate) {
        if (creditDate <= paymentDate) {
            effectiveDate = creditDate;
            withholdingTrigger = 'CREDIT_TO_ACCOUNT';
        } else {
            effectiveDate = paymentDate;
            withholdingTrigger = 'PAYMENT_DISBURSEMENT';
        }
    } else if (creditDate) {
        effectiveDate = creditDate;
        withholdingTrigger = 'CREDIT_TO_ACCOUNT';
    } else if (paymentDate) {
        effectiveDate = paymentDate;
        withholdingTrigger = 'PAYMENT_DISBURSEMENT';
    }

    const isAct2025 = Boolean(effectiveDate && String(effectiveDate).slice(0, 10) >= '2026-04-01');
    const governingSection = isAct2025
        ? 'Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 3(i)'
        : 'Income-tax Act, 1961 Section 194-IA';

    if (!isThresholdMet) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'BELOW_DE_MINIMIS_THRESHOLD',
            applicableTdsRatePercent: 0,
            contractualConsiderationInr: totalContractualConsideration,
            stampDutyValueInr: effectiveSdv,
            taxableBaseInr,
            calculatedTdsInr: 0,
            netPayableToSellerInr: totalContractualConsideration,
            effectiveDate,
            withholdingTrigger,
            governingLaw: governingSection,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Both contractual consideration (INR ${totalContractualConsideration.toLocaleString('en-IN')}) and Stamp Duty Value (INR ${effectiveSdv.toLocaleString('en-IN')}) are below the statutory threshold of INR 50,00,000 under Section 194-IA(2). Zero TDS required.`
        };
    }

    // Rate Determination
    let applicableTdsRatePercent = 1.0;
    let rateType = 'STANDARD_STATUTORY_RATE';
    let isPenaltyRate = false;

    if (!hasPan || isPanInoperative) {
        applicableTdsRatePercent = 20.0;
        rateType = 'PENALTY_RATE_SECTION_206AA';
        isPenaltyRate = true;
    } else if (lowerDeductionRate !== null && !isNaN(Number(lowerDeductionRate)) && Number(lowerDeductionRate) >= 0) {
        applicableTdsRatePercent = Number(lowerDeductionRate);
        rateType = 'CONCESSIONAL_CERTIFICATE_SECTION_197';
    }

    const calculatedTdsInr = roundToPaisa((taxableBaseInr * applicableTdsRatePercent) / 100);
    const netPayableToSellerInr = roundToPaisa(Math.max(0, totalContractualConsideration - calculatedTdsInr));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DETERMINED_APPLICABLE',
        governingLaw: governingSection,
        effectiveDate,
        withholdingTrigger,
        valuationBasis: {
            contractualConsiderationInr: rawBaseConsideration,
            incidentalChargesInr: rawIncidental,
            totalContractualConsiderationInr: totalContractualConsideration,
            stampDutyValueInr: effectiveSdv,
            higherValuationAdopted: effectiveSdv > totalContractualConsideration ? 'STAMP_DUTY_VALUE' : 'CONTRACTUAL_CONSIDERATION',
            taxableBaseInr
        },
        withholding: {
            applicableTdsRatePercent,
            rateType,
            isPenaltyRate,
            calculatedTdsInr,
            netPayableToSellerInr
        },
        complianceRequirements: {
            challanCumStatementForm: 'Form 26QB (Electronic)',
            challanFilingCutoff: 'Within 30 days from the end of the month in which deduction was made',
            tdsCertificateForm: 'Form 16B (TRACES)',
            certificateIssuanceCutoff: 'Within 15 days from the due date for furnishing Form 26QB',
            tanRequired: false
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Section 194-IA Applicable: Taxable base INR ${taxableBaseInr.toLocaleString('en-IN')} (Higher of consideration INR ${totalContractualConsideration.toLocaleString('en-IN')} vs SDV INR ${effectiveSdv.toLocaleString('en-IN')}) taxed at ${applicableTdsRatePercent}% = INR ${calculatedTdsInr.toLocaleString('en-IN')}. Net payable to seller: INR ${netPayableToSellerInr.toLocaleString('en-IN')}. File Form 26QB within 30 days of month-end.`
    };
}

/**
 * Evaluates Tax Deduction at Source on Benefit or Perquisite in respect of Business or Profession under Section 194R.
 * 
 * Rules Enforced:
 * 1. Payer Turnover Prerequisite:
 *    - Individuals & HUFs: Covered only if preceding FY business turnover > 1 Crore or profession receipts > 50 Lakhs.
 *    - Companies, LLPs, Partnership Firms: Covered irrespective of turnover.
 * 2. Cumulative Financial Year Exemption:
 *    - Threshold: INR 20,000 aggregate in a financial year.
 *    - Value up to INR 20,000: No TDS (EXEMPT_BELOW_THRESHOLD).
 *    - Once aggregate exceeds INR 20,000: TDS applies at 10% standard rate.
 * 3. Specific Exclusions per CBDT Circular No. 12/2022 & 18/2022:
 *    - Sales discount, cash discount, and commercial rebates are excluded.
 *    - Bona fide dealer training/launch conferences without leisure/family overstay are excluded.
 * 4. In-Kind / Non-Cash Benefits (Section 194R(1) Proviso):
 *    - If benefit is wholly in kind or partly in kind, provider must ensure advance tax (ITNS 281) has been paid before release.
 *    - Release without advance tax verification flags 30% disallowance under Section 40(a)(ia).
 * 5. Section 206AA / 206AB Rules:
 *    - 20% floor for missing/inoperative PAN.
 *    - Section 206AB non-filer penalty active pre-1 April 2025; omitted post-1 April 2025 by Finance Act, 2025.
 * 6. Statutory Invariant:
 *    - Always returns isAuthorizedToDisburse: false.
 */
export function evaluateSection194R({
    currentBenefitValueInr,
    cumulativePriorBenefitsInr = 0,
    payerEntityType = 'COMPANY',
    payerPrecedingFyTurnoverInr = null,
    payerPrecedingFyProfessionReceiptsInr = null,
    benefitNature = 'CASH', // 'CASH', 'IN_KIND', 'PARTLY_CASH_PARTLY_KIND'
    benefitCategory = 'OTHER', // 'DEALER_INCENTIVE_TRIP', 'SALES_CONFERENCE_EXCESS', 'CAPITAL_ASSET_TRANSFER', 'PRODUCT_SAMPLE_COMMERCIAL', 'GIFT_VOUCHER_OR_GOLD', 'COMMERCIAL_DISCOUNT_REBATE', 'DEALER_CONFERENCE_BONAFIDE', 'OTHER'
    isAdvanceTaxPaidBeforeRelease = false,
    hasPan = true,
    isPanInoperative = false,
    isSpecifiedPerson206AB = false,
    effectiveDate = null
} = {}) {
    const benefitVal = roundToPaisa(Math.max(0, assertMandatoryNumeric(currentBenefitValueInr, 'currentBenefitValueInr', 'evaluateSection194R')));
    const priorVal = roundToPaisa(Math.max(0, cleanNumericAmount(cumulativePriorBenefitsInr)));
    const totalCumulativeVal = roundToPaisa(priorVal + benefitVal);
    const dateStr = effectiveDate ? String(effectiveDate).slice(0, 10) : new Date().toISOString().slice(0, 10);

    // Exclusion 1: CBDT Circular 12/2022 & 18/2022 Non-Perquisite Categories
    if (benefitCategory === 'COMMERCIAL_DISCOUNT_REBATE' || benefitCategory === 'DEALER_CONFERENCE_BONAFIDE') {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXCLUDED_CBDT_CIRCULAR_12_2022',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            benefitValueInr: benefitVal,
            cumulativePriorBenefitsInr: priorVal,
            totalCumulativeInr: totalCumulativeVal,
            disallowanceRiskSection40a_ia: false,
            governingLaw: 'Income-tax Act, 1961 Section 194R read with CBDT Circular No. 12/2022 & Circular No. 18/2022',
            isAuthorizedToDisburse: false,
            statutoryRationale: `Benefit category "${benefitCategory}" is explicitly excluded from Section 194R per CBDT Circular No. 12/2022. Sales discounts, commercial rebates, and bona fide dealer product conferences do not constitute taxable perquisites.`
        };
    }

    // Exclusion 2: Individual / HUF Payer Turnover Floor
    const normalizedEntityType = String(payerEntityType || 'COMPANY').toUpperCase();
    if (normalizedEntityType === 'INDIVIDUAL' || normalizedEntityType === 'HUF' || normalizedEntityType === 'INDIVIDUAL_HUF') {
        const turnover = cleanNumericAmount(payerPrecedingFyTurnoverInr);
        const profReceipts = cleanNumericAmount(payerPrecedingFyProfessionReceiptsInr);
        const exceedsBusinessFloor = turnover > 10000000; // > 1 Crore
        const exceedsProfessionFloor = profReceipts > 5000000; // > 50 Lakhs

        if (!exceedsBusinessFloor && !exceedsProfessionFloor) {
            return {
                status: 'SUCCESS',
                statutoryDetermination: 'EXEMPT_PAYER_TURNOVER_BELOW_STATUTORY_FLOOR',
                applicableTdsRatePercent: 0,
                taxableBaseInr: 0,
                calculatedTdsInr: 0,
                benefitValueInr: benefitVal,
                cumulativePriorBenefitsInr: priorVal,
                totalCumulativeInr: totalCumulativeVal,
                disallowanceRiskSection40a_ia: false,
                governingLaw: 'Income-tax Act, 1961 Section 194R(1) Second Proviso',
                isAuthorizedToDisburse: false,
                statutoryRationale: 'Payer is an Individual/HUF whose business turnover (INR ' + turnover.toLocaleString('en-IN') + ') and professional receipts (INR ' + profReceipts.toLocaleString('en-IN') + ') in preceding FY do not exceed the statutory threshold (> INR 1 Cr business / > INR 50L profession). Section 194R withholding does not apply.'
            };
        }
    }

    // Exclusion 3: Cumulative Financial Year De Minimis Ceiling (<= INR 20,000)
    const STATUTORY_THRESHOLD_INR = 20000;
    if (totalCumulativeVal <= STATUTORY_THRESHOLD_INR) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            benefitValueInr: benefitVal,
            cumulativePriorBenefitsInr: priorVal,
            totalCumulativeInr: totalCumulativeVal,
            thresholdCeilingInr: STATUTORY_THRESHOLD_INR,
            disallowanceRiskSection40a_ia: false,
            governingLaw: 'Income-tax Act, 1961 Section 194R(1) First Proviso',
            isAuthorizedToDisburse: false,
            statutoryRationale: `Cumulative value of perquisites in financial year (INR ${totalCumulativeVal.toLocaleString('en-IN')}) does not exceed the statutory de minimis limit of INR 20,000. No tax deduction at source is required.`
        };
    }

    // Rate Determination
    let applicableRate = 10.0;
    let rateLaw = 'Section 194R(1) standard rate (10%)';

    // Section 206AA: Missing or inoperative PAN
    if (!hasPan || isPanInoperative) {
        applicableRate = 20.0;
        rateLaw = 'Section 206AA penal rate (20%) due to missing or inoperative PAN';
    } else if (isSpecifiedPerson206AB) {
        // Section 206AB: Non-filer higher rate (omitted w.e.f. 1 April 2025)
        if (dateStr < '2025-04-01') {
            applicableRate = 20.0;
            rateLaw = 'Section 206AB higher rate (20% - twice standard rate) for non-filer prior to 1 April 2025';
        } else {
            applicableRate = 10.0;
            rateLaw = 'Section 194R(1) standard rate (10%) - Section 206AB omitted w.e.f. 1 April 2025';
        }
    }

    const calculatedTds = roundToPaisa(benefitVal * (applicableRate / 100));

    // In-Kind Compliance Verification (Section 194R(1) Proviso)
    const normalizedNature = String(benefitNature || 'CASH').toUpperCase();
    const isInKind = normalizedNature === 'IN_KIND' || normalizedNature === 'PARTLY_CASH_PARTLY_KIND';

    if (isInKind && !isAdvanceTaxPaidBeforeRelease) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'MANDATORY_IN_KIND_ADVANCE_TAX_PENDING',
            applicableTdsRatePercent: applicableRate,
            taxableBaseInr: benefitVal,
            calculatedTdsInr: calculatedTds,
            benefitValueInr: benefitVal,
            cumulativePriorBenefitsInr: priorVal,
            totalCumulativeInr: totalCumulativeVal,
            benefitNature: normalizedNature,
            isAdvanceTaxPaidBeforeRelease: false,
            disallowanceRiskSection40a_ia: true,
            disallowanceAmountInr: roundToPaisa(benefitVal * 0.30),
            governingLaw: `${dateStr >= '2026-04-01' ? 'Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 8(iv)' : 'Income-tax Act, 1961 Section 194R(1)'} read with ${rateLaw}`,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Section 194R(1) Proviso mandates that where perquisite is wholly or partly in kind, the provider must ensure tax has been paid in advance before releasing the benefit. Releasing INR ${benefitVal.toLocaleString('en-IN')} benefit in kind without verified advance tax payment creates an immediate 30% expenditure disallowance risk (INR ${roundToPaisa(benefitVal * 0.30).toLocaleString('en-IN')}) under Section 40(a)(ia).`
        };
    }

    return {
        status: 'SUCCESS',
        statutoryDetermination: isInKind ? 'DEDUCTION_MANDATORY_IN_KIND_DISCHARGED' : 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: benefitVal,
        calculatedTdsInr: calculatedTds,
        benefitValueInr: benefitVal,
        cumulativePriorBenefitsInr: priorVal,
        totalCumulativeInr: totalCumulativeVal,
        benefitNature: normalizedNature,
        isAdvanceTaxPaidBeforeRelease: isInKind ? true : null,
        disallowanceRiskSection40a_ia: false,
        governingLaw: `${dateStr >= '2026-04-01' ? 'Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 8(iv)' : 'Income-tax Act, 1961 Section 194R(1)'} read with ${rateLaw}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Cumulative benefits (INR ${totalCumulativeVal.toLocaleString('en-IN')}) exceed statutory threshold of INR 20,000. TDS of INR ${calculatedTds.toLocaleString('en-IN')} (rate: ${applicableRate}%) applies under Section 194R.`
    };
}

/**
 * Evaluates Tax Deduction at Source on Payment of Rent by Certain Individuals or HUF under Section 194-IB.
 * 
 * Rules Enforced:
 * 1. Payer Qualification:
 *    - Individuals and HUFs not liable to tax audit under Section 44AB.
 * 2. Monthly Rent Threshold:
 *    - Rent exceeding INR 50,000 for a month or part of a month in the previous year.
 *    - Rent <= INR 50,000/month: EXEMPT_BELOW_MONTHLY_THRESHOLD.
 * 3. Rate Rationalization (Finance (No. 2) Act, 2024):
 *    - Transactions prior to 1 October 2024: 5.0%.
 *    - Transactions on or after 1 October 2024: 2.0%.
 * 4. Section 206AA Penalty & Statutory Ceiling (Section 194-IB(4)):
 *    - 20.0% penalty rate if landlord has missing/inoperative PAN.
 *    - Mandatory Statutory Cap: Deduction under Section 206AA shall NOT exceed the rent payable for the last month of the tenancy.
 * 5. Compliance Return:
 *    - Form 26QC challan-cum-statement within 30 days of month-end; Form 16C certificate.
 * 6. Statutory Invariant:
 *    - Always returns isAuthorizedToDisburse: false.
 */
export function evaluateSection194IB({
    monthlyRentInr,
    tenancyMonthsInYear = 1,
    lastMonthRentPayableInr = null,
    isPayerTaxAuditCovered = false,
    hasLandlordPan = true,
    isLandlordPanInoperative = false,
    effectiveDate = null
} = {}) {
    const monthRent = roundToPaisa(Math.max(0, assertMandatoryNumeric(monthlyRentInr, 'monthlyRentInr', 'evaluateSection194IB')));
    const months = Math.max(1, Math.min(12, Number(tenancyMonthsInYear) || 1));
    const totalRentForPeriod = roundToPaisa(monthRent * months);
    const lastMonthRent = lastMonthRentPayableInr !== null && lastMonthRentPayableInr !== undefined
        ? roundToPaisa(Math.max(0, cleanNumericAmount(lastMonthRentPayableInr)))
        : monthRent;
    const dateStr = effectiveDate ? String(effectiveDate).slice(0, 10) : new Date().toISOString().slice(0, 10);

    // Condition 1: Payer covered under Section 194-I (Tax audit covered)
    if (isPayerTaxAuditCovered) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'GOVERNED_BY_SECTION_194I',
            applicableTdsRatePercent: null,
            taxableBaseInr: null,
            calculatedTdsInr: null,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194-I (Commercial / Audit Payer Rent)',
            statutoryRationale: 'Payer is subject to tax audit under Section 44AB. Rent payments are governed by Section 194-I (2% machinery / 10% land & building) with Form 26Q quarterly reporting, rather than Section 194-IB.'
        };
    }

    // Condition 2: Monthly Rent Threshold (<= INR 50,000 per month)
    const STATUTORY_MONTHLY_THRESHOLD_INR = 50000;
    if (monthRent <= STATUTORY_MONTHLY_THRESHOLD_INR) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_BELOW_MONTHLY_THRESHOLD',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            monthlyRentInr: monthRent,
            tenancyMonthsInYear: months,
            totalRentInr: totalRentForPeriod,
            monthlyThresholdCeilingInr: STATUTORY_MONTHLY_THRESHOLD_INR,
            netPayableToLandlordInr: totalRentForPeriod,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194-IB(1)',
            statutoryRationale: `Monthly rent of INR ${monthRent.toLocaleString('en-IN')} does not exceed the statutory threshold of INR 50,000 per month. No tax deduction at source is required.`
        };
    }

    // Rate Determination
    const isPostRateCut = dateStr >= '2024-10-01';
    let applicableRate = isPostRateCut ? 2.0 : 5.0;
    let rateDescription = isPostRateCut
        ? 'Section 194-IB amended rate (2% w.e.f. 1 October 2024 by Finance (No. 2) Act, 2024)'
        : 'Section 194-IB historical rate (5% prior to 1 October 2024)';
    let isCappedByLastMonthRent = false;

    // Penalty Rate Check (Section 206AA)
    if (!hasLandlordPan || isLandlordPanInoperative) {
        applicableRate = 20.0;
        rateDescription = 'Section 206AA penal rate (20%) due to missing or inoperative Landlord PAN';
    }

    const rawTdsAmount = roundToPaisa((totalRentForPeriod * applicableRate) / 100);
    let finalTdsAmount = rawTdsAmount;

    // Section 194-IB(4) Statutory Ceiling for Section 206AA
    if ((!hasLandlordPan || isLandlordPanInoperative) && rawTdsAmount > lastMonthRent) {
        finalTdsAmount = lastMonthRent;
        isCappedByLastMonthRent = true;
    }

    const netPayable = roundToPaisa(Math.max(0, totalRentForPeriod - finalTdsAmount));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: totalRentForPeriod,
        calculatedTdsInr: finalTdsAmount,
        uncappedTdsInr: rawTdsAmount,
        monthlyRentInr: monthRent,
        tenancyMonthsInYear: months,
        totalRentInr: totalRentForPeriod,
        lastMonthRentCeilingInr: lastMonthRent,
        isCappedByLastMonthRent,
        netPayableToLandlordInr: netPayable,
        complianceRequirements: {
            challanCumStatementForm: 'Form 26QC (Electronic)',
            challanFilingCutoff: 'Within 30 days from the end of the month in which deduction was made',
            tdsCertificateForm: 'Form 16C (TRACES)',
            certificateIssuanceCutoff: 'Within 15 days from the due date for furnishing Form 26QC',
            tanRequired: false
        },
        governingLaw: `${dateStr >= '2026-04-01' ? 'Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 2(iii)' : 'Income-tax Act, 1961 Section 194-IB'} read with ${rateDescription}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: isCappedByLastMonthRent
            ? `Section 194-IB Applicable: 20% Section 206AA penalty on INR ${totalRentForPeriod.toLocaleString('en-IN')} (INR ${rawTdsAmount.toLocaleString('en-IN')}) was legally capped at INR ${lastMonthRent.toLocaleString('en-IN')} under Section 194-IB(4) (rent payable for last month of tenancy). Net payable: INR ${netPayable.toLocaleString('en-IN')}.`
            : `Section 194-IB Applicable: Total rent INR ${totalRentForPeriod.toLocaleString('en-IN')} for ${months} months taxed at ${applicableRate}% = INR ${finalTdsAmount.toLocaleString('en-IN')}. Net payable: INR ${netPayable.toLocaleString('en-IN')}. File Form 26QC within 30 days of month-end.`
    };
}

/**
 * Evaluates Tax Deduction at Source on Payments to Contractors/Professionals by Certain Individuals or HUF under Section 194M.
 * 
 * Rules Enforced:
 * 1. Payer Qualification:
 *    - Individuals and HUFs not liable to deduct TDS under Section 194C, 194H, or 194J.
 * 2. Cumulative Financial Year Threshold:
 *    - Aggregate sum exceeding INR 50,00,000 in a financial year.
 *    - Aggregate <= INR 50 Lakhs: EXEMPT_BELOW_THRESHOLD.
 * 3. Rate Rationalization (Finance (No. 2) Act, 2024):
 *    - Transactions prior to 1 October 2024: 5.0%.
 *    - Transactions on or after 1 October 2024: 2.0%.
 * 4. Section 206AA / 206AB Penalties:
 *    - 20.0% penalty for missing/inoperative PAN.
 *    - Section 206AB omitted w.e.f. 1 April 2025 by Finance Act, 2025.
 * 5. Compliance Return:
 *    - Form 26QD challan-cum-statement within 30 days of month-end; Form 16D certificate. No TAN required.
 * 6. Statutory Invariant:
 *    - Always returns isAuthorizedToDisburse: false.
 */
export function evaluateSection194M({
    currentPaymentAmountInr,
    cumulativePriorPaymentsInr = 0,
    isPayerCoveredBy194C_H_J = false,
    hasPan = true,
    isPanInoperative = false,
    isSpecifiedPerson206AB = false,
    effectiveDate = null
} = {}) {
    const currentAmt = roundToPaisa(Math.max(0, assertMandatoryNumeric(currentPaymentAmountInr, 'currentPaymentAmountInr', 'evaluateSection194M')));
    const priorAmt = roundToPaisa(Math.max(0, cleanNumericAmount(cumulativePriorPaymentsInr)));
    const totalCumulative = roundToPaisa(priorAmt + currentAmt);
    const dateStr = effectiveDate ? String(effectiveDate).slice(0, 10) : new Date().toISOString().slice(0, 10);

    // Condition 1: Covered by standard 194C, 194H, or 194J
    if (isPayerCoveredBy194C_H_J) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'GOVERNED_BY_PRIMARY_SECTIONS',
            applicableTdsRatePercent: null,
            taxableBaseInr: null,
            calculatedTdsInr: null,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194C, 194H, or 194J',
            statutoryRationale: 'Payer is covered under tax audit or entity thresholds of Section 194C, 194H, or 194J. Payments must be routed through standard quarterly Form 26Q reporting with TAN, not Section 194M.'
        };
    }

    // Condition 2: Cumulative Annual Threshold (<= INR 50,00,000)
    const STATUTORY_ANNUAL_THRESHOLD_INR = 5000000;
    if (totalCumulative <= STATUTORY_ANNUAL_THRESHOLD_INR) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            currentPaymentInr: currentAmt,
            cumulativePriorPaymentsInr: priorAmt,
            totalCumulativeInr: totalCumulative,
            thresholdCeilingInr: STATUTORY_ANNUAL_THRESHOLD_INR,
            netPayableToPayeeInr: currentAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194M(1)',
            statutoryRationale: `Cumulative payments in financial year (INR ${totalCumulative.toLocaleString('en-IN')}) do not exceed statutory threshold of INR 50 Lakhs. No tax deduction is required.`
        };
    }

    // Rate Determination
    const isPostRateCut = dateStr >= '2024-10-01';
    let applicableRate = isPostRateCut ? 2.0 : 5.0;
    let rateDescription = isPostRateCut
        ? 'Section 194M amended rate (2% w.e.f. 1 October 2024 by Finance (No. 2) Act, 2024)'
        : 'Section 194M historical rate (5% prior to 1 October 2024)';

    // Section 206AA: Missing or inoperative PAN
    if (!hasPan || isPanInoperative) {
        applicableRate = 20.0;
        rateDescription = 'Section 206AA penal rate (20%) due to missing or inoperative Payee PAN';
    } else if (isSpecifiedPerson206AB) {
        // Section 206AB: Non-filer higher rate (omitted w.e.f. 1 April 2025)
        if (dateStr < '2025-04-01') {
            applicableRate = 20.0;
            rateDescription = 'Section 206AB higher rate (20%) for non-filer prior to 1 April 2025';
        } else {
            applicableRate = isPostRateCut ? 2.0 : 5.0;
            rateDescription = `${rateDescription} - Section 206AB omitted w.e.f. 1 April 2025`;
        }
    }

    const calculatedTds = roundToPaisa((currentAmt * applicableRate) / 100);
    const netPayable = roundToPaisa(Math.max(0, currentAmt - calculatedTds));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: currentAmt,
        calculatedTdsInr: calculatedTds,
        currentPaymentInr: currentAmt,
        cumulativePriorPaymentsInr: priorAmt,
        totalCumulativeInr: totalCumulative,
        netPayableToPayeeInr: netPayable,
        complianceRequirements: {
            challanCumStatementForm: 'Form 26QD (Electronic)',
            challanFilingCutoff: 'Within 30 days from the end of the month in which deduction was made',
            tdsCertificateForm: 'Form 16D (TRACES)',
            certificateIssuanceCutoff: 'Within 15 days from the due date for furnishing Form 26QD',
            tanRequired: false
        },
        governingLaw: `${dateStr >= '2026-04-01' ? 'Income-tax Act, 2025 Section 393(1), Table 1, Sl. No. 6(i) / 6(iii) / 1(ii)' : 'Income-tax Act, 1961 Section 194M'} read with ${rateDescription}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Cumulative payments (INR ${totalCumulative.toLocaleString('en-IN')}) exceed statutory threshold of INR 50 Lakhs. Current payment of INR ${currentAmt.toLocaleString('en-IN')} taxed at ${applicableRate}% = INR ${calculatedTds.toLocaleString('en-IN')}. Net payable: INR ${netPayable.toLocaleString('en-IN')}. File Form 26QD within 30 days of month-end.`
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: Section 194S (TDS on Virtual Digital Assets / Crypto)
 * Applicable on payment of consideration to a resident for transfer of a virtual digital asset (VDA).
 * Standard Rate: 1.0% (100 bps).
 * Thresholds:
 * - Specified person (Individual/HUF without business turnover > INR 1Cr or professional gross receipts > INR 50L): INR 50,000 in FY.
 * - Any other person: INR 10,000 in FY.
 * Penal Rates:
 * - Missing PAN (Sec 206AA): 20.0%.
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection194S({
    considerationInr,
    cumulativePriorConsiderationInr = 0,
    isSpecifiedPerson = false,
    hasPan = true,
    isPanInoperative = false,
    isSellerResident = true,
    transactionDate = null,
    creditDate = null,
    paymentDate = null
} = {}) {
    const rawAmt = Math.max(0, assertMandatoryNumeric(considerationInr, 'considerationInr', 'evaluateSection194S'));
    const priorAmt = Math.max(0, cleanNumericAmount(cumulativePriorConsiderationInr));
    const totalCumulative = roundToPaisa(priorAmt + rawAmt);

    // Non-resident seller -> governed by Section 195, not 194S
    if (!isSellerResident) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'GOVERNED_BY_SECTION_195',
            applicableTdsRatePercent: null,
            taxableBaseInr: null,
            calculatedTdsInr: null,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 195 (Cross-Border)',
            statutoryRationale: 'Transferor is a non-resident. Withholding is governed by Section 195 DTAA framework, not Section 194S.'
        };
    }

    const thresholdCeiling = isSpecifiedPerson ? 50000 : 10000;

    // Threshold gate check
    if (totalCumulative <= thresholdCeiling) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            currentPaymentInr: rawAmt,
            cumulativePriorPaymentsInr: priorAmt,
            totalCumulativeInr: totalCumulative,
            thresholdCeilingInr: thresholdCeiling,
            netPayableToPayeeInr: rawAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194S(1)',
            statutoryRationale: `Cumulative consideration (INR ${totalCumulative.toLocaleString('en-IN')}) does not exceed statutory threshold of INR ${thresholdCeiling.toLocaleString('en-IN')}. No tax deduction required.`
        };
    }

    // Determine taxable base: if prior was below threshold, tax the excess over threshold; if prior already exceeded, tax entire current
    let taxableBase = 0;
    if (priorAmt < thresholdCeiling) {
        taxableBase = roundToPaisa(totalCumulative - thresholdCeiling);
    } else {
        taxableBase = rawAmt;
    }

    let applicableRate = 1.0;
    let rateDescription = 'Section 194S standard rate (1%)';

    if (!hasPan || isPanInoperative) {
        applicableRate = 20.0;
        rateDescription = 'Section 206AA penal rate (20%) due to missing or inoperative PAN';
    }

    const calculatedTds = roundToPaisa((taxableBase * applicableRate) / 100);
    const netPayable = roundToPaisa(Math.max(0, rawAmt - calculatedTds));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: taxableBase,
        calculatedTdsInr: calculatedTds,
        currentPaymentInr: rawAmt,
        cumulativePriorPaymentsInr: priorAmt,
        totalCumulativeInr: totalCumulative,
        thresholdCeilingInr: thresholdCeiling,
        netPayableToPayeeInr: netPayable,
        complianceRequirements: {
            challanCumStatementForm: isSpecifiedPerson ? 'Form 26QE' : 'Form 26Q',
            tdsCertificateForm: isSpecifiedPerson ? 'Form 16E' : 'Form 16A',
            challanFilingCutoff: 'Within 30 days from end of month'
        },
        governingLaw: `Income-tax Act, 1961 Section 194S read with ${rateDescription}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `VDA consideration exceeds INR ${thresholdCeiling.toLocaleString('en-IN')}. Taxable base INR ${taxableBase.toLocaleString('en-IN')} taxed at ${applicableRate}% = INR ${calculatedTds.toLocaleString('en-IN')}. Net payable: INR ${netPayable.toLocaleString('en-IN')}.`
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: Section 194BA (TDS on Winnings from Online Games)
 * Effective 1 April 2023 (Finance Act, 2023).
 * Standard Rate: 30.0% (3000 bps) on net winnings.
 * Threshold: No minimum monetary threshold.
 * Net Winnings Formula: (Withdrawals + Closing Balance) - (Deposits + Opening Balance).
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection194BA({
    netWinningsInr = null,
    withdrawalAmountInr = null,
    totalDepositsInr = 0,
    openingBalanceInr = 0,
    closingBalanceInr = 0,
    totalPriorWithdrawalsInr = 0,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null
} = {}) {
    let computedNetWinnings = 0;

    if (netWinningsInr !== null && netWinningsInr !== undefined) {
        computedNetWinnings = cleanNumericAmount(netWinningsInr);
    } else {
        const withdrawal = Math.max(0, cleanNumericAmount(withdrawalAmountInr));
        const priorWithdrawals = Math.max(0, cleanNumericAmount(totalPriorWithdrawalsInr));
        const closing = Math.max(0, cleanNumericAmount(closingBalanceInr));
        const deposits = Math.max(0, cleanNumericAmount(totalDepositsInr));
        const opening = Math.max(0, cleanNumericAmount(openingBalanceInr));

        computedNetWinnings = roundToPaisa((withdrawal + priorWithdrawals + closing) - (deposits + opening));
    }

    if (computedNetWinnings <= 0) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_ZERO_OR_NEGATIVE_NET_WINNINGS',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            netWinningsInr: computedNetWinnings,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194BA read with Rule 133',
            statutoryRationale: 'Net winnings are zero or negative. No tax deduction is applicable.'
        };
    }

    let applicableRate = 30.0;
    let rateDescription = 'Section 194BA statutory rate (30% on net online gaming winnings)';

    if (!hasPan || isPanInoperative) {
        applicableRate = 30.0; // Max of 20% and 30% -> 30%
        rateDescription = 'Section 194BA/206AA statutory rate (30% applies regardless of PAN)';
    }

    const calculatedTds = roundToPaisa((computedNetWinnings * applicableRate) / 100);
    const netPayout = roundToPaisa(Math.max(0, computedNetWinnings - calculatedTds));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: computedNetWinnings,
        calculatedTdsInr: calculatedTds,
        netWinningsInr: computedNetWinnings,
        netPayableToUserInr: netPayout,
        complianceRequirements: {
            statementForm: 'Form 26Q (Quarterly)',
            tdsCertificateForm: 'Form 16A',
            remittanceCutoff: '7th of subsequent month (30th April for March)'
        },
        governingLaw: `Income-tax Act, 1961 Section 194BA read with ${rateDescription}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Net winnings of INR ${computedNetWinnings.toLocaleString('en-IN')} taxed at ${applicableRate}% = INR ${calculatedTds.toLocaleString('en-IN')}. Net payout: INR ${netPayout.toLocaleString('en-IN')}.`
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: Section 194N (TDS on High-Value Cash Withdrawals)
 * Effective 1 September 2019.
 * Normal Filer: 2% on aggregate cash withdrawals in excess of INR 1 Crore.
 * Non-Filer (no ITR in past 3 assessment years):
 * - 2% on cash withdrawal exceeding INR 20 Lakhs up to INR 1 Crore.
 * - 5% on cash withdrawal exceeding INR 1 Crore.
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection194N({
    withdrawalAmountInr,
    cumulativePriorWithdrawalsInr = 0,
    isNonFilerPast3Years = false,
    isExemptEntity = false,
    transactionDate = null
} = {}) {
    const rawAmt = Math.max(0, assertMandatoryNumeric(withdrawalAmountInr, 'withdrawalAmountInr', 'evaluateSection194N'));
    const priorAmt = Math.max(0, cleanNumericAmount(cumulativePriorWithdrawalsInr));
    const totalCumulative = roundToPaisa(priorAmt + rawAmt);

    if (isExemptEntity) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_ENTITY_FOURTH_PROVISO',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194N Fourth Proviso',
            statutoryRationale: 'Entity is exempt from Section 194N (Government, banking company, co-operative society, business correspondent, white-label ATM operator).'
        };
    }

    const ONE_CRORE = 10000000;
    const TWENTY_LAKHS = 2000000;

    let calculatedTds = 0;
    let applicableRate = 0;
    let taxableBase = 0;
    let rateDescription = '';

    if (!isNonFilerPast3Years) {
        // Normal Filer: 2% on excess over 1 Crore
        if (totalCumulative <= ONE_CRORE) {
            return {
                status: 'SUCCESS',
                statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD',
                applicableTdsRatePercent: 0,
                taxableBaseInr: 0,
                calculatedTdsInr: 0,
                currentWithdrawalInr: rawAmt,
                cumulativePriorWithdrawalsInr: priorAmt,
                totalCumulativeInr: totalCumulative,
                thresholdCeilingInr: ONE_CRORE,
                isAuthorizedToDisburse: false,
                governingLaw: 'Income-tax Act, 1961 Section 194N(1)',
                statutoryRationale: `Cumulative cash withdrawals (INR ${totalCumulative.toLocaleString('en-IN')}) do not exceed INR 1 Crore. No TDS required.`
            };
        }

        taxableBase = priorAmt >= ONE_CRORE ? rawAmt : roundToPaisa(totalCumulative - ONE_CRORE);
        applicableRate = 2.0;
        calculatedTds = roundToPaisa((taxableBase * applicableRate) / 100);
        rateDescription = '2% on cash withdrawal in excess of INR 1 Crore (First Proviso)';
    } else {
        // Non-Filer: 2% between 20L and 1Cr, 5% over 1Cr
        if (totalCumulative <= TWENTY_LAKHS) {
            return {
                status: 'SUCCESS',
                statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD_NON_FILER',
                applicableTdsRatePercent: 0,
                taxableBaseInr: 0,
                calculatedTdsInr: 0,
                currentWithdrawalInr: rawAmt,
                cumulativePriorWithdrawalsInr: priorAmt,
                totalCumulativeInr: totalCumulative,
                thresholdCeilingInr: TWENTY_LAKHS,
                isAuthorizedToDisburse: false,
                governingLaw: 'Income-tax Act, 1961 Section 194N Second Proviso',
                statutoryRationale: `Cumulative cash withdrawals for non-filer (INR ${totalCumulative.toLocaleString('en-IN')}) do not exceed INR 20 Lakhs. No TDS required.`
            };
        }

        // Tiered calculation
        let tier1Taxable = 0;
        let tier2Taxable = 0;

        const effectiveStart = priorAmt;
        const effectiveEnd = totalCumulative;

        if (effectiveEnd > TWENTY_LAKHS) {
            const start1 = Math.max(TWENTY_LAKHS, effectiveStart);
            const end1 = Math.min(ONE_CRORE, effectiveEnd);
            if (end1 > start1) tier1Taxable = end1 - start1;
        }

        if (effectiveEnd > ONE_CRORE) {
            const start2 = Math.max(ONE_CRORE, effectiveStart);
            const end2 = effectiveEnd;
            if (end2 > start2) tier2Taxable = end2 - start2;
        }

        const tdsTier1 = roundToPaisa((tier1Taxable * 2.0) / 100);
        const tdsTier2 = roundToPaisa((tier2Taxable * 5.0) / 100);
        calculatedTds = roundToPaisa(tdsTier1 + tdsTier2);
        taxableBase = roundToPaisa(tier1Taxable + tier2Taxable);
        applicableRate = tier2Taxable > 0 ? (tier1Taxable > 0 ? 5.0 : 5.0) : 2.0;
        rateDescription = `Non-filer tiered rates: 2% on INR ${tier1Taxable.toLocaleString('en-IN')} (20L-1Cr) + 5% on INR ${tier2Taxable.toLocaleString('en-IN')} (>1Cr)`;
    }

    const netCashDisbursed = roundToPaisa(Math.max(0, rawAmt - calculatedTds));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: taxableBase,
        calculatedTdsInr: calculatedTds,
        currentWithdrawalInr: rawAmt,
        cumulativePriorWithdrawalsInr: priorAmt,
        totalCumulativeInr: totalCumulative,
        netCashDisbursedInr: netCashDisbursed,
        governingLaw: `Income-tax Act, 1961 Section 194N (${rateDescription})`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Cumulative withdrawals (INR ${totalCumulative.toLocaleString('en-IN')}) exceed statutory threshold. TDS of INR ${calculatedTds.toLocaleString('en-IN')} deducted. Net cash disbursed: INR ${netCashDisbursed.toLocaleString('en-IN')}.`
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: Section 194DA (TDS on Life Insurance Policy Payouts)
 * Standard Rate: 5% on the income portion (maturity amount minus total premiums paid).
 * Exemption:
 * - Exempt under Section 10(10D) -> 0% TDS.
 * - Aggregate payout in FY < INR 1,00,000 -> 0% TDS.
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection194DA({
    payoutAmountInr,
    totalPremiumsPaidInr = 0,
    cumulativePriorPayoutsInr = 0,
    isExemptUnderSection10_10D = false,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null
} = {}) {
    const rawAmt = Math.max(0, assertMandatoryNumeric(payoutAmountInr, 'payoutAmountInr', 'evaluateSection194DA'));
    const premiumsPaid = Math.max(0, cleanNumericAmount(totalPremiumsPaidInr));
    const priorAmt = Math.max(0, cleanNumericAmount(cumulativePriorPayoutsInr));
    const totalCumulative = roundToPaisa(priorAmt + rawAmt);

    if (isExemptUnderSection10_10D) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_SECTION_10_10D',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            netPayableInr: rawAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 10(10D)',
            statutoryRationale: 'Life insurance payout is fully exempt under Section 10(10D). No tax deduction is applicable.'
        };
    }

    const THRESHOLD = 100000;
    if (totalCumulative < THRESHOLD) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            currentPayoutInr: rawAmt,
            totalCumulativeInr: totalCumulative,
            thresholdCeilingInr: THRESHOLD,
            netPayableInr: rawAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194DA Proviso',
            statutoryRationale: `Aggregate life insurance payout in financial year (INR ${totalCumulative.toLocaleString('en-IN')}) is below INR 1 Lakh. No TDS required.`
        };
    }

    const netIncome = Math.max(0, roundToPaisa(rawAmt - premiumsPaid));
    if (netIncome <= 0) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_NO_INCOME_COMPONENT',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            netPayableInr: rawAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194DA Explanation',
            statutoryRationale: 'Premiums paid equal or exceed payout amount. No net income component exists.'
        };
    }

    let applicableRate = 5.0;
    let rateDescription = 'Section 194DA standard rate (5% on net income component)';

    if (!hasPan || isPanInoperative) {
        applicableRate = 20.0;
        rateDescription = 'Section 206AA penal rate (20% on net income component due to missing/inoperative PAN)';
    }

    const calculatedTds = roundToPaisa((netIncome * applicableRate) / 100);
    const netPayable = roundToPaisa(Math.max(0, rawAmt - calculatedTds));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: netIncome,
        calculatedTdsInr: calculatedTds,
        grossPayoutInr: rawAmt,
        premiumsPaidInr: premiumsPaid,
        netPayableInr: netPayable,
        governingLaw: `Income-tax Act, 1961 Section 194DA read with ${rateDescription}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Non-exempt insurance payout of INR ${rawAmt.toLocaleString('en-IN')} has net income component of INR ${netIncome.toLocaleString('en-IN')}. Taxed at ${applicableRate}% = INR ${calculatedTds.toLocaleString('en-IN')}. Net payable: INR ${netPayable.toLocaleString('en-IN')}.`
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: Section 194K (TDS on Mutual Fund Income)
 * Effective 1 April 2020.
 * Standard Rate: 10% (1000 bps).
 * Threshold: INR 5,000 aggregate income in FY.
 * Exclusion: Capital gains arising from redemption/sale of units are excluded.
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection194K({
    incomeAmountInr,
    cumulativePriorIncomeInr = 0,
    isCapitalGainDistribution = false,
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null
} = {}) {
    const rawAmt = Math.max(0, assertMandatoryNumeric(incomeAmountInr, 'incomeAmountInr', 'evaluateSection194K'));
    const priorAmt = Math.max(0, cleanNumericAmount(cumulativePriorIncomeInr));
    const totalCumulative = roundToPaisa(priorAmt + rawAmt);

    if (isCapitalGainDistribution) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_CAPITAL_GAINS_EXCLUSION',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            netPayableInr: rawAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194K First Proviso',
            statutoryRationale: 'Section 194K excludes capital gains from transfer/redemption of mutual fund units. No tax deduction under this section.'
        };
    }

    const THRESHOLD = 5000;
    if (totalCumulative <= THRESHOLD) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_BELOW_THRESHOLD',
            applicableTdsRatePercent: 0,
            taxableBaseInr: 0,
            calculatedTdsInr: 0,
            currentIncomeInr: rawAmt,
            totalCumulativeInr: totalCumulative,
            thresholdCeilingInr: THRESHOLD,
            netPayableInr: rawAmt,
            isAuthorizedToDisburse: false,
            governingLaw: 'Income-tax Act, 1961 Section 194K Second Proviso',
            statutoryRationale: `Cumulative mutual fund income in financial year (INR ${totalCumulative.toLocaleString('en-IN')}) does not exceed statutory ceiling of INR 5,000. No TDS required.`
        };
    }

    let applicableRate = 10.0;
    let rateDescription = 'Section 194K standard rate (10%)';

    if (!hasPan || isPanInoperative) {
        applicableRate = 20.0;
        rateDescription = 'Section 206AA penal rate (20% due to missing/inoperative PAN)';
    }

    const calculatedTds = roundToPaisa((rawAmt * applicableRate) / 100);
    const netPayable = roundToPaisa(Math.max(0, rawAmt - calculatedTds));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'DEDUCTION_MANDATORY',
        applicableTdsRatePercent: applicableRate,
        taxableBaseInr: rawAmt,
        calculatedTdsInr: calculatedTds,
        grossIncomeInr: rawAmt,
        netPayableInr: netPayable,
        governingLaw: `Income-tax Act, 1961 Section 194K read with ${rateDescription}`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Mutual fund income exceeds INR 5,000 ceiling. Current distribution of INR ${rawAmt.toLocaleString('en-IN')} taxed at ${applicableRate}% = INR ${calculatedTds.toLocaleString('en-IN')}. Net payable: INR ${netPayable.toLocaleString('en-IN')}.`
    };
}

/**
 * DETERMINISTIC STATUTORY CALCULATOR: Section 206C(1G) (TCS on LRS Foreign Remittance and Overseas Tour Packages)
 * Effective 1 October 2020; amended 1 October 2023.
 * Rates:
 * - Overseas Tour Package: 5% up to INR 7 Lakhs, 20% on excess over INR 7 Lakhs.
 * - Education Loan (Sec 80E): 0% up to INR 7 Lakhs, 0.5% on excess over INR 7 Lakhs.
 * - Education or Medical (Other): 0% up to INR 7 Lakhs, 5% on excess over INR 7 Lakhs.
 * - Other LRS Remittances: 0% up to INR 7 Lakhs, 20% on excess over INR 7 Lakhs.
 * Statutory Invariant: isAuthorizedToDisburse === false.
 */
export function evaluateSection206C1G({
    remittanceAmountInr,
    cumulativePriorRemittancesInr = 0,
    remittanceType = 'OTHER_REMITTANCE',
    hasPan = true,
    isPanInoperative = false,
    transactionDate = null
} = {}) {
    const rawAmt = Math.max(0, assertMandatoryNumeric(remittanceAmountInr, 'remittanceAmountInr', 'evaluateSection206C1G'));
    const priorAmt = Math.max(0, cleanNumericAmount(cumulativePriorRemittancesInr));
    const totalCumulative = roundToPaisa(priorAmt + rawAmt);

    const THRESHOLD = 700000;
    const type = String(remittanceType).toUpperCase();

    let calculatedTcs = 0;
    let applicableRate = 0;
    let taxableBase = 0;
    let rateDescription = '';

    if (type === 'OVERSEAS_TOUR_PACKAGE') {
        // Tour package: 5% up to 7L, 20% above 7L (no threshold exemption)
        let tier1 = 0;
        let tier2 = 0;

        const effectiveStart = priorAmt;
        const effectiveEnd = totalCumulative;

        const start1 = effectiveStart;
        const end1 = Math.min(THRESHOLD, effectiveEnd);
        if (end1 > start1) tier1 = end1 - start1;

        if (effectiveEnd > THRESHOLD) {
            const start2 = Math.max(THRESHOLD, effectiveStart);
            const end2 = effectiveEnd;
            if (end2 > start2) tier2 = end2 - start2;
        }

        const tcsTier1 = roundToPaisa((tier1 * 5.0) / 100);
        const tcsTier2 = roundToPaisa((tier2 * 20.0) / 100);
        calculatedTcs = roundToPaisa(tcsTier1 + tcsTier2);
        taxableBase = rawAmt;
        applicableRate = tier2 > 0 ? (tier1 > 0 ? 20.0 : 20.0) : 5.0;
        rateDescription = `Overseas tour package: 5% on INR ${tier1.toLocaleString('en-IN')} + 20% on INR ${tier2.toLocaleString('en-IN')} (>7L)`;
    } else {
        // Other LRS types: First 7 Lakhs is exempt
        if (totalCumulative <= THRESHOLD) {
            return {
                status: 'SUCCESS',
                statutoryDetermination: 'EXEMPT_BELOW_LRS_THRESHOLD',
                applicableTcsRatePercent: 0,
                taxableBaseInr: 0,
                calculatedTcsInr: 0,
                currentRemittanceInr: rawAmt,
                totalCumulativeInr: totalCumulative,
                thresholdCeilingInr: THRESHOLD,
                totalPayableByRemitterInr: rawAmt,
                isAuthorizedToDisburse: false,
                governingLaw: 'Income-tax Act, 1961 Section 206C(1G) Proviso',
                statutoryRationale: `Cumulative LRS remittances in FY (INR ${totalCumulative.toLocaleString('en-IN')}) do not exceed INR 7 Lakhs threshold. No TCS applicable.`
            };
        }

        taxableBase = priorAmt >= THRESHOLD ? rawAmt : roundToPaisa(totalCumulative - THRESHOLD);

        if (type === 'EDUCATION_LOAN') {
            applicableRate = 0.5;
            rateDescription = '0.5% on excess over INR 7 Lakhs (Education remittance funded by Section 80E loan)';
        } else if (type === 'EDUCATION_OR_MEDICAL') {
            applicableRate = 5.0;
            rateDescription = '5% on excess over INR 7 Lakhs (Education or medical treatment remittance)';
        } else {
            applicableRate = 20.0;
            rateDescription = '20% on excess over INR 7 Lakhs (General LRS remittance amended w.e.f. 1 Oct 2023)';
        }

        if (!hasPan || isPanInoperative) {
            applicableRate = Math.max(applicableRate, 20.0);
            rateDescription = `${rateDescription} - Penal rate (20%) due to missing/inoperative PAN`;
        }

        calculatedTcs = roundToPaisa((taxableBase * applicableRate) / 100);
    }

    const totalPayableByRemitter = roundToPaisa(rawAmt + calculatedTcs);

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'COLLECTION_MANDATORY',
        applicableTcsRatePercent: applicableRate,
        taxableBaseInr: taxableBase,
        calculatedTcsInr: calculatedTcs,
        remittanceAmountInr: rawAmt,
        totalPayableByRemitterInr: totalPayableByRemitter,
        governingLaw: `Income-tax Act, 1961 Section 206C(1G) (${rateDescription})`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `LRS remittance taxable base INR ${taxableBase.toLocaleString('en-IN')} subject to TCS of ${applicableRate}% = INR ${calculatedTcs.toLocaleString('en-IN')}. Total collected from remitter: INR ${totalPayableByRemitter.toLocaleString('en-IN')}.`
    };
}
