/**
 * INTERNATIONAL TAX & CROSS-BORDER CALCULATORS
 * Section 195, Lower Deduction Certificates, Transfer Pricing ALP (Sec 92C & Rule 10CA),
 * Section 79 Loss Continuity, Ind AS 21/109 FX Revaluation, Section 281 Tax Clearance.
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
 * ─────────────────────────────────────────────────────────────────────────────
 * 3. CROSS-BORDER WITHHOLDING & DUAL-ACT FORM ROUTER
 * ─────────────────────────────────────────────────────────────────────────────
 * Anchors the transition between Income-tax Act 1961 vs Income-tax Act 2025:
 * - Post-April 1, 2026: Section 393(2), Form 145 (Rule 220), Form 146
 * - Pre-April 1, 2026: Section 195, Form 15CA (Rule 37BB), Form 15CB
 * - SC Engineering Analysis (2021) automated cloud infra vs human FTS
 * - Form 145 Part A vs Part B (AO order) vs Part C (CA cert)
 */
export function evaluateCrossBorderRemittance({
    amountUsd,
    fxRate = 83.5,
    transactionDate = new Date().toISOString(),
    country = null,
    isAutomatedCloudInfra = true,
    hasHumanTechnicalServices = false,
    hasTrc = false,
    hasNoPe = false,
    hasForm10F = false,
    hasAoCertificate = false
}) {
    const usd = Number(amountUsd);
    const amountInr = Number((usd * fxRate).toFixed(2));
    const txDate = new Date(transactionDate);

    // If treaty relief is claimed (TRC, No-PE, or Form 10F) but country is missing,
    // bilateral DTAA treaty benefits cannot be determined under Section 90 without knowing the Contracting State.
    if ((hasTrc || hasNoPe || hasForm10F) && (!country || typeof country !== 'string' || !country.trim())) {
        return {
            amountUsd: usd,
            amountInr,
            transactionDate,
            statutoryDetermination: 'INDETERMINATE',
            status: 'INDETERMINATE',
            isAuthorizedToDisburse: false,
            effectiveRatePercent: null,
            calculatedTdsInr: null,
            withholdingTaxInr: null,
            netRemittanceInr: null,
            indeterminacyReason: 'MISSING_RECIPIENT_COUNTRY_FOR_DTAA',
            characterization: 'Recipient country of tax residency is mandatory to evaluate bilateral DTAA treaty benefits and rates under Section 90.',
            taxabilityBasis: 'DTAA_TREATY_COUNTRY_REQUIRED',
            statutoryCitations: txDate >= new Date('2026-04-01') ? {
                act: "Income-tax Act, 2025",
                chargingSection: "Section 393(2)"
            } : {
                act: "Income-tax Act, 1961",
                chargingSection: "Section 195"
            },
            requiredDocumentation: ['Recipient Country of Tax Residency', 'Tax Residency Certificate (TRC)', 'Form 10F', 'No-PE Certificate']
        };
    }

    // ── TEMPORAL ACT BRIDGE (1 April 2026 Boundary) ──
    const isAct2025 = txDate >= new Date('2026-04-01');

    const statutoryCitations = isAct2025 ? {
        act: "Income-tax Act, 2025",
        chargingSection: "Section 393(2) (Non-Resident Non-Salary Withholding Table)",
        declarationForm: "Form 145",
        caCertificateForm: "Form 146",
        governingRule: "Rule 220 of Income-tax Rules, 2026",
        penaltySection: "Section 462 (Penalty for non-furnishing of Form 145, up to ₹1,00,000)"
    } : {
        act: "Income-tax Act, 1961",
        chargingSection: "Section 195 (Other Sums Payable to Non-Resident)",
        declarationForm: "Form 15CA",
        caCertificateForm: "Form 15CB",
        governingRule: "Rule 37BB of Income-tax Rules, 1962",
        penaltySection: "Section 271-I (Penalty for failure to furnish information, ₹1,00,000)"
    };

    // ── TAXABILITY & CLOUD CHARACTERIZATION ──
    // Engineering Analysis (2021) SC + Rackspace/AWS ITAT Rulings
    let characterization = "";
    let effectiveRatePercent = 20; // Default domestic non-treaty rate
    let taxabilityBasis = "";

    const isTaxHaven = country ? ['CYPRUS', 'CAYMAN', 'BERMUDA', 'BVI', 'PANAMA'].includes(String(country).toUpperCase().trim()) : false;
    if (isTaxHaven) {
        effectiveRatePercent = 30; // Section 94A NJA rate
        characterization = "Notified Jurisdictional Area (Tax Haven) under Section 94A / 2025 Act equivalent. Mandatory 30% withholding.";
        taxabilityBasis = "TAX_HAVEN_STATUTORY_OVERRIDE";
    } else if (isAutomatedCloudInfra && !hasHumanTechnicalServices) {
        // Pure automated SaaS/database infrastructure
        if (hasTrc && hasNoPe) {
            effectiveRatePercent = 0;
            characterization = "Business Profits under DTAA Article 7. Under Supreme Court Engineering Analysis (2021) and ITAT precedents, automated cloud hosting without human managerial/technical involvement does not constitute Royalty or FTS. In the absence of an Indian Permanent Establishment (PE), income is non-taxable in India.";
            taxabilityBasis = "DTAA_ARTICLE_7_BUSINESS_PROFITS_NON_TAXABLE";
        } else {
            effectiveRatePercent = 20;
            characterization = "Automated cloud infrastructure, but foreign vendor lacks TRC and No-PE certificate. Default domestic withholding applies under Section 195 / Section 393.";
            taxabilityBasis = "UNVERIFIED_TREATY_DOMESTIC_RATE";
        }
    } else {
        // Fees for Technical Services (FTS) / Royalty
        if (hasTrc && hasForm10F) {
            effectiveRatePercent = 10; // Standard DTAA Article 12 cap
            characterization = "Fees for Technical Services (FTS) / Royalty under DTAA Article 12. Eligible for concessional treaty rate of 10% (plus surcharge/cess as applicable).";
            taxabilityBasis = "DTAA_ARTICLE_12_FTS_TREATY_RATE";
        } else {
            effectiveRatePercent = 20;
            characterization = "FTS / Royalty without valid TRC and Form 10F. Defaults to domestic statutory rate of 20%.";
            taxabilityBasis = "DOMESTIC_STATUTORY_FTS_RATE";
        }
    }

    // ── FORM ROUTING CONDITIONALITY TREE (Part A vs Part B vs Part C) ──
    const FIVE_LAKHS_INR = 500000;
    let applicablePart = "";
    let caCertificateRequired = false;
    let proceduralSummary = "";

    if (amountInr <= FIVE_LAKHS_INR && effectiveRatePercent === 0) {
        applicablePart = "PART_A";
        caCertificateRequired = false;
        proceduralSummary = `Aggregate remittance in financial year does not exceed ₹5,00,000. File ${statutoryCitations.declarationForm} Part A only. CA Certificate (${statutoryCitations.caCertificateForm}) is not required.`;
    } else if (hasAoCertificate) {
        applicablePart = "PART_B";
        caCertificateRequired = false;
        proceduralSummary = `Lower/Nil withholding certificate or order obtained from Assessing Officer under Section 197 / Section 195(2). File ${statutoryCitations.declarationForm} Part B. CA Certificate (${statutoryCitations.caCertificateForm}) is waived under law.`;
    } else {
        applicablePart = "PART_C";
        caCertificateRequired = true;
        proceduralSummary = `Remittance exceeds ₹5,00,000 and no AO order has been obtained. Mandatory compliance: Obtain CA Certificate in ${statutoryCitations.caCertificateForm}, then file ${statutoryCitations.declarationForm} Part C online before initiating the bank transfer.`;
    }

    const calculatedTdsInr = Number(((amountInr * effectiveRatePercent) / 100).toFixed(2));
    const netRemittanceInr = Number((amountInr - calculatedTdsInr).toFixed(2));

    // Dual control check
    const requiresAdeshaTier2Card = usd > 500;

    return {
        amountUsd: usd,
        amountInr,
        transactionDate,
        statutoryCitations,
        characterization,
        taxabilityBasis,
        effectiveRatePercent,
        calculatedTdsInr,
        netRemittanceInr,
        applicablePart,
        caCertificateRequired,
        proceduralSummary,
        requiredDocumentation: [
            "Vendor Tax Residency Certificate (TRC) covering relevant financial year",
            "Form 10F (self-declaration filed on income tax portal)",
            "No Permanent Establishment (No-PE) certificate from vendor",
            "Invoice and Cloud Service Level Agreement (confirming hosted outside India)",
            caCertificateRequired ? `${statutoryCitations.caCertificateForm} (Chartered Accountant Certificate)` : null,
            `${statutoryCitations.declarationForm} (${applicablePart} acknowledgement)`
        ].filter(Boolean),
        governanceGate: {
            requiresAdeshaTier2Card,
            policy: requiresAdeshaTier2Card 
                ? "International SWIFT remittance exceeds $500 threshold. Bounded Human-in-the-Loop Attention Card + dual-control sign-off required before release."
                : "Within standard autonomous threshold."
        }
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 4. SECTION 194Q / 206C(1H) / 206AA: DETERMINISTIC GOODS WITHHOLDING ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Computes exact taxable base exceeding ₹50,00,000 annual threshold, applies
 * Section 206AA penalty rate (20%) if PAN is missing/inoperative, models the
 * Section 206C(1H) sunset effective 1 April 2025, and transitions to
 * Section 393(1) of the Income-tax Act, 2025 on 1 April 2026.
 */
/**
 * 3. LOWER DEDUCTION CERTIFICATE STATUTORY EVALUATOR (SECTION 197 / SECTION 395)
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates the statutory validity of lower deduction certificates across regime transitions:
 * - Checks explicit cancellation or revocation status (Assessing Officer / TRACES).
 * - Checks covered period / validity window against transaction effective date.
 * - Enforces Section 536(2) transitional savings for legacy Section 197 certificates
 *   pre-issued for FY 2026-27 under the Income-tax Act, 2025.
 * - Rejects expired certificates (e.g. FY 2025-26 certificates used post-April 2026).
 */
export function evaluateLowerDeductionCertificateValidity({
    effectiveDate = null,
    lowerDeductionCertificate = null,
    certificateSection = null,
    isAct2025 = false
} = {}) {
    if (!lowerDeductionCertificate && !certificateSection) {
        return {
            isValid: true,
            status: 'NO_CERTIFICATE',
            reason: null,
            governingCertificateSection: isAct2025 ? 'Income-tax Act, 2025 Section 395' : 'Income-tax Act, 1961 Section 197',
            isSection536TransitionalSavingsApplied: false,
            transitionalSavingsClause: null,
            certificateSection: isAct2025 ? '395' : '197'
        };
    }

    const certObj = typeof lowerDeductionCertificate === 'object' && lowerDeductionCertificate !== null
        ? lowerDeductionCertificate
        : {};

    const certRef = String(
        certificateSection ||
        certObj.certificateSection ||
        certObj.section ||
        certObj.governingSection ||
        ''
    ).trim();

    const isLegacy197Ref = /197/.test(certRef) || Boolean(certObj.isTransitionalSection197);

    // 1. Explicit Cancellation / Revocation Check
    if (certObj.isCancelled === true || certObj.revoked === true || certObj.status === 'CANCELLED' || certObj.status === 'REVOKED') {
        return {
            isValid: false,
            status: 'CERTIFICATE_CANCELLED_OR_REVOKED',
            reason: 'Lower deduction certificate has been explicitly cancelled or revoked by the Assessing Officer / TRACES authority.',
            governingCertificateSection: isAct2025 ? 'Income-tax Act, 2025 Section 395 (Revoked)' : 'Income-tax Act, 1961 Section 197 (Revoked)',
            isSection536TransitionalSavingsApplied: false,
            transitionalSavingsClause: null,
            certificateSection: certRef || (isAct2025 ? '395' : '197')
        };
    }

    // 2. Covered Period / Expiry Window Check
    const validFrom = certObj.validFrom || null;
    const validTo = certObj.validTo || null;
    const coveredFy = certObj.coveredFinancialYear || certObj.financialYear || null;

    if (effectiveDate) {
        if (validFrom && effectiveDate < validFrom) {
            return {
                isValid: false,
                status: 'CERTIFICATE_PREMATURE_DATE',
                reason: `Transaction date (${effectiveDate}) is prior to certificate commencement date (${validFrom}).`,
                governingCertificateSection: isAct2025 ? 'Income-tax Act, 2025 Section 395 (Premature)' : 'Income-tax Act, 1961 Section 197 (Premature)',
                isSection536TransitionalSavingsApplied: false,
                transitionalSavingsClause: null,
                certificateSection: certRef || (isAct2025 ? '395' : '197')
            };
        }
        if (validTo && effectiveDate > validTo) {
            return {
                isValid: false,
                status: 'CERTIFICATE_EXPIRED_COVERED_PERIOD',
                reason: `Transaction date (${effectiveDate}) is outside certificate covered validity period (expired on ${validTo}).`,
                governingCertificateSection: isAct2025 ? 'Income-tax Act, 2025 Section 395 (Expired)' : 'Income-tax Act, 1961 Section 197 (Expired)',
                isSection536TransitionalSavingsApplied: false,
                transitionalSavingsClause: null,
                certificateSection: certRef || (isAct2025 ? '395' : '197')
            };
        }
        if (coveredFy && isAct2025) {
            // If transaction is in FY 2026-27 or later, but certificate was strictly for FY 2025-26
            if (/2025-26/.test(coveredFy) && effectiveDate >= '2026-04-01') {
                return {
                    isValid: false,
                    status: 'CERTIFICATE_EXPIRED_COVERED_PERIOD',
                    reason: 'Certificate was sanctioned strictly for FY 2025-26 and does not cover receivables for FY 2026-27.',
                    governingCertificateSection: 'Income-tax Act, 1961 Section 197 (Expired FY 2025-26)',
                    isSection536TransitionalSavingsApplied: false,
                    transitionalSavingsClause: null,
                    certificateSection: certRef || '197'
                };
            }
        }
    }

    // 3. Act 2025 vs Act 1961 Regime Resolution
    if (isAct2025) {
        if (isLegacy197Ref) {
            return {
                isValid: true,
                status: 'VALID_LEGACY_CERTIFICATE_UNDER_SEC_536_2',
                reason: 'Certificate issued under Section 197 covering projected receivables for FY 2026-27 remains legally enforceable under Section 536(2) transitional savings.',
                governingCertificateSection: "Income-tax Act, 2025 Section 395 read with Section 536(2) Transitional Savings (Deemed Section 197 Certificate)",
                isSection536TransitionalSavingsApplied: true,
                transitionalSavingsClause: "Income-tax Act, 2025 Section 536(2)",
                certificateSection: certRef || '197'
            };
        }
        return {
            isValid: true,
            status: 'VALID_ACT_2025_SEC_395',
            reason: 'Valid certificate issued under Section 395 on Form 128 of the Income-tax Act, 2025.',
            governingCertificateSection: "Income-tax Act, 2025 Section 395",
            isSection536TransitionalSavingsApplied: false,
            transitionalSavingsClause: null,
            certificateSection: certRef || '395'
        };
    }

    return {
        isValid: true,
        status: 'VALID_ACT_1961_SEC_197',
        reason: 'Valid certificate issued under Section 197 of the Income-tax Act, 1961.',
        governingCertificateSection: "Income-tax Act, 1961 Section 197",
        isSection536TransitionalSavingsApplied: false,
        transitionalSavingsClause: null,
        certificateSection: certRef || '197'
    };
}

export function resolveGoverningCertificateSection(params = {}) {
    return evaluateLowerDeductionCertificateValidity(params);
}

/**
 * -----------------------------------------------------------------------------
 * 30. TRANSFER PRICING ARM'S LENGTH PRICE (ALP) ENGINE (SECTION 92C & RULE 10CA)
 * -----------------------------------------------------------------------------
 * Computes Transfer Pricing compliance:
 * 1. Associated Enterprise (AE) transaction validation under Section 92A.
 * 2. 3% tolerance band for wholesale traders vs 1% for other businesses (proviso to Sec 92C(2)).
 * 3. Rule 10CA Percentile Range: 35th to 65th percentile dataset evaluation for >= 6 comparables.
 * 4. Rule 10CA(7) Median (50th percentile) primary adjustment if tested margin is outside range.
 * 5. Section 92CE Secondary Adjustment threshold warning (> 1 Cr).
 */
export function calculateTransferPricingAlp({
    transactionDescription = 'International Transaction with Associated Enterprise',
    testedParty = 'Indian Enterprise',
    associatedEnterpriseName = 'Foreign AE',
    isAssociatedEnterprise = true,
    internationalTransactionAmountInr = 0,
    testedPartyOperatingMarginPercent = 0,
    isWholesaleTrading = false,
    comparablesData = [],
    transferPricingMethod = 'TNMM'
}) {
    const amount = Number(internationalTransactionAmountInr) || 0;
    const testedMargin = Number(testedPartyOperatingMarginPercent) || 0;
    const tolerancePercent = isWholesaleTrading ? 3.0 : 1.0;

    if (!isAssociatedEnterprise) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'NOT_APPLICABLE',
            governingLaw: 'Income-tax Act, 1961 Section 92A & 92C',
            transactionDescription,
            isAssociatedEnterprise: false,
            statutoryRationale: 'Counterparty is not an Associated Enterprise under Section 92A. Transfer Pricing provisions u/s 92C do not apply to third-party transactions.',
            isAuthorizedToDisburse: false
        };
    }

    const comparables = Array.isArray(comparablesData)
        ? comparablesData.map(x => Number(typeof x === 'object' && x !== null ? (x.operatingMargin || x.margin || 0) : x)).filter(x => !isNaN(x)).sort((a, b) => a - b)
        : [];

    let isArmLengthCompliant = false;
    let primaryAdjustmentInr = 0;
    let percentile35 = null;
    let percentile65 = null;
    let median50 = null;
    let arithmeticMean = null;
    let appliedRule = '';

    if (comparables.length >= 6) {
        appliedRule = 'Rule 10CA Percentile Range (Dataset >= 6 Comparables)';
        const n = comparables.length;

        const getPercentileValue = (p) => {
            const index = (n * p);
            if (Number.isInteger(index)) {
                const i = index - 1;
                return (comparables[i] + comparables[Math.min(i + 1, n - 1)]) / 2;
            } else {
                const i = Math.floor(index);
                return comparables[Math.min(i, n - 1)];
            }
        };

        percentile35 = roundToPaisa(getPercentileValue(0.35));
        percentile65 = roundToPaisa(getPercentileValue(0.65));
        median50 = roundToPaisa(getPercentileValue(0.50));

        if (testedMargin >= percentile35 && testedMargin <= percentile65) {
            isArmLengthCompliant = true;
            primaryAdjustmentInr = 0;
        } else if (testedMargin > percentile65) {
            isArmLengthCompliant = true;
            primaryAdjustmentInr = 0;
        } else {
            isArmLengthCompliant = false;
            const marginShortfall = roundToPaisa(median50 - testedMargin);
            primaryAdjustmentInr = roundToPaisa(amount * (marginShortfall / 100));
        }
    } else if (comparables.length > 0) {
        appliedRule = 'Proviso to Section 92C(2) Arithmetic Mean & Tolerance Band (< 6 Comparables)';
        arithmeticMean = roundToPaisa(comparables.reduce((sum, x) => sum + x, 0) / comparables.length);
        const variationPercent = Math.abs(testedMargin - arithmeticMean);

        if (variationPercent <= tolerancePercent || testedMargin >= arithmeticMean) {
            isArmLengthCompliant = true;
            primaryAdjustmentInr = 0;
        } else {
            isArmLengthCompliant = false;
            const marginShortfall = roundToPaisa(arithmeticMean - testedMargin);
            primaryAdjustmentInr = roundToPaisa(amount * (marginShortfall / 100));
        }
    } else {
        appliedRule = 'Single Transaction Assessment (No Dataset)';
        isArmLengthCompliant = true;
        primaryAdjustmentInr = 0;
    }

    const requiresSecondaryAdjustment = primaryAdjustmentInr > 10000000;

    return {
        status: 'SUCCESS',
        statutoryDetermination: isArmLengthCompliant ? 'ARM_LENGTH_COMPLIANT' : 'PRIMARY_ADJUSTMENT_REQUIRED',
        governingLaw: 'Income-tax Act, 1961 Section 92C, Rule 10B & Rule 10CA',
        testedParty,
        associatedEnterpriseName,
        transferPricingMethod,
        internationalTransactionAmountInr: amount,
        testedPartyOperatingMarginPercent: testedMargin,
        appliedRule,
        toleranceBandPercent: tolerancePercent,
        datasetSummary: {
            comparablesCount: comparables.length,
            percentile35,
            percentile65,
            median50,
            arithmeticMean,
            armLengthRange: percentile35 !== null ? [percentile35, percentile65] : null
        },
        primaryAdjustmentInr,
        secondaryAdjustmentAlert: requiresSecondaryAdjustment ? {
            applicable: true,
            section: 'Section 92CE',
            threshold: 'Exceeds INR 1,00,00,000',
            repatriationPeriodDays: 90,
            statutoryConsequence: 'Excess money must be repatriated into India within 90 days, failing which interest at SBI 1-yr MCLR + 325 bps applies.'
        } : { applicable: false },
        isAuthorizedToDisburse: false,
        statutoryRationale: isArmLengthCompliant
            ? `Transfer Pricing Compliant: Tested party operating margin of ${testedMargin}% is within the statutory arm's length range [${percentile35 ?? arithmeticMean}%, ${percentile65 ?? arithmeticMean}%]. Primary adjustment is INR 0.`
            : `Primary Adjustment Required: Tested margin (${testedMargin}%) falls below statutory arm's length range. Under Rule 10CA(7), adjustment is made to the Median (50th percentile: ${median50}%). Primary TP addition to income: INR ${primaryAdjustmentInr.toLocaleString('en-IN')}.${requiresSecondaryAdjustment ? ' Section 92CE secondary adjustment mandatory.' : ''}`
    };
}

/**
 * -----------------------------------------------------------------------------
 * 31. SECTION 79 TAX LOSS CARRY-FORWARD & SHAREHOLDING CONTINUITY
 * -----------------------------------------------------------------------------
 * Evaluates carry forward and set off of accumulated business losses:
 * 1. General Rule (Sec 79(1)): Denied if >= 51% change in voting power.
 * 2. Startup Carve-Out (Sec 79(2)): Preserved for DPIIT startups u/s 80-IAC if all original shareholders continue.
 * 3. IBC Carve-Out (Sec 79(2)(c)): Preserved for NCLT resolution plans.
 * 4. Section 32(2) Invariant: Unabsorbed depreciation carries forward INDEFINITELY regardless of Section 79.
 */
export function evaluateSection79LossContinuity({
    companyType = 'CLOSELY_HELD_COMPANY',
    isDpiitRecognizedStartup = false,
    isIbcResolutionPlan = false,
    accumulatedBroughtForwardLossInr = 0,
    unabsorbedDepreciationInr = 0,
    shareholdingInLossYear = [],
    shareholdingInSetOffYear = [],
    votingPowerHeldByOriginalShareholdersPercent = null
}) {
    const bfLoss = Math.max(0, Number(accumulatedBroughtForwardLossInr) || 0);
    const unabsDep = Math.max(0, Number(unabsorbedDepreciationInr) || 0);

    let continuityPercent = votingPowerHeldByOriginalShareholdersPercent !== null
        ? Number(votingPowerHeldByOriginalShareholdersPercent)
        : null;

    let allOriginalShareholdersContinue = false;

    if (continuityPercent === null && Array.isArray(shareholdingInLossYear) && shareholdingInLossYear.length > 0 && Array.isArray(shareholdingInSetOffYear)) {
        let commonVotingPower = 0;
        let originalFoundCount = 0;

        for (const orig of shareholdingInLossYear) {
            const pan = orig.shareholderPan || orig.pan || orig.name;
            const origPct = Number(orig.percentage || orig.votingPower || 0);
            const current = shareholdingInSetOffYear.find(x => (x.shareholderPan || x.pan || x.name) === pan);

            if (current && Number(current.percentage || current.votingPower || 0) > 0) {
                originalFoundCount++;
                commonVotingPower += Math.min(origPct, Number(current.percentage || current.votingPower || 0));
            }
        }
        continuityPercent = roundToPaisa(commonVotingPower);
        allOriginalShareholdersContinue = (originalFoundCount === shareholdingInLossYear.length);
    } else if (continuityPercent !== null) {
        allOriginalShareholdersContinue = continuityPercent > 0;
    } else {
        continuityPercent = 100;
        allOriginalShareholdersContinue = true;
    }

    let isLossCarryForwardAllowed = false;
    let governingClause = 'Section 79(1)';
    let rationale = '';

    if (isIbcResolutionPlan) {
        isLossCarryForwardAllowed = true;
        governingClause = 'Section 79(2)(c) (Insolvency and Bankruptcy Code Resolution Plan)';
        rationale = 'Accumulated business losses are protected under Section 79(2)(c): Change in shareholding pursuant to NCLT approved resolution plan under the IBC, 2016 is exempt from Section 79 restrictions.';
    } else if (isDpiitRecognizedStartup) {
        governingClause = 'Section 79(2) (DPIIT Recognized Startup under Section 80-IAC)';
        if (allOriginalShareholdersContinue) {
            isLossCarryForwardAllowed = true;
            rationale = `Accumulated business losses protected under Section 79(2) startup carve-out: Although original voting power is ${continuityPercent}%, all original shareholders continue to hold shares in the company.`;
        } else {
            isLossCarryForwardAllowed = continuityPercent >= 51.0;
            rationale = isLossCarryForwardAllowed
                ? `Section 79 satisfied: Original shareholders retain ${continuityPercent}% voting power (>= 51%).`
                : `Section 79(2) startup carve-out failed: Not all original shareholders continue to hold shares, and retained voting power (${continuityPercent}%) is below 51%. Business loss lapsed.`;
        }
    } else {
        governingClause = 'Section 79(1) General Corporate Rule';
        if (continuityPercent >= 51.0) {
            isLossCarryForwardAllowed = true;
            rationale = `Section 79(1) compliant: Original shareholders hold ${continuityPercent}% of voting power on the last day of the financial year (threshold: >= 51%). Business losses can be set off.`;
        } else {
            isLossCarryForwardAllowed = false;
            rationale = `Section 79(1) violation: Voting power held by original shareholders has fallen to ${continuityPercent}% (< 51%). Accumulated business losses of INR ${bfLoss.toLocaleString('en-IN')} are legally lapsed and forfeited.`;
        }
    }

    const allowableLossSetOffInr = isLossCarryForwardAllowed ? bfLoss : 0;
    const lapsedBusinessLossInr = isLossCarryForwardAllowed ? 0 : bfLoss;

    return {
        status: 'SUCCESS',
        statutoryDetermination: isLossCarryForwardAllowed ? 'LOSS_CARRYFORWARD_ALLOWED' : 'LOSS_CARRYFORWARD_DENIED',
        governingLaw: 'Income-tax Act, 1961 Section 79 & Section 32(2)',
        governingClause,
        companyType,
        isDpiitRecognizedStartup,
        isIbcResolutionPlan,
        beneficialOwnershipContinuityPercent: continuityPercent,
        allOriginalShareholdersContinue,
        lossSchedule: {
            broughtForwardBusinessLossInr: bfLoss,
            allowableLossSetOffInr,
            lapsedBusinessLossInr,
            unabsorbedDepreciationInr: unabsDep,
            unabsorbedDepreciationStatutoryStatus: 'Fully protected under Section 32(2) -- Indefinite carry forward unaffected by shareholding changes.'
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `${rationale} Note: Unabsorbed depreciation of INR ${unabsDep.toLocaleString('en-IN')} is unaffected by Section 79 per Supreme Court precedent (CIT v. Virmani Industries) and carries forward indefinitely.`
    };
}

/**
 * -----------------------------------------------------------------------------
 * 32. FOREIGN EXCHANGE MONETARY REVALUATION (IND AS 21 & IND AS 109)
 * -----------------------------------------------------------------------------
 * Revalues foreign currency monetary items at closing reporting spot rates:
 * 1. Monetary Assets (Receivables, Bank) vs Monetary Liabilities (Payables, Loans).
 * 2. Net Unrealized Gain/Loss routing: P&L vs Other Comprehensive Income (OCI - Cash Flow Hedge Reserve).
 * 3. Ind AS 109 Balance Sheet schedule generation.
 */
export function revalueForeignExchangeIndAs109({
    asOfDate = null,
    reportingCurrency = 'INR',
    items = []
}) {
    const dAsOf = asOfDate ? new Date(asOfDate) : new Date();
    const asOfIso = !isNaN(dAsOf.getTime()) ? dAsOf.toISOString().split('T')[0] : new Date().toISOString().split('T')[0];

    const revaluedItems = [];
    let totalPnlGainInr = 0;
    let totalPnlLossInr = 0;
    let totalOciGainInr = 0;
    let totalOciLossInr = 0;

    const itemList = Array.isArray(items) ? items : [];

    for (const item of itemList) {
        const desc = item.description || item.itemDescription || 'FX Item';
        const itemType = (item.itemType || 'MONETARY_ASSET').toUpperCase();
        const ccy = item.foreignCurrency || item.currency || 'USD';
        const fcyAmt = Number(item.foreignAmount || item.amount || 0);
        const bookRate = Number(item.bookingRate ?? item.historicalRate ?? 1);
        const closeRate = Number(item.closingRate ?? item.spotRate ?? bookRate);
        const isHedge = item.isDesignatedCashFlowHedge === true;

        if (itemType === 'NON_MONETARY') {
            revaluedItems.push({
                description: desc,
                itemType: 'NON_MONETARY',
                foreignCurrency: ccy,
                foreignAmount: fcyAmt,
                bookingRate: bookRate,
                closingRate: closeRate,
                carryingAmountInr: roundToPaisa(fcyAmt * bookRate),
                unrealizedGainLossInr: 0,
                accountingTreatment: 'Carried at historical cost per Ind AS 21.23(b) -- No closing rate revaluation.'
            });
            continue;
        }

        const carryingAmountInr = roundToPaisa(fcyAmt * bookRate);
        const revaluedAmountInr = roundToPaisa(fcyAmt * closeRate);

        let gainLoss = 0;
        if (itemType === 'MONETARY_ASSET') {
            gainLoss = roundToPaisa(revaluedAmountInr - carryingAmountInr);
        } else {
            gainLoss = roundToPaisa(carryingAmountInr - revaluedAmountInr);
        }

        let routing = '';
        if (isHedge) {
            routing = 'OTHER_COMPREHENSIVE_INCOME (Cash Flow Hedge Reserve)';
            if (gainLoss >= 0) totalOciGainInr += gainLoss;
            else totalOciLossInr += Math.abs(gainLoss);
        } else {
            routing = 'PROFIT_AND_LOSS (Foreign Exchange Fluctuation Account)';
            if (gainLoss >= 0) totalPnlGainInr += gainLoss;
            else totalPnlLossInr += Math.abs(gainLoss);
        }

        revaluedItems.push({
            description: desc,
            itemType,
            foreignCurrency: ccy,
            foreignAmount: fcyAmt,
            bookingRate: bookRate,
            closingRate: closeRate,
            carryingAmountInr,
            revaluedAmountInr,
            unrealizedGainLossInr: gainLoss,
            isGain: gainLoss >= 0,
            accountingTreatment: routing
        });
    }

    const netPnlGainLossInr = roundToPaisa(totalPnlGainInr - totalPnlLossInr);
    const netOciGainLossInr = roundToPaisa(totalOciGainInr - totalOciLossInr);

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'APPLICABLE',
        governingLaw: 'Ind AS 21 (The Effects of Changes in Foreign Exchange Rates) & Ind AS 109 (Financial Instruments)',
        asOfDate: asOfIso,
        reportingCurrency,
        totalItemsEvaluated: revaluedItems.length,
        summary: {
            netPnlImpactInr: netPnlGainLossInr,
            netPnlStatus: netPnlGainLossInr >= 0 ? 'NET_UNREALIZED_GAIN' : 'NET_UNREALIZED_LOSS',
            netOciImpactInr: netOciGainLossInr,
            netOciStatus: netOciGainLossInr >= 0 ? 'NET_HEDGE_RESERVE_GAIN' : 'NET_HEDGE_RESERVE_LOSS',
            totalPnlGainInr: roundToPaisa(totalPnlGainInr),
            totalPnlLossInr: roundToPaisa(totalPnlLossInr)
        },
        revaluedItems,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 21 & 109 Revaluation: ${revaluedItems.length} items evaluated as of ${asOfIso}. Net P&L FX impact: INR ${netPnlGainLossInr.toLocaleString('en-IN')}; Net OCI Hedge impact: INR ${netOciGainLossInr.toLocaleString('en-IN')}. Accounting revaluations do not authorize payment disbursement.`
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 31B. SECTION 281 TAX CLEARANCE & VOID TRANSFER RISK ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Evaluates asset transfer voidability and tax clearance safe harbors in corporate
 * restructuring, slump sales, and asset acquisitions under Section 281 of the
 * Income-tax Act, 1961 read with CBDT Circular No. 04/2011:
 * - Asset Scope: Land, buildings, machinery, plant, shares, securities, bank FDs > ₹10,000.
 * - Voidability Trigger: Transfer or charge creation during pending tax proceedings
 *   (Sec 143(2), 148, 153A, 250, 254) or outstanding demand under Section 156.
 * - Statutory Safe Harbors:
 *   1. Section 281(1) Proviso (ii): Prior written permission of Assessing Officer (Circular 04/2011).
 *   2. Section 281(1) Proviso (i): Bona fide purchaser for adequate consideration without notice.
 */
export function evaluateSection281TaxClearance({
    assetType = 'IMMOVABLE_PROPERTY', // 'IMMOVABLE_PROPERTY' | 'PLANT_AND_MACHINERY' | 'SHARES_AND_SECURITIES' | 'FIXED_DEPOSIT' | 'OTHER'
    assetDescription = 'Commercial Asset / Undertaking',
    assetValueInr = 0,
    transferNature = 'SALE', // 'SALE' | 'MORTGAGE' | 'GIFT' | 'EXCHANGE' | 'CHARGE_CREATION'
    hasPendingProceedings = false,
    pendingProceedings = [],
    hasOutstandingDemand = false,
    outstandingDemandInr = 0,
    hasPriorAoPermission = false,
    aoPermissionReference = null,
    aoPermissionDate = null,
    hasAdequateConsideration = true,
    hasNoticeOfProceedingsOrDemand = false,
    alternativeSecurityFurnishedInr = 0
} = {}) {
    const rawAssetVal = Math.max(0, Number(assetValueInr) || 0);
    const assetVal = roundToPaisa(rawAssetVal);
    const rawDemand = Math.max(0, Number(outstandingDemandInr) || 0);
    const demandInr = roundToPaisa(rawDemand);
    const rawAltSec = Math.max(0, Number(alternativeSecurityFurnishedInr) || 0);
    const altSecInr = roundToPaisa(rawAltSec);

    let totalPendingExposureInr = demandInr;
    if (Array.isArray(pendingProceedings)) {
        for (const proc of pendingProceedings) {
            const exp = Math.max(0, Number(proc.estimatedExposureInr || proc.exposureInr || proc.amountInr) || 0);
            totalPendingExposureInr = roundToPaisa(totalPendingExposureInr + exp);
        }
    }

    const hasAnyProceedings = Boolean(hasPendingProceedings || (Array.isArray(pendingProceedings) && pendingProceedings.length > 0));
    const hasAnyDemand = Boolean(hasOutstandingDemand || demandInr > 0);

    // ── GATE 1: DE MINIMIS STATUTORY EXCLUSION ──
    // Section 281 Explanation: Assets exceeding INR 10,000 in value
    if (assetVal <= 10000) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'EXEMPT_DE_MINIMIS',
            isVoidableAgainstRevenue: false,
            riskLevel: 'NONE',
            governingLaw: 'Income-tax Act, 1961 Section 281 Explanation',
            assetParticulars: {
                assetType,
                assetDescription,
                assetValueInr: assetVal,
                transferNature
            },
            safeHarborQualification: 'EXEMPT_BELOW_STATUTORY_FLOOR',
            totalTaxExposureInr: totalPendingExposureInr,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Asset value of INR ${assetVal.toLocaleString('en-IN')} does not exceed the statutory threshold of INR 10,000 under Section 281 Explanation. Section 281 voidability is inapplicable.`
        };
    }

    // ── GATE 2: CLEAN TRANSFEROR (NO PROCEEDINGS OR DEMANDS) ──
    if (!hasAnyProceedings && !hasAnyDemand) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'CLEAN_NO_PROCEEDINGS_PENDING',
            isVoidableAgainstRevenue: false,
            riskLevel: 'LOW',
            governingLaw: 'Income-tax Act, 1961 Section 281',
            assetParticulars: {
                assetType,
                assetDescription,
                assetValueInr: assetVal,
                transferNature
            },
            safeHarborQualification: 'NOT_APPLICABLE_NO_PENDING_PROCEEDINGS',
            totalTaxExposureInr: 0,
            isAuthorizedToDisburse: false,
            statutoryRationale: 'No proceedings under the Income-tax Act are pending against the transferor and no outstanding tax demands exist. Transfer/charge is not subject to statutory voidability under Section 281.'
        };
    }

    // ── GATE 3: SAFE HARBOR 2 — PRIOR ASSESSING OFFICER PERMISSION ──
    if (hasPriorAoPermission) {
        return {
            status: 'SUCCESS',
            statutoryDetermination: 'PROTECTED_AO_PERMISSION',
            isVoidableAgainstRevenue: false,
            riskLevel: 'ZERO',
            governingLaw: 'Income-tax Act, 1961 Section 281(1) Proviso (ii) & CBDT Circular No. 04/2011',
            assetParticulars: {
                assetType,
                assetDescription,
                assetValueInr: assetVal,
                transferNature
            },
            safeHarborQualification: 'SECTION_281_PROVISO_II_AO_PERMISSION',
            clearanceDetails: {
                aoPermissionReference: aoPermissionReference || 'NOC_F_NO_SEC281_VERIFIED',
                aoPermissionDate: aoPermissionDate || new Date().toISOString().split('T')[0],
                alternativeSecurityFurnishedInr: altSecInr
            },
            totalTaxExposureInr: totalPendingExposureInr,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Prior written permission of the Assessing Officer obtained under Section 281(1) Proviso (ii) pursuant to CBDT Circular No. 04/2011. Transfer/charge is legally protected against avoidance or attachment by the Tax Recovery Officer.`
        };
    }

    // ── GATE 4: SAFE HARBOR 1 — BONA FIDE PURCHASER FOR ADEQUATE CONSIDERATION WITHOUT NOTICE ──
    if (hasAdequateConsideration && !hasNoticeOfProceedingsOrDemand) {
        if (hasAnyDemand && demandInr > 0) {
            // Constructive notice risk: If notice of demand under Sec 156 has been served or recorded, Department may challenge lack of notice
            return {
                status: 'SUCCESS',
                statutoryDetermination: 'CONDITIONAL_BONA_FIDE_DEFENSE',
                isVoidableAgainstRevenue: false,
                riskLevel: 'MODERATE_NOTICE_DISPUTE_RISK',
                governingLaw: 'Income-tax Act, 1961 Section 281(1) Proviso (i)',
                assetParticulars: {
                    assetType,
                    assetDescription,
                    assetValueInr: assetVal,
                    transferNature
                },
                safeHarborQualification: 'SECTION_281_PROVISO_I_BONA_FIDE_PURCHASER',
                totalTaxExposureInr: totalPendingExposureInr,
                outstandingDemandInr: demandInr,
                isAuthorizedToDisburse: false,
                statutoryRationale: `Transaction qualifies prima facie as a bona fide transfer for adequate consideration without notice under Section 281(1) Proviso (i). However, due to existing outstanding demand of INR ${demandInr.toLocaleString('en-IN')}, buyer should secure an indemnity escrow holdback or seek formal AO clearance under Circular 04/2011 to preclude constructive notice litigation.`
            };
        }

        return {
            status: 'SUCCESS',
            statutoryDetermination: 'PROTECTED_BONA_FIDE_PURCHASER',
            isVoidableAgainstRevenue: false,
            riskLevel: 'LOW_WITH_WARRANTIES',
            governingLaw: 'Income-tax Act, 1961 Section 281(1) Proviso (i)',
            assetParticulars: {
                assetType,
                assetDescription,
                assetValueInr: assetVal,
                transferNature
            },
            safeHarborQualification: 'SECTION_281_PROVISO_I_BONA_FIDE_PURCHASER',
            totalTaxExposureInr: totalPendingExposureInr,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Transfer is made for adequate consideration and without notice of pending tax proceedings. Fully qualifies for statutory protection under Section 281(1) Proviso (i).`
        };
    }

    // ── GATE 5: UNPROTECTED TRANSFER — STATUTORILY VOIDABLE AGAINST REVENUE ──
    return {
        status: 'SUCCESS',
        statutoryDetermination: 'STATUTORILY_VOIDABLE_AGAINST_REVENUE',
        isVoidableAgainstRevenue: true,
        riskLevel: 'HIGH_CRITICAL',
        governingLaw: 'Income-tax Act, 1961 Section 281(1)',
        assetParticulars: {
            assetType,
            assetDescription,
            assetValueInr: assetVal,
            transferNature
        },
        safeHarborQualification: 'NONE_STATUTORILY_VOIDABLE',
        totalTaxExposureInr: totalPendingExposureInr,
        outstandingDemandInr: demandInr,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Transfer/charge is legally void as against the Income Tax Department under Section 281(1) of the Income-tax Act, 1961. The Tax Recovery Officer can attach and auction the transferred asset to satisfy transferor tax liabilities of INR ${totalPendingExposureInr.toLocaleString('en-IN')}. Mandatory action: obtain AO NOC under CBDT Circular No. 04/2011 or establish an escrow holdback equal to 100% of tax exposure.`
    };
}
