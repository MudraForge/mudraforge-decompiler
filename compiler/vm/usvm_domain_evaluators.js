/**
 * MUDRAFORGE UNIVERSAL STATUTORY VIRTUAL MACHINE (USVM) - DOMAIN EVALUATORS
 * Implements pure evaluation handlers for advanced statutory archetypes:
 * - MSMED Section 15/16 3x Bank Compounding
 * - Section 43B(h) March 31 Fiscal Boundary Engine
 * - MSMED Section 7 Enterprise Classification
 * - CGST Rule 37 180-Day ITC Reversal & Rule 37A Supplier Non-Filing
 * - CGST Rule 42/43 Input & Capital Goods Common Credit Apportionment
 * - International Section 195/393 Cross-Border DTAA Treaty Matrix
 * - Transfer Pricing Section 92C & Rule 10CA Percentile ALP
 * - Section 79 Loss Carryforward & Voting Continuity
 * - Section 281 Tax Clearance & Void Asset Transfer
 *
 * Strict Statutory Invariant: Every evaluation stamps isAuthorizedToDisburse: false.
 */

import {
    roundToPaisa,
    roundSection288B,
    cleanNumericAmount
} from '../calculators/numeric_utils.js';

import {
    calculateMsmeCompoundInterest,
    evaluateSection43bH
} from '../calculators/msme.js';

import {
    calculateItcReversalRule37,
    calculateRule37ASupplierNonFilingReversal,
    calculateGstItcApportionmentRule42_43,
    evaluateSection52GstTcs
} from '../calculators/gst.js';

import {
    evaluateCrossBorderRemittance,
    calculateTransferPricingAlp,
    evaluateSection79LossContinuity,
    evaluateSection281TaxClearance,
    revalueForeignExchangeIndAs109
} from '../calculators/international.js';

import {
    calculateCorporateTaxMatVsConcessional,
    evaluateSection194S,
    evaluateSection194BA,
    evaluateSection194N,
    evaluateSection194DA,
    evaluateSection194K,
    evaluateSection206C1G
} from '../calculators/direct_tax.js';

import {
    evaluateSalaryTds
} from '../calculators/payroll_statutory.js';

import {
    generateMsme1AndScheduleThreeAging
} from '../calculators/msme.js';

import {
    calculateRevenueAllocationIndAs115,
    calculateLeaseAmortizationIndAs116,
    calculateExpectedCreditLossIndAs109,
    calculateEsopAmortizationIndAs102,
    calculateDeferredTaxIndAs12,
    evaluateProvisionsAndContingenciesIndAs37,
    calculateForeignCurrencyIndAs21,
    calculateImpairmentIndAs36,
    calculateBusinessCombinationsIndAs103,
    calculateConsolidationNciIndAs110,
    calculateEquityMethodIndAs28,
    calculateEmployeeBenefitsIndAs19,
    calculateInventoryValuationIndAs2,
    calculateEarningsPerShareIndAs33,
    calculateCashFlowsIndAs7,
    calculatePropertyPlantEquipmentIndAs16,
    evaluateFairValueHierarchyIndAs113,
    evaluateIntangiblesCapitalizationIndAs38
} from '../calculators/global_accounting_calculators.js';

/**
 * Evaluates MSMED Section 15/16 Compound Interest at 3x RBI Bank Rate.
 */
export function evaluateMsmeDelayedPaymentCompoundInterest(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating MSMED Section 15/16 delayed payments compounding`);
    const principal = facts.principal ?? facts.invoiceAmount ?? facts.consideration ?? 0;
    const rbiBankRate = facts.rbiBankRate ?? ruleAst.computation?.defaultBankRate ?? 0.0675;

    const res = calculateMsmeCompoundInterest({
        principal,
        acceptanceDate: facts.acceptanceDate || null,
        deliveryDate: facts.deliveryDate || null,
        writtenObjectionDate: facts.writtenObjectionDate || null,
        objectionRemovalDate: facts.objectionRemovalDate || null,
        agreedPaymentDate: facts.agreedPaymentDate || null,
        hasWrittenAgreement: facts.hasWrittenAgreement,
        vendorNicCode: facts.vendorNicCode || facts.nicCode || null,
        nicCode: facts.nicCode || facts.vendorNicCode || null,
        supplierType: facts.supplierType || null,
        isTrader: facts.isTrader,
        dueDate: facts.dueDate || null,
        accrualStartDate: facts.accrualStartDate || null,
        paymentDate: facts.paymentDate || facts.transactionDate || new Date().toISOString(),
        rbiBankRate,
        dateConvention: facts.dateConvention || 'START_DATE_INCLUSIVE',
        commenceDayAfter: facts.commenceDayAfter || false,
        includePaymentDate: facts.includePaymentDate || false
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false, // Fiduciary invariant
        executionTrace: trace
    };
}

/**
 * Evaluates Section 43B(h) Temporal & Fiscal Boundary State Machine.
 */
export function evaluateSection43bHFiscalBoundary(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating Section 43B(h) fiscal year boundary`);
    const res = evaluateSection43bH({
        invoiceDate: facts.invoiceDate || facts.transactionDate || facts.creditDate,
        acceptanceDate: facts.acceptanceDate || null,
        agreedTermsDays: facts.agreedTermsDays || null,
        disputeStartDate: facts.disputeStartDate || null,
        disputeEndDate: facts.disputeEndDate || null,
        paymentDate: facts.paymentDate || null,
        fyEndDate: facts.fyEndDate || null,
        supplierType: facts.supplierType || 'MANUFACTURER',
        nicCode: facts.nicCode || facts.vendorNicCode || null
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 7 MSME Enterprise Classification (Micro / Small / Medium).
 */
export function evaluateMsmeEnterpriseClassification(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating MSME Section 7 classification criteria`);
    const plantAndMachinery = Number(facts.plantAndMachineryInr ?? facts.investmentInPpmInr ?? 0);
    const turnover = Number(facts.turnoverInr ?? 0);

    let enterpriseCategory = 'LARGE';
    let statutoryBasis = '';

    // Micro: Plant/Machinery <= 1 Cr AND Turnover <= 5 Cr
    if (plantAndMachinery <= 10000000 && turnover <= 50000000) {
        enterpriseCategory = 'MICRO';
        statutoryBasis = 'Investment <= 1 Crore and Turnover <= 5 Crores under Section 7(1) of MSMED Act, 2006';
    } else if (plantAndMachinery <= 100000000 && turnover <= 500000000) {
        enterpriseCategory = 'SMALL';
        statutoryBasis = 'Investment <= 10 Crores and Turnover <= 50 Crores under Section 7(1) of MSMED Act, 2006';
    } else if (plantAndMachinery <= 500000000 && turnover <= 2500000000) {
        enterpriseCategory = 'MEDIUM';
        statutoryBasis = 'Investment <= 50 Crores and Turnover <= 250 Crores under Section 7(1) of MSMED Act, 2006';
    } else {
        enterpriseCategory = 'LARGE';
        statutoryBasis = 'Investment > 50 Crores or Turnover > 250 Crores exceeds MSME ceiling';
    }

    return {
        enterpriseCategory,
        isMicroOrSmall: enterpriseCategory === 'MICRO' || enterpriseCategory === 'SMALL',
        isMedium: enterpriseCategory === 'MEDIUM',
        isMsme: enterpriseCategory !== 'LARGE',
        plantAndMachineryInr: plantAndMachinery,
        turnoverInr: turnover,
        statutoryBasis,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Rule 37 180-Day Non-Payment Proportional ITC Reversal.
 */
export function evaluateGstRule37ProportionalReversal(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating CGST Rule 37 180-day ITC reversal`);
    const res = calculateItcReversalRule37({
        invoiceNumber: facts.invoiceNumber || 'INV-UNKNOWN',
        invoiceDate: facts.invoiceDate || facts.transactionDate,
        calculationDate: facts.calculationDate || facts.paymentDate || new Date().toISOString().slice(0, 10),
        invoiceAmount: facts.invoiceAmount ?? facts.grossAmountInr ?? 0,
        taxAmount: facts.taxAmount ?? facts.itcClaimedInr ?? 0,
        paidAmount: facts.paidAmount ?? facts.valuePaidInr ?? 0,
        itcAvailmentDate: facts.itcAvailmentDate || null,
        isRcm: facts.isRcm === true,
        isScheduleISupply: facts.isScheduleISupply === true,
        annualInterestRatePercent: facts.annualInterestRatePercent ?? ruleAst.computation?.annualInterestRatePercent ?? 18.0
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Rule 37A Supplier GSTR-3B Non-Filing Reversal.
 */
export function evaluateGstRule37ASupplierNonFiling(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating CGST Rule 37A supplier non-filing cutover`);
    const res = calculateRule37ASupplierNonFilingReversal({
        financialYear: facts.financialYear || '2024-25',
        supplierGstin: facts.supplierGstin || 'UNKNOWN',
        itcClaimedInr: facts.itcClaimedInr ?? facts.taxAmount ?? 0,
        hasSupplierFiledGstr3B: facts.hasSupplierFiledGstr3B === true,
        currentEvaluationDate: facts.currentEvaluationDate || facts.transactionDate || new Date().toISOString().slice(0, 10),
        subsequentFilingDate: facts.subsequentFilingDate || null
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Rule 42/43 Common ITC Apportionment and Exempt Ratio Reversal.
 */
export function evaluateGstRule42CommonCreditApportionment(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating CGST Rule 42/43 common credit apportionment`);
    const res = calculateGstItcApportionmentRule42_43({
        taxPeriod: facts.taxPeriod || null,
        financialYear: facts.financialYear || '2025-26',
        totalInputTaxCreditT: facts.totalInputTaxCreditT ?? facts.totalItc ?? facts.taxAmount,
        exclusiveNonBusinessT1: facts.exclusiveNonBusinessT1 || 0,
        exclusiveExemptT2: facts.exclusiveExemptT2 || 0,
        blockedSection17_5_T3: facts.blockedSection17_5_T3 || 0,
        exclusiveTaxableT4: facts.exclusiveTaxableT4 || 0,
        exemptTurnoverE: facts.exemptTurnoverE || 0,
        totalTurnoverF: facts.totalTurnoverF || 0,
        capitalGoodsCommonCreditA: facts.capitalGoodsCommonCreditA || 0,
        capitalGoodsPriorQuartersExempt: facts.capitalGoodsPriorQuartersExempt || 0,
        isAnnualTrueUp: facts.isAnnualTrueUp === true,
        annualExemptTurnover: facts.annualExemptTurnover || null,
        annualTotalTurnover: facts.annualTotalTurnover || null,
        sumOfMonthlyReversalsD1: facts.sumOfMonthlyReversalsD1 || 0,
        isRealEstateProject: facts.isRealEstateProject === true
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 195/393 Cross-Border Remittance & DTAA Form Matrix.
 */
export function evaluateCrossBorderDtaaMatrix(ruleAst, facts, temporal, trace = []) {
    trace.push(`USVM evaluating Section 195/393 cross-border remittance`);
    const res = evaluateCrossBorderRemittance({
        amountUsd: facts.amountUsd ?? (facts.amountInr ? facts.amountInr / (facts.fxRate || 83.5) : 0),
        fxRate: facts.fxRate || 83.5,
        transactionDate: facts.transactionDate || temporal?.crystallizedDate || new Date().toISOString(),
        country: facts.country || null,
        isAutomatedCloudInfra: facts.isAutomatedCloudInfra !== false,
        hasHumanTechnicalServices: facts.hasHumanTechnicalServices === true,
        hasTrc: facts.hasTrc === true,
        hasNoPe: facts.hasNoPe === true,
        hasForm10F: facts.hasForm10F === true,
        hasAoCertificate: facts.hasAoCertificate === true
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 92C & Rule 10CA Transfer Pricing Percentile Range & ALP.
 */
export function evaluateTransferPricingPercentileAlp(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating Section 92C & Rule 10CA Transfer Pricing ALP`);
    const res = calculateTransferPricingAlp({
        transactionDescription: facts.transactionDescription || 'International Transaction with Associated Enterprise',
        testedParty: facts.testedParty || 'Indian Enterprise',
        associatedEnterpriseName: facts.associatedEnterpriseName || 'Foreign AE',
        isAssociatedEnterprise: facts.isAssociatedEnterprise !== false,
        internationalTransactionAmountInr: facts.internationalTransactionAmountInr ?? facts.amountInr ?? 0,
        testedPartyOperatingMarginPercent: facts.testedPartyOperatingMarginPercent ?? facts.marginPercent ?? 0,
        isWholesaleTrading: facts.isWholesaleTrading === true,
        comparablesData: facts.comparablesData || [],
        transferPricingMethod: facts.transferPricingMethod || 'TNMM'
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 79 Loss Carryforward & Voting Continuity.
 */
export function evaluateSection79VotingContinuity(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating Section 79 voting continuity`);
    const res = evaluateSection79LossContinuity({
        companyType: facts.companyType || 'CLOSELY_HELD_COMPANY',
        isDpiitRecognizedStartup: facts.isDpiitRecognizedStartup === true,
        isIbcResolutionPlan: facts.isIbcResolutionPlan === true,
        accumulatedBroughtForwardLossInr: facts.accumulatedBroughtForwardLossInr || 0,
        unabsorbedDepreciationInr: facts.unabsorbedDepreciationInr || 0,
        shareholdingInLossYear: facts.shareholdingInLossYear || [],
        shareholdingInSetOffYear: facts.shareholdingInSetOffYear || [],
        votingPowerHeldByOriginalShareholdersPercent: facts.votingPowerHeldByOriginalShareholdersPercent ?? null
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 281 Void Transfer of Assets during Tax Proceedings.
 */
export function evaluateSection281VoidTransfer(ruleAst, facts, trace = []) {
    trace.push(`USVM evaluating Section 281 tax clearance`);
    const res = evaluateSection281TaxClearance({
        assetType: facts.assetType || 'IMMOVABLE_PROPERTY',
        assetDescription: facts.assetDescription || 'Commercial Asset / Undertaking',
        assetValueInr: facts.assetValueInr ?? facts.consideration ?? 0,
        transferNature: facts.transferNature || 'SALE',
        hasPendingProceedings: facts.hasPendingProceedings === true,
        pendingProceedings: facts.pendingProceedings || [],
        hasOutstandingDemand: facts.hasOutstandingDemand === true,
        outstandingDemandInr: facts.outstandingDemandInr || 0,
        hasPriorAoPermission: facts.hasPriorAoPermission === true,
        aoPermissionReference: facts.aoPermissionReference || null,
        aoPermissionDate: facts.aoPermissionDate || null,
        hasAdequateConsideration: facts.hasAdequateConsideration !== false,
        hasNoticeOfProceedingsOrDemand: facts.hasNoticeOfProceedingsOrDemand === true,
        alternativeSecurityFurnishedInr: facts.alternativeSecurityFurnishedInr || 0
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Corporate Tax Section 115BAA vs MAT Section 115JB.
 */
export function evaluateCorporateTaxMatVsConcessionalUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Corporate Tax Section 115BAA vs MAT 115JB');
    const res = calculateCorporateTaxMatVsConcessional({
        netProfitBeforeTaxInr: facts.netProfitBeforeTaxInr ?? facts.profitBeforeTaxInr ?? 0,
        additionsBookProfitInr: facts.additionsBookProfitInr || 0,
        deductionsBookProfitInr: facts.deductionsBookProfitInr || 0,
        deductionsChapterVIAInr: facts.deductionsChapterVIAInr || 0,
        unabsorbedDepreciationInr: facts.unabsorbedDepreciationInr || 0,
        broughtForwardLossInr: facts.broughtForwardLossInr || 0,
        turnoverPriorYearInr: facts.turnoverPriorYearInr || 0,
        isDomesticCompany: facts.isDomesticCompany !== false
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Salary TDS Section 192 / Section 392.
 */
export function evaluateSalaryProgressiveTds(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Salary TDS Section 192 / Section 392');
    const res = evaluateSalaryTds({
        annualGrossSalaryInr: facts.annualGrossSalaryInr ?? facts.grossSalaryInr ?? 0,
        regime: facts.regime || 'NEW',
        hraExemptionInr: facts.hraExemptionInr || 0,
        section80CDeductionsInr: facts.section80CDeductionsInr || 0,
        section80DDeductionsInr: facts.section80DDeductionsInr || 0,
        otherDeductionsChapterVIAInr: facts.otherDeductionsChapterVIAInr || 0,
        priorTdsDeductedInr: facts.priorTdsDeductedInr || 0,
        remainingMonthsInFy: facts.remainingMonthsInFy || 12
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 206AB Non-Filer Penal Rate.
 */
export function evaluateNonFilerPenalRate206AB(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 206AB non-filer rate');
    const txDate = String(facts.transactionDate || facts.creditDate || facts.paymentDate || '2025-01-01');
    const isOmitted = txDate >= '2025-04-01';
    const baseRatePercent = Number(facts.standardRatePercent ?? facts.baseRatePercent ?? 2.0);
    const effectiveRatePercent = isOmitted ? baseRatePercent : Math.max(5.0, baseRatePercent * 2);
    const amountInr = Number(facts.invoiceAmount ?? facts.grossAmountInr ?? 0);
    const calculatedTdsInr = roundToPaisa((amountInr * effectiveRatePercent) / 100);

    return {
        status: isOmitted ? 'SUNSET_OMITTED_FA_2025' : 'PENAL_RATE_APPLIED',
        isOmittedFinanceAct2025: isOmitted,
        baseRatePercent,
        effectiveRatePercent,
        amountInr,
        calculatedTdsInr,
        statutoryDisposition: isOmitted ? 'SECTION_206AB_OMITTED_USE_STANDARD_RATE' : 'SECTION_206AB_HIGHER_RATE_MANDATORY',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 12 Deferred Taxes.
 */
export function evaluateDeferredTaxIndAs12Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 12 Deferred Tax');
    const res = calculateDeferredTaxIndAs12({
        accountingProfitInr: facts.accountingProfitInr || 0,
        permanentDifferencesInr: facts.permanentDifferencesInr || 0,
        temporaryTaxableDifferencesInr: facts.temporaryTaxableDifferencesInr || 0,
        temporaryDeductibleDifferencesInr: facts.temporaryDeductibleDifferencesInr || 0,
        enactedTaxRatePercent: facts.enactedTaxRatePercent || 25.17,
        futureExpectedRatePercent: facts.futureExpectedRatePercent || 25.17
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 17(5) Blocked ITC.
 */
export function evaluateBlockedCreditSection17_5(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 17(5) blocked credit');
    const category = String(facts.expenseCategory || facts.category || '').toUpperCase();
    const blockedCategories = ['MOTOR_VEHICLE', 'FOOD_AND_BEVERAGES', 'OUTDOOR_CATERING', 'CLUB_MEMBERSHIP', 'HEALTH_FITNESS', 'PERSONAL_CONSUMPTION', 'LOST_STOLEN_GOODS', 'FREE_SAMPLES'];
    const isBlocked = blockedCategories.some(c => category.includes(c));
    const taxAmount = Number(facts.taxAmount ?? facts.itcAmount ?? 0);

    return {
        status: isBlocked ? 'BLOCKED_ITC_INELIGIBLE' : 'ELIGIBLE_ITC',
        isBlocked,
        blockedCategory: isBlocked ? category : null,
        itcBlockedInr: isBlocked ? taxAmount : 0,
        itcEligibleInr: isBlocked ? 0 : taxAmount,
        governingSection: 'CGST Act, 2017 Section 17(5)',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 16(2) ITC Eligibility Conditions.
 */
export function evaluateItcEligibilitySection16_2(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 16(2) ITC eligibility conditions');
    const hasTaxInvoice = facts.hasTaxInvoice !== false;
    const hasReceivedGoods = facts.hasReceivedGoods !== false;
    const isTaxPaidBySupplier = facts.isTaxPaidBySupplier !== false;
    const hasFurnishedReturn = facts.hasFurnishedReturn !== false;

    const allConditionsSatisfied = hasTaxInvoice && hasReceivedGoods && isTaxPaidBySupplier && hasFurnishedReturn;
    const taxAmount = Number(facts.taxAmount ?? facts.itcAmount ?? 0);

    return {
        status: allConditionsSatisfied ? 'ITC_ELIGIBLE' : 'ITC_INELIGIBLE_CONDITIONS_UNMET',
        isEligible: allConditionsSatisfied,
        itcEligibleInr: allConditionsSatisfied ? taxAmount : 0,
        itcIneligibleInr: allConditionsSatisfied ? 0 : taxAmount,
        failedConditions: [
            !hasTaxInvoice ? 'MISSING_TAX_INVOICE' : null,
            !hasReceivedGoods ? 'GOODS_SERVICES_NOT_RECEIVED' : null,
            !isTaxPaidBySupplier ? 'TAX_NOT_PAID_TO_GOVERNMENT' : null,
            !hasFurnishedReturn ? 'GSTR3B_RETURN_NOT_FURNISHED' : null
        ].filter(Boolean),
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 9(3) Reverse Charge Mechanism.
 */
export function evaluateRcmLiabilitySection9_3(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 9(3) Reverse Charge Mechanism');
    const serviceType = String(facts.serviceType || facts.natureOfSupply || '').toUpperCase();
    const rcmList = ['GTA', 'GOODS_TRANSPORT', 'LEGAL_SERVICES', 'ADVOCATE', 'DIRECTOR', 'RECOVERY_AGENT', 'SECURITY_SERVICES'];
    const isRcmApplicable = rcmList.some(r => serviceType.includes(r));
    const amountInr = Number(facts.invoiceAmount ?? facts.grossAmountInr ?? 0);
    const rcmRatePercent = Number(facts.rcmRatePercent || 5.0);
    const rcmTaxInr = isRcmApplicable ? roundToPaisa((amountInr * rcmRatePercent) / 100) : 0;

    return {
        status: isRcmApplicable ? 'RCM_TAX_PAYABLE_BY_RECIPIENT' : 'FORWARD_CHARGE_REGULAR',
        isRcmApplicable,
        recipientRcmLiabilityInr: rcmTaxInr,
        cashLedgerPaymentMandatory: isRcmApplicable,
        governingSection: 'CGST Act, 2017 Section 9(3)',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 8 Composite & Mixed Supplies.
 */
export function evaluateCompositeMixedSuppliesSection8(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 8 composite vs mixed supply');
    const supplyType = String(facts.supplyType || 'COMPOSITE').toUpperCase();
    const isComposite = supplyType === 'COMPOSITE';
    const amountInr = Number(facts.invoiceAmount ?? facts.grossAmountInr ?? 0);
    const applicableRatePercent = isComposite
        ? Number(facts.principalSupplyRatePercent || 18.0)
        : Number(facts.highestItemRatePercent || 28.0);

    const taxAmountInr = roundToPaisa((amountInr * applicableRatePercent) / 100);

    return {
        status: 'SUCCESS',
        supplyType: isComposite ? 'COMPOSITE_SUPPLY' : 'MIXED_SUPPLY',
        applicableRatePercent,
        taxAmountInr,
        governingSection: 'CGST Act, 2017 Section 8',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 10 Composition Levy.
 */
export function evaluateCompositionLevySection10(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 10 Composition Levy');
    const turnoverInr = Number(facts.aggregateTurnoverInr || 0);
    const isEligible = turnoverInr <= 15000000;
    const businessType = String(facts.businessType || 'TRADER').toUpperCase();
    let ratePercent = 1.0;
    if (businessType === 'RESTAURANT') ratePercent = 5.0;
    else if (businessType === 'SERVICE') ratePercent = 6.0;

    const compositionTaxInr = isEligible ? roundToPaisa((turnoverInr * ratePercent) / 100) : null;

    return {
        status: isEligible ? 'COMPOSITION_ELIGIBLE' : 'INELIGIBLE_TURNOVER_EXCEEDED',
        isEligible,
        applicableRatePercent: isEligible ? ratePercent : null,
        compositionTaxInr,
        governingSection: 'CGST Act, 2017 Section 10',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 34 Credit and Debit Note Timelines.
 */
export function evaluateCreditDebitNotesSection34(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 34 Credit Note adjustment timelines');
    const noteDate = String(facts.creditNoteDate || facts.transactionDate || '2025-01-01');
    const cutoffDate = String(facts.cutoffDate || '2025-11-30');
    const isWithinTimelines = noteDate <= cutoffDate;

    return {
        status: isWithinTimelines ? 'WITHIN_STATUTORY_DEADLINE' : 'EXPIRED_POST_NOVEMBER_CUTOFF',
        isWithinTimelines,
        creditNoteDate: noteDate,
        statutoryCutoffDate: cutoffDate,
        governingSection: 'CGST Act, 2017 Section 34(2)',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates IGST Section 16 Zero-Rated Export with LUT.
 */
export function evaluateExportLutSection16(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating IGST Section 16 export of services under LUT');
    const hasValidLut = facts.hasValidLut !== false;
    const exportAmountInr = Number(facts.exportAmountInr ?? facts.invoiceAmount ?? 0);

    return {
        status: hasValidLut ? 'ZERO_RATED_LUT_AUTHORIZED' : 'EXPORT_WITH_PAYMENT_REQUIRED',
        hasValidLut,
        exportAmountInr,
        igstPayableInr: hasValidLut ? 0 : roundToPaisa((exportAmountInr * 18) / 100),
        governingSection: 'IGST Act, 2017 Section 16',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Rule 48(4) E-Invoicing Mandate.
 */
export function evaluateEInvoicingRule48_4(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Rule 48(4) e-invoicing mandate');
    const turnoverInr = Number(facts.aggregateTurnoverInr || 0);
    const isEInvoicingMandatory = turnoverInr > 50000000; // 5 Cr

    return {
        status: isEInvoicingMandatory ? 'EINVOICING_MANDATORY_IRN_REQUIRED' : 'EXEMPT_BELOW_TURNOVER_THRESHOLD',
        isEInvoicingMandatory,
        turnoverInr,
        governingSection: 'CGST Rules, 2017 Rule 48(4)',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Rule 138 E-Way Bill Consignment Threshold.
 */
export function evaluateEWayBillRule138(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Rule 138 e-way bill threshold');
    const consignmentValueInr = Number(facts.consignmentValueInr ?? facts.invoiceAmount ?? 0);
    const isEWayBillRequired = consignmentValueInr > 50000;

    return {
        status: isEWayBillRequired ? 'EWAYBILL_GENERATION_MANDATORY' : 'EXEMPT_BELOW_50K_THRESHOLD',
        isEWayBillRequired,
        consignmentValueInr,
        governingSection: 'CGST Rules, 2017 Rule 138',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates CGST Section 51 GST TDS Deduction.
 */
export function evaluateGstTdsSection51(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating CGST Section 51 GST TDS');
    const contractValueInr = Number(facts.contractValueInr || 0);
    const invoiceAmountInr = Number(facts.invoiceAmount ?? facts.paymentAmountInr ?? 0);
    const isGovernmentRecipient = facts.isGovernmentRecipient === true || facts.isGovernmentDeductor === true;
    const isThresholdBreached = contractValueInr > 250000;
    const isDeductible = isGovernmentRecipient && isThresholdBreached;

    const cgstTdsInr = isDeductible ? roundToPaisa((invoiceAmountInr * 1) / 100) : 0;
    const sgstTdsInr = isDeductible ? roundToPaisa((invoiceAmountInr * 1) / 100) : 0;

    return {
        status: isDeductible ? 'GST_TDS_MANDATORY' : 'NOT_APPLICABLE',
        isDeductible,
        cgstTdsInr,
        sgstTdsInr,
        totalGstTdsInr: roundToPaisa(cgstTdsInr + sgstTdsInr),
        governingSection: 'CGST Act, 2017 Section 51',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates MCA Form MSME-1 Aging & Reporting.
 */
export function evaluateMcaMsme1Aging(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating MCA Form MSME-1 45-day overdue aging');
    const res = generateMsme1AndScheduleThreeAging({
        asOfDate: facts.asOfDate || new Date().toISOString().slice(0, 10),
        payables: facts.payables || []
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates MSMED Section 22 Annual Disclosures.
 */
export function evaluateMsmeDisclosuresSection22(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating MSMED Section 22 balance sheet note disclosures');
    const principalDueInr = Number(facts.principalDueInr ?? facts.unpaidPrincipalInr ?? 0);
    const interestAccruedInr = Number(facts.interestAccruedInr ?? facts.unpaidInterestInr ?? 0);

    const disclosures = [
        'Principal and interest unpaid as at year end',
        'Interest paid u/s 16 during the year',
        'Interest due and payable for delay without payment u/s 16',
        'Interest accrued and remaining unpaid at end of each accounting year'
    ];

    return {
        status: 'SUCCESS',
        principalDueInr,
        interestAccruedInr,
        disclosuresRequired: disclosures,
        isDisclosureMandatory: true,
        governingSection: 'MSMED Act, 2006 Section 22',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 116 Lease Amortization.
 */
export function evaluateLeaseAmortizationIndAs116Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 116 Lease Amortization');
    const res = calculateLeaseAmortizationIndAs116({
        leaseTermMonths: facts.leaseTermMonths || facts.leaseTermPeriods || 36,
        annualDiscountRatePercent: facts.annualDiscountRatePercent || facts.discountRatePercent || 8.5,
        annualDiscountRate: facts.annualDiscountRate,
        monthlyPaymentInr: facts.monthlyPaymentInr || facts.monthlyLeasePaymentInr || facts.leasePaymentPerPeriodInr || 100000,
        paymentTiming: facts.paymentTiming || 'ARREARS',
        initialDirectCostsInr: facts.initialDirectCostsInr || 0,
        leaseIncentivesReceivedInr: facts.leaseIncentivesReceivedInr || 0,
        electionShortTermExemption: facts.electionShortTermExemption,
        electionLowValueExemption: facts.electionLowValueExemption
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 115 Revenue Allocation.
 */
export function evaluateRevenueAllocationIndAs115Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 115 Revenue Allocation');
    const res = calculateRevenueAllocationIndAs115({
        transactionPriceInr: facts.transactionPriceInr || 1000000,
        performanceObligations: facts.performanceObligations || [],
        variableConsiderationConstraintInr: facts.variableConsiderationConstraintInr || 0,
        significantFinancingRatePercent: facts.significantFinancingRatePercent || 0,
        financingPeriodMonths: facts.financingPeriodMonths || 0
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 37 Provisions and Contingencies.
 */
export function evaluateProvisionsAndContingenciesIndAs37Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 37 Provisions & Contingencies');
    const res = evaluateProvisionsAndContingenciesIndAs37({
        obligationType: facts.obligationType || 'LEGAL_DISPUTE',
        probabilityAssessment: facts.probabilityAssessment || 'PROBABLE',
        probabilityPercent: facts.probabilityPercent ?? 75,
        bestEstimateAmountInr: facts.bestEstimateAmountInr || 500000,
        minRangeAmountInr: facts.minRangeAmountInr || 300000,
        maxRangeAmountInr: facts.maxRangeAmountInr || 700000,
        reimbursementExpectedInr: facts.reimbursementExpectedInr || 0,
        discountRatePercent: facts.discountRatePercent || 0,
        timeHorizonYears: facts.timeHorizonYears || 1
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 21 & 109 Foreign Exchange Revaluation.
 */
export function evaluateForeignExchangeRevaluationUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 21 & 109 Foreign Exchange Revaluation');
    const res = calculateForeignCurrencyIndAs21({
        originalForeignAmount: facts.originalForeignAmount || 10000,
        originalSpotRate: facts.originalSpotRate || 82.5,
        closingSpotRate: facts.closingSpotRate || 83.5,
        isMonetaryItem: facts.isMonetaryItem !== false,
        isHedgingDesignated: facts.isHedgingDesignated === true,
        hedgeEffectivenessRatio: facts.hedgeEffectivenessRatio || 1.0,
        foreignCurrency: facts.foreignCurrency || 'USD'
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 109 Expected Credit Loss.
 */
export function evaluateExpectedCreditLossIndAs109Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 109 Expected Credit Loss');
    const res = calculateExpectedCreditLossIndAs109({
        grossCarryingAmountInr: facts.grossCarryingAmountInr || 1000000,
        isCreditImpaired: facts.isCreditImpaired === true,
        isSignificantIncreaseInCreditRisk: facts.isSignificantIncreaseInCreditRisk === true,
        pdStage1Percent: facts.pdStage1Percent || 1.5,
        pdStage2Percent: facts.pdStage2Percent || 5.0,
        lgdPercent: facts.lgdPercent || 40.0,
        forwardLookingMacroFactorPercent: facts.forwardLookingMacroFactorPercent || 1.0
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Foreign Tax Credit Relief (IRC §§ 901/904 / DTAA Art 23).
 */
export function evaluateForeignTaxCreditRelief(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Foreign Tax Credit relief limitation');
    const foreignIncomeInr = Number(facts.foreignIncomeInr || 0);
    const foreignTaxPaidInr = Number(facts.foreignTaxPaidInr || 0);
    const domesticTaxOnForeignIncomeInr = Number(facts.domesticTaxOnForeignIncomeInr || 0);

    const allowableFtcInr = Math.min(foreignTaxPaidInr, domesticTaxOnForeignIncomeInr);

    return {
        status: 'SUCCESS',
        foreignIncomeInr,
        foreignTaxPaidInr,
        domesticTaxOnForeignIncomeInr,
        allowableForeignTaxCreditInr: allowableFtcInr,
        excessUncreditedForeignTaxInr: Math.max(0, foreignTaxPaidInr - allowableFtcInr),
        governingStatute: 'US IRC §§ 901/904 & Income-tax Act Section 90/91',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 50B Slump Sale Net Worth & Capital Gains.
 */
export function evaluateSlumpSaleSection50B(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 50B Slump Sale Net Worth');
    const slumpSaleConsiderationInr = Number(facts.slumpSaleConsiderationInr ?? facts.invoiceAmount ?? 0);
    const bookValueOfAssetsInr = Number(facts.bookValueOfAssetsInr || 0);
    const liabilitiesInr = Number(facts.liabilitiesInr || 0);
    const netWorthInr = facts.netWorthInr !== undefined
        ? Number(facts.netWorthInr)
        : Math.max(0, bookValueOfAssetsInr - liabilitiesInr);
    const capitalGainInr = Math.max(0, slumpSaleConsiderationInr - netWorthInr);
    const isLongTerm = facts.holdingPeriodMonths !== undefined
        ? facts.holdingPeriodMonths >= 36
        : (facts.undertakingAgeYears !== undefined ? facts.undertakingAgeYears >= 3 : true);

    return {
        status: 'SUCCESS',
        slumpSaleConsiderationInr,
        netWorthInr,
        capitalGainInr,
        isLongTermCapitalGain: isLongTerm,
        governingSection: 'Income-tax Act, 1961 Section 50B read with Rule 11UAE',
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 103 Business Combinations.
 */
export function evaluateBusinessCombinationsIndAs103Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 103 Business Combinations');
    const res = calculateBusinessCombinationsIndAs103({
        purchaseConsiderationInr: facts.purchaseConsiderationInr || 10000000,
        fairValueOfNetIdentifiableAssetsInr: facts.fairValueOfNetIdentifiableAssetsInr || 8000000,
        nonControllingInterestValueInr: facts.nonControllingInterestValueInr || 0,
        nciValuationMethod: facts.nciValuationMethod || 'PROPORTIONATE_SHARE',
        previouslyHeldEquityFairValueInr: facts.previouslyHeldEquityFairValueInr || 0,
        acquisitionRelatedCostsInr: facts.acquisitionRelatedCostsInr || 0
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 36 Impairment of Assets.
 */
export function evaluateImpairmentIndAs36Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 36 Impairment of Assets');
    const res = calculateImpairmentIndAs36({
        carryingAmountCguInr: facts.carryingAmountCguInr || 5000000,
        fairValueLessCostsOfDisposalInr: facts.fairValueLessCostsOfDisposalInr || 4000000,
        valueInUseInr: facts.valueInUseInr || 4500000,
        allocatedGoodwillInr: facts.allocatedGoodwillInr || 500000,
        cguAssetsList: facts.cguAssetsList || []
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 102 Share-Based Payments (ESOP).
 */
export function evaluateShareBasedPaymentsIndAs102Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 102 Share-Based Payments (ESOP)');
    const res = calculateEsopAmortizationIndAs102({
        optionsGrantedCount: facts.optionsGrantedCount || 10000,
        grantDateFairValuePerOptionInr: facts.grantDateFairValuePerOptionInr || 50,
        vestingPeriodYears: facts.vestingPeriodYears || 3,
        expectedForfeitureRatePercent: facts.expectedForfeitureRatePercent || 5,
        exercisePriceInr: facts.exercisePriceInr || 100,
        currentMarketPriceInr: facts.currentMarketPriceInr || 150
    });

    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 7 / IAS 7 Statement of Cash Flows.
 */
export function evaluateCashFlowsIndAs7Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 7 Statement of Cash Flows');
    const res = calculateCashFlowsIndAs7(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 16 / IAS 16 Property, Plant and Equipment.
 */
export function evaluatePropertyPlantEquipmentIndAs16Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 16 Property, Plant and Equipment');
    const res = calculatePropertyPlantEquipmentIndAs16(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 19 / IAS 19 Employee Benefits (Defined Benefit Obligations).
 */
export function evaluateEmployeeBenefitsIndAs19Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 19 Employee Benefits');
    const res = calculateEmployeeBenefitsIndAs19(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 2 / IAS 2 Inventories (Lower of Cost or NRV).
 */
export function evaluateInventoryValuationIndAs2Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 2 Inventories Valuation');
    const res = calculateInventoryValuationIndAs2(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 28 / IAS 28 Investments in Associates (Equity Method).
 */
export function evaluateEquityMethodIndAs28Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 28 Investments in Associates');
    const res = calculateEquityMethodIndAs28(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 33 / IAS 33 Earnings Per Share (Basic & Diluted EPS).
 */
export function evaluateEarningsPerShareIndAs33Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 33 Earnings Per Share');
    const res = calculateEarningsPerShareIndAs33(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 38 / IAS 38 Intangible Assets Capitalization.
 */
export function evaluateIntangiblesCapitalizationIndAs38Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 38 Intangibles Capitalization');
    const res = evaluateIntangiblesCapitalizationIndAs38(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 110 / IFRS 10 Consolidated Financial Statements & NCI.
 */
export function evaluateConsolidationNciIndAs110Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 110 Consolidated Financial Statements & NCI');
    const res = calculateConsolidationNciIndAs110(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Ind AS 113 / IFRS 13 Fair Value Measurement Hierarchy.
 */
export function evaluateFairValueHierarchyIndAs113Usvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Ind AS 113 Fair Value Measurement Hierarchy');
    const res = evaluateFairValueHierarchyIndAs113(facts);
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 194S TDS on Virtual Digital Assets (Crypto / VDA).
 */
export function evaluateSection194SUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 194S TDS on Virtual Digital Assets');
    const res = evaluateSection194S({
        considerationInr: facts.considerationInr || facts.invoiceAmount || facts.grossAmountInr || 0,
        cumulativePriorConsiderationInr: facts.cumulativePriorConsiderationInr || facts.priorConsiderationInr || 0,
        isSpecifiedPerson: facts.isSpecifiedPerson || false,
        hasPan: facts.hasPan !== undefined ? facts.hasPan : (facts.payee?.isPanValid !== undefined ? facts.payee.isPanValid : true),
        isPanInoperative: facts.isPanInoperative || false,
        isSellerResident: facts.isSellerResident !== undefined ? facts.isSellerResident : true,
        transactionDate: facts.transactionDate || facts.effectiveDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 194BA TDS on Winnings from Online Games.
 */
export function evaluateSection194BAUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 194BA TDS on Online Gaming Winnings');
    const res = evaluateSection194BA({
        netWinningsInr: facts.netWinningsInr,
        withdrawalAmountInr: facts.withdrawalAmountInr || facts.grossAmountInr || facts.invoiceAmount,
        totalDepositsInr: facts.totalDepositsInr || 0,
        openingBalanceInr: facts.openingBalanceInr || 0,
        closingBalanceInr: facts.closingBalanceInr || 0,
        totalPriorWithdrawalsInr: facts.totalPriorWithdrawalsInr || 0,
        hasPan: facts.hasPan !== undefined ? facts.hasPan : (facts.payee?.isPanValid !== undefined ? facts.payee.isPanValid : true),
        isPanInoperative: facts.isPanInoperative || false,
        transactionDate: facts.transactionDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 194N TDS on High-Value Cash Withdrawals.
 */
export function evaluateSection194NUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 194N TDS on Cash Withdrawals');
    const res = evaluateSection194N({
        withdrawalAmountInr: facts.withdrawalAmountInr || facts.grossAmountInr || facts.invoiceAmount || 0,
        cumulativePriorWithdrawalsInr: facts.cumulativePriorWithdrawalsInr || facts.priorWithdrawalsInr || 0,
        isNonFilerPast3Years: facts.isNonFilerPast3Years || false,
        isExemptEntity: facts.isExemptEntity || false,
        transactionDate: facts.transactionDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 194DA TDS on Life Insurance Policy Payouts.
 */
export function evaluateSection194DAUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 194DA TDS on Life Insurance Payouts');
    const res = evaluateSection194DA({
        payoutAmountInr: facts.payoutAmountInr || facts.grossAmountInr || facts.invoiceAmount || 0,
        totalPremiumsPaidInr: facts.totalPremiumsPaidInr || facts.premiumsPaidInr || 0,
        cumulativePriorPayoutsInr: facts.cumulativePriorPayoutsInr || 0,
        isExemptUnderSection10_10D: facts.isExemptUnderSection10_10D || false,
        hasPan: facts.hasPan !== undefined ? facts.hasPan : (facts.payee?.isPanValid !== undefined ? facts.payee.isPanValid : true),
        isPanInoperative: facts.isPanInoperative || false,
        transactionDate: facts.transactionDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 194K TDS on Mutual Fund Income.
 */
export function evaluateSection194KUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 194K TDS on Mutual Fund Income');
    const res = evaluateSection194K({
        incomeAmountInr: facts.incomeAmountInr || facts.grossAmountInr || facts.invoiceAmount || 0,
        cumulativePriorIncomeInr: facts.cumulativePriorIncomeInr || 0,
        isCapitalGainDistribution: facts.isCapitalGainDistribution || false,
        hasPan: facts.hasPan !== undefined ? facts.hasPan : (facts.payee?.isPanValid !== undefined ? facts.payee.isPanValid : true),
        isPanInoperative: facts.isPanInoperative || false,
        transactionDate: facts.transactionDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates Section 206C(1G) TCS on LRS Foreign Remittances.
 */
export function evaluateSection206C1GUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating Section 206C(1G) TCS on Foreign Remittances');
    const res = evaluateSection206C1G({
        remittanceAmountInr: facts.remittanceAmountInr || facts.grossAmountInr || facts.invoiceAmount || 0,
        cumulativePriorRemittancesInr: facts.cumulativePriorRemittancesInr || 0,
        remittanceType: facts.remittanceType || 'OTHER_REMITTANCE',
        hasPan: facts.hasPan !== undefined ? facts.hasPan : (facts.payee?.isPanValid !== undefined ? facts.payee.isPanValid : true),
        isPanInoperative: facts.isPanInoperative || false,
        transactionDate: facts.transactionDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}

/**
 * Evaluates GST Section 52 TCS by E-Commerce Operators.
 */
export function evaluateSection52GstTcsUsvm(ruleAst, facts, trace = []) {
    trace.push('USVM evaluating GST Section 52 TCS by E-Commerce Operators');
    const res = evaluateSection52GstTcs({
        grossTaxableSuppliesInr: facts.grossTaxableSuppliesInr || facts.taxableAmountInr || facts.grossAmountInr || facts.invoiceAmount || 0,
        returnedSuppliesInr: facts.returnedSuppliesInr || 0,
        isInterState: facts.isInterState || false,
        isSec9_5SpecifiedService: facts.isSec9_5SpecifiedService || false,
        transactionDate: facts.transactionDate || null
    });
    return {
        ...res,
        ruleId: ruleAst.sectionId,
        ruleVersion: ruleAst.version,
        governingStatute: ruleAst.governingStatute,
        governingSection: ruleAst.statutorySection,
        isAuthorizedToDisburse: false,
        executionTrace: trace
    };
}
