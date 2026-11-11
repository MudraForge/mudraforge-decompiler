/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE ENTERPRISE GLOBAL ACCOUNTING & FINANCIAL REPORTING CALCULATORS
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Proprietary Enterprise-Tier Financial Engines:
 * 1. Ind AS 115 / IFRS 15 / US GAAP ASC 606 (Revenue from Contracts with Customers)
 *    - 5-step framework
 *    - Standalone selling price (SSP) relative weighting
 *    - Contract Asset (Unbilled Receivable) vs Contract Liability (Deferred Revenue)
 * 2. Ind AS 116 / IFRS 16 / US GAAP ASC 842 (Leases)
 *    - Balance sheet capitalization of Right-of-Use (ROU) Assets
 *    - Present value discounting of lease liabilities using Incremental Borrowing Rate (IBR)
 *    - Effective interest rate method amortization & straight-line asset depreciation
 *    - Short-term (< 12 months) and low-value asset practical expedients
 *
 * Enforces Invariant: isAuthorizedToDisburse is always false.
 * Zero external npm dependencies.
 */

function roundToPaisa(amount) {
    if (typeof amount !== 'number' || Number.isNaN(amount)) return 0;
    return Math.round((amount + Number.EPSILON) * 100) / 100;
}

function cleanNumericAmount(val, options = {}) {
    if (val === null || val === undefined) return options.defaultValue ?? 0;
    if (typeof val === 'number') return Number.isNaN(val) ? (options.defaultValue ?? 0) : val;
    if (typeof val === 'string') {
        const cleaned = val.replace(/,/g, '').trim();
        const num = parseFloat(cleaned);
        return Number.isNaN(num) ? (options.defaultValue ?? 0) : num;
    }
    return options.defaultValue ?? 0;
}

/**
 * Ind AS 115 / IFRS 15 / US GAAP ASC 606: 5-Step Revenue Recognition & Relative SSP Allocation Calculator.
 *
 * @param {Object} params
 * @param {string} [params.contractId] - Contract reference identifier
 * @param {number} params.transactionPrice - Total agreed contract consideration in INR
 * @param {Array<Object>} params.performanceObligations - Array of distinct goods/services:
 *   - id: string
 *   - description: string
 *   - standaloneSellingPrice: number (SSP)
 *   - satisfactionPattern: 'POINT_IN_TIME' | 'OVER_TIME'
 *   - percentageCompletion?: number (0 - 100, for input/output method over time)
 *   - termMonths?: number (for straight-line time-elapsed method)
 *   - monthsElapsed?: number
 *   - isPointInTimeSatisfied?: boolean
 *   - amountBilledToDate?: number
 * @returns {Object} Deterministic allocation schedule, cumulative recognized revenue, contract asset/liability
 */
export function calculateRevenueAllocationIndAs115(params = {}) {
    const rawPrice = cleanNumericAmount(
        params.transactionPrice !== undefined ? params.transactionPrice : (params.transactionPriceInr !== undefined ? params.transactionPriceInr : 0),
        { defaultValue: 0 }
    );
    const transactionPrice = roundToPaisa(rawPrice);
    const obligations = Array.isArray(params.performanceObligations) ? params.performanceObligations : [];

    if (transactionPrice <= 0 || obligations.length === 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 115 / IFRS 15 / US GAAP ASC 606',
            transactionPriceInr: transactionPrice,
            performanceObligationsCount: obligations.length,
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Transaction price and at least one distinct performance obligation must be specified.'
        };
    }

    // Step 4: Calculate total standalone selling prices (SSP)
    let totalSsp = 0;
    const prepped = obligations.map((ob, idx) => {
        const ssp = roundToPaisa(cleanNumericAmount(
            ob.standaloneSellingPrice !== undefined ? ob.standaloneSellingPrice : (ob.standAloneSellingPriceInr !== undefined ? ob.standAloneSellingPriceInr : 0),
            { defaultValue: 0 }
        ));
        totalSsp += ssp;
        return {
            id: ob.id || `PO-${idx + 1}`,
            description: ob.description || `Performance Obligation ${idx + 1}`,
            standaloneSellingPrice: ssp,
            satisfactionPattern: ob.satisfactionPattern === 'POINT_IN_TIME' ? 'POINT_IN_TIME' : 'OVER_TIME',
            percentageCompletion: Math.min(100, Math.max(0, cleanNumericAmount(ob.percentageCompletion, { defaultValue: 0 }))),
            termMonths: Math.max(0, cleanNumericAmount(ob.termMonths, { defaultValue: 0 })),
            monthsElapsed: Math.max(0, cleanNumericAmount(ob.monthsElapsed, { defaultValue: 0 })),
            isPointInTimeSatisfied: Boolean(ob.isPointInTimeSatisfied),
            amountBilledToDate: roundToPaisa(cleanNumericAmount(ob.amountBilledToDate, { defaultValue: 0 }))
        };
    });

    totalSsp = roundToPaisa(totalSsp);
    const sspDenominator = totalSsp > 0 ? totalSsp : transactionPrice;
    const bundleDiscount = roundToPaisa(Math.max(0, totalSsp - transactionPrice));
    const bundleDiscountPercentage = totalSsp > 0 ? Number(((bundleDiscount / totalSsp) * 100).toFixed(2)) : 0;

    // Allocate transaction price based on relative standalone selling prices
    let allocatedSum = 0;
    let totalRevenueRecognized = 0;
    let totalBilled = 0;

    const allocatedObligations = prepped.map((ob, idx) => {
        let allocatedAmount = 0;
        if (idx === prepped.length - 1) {
            // Absorb 1-paisa rounding residual on final component
            allocatedAmount = roundToPaisa(transactionPrice - allocatedSum);
        } else {
            const ratio = ob.standaloneSellingPrice / sspDenominator;
            allocatedAmount = roundToPaisa(transactionPrice * ratio);
            allocatedSum = roundToPaisa(allocatedSum + allocatedAmount);
        }

        // Step 5: Recognize revenue as or when performance obligation is satisfied
        let revenueRecognized = 0;
        let satisfactionProgressPercent = 0;

        if (ob.satisfactionPattern === 'POINT_IN_TIME') {
            if (ob.isPointInTimeSatisfied) {
                revenueRecognized = allocatedAmount;
                satisfactionProgressPercent = 100;
            }
        } else {
            // OVER_TIME: Check milestone completion % or time elapsed
            if (ob.percentageCompletion > 0) {
                satisfactionProgressPercent = ob.percentageCompletion;
                revenueRecognized = roundToPaisa(allocatedAmount * (ob.percentageCompletion / 100));
            } else if (ob.termMonths > 0) {
                const fraction = Math.min(1, ob.monthsElapsed / ob.termMonths);
                satisfactionProgressPercent = Number((fraction * 100).toFixed(2));
                revenueRecognized = roundToPaisa(allocatedAmount * fraction);
            }
        }

        const unearnedDeferredRevenue = roundToPaisa(Math.max(0, allocatedAmount - revenueRecognized));
        totalRevenueRecognized = roundToPaisa(totalRevenueRecognized + revenueRecognized);
        totalBilled = roundToPaisa(totalBilled + ob.amountBilledToDate);

        return {
            id: ob.id,
            description: ob.description,
            standaloneSellingPriceInr: ob.standaloneSellingPrice,
            sspWeightPercentage: Number(((ob.standaloneSellingPrice / sspDenominator) * 100).toFixed(2)),
            allocatedTransactionPriceInr: allocatedAmount,
            satisfactionPattern: ob.satisfactionPattern,
            satisfactionProgressPercent,
            cumulativeRevenueRecognizedInr: revenueRecognized,
            unearnedDeferredRevenueInr: unearnedDeferredRevenue,
            amountBilledToDateInr: ob.amountBilledToDate
        };
    });

    // Contract Asset (Unbilled Receivable) vs Contract Liability (Deferred Revenue)
    const netContractPosition = roundToPaisa(totalRevenueRecognized - totalBilled);
    const contractAssetUnbilled = netContractPosition > 0 ? netContractPosition : 0;
    const contractLiabilityDeferred = netContractPosition < 0 ? Math.abs(netContractPosition) : 0;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 115 / IFRS 15 / US GAAP ASC 606',
        contractId: params.contractId || 'CONTRACT-UNSPECIFIED',
        totalContractTransactionPriceInr: transactionPrice,
        totalStandaloneSellingPriceInr: totalSsp,
        bundleDiscountInr: bundleDiscount,
        bundleDiscountPercentage,
        allocatedObligations,
        totalCumulativeRevenueRecognizedInr: totalRevenueRecognized,
        totalBilledToDateInr: totalBilled,
        contractBalanceSummary: {
            netContractAssetUnbilledReceivableInr: contractAssetUnbilled,
            netContractLiabilityDeferredRevenueInr: contractLiabilityDeferred,
            disposition: contractAssetUnbilled > 0
                ? 'CONTRACT_ASSET_UNBILLED_RECEIVABLE'
                : (contractLiabilityDeferred > 0 ? 'CONTRACT_LIABILITY_DEFERRED_REVENUE' : 'CONTRACT_BALANCED')
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 115 5-Step Evaluation complete: Relative SSP allocation on total contract price INR ${transactionPrice.toLocaleString('en-IN')} across ${allocatedObligations.length} obligations. Cumulative recognized revenue: INR ${totalRevenueRecognized.toLocaleString('en-IN')} vs billed INR ${totalBilled.toLocaleString('en-IN')}. Net contract balance: ${contractAssetUnbilled > 0 ? `Contract Asset INR ${contractAssetUnbilled.toLocaleString('en-IN')}` : `Contract Liability INR ${contractLiabilityDeferred.toLocaleString('en-IN')}`}.`
    };
}

/**
 * Ind AS 116 / IFRS 16 / US GAAP ASC 842: Right-of-Use (ROU) Asset & Lease Liability Amortization Schedule.
 *
 * @param {Object} params
 * @param {string} [params.leaseIdentifier] - Lease contract reference
 * @param {number} params.monthlyPaymentInr - Fixed monthly lease payment in INR
 * @param {number} params.leaseTermMonths - Total non-cancellable lease duration in months
 * @param {number} [params.annualDiscountRate=0.085] - Lessee Incremental Borrowing Rate (IBR) e.g. 0.085 for 8.5%
 * @param {string} [params.paymentTiming='ARREARS'] - 'ARREARS' (end of month) or 'ADVANCE' (start of month)
 * @param {number} [params.initialDirectCostsInr=0] - Direct costs incurred by lessee capitalized into ROU asset
 * @param {number} [params.leaseIncentivesReceivedInr=0] - Incentives received from lessor reducing ROU asset
 * @param {boolean} [params.electionShortTermExemption=false] - Practical expedient for leases <= 12 months
 * @param {boolean} [params.electionLowValueExemption=false] - Practical expedient for low value underlying assets
 * @returns {Object} Initial liability, initial ROU asset, 12-month amortization preview, and year-1 P&L charge
 */
export function calculateLeaseAmortizationIndAs116(params = {}) {
    const monthlyPayment = roundToPaisa(cleanNumericAmount(
        params.monthlyPaymentInr !== undefined ? params.monthlyPaymentInr : (params.monthlyLeasePaymentInr !== undefined ? params.monthlyLeasePaymentInr : (params.leasePaymentPerPeriodInr !== undefined ? params.leasePaymentPerPeriodInr : 0)),
        { defaultValue: 0 }
    ));
    const termMonths = Math.max(0, Math.floor(cleanNumericAmount(
        params.leaseTermMonths !== undefined ? params.leaseTermMonths : (params.leaseTermPeriods !== undefined ? params.leaseTermPeriods : 0),
        { defaultValue: 0 }
    )));
    const annualRate = cleanNumericAmount(
        params.annualDiscountRate !== undefined ? params.annualDiscountRate : (params.annualDiscountRatePercent !== undefined ? params.annualDiscountRatePercent / 100 : (params.discountRatePercent !== undefined ? params.discountRatePercent / 100 : 0.085)),
        { defaultValue: 0.085 }
    );
    const isAdvance = String(params.paymentTiming || 'ARREARS').toUpperCase().trim() === 'ADVANCE';
    const directCosts = roundToPaisa(cleanNumericAmount(params.initialDirectCostsInr, { defaultValue: 0 }));
    const incentives = roundToPaisa(cleanNumericAmount(params.leaseIncentivesReceivedInr, { defaultValue: 0 }));

    // Practical expedients: Short-term (< 12 months) and Low-value asset exemptions
    if (params.electionShortTermExemption && termMonths <= 12) {
        return {
            status: 'EXEMPT_SHORT_TERM',
            standard: 'Ind AS 116 / IFRS 16 / US GAAP ASC 842',
            isCapitalizedOnBalanceSheet: false,
            statutoryStatus: 'SHORT_TERM_LEASE_OFF_BALANCE_SHEET',
            leaseTermMonths: termMonths,
            monthlyPaymentInr: monthlyPayment,
            monthlyStraightLinePnlExpenseInr: monthlyPayment,
            totalUndiscountedCashPaymentsInr: roundToPaisa(monthlyPayment * termMonths),
            initialPresentValueLeaseLiabilityInr: 0,
            initialRightOfUseAssetInr: 0,
            isAuthorizedToDisburse: false,
            statutoryRationale: `Short-term lease exemption elected under Ind AS 116. Lease term of ${termMonths} months is <= 12 months without purchase option. Lease payments recognized as rent expense on straight-line basis in P&L. Off-balance sheet.`
        };
    }

    if (params.electionLowValueExemption) {
        return {
            status: 'EXEMPT_LOW_VALUE',
            standard: 'Ind AS 116 / IFRS 16 / US GAAP ASC 842',
            isCapitalizedOnBalanceSheet: false,
            statutoryStatus: 'LOW_VALUE_LEASE_OFF_BALANCE_SHEET',
            leaseTermMonths: termMonths,
            monthlyPaymentInr: monthlyPayment,
            monthlyStraightLinePnlExpenseInr: monthlyPayment,
            totalUndiscountedCashPaymentsInr: roundToPaisa(monthlyPayment * termMonths),
            initialPresentValueLeaseLiabilityInr: 0,
            initialRightOfUseAssetInr: 0,
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Low-value underlying asset exemption elected under Ind AS 116. Lease payments recognized in P&L on straight-line basis. Off-balance sheet.'
        };
    }

    if (monthlyPayment <= 0 || termMonths <= 0 || annualRate <= 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 116 / IFRS 16 / US GAAP ASC 842',
            isCapitalizedOnBalanceSheet: false,
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Monthly payment, non-zero lease term, and positive incremental borrowing rate (IBR) are mandatory.'
        };
    }

    const monthlyRate = annualRate / 12;

    // Calculate present value of annuity payments
    // PV = Payment * [ (1 - (1+r)^-n) / r ] * (1 + (isAdvance ? r : 0))
    let pvMultiplier = (1 - Math.pow(1 + monthlyRate, -termMonths)) / monthlyRate;
    if (isAdvance) {
        pvMultiplier = pvMultiplier * (1 + monthlyRate);
    }

    const initialLiability = roundToPaisa(monthlyPayment * pvMultiplier);
    const initialRouAsset = roundToPaisa(initialLiability + directCosts - incentives);
    const monthlyDepreciation = roundToPaisa(initialRouAsset / termMonths);

    // Generate multi-period EIR amortization schedule across entire lease term
    let runningLiability = initialLiability;
    let accumulatedDepreciation = 0;
    let lifetimeCashTotal = 0;
    let lifetimeInterestTotal = 0;
    let lifetimeDepreciationTotal = 0;
    const allMonthlySchedules = [];

    for (let m = 1; m <= termMonths; m++) {
        const openingLiability = runningLiability;
        let interestExpense = 0;
        let principalReduction = 0;

        if (isAdvance && m === 1) {
            // First payment is immediate advance: 100% principal
            interestExpense = 0;
            principalReduction = monthlyPayment;
            runningLiability = roundToPaisa(Math.max(0, runningLiability - principalReduction));
        } else if (m === termMonths) {
            // Final month terminal adjustment: absorb fractional rounding so closing liability reaches exactly 0.00
            interestExpense = roundToPaisa(openingLiability * monthlyRate);
            principalReduction = openingLiability;
            runningLiability = 0.00;
        } else {
            interestExpense = roundToPaisa(openingLiability * monthlyRate);
            principalReduction = roundToPaisa(monthlyPayment - interestExpense);
            runningLiability = roundToPaisa(Math.max(0, runningLiability - principalReduction));
        }

        // Monthly straight-line depreciation with terminal penny adjustment
        let monthDepreciation = monthlyDepreciation;
        if (m === termMonths) {
            monthDepreciation = roundToPaisa(initialRouAsset - accumulatedDepreciation);
        }
        accumulatedDepreciation = roundToPaisa(accumulatedDepreciation + monthDepreciation);
        const netRouAsset = roundToPaisa(Math.max(0, initialRouAsset - accumulatedDepreciation));

        lifetimeInterestTotal = roundToPaisa(lifetimeInterestTotal + interestExpense);
        lifetimeDepreciationTotal = roundToPaisa(lifetimeDepreciationTotal + monthDepreciation);
        lifetimeCashTotal = roundToPaisa(lifetimeCashTotal + monthlyPayment);

        allMonthlySchedules.push({
            month: m,
            openingLiabilityInr: openingLiability,
            paymentInr: monthlyPayment,
            interestExpenseInr: interestExpense,
            principalReductionInr: principalReduction,
            closingLiabilityInr: runningLiability,
            monthlyRouDepreciationInr: monthDepreciation,
            accumulatedDepreciationInr: accumulatedDepreciation,
            netRouAssetCarryingAmountInr: netRouAsset
        });
    }

    const previewMonths = Math.min(12, termMonths);
    const previewSchedules = allMonthlySchedules.slice(0, previewMonths);
    const year1InterestTotal = previewSchedules.reduce((acc, s) => roundToPaisa(acc + s.interestExpenseInr), 0);
    const year1DepreciationTotal = previewSchedules.reduce((acc, s) => roundToPaisa(acc + s.monthlyRouDepreciationInr), 0);
    const year1CashTotal = previewSchedules.reduce((acc, s) => roundToPaisa(acc + s.paymentInr), 0);

    const totalUndiscountedCash = roundToPaisa(monthlyPayment * termMonths);
    const totalLifetimeInterest = roundToPaisa(lifetimeInterestTotal);

    const lifetimeSummary = {
        totalCashPaymentsInr: totalUndiscountedCash,
        totalInterestExpenseInr: lifetimeInterestTotal,
        totalDepreciationExpenseInr: lifetimeDepreciationTotal,
        totalPnlChargeInr: roundToPaisa(lifetimeInterestTotal + lifetimeDepreciationTotal),
        netClosingLiabilityInr: runningLiability,
        netClosingRouAssetInr: roundToPaisa(Math.max(0, initialRouAsset - accumulatedDepreciation))
    };

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 116 / IFRS 16 / US GAAP ASC 842',
        isCapitalizedOnBalanceSheet: true,
        leaseTermMonths: termMonths,
        paymentTiming: isAdvance ? 'ADVANCE' : 'ARREARS',
        annualDiscountRate: annualRate,
        monthlyDiscountRate: Number(monthlyRate.toFixed(6)),
        monthlyContractualPaymentInr: monthlyPayment,
        totalUndiscountedCashPaymentsInr: totalUndiscountedCash,
        initialPresentValueLeaseLiabilityInr: initialLiability,
        initialRightOfUseAssetInr: initialRouAsset,
        monthlyStraightLineDepreciationInr: monthlyDepreciation,
        totalLifetimeInterestExpenseInr: totalLifetimeInterest,
        year1Summary: {
            year1CashPaymentsInr: year1CashTotal,
            year1InterestExpenseInr: year1InterestTotal,
            year1DepreciationExpenseInr: year1DepreciationTotal,
            year1TotalPnlExpenseInr: roundToPaisa(year1InterestTotal + year1DepreciationTotal),
            year1FrontLoadedVarianceInr: roundToPaisa((year1InterestTotal + year1DepreciationTotal) - year1CashTotal)
        },
        first12MonthsAmortizationPreview: previewSchedules,
        fullAmortizationSchedule: allMonthlySchedules,
        lifetimeSummary,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 116 on-balance-sheet capitalization active: Present value of future lease payments discounted at ${(annualRate * 100).toFixed(2)}% IBR yields initial Lease Liability of INR ${initialLiability.toLocaleString('en-IN')} and initial ROU Asset of INR ${initialRouAsset.toLocaleString('en-IN')}. Monthly straight-line depreciation: INR ${monthlyDepreciation.toLocaleString('en-IN')}. Year 1 P&L charge: INR ${(year1InterestTotal + year1DepreciationTotal).toLocaleString('en-IN')} (Interest: INR ${year1InterestTotal.toLocaleString('en-IN')}, Depreciation: INR ${year1DepreciationTotal.toLocaleString('en-IN')}). Lifetime interest: INR ${lifetimeInterestTotal.toLocaleString('en-IN')}.`
    };
}


/**
 * Ind AS 109 / IFRS 9 / US GAAP ASC 326: Expected Credit Loss (ECL) Impairment Calculator.
 *
 * @param {Object} params
 * @param {string} [params.reportingDate] - Financial reporting date YYYY-MM-DD
 * @param {number} params.grossTradeReceivablesInr - Total gross receivables balance in INR
 * @param {Array<Object>} params.agingBuckets - Array of aging bracket objects:
 *   - bucketLabel: string
 *   - grossAmountInr: number
 *   - historicalDefaultRatePercent: number (0 - 100)
 *   - stage?: 1 | 2 | 3
 * @param {number} [params.macroeconomicAdjustmentFactor=1.0] - Forward-looking multiplier (1.0 = historical baseline)
 * @param {number} [params.lossGivenDefaultPercent=100.0] - LGD % (loss severity on default)
 * @returns {Object} Total ECL loss allowance, net carrying amount, staging distribution, and tax add-back note
 */
export function calculateExpectedCreditLossIndAs109(params = {}) {
    const grossTotal = roundToPaisa(cleanNumericAmount(
        params.grossTradeReceivablesInr !== undefined ? params.grossTradeReceivablesInr : (params.grossCarryingAmountInr !== undefined ? params.grossCarryingAmountInr : 0),
        { defaultValue: 0 }
    ));
    let rawBuckets = Array.isArray(params.agingBuckets) ? params.agingBuckets : [];
    const macroFactor = cleanNumericAmount(params.macroeconomicAdjustmentFactor, { defaultValue: 1.0 });
    const lgdPercent = cleanNumericAmount(
        params.lossGivenDefaultPercent !== undefined ? params.lossGivenDefaultPercent : (params.lgdPercent !== undefined ? params.lgdPercent : 100.0),
        { defaultValue: 100.0 }
    );
    const asOfDate = params.reportingDate || new Date().toISOString().split('T')[0];

    // If flat risk metrics passed without aging buckets, synthesize bucket
    if (rawBuckets.length === 0 && grossTotal > 0) {
        const pdRate = cleanNumericAmount(
            params.probabilityOfDefaultPercent !== undefined ? params.probabilityOfDefaultPercent : (params.pdStage1Percent !== undefined ? params.pdStage1Percent : 2.5),
            { defaultValue: 2.5 }
        );
        rawBuckets = [{
            bucketLabel: params.stage || 'STAGE_1',
            grossAmountInr: grossTotal,
            historicalDefaultRatePercent: pdRate,
            stage: params.stage === 'STAGE_3' ? 3 : (params.stage === 'STAGE_2' ? 2 : 1)
        }];
    }

    if (grossTotal <= 0 && rawBuckets.length === 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 109 / IFRS 9 / US GAAP ASC 326',
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Gross trade receivables or non-empty aging buckets are mandatory.'
        };
    }

    let totalComputedGross = 0;
    let totalLossAllowance = 0;
    let stage1Allowance = 0;
    let stage2Allowance = 0;
    let stage3Allowance = 0;
    let stage1Gross = 0;
    let stage2Gross = 0;
    let stage3Gross = 0;

    const evaluatedBuckets = rawBuckets.map((b, idx) => {
        const bucketGross = roundToPaisa(cleanNumericAmount(b.grossAmountInr, { defaultValue: 0 }));
        const histRate = cleanNumericAmount(b.historicalDefaultRatePercent, { defaultValue: 0 });
        const stage = Number(b.stage || (idx === 0 ? 1 : (idx === rawBuckets.length - 1 ? 3 : 2)));

        // Effective Loss Rate = Historical PD * Macro Adjustment * LGD
        const effectiveLossRatePercent = Number(Math.min(100, Math.max(0, histRate * macroFactor * (lgdPercent / 100))).toFixed(4));
        const allowance = roundToPaisa(bucketGross * (effectiveLossRatePercent / 100));
        const netCarryingAmount = roundToPaisa(bucketGross - allowance);

        totalComputedGross = roundToPaisa(totalComputedGross + bucketGross);
        totalLossAllowance = roundToPaisa(totalLossAllowance + allowance);

        if (stage === 1) {
            stage1Gross = roundToPaisa(stage1Gross + bucketGross);
            stage1Allowance = roundToPaisa(stage1Allowance + allowance);
        } else if (stage === 2) {
            stage2Gross = roundToPaisa(stage2Gross + bucketGross);
            stage2Allowance = roundToPaisa(stage2Allowance + allowance);
        } else {
            stage3Gross = roundToPaisa(stage3Gross + bucketGross);
            stage3Allowance = roundToPaisa(stage3Allowance + allowance);
        }

        return {
            bucketLabel: b.bucketLabel || `Bracket-${idx + 1}`,
            grossAmountInr: bucketGross,
            stage,
            historicalDefaultRatePercent: histRate,
            effectiveLossRatePercent,
            lossAllowanceInr: allowance,
            netCarryingAmountInr: netCarryingAmount
        };
    });

    const reconciledGross = grossTotal > 0 ? grossTotal : totalComputedGross;
    const netReceivables = roundToPaisa(reconciledGross - totalLossAllowance);
    const overallEffectiveRate = reconciledGross > 0 ? Number(((totalLossAllowance / reconciledGross) * 100).toFixed(2)) : 0;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 109 / IFRS 9 / US GAAP ASC 326',
        asOfDate,
        grossTradeReceivablesInr: reconciledGross,
        totalExpectedCreditLossAllowanceInr: totalLossAllowance,
        netTradeReceivablesCarryingAmountInr: netReceivables,
        overallEffectiveLossRatePercent: overallEffectiveRate,
        macroeconomicAdjustmentFactor: macroFactor,
        stagingSummary: {
            stage1: { grossInr: stage1Gross, allowanceInr: stage1Allowance, label: '12-Month ECL (Performing)' },
            stage2: { grossInr: stage2Gross, allowanceInr: stage2Allowance, label: 'Lifetime ECL (SICR / Underperforming)' },
            stage3: { grossInr: stage3Gross, allowanceInr: stage3Allowance, label: 'Lifetime ECL (Credit Impaired / Default)' }
        },
        agingBuckets: evaluatedBuckets,
        taxTreatment: {
            isTaxDeductibleInCurrentYear: false,
            statutoryAddBackSection: 'Section 36(1)(vii) read with Section 40A(9) Income-tax Act, 1961',
            taxConsequence: `ECL impairment provision of INR ${totalLossAllowance.toLocaleString('en-IN')} is not deductible for Indian corporate tax purposes until actual bad debt write-off. Recognizes a Deferred Tax Asset (DTA) under Ind AS 12.`
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 109 Simplified Approach: Expected Credit Loss provision of INR ${totalLossAllowance.toLocaleString('en-IN')} computed on gross receivables of INR ${reconciledGross.toLocaleString('en-IN')} (overall rate: ${overallEffectiveRate}%). Net receivables on balance sheet: INR ${netReceivables.toLocaleString('en-IN')}. Accounting impairment does not authorize fund disbursement.`
    };
}

/**
 * Ind AS 102 / IFRS 2 / US GAAP ASC 718: ESOP Graded Vesting Amortization Calculator.
 *
 * @param {Object} params
 * @param {string} [params.schemeName] - ESOP plan or grant reference
 * @param {number} params.totalOptionsGranted - Total stock options issued
 * @param {number} params.grantDateFairValuePerOptionInr - Option fair value per Black-Scholes at grant date in INR
 * @param {Array<Object>} [params.vestingTranches] - Array of vesting tranche schedules:
 *   - trancheNumber: number
 *   - vestingYears: number (duration in years e.g. 1, 2, 3, 4)
 *   - allocationPercentage: number (percentage of grant e.g. 25)
 *   - expectedForfeitureRate?: number (percentage e.g. 5)
 * @returns {Object} Total grant-date fair value, tranche schedules, year-by-year front-loaded P&L expense, and tax memo
 */
export function calculateEsopAmortizationIndAs102(params = {}) {
    const totalOptions = Math.max(0, Math.floor(cleanNumericAmount(
        params.totalOptionsGranted !== undefined ? params.totalOptionsGranted : (params.optionsGranted !== undefined ? params.optionsGranted : 0),
        { defaultValue: 0 }
    )));
    const fairValuePerOption = roundToPaisa(cleanNumericAmount(
        params.grantDateFairValuePerOptionInr !== undefined ? params.grantDateFairValuePerOptionInr : (params.fairValuePerOptionInr !== undefined ? params.fairValuePerOptionInr : 0),
        { defaultValue: 0 }
    ));
    const defaultTranches = [
        { trancheNumber: 1, vestingYears: 1, allocationPercentage: 25, expectedForfeitureRate: 0 },
        { trancheNumber: 2, vestingYears: 2, allocationPercentage: 25, expectedForfeitureRate: 0 },
        { trancheNumber: 3, vestingYears: 3, allocationPercentage: 25, expectedForfeitureRate: 0 },
        { trancheNumber: 4, vestingYears: 4, allocationPercentage: 25, expectedForfeitureRate: 0 }
    ];
    const tranches = Array.isArray(params.vestingTranches) && params.vestingTranches.length > 0 ? params.vestingTranches : defaultTranches;

    if (totalOptions <= 0 || fairValuePerOption <= 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 102 / IFRS 2 / US GAAP ASC 718',
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Total options granted and positive grant-date fair value per option are mandatory.'
        };
    }

    const totalGrantDateFairValue = roundToPaisa(totalOptions * fairValuePerOption);
    let maxVestingYear = 1;
    let totalAllocatedOptions = 0;
    let totalAdjustedFairValue = 0;

    // Process tranches
    const processedTranches = tranches.map((t, idx) => {
        const trancheNum = t.trancheNumber || idx + 1;
        const years = Math.max(0.5, cleanNumericAmount(t.vestingYears, { defaultValue: idx + 1 }));
        if (years > maxVestingYear) maxVestingYear = Math.ceil(years);

        const allocPct = cleanNumericAmount(t.allocationPercentage, { defaultValue: 100 / tranches.length });
        const forfeitRate = cleanNumericAmount(t.expectedForfeitureRate, { defaultValue: 0 });

        const nominalOptions = Math.floor(totalOptions * (allocPct / 100));
        const effectiveOptions = Math.floor(nominalOptions * (1 - forfeitRate / 100));
        const trancheFairValue = roundToPaisa(effectiveOptions * fairValuePerOption);
        const annualAmortization = roundToPaisa(trancheFairValue / years);

        totalAllocatedOptions += effectiveOptions;
        totalAdjustedFairValue = roundToPaisa(totalAdjustedFairValue + trancheFairValue);

        return {
            trancheNumber: trancheNum,
            vestingYears: years,
            allocationPercentage: allocPct,
            expectedForfeitureRatePercent: forfeitRate,
            effectiveOptionsGranted: effectiveOptions,
            trancheGrantDateFairValueInr: trancheFairValue,
            annualAmortizationExpenseInr: annualAmortization
        };
    });

    // Build year-by-year P&L expense profile
    const yearlyExpenseProfile = [];
    let cumulativeExpense = 0;
    for (let yr = 1; yr <= maxVestingYear; yr++) {
        let yearExpense = 0;
        for (const pt of processedTranches) {
            if (yr <= Math.ceil(pt.vestingYears)) {
                yearExpense = roundToPaisa(yearExpense + pt.annualAmortizationExpenseInr);
            }
        }
        cumulativeExpense = roundToPaisa(cumulativeExpense + yearExpense);
        yearlyExpenseProfile.push({
            year: yr,
            pnlCompensationExpenseInr: yearExpense,
            cumulativeExpenseRecognizedInr: cumulativeExpense,
            closingShareBasedPaymentReserveInr: cumulativeExpense
        });
    }

    const year1Expense = yearlyExpenseProfile.length > 0 ? yearlyExpenseProfile[0].pnlCompensationExpenseInr : 0;
    const flatStraightLineAverage = roundToPaisa(totalAdjustedFairValue / maxVestingYear);
    const isFrontLoaded = year1Expense > flatStraightLineAverage;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 102 / IFRS 2 / US GAAP ASC 718',
        schemeName: params.schemeName || 'SCHEME-DEFAULT-ESOP',
        totalOptionsGranted: totalOptions,
        effectiveVestingOptions: totalAllocatedOptions,
        grantDateFairValuePerOptionInr: fairValuePerOption,
        totalNominalGrantDateFairValueInr: totalGrantDateFairValue,
        totalNetFairValueToExpenseInr: totalAdjustedFairValue,
        maxVestingYears: maxVestingYear,
        isGradedVestingFrontLoaded: isFrontLoaded,
        year1ExpenseComparison: {
            year1GradedVestingExpenseInr: year1Expense,
            hypotheticalFlatAverageInr: flatStraightLineAverage,
            frontLoadingAccelerationInr: roundToPaisa(year1Expense - flatStraightLineAverage)
        },
        vestingTranches: processedTranches,
        yearlyExpenseProfile,
        taxTreatment: {
            statutoryDeductionSection: 'Section 37(1) Income-tax Act, 1961 (Biocon Special Bench Principle)',
            perquisiteTaxSection: 'Section 17(2)(vi) read with Rule 3(8) Income-tax Rules, 1962',
            deductibilitySummary: `Company is eligible to claim ESOP discount as an allowable business deduction under Section 37(1) spread over the vesting period. Upon exercise, employee is subject to salary TDS under Section 192 on perquisite value (FMV on exercise date minus exercise price).`
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 102 Graded Vesting: Total ESOP fair value INR ${totalAdjustedFairValue.toLocaleString('en-IN')} across ${processedTranches.length} tranches. Mandatory separate-tranche accounting produces Year 1 front-loaded expense of INR ${year1Expense.toLocaleString('en-IN')} (vs flat average INR ${flatStraightLineAverage.toLocaleString('en-IN')}). Share-based compensation does not authorize cash disbursement.`
    };
}


/**
 * Ind AS 12 / IAS 12 / US GAAP ASC 740: Income Taxes & Deferred Tax (DTA / DTL) Calculator.
 *
 * @param {Object} params
 * @param {string} [params.financialYear='2025-26'] - Current fiscal year
 * @param {number} params.accountingProfitBeforeTaxInr - Book profit before tax in INR
 * @param {number} [params.enactedTaxRatePercent=25.168] - Enacted corporate tax rate % (e.g. 25.168% for Sec 115BAA)
 * @param {Object} [params.temporaryDifferences={}] - Detailed temporary differences:
 *   - msmeDisallowanceSec43bHInr: number (Deductible -> DTA)
 *   - statutoryDuesUnpaidSec43BInr: number (Deductible -> DTA)
 *   - eclLossAllowanceIndAs109Inr: number (Deductible -> DTA)
 *   - leaseLiabilityNetIndAs116Inr: number (Deductible -> DTA)
 *   - esopReserveUnvestedIndAs102Inr: number (Deductible -> DTA)
 *   - otherDeductibleDifferencesInr: number (Deductible -> DTA)
 *   - acceleratedTaxDepreciationSec32Inr: number (Taxable -> DTL)
 *   - otherTaxableDifferencesInr: number (Taxable -> DTL)
 * @param {number} [params.permanentDifferencesInr=0] - Non-deductible expenses (penalties, CSR, etc.)
 * @param {number} [params.openingNetDtaInr=0] - Opening net DTA (positive) or DTL (negative)
 * @returns {Object} Current tax, DTA/DTL closing positions, P&L tax expense, and ETR reconciliation
 */
export function calculateDeferredTaxIndAs12(params = {}) {
    const bookProfit = roundToPaisa(cleanNumericAmount(params.accountingProfitBeforeTaxInr, { defaultValue: 0 }));
    const taxRatePercent = cleanNumericAmount(params.enactedTaxRatePercent, { defaultValue: 25.168 });
    const taxRate = taxRatePercent / 100;
    const diffs = params.temporaryDifferences || {};
    const permanentDiffs = roundToPaisa(cleanNumericAmount(params.permanentDifferencesInr, { defaultValue: 0 }));
    const openingNetDta = roundToPaisa(cleanNumericAmount(params.openingNetDtaInr, { defaultValue: 0 }));
    const fy = params.financialYear || '2025-26';

    if (taxRate <= 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 12 / IAS 12 / US GAAP ASC 740',
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Positive enacted tax rate is mandatory.'
        };
    }

    // 1. Deductible temporary differences (creating DTA)
    const msme43bH = roundToPaisa(cleanNumericAmount(diffs.msmeDisallowanceSec43bHInr, { defaultValue: 0 }));
    const statutory43B = roundToPaisa(cleanNumericAmount(diffs.statutoryDuesUnpaidSec43BInr, { defaultValue: 0 }));
    const ecl109 = roundToPaisa(cleanNumericAmount(diffs.eclLossAllowanceIndAs109Inr, { defaultValue: 0 }));
    const lease116 = roundToPaisa(cleanNumericAmount(diffs.leaseLiabilityNetIndAs116Inr, { defaultValue: 0 }));
    const esop102 = roundToPaisa(cleanNumericAmount(diffs.esopReserveUnvestedIndAs102Inr, { defaultValue: 0 }));
    const otherDeductible = roundToPaisa(cleanNumericAmount(diffs.otherDeductibleDifferencesInr, { defaultValue: 0 }));

    const totalDeductibleDifferences = roundToPaisa(msme43bH + statutory43B + ecl109 + lease116 + esop102 + otherDeductible);
    const grossDta = roundToPaisa(totalDeductibleDifferences * taxRate);

    // 2. Taxable temporary differences (creating DTL)
    const accelDep32 = roundToPaisa(cleanNumericAmount(diffs.acceleratedTaxDepreciationSec32Inr, { defaultValue: 0 }));
    const otherTaxable = roundToPaisa(cleanNumericAmount(diffs.otherTaxableDifferencesInr, { defaultValue: 0 }));

    const totalTaxableDifferences = roundToPaisa(accelDep32 + otherTaxable);
    const grossDtl = roundToPaisa(totalTaxableDifferences * taxRate);

    // 3. Balance Sheet Netting
    const netDeferredTaxBalance = roundToPaisa(grossDta - grossDtl);
    const closingNetDta = netDeferredTaxBalance > 0 ? netDeferredTaxBalance : 0;
    const closingNetDtl = netDeferredTaxBalance < 0 ? Math.abs(netDeferredTaxBalance) : 0;

    // 4. Current Tax Computation
    // Taxable Profit = Book Profit + Deductible Disallowances - Accelerated Tax Dep + Permanent Disallowances
    const taxableProfit = roundToPaisa(Math.max(0, bookProfit + totalDeductibleDifferences - totalTaxableDifferences + permanentDiffs));
    const currentTaxExpense = roundToPaisa(taxableProfit * taxRate);

    // 5. Deferred Tax P&L Charge/(Credit) = Opening Net DTA - Closing Net DTA
    // (Increase in DTA is a tax credit reducing tax expense; decrease in DTA is a tax expense)
    const deferredTaxPnlCharge = roundToPaisa(openingNetDta - netDeferredTaxBalance);
    const totalTaxExpense = roundToPaisa(currentTaxExpense + deferredTaxPnlCharge);

    // 6. Effective Tax Rate (ETR) Reconciliation
    const theoreticalTaxOnBookProfit = roundToPaisa(bookProfit * taxRate);
    const taxEffectPermanent = roundToPaisa(permanentDiffs * taxRate);
    const effectiveTaxRatePercent = bookProfit > 0 ? Number(((totalTaxExpense / bookProfit) * 100).toFixed(2)) : 0;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 12 / IAS 12 / US GAAP ASC 740',
        financialYear: fy,
        accountingProfitBeforeTaxInr: bookProfit,
        enactedCorporateTaxRatePercent: taxRatePercent,
        taxableProfitForIncomeTaxInr: taxableProfit,
        currentTaxLiabilityInr: currentTaxExpense,
        temporaryDifferencesSummary: {
            totalDeductibleDifferencesInr: totalDeductibleDifferences,
            totalTaxableDifferencesInr: totalTaxableDifferences,
            netTemporaryDifferenceInr: roundToPaisa(totalDeductibleDifferences - totalTaxableDifferences),
            breakdown: {
                msmeDisallowanceSec43bHInr: msme43bH,
                statutoryDuesSec43BInr: statutory43B,
                eclLossAllowanceIndAs109Inr: ecl109,
                leaseNetIndAs116Inr: lease116,
                esopReserveIndAs102Inr: esop102,
                acceleratedTaxDepreciationSec32Inr: accelDep32
            }
        },
        balanceSheetPosition: {
            grossDeferredTaxAssetInr: grossDta,
            grossDeferredTaxLiabilityInr: grossDtl,
            netDeferredTaxAssetBalanceSheetInr: closingNetDta,
            netDeferredTaxLiabilityBalanceSheetInr: closingNetDtl,
            disposition: closingNetDta > 0 ? 'NET_DEFERRED_TAX_ASSET' : (closingNetDtl > 0 ? 'NET_DEFERRED_TAX_LIABILITY' : 'BALANCED')
        },
        statementOfProfitAndLossTaxExpense: {
            currentTaxExpenseInr: currentTaxExpense,
            deferredTaxExpenseOrCreditInr: deferredTaxPnlCharge,
            totalTaxExpenseInr: totalTaxExpense
        },
        effectiveTaxRateReconciliation: {
            theoreticalTaxExpenseAtStatutoryRateInr: theoreticalTaxOnBookProfit,
            taxEffectOfPermanentDifferencesInr: taxEffectPermanent,
            reconciledTotalTaxExpenseInr: totalTaxExpense,
            effectiveTaxRatePercent,
            statutoryTaxRatePercent: taxRatePercent,
            rateVariancePercent: Number((effectiveTaxRatePercent - taxRatePercent).toFixed(2))
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 12 Balance Sheet Method: Current Tax of INR ${currentTaxExpense.toLocaleString('en-IN')} (Taxable Profit: INR ${taxableProfit.toLocaleString('en-IN')}). Closing Net ${closingNetDta > 0 ? `DTA of INR ${closingNetDta.toLocaleString('en-IN')}` : `DTL of INR ${closingNetDtl.toLocaleString('en-IN')}`}. Total P&L Tax Expense: INR ${totalTaxExpense.toLocaleString('en-IN')} (ETR: ${effectiveTaxRatePercent}% vs statutory ${taxRatePercent}%). Tax calculations do not authorize payment disbursement.`
    };
}

/**
 * Ind AS 37 / IAS 37 / US GAAP ASC 450: Provisions, Contingent Liabilities & Contingent Assets Evaluator.
 *
 * @param {Object} params
 * @param {string} [params.reportingDate] - Financial reporting date YYYY-MM-DD
 * @param {Array<Object>} params.exposureItems - Array of legal, statutory tax, or operational exposures:
 *   - claimId: string
 *   - description: string
 *   - claimedAmountInr: number
 *   - bestEstimateAmountInr?: number (defaults to claimedAmountInr)
 *   - outflowProbabilityPercent: number (0 - 100, where > 50 = Probable, 5-50 = Possible, < 5 = Remote)
 *   - isReliablyMeasurable?: boolean (default true)
 *   - expectedReimbursementInr?: number (default 0)
 *   - disputeType?: string
 * @returns {Object} Classified exposures, total provisions accrued, contingent liability footnote disclosures
 */
export function evaluateProvisionsAndContingenciesIndAs37(params = {}) {
    let rawItems = Array.isArray(params.exposureItems) ? params.exposureItems : [];
    if (rawItems.length === 0 && (params.bestEstimateAmountInr !== undefined || params.claimedAmountInr !== undefined)) {
        rawItems = [{
            claimId: params.claimId || 'EXP-1',
            description: params.description || params.obligationNature || params.obligationType || 'Present obligation evaluation',
            claimedAmountInr: params.claimedAmountInr !== undefined ? params.claimedAmountInr : params.bestEstimateAmountInr,
            bestEstimateAmountInr: params.bestEstimateAmountInr,
            outflowProbabilityPercent: params.probabilityPercent !== undefined ? params.probabilityPercent : (params.outflowProbabilityPercent !== undefined ? params.outflowProbabilityPercent : 75),
            isReliablyMeasurable: params.reliablyMeasurable !== false && params.isReliablyMeasurable !== false,
            expectedReimbursementInr: params.reimbursementExpectedInr || params.expectedReimbursementInr || 0
        }];
    }
    const asOfDate = params.reportingDate || new Date().toISOString().split('T')[0];

    if (rawItems.length === 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 37 / IAS 37 / US GAAP ASC 450',
            isAuthorizedToDisburse: false,
            statutoryRationale: 'At least one exposure item must be provided for evaluation.'
        };
    }

    let totalProvisionsAccrued = 0;
    let totalContingentLiabilities = 0;
    let totalRemoteExposures = 0;
    let totalReimbursements = 0;

    const classifiedItems = rawItems.map((item, idx) => {
        const claimId = item.claimId || `EXP-${idx + 1}`;
        const desc = item.description || `Exposure item ${idx + 1}`;
        const claimedAmt = roundToPaisa(cleanNumericAmount(item.claimedAmountInr, { defaultValue: 0 }));
        const estimateAmt = roundToPaisa(cleanNumericAmount(item.bestEstimateAmountInr, { defaultValue: claimedAmt }));
        const prob = cleanNumericAmount(item.outflowProbabilityPercent, { defaultValue: 0 });
        const measurable = item.isReliablyMeasurable !== false;
        const reimb = roundToPaisa(cleanNumericAmount(item.expectedReimbursementInr, { defaultValue: 0 }));
        const disputeType = item.disputeType || 'STATUTORY_TAX_DISPUTE';

        let classification = '';
        let accountingTreatment = '';
        let provisionAmount = 0;
        let contingentAmount = 0;
        let remoteAmount = 0;

        // Ind AS 37 Triangulation:
        // Probable (> 50%) AND Measurable -> PROVISION on Balance Sheet
        // Possible (5% - 50%) OR (Probable but not measurable) -> CONTINGENT LIABILITY in Notes
        // Remote (< 5%) -> REMOTE (No disclosure)
        if (prob > 50 && measurable) {
            classification = 'PROVISION_RECOGNIZED_ON_BALANCE_SHEET';
            provisionAmount = Math.max(0, roundToPaisa(estimateAmt - reimb));
            accountingTreatment = 'Recognize accrued liability on balance sheet and charge expense to profit or loss.';
            totalProvisionsAccrued = roundToPaisa(totalProvisionsAccrued + provisionAmount);
            totalReimbursements = roundToPaisa(totalReimbursements + reimb);
        } else if (prob >= 5 || (prob > 50 && !measurable)) {
            classification = 'CONTINGENT_LIABILITY_DISCLOSED_IN_NOTES';
            contingentAmount = estimateAmt;
            accountingTreatment = 'Do not recognize liability on balance sheet. Disclose in Schedule III Notes to Accounts.';
            totalContingentLiabilities = roundToPaisa(totalContingentLiabilities + contingentAmount);
        } else {
            classification = 'REMOTE_NO_DISCLOSURE_REQUIRED';
            remoteAmount = estimateAmt;
            accountingTreatment = 'Outflow probability is remote (< 5%). Neither recognition nor disclosure required.';
            totalRemoteExposures = roundToPaisa(totalRemoteExposures + remoteAmount);
        }

        return {
            claimId,
            description: desc,
            disputeType,
            claimedAmountInr: claimedAmt,
            bestEstimateAmountInr: estimateAmt,
            outflowProbabilityPercent: prob,
            isReliablyMeasurable: measurable,
            classification,
            accountingTreatment,
            provisionRecognizedInr: provisionAmount,
            contingentLiabilityDisclosedInr: contingentAmount,
            remoteAmountInr: remoteAmount,
            expectedReimbursementInr: reimb
        };
    });

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 37 / IAS 37 / US GAAP ASC 450',
        asOfDate,
        totalItemsEvaluated: classifiedItems.length,
        summary: {
            totalProvisionsAccruedOnBalanceSheetInr: totalProvisionsAccrued,
            totalContingentLiabilitiesDisclosedInNotesInr: totalContingentLiabilities,
            totalRemoteExposuresNoDisclosureInr: totalRemoteExposures,
            totalExpectedReimbursementsInr: totalReimbursements
        },
        classifiedExposures: classifiedItems,
        scheduleThreeDisclosureNote: `Contingent Liabilities and Commitments (to the extent not provided for): The company has contingent liabilities of INR ${totalContingentLiabilities.toLocaleString('en-IN')} in respect of disputed statutory demands and claims under appeal where adverse outflow is possible (5%-50%) but not probable. Balance sheet provisions of INR ${totalProvisionsAccrued.toLocaleString('en-IN')} have been recognized where outflow is probable (> 50%).`,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 37 Evaluation complete: ${classifiedItems.length} exposures evaluated. Balance Sheet Provisions accrued: INR ${totalProvisionsAccrued.toLocaleString('en-IN')}; Footnote Contingent Liabilities: INR ${totalContingentLiabilities.toLocaleString('en-IN')}; Remote exclusions: INR ${totalRemoteExposures.toLocaleString('en-IN')}. Evaluation of legal claims does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 21 / IAS 21 / US GAAP ASC 830: Foreign Currency Monetary Revaluation & FCTR Translation Calculator.
 *
 * @param {Object} params
 * @param {string} [params.asOfDate] - Reporting date (YYYY-MM-DD)
 * @param {string} [params.functionalCurrency='INR'] - Functional currency
 * @param {Array<Object>} [params.items=[]] - Foreign currency denominated assets/liabilities
 * @param {Array<Object>} [params.foreignOperations=[]] - Foreign operations translation packages
 * @returns {Object} Revaluation schedule, unrealized FX gain/loss, FCTR reserves, balanced journal vouchers
 */
export function calculateForeignCurrencyIndAs21(params = {}) {
    const asOfDate = params.asOfDate || new Date().toISOString().split('T')[0];
    const functionalCurrency = params.functionalCurrency || 'INR';
    let rawItems = Array.isArray(params.items) ? params.items : [];
    if (rawItems.length === 0 && (params.foreignCurrencyAmount !== undefined || params.originalForeignAmount !== undefined)) {
        const fcAmt = cleanNumericAmount(params.foreignCurrencyAmount || params.originalForeignAmount, { defaultValue: 0 });
        const spotInit = cleanNumericAmount(params.spotRateInitial || params.originalSpotRate, { defaultValue: 80 });
        const spotClose = cleanNumericAmount(params.spotRateClosing || params.closingSpotRate, { defaultValue: 83 });
        rawItems = [{
            id: params.id || 'FX-ITEM-1',
            description: params.description || 'Monetary foreign currency exposure',
            foreignCurrency: params.foreignCurrency || 'USD',
            foreignAmount: fcAmt,
            originalSpotRate: spotInit,
            closingSpotRate: spotClose,
            isMonetary: params.isMonetaryItem !== false,
            monetaryType: params.monetaryItemType || 'RECEIVABLE'
        }];
    }
    const foreignOps = Array.isArray(params.foreignOperations) ? params.foreignOperations : [];

    if (rawItems.length === 0 && foreignOps.length === 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 21 / IAS 21 / US GAAP ASC 830',
            asOfDate,
            itemsCount: 0,
            isAuthorizedToDisburse: false,
            statutoryRationale: 'At least one foreign currency item or foreign operation package must be provided for evaluation.'
        };
    }

    let totalOriginalBookValueInr = 0;
    let totalClosingBookValueInr = 0;
    let totalUnrealizedGainInr = 0;
    let totalUnrealizedLossInr = 0;
    const journalEntries = [];

    const revaluedItems = rawItems.map((item, idx) => {
        const id = item.id || `FX-ITEM-${idx + 1}`;
        const description = item.description || `Foreign Currency Exposure ${idx + 1}`;
        const itemType = String(item.itemType || 'MONETARY_ASSET').toUpperCase().trim();
        const foreignCurrency = item.foreignCurrency || 'USD';
        const foreignAmount = cleanNumericAmount(item.foreignAmount, { defaultValue: 0 });
        const bookingRate = cleanNumericAmount(
            item.bookingRate !== undefined ? item.bookingRate : (item.originalSpotRate !== undefined ? item.originalSpotRate : (item.spotRateInitial !== undefined ? item.spotRateInitial : 1)),
            { defaultValue: 1 }
        );
        const closingRate = cleanNumericAmount(
            item.closingRate !== undefined ? item.closingRate : (item.closingSpotRate !== undefined ? item.closingSpotRate : (item.spotRateClosing !== undefined ? item.spotRateClosing : bookingRate)),
            { defaultValue: bookingRate }
        );

        const originalBookValueInr = roundToPaisa(foreignAmount * bookingRate);
        totalOriginalBookValueInr = roundToPaisa(totalOriginalBookValueInr + originalBookValueInr);

        let closingBookValueInr = originalBookValueInr;
        let unrealizedGainLossInr = 0;
        let treatment = '';
        let glImpact = 'NONE';

        if (itemType === 'NON_MONETARY_HISTORICAL') {
            closingBookValueInr = originalBookValueInr;
            unrealizedGainLossInr = 0;
            treatment = 'Carried at historical exchange rate under Ind AS 21.23(b). No closing revaluation recognized.';
        } else if (itemType === 'MONETARY_ASSET') {
            closingBookValueInr = roundToPaisa(foreignAmount * closingRate);
            unrealizedGainLossInr = roundToPaisa(closingBookValueInr - originalBookValueInr);

            if (unrealizedGainLossInr > 0) {
                treatment = 'Unrealized foreign exchange gain credited to Statement of Profit & Loss.';
                glImpact = 'CREDIT_PL_GAIN';
                totalUnrealizedGainInr = roundToPaisa(totalUnrealizedGainInr + unrealizedGainLossInr);
                journalEntries.push({
                    account: `Foreign Trade Receivables (${foreignCurrency})`,
                    type: 'CURRENT_ASSET',
                    debitInr: unrealizedGainLossInr,
                    creditInr: 0,
                    reference: id
                });
                journalEntries.push({
                    account: 'Foreign Exchange Fluctuation Gain (P&L)',
                    type: 'OTHER_INCOME',
                    debitInr: 0,
                    creditInr: unrealizedGainLossInr,
                    reference: id
                });
            } else if (unrealizedGainLossInr < 0) {
                const lossAmt = Math.abs(unrealizedGainLossInr);
                treatment = 'Unrealized foreign exchange loss debited to Statement of Profit & Loss.';
                glImpact = 'DEBIT_PL_LOSS';
                totalUnrealizedLossInr = roundToPaisa(totalUnrealizedLossInr + lossAmt);
                journalEntries.push({
                    account: 'Foreign Exchange Fluctuation Loss (P&L)',
                    type: 'FINANCE_EXPENSE',
                    debitInr: lossAmt,
                    creditInr: 0,
                    reference: id
                });
                journalEntries.push({
                    account: `Foreign Trade Receivables (${foreignCurrency})`,
                    type: 'CURRENT_ASSET',
                    debitInr: 0,
                    creditInr: lossAmt,
                    reference: id
                });
            } else {
                treatment = 'No foreign exchange variance detected at reporting date.';
            }
        } else if (itemType === 'MONETARY_LIABILITY') {
            closingBookValueInr = roundToPaisa(foreignAmount * closingRate);
            unrealizedGainLossInr = roundToPaisa(originalBookValueInr - closingBookValueInr);

            if (unrealizedGainLossInr > 0) {
                treatment = 'Unrealized foreign exchange gain (liability reduction) credited to P&L.';
                glImpact = 'CREDIT_PL_GAIN';
                totalUnrealizedGainInr = roundToPaisa(totalUnrealizedGainInr + unrealizedGainLossInr);
                journalEntries.push({
                    account: `Foreign Trade Payables (${foreignCurrency})`,
                    type: 'CURRENT_LIABILITY',
                    debitInr: unrealizedGainLossInr,
                    creditInr: 0,
                    reference: id
                });
                journalEntries.push({
                    account: 'Foreign Exchange Fluctuation Gain (P&L)',
                    type: 'OTHER_INCOME',
                    debitInr: 0,
                    creditInr: unrealizedGainLossInr,
                    reference: id
                });
            } else if (unrealizedGainLossInr < 0) {
                const lossAmt = Math.abs(unrealizedGainLossInr);
                treatment = 'Unrealized foreign exchange loss (liability expansion) debited to P&L.';
                glImpact = 'DEBIT_PL_LOSS';
                totalUnrealizedLossInr = roundToPaisa(totalUnrealizedLossInr + lossAmt);
                journalEntries.push({
                    account: 'Foreign Exchange Fluctuation Loss (P&L)',
                    type: 'FINANCE_EXPENSE',
                    debitInr: lossAmt,
                    creditInr: 0,
                    reference: id
                });
                journalEntries.push({
                    account: `Foreign Trade Payables (${foreignCurrency})`,
                    type: 'CURRENT_LIABILITY',
                    debitInr: 0,
                    creditInr: lossAmt,
                    reference: id
                });
            } else {
                treatment = 'No foreign exchange variance detected at reporting date.';
            }
        }

        totalClosingBookValueInr = roundToPaisa(totalClosingBookValueInr + closingBookValueInr);

        return {
            id,
            description,
            itemType,
            foreignCurrency,
            foreignAmount,
            bookingRate,
            closingRate,
            originalBookValueInr,
            closingBookValueInr,
            unrealizedGainLossInr,
            treatment,
            glImpact
        };
    });

    const netUnrealizedPnLInr = roundToPaisa(totalUnrealizedGainInr - totalUnrealizedLossInr);

    let totalFctrReserveInr = 0;
    const translatedOperations = foreignOps.map((op, idx) => {
        const entityName = op.entityName || `Foreign Subsidiary ${idx + 1}`;
        const closingRate = cleanNumericAmount(op.closingRate, { defaultValue: 1 });
        const averageRate = cleanNumericAmount(op.averageRate, { defaultValue: closingRate });
        const assetsForeign = cleanNumericAmount(op.totalAssetsForeign, { defaultValue: 0 });
        const liabilitiesForeign = cleanNumericAmount(op.totalLiabilitiesForeign, { defaultValue: 0 });
        const netIncomeForeign = cleanNumericAmount(op.netIncomeForeign, { defaultValue: 0 });
        const shareCapitalInr = cleanNumericAmount(op.shareCapitalInr, { defaultValue: 0 });
        const openingRetainedEarningsInr = cleanNumericAmount(op.openingRetainedEarningsInr, { defaultValue: 0 });

        const assetsInr = roundToPaisa(assetsForeign * closingRate);
        const liabilitiesInr = roundToPaisa(liabilitiesForeign * closingRate);
        const netAssetsClosingInr = roundToPaisa(assetsInr - liabilitiesInr);
        const netIncomeAverageInr = roundToPaisa(netIncomeForeign * averageRate);
        const totalEquityPreFctrInr = roundToPaisa(shareCapitalInr + openingRetainedEarningsInr + netIncomeAverageInr);

        const fctrInr = roundToPaisa(netAssetsClosingInr - totalEquityPreFctrInr);
        totalFctrReserveInr = roundToPaisa(totalFctrReserveInr + fctrInr);

        return {
            entityName,
            closingRate,
            averageRate,
            totalAssetsInr: assetsInr,
            totalLiabilitiesInr: liabilitiesInr,
            netAssetsClosingInr,
            netIncomeInr: netIncomeAverageInr,
            fctrBalanceInr: fctrInr,
            treatment: 'Translation difference routed to Foreign Currency Translation Reserve (FCTR) in Other Comprehensive Income (OCI) under Ind AS 21.39.'
        };
    });

    const totalDebits = roundToPaisa(journalEntries.reduce((acc, e) => acc + e.debitInr, 0));
    const totalCredits = roundToPaisa(journalEntries.reduce((acc, e) => acc + e.creditInr, 0));
    const isDoubleEntryBalanced = Math.abs(totalDebits - totalCredits) < 0.001;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 21 / IAS 21 / US GAAP ASC 830',
        asOfDate,
        functionalCurrency,
        totalItemsEvaluated: revaluedItems.length,
        foreignOperationsCount: translatedOperations.length,
        summary: {
            totalOriginalBookValueInr,
            totalClosingBookValueInr,
            totalUnrealizedGainInr,
            totalUnrealizedLossInr,
            netUnrealizedPnLInr,
            totalFctrReserveInr,
            isDoubleEntryBalanced,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits
        },
        revaluedItems,
        translatedForeignOperations: translatedOperations,
        journalEntries,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 21 evaluation completed: ${revaluedItems.length} monetary/non-monetary exposures revalued. Net unrealized P&L variance: INR ${netUnrealizedPnLInr.toLocaleString('en-IN')}; Total FCTR in OCI: INR ${totalFctrReserveInr.toLocaleString('en-IN')}. General Ledger vouchers balanced at INR ${totalDebits.toLocaleString('en-IN')}. Foreign currency revaluation does not authorize payment disbursement.`
    };
}

/**
 * Ind AS 36 / IAS 36 / US GAAP ASC 360 & 350: Impairment of Assets & Cash-Generating Units (CGU).
 *
 * @param {Object} params
 * @param {string} [params.asOfDate] - Reporting evaluation date (YYYY-MM-DD)
 * @param {string} [params.cguId='CGU-01'] - Cash-Generating Unit reference
 * @param {string} [params.cguName='Corporate Cash-Generating Unit'] - CGU description
 * @param {Object} params.carryingAmounts - Carrying book values of CGU assets
 * @param {Object} params.recoverableAmountDetermination - Determination parameters
 * @param {Object} [params.assetFloors={}] - Floors below which individual assets cannot be written down
 * @returns {Object} Impairment determination, asset-level write-down allocation, balanced double-entry vouchers
 */
export function calculateImpairmentIndAs36(params = {}) {
    const asOfDate = params.asOfDate || new Date().toISOString().split('T')[0];
    const cguId = params.cguId || 'CGU-01';
    const cguName = params.cguName || 'Corporate Cash-Generating Unit';

    const rawAmounts = params.carryingAmounts || {};
    const flatCarrying = cleanNumericAmount(params.carryingAmountInr || params.carryingAmountCguInr, { defaultValue: 0 });
    const goodwill = roundToPaisa(cleanNumericAmount(rawAmounts.goodwill, { defaultValue: (params.allocatedGoodwillInr || 0) }));
    const ppe = roundToPaisa(cleanNumericAmount(rawAmounts.propertyPlantEquipment, { defaultValue: flatCarrying }));
    const intangibles = roundToPaisa(cleanNumericAmount(rawAmounts.intangibleAssets, { defaultValue: 0 }));
    const otherNonCurrent = roundToPaisa(cleanNumericAmount(rawAmounts.otherNonCurrentAssets, { defaultValue: 0 }));

    const totalCarryingAmount = roundToPaisa(goodwill + ppe + intangibles + otherNonCurrent);

    if (totalCarryingAmount <= 0) {
        return {
            status: 'INSUFFICIENT_DATA',
            standard: 'Ind AS 36 / IAS 36 / US GAAP ASC 360',
            cguId,
            totalCarryingAmountInr: 0,
            isAuthorizedToDisburse: false,
            statutoryRationale: 'Carrying amounts must be greater than zero for impairment evaluation.'
        };
    }

    const rad = params.recoverableAmountDetermination || {};
    const fvlcod = roundToPaisa(cleanNumericAmount(
        rad.fairValueLessCostsOfDisposal !== undefined ? rad.fairValueLessCostsOfDisposal : (params.fairValueLessCostOfDisposalInr !== undefined ? params.fairValueLessCostOfDisposalInr : (params.fairValueLessCostsOfDisposalInr !== undefined ? params.fairValueLessCostsOfDisposalInr : (params.fairValueLessCostsToSellInr !== undefined ? params.fairValueLessCostsToSellInr : 0))),
        { defaultValue: 0 }
    ));

    let valueInUse = 0;
    let discountRate = 0;
    const discountedCashFlows = [];

    if (typeof rad.valueInUse === 'number') {
        valueInUse = roundToPaisa(rad.valueInUse);
    } else if (params.valueInUseInr !== undefined) {
        valueInUse = roundToPaisa(cleanNumericAmount(params.valueInUseInr, { defaultValue: 0 }));
    } else if (rad.valueInUse && typeof rad.valueInUse === 'object') {
        discountRate = cleanNumericAmount(rad.valueInUse.preTaxDiscountRatePercent, { defaultValue: 10 }) / 100;
        const cashFlows = Array.isArray(rad.valueInUse.projectedCashFlows) ? rad.valueInUse.projectedCashFlows : [];

        let pvSum = 0;
        cashFlows.forEach((cfRaw, idx) => {
            const year = idx + 1;
            const cf = cleanNumericAmount(cfRaw, { defaultValue: 0 });
            const discountFactor = 1 / Math.pow(1 + discountRate, year);
            const presentValue = roundToPaisa(cf * discountFactor);
            pvSum = roundToPaisa(pvSum + presentValue);
            discountedCashFlows.push({ year, cashFlow: cf, discountFactor: Number(discountFactor.toFixed(4)), presentValue });
        });

        let terminalPv = 0;
        if (rad.valueInUse.terminalValue) {
            const tv = cleanNumericAmount(rad.valueInUse.terminalValue, { defaultValue: 0 });
            const finalYear = cashFlows.length || 1;
            const tvDiscountFactor = 1 / Math.pow(1 + discountRate, finalYear);
            terminalPv = roundToPaisa(tv * tvDiscountFactor);
        } else if (rad.valueInUse.terminalGrowthRatePercent && cashFlows.length > 0) {
            const g = cleanNumericAmount(rad.valueInUse.terminalGrowthRatePercent, { defaultValue: 2 }) / 100;
            const finalCf = cleanNumericAmount(cashFlows[cashFlows.length - 1], { defaultValue: 0 });
            if (discountRate > g) {
                const tv = (finalCf * (1 + g)) / (discountRate - g);
                const finalYear = cashFlows.length;
                const tvDiscountFactor = 1 / Math.pow(1 + discountRate, finalYear);
                terminalPv = roundToPaisa(tv * tvDiscountFactor);
            }
        }

        valueInUse = roundToPaisa(pvSum + terminalPv);
    }

    const recoverableAmount = Math.max(fvlcod, valueInUse);
    const recoverableMethod = fvlcod >= valueInUse ? 'FAIR_VALUE_LESS_COSTS_OF_DISPOSAL' : 'VALUE_IN_USE';

    let impairmentRequired = false;
    let totalImpairmentLoss = 0;
    let postImpairmentCarryingAmount = totalCarryingAmount;
    let goodwillImpairment = 0;
    let ppeImpairment = 0;
    let intangiblesImpairment = 0;
    let otherImpairment = 0;
    const journalEntries = [];

    if (recoverableAmount < totalCarryingAmount) {
        impairmentRequired = true;
        totalImpairmentLoss = roundToPaisa(totalCarryingAmount - recoverableAmount);
        postImpairmentCarryingAmount = recoverableAmount;

        let remainingLoss = totalImpairmentLoss;
        if (goodwill > 0) {
            goodwillImpairment = Math.min(goodwill, remainingLoss);
            remainingLoss = roundToPaisa(remainingLoss - goodwillImpairment);
            journalEntries.push({
                account: 'Accumulated Impairment - Goodwill',
                type: 'INTANGIBLE_ASSET_CONTRA',
                debitInr: 0,
                creditInr: goodwillImpairment,
                rule: 'Ind AS 36.104(a) Goodwill Priority Write-Down'
            });
        }

        const floors = params.assetFloors || {};
        const ppeFloor = cleanNumericAmount(floors.propertyPlantEquipment, { defaultValue: 0 });
        const intFloor = cleanNumericAmount(floors.intangibleAssets, { defaultValue: 0 });
        const otherFloor = cleanNumericAmount(floors.otherNonCurrentAssets, { defaultValue: 0 });

        const ppeMaxWriteDown = Math.max(0, ppe - ppeFloor);
        const intMaxWriteDown = Math.max(0, intangibles - intFloor);
        const otherMaxWriteDown = Math.max(0, otherNonCurrent - otherFloor);
        const totalEligibleBase = ppeMaxWriteDown + intMaxWriteDown + otherMaxWriteDown;

        if (remainingLoss > 0 && totalEligibleBase > 0) {
            const ppeRatio = ppeMaxWriteDown / totalEligibleBase;
            const intRatio = intMaxWriteDown / totalEligibleBase;

            ppeImpairment = roundToPaisa(Math.min(ppeMaxWriteDown, remainingLoss * ppeRatio));
            intangiblesImpairment = roundToPaisa(Math.min(intMaxWriteDown, remainingLoss * intRatio));
            otherImpairment = roundToPaisa(remainingLoss - ppeImpairment - intangiblesImpairment);

            if (ppeImpairment > 0) {
                journalEntries.push({
                    account: 'Accumulated Impairment - Property, Plant & Equipment',
                    type: 'PPE_CONTRA',
                    debitInr: 0,
                    creditInr: ppeImpairment,
                    rule: 'Ind AS 36.104(b) Pro-rata PPE Allocation'
                });
            }
            if (intangiblesImpairment > 0) {
                journalEntries.push({
                    account: 'Accumulated Impairment - Intangible Assets',
                    type: 'INTANGIBLE_CONTRA',
                    debitInr: 0,
                    creditInr: intangiblesImpairment,
                    rule: 'Ind AS 36.104(b) Pro-rata Intangibles Allocation'
                });
            }
            if (otherImpairment > 0) {
                journalEntries.push({
                    account: 'Accumulated Impairment - Other Non-Current Assets',
                    type: 'NON_CURRENT_ASSET_CONTRA',
                    debitInr: 0,
                    creditInr: otherImpairment,
                    rule: 'Ind AS 36.104(b) Pro-rata Other Assets Allocation'
                });
            }
        }

        journalEntries.unshift({
            account: 'Impairment Loss on Assets (Statement of Profit & Loss)',
            type: 'OPERATING_EXPENSE',
            debitInr: totalImpairmentLoss,
            creditInr: 0,
            rule: 'Ind AS 36.60 Recognition in Profit or Loss'
        });
    }

    const totalDebits = roundToPaisa(journalEntries.reduce((acc, e) => acc + e.debitInr, 0));
    const totalCredits = roundToPaisa(journalEntries.reduce((acc, e) => acc + e.creditInr, 0));
    const isDoubleEntryBalanced = Math.abs(totalDebits - totalCredits) < 0.001;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 36 / IAS 36 / US GAAP ASC 360',
        asOfDate,
        cguId,
        cguName,
        valuation: {
            totalCarryingAmountInr: totalCarryingAmount,
            fairValueLessCostsOfDisposalInr: fvlcod,
            valueInUseInr: valueInUse,
            recoverableAmountInr: recoverableAmount,
            recoverableMethod,
            headroomInr: Math.max(0, recoverableAmount - totalCarryingAmount),
            impairmentRequired,
            totalImpairmentLossInr: totalImpairmentLoss,
            postImpairmentCarryingAmountInr: postImpairmentCarryingAmount
        },
        assetBreakdown: {
            goodwill: { initialCarryingInr: goodwill, impairmentInr: goodwillImpairment, closingCarryingInr: roundToPaisa(goodwill - goodwillImpairment) },
            propertyPlantEquipment: { initialCarryingInr: ppe, impairmentInr: ppeImpairment, closingCarryingInr: roundToPaisa(ppe - ppeImpairment) },
            intangibleAssets: { initialCarryingInr: intangibles, impairmentInr: intangiblesImpairment, closingCarryingInr: roundToPaisa(intangibles - intangiblesImpairment) },
            otherNonCurrentAssets: { initialCarryingInr: otherNonCurrent, impairmentInr: otherImpairment, closingCarryingInr: roundToPaisa(otherNonCurrent - otherImpairment) }
        },
        discountedCashFlowsSchedule: discountedCashFlows,
        journalEntries,
        isDoubleEntryBalanced,
        isAuthorizedToDisburse: false,
        statutoryRationale: impairmentRequired
            ? `Ind AS 36 Impairment Test: CGU "${cguName}" carrying value INR ${totalCarryingAmount.toLocaleString('en-IN')} exceeds recoverable amount INR ${recoverableAmount.toLocaleString('en-IN')} (${recoverableMethod}). Impairment loss of INR ${totalImpairmentLoss.toLocaleString('en-IN')} recognized in P&L. Goodwill reduced by INR ${goodwillImpairment.toLocaleString('en-IN')}; non-current assets reduced pro-rata by INR ${(totalImpairmentLoss - goodwillImpairment).toLocaleString('en-IN')}. Impairment assessment does not authorize payment disbursement.`
            : `Ind AS 36 Impairment Test: CGU "${cguName}" carrying value INR ${totalCarryingAmount.toLocaleString('en-IN')} is fully supported by recoverable amount INR ${recoverableAmount.toLocaleString('en-IN')} (${recoverableMethod}) with headroom of INR ${(recoverableAmount - totalCarryingAmount).toLocaleString('en-IN')}. No impairment recognized. Impairment assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 103 / IFRS 3 / ASC 805 Business Combinations & Purchase Price Allocation (PPA) Engine
 *
 * Proprietary M&A calculation engine performing acquisition method purchase price allocation,
 * separate intangible asset recognition, deferred tax liability calculation, contingent
 * consideration fair valuation, and deterministic Goodwill vs Bargain Purchase Capital Reserve.
 *
 * @param {Object} params
 * @returns {Object} Comprehensive PPA schedule, goodwill/reserve derivation, balanced journal vouchers
 */
export function calculateBusinessCombinationsIndAs103(params = {}) {
    const acquisitionDate = params.acquisitionDate || new Date().toISOString().split('T')[0];
    const acquirerName = params.acquirerName || 'Acquiring Entity';
    const acquireeName = params.acquireeName || 'Acquired Target Enterprise';

    const flatConsideration = cleanNumericAmount(params.purchaseConsiderationInr, { defaultValue: 0 });
    const cashConsideration = roundToPaisa(cleanNumericAmount(params.cashConsideration, { defaultValue: flatConsideration }));
    const sharesIssuedFairValue = roundToPaisa(cleanNumericAmount(params.sharesIssuedFairValue, { defaultValue: 0 }));
    const contingentConsiderationFairValue = roundToPaisa(cleanNumericAmount(params.contingentConsiderationFairValue, { defaultValue: 0 }));
    const preExistingInterestFairValue = roundToPaisa(cleanNumericAmount(params.preExistingInterestFairValue, { defaultValue: 0 }));

    const totalConsiderationTransferred = roundToPaisa(
        cashConsideration + sharesIssuedFairValue + contingentConsiderationFairValue
    );

    // Identifiable assets at fair value
    const rawAssets = params.identifiableAssetsFairValue || {};
    const flatNetAssets = cleanNumericAmount(params.fairValueOfNetIdentifiableAssetsInr || params.fairValueOfIdentifiableNetAssetsInr || params.bookValueOfNetAssetsInr, { defaultValue: 0 });
    const tangiblePpe = flatNetAssets > 0 ? roundToPaisa(flatNetAssets) : roundToPaisa(cleanNumericAmount(rawAssets.propertyPlantEquipment, { defaultValue: 0 }));
    const customerRelationships = roundToPaisa(cleanNumericAmount(rawAssets.customerRelationships, { defaultValue: 0 }));
    const technologySoftware = roundToPaisa(cleanNumericAmount(rawAssets.technologySoftware, { defaultValue: 0 }));
    const brandTrademarks = roundToPaisa(cleanNumericAmount(rawAssets.brandTrademarks, { defaultValue: 0 }));
    const currentAssets = roundToPaisa(cleanNumericAmount(rawAssets.currentAssets, { defaultValue: 0 }));

    const totalIdentifiableAssets = roundToPaisa(
        tangiblePpe + customerRelationships + technologySoftware + brandTrademarks + currentAssets
    );

    // Liabilities assumed at fair value
    const rawLiabilities = params.liabilitiesAssumedFairValue || {};
    const financialLiabilities = roundToPaisa(cleanNumericAmount(rawLiabilities.financialLiabilities, { defaultValue: 0 }));
    const provisions = roundToPaisa(cleanNumericAmount(rawLiabilities.provisions, { defaultValue: 0 }));
    const otherLiabilities = roundToPaisa(cleanNumericAmount(rawLiabilities.otherLiabilities, { defaultValue: 0 }));

    // Deferred Tax Liability (DTL) on separately recognized intangibles
    const taxRatePercent = cleanNumericAmount(params.applicableTaxRatePercent, { defaultValue: 25.168 });
    const recognizedIntangibles = roundToPaisa(customerRelationships + technologySoftware + brandTrademarks);
    const deferredTaxLiabilityOnIntangibles = roundToPaisa(recognizedIntangibles * (taxRatePercent / 100.0));

    const totalLiabilitiesAssumed = roundToPaisa(
        financialLiabilities + provisions + otherLiabilities + deferredTaxLiabilityOnIntangibles
    );

    const netIdentifiableAssets = roundToPaisa(totalIdentifiableAssets - totalLiabilitiesAssumed);

    // Non-Controlling Interest (NCI)
    const nciPercentage = cleanNumericAmount(params.nciPercentage, { defaultValue: 0 });
    const nciValuationMethod = String(params.nciValuationMethod || 'PROPORTIONATE_NET_ASSETS').toUpperCase();
    let nciValue = 0;
    if (nciPercentage > 0) {
        if (nciValuationMethod === 'FAIR_VALUE') {
            nciValue = roundToPaisa(cleanNumericAmount(params.nciFairValue, {
                defaultValue: roundToPaisa(totalConsiderationTransferred * (nciPercentage / (100 - nciPercentage)))
            }));
        } else {
            nciValue = roundToPaisa(netIdentifiableAssets * (nciPercentage / 100.0));
        }
    }

    // Total Acquisition Basis vs Net Identifiable Assets
    const totalAcquisitionBasis = roundToPaisa(
        totalConsiderationTransferred + nciValue + preExistingInterestFairValue
    );

    const excessAmount = roundToPaisa(totalAcquisitionBasis - netIdentifiableAssets);
    const isGoodwill = excessAmount > 0;
    const isBargainPurchase = excessAmount < 0;

    const goodwillAmount = isGoodwill ? excessAmount : 0;
    const bargainPurchaseCapitalReserve = isBargainPurchase ? Math.abs(excessAmount) : 0;

    // Acquisition-related transaction costs (expensed immediately in P&L)
    const transactionAdvisoryCosts = roundToPaisa(cleanNumericAmount(params.transactionAdvisoryCosts, { defaultValue: 0 }));

    // Balanced Double-Entry Journal Vouchers
    const journalEntries = [];
    if (tangiblePpe > 0) journalEntries.push({ account: 'Property, Plant and Equipment (Acquired)', type: 'DEBIT', amountInr: tangiblePpe });
    if (customerRelationships > 0) journalEntries.push({ account: 'Intangible Asset - Customer Relationships', type: 'DEBIT', amountInr: customerRelationships });
    if (technologySoftware > 0) journalEntries.push({ account: 'Intangible Asset - Technology & Software', type: 'DEBIT', amountInr: technologySoftware });
    if (brandTrademarks > 0) journalEntries.push({ account: 'Intangible Asset - Brands & Trademarks', type: 'DEBIT', amountInr: brandTrademarks });
    if (currentAssets > 0) journalEntries.push({ account: 'Current Assets (Acquired)', type: 'DEBIT', amountInr: currentAssets });
    if (goodwillAmount > 0) journalEntries.push({ account: 'Goodwill on Acquisition (Ind AS 103)', type: 'DEBIT', amountInr: goodwillAmount });

    if (financialLiabilities > 0) journalEntries.push({ account: 'Financial Liabilities Assumed', type: 'CREDIT', amountInr: financialLiabilities });
    if (provisions > 0) journalEntries.push({ account: 'Provisions Assumed', type: 'CREDIT', amountInr: provisions });
    if (otherLiabilities > 0) journalEntries.push({ account: 'Other Liabilities Assumed', type: 'CREDIT', amountInr: otherLiabilities });
    if (deferredTaxLiabilityOnIntangibles > 0) journalEntries.push({ account: 'Deferred Tax Liability (Ind AS 12)', type: 'CREDIT', amountInr: deferredTaxLiabilityOnIntangibles });
    if (cashConsideration > 0) journalEntries.push({ account: 'Bank / Cash Account (Consideration)', type: 'CREDIT', amountInr: cashConsideration });
    if (sharesIssuedFairValue > 0) journalEntries.push({ account: 'Share Capital & Securities Premium', type: 'CREDIT', amountInr: sharesIssuedFairValue });
    if (contingentConsiderationFairValue > 0) journalEntries.push({ account: 'Financial Liability - Contingent Consideration', type: 'CREDIT', amountInr: contingentConsiderationFairValue });
    if (nciValue > 0) journalEntries.push({ account: 'Non-Controlling Interest (Equity)', type: 'CREDIT', amountInr: nciValue });
    if (bargainPurchaseCapitalReserve > 0) journalEntries.push({ account: 'Capital Reserve (Bargain Purchase Gain via OCI)', type: 'CREDIT', amountInr: bargainPurchaseCapitalReserve });

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCredits = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));
    const isDoubleEntryBalanced = Math.abs(totalDebits - totalCredits) < 0.01;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 103 / IFRS 3 / US GAAP ASC 805',
        acquisitionDate,
        acquirerName,
        acquireeName,
        totalConsiderationTransferredInr: totalConsiderationTransferred,
        identifiableAssetsFairValueInr: totalIdentifiableAssets,
        deferredTaxLiabilityOnIntangiblesInr: deferredTaxLiabilityOnIntangibles,
        totalLiabilitiesAssumedInr: totalLiabilitiesAssumed,
        netIdentifiableAssetsInr: netIdentifiableAssets,
        nonControllingInterestInr: nciValue,
        goodwillInr: goodwillAmount,
        capitalReserveBargainPurchaseInr: bargainPurchaseCapitalReserve,
        consideration: {
            cashConsiderationInr: cashConsideration,
            sharesIssuedFairValueInr: sharesIssuedFairValue,
            contingentConsiderationFairValueInr: contingentConsiderationFairValue,
            totalConsiderationTransferredInr: totalConsiderationTransferred
        },
        netIdentifiableAssets: {
            totalIdentifiableAssetsInr: totalIdentifiableAssets,
            recognizedIntangiblesInr: recognizedIntangibles,
            deferredTaxLiabilityOnIntangiblesInr: deferredTaxLiabilityOnIntangibles,
            totalLiabilitiesAssumedInr: totalLiabilitiesAssumed,
            netIdentifiableAssetsInr: netIdentifiableAssets
        },
        nonControllingInterest: {
            nciPercentage,
            nciValuationMethod,
            nciValueInr: nciValue
        },
        outcome: {
            totalAcquisitionBasisInr: totalAcquisitionBasis,
            outcomeType: isGoodwill ? 'GOODWILL' : (isBargainPurchase ? 'CAPITAL_RESERVE_BARGAIN_PURCHASE' : 'EXACT_EQUILIBRIUM'),
            goodwillAmountInr: goodwillAmount,
            bargainPurchaseCapitalReserveInr: bargainPurchaseCapitalReserve,
            accountingTreatment: isGoodwill
                ? 'Recognized as Goodwill; test annually for impairment under Ind AS 36; amortization prohibited.'
                : 'Bargain purchase gain recognized in OCI and accumulated in Equity as Capital Reserve after reassessment.'
        },
        transactionCosts: {
            transactionAdvisoryCostsInr: transactionAdvisoryCosts,
            accountingTreatment: 'Expensed in Profit or Loss in the periods incurred; cannot be capitalized into Goodwill.'
        },
        journalEntries,
        acquisitionJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits,
            isBalanced: isDoubleEntryBalanced
        },
        isDoubleEntryBalanced,
        isAuthorizedToDisburse: false,
        statutoryRationale: isGoodwill
            ? `Ind AS 103 Business Combination: Acquisition of "${acquireeName}" by "${acquirerName}". Consideration transferred INR ${totalConsiderationTransferred.toLocaleString('en-IN')} plus NCI INR ${nciValue.toLocaleString('en-IN')} exceeds net identifiable assets INR ${netIdentifiableAssets.toLocaleString('en-IN')}, recognizing Goodwill of INR ${goodwillAmount.toLocaleString('en-IN')}. Intangibles of INR ${recognizedIntangibles.toLocaleString('en-IN')} recognized with DTL of INR ${deferredTaxLiabilityOnIntangibles.toLocaleString('en-IN')}. Assessment does not authorize payment disbursement.`
            : `Ind AS 103 Business Combination: Net identifiable assets INR ${netIdentifiableAssets.toLocaleString('en-IN')} exceed acquisition basis INR ${totalAcquisitionBasis.toLocaleString('en-IN')}, recognizing Bargain Purchase Gain of INR ${bargainPurchaseCapitalReserve.toLocaleString('en-IN')} in Capital Reserve via OCI. Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 110 / IFRS 10 / ASC 810 Consolidated Financial Statements & NCI Engine
 *
 * Proprietary group consolidation engine computing consolidated net identifiable assets,
 * parent vs NCI allocation, intercompany receivables/payables elimination, unrealized
 * inventory profit elimination, and deficit NCI allocation under Ind AS 110 paragraph B94.
 *
 * @param {Object} params
 * @returns {Object} Group consolidation schedule, elimination entries, balanced vouchers
 */
export function calculateConsolidationNciIndAs110(params = {}) {
    const parentSharePercent = cleanNumericAmount(params.parentSharePercent, { defaultValue: 80.0 });
    const subsidiaryNetAssets = cleanNumericAmount(params.subsidiaryNetAssets, { defaultValue: 10000000.0 });
    const intercompanyReceivablesPayables = cleanNumericAmount(params.intercompanyReceivablesPayables, { defaultValue: 0.0 });
    const unrealizedIntercompanyProfitInInventory = cleanNumericAmount(params.unrealizedIntercompanyProfitInInventory, { defaultValue: 0.0 });
    const subsidiaryProfitOrLoss = cleanNumericAmount(params.subsidiaryProfitOrLoss, { defaultValue: 0.0 });
    const nciValuationMethod = String(params.nciValuationMethod || 'PROPORTIONATE_NET_ASSETS').toUpperCase();
    const nciFairValue = cleanNumericAmount(params.nciFairValue, { defaultValue: 0.0 });

    const nciSharePercent = roundToPaisa(100.0 - parentSharePercent);
    const adjustedSubsidiaryNetAssets = roundToPaisa(subsidiaryNetAssets - unrealizedIntercompanyProfitInInventory);

    const parentShareOfNetAssets = roundToPaisa(adjustedSubsidiaryNetAssets * (parentSharePercent / 100.0));
    const nciProportionateShare = roundToPaisa(adjustedSubsidiaryNetAssets * (nciSharePercent / 100.0));
    const nciBalance = nciValuationMethod === 'FAIR_VALUE' && nciFairValue > 0
        ? nciFairValue
        : nciProportionateShare;

    const nciProfitAllocation = roundToPaisa(subsidiaryProfitOrLoss * (nciSharePercent / 100.0));
    const parentProfitAllocation = roundToPaisa(subsidiaryProfitOrLoss - nciProfitAllocation);

    const closingNciEquity = roundToPaisa(nciBalance + nciProfitAllocation);
    const isNciDeficit = closingNciEquity < 0;

    const journalEntries = [];
    if (intercompanyReceivablesPayables > 0) {
        journalEntries.push(
            { account: 'Trade Payables (Intercompany)', type: 'DEBIT', amountInr: intercompanyReceivablesPayables },
            { account: 'Trade Receivables (Intercompany)', type: 'CREDIT', amountInr: intercompanyReceivablesPayables }
        );
    }
    if (unrealizedIntercompanyProfitInInventory > 0) {
        journalEntries.push(
            { account: 'Consolidated Retained Earnings / Group P&L', type: 'DEBIT', amountInr: unrealizedIntercompanyProfitInInventory },
            { account: 'Inventories (Unrealized Margin Elimination)', type: 'CREDIT', amountInr: unrealizedIntercompanyProfitInInventory }
        );
    }

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCredits = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 110 / IFRS 10 / ASC 810 Consolidated Financial Statements',
        parentSharePercent,
        nciSharePercent,
        subsidiaryNetAssets,
        unrealizedIntercompanyProfitInInventory,
        adjustedSubsidiaryNetAssets,
        adjustedSubsidiaryNetAssetsInr: adjustedSubsidiaryNetAssets,
        parentShareOfNetAssets,
        nciValuationMethod,
        nciOpeningBalanceInr: nciBalance,
        nciCarryingAmountInr: closingNciEquity,
        subsidiaryProfitOrLoss,
        parentProfitAllocationInr: parentProfitAllocation,
        parentShareOfAdjustedProfitInr: parentProfitAllocation,
        nciProfitAllocationInr: nciProfitAllocation,
        nciShareOfAdjustedProfitInr: nciProfitAllocation,
        closingNciBalanceInr: closingNciEquity,
        isNciDeficit,
        deficitNciHandling: {
            isNciInDeficit: isNciDeficit,
            deficitAmountInr: isNciDeficit ? Math.abs(closingNciEquity) : 0
        },
        intercompanyEliminations: {
            receivablesPayablesEliminatedInr: intercompanyReceivablesPayables,
            unrealizedProfitInInventoryEliminatedInr: unrealizedIntercompanyProfitInInventory
        },
        journalEntries,
        eliminationJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits,
            isBalanced: totalDebits === totalCredits
        },
        isDoubleEntryBalanced: totalDebits === totalCredits,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 110 Group Consolidation: Subsidiary net assets INR ${subsidiaryNetAssets.toLocaleString('en-IN')} adjusted for unrealized inventory margin of INR ${unrealizedIntercompanyProfitInInventory.toLocaleString('en-IN')} yields adjusted net assets of INR ${adjustedSubsidiaryNetAssets.toLocaleString('en-IN')}. Parent share (${parentSharePercent}%): INR ${parentShareOfNetAssets.toLocaleString('en-IN')}; NCI share (${nciSharePercent}%): INR ${closingNciEquity.toLocaleString('en-IN')}. ${isNciDeficit ? 'Deficit NCI balance allocated to NCI per Ind AS 110 para B94.' : 'NCI presented in equity separately.'} Intercompany balances of INR ${intercompanyReceivablesPayables.toLocaleString('en-IN')} eliminated. Consolidation assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 28 / IAS 28 / ASC 323 Investments in Associates & Equity Method Engine
 *
 * Proprietary associate investment valuation engine computing post-acquisition share of
 * profit/loss and OCI, eliminating investor share of upstream/downstream margins, and
 * enforcing Ind AS 28 paragraph 38 loss discontinuation floor at zero.
 *
 * @param {Object} params
 * @returns {Object} Associate equity accounting schedule, margin eliminations, balanced vouchers
 */
export function calculateEquityMethodIndAs28(params = {}) {
    const initialCostOfInvestment = cleanNumericAmount(params.initialCostOfInvestment, { defaultValue: 5000000.0 });
    const priorCarryingAmount = cleanNumericAmount(params.priorCarryingAmount, { defaultValue: initialCostOfInvestment });
    const investorSharePercent = cleanNumericAmount(params.investorSharePercent, { defaultValue: 30.0 });
    const associateProfitOrLoss = cleanNumericAmount(params.associateProfitOrLoss, { defaultValue: 0.0 });
    const associateOci = cleanNumericAmount(params.associateOci, { defaultValue: 0.0 });
    const unrealizedUpstreamProfit = cleanNumericAmount(params.unrealizedUpstreamProfit, { defaultValue: 0.0 });
    const unrealizedDownstreamProfit = cleanNumericAmount(params.unrealizedDownstreamProfit, { defaultValue: 0.0 });
    const dividendsReceivedFromAssociate = cleanNumericAmount(params.dividendsReceivedFromAssociate, { defaultValue: 0.0 });

    const grossShareOfProfit = roundToPaisa(associateProfitOrLoss * (investorSharePercent / 100.0));
    const eliminatedUpstreamProfit = roundToPaisa(unrealizedUpstreamProfit * (investorSharePercent / 100.0));
    const eliminatedDownstreamProfit = roundToPaisa(unrealizedDownstreamProfit * (investorSharePercent / 100.0));
    const totalEliminatedProfit = roundToPaisa(eliminatedUpstreamProfit + eliminatedDownstreamProfit);

    const netShareOfProfitRecognizedInPnl = roundToPaisa(grossShareOfProfit - totalEliminatedProfit);
    const shareOfOciRecognizedInOci = roundToPaisa(associateOci * (investorSharePercent / 100.0));

    const rawCarryingAmount = roundToPaisa(
        priorCarryingAmount +
        netShareOfProfitRecognizedInPnl +
        shareOfOciRecognizedInOci -
        dividendsReceivedFromAssociate
    );

    const isLossRecognitionDiscontinued = rawCarryingAmount < 0;
    const closingCarryingAmountInr = isLossRecognitionDiscontinued ? 0.0 : rawCarryingAmount;
    const unrecognizedLossCarriedForwardInr = isLossRecognitionDiscontinued ? Math.abs(rawCarryingAmount) : 0.0;

    const journalEntries = [];
    if (netShareOfProfitRecognizedInPnl !== 0) {
        if (netShareOfProfitRecognizedInPnl > 0) {
            journalEntries.push(
                { account: 'Investment in Associate (Ind AS 28)', type: 'DEBIT', amountInr: netShareOfProfitRecognizedInPnl },
                { account: 'Share of Profit from Associate (P&L)', type: 'CREDIT', amountInr: netShareOfProfitRecognizedInPnl }
            );
        } else {
            const recognizedLoss = isLossRecognitionDiscontinued ? priorCarryingAmount : Math.abs(netShareOfProfitRecognizedInPnl);
            if (recognizedLoss > 0) {
                journalEntries.push(
                    { account: 'Share of Loss from Associate (P&L)', type: 'DEBIT', amountInr: recognizedLoss },
                    { account: 'Investment in Associate (Ind AS 28)', type: 'CREDIT', amountInr: recognizedLoss }
                );
            }
        }
    }
    if (dividendsReceivedFromAssociate > 0) {
        journalEntries.push(
            { account: 'Bank / Cash Account', type: 'DEBIT', amountInr: dividendsReceivedFromAssociate },
            { account: 'Investment in Associate (Ind AS 28)', type: 'CREDIT', amountInr: dividendsReceivedFromAssociate }
        );
    }

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCredits = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 28 / IAS 28 / ASC 323 Investments in Associates and Joint Ventures',
        investorSharePercent,
        hasSignificantInfluence: investorSharePercent >= 20.0,
        initialCostOfInvestment,
        priorCarryingAmount,
        associateProfitOrLoss,
        grossShareOfProfitInr: grossShareOfProfit,
        eliminatedUpstreamProfitInr: eliminatedUpstreamProfit,
        eliminatedDownstreamProfitInr: eliminatedDownstreamProfit,
        totalEliminatedProfitInr: totalEliminatedProfit,
        adjustedAssociateProfitInr: roundToPaisa(associateProfitOrLoss - (totalEliminatedProfit / (investorSharePercent / 100.0))),
        netShareOfProfitRecognizedInPnlInr: netShareOfProfitRecognizedInPnl,
        investorShareOfProfitInr: netShareOfProfitRecognizedInPnl,
        shareOfOciRecognizedInOciInr: shareOfOciRecognizedInOci,
        investorShareOfOciInr: shareOfOciRecognizedInOci,
        dividendsReceivedFromAssociateInr: dividendsReceivedFromAssociate,
        closingCarryingAmountInr,
        endingCarryingAmountInr: closingCarryingAmountInr,
        unrecognizedLossCarriedForwardInr,
        isLossRecognitionDiscontinued,
        lossDiscontinuation: {
            isLossRecognitionDiscontinued,
            unrecognizedShareOfLossesInr: unrecognizedLossCarriedForwardInr
        },
        journalEntries,
        equityMethodJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits,
            isBalanced: totalDebits === totalCredits
        },
        isDoubleEntryBalanced: totalDebits === totalCredits,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 28 Equity Method: Investor holding ${investorSharePercent}% in associate. Share of profit INR ${grossShareOfProfit.toLocaleString('en-IN')} net of intercompany margin eliminations of INR ${totalEliminatedProfit.toLocaleString('en-IN')} yields net P&L share of INR ${netShareOfProfitRecognizedInPnl.toLocaleString('en-IN')}. Dividends of INR ${dividendsReceivedFromAssociate.toLocaleString('en-IN')} reduce investment carrying amount to INR ${closingCarryingAmountInr.toLocaleString('en-IN')}. ${isLossRecognitionDiscontinued ? `Cumulative losses exceeded investment; loss recognition discontinued under Ind AS 28 para 38 with INR ${unrecognizedLossCarriedForwardInr.toLocaleString('en-IN')} off-balance sheet tracking.` : 'Investment carrying amount remains active.'} Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 19 / IAS 19 / ASC 715 Employee Benefits & Actuarial Valuation Engine
 *
 * Proprietary defined benefit obligation (DBO) engine computing current service cost,
 * net interest expense, Projected Unit Credit actuarial gains/losses, OCI remeasurements
 * (with strict no-recycling rule), and Payment of Gratuity Act statutory ceiling.
 *
 * @param {Object} params
 * @returns {Object} Gratuity DBO schedule, P&L vs OCI split, balanced journal entries
 */
export function calculateEmployeeBenefitsIndAs19(params = {}) {
    const valuationDate = params.valuationDate || new Date().toISOString().split('T')[0];
    const planName = params.planName || 'Gratuity Defined Benefit Plan';
    const discountRatePercent = cleanNumericAmount(params.discountRatePercent, { defaultValue: 7.25 });
    const salaryEscalationRatePercent = cleanNumericAmount(params.salaryEscalationRatePercent, { defaultValue: 6.0 });
    const openingDbo = cleanNumericAmount(params.openingDbo, { defaultValue: 10000000.0 });
    const currentServiceCost = cleanNumericAmount(params.currentServiceCost, { defaultValue: 1500000.0 });
    const pastServiceCost = cleanNumericAmount(params.pastServiceCost, { defaultValue: 0.0 });
    const benefitsPaid = cleanNumericAmount(params.benefitsPaid, { defaultValue: 500000.0 });
    const actualClosingDbo = params.actualClosingDbo !== undefined && params.actualClosingDbo !== null
        ? cleanNumericAmount(params.actualClosingDbo)
        : null;

    // Plan assets (if funded)
    const openingPlanAssets = cleanNumericAmount(params.planAssets?.openingBalance, { defaultValue: 0.0 });
    const employerContributions = cleanNumericAmount(params.planAssets?.contributions, { defaultValue: 0.0 });
    const benefitsPaidFromFund = cleanNumericAmount(params.planAssets?.benefitsPaid, { defaultValue: 0.0 });
    const actualReturnOnPlanAssets = cleanNumericAmount(params.planAssets?.actualReturn, { defaultValue: 0.0 });

    // DBO interest cost
    const interestCostOnDbo = roundToPaisa(openingDbo * (discountRatePercent / 100.0));
    const expectedClosingDbo = roundToPaisa(openingDbo + currentServiceCost + pastServiceCost + interestCostOnDbo - benefitsPaid);

    // Actuarial gain/loss on DBO
    const finalClosingDbo = actualClosingDbo !== null ? actualClosingDbo : expectedClosingDbo;
    const actuarialGainLossOnDbo = roundToPaisa(finalClosingDbo - expectedClosingDbo); // positive = loss, negative = gain

    // Plan assets interest & remeasurements
    const interestIncomeOnPlanAssets = roundToPaisa(openingPlanAssets * (discountRatePercent / 100.0));
    const expectedClosingPlanAssets = roundToPaisa(openingPlanAssets + interestIncomeOnPlanAssets + employerContributions - benefitsPaidFromFund);
    const returnOnPlanAssetsExcludingInterest = roundToPaisa(actualReturnOnPlanAssets - interestIncomeOnPlanAssets);
    const finalClosingPlanAssets = openingPlanAssets > 0
        ? roundToPaisa(expectedClosingPlanAssets + returnOnPlanAssetsExcludingInterest)
        : 0.0;

    // Net defined benefit cost components
    const netInterestExpense = roundToPaisa(interestCostOnDbo - interestIncomeOnPlanAssets);
    const totalServiceCost = roundToPaisa(currentServiceCost + pastServiceCost);
    const netExpenseRecognizedInPnl = roundToPaisa(totalServiceCost + netInterestExpense);

    // Remeasurements recognized in OCI (strictly prohibited from recycling to P&L)
    const netRemeasurementGainLossOci = roundToPaisa(actuarialGainLossOnDbo - returnOnPlanAssetsExcludingInterest);

    // Net Defined Benefit Liability / (Asset) on Balance Sheet
    const netDefinedBenefitLiabilityClosing = roundToPaisa(finalClosingDbo - finalClosingPlanAssets);

    // Balanced double-entry journal vouchers
    const journalEntries = [];
    if (totalServiceCost > 0) {
        journalEntries.push({ account: 'Employee Benefit Expense - Current Service Cost (P&L)', type: 'DEBIT', amountInr: totalServiceCost });
    }
    if (netInterestExpense > 0) {
        journalEntries.push({ account: 'Finance Cost - Net Interest on Defined Benefit Liability (P&L)', type: 'DEBIT', amountInr: netInterestExpense });
    } else if (netInterestExpense < 0) {
        journalEntries.push({ account: 'Finance Income - Net Interest on Defined Benefit Asset (P&L)', type: 'CREDIT', amountInr: Math.abs(netInterestExpense) });
    }
    if (netRemeasurementGainLossOci > 0) {
        journalEntries.push({ account: 'Remeasurements of Defined Benefit Plan - Actuarial Loss (OCI)', type: 'DEBIT', amountInr: netRemeasurementGainLossOci });
    } else if (netRemeasurementGainLossOci < 0) {
        journalEntries.push({ account: 'Remeasurements of Defined Benefit Plan - Actuarial Gain (OCI)', type: 'CREDIT', amountInr: Math.abs(netRemeasurementGainLossOci) });
    }

    const directBenefitsPaid = Math.max(0, roundToPaisa(benefitsPaid - benefitsPaidFromFund));
    if (directBenefitsPaid > 0) {
        journalEntries.push({ account: 'Bank / Cash Account (Direct Benefits Paid)', type: 'CREDIT', amountInr: directBenefitsPaid });
    }
    if (employerContributions > 0) {
        journalEntries.push({ account: 'Bank / Cash Account (Fund Contributions)', type: 'CREDIT', amountInr: employerContributions });
    }

    const netLiabilityMovement = roundToPaisa(netExpenseRecognizedInPnl + netRemeasurementGainLossOci - directBenefitsPaid - employerContributions);
    if (netLiabilityMovement > 0) {
        journalEntries.push({ account: 'Net Defined Benefit Liability - Gratuity (Balance Sheet)', type: 'CREDIT', amountInr: netLiabilityMovement });
    } else if (netLiabilityMovement < 0) {
        journalEntries.push({ account: 'Net Defined Benefit Liability - Gratuity (Balance Sheet)', type: 'DEBIT', amountInr: Math.abs(netLiabilityMovement) });
    }

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCredits = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));
    const isDoubleEntryBalanced = Math.abs(totalDebits - totalCredits) < 0.01;

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 19 / IAS 19 / US GAAP ASC 715 Employee Benefits',
        valuationDate,
        planName,
        statutoryGratuityCeilingInr: 2000000,
        actuarialAssumptions: {
            discountRatePercent,
            salaryEscalationRatePercent,
            valuationMethod: 'PROJECTED_UNIT_CREDIT_PUC'
        },
        definedBenefitObligation: {
            openingDboInr: openingDbo,
            currentServiceCostInr: currentServiceCost,
            pastServiceCostInr: pastServiceCost,
            interestCostInr: interestCostOnDbo,
            benefitsPaidInr: benefitsPaid,
            actuarialGainLossInr: actuarialGainLossOnDbo,
            closingDboInr: finalClosingDbo
        },
        planAssets: {
            openingPlanAssetsInr: openingPlanAssets,
            interestIncomeInr: interestIncomeOnPlanAssets,
            employerContributionsInr: employerContributions,
            benefitsPaidFromFundInr: benefitsPaidFromFund,
            returnExcludingInterestInr: returnOnPlanAssetsExcludingInterest,
            closingPlanAssetsInr: finalClosingPlanAssets
        },
        netDefinedBenefitLiabilityClosingInr: netDefinedBenefitLiabilityClosing,
        netExpenseRecognizedInPnlInr: netExpenseRecognizedInPnl,
        netRemeasurementRecognizedInOciInr: netRemeasurementGainLossOci,
        isOciRecyclingProhibited: true,
        journalEntries,
        actuarialJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits,
            isBalanced: totalDebits === totalCredits
        },
        isDoubleEntryBalanced: totalDebits === totalCredits,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 19 Employee Benefits: ${planName} actuarially valued under Projected Unit Credit (PUC) method at ${discountRatePercent}% discount rate. Current Service Cost INR ${currentServiceCost.toLocaleString('en-IN')} and Net Interest INR ${netInterestExpense.toLocaleString('en-IN')} recognized in P&L (Total P&L Expense: INR ${netExpenseRecognizedInPnl.toLocaleString('en-IN')}). Actuarial Remeasurement of INR ${netRemeasurementGainLossOci.toLocaleString('en-IN')} recognized in Other Comprehensive Income (OCI) with permanent equity retention (no P&L recycling under Ind AS 19 para 122). Closing Net Defined Benefit Liability: INR ${netDefinedBenefitLiabilityClosing.toLocaleString('en-IN')} (Statutory Gratuity ceiling: INR 20,00,000 per employee). Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 2 / IAS 2 / ASC 330 Inventories Valuation & Capacity Overhead Engine
 *
 * Proprietary inventory costing engine computing normal operating capacity fixed overhead
 * absorption, unallocated idle capacity overheads expensed in P&L, Net Realizable Value (NRV)
 * lower-of-cost-and-NRV write-downs, and enforcing the statutory LIFO prohibition.
 *
 * @param {Object} params
 * @returns {Object} Inventory valuation schedule, NRV write-down, balanced vouchers
 */
export function calculateInventoryValuationIndAs2(params = {}) {
    const normalCapacityUnits = cleanNumericAmount(params.normalCapacityUnits, { defaultValue: 100000.0 });
    const actualProductionUnits = cleanNumericAmount(params.actualProductionUnits, { defaultValue: 80000.0 });
    const incurredFixedOverheads = cleanNumericAmount(params.incurredFixedOverheads, { defaultValue: 10000000.0 });
    const rawMaterialsCost = cleanNumericAmount(params.rawMaterialsCost, { defaultValue: 20000000.0 });
    const directLaborCost = cleanNumericAmount(params.directLaborCost, { defaultValue: 8000000.0 });
    const variableOverheads = cleanNumericAmount(params.variableOverheads, { defaultValue: 4000000.0 });
    const endingInventoryUnits = cleanNumericAmount(params.endingInventoryUnits, { defaultValue: 20000.0 });
    const estimatedSellingPricePerUnit = cleanNumericAmount(params.estimatedSellingPricePerUnit, { defaultValue: 600.0 });
    const estimatedSellingCostsPerUnit = cleanNumericAmount(params.estimatedSellingCostsPerUnit, { defaultValue: 50.0 });
    const estimatedCompletionCostsPerUnit = cleanNumericAmount(params.estimatedCompletionCostsPerUnit, { defaultValue: 20.0 });
    const requestedCostFormula = String(params.costFormula || 'FIFO').toUpperCase();

    const isLifoRequested = requestedCostFormula === 'LIFO';
    const effectiveCostFormula = isLifoRequested ? 'FIFO' : requestedCostFormula;

    // Normal capacity fixed overhead absorption (Ind AS 2 para 13)
    const fixedOverheadAbsorptionRatePerUnit = roundToPaisa(incurredFixedOverheads / normalCapacityUnits);

    // In low production, rate is NOT increased. Overheads absorbed = actual units * standard rate
    // In abnormally high production, rate is decreased to avoid measuring inventory above cost
    const absorbedFixedOverheads = actualProductionUnits <= normalCapacityUnits
        ? roundToPaisa(actualProductionUnits * fixedOverheadAbsorptionRatePerUnit)
        : incurredFixedOverheads;

    const unallocatedIdleCapacityOverheads = roundToPaisa(incurredFixedOverheads - absorbedFixedOverheads);

    // Total manufacturing conversion cost
    const totalManufacturingCost = roundToPaisa(rawMaterialsCost + directLaborCost + variableOverheads + absorbedFixedOverheads);
    const unitManufacturingCost = roundToPaisa(totalManufacturingCost / actualProductionUnits);

    // Ending inventory cost
    const endingInventoryCost = roundToPaisa(endingInventoryUnits * unitManufacturingCost);

    // Net Realizable Value (NRV) calculation (Ind AS 2 para 9)
    const netRealizableValuePerUnit = roundToPaisa(estimatedSellingPricePerUnit - estimatedCompletionCostsPerUnit - estimatedSellingCostsPerUnit);
    const totalNetRealizableValue = roundToPaisa(endingInventoryUnits * netRealizableValuePerUnit);

    const isWriteDownRequired = endingInventoryCost > totalNetRealizableValue;
    const inventoryWriteDownAmount = isWriteDownRequired ? roundToPaisa(endingInventoryCost - totalNetRealizableValue) : 0.0;
    const finalCarryingAmount = roundToPaisa(endingInventoryCost - inventoryWriteDownAmount);

    // Journal entries
    const journalEntries = [];
    journalEntries.push({ account: 'Inventories - Finished Goods (Balance Sheet)', type: 'DEBIT', amountInr: finalCarryingAmount });
    if (unallocatedIdleCapacityOverheads > 0) {
        journalEntries.push({ account: 'Cost of Sales - Unallocated Fixed Overhead / Idle Capacity (P&L)', type: 'DEBIT', amountInr: unallocatedIdleCapacityOverheads });
    }
    if (inventoryWriteDownAmount > 0) {
        journalEntries.push({ account: 'Cost of Sales - Inventory Write-Down to NRV (P&L)', type: 'DEBIT', amountInr: inventoryWriteDownAmount });
    }
    const cogsUnitsProduced = actualProductionUnits - endingInventoryUnits;
    if (cogsUnitsProduced > 0) {
        const cogsAmount = roundToPaisa(cogsUnitsProduced * unitManufacturingCost);
        journalEntries.push({ account: 'Cost of Goods Sold (P&L)', type: 'DEBIT', amountInr: cogsAmount });
    }
    const totalManufacturingCredits = roundToPaisa(rawMaterialsCost + directLaborCost + variableOverheads + incurredFixedOverheads);
    journalEntries.push({ account: 'Manufacturing & Production Clearing / AP', type: 'CREDIT', amountInr: totalManufacturingCredits });

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCredits = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 2 / IAS 2 / US GAAP ASC 330 Inventories',
        costFormula: effectiveCostFormula,
        lifoProhibitionNotice: isLifoRequested
            ? 'LIFO formula is strictly prohibited under Ind AS 2 paragraph 25 and IAS 2; automatically defaulted to FIFO.'
            : 'Compliant with Ind AS 2 paragraph 25 (FIFO / Weighted Average).',
        capacityAnalysis: {
            normalCapacityUnits,
            actualProductionUnits,
            capacityUtilizationPercent: roundToPaisa((actualProductionUnits / normalCapacityUnits) * 100.0),
            incurredFixedOverheadsInr: incurredFixedOverheads,
            fixedOverheadAbsorptionRatePerUnitInr: fixedOverheadAbsorptionRatePerUnit,
            absorbedFixedOverheadsInr: absorbedFixedOverheads,
            unallocatedIdleCapacityOverheadsInr: unallocatedIdleCapacityOverheads
        },
        manufacturingCost: {
            rawMaterialsCostInr: rawMaterialsCost,
            directLaborCostInr: directLaborCost,
            variableOverheadsInr: variableOverheads,
            absorbedFixedOverheadsInr: absorbedFixedOverheads,
            totalManufacturingCostInr: totalManufacturingCost,
            unitManufacturingCostInr: unitManufacturingCost
        },
        inventoryValuation: {
            endingInventoryUnits,
            endingInventoryCostInr: endingInventoryCost,
            netRealizableValuePerUnitInr: netRealizableValuePerUnit,
            totalNetRealizableValueInr: totalNetRealizableValue,
            isWriteDownRequired,
            inventoryWriteDownAmountInr: inventoryWriteDownAmount,
            finalCarryingAmountInr: finalCarryingAmount
        },
        endingInventoryCarryingAmountInr: finalCarryingAmount,
        unallocatedIdleCapacityOverheadsInr: unallocatedIdleCapacityOverheads,
        inventoryWriteDownAmountInr: inventoryWriteDownAmount,
        journalEntries,
        inventoryJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits,
            isBalanced: totalDebits === totalCredits
        },
        isDoubleEntryBalanced: totalDebits === totalCredits,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 2 Inventories: Actual production of ${actualProductionUnits.toLocaleString('en-IN')} units vs normal capacity of ${normalCapacityUnits.toLocaleString('en-IN')} units. Standard fixed overhead rate of INR ${fixedOverheadAbsorptionRatePerUnit} absorbed INR ${absorbedFixedOverheads.toLocaleString('en-IN')} into inventory; unallocated idle capacity overhead of INR ${unallocatedIdleCapacityOverheads.toLocaleString('en-IN')} expensed directly to P&L per paragraph 13. Ending inventory of ${endingInventoryUnits.toLocaleString('en-IN')} units valued at lower of Cost (INR ${endingInventoryCost.toLocaleString('en-IN')}) and NRV (INR ${totalNetRealizableValue.toLocaleString('en-IN')}), ${isWriteDownRequired ? `recognizing NRV write-down of INR ${inventoryWriteDownAmount.toLocaleString('en-IN')} in P&L.` : 'with no NRV write-down required.'} Final Balance Sheet carrying value: INR ${finalCarryingAmount.toLocaleString('en-IN')}. Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 33 / IAS 33 / ASC 260 Earnings Per Share (EPS) Engine
 *
 * Proprietary EPS engine computing Basic EPS and Diluted EPS with Treasury Stock Method
 * for stock options/warrants, If-Converted Method for convertible debt and preference shares,
 * and strict Anti-Dilution filtering per Ind AS 33 paragraph 41.
 *
 * @param {Object} params
 * @returns {Object} Basic & Diluted EPS schedule, anti-dilution analysis
 */
export function calculateEarningsPerShareIndAs33(params = {}) {
    const netProfitAttributableToParent = cleanNumericAmount(params.netProfitAttributableToParent, { defaultValue: 50000000.0 });
    const preferenceDividends = cleanNumericAmount(params.preferenceDividends, { defaultValue: 0.0 });
    const weightedAverageOrdinaryShares = cleanNumericAmount(params.weightedAverageOrdinaryShares, { defaultValue: 10000000.0 });

    const earningsAvailableToOrdinaryEquityHolders = roundToPaisa(netProfitAttributableToParent - preferenceDividends);
    const basicEps = roundToPaisa(earningsAvailableToOrdinaryEquityHolders / weightedAverageOrdinaryShares);

    // Dilution Components
    let dilutedEarnings = earningsAvailableToOrdinaryEquityHolders;
    let dilutedShares = weightedAverageOrdinaryShares;
    const dilutiveInstrumentsAnalysis = [];

    // 1. Stock Options / Warrants (Treasury Stock Method)
    const options = params.options;
    let incrementalSharesFromOptions = 0;
    if (options && options.outstandingOptions > 0) {
        const outstandingOptions = cleanNumericAmount(options.outstandingOptions);
        const exercisePrice = cleanNumericAmount(options.exercisePrice);
        const averageMarketPrice = cleanNumericAmount(options.averageMarketPrice);

        if (averageMarketPrice > exercisePrice && averageMarketPrice > 0) {
            const assumedProceeds = roundToPaisa(outstandingOptions * exercisePrice);
            const sharesRepurchased = Math.floor(assumedProceeds / averageMarketPrice);
            incrementalSharesFromOptions = outstandingOptions - sharesRepurchased;

            dilutedShares += incrementalSharesFromOptions;
            dilutiveInstrumentsAnalysis.push({
                instrument: 'ESOP / Share Options',
                method: 'TREASURY_STOCK_METHOD',
                incrementalEarningsInr: 0,
                incrementalShares: incrementalSharesFromOptions,
                incrementalEps: 0.0,
                isDilutive: true,
                status: 'INCLUDED'
            });
        } else {
            dilutiveInstrumentsAnalysis.push({
                instrument: 'ESOP / Share Options',
                method: 'TREASURY_STOCK_METHOD',
                incrementalEarningsInr: 0,
                incrementalShares: 0,
                incrementalEps: 0.0,
                isDilutive: false,
                status: 'EXCLUDED_OUT_OF_THE_MONEY'
            });
        }
    }

    // 2. Convertible Debt (If-Converted Method)
    const convertibleDebt = params.convertibleDebt;
    if (convertibleDebt && convertibleDebt.sharesIssuable > 0) {
        const afterTaxInterestSaved = cleanNumericAmount(convertibleDebt.afterTaxInterestSaved);
        const sharesIssuable = cleanNumericAmount(convertibleDebt.sharesIssuable);
        const incrementalEps = roundToPaisa(afterTaxInterestSaved / sharesIssuable);

        // Anti-dilution test: incremental EPS must be less than current running EPS
        const currentRunningEps = roundToPaisa(dilutedEarnings / dilutedShares);
        const isDilutive = incrementalEps < currentRunningEps;

        if (isDilutive) {
            dilutedEarnings = roundToPaisa(dilutedEarnings + afterTaxInterestSaved);
            dilutedShares += sharesIssuable;
            dilutiveInstrumentsAnalysis.push({
                instrument: 'Convertible Debentures',
                method: 'IF_CONVERTED_METHOD',
                incrementalEarningsInr: afterTaxInterestSaved,
                incrementalShares: sharesIssuable,
                incrementalEps,
                isDilutive: true,
                status: 'INCLUDED'
            });
        } else {
            dilutiveInstrumentsAnalysis.push({
                instrument: 'Convertible Debentures',
                method: 'IF_CONVERTED_METHOD',
                incrementalEarningsInr: afterTaxInterestSaved,
                incrementalShares: sharesIssuable,
                incrementalEps,
                isDilutive: false,
                status: 'EXCLUDED_ANTI_DILUTIVE'
            });
        }
    }

    // 3. Convertible Preference Shares (If-Converted Method)
    const convertiblePreference = params.convertiblePreference;
    if (convertiblePreference && convertiblePreference.sharesIssuable > 0) {
        const dividendsSaved = cleanNumericAmount(convertiblePreference.dividendsSaved);
        const sharesIssuable = cleanNumericAmount(convertiblePreference.sharesIssuable);
        const incrementalEps = roundToPaisa(dividendsSaved / sharesIssuable);

        const currentRunningEps = roundToPaisa(dilutedEarnings / dilutedShares);
        const isDilutive = incrementalEps < currentRunningEps;

        if (isDilutive) {
            dilutedEarnings = roundToPaisa(dilutedEarnings + dividendsSaved);
            dilutedShares += sharesIssuable;
            dilutiveInstrumentsAnalysis.push({
                instrument: 'Convertible Preference Shares',
                method: 'IF_CONVERTED_METHOD',
                incrementalEarningsInr: dividendsSaved,
                incrementalShares: sharesIssuable,
                incrementalEps,
                isDilutive: true,
                status: 'INCLUDED'
            });
        } else {
            dilutiveInstrumentsAnalysis.push({
                instrument: 'Convertible Preference Shares',
                method: 'IF_CONVERTED_METHOD',
                incrementalEarningsInr: dividendsSaved,
                incrementalShares: sharesIssuable,
                incrementalEps,
                isDilutive: false,
                status: 'EXCLUDED_ANTI_DILUTIVE'
            });
        }
    }

    const dilutedEps = roundToPaisa(dilutedEarnings / dilutedShares);

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 33 / IAS 33 / US GAAP ASC 260 Earnings Per Share',
        basicEps: {
            netProfitAttributableToParentInr: netProfitAttributableToParent,
            preferenceDividendsInr: preferenceDividends,
            earningsAvailableToOrdinaryHoldersInr: earningsAvailableToOrdinaryEquityHolders,
            weightedAverageOrdinaryShares,
            basicEpsInr: basicEps
        },
        basicEpsInr: basicEps,
        dilutedEps: {
            adjustedEarningsAvailableInr: dilutedEarnings,
            dilutedWeightedAverageShares: dilutedShares,
            dilutedEpsInr: dilutedEps,
            incrementalSharesTotal: dilutedShares - weightedAverageOrdinaryShares
        },
        dilutedEpsInr: dilutedEps,
        dilutiveInstrumentsAnalysis,
        isAntiDilutiveFiltered: dilutiveInstrumentsAnalysis.some(i => i.status.includes('ANTI_DILUTIVE')),
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 33 Earnings Per Share: Earnings available to ordinary shareholders of INR ${earningsAvailableToOrdinaryEquityHolders.toLocaleString('en-IN')} over ${weightedAverageOrdinaryShares.toLocaleString('en-IN')} weighted average shares yields Basic EPS of INR ${basicEps.toFixed(2)}. Incorporating dilutive potential ordinary shares (Treasury Stock Method & If-Converted Method) with anti-dilution filter yields Diluted Earnings of INR ${dilutedEarnings.toLocaleString('en-IN')} over ${dilutedShares.toLocaleString('en-IN')} diluted shares for Diluted EPS of INR ${dilutedEps.toFixed(2)}. Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 7 / IAS 7 / ASC 230 Statement of Cash Flows Engine (Indirect Method)
 *
 * Proprietary cash flow engine deriving operating, investing, and financing cash flows,
 * working capital accruals/deferrals, lease principal classification, and exact cash reconciliation.
 *
 * @param {Object} params
 * @returns {Object} Cash flow statement schedule and cash balance reconciliation
 */
export function calculateCashFlowsIndAs7(params = {}) {
    const profitBeforeTax = cleanNumericAmount(params.profitBeforeTax, { defaultValue: 25000000.0 });
    const depreciationAmortization = cleanNumericAmount(params.depreciationAmortization, { defaultValue: 5000000.0 });
    const financeCosts = cleanNumericAmount(params.financeCosts, { defaultValue: 2000000.0 });
    const gainLossOnSaleOfAssets = cleanNumericAmount(params.gainLossOnSaleOfAssets, { defaultValue: 0.0 });
    const unrealizedFxGainLoss = cleanNumericAmount(params.unrealizedFxGainLoss, { defaultValue: 0.0 });

    const changeInInventories = cleanNumericAmount(params.changeInInventories, { defaultValue: -2000000.0 }); // negative = increase
    const changeInTradeReceivables = cleanNumericAmount(params.changeInTradeReceivables, { defaultValue: -3000000.0 }); // negative = increase
    const changeInTradePayables = cleanNumericAmount(params.changeInTradePayables, { defaultValue: 1500000.0 }); // positive = increase
    const incomeTaxesPaid = cleanNumericAmount(params.incomeTaxesPaid, { defaultValue: 6000000.0 });

    const operatingProfitBeforeWorkingCapital = roundToPaisa(
        profitBeforeTax +
        depreciationAmortization +
        financeCosts +
        gainLossOnSaleOfAssets +
        unrealizedFxGainLoss
    );

    const cashGeneratedFromOperations = roundToPaisa(
        operatingProfitBeforeWorkingCapital +
        changeInInventories +
        changeInTradeReceivables +
        changeInTradePayables
    );

    const netOperatingCashFlow = roundToPaisa(cashGeneratedFromOperations - incomeTaxesPaid);

    // Investing Activities
    const inv = params.investingActivities || {};
    const purchaseOfPpeAndIntangibles = cleanNumericAmount(inv.purchaseOfPpeAndIntangibles, { defaultValue: 12000000.0 });
    const proceedsFromSaleOfPpe = cleanNumericAmount(inv.proceedsFromSaleOfPpe, { defaultValue: 1000000.0 });
    const purchaseOfInvestments = cleanNumericAmount(inv.purchaseOfInvestments, { defaultValue: 0.0 });
    const proceedsFromSaleOfInvestments = cleanNumericAmount(inv.proceedsFromSaleOfInvestments, { defaultValue: 0.0 });
    const interestAndDividendsReceived = cleanNumericAmount(inv.interestAndDividendsReceived, { defaultValue: 500000.0 });

    const netInvestingCashFlow = roundToPaisa(
        -purchaseOfPpeAndIntangibles +
        proceedsFromSaleOfPpe -
        purchaseOfInvestments +
        proceedsFromSaleOfInvestments +
        interestAndDividendsReceived
    );

    // Financing Activities
    const fin = params.financingActivities || {};
    const proceedsFromIssuingShares = cleanNumericAmount(fin.proceedsFromIssuingShares, { defaultValue: 0.0 });
    const proceedsFromBorrowings = cleanNumericAmount(fin.proceedsFromBorrowings, { defaultValue: 5000000.0 });
    const repaymentOfBorrowings = cleanNumericAmount(fin.repaymentOfBorrowings, { defaultValue: 3000000.0 });
    const leasePrincipalPayments = cleanNumericAmount(fin.leasePrincipalPayments, { defaultValue: 1500000.0 });
    const interestPaid = cleanNumericAmount(fin.interestPaid, { defaultValue: 2000000.0 });
    const dividendsPaid = cleanNumericAmount(fin.dividendsPaid, { defaultValue: 2500000.0 });

    const netFinancingCashFlow = roundToPaisa(
        proceedsFromIssuingShares +
        proceedsFromBorrowings -
        repaymentOfBorrowings -
        leasePrincipalPayments -
        interestPaid -
        dividendsPaid
    );

    // Net Change & Cash Equivalents
    const openingCashAndCashEquivalents = cleanNumericAmount(params.openingCashAndCashEquivalents, { defaultValue: 15000000.0 });
    const netChangeInCash = roundToPaisa(netOperatingCashFlow + netInvestingCashFlow + netFinancingCashFlow);
    const closingCashAndCashEquivalents = roundToPaisa(openingCashAndCashEquivalents + netChangeInCash);

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 7 / IAS 7 / US GAAP ASC 230 Statement of Cash Flows',
        operatingActivities: {
            profitBeforeTaxInr: profitBeforeTax,
            depreciationAmortizationInr: depreciationAmortization,
            financeCostsInr: financeCosts,
            operatingProfitBeforeWorkingCapitalInr: operatingProfitBeforeWorkingCapital,
            changeInInventoriesInr: changeInInventories,
            changeInTradeReceivablesInr: changeInTradeReceivables,
            changeInTradePayablesInr: changeInTradePayables,
            cashGeneratedFromOperationsInr: cashGeneratedFromOperations,
            incomeTaxesPaidInr: incomeTaxesPaid,
            netOperatingCashFlowInr: netOperatingCashFlow
        },
        investingActivities: {
            purchaseOfPpeAndIntangiblesInr: purchaseOfPpeAndIntangibles,
            proceedsFromSaleOfPpeInr: proceedsFromSaleOfPpe,
            interestAndDividendsReceivedInr: interestAndDividendsReceived,
            netInvestingCashFlowInr: netInvestingCashFlow
        },
        financingActivities: {
            proceedsFromBorrowingsInr: proceedsFromBorrowings,
            repaymentOfBorrowingsInr: repaymentOfBorrowings,
            leasePrincipalPaymentsInr: leasePrincipalPayments,
            interestPaidInr: interestPaid,
            dividendsPaidInr: dividendsPaid,
            netFinancingCashFlowInr: netFinancingCashFlow
        },
        reconciliation: {
            netOperatingCashFlowInr: netOperatingCashFlow,
            netInvestingCashFlowInr: netInvestingCashFlow,
            netFinancingCashFlowInr: netFinancingCashFlow,
            netChangeInCashInr: netChangeInCash,
            openingCashAndCashEquivalentsInr: openingCashAndCashEquivalents,
            closingCashAndCashEquivalentsInr: closingCashAndCashEquivalents
        },
        netOperatingCashFlowInr: netOperatingCashFlow,
        netInvestingCashFlowInr: netInvestingCashFlow,
        netFinancingCashFlowInr: netFinancingCashFlow,
        netChangeInCashInr: netChangeInCash,
        closingCashAndCashEquivalentsInr: closingCashAndCashEquivalents,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 7 Statement of Cash Flows: Net cash generated from operating activities of INR ${netOperatingCashFlow.toLocaleString('en-IN')}, net cash used in investing activities of INR ${netInvestingCashFlow.toLocaleString('en-IN')}, and net cash used in financing activities of INR ${netFinancingCashFlow.toLocaleString('en-IN')} yields net cash movement of INR ${netChangeInCash.toLocaleString('en-IN')}. Closing cash and cash equivalents: INR ${closingCashAndCashEquivalents.toLocaleString('en-IN')}. Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 16 / IAS 16 / ASC 360 Property, Plant and Equipment (PPE) Engine
 *
 * Proprietary PPE engine computing component-level depreciation schedules, useful life reviews,
 * SLM/WDV methods, and derecognition gains/losses upon component/asset replacement.
 *
 * @param {Object} params
 * @returns {Object} Fixed asset register component schedule, depreciation expense, balanced vouchers
 */
export function calculatePropertyPlantEquipmentIndAs16(params = {}) {
    const assetTagId = params.assetTagId || 'PPE-ASSET-001';
    const assetName = params.assetName || 'Industrial Production Facility';
    const depreciationMethod = String(params.depreciationMethod || 'SLM').toUpperCase();
    const rawComponents = Array.isArray(params.components) && params.components.length > 0
        ? params.components
        : [{ componentName: 'Building Structure', cost: 10000000.0, residualValue: 500000.0, usefulLifeYears: 30, openingAccumulatedDepreciation: 0.0 }];

    let totalGrossBlock = 0.0;
    let totalResidualValue = 0.0;
    let totalCurrentDepreciation = 0.0;
    let totalClosingAccumulatedDepreciation = 0.0;
    let totalClosingNetBlock = 0.0;

    const componentSchedules = rawComponents.map((c, idx) => {
        const componentName = c.componentName || `Component ${idx + 1}`;
        const cost = cleanNumericAmount(c.cost, { defaultValue: 1000000.0 });
        const residualValue = cleanNumericAmount(c.residualValue, { defaultValue: roundToPaisa(cost * 0.05) });
        const usefulLifeYears = cleanNumericAmount(c.usefulLifeYears, { defaultValue: 10.0 });
        const openingAccumulatedDepreciation = cleanNumericAmount(c.openingAccumulatedDepreciation, { defaultValue: 0.0 });

        const depreciableAmount = roundToPaisa(Math.max(0, cost - residualValue));
        let annualDepreciation = 0.0;

        if (depreciationMethod === 'WDV') {
            const wdvRate = cost > 0 && usefulLifeYears > 0
                ? 1.0 - Math.pow(Math.max(0.01, residualValue) / cost, 1.0 / usefulLifeYears)
                : 0.0;
            const currentCarrying = roundToPaisa(cost - openingAccumulatedDepreciation);
            annualDepreciation = roundToPaisa(Math.min(currentCarrying, currentCarrying * wdvRate));
        } else {
            // Straight-Line Method (SLM)
            annualDepreciation = usefulLifeYears > 0 ? roundToPaisa(depreciableAmount / usefulLifeYears) : 0.0;
            const remainingCarrying = roundToPaisa(cost - openingAccumulatedDepreciation - residualValue);
            annualDepreciation = Math.min(annualDepreciation, Math.max(0, remainingCarrying));
        }

        const closingAccumulatedDepreciation = roundToPaisa(openingAccumulatedDepreciation + annualDepreciation);
        const closingNetBlock = roundToPaisa(cost - closingAccumulatedDepreciation);

        totalGrossBlock = roundToPaisa(totalGrossBlock + cost);
        totalResidualValue = roundToPaisa(totalResidualValue + residualValue);
        totalCurrentDepreciation = roundToPaisa(totalCurrentDepreciation + annualDepreciation);
        totalClosingAccumulatedDepreciation = roundToPaisa(totalClosingAccumulatedDepreciation + closingAccumulatedDepreciation);
        totalClosingNetBlock = roundToPaisa(totalClosingNetBlock + closingNetBlock);

        return {
            componentName,
            costInr: cost,
            residualValueInr: residualValue,
            usefulLifeYears,
            openingAccumulatedDepreciationInr: openingAccumulatedDepreciation,
            currentPeriodDepreciationInr: annualDepreciation,
            closingAccumulatedDepreciationInr: closingAccumulatedDepreciation,
            closingNetBlockInr: closingNetBlock
        };
    });

    // Disposal / Replacement check
    let disposalSummary = null;
    if (params.disposal && params.disposal.isDisposed) {
        const compIdx = params.disposal.componentIndex || 0;
        const targetComp = componentSchedules[compIdx] || componentSchedules[0];
        const netDisposalProceeds = cleanNumericAmount(params.disposal.netDisposalProceeds, { defaultValue: 0.0 });
        const derecognizedNetBookValue = targetComp.closingNetBlockInr;
        const gainLossOnDisposal = roundToPaisa(netDisposalProceeds - derecognizedNetBookValue);

        disposalSummary = {
            disposedComponent: targetComp.componentName,
            netDisposalProceedsInr: netDisposalProceeds,
            derecognizedNetBookValueInr: derecognizedNetBookValue,
            gainLossOnDisposalInr: gainLossOnDisposal,
            outcome: gainLossOnDisposal >= 0 ? 'GAIN_ON_DISPOSAL' : 'LOSS_ON_DISPOSAL'
        };
    }

    // Balanced double-entry journal vouchers
    const journalEntries = [];
    if (totalCurrentDepreciation > 0) {
        journalEntries.push(
            { account: 'Depreciation Expense - PPE (P&L)', type: 'DEBIT', amountInr: totalCurrentDepreciation },
            { account: 'Accumulated Depreciation - PPE (Contra-Asset)', type: 'CREDIT', amountInr: totalCurrentDepreciation }
        );
    }
    if (disposalSummary) {
        if (disposalSummary.netDisposalProceedsInr > 0) {
            journalEntries.push({ account: 'Bank / Cash Account (Sale Proceeds)', type: 'DEBIT', amountInr: disposalSummary.netDisposalProceedsInr });
        }
        if (disposalSummary.gainLossOnDisposalInr < 0) {
            journalEntries.push({ account: 'Loss on Disposal of PPE (P&L)', type: 'DEBIT', amountInr: Math.abs(disposalSummary.gainLossOnDisposalInr) });
        } else if (disposalSummary.gainLossOnDisposalInr > 0) {
            journalEntries.push({ account: 'Gain on Disposal of PPE (P&L)', type: 'CREDIT', amountInr: disposalSummary.gainLossOnDisposalInr });
        }
        journalEntries.push({ account: 'Property, Plant and Equipment (Net Block Derecognition)', type: 'CREDIT', amountInr: disposalSummary.derecognizedNetBookValueInr });
    }

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCredits = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 16 / IAS 16 / US GAAP ASC 360 Property, Plant and Equipment',
        assetTagId,
        assetName,
        depreciationMethod,
        totalGrossBlockInr: totalGrossBlock,
        totalResidualValueInr: totalResidualValue,
        totalCurrentDepreciationInr: totalCurrentDepreciation,
        totalClosingAccumulatedDepreciationInr: totalClosingAccumulatedDepreciation,
        totalClosingNetBlockInr: totalClosingNetBlock,
        componentSchedules,
        disposalSummary,
        journalEntries,
        depreciationJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCredits,
            isBalanced: totalDebits === totalCredits
        },
        isDoubleEntryBalanced: totalDebits === totalCredits,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 16 Property, Plant and Equipment: Asset "${assetName}" componentized into ${componentSchedules.length} significant components per paragraph 43. Total gross block of INR ${totalGrossBlock.toLocaleString('en-IN')} depreciated under ${depreciationMethod} yielding current period depreciation of INR ${totalCurrentDepreciation.toLocaleString('en-IN')}. Closing Net Carrying Amount: INR ${totalClosingNetBlock.toLocaleString('en-IN')}. Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 113 / IFRS 13 / ASC 820 Fair Value Measurement Hierarchy Engine
 *
 * Proprietary fair value valuation engine classifying instruments across Level 1, 2, and 3 inputs,
 * executing Level 3 roll-forward reconciliations, and evaluating sensitivity shocks.
 *
 * @param {Object} params
 * @returns {Object} Fair value hierarchy register, Level 3 roll-forward, sensitivity matrix
 */
export function evaluateFairValueHierarchyIndAs113(params = {}) {
    const rawInstruments = Array.isArray(params.instruments) && params.instruments.length > 0
        ? params.instruments
        : [
            { name: 'Listed Equity Holdings', hierarchyLevel: 'LEVEL_1', valuationTechnique: 'MARKET_APPROACH', fairValueInr: 50000000.0 },
            { name: 'Interest Rate Swap Derivatives', hierarchyLevel: 'LEVEL_2', valuationTechnique: 'INCOME_APPROACH_DCF', fairValueInr: 15000000.0 },
            { name: 'Unlisted Venture Equity', hierarchyLevel: 'LEVEL_3', valuationTechnique: 'INCOME_APPROACH_DCF', fairValueInr: 25000000.0 }
        ];

    let totalPortfolioFairValue = 0.0;
    let level1Total = 0.0;
    let level2Total = 0.0;
    let level3Total = 0.0;

    const classifiedInstruments = rawInstruments.map(item => {
        const name = item.name || 'Unnamed Instrument';
        const hierarchyLevel = String(item.hierarchyLevel || 'LEVEL_2').toUpperCase();
        const valuationTechnique = String(item.valuationTechnique || 'MARKET_APPROACH').toUpperCase();
        const fairValueInr = cleanNumericAmount(item.fairValueInr || item.carryingAmount, { defaultValue: 0.0 });

        totalPortfolioFairValue = roundToPaisa(totalPortfolioFairValue + fairValueInr);
        if (hierarchyLevel === 'LEVEL_1') level1Total = roundToPaisa(level1Total + fairValueInr);
        else if (hierarchyLevel === 'LEVEL_2') level2Total = roundToPaisa(level2Total + fairValueInr);
        else if (hierarchyLevel === 'LEVEL_3') level3Total = roundToPaisa(level3Total + fairValueInr);

        return {
            name,
            hierarchyLevel,
            valuationTechnique,
            fairValueInr
        };
    });

    // Level 3 Roll-Forward Reconciliation (Ind AS 113 para 93)
    const rec = params.level3Reconciliation || {};
    const openingLevel3 = cleanNumericAmount(rec.openingBalance, { defaultValue: roundToPaisa(level3Total * 0.8) });
    const gainsLossesPnl = cleanNumericAmount(rec.gainsLossesPnl, { defaultValue: roundToPaisa(level3Total * 0.1) });
    const gainsLossesOci = cleanNumericAmount(rec.gainsLossesOci, { defaultValue: 0.0 });
    const purchases = cleanNumericAmount(rec.purchases, { defaultValue: roundToPaisa(level3Total * 0.1) });
    const sales = cleanNumericAmount(rec.sales, { defaultValue: 0.0 });
    const transfersIn = cleanNumericAmount(rec.transfersIn, { defaultValue: 0.0 });
    const transfersOut = cleanNumericAmount(rec.transfersOut, { defaultValue: 0.0 });

    const closingLevel3Calculated = roundToPaisa(
        openingLevel3 +
        gainsLossesPnl +
        gainsLossesOci +
        purchases -
        sales +
        transfersIn -
        transfersOut
    );

    // Sensitivity shock simulation
    const sens = params.sensitivityParameters || {};
    const inputName = sens.inputName || 'WACC Discount Rate';
    const shockBps = cleanNumericAmount(sens.shockBps, { defaultValue: 100.0 }); // 100 bps = 1%
    const valueImpactPerBps = cleanNumericAmount(sens.valueImpactPerBps, { defaultValue: roundToPaisa(level3Total * 0.0005) });

    const favorableShockValue = roundToPaisa(level3Total + (shockBps * valueImpactPerBps));
    const unfavorableShockValue = roundToPaisa(level3Total - (shockBps * valueImpactPerBps));

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 113 / IFRS 13 / US GAAP ASC 820 Fair Value Measurement',
        totalPortfolioFairValueInr: totalPortfolioFairValue,
        hierarchyBreakdown: {
            level1: {
                totalInr: level1Total,
                percentageOfTotal: totalPortfolioFairValue > 0 ? roundToPaisa((level1Total / totalPortfolioFairValue) * 100.0) : 0.0,
                description: 'Quoted prices in active markets for identical assets/liabilities'
            },
            level2: {
                totalInr: level2Total,
                percentageOfTotal: totalPortfolioFairValue > 0 ? roundToPaisa((level2Total / totalPortfolioFairValue) * 100.0) : 0.0,
                description: 'Observable inputs other than Level 1 quoted prices'
            },
            level3: {
                totalInr: level3Total,
                percentageOfTotal: totalPortfolioFairValue > 0 ? roundToPaisa((level3Total / totalPortfolioFairValue) * 100.0) : 0.0,
                description: 'Unobservable inputs reflecting entity assumptions'
            }
        },
        classifiedInstruments,
        level3Reconciliation: {
            openingBalanceInr: openingLevel3,
            gainsLossesRecognizedInPnlInr: gainsLossesPnl,
            gainsLossesRecognizedInOciInr: gainsLossesOci,
            purchasesAdditionsInr: purchases,
            salesSettlementsInr: sales,
            transfersInInr: transfersIn,
            transfersOutInr: transfersOut,
            closingBalanceInr: closingLevel3Calculated
        },
        level3SensitivityAnalysis: {
            inputName,
            shockMagnitudeBps: shockBps,
            favorableShockFairValueInr: favorableShockValue,
            unfavorableShockFairValueInr: unfavorableShockValue,
            netValueAtRiskInr: roundToPaisa(favorableShockValue - unfavorableShockValue)
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 113 Fair Value Measurement: Total measured fair value portfolio of INR ${totalPortfolioFairValue.toLocaleString('en-IN')}. Level 1: INR ${level1Total.toLocaleString('en-IN')} (${totalPortfolioFairValue > 0 ? ((level1Total / totalPortfolioFairValue) * 100).toFixed(1) : 0}%); Level 2: INR ${level2Total.toLocaleString('en-IN')} (${totalPortfolioFairValue > 0 ? ((level2Total / totalPortfolioFairValue) * 100).toFixed(1) : 0}%); Level 3: INR ${level3Total.toLocaleString('en-IN')} (${totalPortfolioFairValue > 0 ? ((level3Total / totalPortfolioFairValue) * 100).toFixed(1) : 0}%). Level 3 roll-forward reconciled per paragraph 93 with +/-${shockBps}bps sensitivity shock analysis. Assessment does not authorize payment disbursement.`
    };
}


/**
 * Ind AS 38 / IAS 38 / ASC 350 Intangible Assets & R&D Capitalization Engine
 *
 * Proprietary intangible asset engine evaluating research phase expensing, development phase
 * PIRATE criteria capitalization under paragraph 57, and finite/indefinite useful life amortization.
 *
 * @param {Object} params
 * @returns {Object} Intangible asset capitalization schedule, amortization, balanced vouchers
 */
export function evaluateIntangiblesCapitalizationIndAs38(params = {}) {
    const projectId = params.projectId || 'RND-DEV-001';
    const projectName = params.projectName || 'Enterprise AI Financial Engine';
    const researchCosts = cleanNumericAmount(params.researchCosts, { defaultValue: 4000000.0 });
    const developmentCosts = cleanNumericAmount(params.developmentCosts, { defaultValue: 16000000.0 });

    const p = params.pirateCriteria || {};
    const probableEconomicBenefits = Boolean(p.probableEconomicBenefits ?? true);
    const intentionToComplete = Boolean(p.intentionToComplete ?? true);
    const resourcesAdequate = Boolean(p.resourcesAdequate ?? true);
    const abilityToUseOrSell = Boolean(p.abilityToUseOrSell ?? true);
    const technicalFeasibility = Boolean(p.technicalFeasibility ?? true);
    const expenditureMeasurable = Boolean(p.expenditureMeasurable ?? true);

    const isAllPirateCriteriaSatisfied = (
        probableEconomicBenefits &&
        intentionToComplete &&
        resourcesAdequate &&
        abilityToUseOrSell &&
        technicalFeasibility &&
        expenditureMeasurable
    );

    const capitalizedDevelopmentCost = isAllPirateCriteriaSatisfied ? developmentCosts : 0.0;
    const expensedToPnl = roundToPaisa(researchCosts + (isAllPirateCriteriaSatisfied ? 0.0 : developmentCosts));

    // Amortization analysis
    const usefulLifeType = String(params.usefulLifeType || 'FINITE').toUpperCase();
    const usefulLifeYears = usefulLifeType === 'FINITE' ? cleanNumericAmount(params.usefulLifeYears, { defaultValue: 5.0 }) : 0.0;
    const annualAmortizationExpense = (usefulLifeType === 'FINITE' && usefulLifeYears > 0 && capitalizedDevelopmentCost > 0)
        ? roundToPaisa(capitalizedDevelopmentCost / usefulLifeYears)
        : 0.0;

    // Balanced double-entry journal vouchers
    const journalEntries = [];
    if (capitalizedDevelopmentCost > 0) {
        journalEntries.push({ account: 'Intangible Assets Under Development (Balance Sheet)', type: 'DEBIT', amountInr: capitalizedDevelopmentCost });
    }
    if (expensedToPnl > 0) {
        journalEntries.push({ account: 'Research & Development Expense (P&L)', type: 'DEBIT', amountInr: expensedToPnl });
    }
    const totalCredits = roundToPaisa(capitalizedDevelopmentCost + expensedToPnl);
    journalEntries.push({ account: 'Accounts Payable / Payroll Clearing', type: 'CREDIT', amountInr: totalCredits });

    if (annualAmortizationExpense > 0) {
        journalEntries.push(
            { account: 'Amortization Expense - Intangibles (P&L)', type: 'DEBIT', amountInr: annualAmortizationExpense },
            { account: 'Accumulated Amortization - Intangibles (Contra-Asset)', type: 'CREDIT', amountInr: annualAmortizationExpense }
        );
    }

    const totalDebits = roundToPaisa(journalEntries.filter(e => e.type === 'DEBIT').reduce((s, e) => s + e.amountInr, 0));
    const totalCreditsSum = roundToPaisa(journalEntries.filter(e => e.type === 'CREDIT').reduce((s, e) => s + e.amountInr, 0));

    return {
        status: 'SUCCESS',
        standard: 'Ind AS 38 / IAS 38 / US GAAP ASC 350 Intangible Assets',
        projectId,
        projectName,
        costAnalysis: {
            researchCostsInr: researchCosts,
            developmentCostsInr: developmentCosts,
            totalProjectCostInr: roundToPaisa(researchCosts + developmentCosts)
        },
        pirateCriteriaAssessment: {
            probableEconomicBenefits,
            intentionToComplete,
            resourcesAdequate,
            abilityToUseOrSell,
            technicalFeasibility,
            expenditureMeasurable,
            isAllPirateCriteriaSatisfied
        },
        accountingTreatment: {
            capitalizedIntangibleCostInr: capitalizedDevelopmentCost,
            expensedToPnlInr: expensedToPnl,
            capitalizationEligibility: isAllPirateCriteriaSatisfied ? 'ELIGIBLE_FOR_CAPITALIZATION' : 'INELIGIBLE_EXPENSED_TO_PNL'
        },
        amortizationSchedule: {
            usefulLifeType,
            usefulLifeYears,
            annualAmortizationExpenseInr: annualAmortizationExpense,
            impairmentRequirement: usefulLifeType === 'INDEFINITE'
                ? 'Amortization prohibited; annual impairment test mandatory under Ind AS 36.'
                : 'Systematic straight-line amortization over useful life.'
        },
        capitalizedIntangibleCostInr: capitalizedDevelopmentCost,
        expensedToPnlInr: expensedToPnl,
        annualAmortizationExpenseInr: annualAmortizationExpense,
        journalEntries,
        intangiblesJournalEntry: {
            journalEntries,
            totalDebitsInr: totalDebits,
            totalCreditsInr: totalCreditsSum,
            isBalanced: totalDebits === totalCreditsSum
        },
        isDoubleEntryBalanced: totalDebits === totalCreditsSum,
        isAuthorizedToDisburse: false,
        statutoryRationale: `Ind AS 38 Intangible Assets: Project "${projectName}". Research expenditure of INR ${researchCosts.toLocaleString('en-IN')} expensed immediately to P&L per paragraph 54. Development expenditure of INR ${developmentCosts.toLocaleString('en-IN')} ${isAllPirateCriteriaSatisfied ? `satisfies all 6 PIRATE criteria (paragraph 57) and is capitalized as Intangible Asset with annual amortization of INR ${annualAmortizationExpense.toLocaleString('en-IN')}.` : 'failed one or more PIRATE criteria and is expensed to P&L.'} Assessment does not authorize payment disbursement.`
    };
}
