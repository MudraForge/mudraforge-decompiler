/**
 * MSME & DELAYED PAYMENTS DETERMINISTIC CALCULATORS
 * Section 15/16 compounding at 3x Bank Rate, Section 43B(h), Schedule III aging.
 */
import { roundToPaisa, roundSection288B } from './numeric_utils.js';

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * DETERMINISTIC STATUTORY & FINANCIAL CALCULATOR (v1.0 -- Adesha Invariants)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements deterministic arithmetic, temporal state machines, and statutory
 * conditionality trees across:
 * 1. MSMED Act Section 16: Exact compound interest with monthly rests at 3x Bank Rate.
 * 2. Section 43B(h) Temporal Engine: March 31 fiscal boundaries, dispute pauses, and trader exclusions.
 * 3. Cross-Border Withholding Engine: Income-tax Act 2025 vs 1961 dual-act mapping,
 *    Form 145/146 routing, and Supreme Court Engineering Analysis (2021) cloud characterization.
 */
// Current RBI Bank Rate (H2 2024 - 2026 baseline): 6.75%
export const DEFAULT_RBI_BANK_RATE = 0.0675;

// Dated RBI Bank Rate Schedule notified under Section 49 of the RBI Act, 1934
export const RBI_BANK_RATE_SCHEDULE = [
    { effectiveFrom: '2023-02-08', effectiveTo: null, rate: 0.0675, notificationRef: 'RBI/2022-23/177' },
    { effectiveFrom: '2022-12-07', effectiveTo: '2023-02-07', rate: 0.0650, notificationRef: 'RBI/2022-23/150' },
    { effectiveFrom: '2022-09-30', effectiveTo: '2022-12-06', rate: 0.0615, notificationRef: 'RBI/2022-23/120' }
];

/**
 * Resolves the official RBI Bank Rate in effect on a specific transaction or due date.
 */
export function getRbiBankRateForDate(dateString) {
    const d = dateString ? new Date(dateString) : new Date();
    const dateIso = !isNaN(d.getTime()) ? d.toISOString().split('T')[0] : new Date().toISOString().split('T')[0];
    const match = RBI_BANK_RATE_SCHEDULE.find(item => {
        if (item.effectiveFrom && dateIso < item.effectiveFrom) return false;
        if (item.effectiveTo && dateIso > item.effectiveTo) return false;
        return true;
    });
    return match ? match.rate : DEFAULT_RBI_BANK_RATE;
}

// Statutory Multiple per MSMED Act 2006 Section 16: 3x Bank Rate
export const MSMED_RATE_MULTIPLIER = 3;

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 1. MSMED SECTION 16: DETERMINISTIC COMPOUND INTEREST ENGINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Formula: Compound interest with monthly rests at 3x RBI Bank Rate.
 * For partial months, applies prorated day-count fractions.
 */
export function calculateMsmeCompoundInterest({
    principal,
    // First-class MSMED statutory parameters:
    acceptanceDate = null,        // Day of acceptance / deemed acceptance u/s 2(h)
    deliveryDate = null,          // Day of actual delivery u/s 2(h)(i)
    writtenObjectionDate = null,  // Day of written objection u/s 2(h)(ii) (within 15 days of delivery)
    objectionRemovalDate = null,  // Day objection removed by supplier u/s 2(h)(ii)
    agreedPaymentDate = null,     // Date agreed upon in writing u/s 15
    hasWrittenAgreement = null,   // Boolean: true if written agreement exists, false if absent
    // Trader Exclusion & Enterprise Classification (Ministry of MSME OM 02.07.2021):
    vendorNicCode = null,         // 2-digit or 5-digit NIC code (e.g. '45', '46', '47')
    nicCode = null,               // Alias for vendorNicCode
    supplierType = null,          // 'MANUFACTURER' | 'SERVICE' | 'TRADER' | 'RETAIL_TRADER' | 'WHOLESALE_TRADER'
    isTrader = null,              // Explicit boolean override
    // Direct / legacy parameters:
    dueDate = null,
    accrualStartDate = null,
    paymentDate = new Date().toISOString(),
    rbiBankRate = null,
    dateConvention = 'START_DATE_INCLUSIVE', // 'START_DATE_INCLUSIVE' | 'SECTION_16_DAY_AFTER_DUE' | 'FROM_APPOINTED_DAY' | 'FROM_DAY_AFTER_AGREED_DATE'
    commenceDayAfter = false, // Legacy flag: when true, adds 1 day to dueDate per Section 16 agreed date rule
    includePaymentDate = false // When false, payment date is the settlement day (interest ceases at start of payment date)
}) {
    const p = Number(principal);
    if (isNaN(p) || p <= 0) {
        throw new Error("Invalid principal amount for MSME interest calculation.");
    }

    // ── 0A. TRADER EXCLUSION GATE (MINISTRY OF MSME OM DATED 02.07.2021) ──
    // Retail and Wholesale Trade (NIC Codes 45, 46, 47) were admitted to Udyam
    // solely for Priority Sector Lending (PSL) purposes. Delayed payment interest
    // under Chapter V of MSMED Act (Section 16) and Section 43B(h) disallowance
    // do NOT apply to traders.
    const resolvedNic = String(vendorNicCode || nicCode || '').trim();
    const isTraderByNic = /^(45|46|47)/.test(resolvedNic);
    const isTraderByType = typeof supplierType === 'string' && /TRADER|RETAIL|WHOLESALE/i.test(supplierType);
    const traderExclusionActive = isTrader === true || isTraderByNic || isTraderByType;

    if (traderExclusionActive) {
        return {
            principal: p,
            status: 'EXEMPT_TRADER_EXCLUSION',
            isTrader: true,
            isTraderExclusionApplied: true,
            vendorNicCode: resolvedNic || null,
            supplierType: supplierType || (isTraderByNic ? 'TRADER' : null),
            overdueDays: 0,
            totalInterest: 0,
            penalInterestAmount: 0,
            totalInterestRoundedSection288B: 0,
            totalPayable: p,
            effectiveAnnualRatePercent: 0,
            effectiveAnnualInterestRatePercent: 0,
            monthlyRestsCount: 0,
            monthlyRests: [],
            statutoryRule: 'MSME_TRADER_EXCLUSION_OM_2021',
            statutoryCitation: 'Ministry of MSME OM No. 5/2(2)/2021-E/P & G/Policy dated 02-07-2021 and RBI Circular FIDD.MSME & NFS.BC.No.13/06.02.31/2021-22',
            statutoryBasis: 'Wholesale and Retail Traders (NIC Codes 45, 46, 47) are admitted to Udyam solely for Priority Sector Lending (PSL). Chapter V delayed payment interest u/s 16 of MSMED Act and Section 43B(h) do NOT apply.',
            statutoryRationale: 'Enterprises registered under Retail and Wholesale Trade (NIC Codes 45, 46, 47) are restricted strictly to Priority Sector Lending (PSL) benefits. Delayed payment interest under Section 16 of MSMED Act and Section 43B(h) disallowance do NOT apply to traders.'
        };
    }

    if (!dueDate && !accrualStartDate && !acceptanceDate && !agreedPaymentDate && !deliveryDate) {
        throw new Error("Invalid statutory dates: at least one of acceptanceDate, deliveryDate, agreedPaymentDate, dueDate, or accrualStartDate must be provided for MSME interest calculation.");
    }

    // ── 0. SECTION 2(h) DAY OF ACCEPTANCE & DEEMED ACCEPTANCE RESOLUTION ──
    let effectiveAcceptanceDate = acceptanceDate;
    let acceptanceType = acceptanceDate ? 'EXPLICIT_ACCEPTANCE_DATE' : null;

    if (!effectiveAcceptanceDate && deliveryDate) {
        const dDelivery = new Date(String(deliveryDate).slice(0, 10) + 'T00:00:00.000Z');
        if (isNaN(dDelivery.getTime())) {
            throw new Error("Invalid deliveryDate format (expected YYYY-MM-DD).");
        }

        if (writtenObjectionDate) {
            const dObjection = new Date(String(writtenObjectionDate).slice(0, 10) + 'T00:00:00.000Z');
            const diffDays = Math.round((dObjection.getTime() - dDelivery.getTime()) / 86400000);

            // Section 2(h)(ii): Objection must be made within 15 days from delivery
            if (diffDays >= 0 && diffDays <= 15) {
                if (objectionRemovalDate) {
                    effectiveAcceptanceDate = String(objectionRemovalDate).slice(0, 10);
                    acceptanceType = 'ACCEPTANCE_UPON_OBJECTION_REMOVAL';
                } else {
                    return {
                        principal: p,
                        status: 'DISPUTE_CLOCK_PAUSED',
                        statutoryDetermination: 'INDETERMINATE_DISPUTE_UNRESOLVED',
                        isDisputePaused: true,
                        isAuthorizedToDisburse: false,
                        reason: 'Written objection filed within 15 days of delivery remains unresolved. Acceptance clock paused under Section 2(h)(ii). Objection removal date required to determine due date and interest.',
                        acceptanceDate: null,
                        statutoryDueDate: null,
                        overdueDays: null,
                        totalInterest: null,
                        penalInterestAmount: null,
                        totalInterestRoundedSection288B: null,
                        totalPayable: null,
                        monthlyRestsCount: 0,
                        monthlyRests: []
                    };
                }
            } else {
                effectiveAcceptanceDate = String(deliveryDate).slice(0, 10);
                acceptanceType = 'DEEMED_ACCEPTANCE_UNTIMELY_OBJECTION_DISREGARDED';
            }
        } else {
            effectiveAcceptanceDate = String(deliveryDate).slice(0, 10);
            acceptanceType = 'DEEMED_ACCEPTANCE_NO_OBJECTION';
        }
    }

    // ── 1. STATUTORY DATE RESOLUTION (MSMED ACT 2006 SECTIONS 15, 16 & 2(b)) ──
    // Section 16 prescribes two distinct commencement rules depending on whether a written agreement exists:
    //
    // Branch A (No written agreement):
    //   - Section 15: Buyer shall make payment "before the appointed day".
    //   - Section 2(b): "appointed day means the day following immediately after the expiry of the period of fifteen days from the day of acceptance".
    //   - Section 16: Buyer is liable to pay interest "from the appointed day".
    //   - Execution: Interest accrues ON the appointed day itself (acceptanceDate + 16 days).
    //
    // Branch B (Written agreement):
    //   - Section 15: Buyer shall make payment "on or before the date agreed upon".
    //   - Section 15 Proviso: In no case shall the agreed period exceed 45 days from acceptance.
    //   - Section 16: Buyer is liable to pay interest "from the date immediately following the date agreed upon".
    //   - Execution: Interest accrues starting agreedDate + 1 day (or cappedDate + 1 day).

    let startStr = null;
    let statutoryDueDateStr = null;
    let appointedDayStr = null;
    let effectiveAgreedDateStr = null;
    let statutoryCapApplied = false;
    let statutoryRule = 'LEGACY_DUE_DATE';
    let statutoryCitation = 'MSMED Act, 2006 Section 16';

    const endStr = String(paymentDate).slice(0, 10);
    const payment = new Date(endStr + 'T00:00:00.000Z');

    if (effectiveAcceptanceDate) {
        const dAccept = new Date(String(effectiveAcceptanceDate).slice(0, 10) + 'T00:00:00.000Z');
        if (isNaN(dAccept.getTime())) {
            throw new Error("Invalid acceptanceDate format (expected YYYY-MM-DD).");
        }

        // Section 2(b): Appointed day = day following immediately after expiry of 15 days
        const dAppointed = new Date(dAccept.getTime() + 16 * 86400000);
        appointedDayStr = dAppointed.toISOString().split('T')[0];

        const isWritten = hasWrittenAgreement === true || (hasWrittenAgreement === null && Boolean(agreedPaymentDate));

        if (isWritten && agreedPaymentDate) {
            // Branch B: Written agreement
            statutoryRule = 'SECTION_16_FROM_DAY_AFTER_AGREED_DATE';
            statutoryCitation = 'MSMED Act, 2006 Section 16 (interest running from date immediately following agreed date u/s 15 proviso)';
            
            const dAgreed = new Date(String(agreedPaymentDate).slice(0, 10) + 'T00:00:00.000Z');
            if (isNaN(dAgreed.getTime())) {
                throw new Error("Invalid agreedPaymentDate format (expected YYYY-MM-DD).");
            }

            // Section 15 Proviso: Maximum 45 days from day of acceptance
            const dCap = new Date(dAccept.getTime() + 45 * 86400000);
            if (dAgreed > dCap) {
                statutoryCapApplied = true;
                effectiveAgreedDateStr = dCap.toISOString().split('T')[0];
                statutoryDueDateStr = effectiveAgreedDateStr;
                const dAccrual = new Date(dCap.getTime() + 86400000);
                startStr = dAccrual.toISOString().split('T')[0];
            } else {
                statutoryCapApplied = false;
                effectiveAgreedDateStr = dAgreed.toISOString().split('T')[0];
                statutoryDueDateStr = effectiveAgreedDateStr;
                const dAccrual = new Date(dAgreed.getTime() + 86400000);
                startStr = dAccrual.toISOString().split('T')[0];
            }
        } else {
            // Branch A: No written agreement (Appointed Day rule)
            statutoryRule = 'SECTION_16_FROM_APPOINTED_DAY';
            statutoryCitation = 'MSMED Act, 2006 Section 16 (interest running from appointed day under Section 2(b))';
            
            // Payment due before appointed day (acceptance + 15 days)
            const dDue = new Date(dAccept.getTime() + 15 * 86400000);
            statutoryDueDateStr = dDue.toISOString().split('T')[0];
            // Interest runs FROM the appointed day (acceptance + 16 days)
            startStr = appointedDayStr;
        }
    } else {
        // Direct / legacy parameter fallback
        const dueDateStr = dueDate ? String(dueDate).slice(0, 10) : null;
        const isDayAfter = commenceDayAfter || dateConvention === 'SECTION_16_DAY_AFTER_DUE' || dateConvention === 'FROM_DAY_AFTER_AGREED_DATE';
        const isFromAppointed = dateConvention === 'FROM_APPOINTED_DAY';

        if (accrualStartDate) {
            startStr = String(accrualStartDate).slice(0, 10);
            statutoryDueDateStr = dueDateStr || startStr;
            statutoryRule = 'EXPLICIT_ACCRUAL_START_DATE';
        } else if (isFromAppointed && dueDateStr) {
            // Caller passed appointed day as dueDate; interest runs from appointed day
            startStr = dueDateStr;
            const dDue = new Date(dueDateStr + 'T00:00:00.000Z');
            const dBeforeAppointed = new Date(dDue.getTime() - 86400000);
            statutoryDueDateStr = dBeforeAppointed.toISOString().split('T')[0];
            statutoryRule = 'SECTION_16_FROM_APPOINTED_DAY';
            statutoryCitation = 'MSMED Act, 2006 Section 16 (interest running from appointed day under Section 2(b))';
        } else if (isDayAfter && dueDateStr) {
            // Caller passed agreed payment date; interest runs from day immediately following
            const dDue = new Date(dueDateStr + 'T00:00:00.000Z');
            if (isNaN(dDue.getTime())) {
                throw new Error("Invalid dueDate format (expected YYYY-MM-DD).");
            }
            statutoryDueDateStr = dueDateStr;
            const dAccrual = new Date(dDue.getTime() + 86400000);
            startStr = dAccrual.toISOString().split('T')[0];
            statutoryRule = 'SECTION_16_FROM_DAY_AFTER_AGREED_DATE';
            statutoryCitation = 'MSMED Act, 2006 Section 16 (interest running from date immediately following agreed date u/s 15 proviso)';
        } else {
            // Default legacy inclusive
            startStr = dueDateStr;
            statutoryDueDateStr = dueDateStr;
            statutoryRule = 'LEGACY_DUE_DATE';
        }
    }

    const start = new Date(startStr + 'T00:00:00.000Z');

    if (isNaN(start.getTime()) || isNaN(payment.getTime())) {
        throw new Error("Invalid dueDate or paymentDate format (expected YYYY-MM-DD).");
    }

    let effectiveBankRate = (rbiBankRate !== null && rbiBankRate !== undefined)
        ? Number(rbiBankRate)
        : getRbiBankRateForDate(statutoryDueDateStr || startStr || paymentDate);

    if (isNaN(effectiveBankRate) || !Number.isFinite(effectiveBankRate) || effectiveBankRate <= 0) {
        throw new Error(`Invalid rbiBankRate for MSME calculation: expected finite positive rate, received ${rbiBankRate}.`);
    }

    if (effectiveBankRate > 1) {
        // Defensive normalization: caller passed whole percentage e.g. 6.5 instead of 0.065
        effectiveBankRate = effectiveBankRate / 100;
    }

    const annualRate = effectiveBankRate * MSMED_RATE_MULTIPLIER; // e.g. 0.195 (19.5%)
    const activeDateConvention = dateConvention || 'START_DATE_INCLUSIVE';

    // Zero-Liability Check: Payment made on or before statutory due date
    const dDueRef = statutoryDueDateStr ? new Date(statutoryDueDateStr + 'T00:00:00.000Z') : start;
    if (payment <= dDueRef) {
        return {
            principal: p,
            effectiveAnnualRatePercent: Number((effectiveBankRate * MSMED_RATE_MULTIPLIER * 100).toFixed(2)),
            annualRateDecimal: effectiveBankRate * MSMED_RATE_MULTIPLIER,
            rbiBankRate: effectiveBankRate,
            statutoryRule,
            statutoryCitation,
            statutoryDueDate: statutoryDueDateStr,
            firstAccrualDate: startStr,
            paymentDate: endStr,
            acceptanceDate: effectiveAcceptanceDate ? String(effectiveAcceptanceDate).slice(0, 10) : null,
            acceptanceType,
            appointedDay: appointedDayStr,
            agreedPaymentDate: effectiveAgreedDateStr || (agreedPaymentDate ? String(agreedPaymentDate).slice(0, 10) : null),
            statutoryCapApplied,
            dateConvention: activeDateConvention,
            includePaymentDate,
            dayCountConvention: 'ACTUAL/365_OR_366',
            dayCountConventionBasis: 'DOCUMENTED_CALCULATION_CONVENTION',
            overdueDays: 0,
            totalInterest: 0,
            penalInterestAmount: 0,
            totalInterestRoundedSection288B: 0,
            totalPayable: p,
            monthlyRestsCount: 0,
            monthlyRests: [],
            statutoryBasis: `Payment made on or before statutory due date (${statutoryDueDateStr}). Zero penal interest accrued under Section 16 of MSMED Act, 2006.`
        };
    }

    // Accrual End Boundary:
    // If includePaymentDate is true, accrual continues through the end of paymentDate.
    // If includePaymentDate is false, accrual terminates at opening of paymentDate (paymentDate - 1 day).
    const accrualEnd = includePaymentDate
        ? new Date(payment.getTime() + 86400000)
        : payment;

    if (accrualEnd <= start) {
        return {
            principal: p,
            effectiveAnnualRatePercent: Number((annualRate * 100).toFixed(2)),
            annualRateDecimal: annualRate,
            rbiBankRate: effectiveBankRate,
            dueDate: dueDateStr,
            firstAccrualDate: startStr,
            paymentDate: endStr,
            dateConvention: activeDateConvention,
            includePaymentDate,
            dayCountConvention: 'ACTUAL/365_OR_366',
            overdueDays: 0,
            totalInterest: 0,
            penalInterestAmount: 0,
            totalInterestRoundedSection288B: 0,
            totalPayable: p,
            monthlyRestsCount: 0,
            monthlyRests: [],
            statutoryBasis: "Zero overdue accrual days under selected payment date boundary."
        };
    }

    const totalDays = Math.round((accrualEnd.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));

    // Calculate month-by-month compounding with monthly rests strictly in UTC
    const monthlyRests = [];
    let currentBalance = p;
    let iterDate = new Date(start.getTime());
    let restIndex = 1;

    while (iterDate < accrualEnd) {
        const year = iterDate.getUTCFullYear();
        const month = iterDate.getUTCMonth();
        // Start of next calendar month in UTC
        const nextMonthStart = new Date(Date.UTC(year, month + 1, 1, 0, 0, 0, 0));
        const restEnd = accrualEnd < nextMonthStart ? accrualEnd : nextMonthStart;

        const daysInRest = Math.round((restEnd.getTime() - iterDate.getTime()) / (1000 * 60 * 60 * 24));
        if (daysInRest > 0) {
            const daysInYear = ((year % 4 === 0 && year % 100 !== 0) || year % 400 === 0) ? 366 : 365;
            const periodInterest = currentBalance * annualRate * (daysInRest / daysInYear);
            const closing = currentBalance + periodInterest;

            const inclusivePeriodEnd = new Date(restEnd.getTime() - 86400000).toISOString().split('T')[0];

            monthlyRests.push({
                restNumber: restIndex++,
                periodStart: iterDate.toISOString().split('T')[0],
                periodEnd: inclusivePeriodEnd,
                daysInPeriod: daysInRest,
                yearDenominator: daysInYear,
                openingPrincipal: Number(currentBalance.toFixed(2)),
                openingPrincipalFullPrecision: currentBalance,
                interestAccrued: Number(periodInterest.toFixed(2)),
                interestAccruedFullPrecision: periodInterest,
                closingBalance: Number(closing.toFixed(2)),
                closingBalanceFullPrecision: closing,
                capitalizedAt: restEnd < accrualEnd ? `End of calendar month (${inclusivePeriodEnd} 24:00 UTC)` : `Payment settlement (${endStr})`
            });

            currentBalance = closing;
        }

        iterDate = restEnd;
    }

    const totalInterest = Number((currentBalance - p).toFixed(2));
    const totalPayable = Number(currentBalance.toFixed(2));

    return {
        principal: p,
        effectiveAnnualRatePercent: Number((annualRate * 100).toFixed(2)),
        annualRateDecimal: annualRate,
        rbiBankRate: effectiveBankRate,
        statutoryRule,
        statutoryCitation,
        statutoryDueDate: statutoryDueDateStr,
        dueDate: statutoryDueDateStr,
        firstAccrualDate: startStr,
        paymentDate: endStr,
        acceptanceDate: effectiveAcceptanceDate ? String(effectiveAcceptanceDate).slice(0, 10) : null,
        acceptanceType,
        appointedDay: appointedDayStr,
        agreedPaymentDate: effectiveAgreedDateStr || (agreedPaymentDate ? String(agreedPaymentDate).slice(0, 10) : null),
        statutoryCapApplied,
        dateConvention: activeDateConvention,
        includePaymentDate,
        dayCountConvention: 'ACTUAL/365_OR_366',
        dayCountConventionBasis: 'DOCUMENTED_CALCULATION_CONVENTION',
        overdueDays: totalDays,
        totalInterest,
        penalInterestAmount: totalInterest,
        totalInterestRoundedSection288B: roundSection288B(totalInterest),
        totalPayable,
        monthlyRestsCount: monthlyRests.length,
        monthlyRests,
        statutoryBasis: `${statutoryCitation} (Compound interest with monthly rests at 3x RBI Bank Rate: ${Number((annualRate * 100).toFixed(2))}%)`
    };
}

/**
 * ─────────────────────────────────────────────────────────────────────────────
 * 2. SECTION 43B(h) TEMPORAL & FISCAL BOUNDARY STATE MACHINE
 * ─────────────────────────────────────────────────────────────────────────────
 * Accurately branches:
 * - Trader vs Manufacturer qualification (NIC 45, 46, 47 exclusion)
 * - 45-day written contract vs 15-day statutory baseline
 * - Dispute clock pauses (Section 15 Explanation)
 * - Same FY (before March 31) payment vs crossing FY boundary
 */
export function evaluateSection43bH({
    invoiceDate,
    acceptanceDate = null,
    agreedTermsDays = null,
    disputeStartDate = null,
    disputeEndDate = null,
    paymentDate = null,
    fyEndDate = null,
    supplierType = 'MANUFACTURER', // 'MICRO_MANUFACTURER', 'SMALL_MANUFACTURER', 'TRADER', 'MEDIUM'
    nicCode = null
}) {
    const baseAcceptanceDate = new Date(acceptanceDate || invoiceDate);
    
    // Dynamically calculate fiscal year end based on invoice/acceptance date if not explicitly overridden
    let effectiveFyEndDate = fyEndDate;
    if (!effectiveFyEndDate) {
        const refDate = new Date(acceptanceDate || invoiceDate);
        if (!isNaN(refDate.getTime())) {
            const month = refDate.getMonth(); // Jan=0, Feb=1, Mar=2, Apr=3
            const year = refDate.getFullYear();
            const endYear = month >= 3 ? year + 1 : year;
            effectiveFyEndDate = `${endYear}-03-31`;
        } else {
            effectiveFyEndDate = '2026-03-31';
        }
    }
    const fyEnd = new Date(effectiveFyEndDate);
    const actualPayment = paymentDate ? new Date(paymentDate) : null;

    // ── GATE 1: SUPPLIER ELIGIBILITY CHECK ──
    const isTrader = supplierType?.toUpperCase()?.includes('TRADER') 
        || (nicCode && /^(45|46|47)/.test(String(nicCode).trim()));
    const isMedium = supplierType?.toUpperCase()?.includes('MEDIUM');

    if (isTrader) {
        return {
            isCoveredBy43Bh: false,
            qualificationStatus: 'EXCLUDED_TRADER',
            reason: 'Supplier is registered as a Retail or Wholesale Trader (NIC 45/46/47). Ministry of MSME OM No. 2/1(5)/2021-P&G/Policy dated 02.07.2021 restricts Chapter V and Section 43B(h) protection to Manufacturers and Service Providers only.',
            disallowedInCurrentFY: false,
            taxDeductionStatus: 'ALLOWABLE_ON_ACCRUAL',
            statutoryAction: 'Expense is 100% deductible in the current financial year on mercantile accrual basis, regardless of payment date.'
        };
    }

    if (isMedium) {
        return {
            isCoveredBy43Bh: false,
            qualificationStatus: 'EXCLUDED_MEDIUM_ENTERPRISE',
            reason: 'Supplier is a Medium Enterprise. Section 43B(h) exclusively protects Micro and Small enterprises. Medium enterprises are outside the purview.',
            disallowedInCurrentFY: false,
            taxDeductionStatus: 'ALLOWABLE_ON_ACCRUAL',
            statutoryAction: 'Expense is deductible on standard mercantile accrual basis.'
        };
    }

    // ── GATE 2: STATUTORY PAYMENT TIMELINE & DISPUTE PAUSE ──
    // Without written agreement: strictly 15 days from acceptance
    // With written agreement: agreed term up to a maximum cap of 45 days
    let allowedDays = 15;
    let agreementType = 'STATUTORY_DEFAULT_15_DAYS';

    if (agreedTermsDays && Number(agreedTermsDays) > 0) {
        const terms = Number(agreedTermsDays);
        if (terms <= 45) {
            allowedDays = terms;
            agreementType = `WRITTEN_AGREEMENT_${terms}_DAYS`;
        } else {
            allowedDays = 45; // Section 15 statutory cap
            agreementType = 'WRITTEN_AGREEMENT_CAPPED_AT_45_DAYS';
        }
    }

    // Check dispute pause (Section 15 Explanation)
    let disputeDurationDays = 0;
    if (disputeStartDate && disputeEndDate) {
        const dStart = new Date(disputeStartDate);
        const dEnd = new Date(disputeEndDate);
        if (dEnd > dStart) {
            disputeDurationDays = Math.ceil((dEnd.getTime() - dStart.getTime()) / (1000 * 60 * 60 * 24));
        }
    }

    const statutoryDueDate = new Date(baseAcceptanceDate.getTime() + ((allowedDays + disputeDurationDays) * 24 * 60 * 60 * 1000));

    // ── GATE 3: TEMPORAL TAX ASSESSMENT (MARCH 31 FY BOUNDARY) ──
    const dueDateStr = statutoryDueDate.toISOString().split('T')[0];
    const fyEndStr = fyEnd.toISOString().split('T')[0];

    // Case A: Payment already made
    if (actualPayment) {
        const paymentDateStr = actualPayment.toISOString().split('T')[0];
        const isPaidWithinStatutoryTerms = actualPayment <= statutoryDueDate;
        const isPaidBeforeMarch31 = actualPayment <= fyEnd;

        if (isPaidWithinStatutoryTerms) {
            return {
                isCoveredBy43Bh: true,
                qualificationStatus: 'QUALIFIED_MICRO_SMALL',
                statutoryDueDate: dueDateStr,
                actualPaymentDate: paymentDateStr,
                disputeDurationDays,
                agreementType,
                isPaidWithinTerms: true,
                isPaidBeforeMarch31,
                disallowedInCurrentFY: false,
                taxDeductionStatus: 'FULLY_DEDUCTIBLE_CURRENT_FY',
                statutoryAction: `Payment made on ${paymentDateStr} within the statutory clock (due ${dueDateStr}). Expense is 100% allowable on accrual basis in the current assessment year.`
            };
        }

        // Paid LATE, but before March 31 of same FY:
        if (isPaidBeforeMarch31) {
            return {
                isCoveredBy43Bh: true,
                qualificationStatus: 'QUALIFIED_MICRO_SMALL',
                statutoryDueDate: dueDateStr,
                actualPaymentDate: paymentDateStr,
                disputeDurationDays,
                agreementType,
                isPaidWithinTerms: false,
                isPaidBeforeMarch31: true,
                disallowedInCurrentFY: false, // CRITICAL CA INVARIANT: Paid before March 31 = Deductible on actual payment in same FY!
                taxDeductionStatus: 'FULLY_DEDUCTIBLE_CURRENT_FY_ON_PAYMENT',
                msmePenalInterestApplicable: true,
                statutoryAction: `Payment was delayed past statutory due date (${dueDateStr}), but made BEFORE fiscal year-end (${paymentDateStr} <= ${fyEndStr}). Under Section 43B(h), the deduction is allowable on actual payment basis in this same financial year (ZERO year-end tax disallowance). Note: MSMED Section 16 compound interest remains payable to the supplier for the delayed days.`
            };
        }

        // Paid LATE and crossed March 31 into next FY:
        const currentFY = `${fyEnd.getFullYear() - 1}-${String(fyEnd.getFullYear()).slice(-2)}`;
        const nextFY = `${fyEnd.getFullYear()}-${String(fyEnd.getFullYear() + 1).slice(-2)}`;

        return {
            isCoveredBy43Bh: true,
            qualificationStatus: 'QUALIFIED_MICRO_SMALL',
            statutoryDueDate: dueDateStr,
            actualPaymentDate: paymentDateStr,
            disputeDurationDays,
            agreementType,
            isPaidWithinTerms: false,
            isPaidBeforeMarch31: false,
            disallowedInCurrentFY: true,
            assessmentYearDisallowed: `AY ${fyEnd.getFullYear() + 1}-${String(fyEnd.getFullYear() + 2).slice(-2)} (FY ${currentFY})`,
            assessmentYearAllowed: `AY ${actualPayment.getFullYear() + 1}-${String(actualPayment.getFullYear() + 2).slice(-2)} (FY ${nextFY})`,
            taxDeductionStatus: 'DISALLOWED_CURRENT_FY_CARRIED_FORWARD',
            msmePenalInterestApplicable: true,
            statutoryAction: `Overdue invoice remained unpaid as of March 31 (${fyEndStr}) and crossed into next FY. The expense is strictly DISALLOWED in FY ${currentFY} under Section 43B(h), and becomes deductible in FY ${nextFY} (the year of actual payment). MSMED Section 16 penal interest is applicable and is non-deductible under Section 23 of MSMED Act.`
        };
    }

    // Case B: Currently unpaid
    const today = new Date();
    const isCurrentlyOverdue = today > statutoryDueDate;

    return {
        isCoveredBy43Bh: true,
        qualificationStatus: 'QUALIFIED_MICRO_SMALL',
        statutoryDueDate: dueDateStr,
        actualPaymentDate: null,
        disputeDurationDays,
        agreementType,
        isCurrentlyOverdue,
        statutoryAction: isCurrentlyOverdue
            ? `Invoice is currently overdue past ${dueDateStr}. To prevent Section 43B(h) disallowance, payment must be completed before ${fyEndStr} (March 31).`
            : `Invoice is within terms until ${dueDateStr}. Ensure payment is settled before the due date.`
    };
}

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CORPORATE FINANCE: MSME REPORTING & SCHEDULE III PAYABLE AGING
 * MCA Order 22.01.2019 (Form MSME-1) & Companies Act 2013 Schedule III
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Strictly separates:
 * 1. Accounting Age: Days from invoice date to reporting date (Schedule III brackets).
 * 2. Statutory Overdue Status: Days past Section 15 statutory due date (Appointed Day).
 * 3. MCA Form MSME-1: Bi-annual filing of Micro/Small dues exceeding 45 days.
 */
export function generateMsme1AndScheduleThreeAging({
    asOfDate = new Date().toISOString().slice(0, 10),
    halfYearPeriod = null, // 'APRIL_TO_SEPTEMBER' | 'OCTOBER_TO_MARCH'
    payables = []
}) {
    if (!Array.isArray(payables)) {
        throw new Error('[@mudraforge/statutory] Invalid payables: expected an array of payable records.');
    }

    const dAsOf = new Date(String(asOfDate).slice(0, 10) + 'T00:00:00.000Z');
    if (isNaN(dAsOf.getTime())) {
        throw new Error(`[@mudraforge/statutory] Invalid asOfDate format "${asOfDate}". Expected YYYY-MM-DD.`);
    }

    // Schedule III 4-Tier Accumulators
    const agingMatrix = {
        msmeUndisputed: { lessThan1Year: 0, oneToTwoYears: 0, twoToThreeYears: 0, moreThan3Years: 0, total: 0 },
        msmeDisputed:   { lessThan1Year: 0, oneToTwoYears: 0, twoToThreeYears: 0, moreThan3Years: 0, total: 0 },
        othersUndisputed: { lessThan1Year: 0, oneToTwoYears: 0, twoToThreeYears: 0, moreThan3Years: 0, total: 0 },
        othersDisputed:   { lessThan1Year: 0, oneToTwoYears: 0, twoToThreeYears: 0, moreThan3Years: 0, total: 0 },
        unbilledDues: 0,
        grandTotalTradePayables: 0
    };

    const msme1ReportableList = [];
    const assessedPayables = [];

    for (const item of payables) {
        const amount = Math.max(0, Number(item.amount || item.invoiceAmount || 0));
        if (amount === 0) continue;

        const isUnbilled = item.isUnbilled === true;
        if (isUnbilled) {
            agingMatrix.unbilledDues = roundToPaisa(agingMatrix.unbilledDues + amount);
            agingMatrix.grandTotalTradePayables = roundToPaisa(agingMatrix.grandTotalTradePayables + amount);
            assessedPayables.push({
                invoiceNumber: item.invoiceNumber || 'UNBILLED',
                supplierName: item.supplierName || 'N/A',
                amount,
                isUnbilled: true,
                accountingAgeDays: 0,
                statutoryOverdueDays: 0,
                msme1Reportable: false
            });
            continue;
        }

        // 1. ACCOUNTING AGE (Schedule III): Days from invoice date to asOfDate
        const invDateStr = item.invoiceDate ? String(item.invoiceDate).slice(0, 10) : asOfDate;
        const dInvoice = new Date(invDateStr + 'T00:00:00.000Z');
        const accountingAgeDays = Math.max(0, Math.round((dAsOf.getTime() - dInvoice.getTime()) / 86400000));

        // Enterprise Classification: Micro vs Small vs Medium vs Others
        // Trader exclusion: Retail & Wholesale Trade (NIC 45/46/47) are NOT eligible for MSMED Chapter V / MSME-1
        const nic = String(item.nicCode || item.vendorNicCode || '').trim();
        const isTrader = item.isTrader === true || /^(45|46|47)/.test(nic);
        const isMicroOrSmall = (item.isMicro === true || item.isSmall === true || item.enterpriseType === 'MICRO' || item.enterpriseType === 'SMALL') && !isTrader;
        const isMsme = (isMicroOrSmall || item.isMedium === true || item.enterpriseType === 'MEDIUM') && !isTrader;
        const isDisputed = item.isDisputed === true || Boolean(item.writtenObjectionDate);

        // Map to Schedule III aging bucket
        let bracket = 'lessThan1Year';
        if (accountingAgeDays > 1095) {
            bracket = 'moreThan3Years'; // > 3 years
        } else if (accountingAgeDays > 730) {
            bracket = 'twoToThreeYears'; // 2–3 years
        } else if (accountingAgeDays > 365) {
            bracket = 'oneToTwoYears'; // 1–2 years
        }

        // Accumulate into Schedule III matrix
        let targetCategory;
        if (isMsme) {
            targetCategory = isDisputed ? agingMatrix.msmeDisputed : agingMatrix.msmeUndisputed;
        } else {
            targetCategory = isDisputed ? agingMatrix.othersDisputed : agingMatrix.othersUndisputed;
        }
        targetCategory[bracket] = roundToPaisa(targetCategory[bracket] + amount);
        targetCategory.total = roundToPaisa(targetCategory.total + amount);
        agingMatrix.grandTotalTradePayables = roundToPaisa(agingMatrix.grandTotalTradePayables + amount);

        // 2. STATUTORY OVERDUE STATUS (Section 15 MSMED Act)
        // Statutory due date: 15 days without written agreement; maximum 45 days with written agreement.
        // Day of acceptance: If delivery objection occurred within 15 days, acceptance is date objection removed.
        const deliveryStr = item.deliveryDate ? String(item.deliveryDate).slice(0, 10) : invDateStr;
        const dDelivery = new Date(deliveryStr + 'T00:00:00.000Z');
        
        let dAcceptance = dDelivery;
        let isDisputePaused = false;

        if (item.writtenObjectionDate) {
            const dObj = new Date(String(item.writtenObjectionDate).slice(0, 10) + 'T00:00:00.000Z');
            const objectionDays = Math.round((dObj.getTime() - dDelivery.getTime()) / 86400000);
            if (objectionDays <= 15) {
                if (item.objectionRemovalDate) {
                    dAcceptance = new Date(String(item.objectionRemovalDate).slice(0, 10) + 'T00:00:00.000Z');
                } else {
                    isDisputePaused = true; // Objection active and uncured
                }
            }
        }

        const hasAgreement = item.hasWrittenAgreement === true || Boolean(item.agreedPaymentDate);
        const maxDaysAllowed = hasAgreement ? 45 : 15;
        const dStatutoryDue = new Date(dAcceptance.getTime() + maxDaysAllowed * 86400000);

        let statutoryOverdueDays = 0;
        let isStatutoryOverdue = false;

        if (!isDisputePaused && dAsOf > dStatutoryDue) {
            statutoryOverdueDays = Math.round((dAsOf.getTime() - dStatutoryDue.getTime()) / 86400000);
            isStatutoryOverdue = statutoryOverdueDays > 0;
        }

        // 3. FORM MSME-1 REPORTABLE CRITERIA:
        // Must be Micro or Small enterprise (excluding traders) AND
        // Payment must exceed 45 days from date of acceptance/deemed acceptance AND
        // Outstanding as of half-year end.
        const daysFromAcceptance = isDisputePaused ? 0 : Math.round((dAsOf.getTime() - dAcceptance.getTime()) / 86400000);
        const isMsme1Reportable = isMicroOrSmall && !isDisputePaused && daysFromAcceptance > 45;

        if (isMsme1Reportable) {
            msme1ReportableList.push({
                supplierName: item.supplierName || 'Unknown Vendor',
                supplierPan: item.supplierPan || item.pan || 'PAN_NOT_PROVIDED',
                udyamRegistrationNumber: item.udyamRegistrationNumber || item.udyamRegNo || 'UDYAM_NOT_PROVIDED',
                invoiceNumber: item.invoiceNumber || 'INV_REF',
                invoiceDate: invDateStr,
                acceptanceDate: dAcceptance.toISOString().slice(0, 10),
                amountDueInr: amount,
                accountingAgeDays,
                statutoryOverdueDays,
                daysFromAcceptance,
                reasonForDelay: item.reasonForDelay || 'Operational billing reconciliation'
            });
        }

        assessedPayables.push({
            invoiceNumber: item.invoiceNumber || 'INV_REF',
            supplierName: item.supplierName || 'Unknown Vendor',
            amount,
            accountingAgeDays,
            accountingAgeBracket: bracket,
            isMicroOrSmall,
            isTrader,
            isDisputePaused,
            statutoryDueDate: isDisputePaused ? null : dStatutoryDue.toISOString().slice(0, 10),
            statutoryOverdueDays,
            isStatutoryOverdue,
            msme1Reportable: isMsme1Reportable
        });
    }

    const totalMsme1ReportableAmount = roundToPaisa(msme1ReportableList.reduce((sum, x) => sum + x.amountDueInr, 0));

    return {
        status: 'SUCCESS',
        statutoryDetermination: 'APPLICABLE',
        governingLaw: 'Companies Act, 2013 Schedule III & MCA Order 22.01.2019 (Form MSME-1)',
        asOfDate,
        halfYearPeriod: halfYearPeriod || (dAsOf.getUTCMonth() >= 3 && dAsOf.getUTCMonth() <= 8 ? 'APRIL_TO_SEPTEMBER' : 'OCTOBER_TO_MARCH'),
        totalInvoicesEvaluated: payables.length,
        scheduleThreeAgingMatrix: agingMatrix,
        mcaFormMsme1: {
            reportableVendorsCount: msme1ReportableList.length,
            totalReportableAmountInr: totalMsme1ReportableAmount,
            filingDeadline: halfYearPeriod === 'APRIL_TO_SEPTEMBER' || (dAsOf.getUTCMonth() >= 3 && dAsOf.getUTCMonth() <= 8) ? 'October 31' : 'April 30',
            reportablePayables: msme1ReportableList
        },
        assessedPayablesSummary: {
            totalPayablesCount: assessedPayables.length,
            statutoryOverdueCount: assessedPayables.filter(x => x.isStatutoryOverdue).length,
            disputePausedCount: assessedPayables.filter(x => x.isDisputePaused).length
        },
        isAuthorizedToDisburse: false,
        statutoryRationale: `Schedule III Trade Payables total: ₹${agingMatrix.grandTotalTradePayables.toLocaleString('en-IN')}. MCA Form MSME-1 identified ${msme1ReportableList.length} overdue Micro/Small items exceeding 45 days (₹${totalMsme1ReportableAmount.toLocaleString('en-IN')}). Accounting age is segregated from statutory overdue days.`
    };
}
