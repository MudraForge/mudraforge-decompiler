/**
 * DETERMINISTIC STATUTORY PAYROLL & SALARY WITHHOLDING CALCULATORS
 * ==============================================================================
 * Implements pure mathematical functions for Indian payroll compliance:
 * 1. Salary TDS under Income-tax Act, 1961 Section 192 & Income-tax Act, 2025 Section 392(1)
 *    - Default Section 115BAC(1A) New Tax Regime with ₹75,000 standard deduction
 *    - Old Tax Regime opt-out with ₹50,000 standard deduction and Form 12BB deductions
 *    - Section 87A rebate and 4% Health & Education Cess
 * 2. Employees' Provident Fund (EPF & EPS) under EPF Act, 1952
 *    - ₹15,000 standard wage ceiling vs. Paragraph 26(6) joint higher wage declarations
 *    - Age 58 EPS cessation rule
 * 3. Employees' State Insurance (ESI) under ESI Act, 1948
 *    - ₹21,000 gross wage threshold & ₹176/day exempt employee floor
 * 4. Professional Tax (PT) across Indian state statutory slabs
 *
 * Statutory Invariant: Every calculator strictly returns isAuthorizedToDisburse: false.
 * Zero external npm dependencies. Integer paise arithmetic to prevent float drift.
 * ==============================================================================
 */

import { roundToPaisa, roundSection288B } from './numeric_utils.js';

export const TAX_REGIMES = Object.freeze({
    NEW: 'NEW',
    OLD: 'OLD'
});

export const NEW_REGIME_SLABS_AY2025_26 = Object.freeze([
    { min: 0, max: 300000, ratePercent: 0 },
    { min: 300000, max: 700000, ratePercent: 5 },
    { min: 700000, max: 1000000, ratePercent: 10 },
    { min: 1000000, max: 1200000, ratePercent: 15 },
    { min: 1200000, max: 1500000, ratePercent: 20 },
    { min: 1500000, max: Infinity, ratePercent: 30 }
]);

export const OLD_REGIME_SLABS = Object.freeze([
    { min: 0, max: 250000, ratePercent: 0 },
    { min: 250000, max: 500000, ratePercent: 5 },
    { min: 500000, max: 1000000, ratePercent: 20 },
    { min: 1000000, max: Infinity, ratePercent: 30 }
]);

export const EPF_STATUTORY_CONSTANTS = Object.freeze({
    STANDARD_WAGE_CEILING_INR: 15000,
    EMPLOYEE_RATE_PERCENT: 12.0,
    EMPLOYER_TOTAL_RATE_PERCENT: 12.0,
    EPS_RATE_PERCENT: 8.33,
    EPS_MAX_MONTHLY_CAP_INR: 1250,
    EPS_SUPERANNUATION_AGE_YEARS: 58
});

export const ESI_STATUTORY_CONSTANTS = Object.freeze({
    GROSS_WAGE_CEILING_INR: 21000,
    EMPLOYEE_RATE_PERCENT: 0.75,
    EMPLOYER_RATE_PERCENT: 3.25,
    DAILY_EXEMPT_WAGE_FLOOR_INR: 176
});

/**
 * Evaluates annual and monthly Salary TDS under Section 192 (1961 Act)
 * or Section 392(1) (2025 Act).
 */
export function evaluateSalaryTds({
    annualGrossSalaryInr,
    taxRegime = TAX_REGIMES.NEW,
    form12BBClaims = {},
    standardDeductionOverride = null,
    monthsElapsedInFy = 0,
    priorTdsDeductedInr = 0,
    transactionDate = null,
    isSeniorCitizen = false,
    isSuperSeniorCitizen = false
} = {}) {
    const gross = Math.max(0, Number(annualGrossSalaryInr) || 0);
    const dateStr = transactionDate ? String(transactionDate).slice(0, 10) : new Date().toISOString().slice(0, 10);
    const isAct2025 = Boolean(dateStr >= '2026-04-01');

    const normalizedRegime = String(taxRegime || '').toUpperCase().trim() === TAX_REGIMES.OLD
        ? TAX_REGIMES.OLD
        : TAX_REGIMES.NEW;

    // Standard Deduction: ₹75,000 for New Regime (Finance (No. 2) Act, 2024), ₹50,000 for Old Regime
    let standardDeductionInr = standardDeductionOverride !== null && standardDeductionOverride !== undefined
        ? Math.max(0, Number(standardDeductionOverride) || 0)
        : (normalizedRegime === TAX_REGIMES.NEW ? 75000 : 50000);

    let allowedDeductionsInr = standardDeductionInr;
    const deductionsBreakdown = {
        standardDeductionInr
    };

    if (normalizedRegime === TAX_REGIMES.OLD) {
        // Section 80C capped at ₹1,50,000
        const claim80C = Math.min(150000, Math.max(0, Number(form12BBClaims.section80CInr) || 0));
        // Section 80D (Health Insurance) capped at ₹25,000 (or ₹50,000 for senior citizen)
        const cap80D = isSeniorCitizen ? 50000 : 25000;
        const claim80D = Math.min(cap80D, Math.max(0, Number(form12BBClaims.section80DInr) || 0));
        // Section 24(b) Home loan interest capped at ₹2,00,000
        const claimHomeLoan = Math.min(200000, Math.max(0, Number(form12BBClaims.homeLoanInterestSec24Inr) || 0));
        // HRA Exemption under Section 10(13A)
        const claimHra = Math.max(0, Number(form12BBClaims.hraExemptionInr) || 0);
        // LTA Exemption under Section 10(5)
        const claimLta = Math.max(0, Number(form12BBClaims.ltaExemptionInr) || 0);
        // Other eligible deductions
        const claimOther = Math.max(0, Number(form12BBClaims.otherDeductionsInr) || 0);

        deductionsBreakdown.section80CInr = claim80C;
        deductionsBreakdown.section80DInr = claim80D;
        deductionsBreakdown.homeLoanInterestSec24Inr = claimHomeLoan;
        deductionsBreakdown.hraExemptionInr = claimHra;
        deductionsBreakdown.ltaExemptionInr = claimLta;
        deductionsBreakdown.otherDeductionsInr = claimOther;

        allowedDeductionsInr += (claim80C + claim80D + claimHomeLoan + claimHra + claimLta + claimOther);
    }

    const netTaxableIncomeInr = Math.max(0, gross - allowedDeductionsInr);

    // Compute Tax on Slabs
    let taxBeforeRebate = 0;
    const slabs = normalizedRegime === TAX_REGIMES.NEW
        ? NEW_REGIME_SLABS_AY2025_26
        : OLD_REGIME_SLABS;

    let previousMax = 0;
    if (normalizedRegime === TAX_REGIMES.OLD && isSuperSeniorCitizen) {
        previousMax = 500000;
    } else if (normalizedRegime === TAX_REGIMES.OLD && isSeniorCitizen) {
        previousMax = 300000;
    }

    for (const slab of slabs) {
        const slabFloor = Math.max(slab.min, previousMax);
        if (netTaxableIncomeInr > slabFloor) {
            const taxableInThisSlab = Math.min(netTaxableIncomeInr, slab.max) - slabFloor;
            if (taxableInThisSlab > 0 && slab.ratePercent > 0) {
                taxBeforeRebate += taxableInThisSlab * (slab.ratePercent / 100);
            }
        }
    }

    // Section 87A Rebate
    // New Regime: Taxable income up to ₹7,00,000 gets full rebate (₹25,000 max rebate)
    // Old Regime: Taxable income up to ₹5,00,000 gets full rebate (₹12,500 max rebate)
    let section87ARebateInr = 0;
    if (normalizedRegime === TAX_REGIMES.NEW && netTaxableIncomeInr <= 700000) {
        section87ARebateInr = Math.min(taxBeforeRebate, 25000);
    } else if (normalizedRegime === TAX_REGIMES.OLD && netTaxableIncomeInr <= 500000) {
        section87ARebateInr = Math.min(taxBeforeRebate, 12500);
    }

    const taxAfterRebate = Math.max(0, taxBeforeRebate - section87ARebateInr);

    // 4% Health and Education Cess
    const cessInr = taxAfterRebate > 0 ? taxAfterRebate * 0.04 : 0;
    const totalAnnualTdsInr = roundSection288B(taxAfterRebate + cessInr);

    // Monthly Withholding Amortization
    const validElapsed = Math.min(11, Math.max(0, Number(monthsElapsedInFy) || 0));
    const remainingMonths = Math.max(1, 12 - validElapsed);
    const priorDeducted = Math.max(0, Number(priorTdsDeductedInr) || 0);
    const unrecoveredTax = Math.max(0, totalAnnualTdsInr - priorDeducted);
    const monthlyTdsDeductionInr = roundSection288B(unrecoveredTax / remainingMonths);

    const governingSection = isAct2025
        ? 'Income-tax Act, 2025 Section 392(1)'
        : (normalizedRegime === TAX_REGIMES.NEW
            ? 'Income-tax Act, 1961 Section 192 read with Section 115BAC'
            : 'Income-tax Act, 1961 Section 192 (Old Tax Regime Opt-Out)');

    return {
        status: 'SUCCESS',
        governingSection,
        isAct2025,
        taxRegime: normalizedRegime,
        annualGrossSalaryInr: roundToPaisa(gross),
        totalDeductionsAllowedInr: roundToPaisa(allowedDeductionsInr),
        deductionsBreakdown,
        netTaxableIncomeInr: roundToPaisa(netTaxableIncomeInr),
        taxBeforeRebateInr: roundToPaisa(taxBeforeRebate),
        section87ARebateInr: roundToPaisa(section87ARebateInr),
        taxAfterRebateInr: roundToPaisa(taxAfterRebate),
        healthAndEducationCessInr: roundToPaisa(cessInr),
        totalAnnualTdsInr,
        priorTdsDeductedInr: roundToPaisa(priorDeducted),
        remainingMonthsInFy: remainingMonths,
        monthlyTdsDeductionInr,
        averageTaxRatePercent: gross > 0 ? roundToPaisa((totalAnnualTdsInr / gross) * 100) : 0,
        isAuthorizedToDisburse: false
    };
}

/**
 * Evaluates Employees' Provident Fund (EPF) and Employees' Pension Scheme (EPS)
 * contributions under the Employees' Provident Funds and Miscellaneous Provisions Act, 1952.
 */
export function evaluateEpfContribution({
    monthlyBasicAndDaInr,
    epfOnActualWages = false,
    employeeAgeYears = 30,
    isInternationalWorker = false,
    optedOutOfEpf = false
} = {}) {
    if (optedOutOfEpf) {
        return {
            status: 'OPTED_OUT',
            statutoryWageCeilingInr: EPF_STATUTORY_CONSTANTS.STANDARD_WAGE_CEILING_INR,
            effectiveEpfWageInr: 0,
            employeeContributionInr: 0,
            employerEpsContributionInr: 0,
            employerEpfContributionInr: 0,
            totalEmployerContributionInr: 0,
            totalEpfRemittanceInr: 0,
            isAuthorizedToDisburse: false
        };
    }

    const wage = Math.max(0, Number(monthlyBasicAndDaInr) || 0);
    const age = Number(employeeAgeYears) || 30;

    // Wage ceiling applies unless employee has joint declaration under Para 26(6) or is International Worker
    const isCeilingApplicable = !epfOnActualWages && !isInternationalWorker;
    const effectiveEpfWageInr = isCeilingApplicable
        ? Math.min(wage, EPF_STATUTORY_CONSTANTS.STANDARD_WAGE_CEILING_INR)
        : wage;

    // Employee Share: Exactly 12% of effective EPF wage
    const employeeContributionInr = roundToPaisa(effectiveEpfWageInr * (EPF_STATUTORY_CONSTANTS.EMPLOYEE_RATE_PERCENT / 100));

    // Employer Share: Exactly 12% total, split into EPS (8.33%) and EPF (3.67%)
    let employerEpsContributionInr = 0;
    let employerEpfContributionInr = 0;

    if (age >= EPF_STATUTORY_CONSTANTS.EPS_SUPERANNUATION_AGE_YEARS) {
        // EPS membership ceases at age 58; entire 12% employer share goes to EPF
        employerEpsContributionInr = 0;
        employerEpfContributionInr = employeeContributionInr;
    } else {
        // EPS is calculated at 8.33% of capped wage (max ₹1,250/month)
        const epsWage = Math.min(effectiveEpfWageInr, EPF_STATUTORY_CONSTANTS.STANDARD_WAGE_CEILING_INR);
        // Under EPFO Rule 38(1), EPS contribution rounds to nearest integer rupee (e.g. 15,000 * 8.33% = 1,249.50 -> 1,250)
        employerEpsContributionInr = Math.min(
            EPF_STATUTORY_CONSTANTS.EPS_MAX_MONTHLY_CAP_INR,
            Math.round(epsWage * (EPF_STATUTORY_CONSTANTS.EPS_RATE_PERCENT / 100))
        );
        employerEpfContributionInr = roundToPaisa(employeeContributionInr - employerEpsContributionInr);
    }

    const totalEmployerContributionInr = roundToPaisa(employerEpsContributionInr + employerEpfContributionInr);
    const totalEpfRemittanceInr = roundToPaisa(employeeContributionInr + totalEmployerContributionInr);

    return {
        status: 'SUCCESS',
        governingStatute: "Employees' Provident Funds and Miscellaneous Provisions Act, 1952",
        statutoryWageCeilingInr: isCeilingApplicable ? EPF_STATUTORY_CONSTANTS.STANDARD_WAGE_CEILING_INR : null,
        epfOnActualWages: Boolean(epfOnActualWages || isInternationalWorker),
        effectiveEpfWageInr: roundToPaisa(effectiveEpfWageInr),
        employeeContributionInr,
        employerEpsContributionInr,
        employerEpfContributionInr,
        totalEmployerContributionInr,
        totalEpfRemittanceInr,
        isSuperannuationAge: age >= EPF_STATUTORY_CONSTANTS.EPS_SUPERANNUATION_AGE_YEARS,
        isAuthorizedToDisburse: false
    };
}

/**
 * Evaluates Employees' State Insurance (ESI) contributions under the ESI Act, 1948.
 */
export function evaluateEsiContribution({
    monthlyGrossSalaryInr,
    workingDaysInMonth = 26,
    optedOutOfEsi = false
} = {}) {
    const gross = Math.max(0, Number(monthlyGrossSalaryInr) || 0);

    if (optedOutOfEsi || gross > ESI_STATUTORY_CONSTANTS.GROSS_WAGE_CEILING_INR) {
        return {
            status: gross > ESI_STATUTORY_CONSTANTS.GROSS_WAGE_CEILING_INR ? 'EXCEEDS_WAGE_CEILING_EXEMPT' : 'OPTED_OUT',
            isCovered: false,
            monthlyGrossSalaryInr: roundToPaisa(gross),
            wageCeilingInr: ESI_STATUTORY_CONSTANTS.GROSS_WAGE_CEILING_INR,
            employeeContributionInr: 0,
            employerContributionInr: 0,
            totalEsiRemittanceInr: 0,
            isAuthorizedToDisburse: false
        };
    }

    // Exempt Employee Check: Daily average wage <= ₹176 pays 0% employee share
    const days = Math.max(1, Number(workingDaysInMonth) || 26);
    const dailyAverageWage = gross / days;
    const isExemptEmployee = dailyAverageWage <= ESI_STATUTORY_CONSTANTS.DAILY_EXEMPT_WAGE_FLOOR_INR;

    const employeeRate = isExemptEmployee ? 0 : (ESI_STATUTORY_CONSTANTS.EMPLOYEE_RATE_PERCENT / 100);
    const employerRate = ESI_STATUTORY_CONSTANTS.EMPLOYER_RATE_PERCENT / 100;

    const employeeContributionInr = roundToPaisa(gross * employeeRate);
    const employerContributionInr = roundToPaisa(gross * employerRate);
    const totalEsiRemittanceInr = roundToPaisa(employeeContributionInr + employerContributionInr);

    return {
        status: 'SUCCESS',
        governingStatute: "Employees' State Insurance Act, 1948",
        isCovered: true,
        isExemptEmployee,
        monthlyGrossSalaryInr: roundToPaisa(gross),
        employeeContributionInr,
        employerContributionInr,
        totalEsiRemittanceInr,
        isAuthorizedToDisburse: false
    };
}

/**
 * Evaluates State Professional Tax (PT) across major Indian jurisdictions.
 */
export function evaluateProfessionalTax({
    monthlyGrossSalaryInr,
    stateCode = '27', // Default Maharashtra ('27')
    salaryMonth = 1,  // 1-12 (Month 2 = February has special slab in MH)
    gender = 'MALE'
} = {}) {
    const gross = Math.max(0, Number(monthlyGrossSalaryInr) || 0);
    const cleanState = String(stateCode || '27').trim().toUpperCase();
    const cleanGender = String(gender || 'MALE').trim().toUpperCase();
    const month = Number(salaryMonth) || 1;

    let ptInr = 0;
    let stateName = 'Other';

    // Maharashtra (27 / MH)
    if (cleanState === '27' || cleanState === 'MH' || cleanState === 'MAHARASHTRA') {
        stateName = 'Maharashtra';
        // Women earning <= ₹25,000 are exempt in Maharashtra w.e.f. 2023 amendment
        if (cleanGender === 'FEMALE' && gross <= 25000) {
            ptInr = 0;
        } else if (gross <= 7500) {
            ptInr = 0;
        } else if (gross <= 10000) {
            ptInr = 175;
        } else {
            // ₹200 for 11 months, ₹300 in February
            ptInr = (month === 2) ? 300 : 200;
        }
    }
    // Karnataka (29 / KA)
    else if (cleanState === '29' || cleanState === 'KA' || cleanState === 'KARNATAKA') {
        stateName = 'Karnataka';
        // Gross > ₹25,000 pays ₹200/month (w.e.f. 2023 amendment)
        ptInr = gross > 25000 ? 200 : 0;
    }
    // West Bengal (19 / WB)
    else if (cleanState === '19' || cleanState === 'WB' || cleanState === 'WEST BENGAL') {
        stateName = 'West Bengal';
        if (gross <= 10000) ptInr = 0;
        else if (gross <= 15000) ptInr = 110;
        else if (gross <= 20000) ptInr = 130;
        else if (gross <= 40000) ptInr = 150;
        else ptInr = 200;
    }
    // Gujarat (24 / GJ)
    else if (cleanState === '24' || cleanState === 'GJ' || cleanState === 'GUJARAT') {
        stateName = 'Gujarat';
        if (gross <= 12000) ptInr = 0;
        else ptInr = 200;
    }
    // Telangana (36 / TS) & Andhra Pradesh (37 / AP)
    else if (cleanState === '36' || cleanState === 'TS' || cleanState === '37' || cleanState === 'AP') {
        stateName = 'Telangana / Andhra Pradesh';
        if (gross <= 15000) ptInr = 0;
        else if (gross <= 20000) ptInr = 150;
        else ptInr = 200;
    }
    // Delhi (07), Haryana (06), UP (09) do not levy employee PT
    else {
        stateName = 'Exempt / No PT Jurisdiction';
        ptInr = 0;
    }

    return {
        status: 'SUCCESS',
        stateName,
        stateCode: cleanState,
        monthlyGrossSalaryInr: roundToPaisa(gross),
        salaryMonth: month,
        professionalTaxInr: ptInr,
        isAuthorizedToDisburse: false
    };
}

/**
 * Composite Line-Item Evaluator for an Individual Employee
 */
export function evaluateEmployeeSalaryPackage({
    employeeId,
    employeeName,
    pan,
    bankAccountNumber,
    bankIfscCode,
    monthlyGrossSalaryInr,
    monthlyBasicAndDaInr = null,
    annualGrossSalaryInr = null,
    taxRegime = TAX_REGIMES.NEW,
    form12BBClaims = {},
    epfOnActualWages = false,
    employeeAgeYears = 30,
    stateCode = '27',
    salaryMonth = 1,
    gender = 'MALE',
    otherDeductionsInr = 0,
    transactionDate = null
} = {}) {
    const grossMonthly = Math.max(0, Number(monthlyGrossSalaryInr) || 0);
    const basicMonthly = monthlyBasicAndDaInr !== null && monthlyBasicAndDaInr !== undefined
        ? Math.max(0, Number(monthlyBasicAndDaInr) || 0)
        : roundToPaisa(grossMonthly * 0.50); // Default standard: Basic is 50% of gross
    const grossAnnual = annualGrossSalaryInr !== null && annualGrossSalaryInr !== undefined
        ? Math.max(0, Number(annualGrossSalaryInr) || 0)
        : roundToPaisa(grossMonthly * 12);

    // 1. Salary TDS
    const tdsResult = evaluateSalaryTds({
        annualGrossSalaryInr: grossAnnual,
        taxRegime,
        form12BBClaims,
        transactionDate
    });

    // 2. EPF
    const epfResult = evaluateEpfContribution({
        monthlyBasicAndDaInr: basicMonthly,
        epfOnActualWages,
        employeeAgeYears
    });

    // 3. ESI
    const esiResult = evaluateEsiContribution({
        monthlyGrossSalaryInr: grossMonthly
    });

    // 4. Professional Tax
    const ptResult = evaluateProfessionalTax({
        monthlyGrossSalaryInr: grossMonthly,
        stateCode,
        salaryMonth,
        gender
    });

    const otherDeductions = Math.max(0, Number(otherDeductionsInr) || 0);

    const totalEmployeeDeductionsInr = roundToPaisa(
        tdsResult.monthlyTdsDeductionInr +
        epfResult.employeeContributionInr +
        esiResult.employeeContributionInr +
        ptResult.professionalTaxInr +
        otherDeductions
    );

    const netSalaryInr = roundToPaisa(Math.max(0, grossMonthly - totalEmployeeDeductionsInr));

    const totalEmployerContributionsInr = roundToPaisa(
        epfResult.totalEmployerContributionInr +
        esiResult.employerContributionInr
    );

    const totalCostToCompanyInr = roundToPaisa(grossMonthly + totalEmployerContributionsInr);

    return {
        status: 'SUCCESS',
        employeeId: String(employeeId || '').trim(),
        employeeName: String(employeeName || '').trim(),
        pan: String(pan || '').toUpperCase().trim(),
        bankAccountNumber: String(bankAccountNumber || '').trim(),
        bankIfscCode: String(bankIfscCode || '').toUpperCase().trim(),
        monthlyGrossSalaryInr: grossMonthly,
        monthlyBasicAndDaInr: basicMonthly,
        annualGrossSalaryInr: grossAnnual,
        withholdings: {
            salaryTdsMonthlyInr: tdsResult.monthlyTdsDeductionInr,
            epfEmployeeInr: epfResult.employeeContributionInr,
            esiEmployeeInr: esiResult.employeeContributionInr,
            professionalTaxInr: ptResult.professionalTaxInr,
            otherDeductionsInr: otherDeductions,
            totalEmployeeDeductionsInr
        },
        employerContributions: {
            epfEmployerTotalInr: epfResult.totalEmployerContributionInr,
            epfEmployerShareInr: epfResult.employerEpfContributionInr,
            epsEmployerShareInr: epfResult.employerEpsContributionInr,
            esiEmployerInr: esiResult.employerContributionInr,
            totalEmployerContributionsInr
        },
        netSalaryInr,
        totalCostToCompanyInr,
        statutoryDetails: {
            tds: tdsResult,
            epf: epfResult,
            esi: esiResult,
            pt: ptResult
        },
        isAuthorizedToDisburse: false
    };
}
