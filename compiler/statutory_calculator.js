/**
 * DETERMINISTIC STATUTORY & FINANCIAL CALCULATOR BARREL
 * Re-exports modular calculators for numeric utils, MSME, Direct Tax, GST,
 * International Tax, and Enterprise Global Accounting.
 * Preserves 100% backward compatibility across all existing imports.
 */

export * from './calculators/numeric_utils.js';
export * from './calculators/msme.js';
export * from './calculators/direct_tax.js';
export * from './calculators/gst.js';
export * from './calculators/international.js';
export * from './calculators/payroll_statutory.js';

// Proprietary Enterprise Global Accounting Calculators
export {
    calculateRevenueAllocationIndAs115,
    calculateLeaseAmortizationIndAs116,
    calculateExpectedCreditLossIndAs109,
    calculateEsopAmortizationIndAs102,
    calculateDeferredTaxIndAs12,
    evaluateProvisionsAndContingenciesIndAs37,
    calculateForeignCurrencyIndAs21,
    calculateImpairmentIndAs36,
    calculateBusinessCombinationsIndAs103
} from '../enterprise/global_accounting_calculators.js';
