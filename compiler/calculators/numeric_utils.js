/**
 * DETERMINISTIC NUMERIC & IDENTIFIER UTILITIES
 * Zero floating-point drift integer paise arithmetic, GSTIN/PAN validators and errors.
 */

/**
 * Standard Paisa Precision Rounding (2 decimal places) using integer math.
 * Eliminates floating point drift (e.g. 0.00000000001).
 * Rejects negative zero (-0 normalized to 0).
 */
export function roundToPaisa(amount) {
    const n = Number(amount);
    if (!Number.isFinite(n) || isNaN(n) || Object.is(n, -0)) return 0;
    return Math.round(n * 100) / 100;
}

/**
 * Section 288B Rounding (Rounding off of tax):
 * The amount of tax under the Income-tax Act is rounded off to the nearest rupee,
 * and if the fraction is 50 paise or more, it is rounded up; otherwise ignored.
 */
export function roundSection288B(amount) {
    const n = Number(amount);
    if (!Number.isFinite(n) || isNaN(n) || Object.is(n, -0)) return 0;
    return Math.round(n);
}

/**
 * Standard Indian State & Union Territory Codes (Census 2011 / GSTN 2-digit prefix).
 */
export const INDIAN_STATE_CODES = {
    '01': 'Jammu & Kashmir',
    '02': 'Himachal Pradesh',
    '03': 'Punjab',
    '04': 'Chandigarh',
    '05': 'Uttarakhand',
    '06': 'Haryana',
    '07': 'Delhi',
    '08': 'Rajasthan',
    '09': 'Uttar Pradesh',
    '10': 'Bihar',
    '11': 'Sikkim',
    '12': 'Arunachal Pradesh',
    '13': 'Nagaland',
    '14': 'Manipur',
    '15': 'Mizoram',
    '16': 'Tripura',
    '17': 'Meghalaya',
    '18': 'Assam',
    '19': 'West Bengal',
    '20': 'Jharkhand',
    '21': 'Odisha',
    '22': 'Chhattisgarh',
    '23': 'Madhya Pradesh',
    '24': 'Gujarat',
    '26': 'Dadra & Nagar Haveli and Daman & Diu',
    '27': 'Maharashtra',
    '29': 'Karnataka',
    '30': 'Goa',
    '31': 'Lakshadweep',
    '32': 'Kerala',
    '33': 'Tamil Nadu',
    '34': 'Puducherry',
    '35': 'Andaman & Nicobar Islands',
    '36': 'Telangana',
    '37': 'Andhra Pradesh',
    '38': 'Ladakh',
    '97': 'Other Territory',
    '99': 'Centre Jurisdiction'
};

/**
 * Statutory 4th character legal status classification for Indian Permanent Account Numbers (PAN).
 * Enacted under Section 139A of the Income-tax Act, 1961 read with Rule 114 of Income Tax Rules.
 */
export const PAN_ENTITY_TYPES = {
    'P': { code: 'P', name: 'Individual', isIndividualOrHuf: true, standard194cRatePercent: 1.0 },
    'H': { code: 'H', name: 'Hindu Undivided Family (HUF)', isIndividualOrHuf: true, standard194cRatePercent: 1.0 },
    'C': { code: 'C', name: 'Company', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'F': { code: 'F', name: 'Partnership Firm / LLP', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'T': { code: 'T', name: 'Trust', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'A': { code: 'A', name: 'Association of Persons (AOP)', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'B': { code: 'B', name: 'Body of Individuals (BOI)', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'L': { code: 'L', name: 'Local Authority', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'J': { code: 'J', name: 'Artificial Juridical Person', isIndividualOrHuf: false, standard194cRatePercent: 2.0 },
    'G': { code: 'G', name: 'Government Agency', isIndividualOrHuf: false, standard194cRatePercent: 2.0 }
};

export const BASE36_ALPHANUMERIC = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';

/**
 * Validates the check digit of a 15-character Goods and Services Tax Identification Number (GSTIN)
 * using the official ISO/IEC 7064 Mod 36, 36 check digit algorithm.
 * 
 * Offline, zero-dependency, and deterministic. Requires zero network calls.
 * 
 * @param {string} gstin 15-character alphanumeric GSTIN string
 * @returns {boolean} True if the 15th character exactly matches the calculated check character
 */
export function isValidGstinChecksum(gstin) {
    if (!gstin || typeof gstin !== 'string') return false;
    const clean = gstin.trim().toUpperCase();

    // Structural syntax: State (2 digits) + PAN (10 chars) + Entity (1 char) + Z (1 char) + Check (1 char)
    const pattern = /^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]$/;
    if (!pattern.test(clean)) return false;

    let total = 0;
    for (let i = 0; i < 14; i++) {
        const val = BASE36_ALPHANUMERIC.indexOf(clean[i]);
        if (val === -1) return false;
        const factor = (i % 2 === 0) ? 1 : 2;
        const product = val * factor;
        total += Math.floor(product / 36) + (product % 36);
    }

    const expectedIndex = (36 - (total % 36)) % 36;
    const expectedChar = BASE36_ALPHANUMERIC[expectedIndex];
    return clean[14] === expectedChar;
}

/**
 * Deterministically decodes an Indian Tax Identifier (PAN or GSTIN).
 * 
 * Capabilities:
 * - Structural regular expression pattern validation
 * - Offline ISO/IEC 7064 Mod 36, 36 checksum calculation for GSTIN
 * - State code and name extraction from GSTIN prefix
 * - Embedded 10-character PAN extraction from GSTIN positions 3..12
 * - Statutory legal entity classification via 4th character of PAN (Individual, HUF, Company, Firm, etc.)
 * - Statutory Section 194C withholding rate derivation (1% for Individual/HUF vs 2% for Other Entities)
 * 
 * Zero dependencies, 0-neuron, zero network calls.
 * 
 * @param {string} rawId PAN (10 characters) or GSTIN (15 characters)
 * @returns {Object} Decoded tax entity metadata
 */
export function decodeTaxIdentifier(rawId) {
    if (!rawId || typeof rawId !== 'string') {
        return {
            isValid: false,
            error: 'Tax identifier must be a non-empty string.'
        };
    }

    const clean = rawId.trim().toUpperCase();

    // 1. Process 15-character GSTIN
    if (clean.length === 15) {
        const pattern = /^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]$/;
        if (!pattern.test(clean)) {
            return {
                isValid: false,
                type: 'GSTIN',
                gstin: clean,
                error: 'Malformed GSTIN format: expected 2 digits state + 10 chars PAN + 1 entity char + Z + 1 check char.'
            };
        }

        const isChecksumValid = isValidGstinChecksum(clean);
        const stateCode = clean.slice(0, 2);
        const pan = clean.slice(2, 12);
        const entityCode = pan[3];
        const entityMeta = PAN_ENTITY_TYPES[entityCode] || {
            code: entityCode,
            name: 'Other Entity / Non-Individual',
            isIndividualOrHuf: false,
            standard194cRatePercent: 2.0
        };

        return {
            isValid: isChecksumValid,
            checksumValid: isChecksumValid,
            type: 'GSTIN',
            gstin: clean,
            pan,
            stateCode,
            stateName: INDIAN_STATE_CODES[stateCode] || 'Unknown Territory',
            entityCode,
            entityType: entityMeta.name,
            isIndividualOrHuf: entityMeta.isIndividualOrHuf,
            standard194cRatePercent: entityMeta.standard194cRatePercent,
            error: isChecksumValid ? null : 'GSTIN checksum mismatch under ISO/IEC 7064 Mod 36, 36.'
        };
    }

    // 2. Process 10-character PAN
    if (clean.length === 10) {
        const panPattern = /^[A-Z]{5}\d{4}[A-Z]$/;
        if (!panPattern.test(clean)) {
            return {
                isValid: false,
                type: 'PAN',
                pan: clean,
                error: 'Malformed PAN format: expected 5 letters + 4 digits + 1 letter.'
            };
        }

        const entityCode = clean[3];
        const entityMeta = PAN_ENTITY_TYPES[entityCode] || {
            code: entityCode,
            name: 'Other Entity / Non-Individual',
            isIndividualOrHuf: false,
            standard194cRatePercent: 2.0
        };

        return {
            isValid: true,
            type: 'PAN',
            pan: clean,
            entityCode,
            entityType: entityMeta.name,
            isIndividualOrHuf: entityMeta.isIndividualOrHuf,
            standard194cRatePercent: entityMeta.standard194cRatePercent,
            error: null
        };
    }

    return {
        isValid: false,
        error: `Invalid identifier length (${clean.length}): Expected 10 characters for PAN or 15 characters for GSTIN.`
    };
}

/**
 * Error thrown when a required numeric field cannot be parsed from input.
 */
export class MalformedNumericInputError extends Error {
    constructor(val, fieldName = 'amount') {
        super(`[statutory_calculator] Malformed numeric input for field "${fieldName}": "${val}". Input could not be safely resolved to a finite number.`);
        this.name = 'MalformedNumericInputError';
        this.code = 'MALFORMED_NUMERIC_INPUT';
        this.field = fieldName;
        this.rawValue = val;
    }
}

/**
 * Error thrown when a mandatory statutory calculation parameter is missing or null.
 */
export class MissingMandatoryParameterError extends Error {
    constructor(fieldName, functionName) {
        super(`[statutory_calculator] ${functionName}: Mandatory parameter "${fieldName}" is missing or undefined.`);
        this.name = 'MissingMandatoryParameterError';
        this.code = 'MISSING_MANDATORY_PARAMETER';
        this.field = fieldName;
        this.functionName = functionName;
    }
}

/**
 * Validates that a required statutory input parameter is present and resolves cleanly
 * to a finite numeric value, rejecting missing, null, or malformed input without silent coercion.
 * 
 * @param {any} val - The input value to validate
 * @param {string} fieldName - The parameter name for diagnostics
 * @param {string} functionName - The calling calculator function name
 * @param {Object} [options]
 * @param {boolean} [options.interpretCreditAsNegative=false]
 * @returns {number} Clean normalized number
 */
export function assertMandatoryNumeric(val, fieldName, functionName, options = {}) {
    if (val === undefined || val === null) {
        throw new MissingMandatoryParameterError(fieldName, functionName);
    }
    return cleanNumericAmount(val, {
        strict: true,
        fieldName,
        interpretCreditAsNegative: options.interpretCreditAsNegative || false
    });
}

/**
 * Safely normalizes numeric inputs from numbers, strings with currency symbols (₹, $, €, Rs, INR),
 * thousands separators (commas), and whitespace into valid finite numbers.
 * 
 * Accounting conventions handled:
 * - Parentheses: "(1,500.00)" or "(₹ 1,500.00)" -> -1500.00
 * - Trailing minus: "1,500.00-" -> -1500.00
 * - Leading minus with spaces: "- ₹ 1,500.00" -> -1500.00
 * - Credit suffix: "1,500.00 CR" -> magnitude 1500.00. Negated only if interpretCreditAsNegative: true.
 * - Negative zero (-0) normalized to unsigned 0.
 * 
 * @param {any} val - Input number or formatted numeric string
 * @param {Object} [options]
 * @param {boolean} [options.strict=false] - When true, throws MalformedNumericInputError on unparseable strings
 * @param {string} [options.fieldName='amount'] - Diagnostic field name for error reporting
 * @param {boolean} [options.interpretCreditAsNegative=false] - Whether "CR" suffix inverts sign
 * @returns {number} Normalized float
 */
export function cleanNumericAmount(val, options = {}) {
    const {
        strict = false,
        fieldName = 'amount',
        interpretCreditAsNegative = false,
        defaultValue = 0
    } = options;

    if (typeof val === 'number') {
        if (!Number.isFinite(val) || isNaN(val)) {
            if (strict) {
                throw new MalformedNumericInputError(val, fieldName);
            }
            return defaultValue;
        }
        return Object.is(val, -0) ? 0 : val;
    }
    if (val === null || val === undefined || typeof val === 'boolean') {
        return defaultValue;
    }
    if (typeof val === 'string') {
        const trimmed = val.trim();
        if (!trimmed) return defaultValue;

        let isNegative = false;
        let s = trimmed;

        // Financial malformation checks:
        // 1. Digits separated by whitespace (e.g. "100 200" or concatenated invoice numbers)
        // 2. Double commas or double decimal points (e.g. "1,,000" or "100..50")
        // 3. Comma appearing after a decimal point (e.g. "100.50,25")
        if (/\d\s+\d/.test(s) || /(?:,,|\.\.)/.test(s) || /\.\d*,/.test(s)) {
            if (strict) {
                throw new MalformedNumericInputError(val, fieldName);
            }
            return defaultValue;
        }

        // Strip leading currency label before parenthesis or sign: "₹ (50,000.00)" or "INR -50,000"
        s = s.replace(/^(?:₹|\$|€|INR|Rs\.?)\s*/i, '');

        // Accounting parentheses: (1,500.00) or (₹ 1,500.00)
        if (/^\(.*\)$/.test(s)) {
            isNegative = true;
            s = s.slice(1, -1).trim();
        } else if (/-\s*$/.test(s)) {
            // Trailing minus from legacy ERP dumps: "1,500.00-"
            isNegative = true;
            s = s.replace(/-\s*$/, '').trim();
        } else if (/^\s*-\s*/.test(s)) {
            // Explicit leading minus with possible whitespace: "- ₹ 1,500.00" or "-1500"
            isNegative = true;
            s = s.replace(/^\s*-\s*/, '').trim();
        } else if (/(?:\s+)?(?:cr|credit)\.?$/i.test(s)) {
            // Accounting credit suffix: "1,500.00 CR"
            // Note: CR means credit, whose signed interpretation depends on ledger orientation.
            // Only negate if caller explicitly requests interpretCreditAsNegative: true.
            if (interpretCreditAsNegative) {
                isNegative = true;
            }
            s = s.replace(/(?:\s+)?(?:cr|credit)\.?$/i, '').trim();
        } else if (/(?:\s+)?(?:dr|debit)\.?$/i.test(s)) {
            // Accounting debit suffix: "1,500.00 DR"
            s = s.replace(/(?:\s+)?(?:dr|debit)\.?$/i, '').trim();
        }

        // Strip remaining currency symbols, commas, and currency labels
        const cleaned = s.replace(/[₹$€,\s]|(?:INR|Rs\.?)/gi, '');
        const n = parseFloat(cleaned);
        
        // Check for malformed strings that had non-numeric text left over
        const isCleanNumber = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(cleaned);
        if (!isCleanNumber || !Number.isFinite(n) || isNaN(n)) {
            if (strict) {
                throw new MalformedNumericInputError(val, fieldName);
            }
            return defaultValue;
        }

        const res = isNegative ? -Math.abs(n) : n;
        return Object.is(res, -0) ? 0 : res;
    }
    return defaultValue;
}

/**
 * High-precision conversion of rupees to integer paise.
 * Avoids IEEE-754 floating point drift during threshold and summation operations.
 * 
 * Safety Bounds & Precision Policy:
 * - Bounds: Asserts absolute value does not exceed JavaScript Number.MAX_SAFE_INTEGER (53-bit integer limit).
 * - Precision: Applies standard statutory half-up rounding (Math.round) to the nearest integer paisa.
 * - Strict sub-paisa: When options.strictSubPaisa is true, rejects amounts having fractional sub-paisa (>2 decimals).
 * 
 * @param {any} amount - Rupees amount or formatted string
 * @param {string|Object} [options='amount'] - Diagnostic field name or options object
 * @param {string} [options.fieldName='amount']
 * @param {boolean} [options.strict=false]
 * @param {boolean} [options.strictSubPaisa=false]
 * @returns {number} Integer paise (e.g. 50.25 -> 5025)
 */
export function toIntegerPaise(amount, options = 'amount') {
    const opts = typeof options === 'string' ? { fieldName: options } : (options || {});
    const {
        fieldName = 'amount',
        strict = false,
        strictSubPaisa = false
    } = opts;

    const n = cleanNumericAmount(amount, { strict, fieldName });
    const rawPaise = n * 100;

    // Safe integer boundary assertion (9 quadrillion paise / ~₹90 Trillion limit)
    if (Math.abs(rawPaise) > Number.MAX_SAFE_INTEGER) {
        throw new RangeError(`[statutory_calculator] Amount exceeds JavaScript MAX_SAFE_INTEGER (53-bit limit) in field "${fieldName}".`);
    }

    const roundedPaise = Math.round(rawPaise);

    // Explicit sub-paisa fractional detection
    if (strictSubPaisa && Math.abs(rawPaise - roundedPaise) > 1e-4) {
        throw new MalformedNumericInputError(amount, fieldName);
    }

    return roundedPaise;
}

/**
 * Converts integer paise back to rupees with 2 decimal places.
 * 
 * Safety Bounds:
 * - Asserts input paise does not exceed Number.MAX_SAFE_INTEGER.
 * 
 * @param {number} paise - Integer paise
 * @returns {number} Rounded rupees
 */
export function fromIntegerPaise(paise) {
    if (!Number.isFinite(paise) || isNaN(paise)) return 0;
    if (Math.abs(paise) > Number.MAX_SAFE_INTEGER) {
        throw new RangeError('[statutory_calculator] Integer paise exceeds JavaScript MAX_SAFE_INTEGER (53-bit limit).');
    }
    const rupees = paise / 100;
    return Object.is(rupees, -0) ? 0 : Math.round(rupees * 100) / 100;
}
