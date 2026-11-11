/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MUDRAFORGE UNIVERSAL STATUTORY VIRTUAL MACHINE (USVM) - RATIONAL ARITHMETIC
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Implements 5-stage rational arithmetic, sub-paisa remainder preservation,
 * commercial currency quantization, and statutory assessment rounding (Section 288B/516).
 *
 * Guarantees zero floating-point drift and eliminates premature integer truncation.
 *
 * Specification: RFC-009 (Revision 2.1)
 */

/**
 * Converts any numeric or string representation of an INR rupee amount into
 * an exact integer number of paise represented as a BigInt.
 * Avoids floating-point drift by splitting on decimal point.
 */
export function toBigIntPaise(amount) {
    if (amount === undefined || amount === null) return 0n;
    if (typeof amount === 'bigint') return amount;

    if (typeof amount === 'number') {
        if (!Number.isFinite(amount) || isNaN(amount) || Object.is(amount, -0)) return 0n;
        // Format to fixed 2 decimal places to avoid standard IEEE-754 precision artifacts
        const str = amount.toFixed(2);
        return toBigIntPaise(str);
    }

    if (typeof amount === 'string') {
        const clean = amount.trim().replace(/,/g, '');
        if (!clean || clean === '0' || clean === '-0') return 0n;

        const isNegative = clean.startsWith('-');
        const absStr = isNegative ? clean.slice(1) : clean;

        const parts = absStr.split('.');
        const wholeRupeesStr = parts[0] || '0';
        let paiseStr = parts[1] || '00';

        // Pad or truncate paise to exactly 2 digits
        if (paiseStr.length === 1) paiseStr += '0';
        else if (paiseStr.length > 2) paiseStr = paiseStr.slice(0, 2);

        const wholeRupees = BigInt(wholeRupeesStr);
        const paise = BigInt(paiseStr);
        const totalPaise = (wholeRupees * 100n) + paise;

        return isNegative ? -totalPaise : totalPaise;
    }

    return 0n;
}

/**
 * Converts integer paise BigInt back to a standard floating-point rupee number
 * strictly rounded to 2 decimal places.
 */
export function fromBigIntPaise(paiseBigInt) {
    if (typeof paiseBigInt !== 'bigint') {
        paiseBigInt = toBigIntPaise(paiseBigInt);
    }
    const isNegative = paiseBigInt < 0n;
    const absPaise = isNegative ? -paiseBigInt : paiseBigInt;

    const wholeRupees = absPaise / 100n;
    const remPaise = absPaise % 100n;

    const num = Number(wholeRupees) + (Number(remPaise) / 100);
    return isNegative ? -num : num;
}

/**
 * Stage 1 & 2: Exact Rational Multiplication & Remainder Extraction.
 * Multiplies base in paise by rate in basis points (1 bps = 0.01% = 1/10000).
 * Preserves the exact rational sub-paisa remainder without truncation.
 */
export function multiplyRational(basePaise, rateBps, denominator = 10000n) {
    const base = typeof basePaise === 'bigint' ? basePaise : toBigIntPaise(basePaise);
    const rate = typeof rateBps === 'bigint' ? rateBps : BigInt(Math.round(Number(rateBps) || 0));
    const denom = typeof denominator === 'bigint' ? denominator : BigInt(denominator);

    const numerator = base * rate;
    const quotient = numerator / denom;
    const remainder = numerator % denom;

    return {
        numerator,
        denominator: denom,
        quotientPaise: quotient,
        subPaisaRemainder: remainder
    };
}

/**
 * Stage 3: Currency Quantization (Commercial Paisa Rounding).
 * Converts rational calculation into exact integer commercial paise using
 * standard half-up rounding (>= 0.5 paise rounds up to next paisa).
 */
export function quantizeCurrencyPaisa(quotientPaise, subPaisaRemainder, denominator = 10000n) {
    let q = quotientPaise;
    const isNegative = q < 0n || subPaisaRemainder < 0n;
    const absRem = subPaisaRemainder < 0n ? -subPaisaRemainder : subPaisaRemainder;

    // Half-up: if remainder * 2 >= denominator, add 1 paisa
    if ((absRem * 2n) >= denominator) {
        q = isNegative ? q - 1n : q + 1n;
    }

    return q;
}

/**
 * Stage 4: Statutory Assessment Rounding.
 * Applies statute-specific rounding rules:
 * - INCOME_TAX_ACT_1961_SEC288B / INCOME_TAX_ACT_2025_SEC516:
 *   Ignores fractional paise, then rounds whole rupees to nearest multiple of 10.
 *   (e.g., ₹145.40 -> ignore paise -> ₹145 -> round to nearest 10 -> ₹150)
 *   (e.g., ₹144.90 -> ignore paise -> ₹144 -> round to nearest 10 -> ₹140)
 *
 * For standard transaction withholding line items, COMMERCIAL_PAISA is retained.
 */
export function roundStatutoryAssessment(exactPaiseBigInt, policy = 'COMMERCIAL_PAISA') {
    const paise = typeof exactPaiseBigInt === 'bigint' ? exactPaiseBigInt : toBigIntPaise(exactPaiseBigInt);

    if (policy === 'COMMERCIAL_PAISA' || policy === 'NONE') {
        return paise;
    }

    if (policy === 'INCOME_TAX_ACT_1961_SEC288B' || policy === 'INCOME_TAX_ACT_2025_SEC516') {
        const isNegative = paise < 0n;
        const absPaise = isNegative ? -paise : paise;

        // Section 288B & 516 mandate: Ignore paise first
        const wholeRupees = absPaise / 100n;
        const rupeeRemainder = wholeRupees % 10n;

        let roundedRupees = wholeRupees - rupeeRemainder;
        if (rupeeRemainder >= 5n) {
            roundedRupees += 10n;
        }

        const finalPaise = roundedRupees * 100n;
        return isNegative ? -finalPaise : finalPaise;
    }

    return paise;
}

/**
 * Stage 5: Formats a BigInt paise amount into a validated decimal string representation ("50000.00").
 * Used at JSON disk, database, and network serialization boundaries.
 */
export function formatPaiseToDecimalString(paiseBigInt) {
    const paise = typeof paiseBigInt === 'bigint' ? paiseBigInt : toBigIntPaise(paiseBigInt);
    const isNegative = paise < 0n;
    const absPaise = isNegative ? -paise : paise;

    const wholeRupees = absPaise / 100n;
    const remPaise = absPaise % 100n;

    const remStr = remPaise < 10n ? `0${remPaise}` : `${remPaise}`;
    const sign = isNegative ? '-' : '';
    return `${sign}${wholeRupees}.${remStr}`;
}
