"""
Canonical fact bindings shared by the Catala emitter and the USVM bridge.

A fact binding maps a canonical USVM fact key to the pair
``(catala_field_name, catala_type)`` used when the same executable AST is
lowered to Catala. Keeping this map in one module is deliberate: the Catala
emitter and the USVM input adapter must agree on a fact's semantic type, or the
two engines will silently disagree about units.

``integer`` bindings are *scalar* values (counts and basis-point rates), never
money. The USVM convention for scalars is subtle:

* a JavaScript ``number`` is interpreted as major units and multiplied by 100
  whenever an arithmetic operator coerces it to paise;
* a bare digit string is interpreted as an already-minor-unit integer.

A count used as a divisor (``quotientFamilialParts``) must therefore reach the
VM as a digit string, otherwise ``DIV`` divides by ``parts * 100``. The bridge
uses :data:`INTEGER_FACT_KEYS` to perform that encoding; the emitter uses the
same types to declare Catala inputs, so a fact can never be a scalar in one
engine and money in the other.
"""

from typing import Final


# Canonical runtime fact bindings injected by evaluation_vm.js for pipelines.
# Each maps a USVM fact path to (catala_field, catala_type). Types are Catala
# built-ins (money/integer/decimal/boolean) or "enum:<EnumName>".
FACT_BINDINGS: Final[dict[str, tuple[str, str]]] = {
    # INR / paise (canonical USVM facts).
    "currentInvoicePaisa": ("current_invoice", "money"),
    "invoiceAmountPaisa": ("current_invoice", "money"),
    "primaryBasePaisa": ("primary_base", "money"),
    "priorPurchasesPaisa": ("prior_purchases", "money"),
    "invoiceAmountInr": ("current_invoice", "money"),
    # EUR / cents (French tax 2026 fixtures).
    "transactionAmountEur": ("transaction_amount", "money"),
    "euB2cSalesEur": ("eu_b2c_sales", "money"),
    "customerType": ("customer_type", "enum:CustomerType"),
    "customerCountry": ("customer_country", "enum:CustomerCountry"),
    "customerCountryVatRateBps": ("customer_country_vat_rate_bps", "integer"),
    "hasValidVatId": ("has_valid_vat_id", "boolean"),
    "goodsCategory": ("goods_category", "enum:GoodsCategory"),
    "grossSalaryEur": ("gross_salary", "money"),
    "rgduReliefEur": ("rgdu_relief", "money"),
    "isSmeEligible": ("is_sme_eligible", "boolean"),
    "quotientFamilialParts": ("quotient_familial_parts", "integer"),
    "taxableIncomeEur": ("taxable_income", "money"),
    "taxableProfitEur": ("taxable_profit", "money"),
    "cadastralValueEur": ("cadastral_value", "money"),
    "municipalRateBps": ("municipal_rate_bps", "integer"),
}


#: Facts whose semantic type is a scalar integer rather than money.
INTEGER_FACT_KEYS: Final[frozenset[str]] = frozenset(
    fact for fact, (_, typ) in FACT_BINDINGS.items() if typ == "integer"
)
