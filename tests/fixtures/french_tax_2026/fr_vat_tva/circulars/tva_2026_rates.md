# French VAT (TVA) - Taux 2026

## Source
- ClearTax - "French VAT Rates 2026: 20%, 10%, 5.5% & 2.1%"
- Avalara - "French VAT rates and VAT compliance"
- Service Public / DILA
- Verified: April 2026

## VAT Rates 2026

| Rate Type | Rate | Goods / Services |
|-----------|------|------------------|
| Standard (Normal) | 20% | Most goods and services: clothing, alcohol, household appliances, cosmetics, cars, fuels, electronics |
| Intermediate / Reduced 1 | 10% | Prepared foods, restaurants, catering, hotel stays, cable/pay TV, some renovation work |
| Reduced 2 | 5.5% | Basic food products, books, soft drinks, medical devices, sports/cultural event tickets, energy, disability equipment |
| Super-reduced | 2.1% | Newspapers, magazines, TV license fees, certain medicines, entry to cultural programs, livestock |
| Zero | 0% | Exports outside EU, international transport |

## Cross-Border Rules

| Scenario | Rule |
|----------|------|
| Domestic France (B2B or B2C) | French VAT applies (20% / 10% / 5.5% / 2.1%); no reverse charge |
| B2B within EU (valid VAT id) | Reverse charge - customer accounts for VAT (0% invoiced) |
| B2C within EU | €10,000 OSS threshold; up to the threshold French VAT applies, above it the customer's country VAT |
| Exports outside EU | VAT-free (0%) (CGI art. 262-I) |
| Imports into EU | VAT payable at entry (deductible if VAT-registered) |

`FRANCE` is modelled as its own customer country, distinct from `EU`: a
domestic French B2B sale is not a reverse charge and must not be treated as a
non-EU export.

## E-Invoicing
France e-invoicing mandate: phased rollout September 2026 - 2027 for B2B and B2G transactions via PDP/PPf platform.

## Special Cases
- Electricity subscription: 20% from August 1, 2025 (EU harmonization)
- Corsica: reduced rates apply (different from mainland)
- Overseas departments: different VAT rates