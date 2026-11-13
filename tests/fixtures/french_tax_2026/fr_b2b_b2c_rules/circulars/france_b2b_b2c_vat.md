# French B2B / B2C VAT Rules - 2026

## Source
- ClearTax - "French VAT Rates 2026"
- EU VAT Directive 2006/112/CE
- Verified: April 2026

## B2B Rules (Business to Business)

### Within EU
- Reverse charge mechanism: supplier does not charge VAT
- Customer self-assesses VAT in their own country
- Supplier must have valid EU VAT number
- Customer must provide valid EU VAT number (VIES verification)
- Invoice must state "Reverse charge" or "Autoliquidation"
- No VAT amount on invoice

### Outside EU (Exports)
- VAT-free (0%)
- Export documentation required
- Proof of export (customs declaration, transport document)

## B2C Rules (Business to Consumer)

### Within EU (Distance Selling)
- EU-wide threshold: €10,000 per year
- Below threshold: charge French VAT (20% / 10% / 5.5% / 2.1%)
- Above threshold: charge customer's country VAT rate
- OSS (One Stop Shop) registration for cross-border B2C

### Outside EU
- VAT-free (0%) for exports
- Import VAT payable by customer at destination

## E-Invoicing Mandate 2026-2027

| Phase | Date | Scope |
|-------|------|-------|
| Phase 1 | September 2026 | Large enterprises (B2B e-invoicing) |
| Phase 2 | 2027 | All B2B and B2G transactions |
| Platform | PDP / PPF | Partner Dematerialisation / Public Invoicing Portal |

## Domestic B2B
- Standard French VAT rates apply
- No reverse charge within France
- Both parties must be VAT-registered

## Executable Coverage

This section is a rules summary and intentionally has no `ast.json`: the
executable domestic/cross-border pipeline, including the distinct `FRANCE`
country case, lives in `fr_vat_tva`.