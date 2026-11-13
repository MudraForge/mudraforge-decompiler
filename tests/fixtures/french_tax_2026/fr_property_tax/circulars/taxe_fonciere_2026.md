# French Property Taxes - 2026

## Source
- Service Public / DGFiP
- Verified: 2026

## Taxes

### Taxe Foncière (Land/Property Tax)
- Paid by property owners
- Based on cadastral rental value (valeur locative cadastrale)
- Rates set by local municipalities (communes), departments, and regions
- No single national rate: each commune votes its own rate (the indicative
  range is 10-50% for built property)
- Built property benefits from a 50% abatement on the cadastral rental value
- The municipal rate is a required computation input; the pipeline fails
  closed when it is missing rather than guessing a rate

### Taxe d'Habitation (Residence Tax)
- Phased out for primary residences (completed 2023)
- Still applies to secondary residences and vacant properties
- Surcharge possible in high-demand areas (taxe sur les logements vacants)

### Contribution Economique Territoriale (CET)
- Replaced taxe professionnelle
- Composed of CFE (cotisation foncière des entreprises) and CVAE (cotisation sur la valeur ajoutée des entreprises)
- CVAE being phased out

## Key Rates (indicative, vary by locality)

| Tax | Typical Range | Base |
|-----|--------------|------|
| Taxe foncière bâtie | 10-50% | 50% of cadastral value |
| Taxe foncière non bâtie | 15-80% | 80% of cadastral value |
| Taxe d'habitation (secondary) | 5-60% | Cadastral rental value |
| CFE | Variable by commune | Rental value of business premises |