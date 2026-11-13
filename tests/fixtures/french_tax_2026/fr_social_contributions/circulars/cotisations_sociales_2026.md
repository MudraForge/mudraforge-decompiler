# French Social Contributions (Cotisations Sociales) - Taux 2026

## Source
- Intelligent Payroll - "France payroll 2026: key facts and law changes"
- Urssaf - "Taux de cotisations - Secteur privé" (updated 1 January 2026)
- Service Public - "Sécurité sociale : plafond 2026"
- Verified: 4 October 2026

## Social Security Ceiling (PMSS/PASS) 2026

| Period | Amount |
|--------|--------|
| Monthly | €4,005 |
| Annual | €48,060 |
| Quarterly | €12,015 |
| Weekly | €924 |
| Daily | €220 |
| Hourly | €30 |

Separate limit for unemployment, AGS, CSG, CRDS: €192,240

## Employer Contributions (Secteur Privé)

| Contribution | Rate | Base |
|-------------|------|------|
| Health/Maternity/Disability/Death | 13% | Total payroll; RGDU relief applies up to 3x SMIC (see below) |
| Old-age | 2.11% + 8.55% (within ceiling) | Total payroll / within ceiling (capped at PMSS: `MIN(gross, €48,060)`) |
| Family allowances | 5.25% | Total payroll; RGDU relief applies up to 3x SMIC (see below) |
| Unemployment | 4.00% | Within €192,240 |
| AGS (wage guarantee) | 0.25% | Within €192,240 |
| CSA | 0.30% | Total payroll |
| Fnal | 0.10% (<50 employees) / 0.50% (50+ employees) | Total payroll |

Not covered: accident at work (Carsat notified), versement mobilité, Fnal 0.50%, 0.016% social dialogue, training levies.

## Employee Contributions (Secteur Privé)

| Contribution | Rate | Base |
|-------------|------|------|
| Old-age | 0.40% + 6.90% (within ceiling) | Total payroll / within ceiling (capped at PMSS) |
| CSG imposable | 2.40% | 98.25% of gross up to €192,240, then 100% above |
| CSG non-imposable | 6.80% | 98.25% of gross up to €192,240, then 100% above |
| CRDS | 0.50% | 98.25% of gross up to €192,240, then 100% above |
| Extra health contribution | 1.30% | Haut-Rhin, Bas-Rhin, Moselle only |

No employee unemployment contribution.

The 1.75% expense deduction (giving the 98.25% base) is capped at 4 x PASS
(€192,240). Above that ceiling, CSG/CRDS apply to the full gross excess:
`MIN(gross, €192,240) x 98.25% + MAX(0, gross - €192,240)`.
Unemployment (4.00%) and AGS (0.25%) use the same 4 x PASS ceiling, so their
base is `MIN(gross, €192,240)`.

## 2026 LFSS Reform: RGDU

Effective 1 January 2026, the 2026 LFSS abolished the reduced 7% health rate
and the reduced 3.45% family rate. Employer health and family contributions are
therefore due at the full 13% and 5.25% rates across all salaries. In exchange,
a single degressive relief ("RGDU") applies to employer contributions up to
3 x SMIC (approximately €65,600 at the 2026 SMIC).

This fixture models the RGDU amount as an explicit caller-supplied input
(`rgdu_relief`), because the statutory coefficient depends on annual SMIC and
employer-specific parameters. The pipeline applies the full rates and then
subtracts the supplied relief; it never hardcodes a reduced rate.

## SMIC (Minimum Wage)

| Effective Date | Hourly | Monthly (35h) |
|----------------|--------|---------------|
| 1 Jan 2026 | €12.02 | €1,823.03 |
| 1 Jun 2026 | €12.31 | €1,867.02 |

Revalued 2.41% on 1 June 2026 (arrêté of 22 May 2026)