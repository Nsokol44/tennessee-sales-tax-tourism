# What sales-tax receipts can (and cannot) tell us about tourism: evidence from Tennessee counties

Nicholas J. Sokol, Department of Geography & Sustainability, University of Tennessee, Knoxville (nsokol@utk.edu)

This repository holds the data and code for a study of what monthly county local-option sales-tax collections in Tennessee can and cannot say about tourism. It uses a panel of 95 counties and 80 months (January 2020 to August 2026), compares collections with published county visitor spending for 2024 and 2025, and tests for spatial dependence.

Preprint: Sokol, N. J. (2026). What sales-tax receipts can (and cannot) tell us about tourism: evidence from Tennessee's counties. SocArXiv. https://osf.io/preprints/socarxiv/q74pn_v1 (CC-BY 4.0).

## Contents

- `code/` - numbered analysis scripts, `tn_fips.py` (county name to FIPS lookup) and `run_all.py`.
- `data/` - processed inputs. `panel.parquet` is built by script 01.
- `data_sources/` - the TDTD county visitor-spending table, the Affinity county-month file, `SOURCE_NOTES.md`, `extraction_script.py` and `fetch_data.py`.
- `results/` - tables and JSON written by the scripts, including `results/extended/`, `results/review_checks/` and `results/affinity_check/`.
- `figures/` - Figures 1-10 as 300-dpi PNG and PDF.
- `run_manifest.json` - record of the run that produced the files in `results/` and `figures/`.

## Install and run

```bash
pip install -r requirements.txt
python code/run_all.py
```

The versions in `requirements.txt` are pinned; the run was done with Python 3.12. The full run takes several minutes, mostly the 9,999-permutation LISA and the 2,000-replication bootstrap. Permutation and bootstrap steps use fixed seeds (20261005; 20261006 for script 11).

`run_all.py` runs the scripts in `code/` in this order:

| Script | Purpose |
|---|---|
| `01_build_panel.py` | Build the county-month panel |
| `02_core_models.py` | Pooled OLS, two-way fixed effects, Moran's I, spatial lag and error models, 2025 holdout forecast |
| `03_lisa.py` | Local Moran's I for 2024 and 2025 |
| `04_validation.py` | Collections against published visitor spending |
| `05_figures.py` | Basic figures |
| `06_robustness_real_terms.py` | Nominal against CPI-deflated results |
| `07_review_checks.py` | Alternative standard errors, rolling-origin forecasts, state-tax comparison, permutation tests, seasonality |
| `08_figures_v2.py` | Two-panel LISA and scatter figures |
| `09_affinity_check.py` | Monthly comparison with Affinity card spending |
| `10_population_controls.py` | Models with population and employment controls |
| `11_extended_checks.py` | Visitor share of collections, rate-step screen, forecast bootstrap |
| `12_residual_map.py` | Residual map of visitor spending against collections |
| `13_figures_standard.py` | Final versions of Figures 1-10 |

Script 10 tries to download the ACS population and employment extract from Census Reporter and, if that fails, uses `data/acs_county_pop_emp.csv`.

## Data

| File | Source |
|---|---|
| `data/tndor_county_month.csv` | Tennessee Department of Revenue, monthly county local and state sales-tax collections |
| `data/fred_gasregw.csv`, `fred_umcsent.csv`, `fred_cpi.csv` | FRED: US regular gasoline price (weekly), University of Michigan consumer sentiment, CPIAUCSL |
| `data/tn_counties.geojson` | Census TIGER 2023 county boundaries |
| `data/acs_county_pop_emp.csv` | ACS 2024 5-year county population and employment, via Census Reporter |
| `data_sources/tdtd_county_visitor_spending.csv` | Tennessee Department of Tourist Development, 2025 Economic Impact of Travel on Tennessee (prepared by Tourism Economics), county direct visitor spending, 2024 and 2025 |
| `data_sources/affinity_county_monthly.csv` | Affinity Solutions card spending by county of cardholder residence, from the Opportunity Insights Economic Tracker |

`data_sources/SOURCE_NOTES.md` describes the TDTD table and how it was extracted. The 30 MB source PDF is not in the repository; `data_sources/fetch_data.py` downloads it and checks its SHA-256. The bundled CSV is enough to run the pipeline.

## Outputs

`results/` has the coefficient tables, LISA classes, validation tables and JSON summaries (for example `validation_summary.json`, `lisa_summary.json`, `diagnostics.json`, `results/extended/extended_summary.json`). `figures/` has Figures 1-10.

## Headline numbers

These come from the files in `results/`.

| Quantity | Value |
|---|---|
| Panel | 95 counties, 80 months (Jan 2020 to Aug 2026), 7,600 county-months |
| Log collections vs log visitor spending, levels | r = 0.913 pooled (n = 190); log state sales tax gives r = 0.915 |
| Log-log elasticity | 1.109 (2024), 1.126 (2025), 1.117 pooled; p against 1 = 0.085, 0.048, 0.052 |
| Growth correlation, 2024 to 2025 | r = 0.372; 0.19 with 5-95% trimming; 0.16 excluding 11 counties with a large local-state growth gap; 0.20 excluding 28 rate-step counties; 0.23 using state-tax growth |
| Visitor-generated local tax as a share of collections, 2025 | median county 11.4% (IQR 7.7% to 17.3%); 28.6% statewide; 3 counties above 40% |
| Benchmark local tax figures | 4.2% of visitor spending on average (CV 14%); log correlation with spending 0.996 |
| Moran's I, 2025 | collections 0.182 (permutation p = 0.002); visitor spending 0.169; residual 0.035 (not significant) |
| LISA after Benjamini-Hochberg FDR | 2 counties (Davidson, Cheatham), collections only |
| Population control | collections elasticity 1.36 with population controlled (2025); adding collections to a population-only model F = 32.3, p = 1.5e-7; adding population to a collections model p = 0.31 |
| Forecast MAPE, % (2023 / 2024 / 2025 holdouts) | ex-post LSDV 19.9 / 7.3 / 7.4; ex-ante LSDV 15.8 / 11.2 / 10.8; ex-ante LSDV trained 2021+ 9.0 / 6.9 / 7.1; seasonal naive times statewide growth 9.5 / 7.4 / 5.7 |
| Bootstrap difference, LSDV 2021+ minus naive times statewide growth (pp, 95% CI) | -0.5 (-1.0 to 0.0); -0.6 (-1.6 to 0.3); +1.4 (0.6 to 2.2) |
| Growth 2020 to 2025 | +56.0% nominal, +25.4% real |
| Monthly check, Affinity card spending | month-on-month growth r = 0.48 with the card series leading one month; year-over-year r between 0.02 and 0.06 after 2021 |

## License

The code in this repository is released under the MIT License (see `LICENSE`).
