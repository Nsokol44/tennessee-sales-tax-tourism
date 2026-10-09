#!/usr/bin/env python3
"""Nominal vs real (CPI-deflated) collections.

Deflates by each calendar year's mean CPIAUCSL (data/fred_cpi.csv) to 2026 dollars and recomputes the
pooled-OLS trend, statewide totals and 2020-2025 growth, and the validation growth correlation.
Inputs: data/panel.parquet, data/fred_cpi.csv, results/validation_merged.csv. Output: results/robustness_real_terms.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    cpi = pd.read_csv(ROOT / "data" / "fred_cpi.csv", parse_dates=["observation_date"])
    cpi["ym"] = cpi["observation_date"].dt.to_period("M").dt.to_timestamp()
    # CPIAUCSL has no October 2025 value, so deflate by each year's mean CPI
    cpi["year"] = cpi["ym"].dt.year
    yr_mean = cpi.groupby("year")["CPIAUCSL"].mean()
    base = float(yr_mean.loc[2026])
    panel["cpi_year_mean"] = panel["year"].map(yr_mean)
    assert panel["cpi_year_mean"].notna().all()
    panel["real_tax"] = panel["local_sales_tax"] * base / panel["cpi_year_mean"]

    out = {"cpi_base": "mean CPIAUCSL of 2026 months available (= %.1f); "
                       "year-mean step deflator; Oct 2025 CPI missing in "
                       "source (BLS shutdown gap), not interpolated" % base}

    totals = {}
    for year in (2020, 2025):
        s = panel[panel["year"] == year]
        totals[year] = {"nominal_B": float(s["local_sales_tax"].sum() / 1e9),
                        "real_B_2026avg": float(s["real_tax"].sum() / 1e9)}
    out["statewide_totals"] = totals
    out["growth_2020_2025_pct"] = {
        "nominal": float((totals[2025]["nominal_B"] / totals[2020]["nominal_B"] - 1) * 100),
        "real": float((totals[2025]["real_B_2026avg"] / totals[2020]["real_B_2026avg"] - 1) * 100),
    }

    # trend in real terms: same design as script 02, on log real collections
    d = panel[panel["local_sales_tax"] > 0].copy()
    d["l_real"] = np.log(d["real_tax"])
    d["trend"] = (d["year"] - 2020) * 12 + (d["month"] - 1)
    X = pd.get_dummies(d[["fips", "month"]].astype(str),
                       columns=["fips", "month"], drop_first=True, dtype=float)
    X["trend"] = d["trend"].values
    X["gas_price_usd_gal"] = d["gas_price_usd_gal"].astype(float).values
    X["consumer_sentiment"] = d["consumer_sentiment"].astype(float).values
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(d["l_real"], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips"].values})
    out["trend_pct_per_month"] = {
        "real": float((np.exp(fit.params["trend"]) - 1) * 100),
        "real_se": float(fit.bse["trend"]),
        "nominal_reference": 0.578,
    }

    # validation growth in real terms, using the same year-mean factor for both series
    yr_cpi = panel.groupby("year")["cpi_year_mean"].first()
    m = pd.read_csv(RES / "validation_merged.csv")
    m["collections_real"] = m["collections"] * base / m["year"].map(yr_cpi)
    m["spending_real"] = m["visitor_spending_usd"] * base / m["year"].map(yr_cpi)
    wide = m.pivot(index="county_name", columns="year",
                   values=["collections_real", "spending_real"])
    cg = (wide[("collections_real", 2025)] / wide[("collections_real", 2024)] - 1) * 100
    sg = (wide[("spending_real", 2025)] / wide[("spending_real", 2024)] - 1) * 100
    ok = cg.notna() & sg.notna()
    out["validation_growth_real"] = {
        "pearson_r": float(np.corrcoef(cg[ok], sg[ok])[0, 1]),
        "spearman_rho": float(stats.spearmanr(cg[ok], sg[ok]).statistic),
        "n": int(ok.sum()),
    }

    with open(RES / "robustness_real_terms.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
