#!/usr/bin/env python3
"""Compare county collections with published county visitor spending (TDTD, 2024 and 2025).

Computes levels association (Pearson, Spearman, log-log OLS), the 2024-2025 growth correlation,
spending per tax dollar, and county fixed effects against 2025 spending.
Inputs: data/panel.parquet, data_sources/tdtd_county_visitor_spending.csv, results/county_effects.csv.
Outputs: results/validation_merged.csv, validation_summary.json, validation_regression.csv, validation_growth.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def assoc(x, y):
    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": float(stats.spearmanr(x, y).statistic),
        "n": int(len(x)),
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    annual = (panel.groupby(["county_name", "year"], as_index=False)
              .agg(collections=("local_sales_tax", "sum")))
    tdtd = pd.read_csv(ROOT / "data_sources" / "tdtd_county_visitor_spending.csv")
    # join on FIPS, matching names case-insensitively (Mcnairy vs McNairy)
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from tn_fips import TN_FIPS
    fips_of = {name.lower(): fips for name, fips in TN_FIPS.items()}
    annual["fips"] = annual["county_name"].str.lower().map(fips_of)
    tdtd["fips"] = tdtd["county_name"].str.lower().map(fips_of)
    assert annual["fips"].notna().all() and tdtd["fips"].notna().all()
    m = tdtd.merge(annual.drop(columns=["county_name"]),
                   on=["fips", "year"], how="inner")
    assert len(m) == 190, f"expected 190 merged rows, got {len(m)}"
    assert m.groupby("year")["county_name"].nunique().eq(95).all()
    m["log_collections"] = np.log(m["collections"])
    m["log_spending"] = np.log(m["visitor_spending_usd"])
    m["spending_per_tax_dollar"] = m["visitor_spending_usd"] / m["collections"]
    m.to_csv(RES / "validation_merged.csv", index=False)

    out = {"source": "TDTD (2026) 2025 Economic Impact of Travel on "
                     "Tennessee (Tourism Economics), 95 counties x 2024-2025"}

    # A. levels association
    for year in (2024, 2025):
        s = m[m["year"] == year]
        out[f"levels_{year}"] = assoc(s["log_collections"], s["log_spending"])
    out["levels_pooled"] = assoc(m["log_collections"], m["log_spending"])

    X = sm.add_constant(m["log_collections"])
    pooled = sm.OLS(m["log_spending"], X).fit(
        cov_type="cluster", cov_kwds={"groups": m["county_name"]})
    rows = []
    for year in (2024, 2025):
        s = m[m["year"] == year]
        fit = sm.OLS(s["log_spending"],
                     sm.add_constant(s["log_collections"])).fit()
        rows.append({"sample": str(year), "n": int(len(s)),
                     "intercept": float(fit.params.iloc[0]),
                     "elasticity": float(fit.params.iloc[1]),
                     "se": float(fit.bse.iloc[1]),
                     "r_squared": float(fit.rsquared)})
    rows.append({"sample": "pooled (clustered SE)", "n": int(len(m)),
                 "intercept": float(pooled.params.iloc[0]),
                 "elasticity": float(pooled.params.iloc[1]),
                 "se": float(pooled.bse.iloc[1]),
                 "r_squared": float(pooled.rsquared)})
    pd.DataFrame(rows).to_csv(RES / "validation_regression.csv", index=False)
    out["log_log_regression"] = rows
    out["old_n3_bridge_for_comparison"] = {
        "elasticity": 0.9728, "r_squared": 0.8677, "n": 3,
        "note": "Production model's three-point scale check (Knox 2025, "
                "Davidson 2024, TN statewide 2023). Superseded by the n=190 "
                "estimates above; retained for comparison."}

    # B. growth co-movement
    wide = m.pivot(index="county_name", columns="year",
                   values=["collections", "visitor_spending_usd"])
    g = pd.DataFrame({
        "collections_growth_pct":
            (wide[("collections", 2025)] / wide[("collections", 2024)] - 1) * 100,
        "spending_growth_pct":
            (wide[("visitor_spending_usd", 2025)]
             / wide[("visitor_spending_usd", 2024)] - 1) * 100,
    }).dropna()
    out["growth_2024_2025"] = {
        **assoc(g["collections_growth_pct"], g["spending_growth_pct"]),
        "pearson_p": float(stats.pearsonr(g["collections_growth_pct"],
                                          g["spending_growth_pct"]).pvalue),
        "spearman_p": float(stats.spearmanr(g["collections_growth_pct"],
                                            g["spending_growth_pct"]).pvalue),
    }
    g.to_csv(RES / "validation_growth.csv")

    # C. spending per tax dollar
    out["spending_per_tax_dollar"] = {
        str(year): {
            "median": float(s["spending_per_tax_dollar"].median()),
            "q1": float(s["spending_per_tax_dollar"].quantile(0.25)),
            "q3": float(s["spending_per_tax_dollar"].quantile(0.75)),
        } for year, s in m.groupby("year")}

    # D. county effects vs 2025 visitor spending
    ce = pd.read_csv(RES / "county_effects.csv", dtype={"fips": str})
    s25 = m[m["year"] == 2025].merge(
        ce[["fips", "tourism_intensity_index"]], on="fips")
    assert len(s25) == 95
    out["county_fe_vs_spending_2025"] = assoc(
        np.log(s25["tourism_intensity_index"]), s25["log_spending"])

    with open(RES / "validation_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
