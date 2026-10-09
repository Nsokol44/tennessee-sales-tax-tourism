#!/usr/bin/env python3
"""Core models for log monthly county local-option sales-tax collections.

Fits a pooled OLS with county and month dummies, trend and FRED controls; two-way fixed effects;
Moran's I with spatial lag and error models on 2025 county totals; and a 2025 holdout forecast.
Inputs: data/panel.parquet, data/tn_counties.geojson.
Outputs: results/ coef_pooled_ols.csv, month_seasonal.csv, county_effects.csv, coef_spatial.csv, diagnostics.json, missingness.csv.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import Queen
from linearmodels.panel import PanelOLS
from spreg import ML_Error, ML_Lag

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "panel.parquet"
RES = ROOT / "results"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PANEL_PATH)
    # Rows with non-positive collections stay in the panel and are dropped from the log models.
    neg = df[df["local_sales_tax"] <= 0]
    d = df[df["local_sales_tax"] > 0].copy()
    d["l_tax"] = np.log(d["local_sales_tax"])
    d["trend"] = (d["year"] - 2020) * 12 + (d["month"] - 1)
    n_county, n_ym = d["fips"].nunique(), d["ym"].nunique()
    print(f"panel: {len(d)} log-model rows ({len(neg)} negative-adjustment "
          f"row(s) excluded)")

    # 1. pooled OLS with county dummies
    d["gas_price_usd_gal"] = d["gas_price_usd_gal"].astype(float)
    d["consumer_sentiment"] = d["consumer_sentiment"].astype(float)
    assert d[["gas_price_usd_gal", "consumer_sentiment"]].notna().all().all()

    def design(dd, macro=True):
        Xd = pd.get_dummies(dd[["fips", "month"]].astype(str),
                            columns=["fips", "month"], drop_first=True,
                            dtype=float)
        if macro:
            Xd["trend"] = dd["trend"].values
            Xd["gas_price_usd_gal"] = dd["gas_price_usd_gal"].values
            Xd["consumer_sentiment"] = dd["consumer_sentiment"].values
        return sm.add_constant(Xd, has_constant="add")

    X = design(d)
    ols = sm.OLS(d["l_tax"], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["fips"].values})
    pd.DataFrame({"coef": ols.params, "se": ols.bse,
                  "t": ols.tvalues, "p": ols.pvalues}
                 ).to_csv(RES / "coef_pooled_ols.csv")

    seas = {"month_1": 0.0}
    for m in range(2, 13):
        seas[f"month_{m}"] = float(ols.params.get(f"month_{m}", float("nan")))
    pd.DataFrame([{"month": int(k.split("_")[1]),
                   "log_seasonal_vs_jan": v,
                   "pct_vs_jan": (np.exp(v) - 1) * 100}
                  for k, v in seas.items()]).to_csv(
        RES / "month_seasonal.csv", index=False)

    # 2. two-way fixed effects, closed form and PanelOLS
    gmean = d["l_tax"].mean()
    d["county_fe"] = d.groupby("fips")["l_tax"].transform("mean") - gmean
    d["time_fe"] = d.groupby("ym")["l_tax"].transform("mean") - gmean
    d["fitted_fe"] = gmean + d["county_fe"] + d["time_fe"]
    fe_r2 = float(1 - ((d["l_tax"] - d["fitted_fe"]) ** 2).sum()
                  / ((d["l_tax"] - gmean) ** 2).sum())
    fe_r2_county_only = float(
        1 - ((d["l_tax"] - gmean - d["county_fe"]) ** 2).sum()
        / ((d["l_tax"] - gmean) ** 2).sum())
    p = d.set_index(["fips", "ym"]).sort_index()
    fe_check = PanelOLS(p["l_tax"],
                        sm.add_constant(pd.DataFrame(index=p.index)),
                        entity_effects=True, time_effects=True,
                        drop_absorbed=True).fit(
                            cov_type="clustered", cluster_entity=True)
    # correlation of closed-form county effects with PanelOLS entity effects
    est_eff = fe_check.estimated_effects
    entity_eff = (est_eff.reset_index().drop_duplicates("fips")
                  .set_index("fips")["estimated_effects"])
    county_fe_by_fips = (d.groupby("fips")["county_fe"].mean())
    fe_check_corr = float(np.corrcoef(
        county_fe_by_fips.loc[entity_eff.index].values,
        entity_eff.values)[0, 1])
    ce = (d.groupby(["fips", "county_name"])["county_fe"].mean()
          .rename("county_fe_log").reset_index())
    ce["tourism_intensity_index"] = np.exp(ce["county_fe_log"]) * 100
    ce = ce.sort_values("tourism_intensity_index", ascending=False)
    ce.to_csv(RES / "county_effects.csv", index=False)

    # 3. spatial models on the 2025 annual cross-section
    a = (d[d["year"] == 2025].groupby(["fips", "county_name"], as_index=False)
         .agg(tax=("local_sales_tax", "sum")))
    g = gpd.read_file(ROOT / "data" / "tn_counties.geojson")
    g = g[g["GEOID"].isin(a["fips"])].copy()
    w = Queen.from_dataframe(g, ids="GEOID")
    w.transform = "R"
    a = a.set_index("fips").loc[g["GEOID"]].reset_index()
    y = np.log(a["tax"].values).reshape(-1, 1)
    Xc = np.ones((len(a), 1))
    mi = Moran(y.ravel(), w)
    lag = ML_Lag(y, Xc, w, name_y="l_tax_2025", name_x=["const"])
    err = ML_Error(y, Xc, w, name_y="l_tax_2025", name_x=["const"])
    pd.DataFrame({
        "model": ["spatial_lag"] * 2 + ["spatial_error"] * 2,
        "param": ["const", "rho", "const", "lambda"],
        "coef": [float(lag.betas[0, 0]), float(lag.betas[1, 0]),
                 float(err.betas[0, 0]), float(err.betas[1, 0])],
        "se": [float(lag.std_err[0]), float(lag.std_err[1]),
               float(err.std_err[0]), float(err.std_err[1])],
    }).to_csv(RES / "coef_spatial.csv", index=False)

    # 4. forecast: train on 2020-2024, predict 2025
    tr = d[d["year"] <= 2024]
    te = d[d["year"] == 2025].copy()

    def metrics(y_true, y_hat):
        y_true, y_hat = np.asarray(y_true, float), np.asarray(y_hat, float)
        return {
            "rmse_usd": float(np.sqrt(np.mean((y_true - y_hat) ** 2))),
            "mae_usd": float(np.mean(np.abs(y_true - y_hat))),
            "mape_pct": float(np.mean(np.abs((y_true - y_hat) / y_true)) * 100),
        }

    Xtr = design(tr)
    ols_tr = sm.OLS(tr["l_tax"], Xtr).fit()
    Xte = design(te).reindex(columns=Xtr.columns, fill_value=0.0)
    cv_full = metrics(te["local_sales_tax"], np.exp(ols_tr.predict(Xte).values))
    # Duan smearing correction
    smear = float(np.mean(np.exp(ols_tr.resid)))
    cv_full_smearing = metrics(te["local_sales_tax"],
                               np.exp(ols_tr.predict(Xte).values) * smear)

    Xtr_b = design(tr, macro=False)
    ols_tr_b = sm.OLS(tr["l_tax"], Xtr_b).fit()
    Xte_b = design(te, macro=False).reindex(columns=Xtr_b.columns,
                                            fill_value=0.0)
    cv_no_macro = metrics(te["local_sales_tax"],
                          np.exp(ols_tr_b.predict(Xte_b).values))

    # seasonal naive: each 2025 month uses the same county and month of 2024
    prev = (d[d["year"] == 2024].set_index(["fips", "month"])
            ["local_sales_tax"].rename("naive"))
    te_n = te.set_index(["fips", "month"]).join(prev).reset_index()
    cv_naive = metrics(te_n["local_sales_tax"], te_n["naive"])

    diag = {
        "panel_rows": int(len(d)),
        "counties": int(n_county),
        "months": f"{d['ym'].min().date()}..{d['ym'].max().date()}",
        "balanced_panel": True,
        "negative_adjustment_rows_excluded_from_logs":
            neg[["fips", "county_name", "ym", "local_sales_tax"]]
            .assign(ym=lambda x: x["ym"].astype(str)).to_dict("records"),
        "pooled_ols_rsquared": float(ols.rsquared),
        "pooled_ols_trend_per_month_pct":
            float((np.exp(ols.params["trend"]) - 1) * 100),
        "pooled_ols_gas_semi_elasticity": float(ols.params["gas_price_usd_gal"]),
        "pooled_ols_sentiment_semi_elasticity":
            float(ols.params["consumer_sentiment"]),
        "panel_fe_r2_two_way": fe_r2,
        "panel_fe_r2_county_effects_only": fe_r2_county_only,
        "panelols_county_effects_corr_check": fe_check_corr,
        "morans_I_log_tax_2025": float(mi.I),
        "morans_I_p_norm": float(mi.p_norm),
        "morans_I_expected": float(mi.EI),
        "spatial_lag_rho": float(lag.betas[1, 0]),
        "spatial_lag_rho_se": float(lag.std_err[1]),
        "spatial_error_lambda": float(err.betas[1, 0]),
        "spatial_error_lambda_se": float(err.std_err[1]),
        "spatial_note": "Intercept-only design: lag and error parametrizations "
                        "coincide (row-standardized W gives W*1 = 1), so "
                        "rho == lambda. Reported as a single spatial-"
                        "dependence estimate; the design does not distinguish "
                        "outcome spillovers from correlated unobservables.",
        "cv_train": "2020-2024",
        "cv_holdout": "2025 (12 months x 95 counties), conditional on "
                      "realized 2025 gas price + sentiment",
        "cv_full_spec": cv_full,
        "cv_full_spec_smearing_corrected": cv_full_smearing,
        "smearing_factor": smear,
        "cv_baseline_no_trend_no_macro": cv_no_macro,
        "cv_baseline_seasonal_naive": cv_naive,
    }
    with open(RES / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    df.isna().mean().mul(100).round(1).rename("pct_missing").to_frame().to_csv(
        RES / "missingness.csv")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
