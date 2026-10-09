#!/usr/bin/env python3
"""Further checks of collections against the TDTD benchmark.

A: visitor-generated local taxes as a share of collections, by county. B: spending categories against collections.
C: local tax-rate or allocation changes flagged from the local-to-state ratio, with growth correlations with and without them.
D: forecast comparison with county-cluster bootstrap intervals.
Inputs: data/panel.parquet, results/validation_merged.csv. Outputs: results/extended/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extended"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261006)
S = {}


def load():
    P = pd.read_parquet(ROOT / "data" / "panel.parquet")
    V = pd.read_csv(ROOT / "results" / "validation_merged.csv", dtype={"fips": str})
    st = P.groupby(["fips", "year"], as_index=False).agg(state=("state_sales_tax", "sum"))
    V = V.merge(st, on=["fips", "year"])
    return P, V


def block_a(V):
    s = V[V.year == 2025].copy()
    s["eff_local_rate"] = s.local_tax_receipts_usd / s.visitor_spending_usd
    s["eff_state_rate"] = s.state_tax_receipts_usd / s.visitor_spending_usd
    s["visitor_local_share_of_collections"] = s.local_tax_receipts_usd / s.collections
    s["visitor_state_share_of_state_tax"] = s.state_tax_receipts_usd / s.state
    cols = ["county_name", "visitor_spending_usd", "collections", "eff_local_rate",
            "eff_state_rate", "visitor_local_share_of_collections", "visitor_state_share_of_state_tax"]
    s[cols].sort_values("visitor_local_share_of_collections", ascending=False).to_csv(
        OUT / "A_visitor_share_by_county_2025.csv", index=False)
    q = s.visitor_local_share_of_collections.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
    S["A"] = {
        "statewide_visitor_local_taxes_over_collections_2025":
            float(s.local_tax_receipts_usd.sum() / s.collections.sum()),
        "county_share_quantiles": q, "n_counties_share_gt_0.4": int((s.visitor_local_share_of_collections > .4).sum()),
        "top5": s.sort_values("visitor_local_share_of_collections", ascending=False)
                 .head(5)[["county_name", "visitor_local_share_of_collections"]].round(3).to_dict("records"),
        "eff_local_rate_mean_sd_cv": [float(s.eff_local_rate.mean()), float(s.eff_local_rate.std()),
                                      float(s.eff_local_rate.std() / s.eff_local_rate.mean())],
        "eff_state_rate_mean_sd_cv": [float(s.eff_state_rate.mean()), float(s.eff_state_rate.std()),
                                      float(s.eff_state_rate.std() / s.eff_state_rate.mean())],
        "corr_log_visitor_local_tax_vs_log_spending": float(np.corrcoef(
            np.log(s.local_tax_receipts_usd), np.log(s.visitor_spending_usd))[0, 1]),
    }


def block_b(V):
    s = V[V.year == 2025]
    comps = ["spending_lodging_usd", "spending_food_beverages_usd", "spending_retail_usd",
             "spending_recreation_usd", "spending_transportation_usd", "visitor_spending_usd",
             "labor_income_usd", "employment_jobs"]
    rows = []
    for c in comps:
        x = np.log(s[c].replace(0, np.nan)).dropna()
        j = s.loc[x.index]
        rows.append({"series": c, "corr_with_log_collections": np.corrcoef(np.log(j.collections), x)[0, 1],
                     "corr_with_log_state_tax": np.corrcoef(np.log(j.state), x)[0, 1], "n": len(x)})
    pd.DataFrame(rows).round(3).to_csv(OUT / "B_component_correlations_2025.csv", index=False)
    X = np.log(s[comps[:5]].replace(0, np.nan)).dropna()
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    S["B"] = {"vif_log_components": {c: float(vif(sm.add_constant(X).values, i + 1))
                                     for i, c in enumerate(comps[:5])}}


def block_c(P, V):
    P = P.copy()
    P = P[P.local_sales_tax > 0]
    P["lr"] = np.log(P.local_sales_tax / P.state_sales_tax)
    P = P.sort_values(["fips", "ym"])
    rows = []
    for f, g in P.groupby("fips"):
        lr = g.set_index("ym").lr
        pre = lr.rolling(12).median(); post = lr[::-1].rolling(12).median()[::-1]
        step = (post.shift(-1) - pre).dropna()
        if len(step):
            m = step.abs().idxmax()
            rows.append({"fips": f, "county": g.county_name.iloc[0], "max_step_log": float(step[m]),
                         "step_month": str(m.date())})
    st = pd.DataFrame(rows).sort_values("max_step_log", key=abs, ascending=False)
    st.to_csv(OUT / "C_ratio_step_changes.csv", index=False)
    flagged = set(st.loc[st.max_step_log.abs() > 0.10, "fips"])
    A = P.groupby(["fips", "year"]).agg(loc=("local_sales_tax", "sum"), sta=("state_sales_tax", "sum")).reset_index()
    w = A.pivot(index="fips", columns="year", values=["loc", "sta"])
    sp = V.pivot(index="fips", columns="year", values="visitor_spending_usd")
    g = pd.DataFrame({"loc": (w[("loc", 2025)] / w[("loc", 2024)] - 1) * 100,
                      "sta": (w[("sta", 2025)] / w[("sta", 2024)] - 1) * 100,
                      "spend": (sp[2025] / sp[2024] - 1) * 100}).dropna()
    def c(a, b, d):
        return {"pearson": float(np.corrcoef(d[a], d[b])[0, 1]), "spearman": float(stats.spearmanr(d[a], d[b]).statistic),
                "n": int(len(d))}
    keep = g[~g.index.isin(flagged)]
    S["C"] = {"n_flagged_step_gt_10pct": len(flagged),
              "flagged": sorted(st.loc[st.max_step_log.abs() > .10, "county"].tolist()),
              "growth_local_vs_spending_all": c("loc", "spend", g),
              "growth_local_vs_spending_unflagged": c("loc", "spend", keep),
              "growth_state_tax_vs_spending_all": c("sta", "spend", g),
              "growth_state_tax_vs_spending_unflagged": c("sta", "spend", keep)}


def block_d(P):
    d = P[P.local_sales_tax > 0].copy()
    d["l"] = np.log(d.local_sales_tax); d["trend"] = (d.year - 2020) * 12 + d.month - 1
    idx = d.set_index(["fips", "ym"]).l
    def design(dd):
        X = pd.get_dummies(dd[["fips", "month"]].astype(str), columns=["fips", "month"], drop_first=True, dtype=float)
        X["trend"] = dd.trend.values
        return sm.add_constant(X, has_constant="add")
    stt = d.groupby("year").local_sales_tax.sum()
    rows = []
    for H in (2023, 2024, 2025):
        tr, te = d[d.year < H], d[d.year == H].copy()
        Xtr = design(tr[tr.year >= 2021]); m = sm.OLS(tr[tr.year >= 2021].l, Xtr).fit()
        lsdv = np.exp(m.predict(design(te).reindex(columns=Xtr.columns, fill_value=0.0)).values)
        key = pd.MultiIndex.from_arrays([te.fips, te.ym - pd.DateOffset(years=1)])
        n1 = np.exp(idx.reindex(key).values)
        naive_g = n1 * (stt[H - 1] / stt[H - 2])
        y = te.local_sales_tax.values
        e1 = np.abs((y - lsdv) / y); e2 = np.abs((y - naive_g) / y)
        diff = pd.Series(e1 - e2).groupby(te.fips.values).mean()      # county mean of the error difference
        boot = [diff.sample(len(diff), replace=True, random_state=int(RNG.integers(1e9))).mean() for _ in range(2000)]
        rows.append({"holdout": H, "mape_lsdv_ex_ante_2021plus": e1.mean() * 100, "mape_naive_x_state_growth": e2.mean() * 100,
                     "diff_pp": diff.mean() * 100, "ci_lo": np.percentile(boot, 2.5) * 100, "ci_hi": np.percentile(boot, 97.5) * 100})
    pd.DataFrame(rows).round(2).to_csv(OUT / "D_forecast_bootstrap.csv", index=False)


if __name__ == "__main__":
    P, V = load()
    block_a(V); block_b(V); block_c(P, V); block_d(P)
    (OUT / "extended_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(json.dumps(S, indent=2, default=float))
