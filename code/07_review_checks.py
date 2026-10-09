#!/usr/bin/env python3
"""Additional checks on the core results.

A: standard errors for the pooled model. B: rolling-origin forecasts. C: state-tax comparison, elasticity tests and growth correlations.
D: permutation Moran's I, LISA with 9,999 permutations and FDR, state-border counties. E: seasonality by sample period. F: inflation share of growth.
Inputs: data/panel.parquet, results/validation_merged.csv, results/robustness_real_terms.json. Outputs: results/review_checks/.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran, Moran_Local
from libpysal.weights import Queen
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "review_checks"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261005
summary = {}


def load():
    P = pd.read_parquet(ROOT / "data" / "panel.parquet")
    d = P[P["local_sales_tax"] > 0].copy()
    d["l"] = np.log(d["local_sales_tax"])
    d["trend"] = (d["year"] - 2020) * 12 + d["month"] - 1
    V = pd.read_csv(ROOT / "results" / "validation_merged.csv", dtype={"fips": str})
    return P, d, V


def design(dd, macro=True, trend=True):
    X = pd.get_dummies(dd[["fips", "month"]].astype(str), columns=["fips", "month"],
                       drop_first=True, dtype=float)
    if trend:
        X["trend"] = dd["trend"].values
    if macro:
        X["gas"] = dd["gas_price_usd_gal"].astype(float).values
        X["sent"] = dd["consumer_sentiment"].astype(float).values
    return sm.add_constant(X, has_constant="add")


# A: standard errors
def block_a(d):
    d = d.sort_values(["ym", "fips"]).reset_index(drop=True)
    X = design(d)
    m = sm.OLS(d["l"], X)
    tcode = pd.factorize(d["ym"])[0]
    fcode = pd.factorize(d["fips"])[0]
    fits = {
        "county-clustered (paper)": m.fit(cov_type="cluster", cov_kwds={"groups": fcode}),
        "month-clustered": m.fit(cov_type="cluster", cov_kwds={"groups": tcode}),
        "two-way clustered": m.fit(cov_type="cluster",
                                   cov_kwds={"groups": np.column_stack([fcode, tcode])}),
        "Driscoll-Kraay (12 lags)": m.fit(cov_type="hac-groupsum",
                                          cov_kwds={"time": tcode, "maxlags": 12}),
    }
    rows = []
    for name, f in fits.items():
        for v in ("trend", "gas", "sent"):
            rows.append({"se_type": name, "term": v, "coef": f.params[v],
                         "se": f.bse[v], "t": f.params[v] / f.bse[v]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "A_standard_errors.csv", index=False)
    base = t[t.se_type == "county-clustered (paper)"].set_index("term")["se"]
    summary["A_se_inflation_vs_paper"] = {
        name: {v: float(t[(t.se_type == name) & (t.term == v)]["se"].iloc[0] / base[v])
               for v in ("trend", "gas", "sent")}
        for name in fits}
    return t


# B: forecasts
def met(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    e = y - f
    return {"RMSE_k": np.sqrt((e ** 2).mean()) / 1e3, "MAE_k": np.abs(e).mean() / 1e3,
            "MAPE_pct": np.abs(e / y).mean() * 100, "bias_pct": (e / y).mean() * 100}


def block_b(d):
    idx = d.set_index(["fips", "ym"])["l"]
    stt = d.groupby("year")["local_sales_tax"].sum()
    rows = []
    for H in (2023, 2024, 2025):
        tr, te = d[d.year < H], d[d.year == H].copy()

        def lag(k):
            key = pd.MultiIndex.from_arrays([te.fips, te.ym - pd.DateOffset(years=k)])
            return np.exp(idx.reindex(key).values)

        def fit_pred(train, **kw):
            Xtr = design(train, **kw)
            mod = sm.OLS(train["l"], Xtr).fit()
            Xte = design(te, **kw).reindex(columns=Xtr.columns, fill_value=0.0)
            return np.exp(mod.predict(Xte).values)

        n1, n2 = lag(1), lag(2)
        g = stt[H - 1] / stt[H - 2]
        preds = {
            "Full LSDV, realized macro (paper; ex post)": fit_pred(tr),
            "LSDV trend+month, no macro (ex ante)": fit_pred(tr, macro=False),
            "LSDV trend+month, trained 2021+": fit_pred(tr[tr.year >= 2021], macro=False),
            "Seasonal naive (paper)": n1,
            "Seasonal naive x county trailing growth": n1 * (n1 / n2),
            "Seasonal naive x statewide trailing growth": n1 * g,
        }
        for k, v in preds.items():
            ok = ~np.isnan(v)
            rows.append({"holdout": H, "model": k, "n": int(ok.sum()),
                         **{a: round(b, 2) for a, b in
                            met(te["local_sales_tax"].values[ok], v[ok]).items()}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "B_rolling_origin_forecasts.csv", index=False)
    return t


# C: validation
def block_c(P, V):
    out = {}
    st = P.groupby(["fips", "year"], as_index=False).agg(state=("state_sales_tax", "sum"))
    V = V.merge(st, on=["fips", "year"])
    V["l_state"] = np.log(V["state"])
    for yr in (2024, 2025):
        s = V[V.year == yr]
        out[f"r_{yr}"] = {
            "local_vs_spending": float(np.corrcoef(s.log_collections, s.log_spending)[0, 1]),
            "state_vs_spending": float(np.corrcoef(s.l_state, s.log_spending)[0, 1]),
            "local_vs_state": float(np.corrcoef(s.log_collections, s.l_state)[0, 1])}
    s = V[V.year == 2025]
    m2 = sm.OLS(s.log_spending, sm.add_constant(s[["l_state", "log_collections"]])).fit()
    out["local_adds_beyond_state_2025"] = {
        "coef": float(m2.params["log_collections"]), "se": float(m2.bse["log_collections"])}
    rows = []
    for yr in (2024, 2025):
        s = V[V.year == yr]
        f = sm.OLS(s.log_spending, sm.add_constant(s.log_collections)).fit(cov_type="HC3")
        rev = sm.OLS(s.log_collections, sm.add_constant(s.log_spending)).fit()
        b, se = f.params.iloc[1], f.bse.iloc[1]
        rows.append({"sample": str(yr), "b": b, "se_HC3": se, "t_vs_1": (b - 1) / se,
                     "p_vs_1": 2 * (1 - stats.t.cdf(abs((b - 1) / se), len(s) - 2)),
                     "reverse_implied_b": 1 / rev.params.iloc[1],
                     "orthogonal_b": s.log_spending.std() / s.log_collections.std()})
    f = sm.OLS(V.log_spending, sm.add_constant(V.log_collections)).fit(
        cov_type="cluster", cov_kwds={"groups": V.county_name})
    b, se = f.params.iloc[1], f.bse.iloc[1]
    rows.append({"sample": "pooled (county-clustered)", "b": b, "se_HC3": se,
                 "t_vs_1": (b - 1) / se,
                 "p_vs_1": 2 * (1 - stats.norm.cdf(abs((b - 1) / se)))})
    pd.DataFrame(rows).to_csv(OUT / "C_elasticity_tests.csv", index=False)

    # growth, 2024 to 2025
    A = P.groupby(["fips", "year"]).agg(loc=("local_sales_tax", "sum"),
                                       sta=("state_sales_tax", "sum")).reset_index()
    w = A.pivot(index="fips", columns="year", values=["loc", "sta"])
    sp = V.pivot(index="fips", columns="year", values="visitor_spending_usd")
    g = pd.DataFrame({"loc": (w[("loc", 2025)] / w[("loc", 2024)] - 1) * 100,
                      "sta": (w[("sta", 2025)] / w[("sta", 2024)] - 1) * 100,
                      "spend": (sp[2025] / sp[2024] - 1) * 100}).dropna()
    g["gap_local_minus_state"] = g["loc"] - g["sta"]
    names = P.drop_duplicates("fips").set_index("fips")["county_name"]
    g["county"] = names.reindex(g.index)
    g.to_csv(OUT / "C_growth_local_state_spending.csv")

    def corr(a, b):
        return {"pearson": float(np.corrcoef(a, b)[0, 1]),
                "spearman": float(stats.spearmanr(a, b).statistic)}

    lo, hi = g["loc"].quantile([.05, .95])
    trimmed = g[g["loc"].between(lo, hi)]
    clean = g[g.gap_local_minus_state.abs() <= 8]
    out["growth"] = {
        "local_vs_spending": corr(g["loc"], g.spend),
        "state_tax_vs_spending": corr(g.sta, g.spend),
        "local_vs_state": corr(g["loc"], g.sta),
        "local_vs_spending_trimmed_5_95_on_local": corr(trimmed["loc"], trimmed.spend),
        "n_trimmed": int(len(trimmed)),
        "local_vs_spending_excl_big_local_state_gap": corr(clean["loc"], clean.spend),
        "n_excl_gap": int(len(clean)),
        "n_counties_gap_gt_8pp": int((g.gap_local_minus_state.abs() > 8).sum()),
        "largest_gaps": g.reindex(g.gap_local_minus_state.abs().sort_values(
            ascending=False).index[:6])[["county", "loc", "sta", "spend"]]
            .round(1).to_dict("records"),
    }
    summary["C_validation"] = out
    return V


# D: spatial
def block_d(V):
    g = gpd.read_file(ROOT / "data" / "tn_counties.geojson")
    v = V[V.year == 2025].set_index("fips")
    g = g[g.GEOID.isin(v.index)].reset_index(drop=True)
    w = Queen.from_dataframe(g, ids="GEOID")
    w.transform = "R"
    ids = g["GEOID"]
    yc = np.log(v.reindex(ids).collections.values)
    ys = np.log(v.reindex(ids).visitor_spending_usd.values)
    res = sm.OLS(ys, sm.add_constant(yc)).fit().resid
    out = {}
    for label, x in (("collections", yc), ("spending", ys), ("residual", res)):
        np.random.seed(SEED)
        mi = Moran(x, w, permutations=9999)
        out[f"moran_{label}"] = {"I": float(mi.I), "p_norm": float(mi.p_norm),
                                 "p_perm": float(mi.p_sim)}
    names = pd.read_csv(ROOT / "results" / "lisa_2025.csv", dtype={"fips": str}) \
        .set_index("fips")["county_name"].reindex(ids).values
    q = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    allrows = []
    for yr in (2024, 2025):
        vy = V[V.year == yr].set_index("fips")
        for label, col in (("collections", "collections"), ("spending", "visitor_spending_usd")):
            x = np.log(vy.reindex(ids)[col].values)
            lm = Moran_Local(x, w, permutations=9999, seed=SEED)
            rej = multipletests(lm.p_sim, alpha=0.05, method="fdr_bh")[0]
            quad = np.array([q[i] for i in lm.q])
            allrows.append(pd.DataFrame({
                "county": names, "year": yr, "series": label, "p_sim": lm.p_sim,
                "class_p05": np.where(lm.p_sim < 0.05, quad, "ns"),
                "class_fdr": np.where(rej, quad, "ns")}))
    L = pd.concat(allrows)
    L.to_csv(OUT / "D_lisa_9999perm_long.csv", index=False)
    cnt = (L[L.class_p05 != "ns"].groupby(["year", "series", "class_p05"]).size()
           .unstack(fill_value=0))
    cnt.to_csv(OUT / "D_lisa_class_counts_p05.csv")
    out["lisa_class_counts_p05"] = cnt.reset_index().to_dict("records")
    for cls in ("HH", "LL"):
        for yr in (2024, 2025):
            sub = L[L.year == yr]
            a = set(sub[(sub.series == "collections") & (sub.class_p05 == cls)].county)
            b = set(sub[(sub.series == "spending") & (sub.class_p05 == cls)].county)
            out[f"lisa_{cls}_{yr}"] = {"collections": sorted(a), "spending": sorted(b),
                                       "overlap": sorted(a & b)}
    out["lisa_n_fdr"] = {f"{yr}_{k}": sorted(L[(L.year == yr) & (L.series == k)
                                              & (L.class_fdr != "ns")].county)
                         for yr in (2024, 2025) for k in ("collections", "spending")}
    gg = g.to_crs(5070)
    ext = gg.dissolve().geometry.iloc[0].boundary
    on_border = gg.geometry.apply(lambda x: x.boundary.intersection(ext).length > 1000)
    card = pd.Series(w.cardinalities)
    out["edge"] = {"n_counties_on_state_border": int(on_border.sum()),
                   "queen_neighbors": {n: int(card[f]) for n, f in
                                       (("Shelby", "47157"), ("Davidson", "47037"),
                                        ("Knox", "47093"), ("Hamilton", "47065"))}}
    r = pd.Series(res, index=names).sort_values()
    out["residual_extremes"] = {"spending_above_collections": r.tail(6).round(2)[::-1].to_dict(),
                                "spending_below_collections": r.head(5).round(2).to_dict()}
    summary["D_spatial"] = out


# E: seasonality
def block_e(d):
    def prof(dd):
        m = sm.OLS(dd["l"], design(dd)).fit()
        return pd.Series({mm: (np.exp(m.params.get(f"month_{mm}", 0.0)) - 1) * 100
                          for mm in range(1, 13)})
    t = pd.DataFrame({"all_2020_2026 (paper)": prof(d),
                      "2021_2026": prof(d[d.year >= 2021]),
                      "2022_2026": prof(d[d.year >= 2022])}).round(1)
    t.index.name = "collections_month"
    t.to_csv(OUT / "E_seasonal_profile_sensitivity.csv")
    summary["E_trough_month"] = {c: int(t[c].idxmin()) for c in t.columns}


# F: deflator
def block_f(P):
    s = P.groupby("year")["local_sales_tax"].sum()
    rr = json.load(open(ROOT / "results" / "robustness_real_terms.json"))
    nom = s[2025] / s[2020] - 1
    real = rr["growth_2020_2025_pct"]["real"] / 100
    price = (1 + nom) / (1 + real) - 1
    summary["F_inflation_share"] = {
        "nominal_growth_pct": nom * 100, "real_growth_pct": real * 100,
        "implied_price_rise_pct": price * 100,
        "share_in_percentage_points": price / nom,
        "share_in_logs": float(np.log(1 + price) / np.log(1 + nom))}


def main():
    P, d, V = load()
    block_a(d)
    t = block_b(d)
    V = block_c(P, V)
    block_d(V)
    block_e(d)
    block_f(P)
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(t.to_string(index=False))
    print(json.dumps(summary, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
