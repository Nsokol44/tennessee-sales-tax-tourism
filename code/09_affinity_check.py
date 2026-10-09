"""Monthly check of county collections against Affinity card spending.

Affinity spending is by county of cardholder residence, indexed to Jan 2020 (Opportunity Insights Economic Tracker); 53 of 95 counties are covered.
Fits two-way FE log-level models and pooled and per-county growth correlations, for the full overlap (Jan 2020-Jul 2026) and post-2021.
Inputs: data/panel.parquet, data_sources/affinity_county_monthly.csv. Outputs: results/affinity_check/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from tn_fips import NAME_BY_FIPS  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "affinity_check"
OUT.mkdir(parents=True, exist_ok=True)

panel = pd.read_parquet(BASE / "data" / "panel.parquet").drop(
    columns=["year", "month"]).rename(columns={"county_name": "county", "ym": "month"})
aff = pd.read_csv(BASE / "data_sources" / "affinity_county_monthly.csv", low_memory=False)
M = {int(k): v for k, v in NAME_BY_FIPS.items()}
aff = aff[aff.countyfips.isin(M.keys())].copy()
aff["county"] = aff.countyfips.map(M)
aff["month"] = pd.to_datetime(dict(year=aff.year, month=aff.month, day=1))
aff = aff[["county", "month", "spend_all", "spend_s_all"]]

d = panel.merge(aff, on=["county", "month"], how="inner")
d = d.sort_values(["county", "month"]).reset_index(drop=True)
covered = sorted(d.county.unique())
missing = sorted(set(M.values()) - set(covered))
pd.DataFrame({"covered_county": covered}).to_csv(OUT / "covered_counties.csv", index=False)
pd.DataFrame({"uncovered_county": missing}).to_csv(OUT / "uncovered_counties.csv", index=False)

d["card_raw"] = 1.0 + d.spend_all      # raw level index, Jan 2020 = 1
d["card_sa"] = 1.0 + d.spend_s_all     # seasonally adjusted level index
d["ln_card_sa"] = np.log(d.card_sa)
d["ln_local"] = np.log(d.local_sales_tax.where(d.local_sales_tax > 0))
d["ln_state"] = np.log(d.state_sales_tax.where(d.state_sales_tax > 0))

d["ln_card_raw"] = np.log(d.card_raw)
d["ln_card_sa"] = np.log(d.card_sa)

g = d.groupby("county")
d["d12_ln_local"] = g.ln_local.diff(12)
d["d12_ln_state"] = g.ln_state.diff(12)
d["d12_ln_card"] = g.ln_card_raw.diff(12)
d["d1_ln_local"] = g.ln_local.diff(1)
d["d1_ln_state"] = g.ln_state.diff(1)
d["d1_ln_card"] = g.ln_card_sa.diff(1)
g2 = d.groupby("county")
d["d12_ln_card_lead1"] = g2.d12_ln_card.shift(1)
d["d1_ln_card_lead1"] = g2.d1_ln_card.shift(1)

results = {"n_counties": int(d.county.nunique()),
           "n_county_months": int(len(d)),
           "window": [str(d.month.min().date()), str(d.month.max().date())],
           "uncovered_counties": missing}

# A: two-way FE, log levels
fe_rows = []
for window_name, sub in [("full", d), ("post2021", d[d.month >= "2022-01-01"])]:
    for y in ["ln_local", "ln_state"]:
        s2 = sub.dropna(subset=[y, "ln_card_sa"])
        m = smf.ols(f"{y} ~ ln_card_sa + C(county) + C(month)", data=s2).fit(
            cov_type="cluster", cov_kwds={"groups": s2.county})
        fe_rows.append({"window": window_name, "outcome": y,
                        "card_elasticity": float(m.params["ln_card_sa"]),
                        "se_cluster_county": float(m.bse["ln_card_sa"]),
                        "p": float(m.pvalues["ln_card_sa"]),
                        "r2": float(m.rsquared), "n": int(m.nobs)})
fe = pd.DataFrame(fe_rows)
fe.to_csv(OUT / "A_fe_levels.csv", index=False)
results["A_fe"] = fe_rows

# B, C: pooled growth correlations
def corr_row(df, x, y, name):
    s = df[[x, y]].dropna()
    return {"pair": name, "n": int(len(s)),
            "pearson": float(s[x].corr(s[y])),
            "spearman": float(s[x].corr(s[y], method="spearman"))}

grow_rows = []
for window_name, sub in [("full", d), ("post2021", d[d.month >= "2022-01-01"])]:
    for y, lab in [("d12_ln_local", "local YoY"), ("d12_ln_state", "state YoY")]:
        grow_rows.append({"window": window_name, **corr_row(sub, y, "d12_ln_card", f"{lab} vs card YoY")})
        grow_rows.append({"window": window_name, **corr_row(sub, y, "d12_ln_card_lead1", f"{lab} vs card YoY (card leads 1m)")})
    for y, lab in [("d1_ln_local", "local MoM"), ("d1_ln_state", "state MoM")]:
        grow_rows.append({"window": window_name, **corr_row(sub, y, "d1_ln_card", f"{lab} vs card MoM (SA)")})
        grow_rows.append({"window": window_name, **corr_row(sub, y, "d1_ln_card_lead1", f"{lab} vs card MoM (card leads 1m)")})
grow = pd.DataFrame(grow_rows)
grow.to_csv(OUT / "B_growth_correlations.csv", index=False)
results["B_growth"] = grow_rows

# D: per-county year-over-year correlations
pc = []
for county, sub in d.groupby("county"):
    s = sub[["d12_ln_local", "d12_ln_card"]].dropna()
    if len(s) >= 24:
        pc.append({"county": county, "n": int(len(s)),
                   "pearson_yoy": float(s.d12_ln_local.corr(s.d12_ln_card))})
pc = pd.DataFrame(pc).sort_values("pearson_yoy", ascending=False)
pc.to_csv(OUT / "C_per_county_yoy.csv", index=False)
results["C_per_county"] = {
    "median_r": float(pc.pearson_yoy.median()),
    "q25": float(pc.pearson_yoy.quantile(0.25)), "q75": float(pc.pearson_yoy.quantile(0.75)),
    "share_positive": float((pc.pearson_yoy > 0).mean()),
    "named": {c: float(pc.loc[pc.county == c, "pearson_yoy"].iloc[0])
              for c in ["Davidson", "Shelby", "Knox", "Sevier", "Hamilton"] if c in set(pc.county)},
    "top": pc.head(3).to_dict("records"), "bottom": pc.tail(3).to_dict("records")}

(OUT / "summary.json").write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
