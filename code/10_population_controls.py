"""Check whether collections predict visitor spending after controlling for population and employment.

Log-log models: M1 spending on collections, M2 on population, M3 on collections and population, M4 adding employment.
Controls are ACS 2024 5-year population (B01003_001) and employed civilians 16+ (B23025_004) from the Census Reporter API.
Inputs: results/validation_merged.csv, data/acs_county_pop_emp.csv (used if the fetch fails).
Outputs: results/population_controls.csv, results/population_controls_summary.json, data/acs_county_pop_emp.csv.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results"
RELEASE = "acs2024_5yr"


def fetch_acs(release):
    url = (f"https://api.censusreporter.org/1.0/data/show/{release}"
           "?table_ids=B01003,B23025&geo_ids=050|04000US47")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.load(r)
    rows = []
    for geoid, tables in d["data"].items():
        fips = geoid[-5:]
        if not fips.startswith("47") or len(fips) != 5:
            continue
        rows.append({"fips": fips,
                     "population": tables["B01003"]["estimate"]["B01003001"],
                     "employed": tables["B23025"]["estimate"]["B23025004"]})
    return pd.DataFrame(rows)


try:
    acs = fetch_acs(RELEASE)
    acs["acs_release"] = RELEASE
    acs.to_csv(BASE / "data" / "acs_county_pop_emp.csv", index=False)
except Exception as e:  # fetch failed: use the ACS file in data/
    print(f"Census Reporter fetch failed ({e}); using packaged data/acs_county_pop_emp.csv")
    acs = pd.read_csv(BASE / "data" / "acs_county_pop_emp.csv", dtype={"fips": str})
assert len(acs) == 95, len(acs)

v = pd.read_csv(BASE / "results" / "validation_merged.csv")
v["fips"] = v.fips.astype(str).str.zfill(5)
d = v.merge(acs, on="fips", how="inner")
assert len(d) == 190, len(d)
d["ln_pop"] = np.log(d.population)
d["ln_emp"] = np.log(d.employed)

SPECS = {"M1_coll": "log_spending ~ log_collections",
         "M2_pop": "log_spending ~ ln_pop",
         "M3_coll_pop": "log_spending ~ log_collections + ln_pop",
         "M4_coll_pop_emp": "log_spending ~ log_collections + ln_pop + ln_emp"}
rows, summary = [], {}
for year in [2024, 2025]:
    sub = d[d.year == year]
    fits = {}
    for name, f in SPECS.items():
        m = smf.ols(f, data=sub).fit(cov_type="HC1")
        fits[name] = m
        rows.append({"sample": str(year), "model": name, "n": int(m.nobs),
                     "r2": float(m.rsquared),
                     "b_collections": float(m.params.get("log_collections", np.nan)),
                     "se_collections": float(m.bse.get("log_collections", np.nan)),
                     "p_collections": float(m.pvalues.get("log_collections", np.nan)),
                     "b_population": float(m.params.get("ln_pop", np.nan)),
                     "se_population": float(m.bse.get("ln_pop", np.nan)),
                     "p_population": float(m.pvalues.get("ln_pop", np.nan)),
                     "b_employed": float(m.params.get("ln_emp", np.nan)),
                     "p_employed": float(m.pvalues.get("ln_emp", np.nan))})
    f_coll = fits["M3_coll_pop"].compare_f_test(fits["M2_pop"])
    f_pop = fits["M3_coll_pop"].compare_f_test(fits["M1_coll"])
    summary[str(year)] = {
        "r2": {k: float(v.rsquared) for k, v in fits.items()},
        "collections_in_M3": {"b": float(fits["M3_coll_pop"].params["log_collections"]),
                              "p": float(fits["M3_coll_pop"].pvalues["log_collections"])},
        "collections_in_M4": {"b": float(fits["M4_coll_pop_emp"].params["log_collections"]),
                              "p": float(fits["M4_coll_pop_emp"].pvalues["log_collections"])},
        "F_add_collections_to_pop": {"F": float(f_coll[0]), "p": float(f_coll[1])},
        "F_add_pop_to_collections": {"F": float(f_pop[0]), "p": float(f_pop[1])}}

pooled = {}
for name, f in SPECS.items():
    m = smf.ols(f + " + C(year)", data=d).fit(cov_type="cluster",
                                              cov_kwds={"groups": d.fips})
    pooled[name] = {"r2": float(m.rsquared),
                    "b_collections": float(m.params.get("log_collections", np.nan)),
                    "p_collections": float(m.pvalues.get("log_collections", np.nan)),
                    "b_population": float(m.params.get("ln_pop", np.nan)),
                    "p_population": float(m.pvalues.get("ln_pop", np.nan))}
    rows.append({"sample": "pooled", "model": name, "n": int(m.nobs),
                 "r2": float(m.rsquared),
                 "b_collections": pooled[name]["b_collections"],
                 "se_collections": float(m.bse.get("log_collections", np.nan)),
                 "p_collections": pooled[name]["p_collections"],
                 "b_population": pooled[name]["b_population"],
                 "se_population": float(m.bse.get("ln_pop", np.nan)),
                 "p_population": pooled[name]["p_population"],
                 "b_employed": float(m.params.get("ln_emp", np.nan)),
                 "p_employed": float(m.pvalues.get("ln_emp", np.nan))})
summary["pooled"] = pooled
summary["corr_ln_pop_ln_emp"] = float(d[["ln_pop", "ln_emp"]].corr().iloc[0, 1])
summary["corr_ln_coll_ln_pop_2025"] = float(
    d[d.year == 2025][["log_collections", "ln_pop"]].corr().iloc[0, 1])

pd.DataFrame(rows).to_csv(OUT / "population_controls.csv", index=False)
(OUT / "population_controls_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
