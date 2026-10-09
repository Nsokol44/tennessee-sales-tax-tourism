#!/usr/bin/env python3
"""Local Moran's I (LISA) on log annual county collections, 2024 and 2025.

Queen contiguity weights, row-standardized; 999 permutations with a fixed seed; classes at pseudo p < 0.05.
Inputs: data/panel.parquet, data/tn_counties.geojson.
Outputs: results/lisa_2024.csv, results/lisa_2025.csv, results/lisa_summary.json.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights import Queen

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEED = 20261005

QUAD = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "panel.parquet")
    g = gpd.read_file(ROOT / "data" / "tn_counties.geojson")
    summary = {}
    for year in (2024, 2025):
        a = (df[df["year"] == year]
             .groupby(["fips", "county_name"], as_index=False)
             .agg(tax=("local_sales_tax", "sum")))
        gg = g[g["GEOID"].isin(a["fips"])].copy()
        w = Queen.from_dataframe(gg, ids="GEOID")
        w.transform = "R"
        a = a.set_index("fips").loc[gg["GEOID"]].reset_index()
        y = np.log(a["tax"].values)
        lm = Moran_Local(y, w, permutations=999, seed=SEED)
        a["local_I"] = lm.Is
        a["p_sim"] = lm.p_sim
        a["quadrant"] = [QUAD[q] for q in lm.q]
        a["class"] = np.where(lm.p_sim < 0.05, a["quadrant"], "Not significant")
        a["log_tax"] = y
        a.to_csv(RES / f"lisa_{year}.csv", index=False)
        counts = a["class"].value_counts().to_dict()
        summary[year] = {
            "n_counties": int(len(a)),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "hh_counties": sorted(a.loc[a["class"] == "HH", "county_name"]),
            "ll_counties": sorted(a.loc[a["class"] == "LL", "county_name"]),
        }
        print(year, summary[year]["class_counts"])
    with open(RES / "lisa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
