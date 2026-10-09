#!/usr/bin/env python3
"""Build the Tennessee county-month panel.

Inputs (data/): tndor_county_month.csv, tn_counties.geojson, fred_gasregw.csv, fred_umcsent.csv.
The FRED series are national and are repeated for every county in the same month.
Output: data/panel.parquet, one row per county and month.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "panel.parquet"


def main():
    tndor = pd.read_csv(DATA / "tndor_county_month.csv", dtype={"fips": str})
    tndor["ym"] = pd.to_datetime(tndor["ym"] + "-01")
    panel = tndor[["fips", "county_name", "ym", "local_sales_tax",
                   "state_sales_tax"]].copy()
    panel["year"] = panel["ym"].dt.year
    panel["month"] = panel["ym"].dt.month
    panel = panel.sort_values(["fips", "ym"]).reset_index(drop=True)

    gas = pd.read_csv(DATA / "fred_gasregw.csv", parse_dates=["date"])
    gas["ym"] = gas["date"].dt.to_period("M").dt.to_timestamp()
    gas_m = gas.groupby("ym", as_index=False).agg(
        gas_price_usd_gal=("gas_price_usd_gal", "mean"))
    sent = pd.read_csv(DATA / "fred_umcsent.csv", parse_dates=["date"])
    sent["ym"] = sent["date"].dt.to_period("M").dt.to_timestamp()
    fred = gas_m.merge(sent[["ym", "consumer_sentiment"]], on="ym", how="outer")
    panel = panel.merge(fred, on="ym", how="left")
    for c in ["gas_price_usd_gal", "consumer_sentiment"]:
        print(f"  {c} missing: {panel[c].isna().mean()*100:.1f}% "
              f"(national series; NaN months left missing, never filled)")

    import geopandas as gpd
    g = gpd.read_file(DATA / "tn_counties.geojson")
    missing_fips = set(g["GEOID"]) - set(panel["fips"])
    if missing_fips:
        print(f"WARNING: {len(missing_fips)} TIGER counties absent from panel")

    panel.to_parquet(OUT, index=False)
    cov = panel["ym"].agg(["min", "max"])
    print(f"wrote {OUT}: {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"  counties: {panel['fips'].nunique()}, months: "
          f"{cov['min'].date()}..{cov['max'].date()}")
    print(f"  local_sales_tax missing: "
          f"{panel['local_sales_tax'].isna().mean()*100:.1f}%")


if __name__ == "__main__":
    main()
