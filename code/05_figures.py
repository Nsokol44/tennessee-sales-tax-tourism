#!/usr/bin/env python3
"""Basic figures (200 dpi PNG) in figures/.

Inputs: data/panel.parquet, data/tn_counties.geojson, results/lisa_2025.csv, results/validation_merged.csv.
Outputs: fig1_statewide_series, fig2_choropleth_2025, fig3_lisa_2025, fig4_validation_scatter, fig5_seasonality.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
INK = "#1E2320"
GREEN = "#2F5D50"
BRICK = "#B5533C"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    geo = gpd.read_file(ROOT / "data" / "tn_counties.geojson")

    # Fig 1: statewide monthly collections
    sw = panel.groupby("ym")["local_sales_tax"].sum() / 1e9
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(sw.index, sw.values, color=GREEN, lw=1.6)
    ax.fill_between(sw.index, sw.values, color=GREEN, alpha=0.12)
    ax.set_ylabel("Statewide collections ($B / month)")
    ax.set_title("Tennessee local-option sales-tax collections, monthly total",
                 loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig1_statewide_series.png", dpi=200)
    plt.close(fig)

    # Fig 2: choropleth of log 2025 collections
    a25 = (panel[panel["year"] == 2025].groupby("fips", as_index=False)
           .agg(tax=("local_sales_tax", "sum")))
    a25["log_tax"] = np.log(a25["tax"])
    g2 = geo.merge(a25, left_on="GEOID", right_on="fips")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    g2.plot(column="log_tax", cmap="YlGn", ax=ax, edgecolor="white",
            linewidth=0.3, legend=True,
            legend_kwds={"label": "log annual collections, 2025"})
    ax.axis("off")
    ax.set_title("County local-option sales-tax collections, 2025 (log scale)",
                 loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig2_choropleth_2025.png", dpi=200)
    plt.close(fig)

    # Fig 3: LISA clusters, 2025
    lisa = pd.read_csv(ROOT / "results" / "lisa_2025.csv", dtype={"fips": str})
    g3 = geo.merge(lisa, left_on="GEOID", right_on="fips")
    colors = {"HH": BRICK, "LL": "#3C6E9F", "HL": "#E0A458", "LH": "#9BC4B5",
              "Not significant": "#E8E4DA"}
    g3["color"] = g3["class"].map(colors)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    g3.plot(color=g3["color"], ax=ax, edgecolor="white", linewidth=0.3)
    ax.axis("off")
    ax.set_title("LISA clusters of log collections, 2025 "
                 "(Local Moran's I, p < 0.05, 999 permutations)",
                 loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k)
               for k, c in colors.items()]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig3_lisa_2025.png", dpi=200)
    plt.close(fig)

    # Fig 4: collections vs visitor spending, 2025
    m = pd.read_csv(ROOT / "results" / "validation_merged.csv")
    s = m[m["year"] == 2025]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(s["log_collections"], s["log_spending"], s=22,
               color=GREEN, alpha=0.75, edgecolor="white", linewidth=0.4)
    b, a0 = np.polyfit(s["log_collections"], s["log_spending"], 1)
    xs = np.linspace(s["log_collections"].min(), s["log_collections"].max(), 50)
    ax.plot(xs, a0 + b * xs, color=BRICK, lw=1.4)
    for name in ["Davidson", "Sevier", "Knox", "Shelby", "Hamilton"]:
        r = s[s["county_name"] == name]
        if len(r):
            ax.annotate(name, (r["log_collections"].iloc[0],
                               r["log_spending"].iloc[0]),
                        fontsize=8.5, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("log annual local-option sales-tax collections, 2025")
    ax.set_ylabel("log direct visitor spending, 2025 (TDTD / Tourism Economics)")
    ax.set_title("Validation: collections vs. published visitor spending",
                 loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_validation_scatter.png", dpi=200)
    plt.close(fig)

    # Fig 5: mean monthly share of annual collections
    full = panel[panel["year"].between(2020, 2025)].copy()
    annual = full.groupby(["fips", "year"])["local_sales_tax"].transform("sum")
    full["share"] = full["local_sales_tax"] / annual * 100
    seas = full.groupby("month")["share"].mean()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(seas.index, seas.values, color=GREEN, width=0.72)
    ax.axhline(100 / 12, color=BRICK, lw=1.2, ls="--",
               label="Uniform month (8.33%)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Mean share of annual collections (%)")
    ax.set_title("Seasonal profile of collections, county average, 2020–2025",
                 loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig5_seasonality.png", dpi=200)
    plt.close(fig)

    print("figures written:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
