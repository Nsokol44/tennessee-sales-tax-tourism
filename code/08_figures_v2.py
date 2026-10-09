#!/usr/bin/env python3
"""Two-panel figures, 2025.

fig3_lisa_2025_v2: LISA maps for collections and visitor spending; counties passing the FDR correction are outlined.
fig4_validation_scatter_v2: log visitor spending against log local collections and log state sales tax.
Inputs: data/, results/validation_merged.csv, results/review_checks/D_lisa_9999perm_long.csv. Outputs: figures/ (200 dpi PNG).
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
INK, GREEN, BRICK = "#1E2320", "#2F5D50", "#B5533C"
CAT_COLORS = {"HH": BRICK, "LL": "#3E6FA3", "HL": "#D9A441",
              "LH": "#7FB3A2", "ns": "#E9E4D8"}


def fig3():
    geo = gpd.read_file(ROOT / "data" / "tn_counties.geojson")
    lisa = pd.read_csv(ROOT / "results" / "review_checks" /
                       "D_lisa_9999perm_long.csv")
    lisa = lisa[lisa["year"] == 2025]
    name2geoid = dict(zip(geo["NAME"], geo["GEOID"]))
    lisa["GEOID"] = lisa["county"].map(name2geoid)
    assert lisa["GEOID"].notna().all()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, series, title in [
            (axes[0], "collections", "Local-option collections, 2025"),
            (axes[1], "spending", "Published visitor spending, 2025")]:
        d = lisa[lisa["series"] == series]
        g = geo.merge(d[["GEOID", "class_p05", "class_fdr"]], on="GEOID",
                      how="left")
        g["color"] = g["class_p05"].map(CAT_COLORS).fillna(CAT_COLORS["ns"])
        g.plot(ax=ax, color=g["color"], edgecolor="white", linewidth=0.3)
        surv = g[g["class_fdr"] != "ns"]
        if len(surv):
            surv.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.1)
        ax.set_title(title, fontsize=11, color=INK)
        ax.axis("off")
    handles = [Patch(color=c, label=l) for l, c in
               [("HH", CAT_COLORS["HH"]), ("LL", CAT_COLORS["LL"]),
                ("HL", CAT_COLORS["HL"]), ("LH", CAT_COLORS["LH"]),
                ("Not significant", CAT_COLORS["ns"])]]
    axes[0].legend(handles=handles, loc="lower left", frameon=False,
                   fontsize=9)
    fig.suptitle("LISA clusters, log annual values (Local Moran's I, "
                 "p < 0.05, 9,999 permutations)\nBlack outline: survives "
                 "Benjamini–Hochberg FDR correction", fontsize=11.5,
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG / "fig3_lisa_2025_v2.png", dpi=200)
    plt.close(fig)


def fig4():
    m = pd.read_csv(ROOT / "results" / "validation_merged.csv",
                    dtype={"fips": str})
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    state = (panel.groupby(["fips", "year"], as_index=False)
             .agg(state_tax=("state_sales_tax", "sum")))
    m = m.merge(state, on=["fips", "year"], how="left")
    s = m[m["year"] == 2025].copy()
    s["log_spend"] = np.log(s["visitor_spending_usd"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, col, title in [
            (axes[0], "collections",
             "Local-option collections vs visitor spending"),
            (axes[1], "state_tax",
             "State sales tax vs visitor spending (falsification)")]:
        x = np.log(s[col])
        ax.scatter(x, s["log_spend"], s=16, color=GREEN, alpha=0.75)
        b, a = np.polyfit(x, s["log_spend"], 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=BRICK, lw=1.4)
        r = np.corrcoef(x, s["log_spend"])[0, 1]
        ax.set_title(f"{title}\nr = {r:.3f}, slope = {b:.2f}", fontsize=10.5,
                     color=INK)
        ax.set_xlabel(f"log annual {col.replace('_', ' ')}, 2025")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("log direct visitor spending, 2025 "
                       "(TDTD / Tourism Economics)")
    for name in ["Sevier", "Van Buren", "Davidson", "Cannon"]:
        row = s[s["county_name"] == name]
        if len(row):
            row = row.iloc[0]
            axes[0].annotate(name,
                             (np.log(row["collections"]), row["log_spend"]),
                             fontsize=8.5, xytext=(4, 4),
                             textcoords="offset points")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_validation_scatter_v2.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig3()
    fig4()
    print("v2 figures written")
