"""Map of 2025 visitor spending relative to the log-log fit on local-option collections.

Input: results/validation_merged.csv.
Outputs: figures/fig6_residual_map_shareable.png (with ranked panel) and figures/fig6_residual_map.png (map only).
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "figures" / "fig6_residual_map_shareable.png"

v = pd.read_csv(BASE / "results" / "validation_merged.csv")
v25 = v[v.year == 2025].copy()
X = sm.add_constant(v25.log_collections)
fit = sm.OLS(v25.log_spending, X).fit()
v25["resid"] = fit.resid
v25["ratio_x"] = np.exp(v25.resid)
print(f"2025 fit: elasticity={fit.params['log_collections']:.3f} R2={fit.rsquared:.3f}")
print(v25.nlargest(5, "resid")[["county_name", "resid"]].to_string(index=False))
print(v25.nsmallest(3, "resid")[["county_name", "resid"]].to_string(index=False))

geo = gpd.read_file(BASE / "data" / "tn_counties.geojson")
key = next(c for c in geo.columns if c.lower() in ("geoid", "fips", "countyfp"))
geo["fips5"] = geo[key].astype(str).str.zfill(5)
v25["fips5"] = v25.fips.astype(str).str.zfill(5)
m = geo.merge(v25[["fips5", "county_name", "resid", "ratio_x"]], on="fips5", how="left")
assert m.resid.notna().all(), "unmatched counties"

PAPER, INK = "#FAF6EF", "#1E2320"
cmap = LinearSegmentedColormap.from_list("tv", ["#B5533C", "#E8CFC0", "#F3EEE3", "#BFD8CF", "#2F5D50"])
norm = TwoSlopeNorm(vmin=-1.8, vcenter=0.0, vmax=2.6)

fig = plt.figure(figsize=(13.2, 6.9))
fig.patch.set_facecolor(PAPER)
ax = fig.add_axes([0.015, 0.10, 0.70, 0.76])
ax.set_facecolor(PAPER)
m.plot(ax=ax, column="resid", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.45)
ax.set_axis_off()

label_top = ["Van Buren", "Sevier", "Moore", "Pickett", "Davidson", "Cannon"]
cen = m.set_geometry(m.geometry.representative_point())
for _, r in cen[cen.county_name.isin(label_top)].iterrows():
    ax.annotate(r.county_name, (r.geometry.x, r.geometry.y), ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK,
                path_effects=[])

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                  orientation="horizontal", fraction=0.045, pad=0.055, aspect=32)
cb.ax.xaxis.set_label_position("top")
cb.set_label("log points above / below the collections-only prediction", fontsize=9.5, color=INK, labelpad=6)
cb.ax.tick_params(labelsize=9, colors=INK)
cb.outline.set_visible(False)

fig.text(0.03, 0.955, "Where visitor spending outruns the local tax base", fontsize=21,
         fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.03, 0.905, "Tennessee counties, 2025 — visitor spending compared with the level predicted\n"
                     "by local sales-tax collections alone (statewide log–log fit)",
         fontsize=11.5, color=INK, ha="left", va="top")

axr = fig.add_axes([0.735, 0.30, 0.245, 0.52])
axr.set_facecolor(PAPER)
top = v25.nlargest(5, "resid").iloc[::-1]  # five largest positive residuals
bot = v25.nsmallest(3, "resid")            # three most negative residuals
rows = pd.concat([bot, top]).reset_index(drop=True)  # plotted bottom to top
ypos = np.arange(len(rows))
colors = ["#B5533C" if x < 0 else "#2F5D50" for x in rows.resid]
axr.barh(ypos, rows.resid, color=colors, height=0.66)
axr.set_yticks(ypos, rows.county_name)
axr.axvline(0, color=INK, lw=0.8)
for y, (_, r) in zip(ypos, rows.iterrows()):
    if r.resid >= 0:
        axr.text(r.resid + 0.07, y, f"{r.ratio_x:.1f}×", va="center", ha="left",
                 fontsize=10.5, fontweight="bold", color=INK)
    else:
        axr.text(r.resid + 0.09, y, f"{r.ratio_x:.2f}×", va="center", ha="left",
                 fontsize=10.5, fontweight="bold", color="white")
axr.set_xlim(-2.05, 3.1)
axr.tick_params(labelsize=10.5, colors=INK)
for s in axr.spines.values():
    s.set_visible(False)
axr.set_xticks([])
axr.set_title("Visitor spending vs prediction", fontsize=11.5, color=INK, loc="left", pad=8)

fig.text(0.03, 0.028,
         "Sources: TDTD / Tourism Economics, 2025 Economic Impact of Travel on Tennessee (county visitor spending); "
         "TN Department of Revenue (2025 local-option collections).\n"
         "Residuals from a statewide regression of log visitor spending on log collections (elasticity 1.13, R² 0.83). "
         "Analysis: N. Sokol, University of Tennessee, Knoxville.",
         fontsize=8.8, color="#4A4A44", ha="left", va="bottom")

fig.savefig(OUT, dpi=200, facecolor=PAPER)
print("wrote", OUT)

# map-only version
OUTJ = BASE / "figures" / "fig6_residual_map.png"
fig2, ax2 = plt.subplots(figsize=(10.6, 5.4))
fig2.patch.set_facecolor("white")
m.plot(ax=ax2, column="resid", cmap=cmap, norm=norm, edgecolor="white", linewidth=0.45)
ax2.set_axis_off()
cen2 = m.set_geometry(m.geometry.representative_point())
for _, r in cen2[cen2.county_name.isin(label_top)].iterrows():
    ax2.annotate(r.county_name, (r.geometry.x, r.geometry.y), ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=INK)
cb2 = fig2.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2,
                    orientation="horizontal", fraction=0.05, pad=0.03, aspect=34)
cb2.set_label("Residual (log points): log visitor spending minus value predicted by log collections, 2025",
              fontsize=9.5, color=INK)
cb2.ax.tick_params(labelsize=9, colors=INK)
cb2.outline.set_visible(False)
fig2.savefig(OUTJ, dpi=300, facecolor="white", bbox_inches="tight")
print("wrote", OUTJ)
