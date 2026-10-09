#!/usr/bin/env python3
"""Figures 1-10 as 300-dpi PNG and vector PDF; run last.

Maps use Albers Equal Area (EPSG:5070) with a scale bar and north arrow. Values come from results/validation_merged.csv,
results/extended/ and results/review_checks/; the real-terms series uses the year-mean CPI deflator of script 06.
Outputs: figures/fig1_statewide_series through figures/fig10_forecast_origins.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
INK = "#1E2320"
GREEN = "#2F5D50"
BRICK = "#B5533C"
SRC_TNDOR = "Source: Tennessee Department of Revenue, Monthly & Fiscal Year Tax Collections (Local Sales by County)."
SRC_BENCH = "Sources: TDTD / Tourism Economics (2026), 2025 Economic Impact of Travel on Tennessee; TN Department of Revenue."
ALBERS = "EPSG:5070"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})

# Okabe-Ito palette for LISA classes
LISA_COLORS = {"HH": "#D55E00", "LL": "#0072B2", "HL": "#E69F00",
               "LH": "#56B4E9", "ns": "#E9E4D8"}


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem, "(png+pdf)")


def load_geo():
    geo = gpd.read_file(ROOT / "data" / "tn_counties.geojson")
    geo["GEOID"] = geo["GEOID"].astype(str).str.zfill(5)
    return geo.to_crs(ALBERS)


def add_map_furniture(ax, scalebar=True, north=True):
    """Draw a 0-50-100 km scale bar and a north arrow."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    if scalebar:
        bx, by = x0 + 0.015 * dx, y0 + 0.055 * dy
        seg, h = 50_000.0, 0.035 * dy
        for i in range(2):
            ax.add_patch(plt.Rectangle((bx + i * seg, by), seg, h,
                                       facecolor=INK if i == 0 else "white",
                                       edgecolor=INK, linewidth=0.8, zorder=10))
        for i, lab in enumerate(["0", "50", "100 km"]):
            ax.text(bx + i * seg + (4_000 if i == 2 else 0),
                    by - 0.045 * dy, lab,
                    ha="left" if i == 2 else "center",
                    va="top", fontsize=8, color=INK)
    if north:
        ax.annotate("N", xy=(x0 + 0.975 * dx, y0 + 0.97 * dy),
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color=INK)
        ax.annotate("", xy=(x0 + 0.975 * dx, y0 + 0.90 * dy),
                    xytext=(x0 + 0.975 * dx, y0 + 0.78 * dy),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))


def pad_bottom(ax, frac=0.20):
    """Extend the y-limits downward to leave room for the scale bar."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - frac * (y1 - y0), y1)


def source_note(fig, text, y=0.005):
    fig.text(0.01, y, text, fontsize=7.5, color="#4A4A44", ha="left",
             va="bottom", wrap=True)


# Fig 1
def fig1():
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    cpi = pd.read_csv(ROOT / "data" / "fred_cpi.csv",
                      parse_dates=["observation_date"])
    cpi["year"] = cpi["observation_date"].dt.year
    yr_mean = cpi.groupby("year")["CPIAUCSL"].mean()
    base = float(yr_mean.loc[2026])
    annual = panel.groupby("year")["local_sales_tax"].sum() / 1e9
    years = [y for y in range(2020, 2026)]
    real = pd.Series({y: annual[y] * base / yr_mean[y] for y in years})
    idx_nom = annual[years] / annual[2020] * 100
    idx_real = real[years] / real[2020] * 100
    print(f"fig1 check: 2025 nominal index {idx_nom[2025]:.1f} "
          f"(+{idx_nom[2025]-100:.1f}%), real {idx_real[2025]:.1f} "
          f"(+{idx_real[2025]-100:.1f}%)")

    sw = panel.groupby("ym")["local_sales_tax"].sum() / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    ax.plot(sw.index, sw.values, color=GREEN, lw=1.5)
    ax.fill_between(sw.index, sw.values, color=GREEN, alpha=0.10)
    ax.set_ylabel("Statewide collections ($B / month)")
    ax.set_title("Monthly collections, Jan 2020–Aug 2026 (nominal)",
                 loc="left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    x = np.arange(len(years))
    w = 0.38
    ax.bar(x - w / 2, idx_nom.values, width=w, color=GREEN,
           label="Nominal")
    ax.bar(x + w / 2, idx_real.values, width=w, color=BRICK,
           label="Real (CPI-deflated)")
    ax.axhline(100, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel("Annual total, index (2020 = 100)")
    ax.set_title("2025: +56.0% nominal, +25.4% real", loc="left",
                 fontsize=10.5)
    for xi, yr, v in zip(x - w / 2, years, idx_nom.values):
        if yr != 2020:
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    for xi, yr, v in zip(x + w / 2, years, idx_real.values):
        if yr != 2020:
            ax.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(idx_nom.values) * 1.18)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tennessee local-option sales-tax collections",
                 x=0.01, ha="left", fontsize=12)
    source_note(fig, SRC_TNDOR + " Real terms: FRED CPIAUCSL, year-mean "
                     "step deflator (script 06).")
    save(fig, "fig1_statewide_series")


# Fig 2
def fig2():
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    geo = load_geo()
    a25 = (panel[panel["year"] == 2025]
           .groupby("fips", as_index=False)
           .agg(tax=("local_sales_tax", "sum")))
    a25["fips"] = a25["fips"].astype(str).str.zfill(5)
    a25["log_tax"] = np.log(a25["tax"])
    g = geo.merge(a25, left_on="GEOID", right_on="fips")
    fig, ax = plt.subplots(figsize=(10, 4.3))
    g.plot(column="log_tax", cmap="viridis", ax=ax, edgecolor="#CCCCCC",
           linewidth=0.35)
    ax.axis("off")
    ax.set_title("County local-option sales-tax collections, 2025 "
                 "(log scale)", loc="left", fontsize=12)
    smap = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(g["log_tax"].min(), g["log_tax"].max()))
    cb = fig.colorbar(smap, ax=ax, orientation="horizontal",
                      fraction=0.055, pad=0.16, aspect=36)
    tick_vals = [(1e6, "$1M"), (1e7, "$10M"), (1e8, "$100M"),
                 (5e8, "$500M")]
    lo, hi = g["log_tax"].min(), g["log_tax"].max()
    keep = [(v, lab) for v, lab in tick_vals if lo <= np.log(v) <= hi]
    cb.set_ticks([np.log(v) for v, _ in keep])
    cb.set_ticklabels([lab for _, lab in keep])
    cb.set_label("Annual collections, 2025 (US$, log scale)", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)
    pad_bottom(ax)
    add_map_furniture(ax)
    source_note(fig, SRC_TNDOR + " Boundaries: Census TIGER 2023. "
                     "Albers Equal Area (EPSG:5070).", y=-0.05)
    save(fig, "fig2_choropleth_2025")


# Fig 3
def fig3():
    geo = load_geo()
    lisa = pd.read_csv(ROOT / "results" / "review_checks" /
                       "D_lisa_9999perm_long.csv")
    lisa = lisa[lisa["year"] == 2025]
    name2geoid = dict(zip(geo["NAME"], geo["GEOID"]))
    lisa["GEOID"] = lisa["county"].map(name2geoid)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
    for ax, series, title in [
            (axes[0], "collections", "Local-option collections, 2025"),
            (axes[1], "spending", "Published visitor spending, 2025")]:
        d = lisa[lisa["series"] == series]
        g = geo.merge(d[["GEOID", "class_p05", "class_fdr"]], on="GEOID",
                      how="left")
        g["color"] = g["class_p05"].map(LISA_COLORS).fillna(
            LISA_COLORS["ns"])
        g.plot(ax=ax, color=g["color"], edgecolor="#CCCCCC",
               linewidth=0.35)
        surv = g[g["class_fdr"] != "ns"]
        if len(surv):
            surv.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.axis("off")
    add_map_furniture(axes[0], scalebar=False, north=True)
    add_map_furniture(axes[1], scalebar=True, north=False)
    handles = [Patch(color=c, label=l) for l, c in
               [("High–high", LISA_COLORS["HH"]),
                ("Low–low", LISA_COLORS["LL"]),
                ("High–low", LISA_COLORS["HL"]),
                ("Low–high", LISA_COLORS["LH"]),
                ("Not significant", LISA_COLORS["ns"])]]
    axes[1].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False,
                   fontsize=9)
    fig.suptitle("LISA clusters, log annual values (Local Moran's I, "
                 "p < 0.05, 9,999 permutations)\nBlack outline: survives "
                 "Benjamini–Hochberg FDR correction", x=0.01, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    source_note(fig, SRC_BENCH + " Albers Equal Area (EPSG:5070).")
    save(fig, "fig3_lisa_2025_v2")


# Fig 4
def fig4():
    m = pd.read_csv(ROOT / "results" / "validation_merged.csv",
                    dtype={"fips": str})
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    state = (panel.groupby(["fips", "year"], as_index=False)
             .agg(state_tax=("state_sales_tax", "sum")))
    state["fips"] = state["fips"].astype(str).str.zfill(5)
    m["fips"] = m["fips"].astype(str).str.zfill(5)
    m = m.merge(state, on=["fips", "year"], how="left")
    s = m[m["year"] == 2025].copy()
    s["log_spend"] = np.log(s["visitor_spending_usd"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharey=True)
    for ax, col, title in [
            (axes[0], "collections",
             "Local-option collections vs visitor spending"),
            (axes[1], "state_tax",
             "State sales tax vs visitor spending (falsification)")]:
        x = np.log(s[col])
        ax.scatter(x, s["log_spend"], s=20, color=GREEN, alpha=0.75,
                   edgecolor="white", linewidth=0.3)
        b, a = np.polyfit(x, s["log_spend"], 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=BRICK, lw=1.4)
        r = np.corrcoef(x, s["log_spend"])[0, 1]
        ax.set_title(f"{title}\nr = {r:.3f}, slope = {b:.2f} (n = 95)",
                     fontsize=10.5)
        ax.set_xlabel(f"log annual {col.replace('_', ' ')}, 2025")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(x.min() - 0.4, x.max() + 0.9)
    axes[0].set_ylabel("log direct visitor spending, 2025 "
                       "(TDTD / Tourism Economics)")
    offsets = {"Sevier": (5, 5), "Van Buren": (5, 5),
               "Davidson": (-52, 2), "Cannon": (5, 5)}
    for name in ["Sevier", "Van Buren", "Davidson", "Cannon"]:
        row = s[s["county_name"] == name]
        if len(row):
            row = row.iloc[0]
            axes[0].annotate(name,
                             (np.log(row["collections"]), row["log_spend"]),
                             fontsize=9, xytext=offsets[name],
                             textcoords="offset points")
    source_note(fig, SRC_BENCH + " Levels track scale/rank only "
                     "(see text, Section 5.5).")
    save(fig, "fig4_validation_scatter_v2")


# Fig 5
def fig5():
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    full = panel[panel["year"].between(2020, 2025)].copy()
    annual = full.groupby(["fips", "year"])["local_sales_tax"].transform(
        "sum")
    full["share"] = full["local_sales_tax"] / annual * 100
    seas = full.groupby("month")["share"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(seas.index, seas.values, color=GREEN, width=0.72)
    ax.axhline(100 / 12, color=BRICK, lw=1.2, ls="--",
               label="Uniform month (8.33%)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Mean share of annual collections (%)")
    ax.set_title("Seasonal profile of collections, county average, "
                 "2020–2025 (collections months)", loc="left",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    source_note(fig, SRC_TNDOR)
    save(fig, "fig5_seasonality")


# shared map data
def merged_2025():
    v = pd.read_csv(ROOT / "results" / "validation_merged.csv",
                    dtype={"fips": str})
    v["fips"] = v["fips"].astype(str).str.zfill(5)
    return v


# Fig 6
def fig6():
    v = merged_2025()
    v25 = v[v.year == 2025].copy()
    b, a = np.polyfit(v25["log_collections"], v25["log_spending"], 1)
    v25["resid"] = v25["log_spending"] - (a + b * v25["log_collections"])
    geo = load_geo()
    g = geo.merge(v25[["fips", "county_name", "resid"]],
                  left_on="GEOID", right_on="fips", how="left")
    assert g["resid"].notna().all()
    cmap = "RdBu"
    norm = TwoSlopeNorm(vmin=-1.8, vcenter=0.0, vmax=2.6)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    g.plot(ax=ax, column="resid", cmap=cmap, norm=norm,
           edgecolor="#CCCCCC", linewidth=0.35)
    ax.axis("off")
    ax.set_title("Visitor spending relative to the collections-only "
                 "prediction, 2025 (log–log residuals)", loc="left",
                 fontsize=11.5)
    labels = {"Van Buren": (14, -2), "Sevier": (6, 4), "Moore": (-34, -14),
              "Pickett": (6, 6), "Davidson": (-58, 6), "Cannon": (-56, 10)}
    cen = g.set_geometry(g.geometry.representative_point())
    for _, r in cen[cen["county_name"].isin(labels)].iterrows():
        dx, dy = labels[r["county_name"]]
        ax.annotate(r["county_name"], (r.geometry.x, r.geometry.y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color=INK)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      orientation="horizontal", fraction=0.055, pad=0.06,
                      aspect=36)
    cb.set_label("Residual (log points): log visitor spending minus "
                 "value predicted by log collections", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)
    pad_bottom(ax)
    add_map_furniture(ax)
    source_note(fig, SRC_BENCH + " Albers Equal Area (EPSG:5070).", y=-0.05)
    save(fig, "fig6_residual_map")


# Fig 7
def fig7():
    share = pd.read_csv(
        ROOT / "results" / "extended" / "A_visitor_share_by_county_2025.csv")
    share["share_pct"] = share["visitor_local_share_of_collections"] * 100
    geo = load_geo()
    geo["name_key"] = geo["NAME"].str.lower()
    share["name_key"] = share["county_name"].str.lower()
    g = geo.merge(share, on="name_key", how="left")
    assert g["share_pct"].notna().all()
    fig, ax = plt.subplots(figsize=(10, 4.3))
    g.plot(column="share_pct", cmap="cividis", ax=ax,
           edgecolor="#CCCCCC", linewidth=0.35, vmin=0, vmax=100)
    ax.axis("off")
    ax.set_title("Upper bound on the visitor share of collections, 2025 "
                 "— median county 11.4%; statewide 28.6%",
                 loc="left", fontsize=11.5)
    smap = plt.cm.ScalarMappable(cmap="cividis",
                                 norm=plt.Normalize(0, 100))
    cb = fig.colorbar(smap, ax=ax, orientation="horizontal",
                      fraction=0.055, pad=0.16, aspect=36)
    cb.set_label("Visitor-generated local taxes / local-option "
                 "collections, 2025 (%, upper bound)", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)
    cen = g.set_geometry(g.geometry.representative_point())
    offs7 = {"Sevier": (-34, 0), "Van Buren": (0, 0), "Davidson": (0, 0)}
    for _, r in cen[cen["county_name"].isin(
            ["Sevier", "Van Buren", "Davidson"])].iterrows():
        ax.annotate(f"{r['county_name']} {r['share_pct']:.1f}%",
                    (r.geometry.x, r.geometry.y), ha="center",
                    xytext=offs7[r["county_name"]],
                    textcoords="offset points",
                    fontsize=9, fontweight="bold", color="white")
    pad_bottom(ax)
    add_map_furniture(ax)
    source_note(fig, "Bound: benchmark visitor-generated local taxes "
                     "(TDTD / Tourism Economics, 2026) divided by "
                     "collections (TN Dept. of Revenue). Albers Equal "
                     "Area (EPSG:5070).", y=-0.05)
    save(fig, "fig7_visitor_share_map")


# growth helpers
def growth_frame():
    v = merged_2025()
    w = v.pivot(index=["fips", "county_name"], columns="year",
                values=["collections", "visitor_spending_usd"])
    g = pd.DataFrame({
        "coll_growth": (w[("collections", 2025)]
                        / w[("collections", 2024)] - 1) * 100,
        "spend_growth": (w[("visitor_spending_usd", 2025)]
                         / w[("visitor_spending_usd", 2024)] - 1) * 100,
    }).reset_index()
    steps = pd.read_csv(ROOT / "results" / "extended" /
                        "C_ratio_step_changes.csv", dtype={"fips": str})
    steps["fips"] = steps["fips"].astype(str).str.zfill(5)
    flagged = set(steps.loc[steps["max_step_log"].abs() > 0.10, "fips"])
    g["flagged"] = g["fips"].isin(flagged)
    print(f"growth check: n={len(g)}, flagged={g['flagged'].sum()}, "
          f"r_all={np.corrcoef(g['coll_growth'], g['spend_growth'])[0,1]:.3f}, "
          f"r_excl={np.corrcoef(g.loc[~g['flagged'],'coll_growth'], g.loc[~g['flagged'],'spend_growth'])[0,1]:.3f}")
    return g


# Fig 8
def fig8():
    g = growth_frame()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    keep = g[~g["flagged"]]
    flag = g[g["flagged"]]
    ax.scatter(keep["coll_growth"], keep["spend_growth"], s=30,
               color=GREEN, alpha=0.8, edgecolor="white", linewidth=0.4,
               label=f"No rate step (n = {len(keep)})", zorder=3)
    ax.scatter(flag["coll_growth"], flag["spend_growth"], s=64,
               marker="x", color="#D55E00", linewidths=1.6,
               label=f"Rate-step county (n = {len(flag)})", zorder=4)
    for d, lab in [(g, "all counties"), (keep, "excl. rate-step")]:
        b, a = np.polyfit(d["coll_growth"], d["spend_growth"], 1)
        xs = np.linspace(d["coll_growth"].min(), d["coll_growth"].max(),
                         50)
        r = np.corrcoef(d["coll_growth"], d["spend_growth"])[0, 1]
        ax.plot(xs, a + b * xs,
                color=GREEN if lab.startswith("excl") else "#999999",
                lw=1.3, ls="-" if lab.startswith("excl") else "--")
        print(f"fig8 fit {lab}: r={r:.3f}")
    ax.axhline(0, color=INK, lw=0.7); ax.axvline(0, color=INK, lw=0.7)
    offs = {"Johnson": (6, 6), "Trousdale": (6, -12),
            "Cannon": (6, 6), "Haywood": (6, -13)}
    for name, off in offs.items():
        row = g[g["county_name"] == name].iloc[0]
        ax.annotate(name, (row["coll_growth"], row["spend_growth"]),
                    fontsize=9, xytext=off, textcoords="offset points")
    ax.set_xlabel("Growth in local-option collections, 2024→2025 (%)")
    ax.set_ylabel("Growth in published visitor spending, "
                  "2024→2025 (%)")
    ax.set_title("Growth co-movement is weak once rate-step counties "
                 "are excluded\nr = 0.372 (all) → r = 0.197 (excl. 28 "
                 "flagged)", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    source_note(fig, SRC_BENCH + " Step screen: script 11.")
    save(fig, "fig8_growth_scatter")


# Fig 9
def fig9():
    g = growth_frame()
    geo = load_geo()
    gm = geo.merge(g, left_on="GEOID", right_on="fips", how="left")
    assert gm["coll_growth"].notna().all()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6))
    ax = axes[0]
    norm = TwoSlopeNorm(vmin=-32, vcenter=0.0, vmax=44)
    gm.plot(ax=ax, column="coll_growth", cmap="RdBu", norm=norm,
            edgecolor="#CCCCCC", linewidth=0.35)
    ax.axis("off")
    ax.set_title("Growth in annual local-option collections, "
                 "2024→2025 (%)", loc="left", fontsize=11)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu"),
                      ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.14, aspect=36)
    cb.set_label("Collections growth, 2024→2025 (%)", fontsize=9.5)
    cb.ax.tick_params(labelsize=9); cb.outline.set_visible(False)
    add_map_furniture(ax, scalebar=False, north=True)
    ax = axes[1]
    gm["cat"] = np.where(gm["flagged"], "Rate-step county (n = 28)",
                         "No step detected (n = 67)")
    gm["color"] = np.where(gm["flagged"], "#D55E00", "#E9E4D8")
    gm.plot(ax=ax, color=gm["color"], edgecolor="#CCCCCC",
            linewidth=0.35)
    ax.axis("off")
    ax.set_title("Counties flagged by the local-to-state ratio step "
                 "screen (> 0.10 log points)", loc="left", fontsize=11)
    ax.legend(handles=[Patch(color="#D55E00",
                             label="Rate-step county (n = 28)"),
                       Patch(color="#E9E4D8",
                             label="No step detected (n = 67)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=9.5)
    add_map_furniture(ax, scalebar=True, north=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    source_note(fig, SRC_TNDOR + " Step screen: script 11 "
                     "(results/extended/C_ratio_step_changes.csv). "
                     "Albers Equal Area (EPSG:5070).")
    save(fig, "fig9_growth_and_step_maps")


# Fig 10
def fig10():
    boot = pd.read_csv(ROOT / "results" / "extended" /
                       "D_forecast_bootstrap.csv")
    fc = pd.read_csv(ROOT / "results" / "review_checks" /
                     "B_rolling_origin_forecasts.csv")
    lsdv = fc[fc["model"] == "LSDV trend+month, trained 2021+"].set_index(
        "holdout")["MAPE_pct"]
    naive = fc[fc["model"] ==
               "Seasonal naive x statewide trailing growth"].set_index(
        "holdout")["MAPE_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    yrs = boot["holdout"].astype(int).values
    x = np.arange(len(yrs)); w = 0.38
    ax.bar(x - w / 2, [lsdv[y] for y in yrs], width=w, color=GREEN,
           label="Ex-ante LSDV (2021+)")
    ax.bar(x + w / 2, [naive[y] for y in yrs], width=w, color=BRICK,
           label="Seasonal naive × statewide growth")
    for xi, y in zip(x - w / 2, yrs):
        ax.text(xi, lsdv[y] + 0.25, f"{lsdv[y]:.1f}", ha="center",
                fontsize=9)
    for xi, y in zip(x + w / 2, yrs):
        ax.text(xi, naive[y] + 0.25, f"{naive[y]:.1f}", ha="center",
                fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Holdout (forecast origin) year")
    ax.set_title("MAPE by origin: the ranking flips", loc="left",
                 fontsize=11)
    ax.set_ylim(0, max(max(lsdv), max(naive)) * 1.22)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ypos = np.arange(len(boot))[::-1]
    ax.errorbar(boot["diff_pp"], ypos,
                xerr=[boot["diff_pp"] - boot["ci_lo"],
                      boot["ci_hi"] - boot["diff_pp"]],
                fmt="o", color=INK, ecolor=INK, capsize=4, markersize=5)
    ax.axvline(0, color=BRICK, lw=1.2, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([str(int(y)) for y in yrs])
    ax.set_xlabel("MAPE difference, LSDV − naive (percentage points)")
    ax.set_title("Bootstrap difference (95% intervals):\n"
                 "neither forecast dominates", loc="left", fontsize=11)
    for yi, (_, r) in zip(ypos, boot.iterrows()):
        ax.text(r["ci_hi"] + 0.18, yi,
                f"{r['diff_pp']:+.2f} ({r['ci_lo']:+.2f} to "
                f"{r['ci_hi']:+.2f})", va="center", fontsize=9)
    ax.set_xlim(-2.6, 3.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    source_note(fig, "Forecast evaluation: scripts 07/11 "
                     "(results/review_checks/B_rolling_origin_forecasts"
                     ".csv; results/extended/D_forecast_bootstrap.csv; "
                     "2,000 county-bootstrap replications).")
    save(fig, "fig10_forecast_origins")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    fig6(); fig7(); fig8(); fig9(); fig10()
    print("standard figures complete")
