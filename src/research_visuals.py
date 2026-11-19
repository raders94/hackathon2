"""Statistical figures built only from the bundled research evidence.

Functions return Matplotlib / Plotly figures for both Streamlit and local
preview rendering. Distribution plots require individual observed records;
summary quartiles are never expanded into synthetic observations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from viz_theme import seaborn_style, TERMINAL_COLORS
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import seaborn as sns

TERMINALS = ["BCT", "BNCT", "HPNT", "PNC", "PNIT", "BPTC_G", "BPT_S"]
FACTORS = {"wind": "강풍", "wave": "고파고", "visibility": "저시정"}
NAVY, TEAL, CORAL, SLATE = "#17364A", "#079A8A", "#E57259", "#8898AC"


def weather_effect_heatmap(effects: pd.DataFrame, exposure: str, selected_factor: str):
    """Color only FDR-supported effects; retain estimates in uncertain cells."""
    selected = effects.loc[effects.exposure.eq(exposure)]
    values = selected.pivot(index="terminal", columns="variable", values="adjusted_eff_change_pct").reindex(index=TERMINALS, columns=FACTORS)
    qvalues = selected.pivot(index="terminal", columns="variable", values="fdr_q_7terminal").reindex(index=TERMINALS, columns=FACTORS)
    significant = qvalues.lt(.05) & values.notna()
    # Fixed scale over both exposure choices keeps colors comparable when toggled.
    scale = max(5.0, float(np.ceil(effects.adjusted_eff_change_pct.abs().max())))
    with seaborn_style():
        fig, ax = plt.subplots(figsize=(5.0, 5.25))
        ax.set_facecolor("#EDF1F5")
        sns.heatmap(values, mask=~significant, ax=ax, cmap=sns.blend_palette([CORAL, "#FFFFFF", TEAL], as_cmap=True),
                    vmin=-scale, vmax=scale, center=0, linewidths=2.4, linecolor="white", annot=False,
                    cbar_kws={"label": "조정 작업효율 차이 (%)", "shrink": .68, "pad": .04})
        for row in range(len(TERMINALS)):
            for col in range(len(FACTORS)):
                value = values.iloc[row, col]
                sig = bool(significant.iloc[row, col])
                if pd.isna(value):
                    ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#FFFFFF", hatch="///", edgecolor="#CAD4DF", linewidth=.4))
                    label, color = "자료 없음", "#8192A3"
                else:
                    label = ("0.0%" if abs(value) < .05 else f"{value:+.1f}%") + (" *" if sig else "")
                    color = "white" if sig and abs(value) >= scale*.63 else NAVY
                ax.text(col+.5, row+.5, label, ha="center", va="center", fontsize=10.5, color=color,
                        fontweight="bold" if sig else "normal")
        # The selected factor column matches the adjacent confidence-interval plot.
        col = list(FACTORS).index(selected_factor)
        ax.add_patch(Rectangle((col+.025, .025), .95, len(TERMINALS)-.05, fill=False, edgecolor=NAVY, linewidth=1.8))
        ax.set_xticklabels(list(FACTORS.values()), rotation=0)
        ax.set_yticklabels(TERMINALS, rotation=0)
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="both", length=0)
        ax.set_title(f"기상별 조정 차이 · {exposure.replace('-', '–')}회", loc="left", fontsize=13, fontweight="bold", pad=15)
        fig.tight_layout()
    return fig


def terminal_bubble_map(frame: pd.DataFrame, period: str) -> go.Figure:
    """Area encodes observed calls; axes show directly supplied medians."""
    d = frame.dropna(subset=["calls", "median_cargo", "median_rate"]).copy()
    fig = go.Figure()
    max_calls = float(d.calls.max()) if not d.empty else 1.0
    positions = {"BCT": "bottom center", "BNCT": "top center", "HPNT": "bottom center", "PNC": "top center", "PNIT": "top right", "BPTC_G": "top left", "BPT_S": "bottom left"}
    for _, row in d.iterrows():
        terminal = str(row.terminal)
        fig.add_trace(go.Scatter(
            x=[row.median_cargo], y=[row.median_rate], mode="markers+text", name=terminal,
            marker=dict(size=[row.calls], sizemode="area", sizeref=2*max_calls/(90**2),
                        color=TERMINAL_COLORS[terminal], opacity=.82, line=dict(color="white", width=2)),
            text=[terminal], textposition=positions.get(terminal, "top center"),
            textfont=dict(size=12, color=NAVY), cliponaxis=False,
            customdata=[[row.calls, row.median_berth_h]],
            hovertemplate=(f"<b>{terminal}</b><br>중앙 작업량 %{{x:,.0f}}<br>중앙 작업률 %{{y:.1f}}/h"
                           "<br>기항 수 %{customdata[0]:,.0f}건<br>중앙 체류시간 %{customdata[1]:.1f}h<extra></extra>"),
        ))
    if not d.empty:
        fig.update_xaxes(range=[0, float(d.median_cargo.max())*1.23])
        fig.update_yaxes(range=[0, float(d.median_rate.max())*1.23])
    fig.update_layout(title=f"작업 규모와 작업률 · {period}", showlegend=False,
                      xaxis_title="중앙 작업량 (작업단위)", yaxis_title="중앙 작업률 (/h)",
                      margin=dict(l=15, r=35, t=62, b=55), height=450)
    return fig


def terminal_violin(records: pd.DataFrame, period: str):
    """Use real per-call records; each violin has its own normalized width."""
    clean = records.loc[np.isfinite(records.work_per_hour) & records.work_per_hour.gt(0)].copy()
    counts = clean.groupby("terminal").size()
    order = [t for t in TERMINALS if counts.get(t, 0) > 0]
    sufficiently_sampled = [t for t in order if counts[t] >= 20 and clean.loc[clean.terminal.eq(t), "work_per_hour"].nunique() >= 3]
    with seaborn_style():
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        if sufficiently_sampled:
            sns.violinplot(data=clean.loc[clean.terminal.isin(sufficiently_sampled)], x="work_per_hour", y="terminal",
                           hue="terminal", order=order, hue_order=order, palette=TERMINAL_COLORS,
                           inner="quart", cut=0, density_norm="width", common_norm=False,
                           linewidth=1, linecolor="#FFFFFF", saturation=.86, bw_adjust=.8, legend=False, ax=ax)
        # n<20 / degenerate distributions remain actual points, without KDE.
        for i, terminal in enumerate(order):
            values = clean.loc[clean.terminal.eq(terminal), "work_per_hour"]
            if terminal not in sufficiently_sampled:
                ax.scatter(values, np.full(len(values), i), s=16, color=TERMINAL_COLORS[terminal], alpha=.7)
            ax.scatter([values.median()], [i], s=24, color=NAVY, edgecolor="white", linewidth=.7, zorder=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{t}  n={int(counts[t]):,}" for t in order], fontsize=9.5)
        if not sufficiently_sampled:
            ax.set_ylim(len(order)-.5, -.5)
        ax.set_xlim(left=0)
        ax.set(xlabel="기항별 작업률 (/h)", ylabel="")
        ax.set_title(f"실제 기항의 분포 · {period}", loc="left", fontsize=13, fontweight="bold", pad=15)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=True)
        fig.tight_layout()
    return fig


def annual_rate_heatmap(annual: pd.DataFrame, selected_year):
    """Direct annual medians, no interpolation for absent terminal-years."""
    years = sorted(annual.year.unique())
    values = annual.pivot(index="terminal", columns="year", values="median_rate").reindex(index=TERMINALS, columns=years)
    annotations = values.map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    with seaborn_style():
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        ax.set_facecolor("#EDF1F5")
        sns.heatmap(values, annot=annotations, fmt="", cmap=sns.light_palette("#367DD5", as_cmap=True),
                    mask=values.isna(), linewidths=2.6, linecolor="white", ax=ax,
                    vmin=0, vmax=float(values.max().max()), cbar_kws={"label": "중앙 작업률 (/h)", "shrink": .78})
        for row in range(len(TERMINALS)):
            for col in range(len(years)):
                if pd.isna(values.iloc[row, col]):
                    ax.text(col+.5, row+.5, "자료 없음", ha="center", va="center", fontsize=9, color="#8192A3")
        if selected_year != "전체 기간" and selected_year in years:
            col = years.index(selected_year)
            ax.add_patch(Rectangle((col+.025, .025), .95, len(TERMINALS)-.05, fill=False, edgecolor=NAVY, linewidth=1.8))
        ax.set(xlabel="접안 연도", ylabel="")
        ax.set_yticklabels(TERMINALS, rotation=0)
        ax.tick_params(axis="both", length=0)
        ax.set_title("연도와 터미널을 함께 보는 중앙 작업률", loc="left", fontsize=14, fontweight="bold", pad=15)
        fig.tight_layout()
    return fig
