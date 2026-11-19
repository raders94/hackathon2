"""Accessible Seaborn styling and Korean fonts for static research figures.

The lock covers Matplotlib's shared state. Figures are closed after display;
interactive charts continue to use Plotly via ui.chart.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
from threading import RLock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import streamlit as st

PLOT_LOCK = RLock()
NAVY = "#15374F"
BLUE = "#367DD5"
TEAL = "#079A8A"
CORAL = "#E57259"
AMBER = "#DCA643"
SLATE = "#8898AC"
TERMINAL_COLORS = {
    "BCT": "#EF795C", "BNCT": "#7468D9", "HPNT": "#3584CC",
    "PNC": "#119C90", "PNIT": "#6E9D45", "BPTC_G": "#C58D39", "BPT_S": "#B96BA0",
}


@lru_cache(maxsize=1)
def korean_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR", "Arial Unicode MS"):
        if name in available:
            return name
    return "DejaVu Sans"


@contextmanager
def seaborn_style():
    with PLOT_LOCK, sns.axes_style("whitegrid"), matplotlib.rc_context({
        "font.family": korean_font(), "font.size": 11, "axes.unicode_minus": False,
        "axes.labelcolor": NAVY, "text.color": NAVY, "xtick.color": "#526C81",
        "ytick.color": "#526C81", "axes.edgecolor": "#D8E3EE",
        "grid.color": "#ECF1F7", "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.dpi": 140,
    }):
        yield


def figure_to_png(fig, dpi: int = 170) -> bytes:
    with PLOT_LOCK:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        return buffer.getvalue()


def render_seaborn(fig, alt: str = "", **kwargs) -> None:
    """Render a static statistical figure with an accessible description."""
    with PLOT_LOCK:
        st.pyplot(fig, width="stretch", clear_figure=False, alt=alt or "PORTaI 통계 시각화")
        plt.close(fig)
