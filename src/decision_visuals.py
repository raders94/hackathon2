"""Distinct visual forms for case retrieval and exploratory risk prediction.

All outcome positions are observed values.  Jitter separates overlapping marks
vertically; it never changes work-rate values.  Density is decorative smoothing
of the retrieved sample, is clipped to its observed range, and is omitted for
fewer than 20 cases or for a degenerate outcome distribution.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui import BLUE, SLATE, AMBER, NAVY


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip('#')
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def _case_jitter(case_ids) -> np.ndarray:
    """Stable, identifier-based mark placement, independent of Python hash seed."""
    offsets = []
    for case_id in case_ids:
        digest = hashlib.blake2b(str(case_id).encode('utf-8'), digest_size=4).digest()
        unit = int.from_bytes(digest, 'big') / (2**32 - 1)
        offsets.append((unit - .5) * .16)
    return np.asarray(offsets, dtype=float)


def case_raincloud(rows: pd.DataFrame, max_distance: float = 1.0):
    """Return a figure and whether a density was actually drawn."""
    data = rows.copy()
    rates = data.work_per_hour.to_numpy(dtype=float)
    n = len(data)
    if not n or not np.isfinite(rates).all():
        raise ValueError('A case plot needs non-empty, finite observed work rates.')
    q1, median, q3 = np.quantile(rates, [.25, .50, .75])
    spread = float(np.std(rates, ddof=1)) if n > 1 else 0.
    show_density = n >= 20 and len(np.unique(rates)) >= 3 and spread > 0
    fig = go.Figure()
    if show_density:
        robust_scale = min(spread, float(q3 - q1) / 1.34) if q3 > q1 else spread
        bandwidth = max(.9 * robust_scale * n ** (-.2), np.finfo(float).eps)
        fig.add_trace(go.Violin(
            x=rates, y=np.full(n, .24), orientation='h', side='positive', width=.82,
            name='분포 형태', spanmode='hard', bandwidth=bandwidth, points=False,
            meanline_visible=False, box_visible=False, line=dict(color=BLUE, width=1.5),
            fillcolor=_rgba(BLUE, .22), hoverinfo='skip', showlegend=False,
        ))
    fig.add_trace(go.Box(
        x=rates, y=np.full(n, .025), orientation='h', width=.105, quartilemethod='linear',
        boxpoints=False, fillcolor=_rgba(BLUE, .48), line=dict(color=BLUE, width=1.8),
        whiskerwidth=.65, name='중간 50%', hoverinfo='skip', showlegend=False,
    ))
    date_labels = pd.to_datetime(data.start).dt.strftime('%Y-%m-%d %H:%M').to_numpy()
    custom = np.column_stack([
        data.case_id.astype(str).to_numpy(), date_labels,
        data.distance.to_numpy(dtype=float), data.duration_h.to_numpy(dtype=float),
        data.departure_source.astype(str).to_numpy(),
    ])
    fig.add_trace(go.Scatter(
        x=rates, y=-.28 + _case_jitter(data.case_id), mode='markers', name='개별 기항',
        marker=dict(size=10 if n < 20 else 9, color=data.distance.to_numpy(dtype=float),
                    colorscale=[[0, NAVY], [1, '#C6D7E0']], cmin=0, cmax=float(max_distance),
                    line=dict(color='white', width=1), opacity=.95, showscale=False),
        customdata=custom,
        hovertemplate=('<b>%{customdata[0]}</b><br>%{customdata[1]}<br>'
                       '기록 작업률 %{x:.2f}/h<br>조건 거리 %{customdata[2]:.3f}<br>'
                       '체류시간 %{customdata[3]:.2f}h<br>%{customdata[4]}<extra></extra>'),
        showlegend=False,
    ))
    fig.add_annotation(x=float(median), y=.025, text=f'<b>중앙값 {median:.1f}/h</b>',
                       showarrow=False, yshift=27, font=dict(size=13, color=NAVY),
                       bgcolor='rgba(255,255,255,.85)', borderpad=3)
    if not show_density:
        message = f'{n}건 · 밀도 곡선 생략' if n < 20 else '결과의 변이가 작아 밀도 곡선 생략'
        fig.add_annotation(x=.5, y=.84, xref='paper', yref='paper', text=message,
                           showarrow=False, font=dict(size=13, color=SLATE))
    y_ticks = [-.28, .025, .5] if show_density else [-.28, .025]
    y_labels = ['실제 기록', '중간 50%', '분포 형태'] if show_density else ['실제 기록', '중간 50%']
    fig.update_layout(
        height=350 if show_density else 295,
        title=f'유사 기항 {n:,}건 · 분포와 개별 기록을 함께 읽기',
        margin=dict(t=57, b=55, l=18, r=25),
        xaxis=dict(title='기록 작업률 · 원자료 작업단위/h', range=[0, max(float(rates.max()) * 1.09, 1)]),
        yaxis=dict(range=[-.48, .84 if show_density else .40], tickvals=y_ticks,
                   ticktext=y_labels, showgrid=False, zeroline=False, fixedrange=True),
        hovermode='closest', showlegend=False,
    )
    return fig, show_density


def risk_donut(probability: float):
    """A neutral probability display; callers must suppress unsupported values."""
    p = float(probability)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('A probability ring requires a finite supported probability.')
    fig = go.Figure(go.Pie(
        values=[100 * p, 100 * (1 - p)],
        labels=['정의한 효율저하 사건', '그 외 결과'],
        marker=dict(colors=[BLUE, '#E8EEF2'], line=dict(color='white', width=2)),
        hole=.76, sort=False, direction='clockwise', rotation=0,
        textinfo='none', hovertemplate='%{label}<br>모형 확률 %{value:.1f}%<extra></extra>',
        domain=dict(x=[.04, .96], y=[.06, .96]), showlegend=False,
    ))
    fig.add_annotation(x=.5, y=.55, xref='paper', yref='paper', text=f'<b>{100 * p:.1f}%</b>',
                       showarrow=False, font=dict(size=38, color=BLUE))
    fig.add_annotation(x=.5, y=.405, xref='paper', yref='paper', text='탐색적 예측',
                       showarrow=False, font=dict(size=13, color=SLATE))
    fig.update_layout(height=310, margin=dict(t=14, b=10, l=10, r=10), showlegend=False)
    return fig


def brier_comparison(evaluation: dict):
    """Three-model dot comparison with an explicitly enlarged value axis."""
    pairs = [('전체 발생률 기준', float(evaluation['baseline_brier']), False)]
    terminal_score = evaluation.get('terminal_baseline_brier')
    if terminal_score is not None and math.isfinite(float(terminal_score)):
        pairs.append(('터미널별 발생률 기준', float(terminal_score), False))
    model_score = float(evaluation['brier'])
    best_baseline = min(p[1] for p in pairs)
    pairs.append(('새 위험모델', model_score, True))
    pairs.sort(key=lambda item: item[1])
    scores = [p[1] for p in pairs]
    labels = [p[0] for p in pairs]
    colors = [AMBER if p[2] and model_score >= best_baseline else (BLUE if p[2] else SLATE) for p in pairs]
    span = max(max(scores) - min(scores), .005)
    fig = go.Figure(go.Scatter(
        x=scores, y=labels, mode='markers+text',
        marker=dict(size=[17 if p[2] else 13 for p in pairs], color=colors,
                    symbol=['circle' if p[2] else 'diamond' for p in pairs],
                    line=dict(color='white', width=1.5)),
        text=[f'{v:.4f}' for v in scores], textposition='middle right', textfont=dict(size=14, color=NAVY),
        cliponaxis=False, hovertemplate='%{y}<br>Brier %{x:.5f}<extra></extra>',
        showlegend=False,
    ))
    fig.add_vline(x=best_baseline, line_color='#D2DEE5', line_dash='dot', line_width=1)
    fig.update_layout(
        height=265, title='확률오차 비교 · 왼쪽일수록 오차가 작습니다',
        xaxis=dict(title='2025년 Brier score · 차이를 보기 위한 확대 눈금',
                   range=[max(0, min(scores) - span * .45), max(scores) + span * .80], tickformat='.3f'),
        yaxis=dict(autorange='reversed', showgrid=False, zeroline=False),
        margin=dict(t=55, b=55, l=15, r=25), showlegend=False,
    )
    return fig
