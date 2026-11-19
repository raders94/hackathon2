"""Paired predictions and additive SHAP charts; no model calculations change."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from ui import BLUE, TEAL, CORAL, SLATE, NAVY


def prediction_pair(result, selected_color=CORAL):
    normal, selected = float(result['normal']), float(result['vessel'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[normal, selected], y=[0, 0], mode='lines',
                            line=dict(color='#CBD9E8', width=10), hoverinfo='skip', showlegend=False))
    for value, label, color, symbol, position in [
        (normal, '평온 기준', BLUE, 'circle', 'top center'),
        (selected, '선택한 날씨', selected_color, 'diamond', 'bottom center'),
    ]:
        fig.add_trace(go.Scatter(x=[value], y=[0], mode='markers+text',
            marker=dict(size=21, color=color, symbol=symbol, line=dict(color='white', width=2)),
            text=[f'<b>{label} {value:.1f} /h</b>'], textposition=position,
            textfont=dict(color=color, size=14), cliponaxis=False, name=label,
            hovertemplate=f'{label}<br>%{{x:.3f}} 작업단위/h<extra></extra>'))
    fig.update_layout(title='두 점 사이의 간격 = 날씨 조건을 바꿨을 때의 예측 차이',
        height=240, margin=dict(t=60,b=55,l=20,r=28), showlegend=False,
        xaxis=dict(title='선박당 평균 작업률 · 원자료 작업단위/h', range=[0, max(normal, selected, 1)*1.22]),
        yaxis=dict(visible=False, range=[-.8,.8]))
    return fig


def shap_waterfall(explanation, selected):
    """All subsets reconcile by a visible 'other inputs' group on log scale."""
    table=explanation['table']
    picked=selected.sort_values('abs_log',ascending=False)
    names=picked.label.tolist()
    contributions=picked.shap_log.astype(float).tolist()
    inputs=picked.value.astype(str).tolist()
    rest=table.loc[~table.feature.isin(picked.feature)]
    if len(rest):
        names.append(f'그 외 {len(rest)}개 입력')
        contributions.append(float(rest.shap_log.sum()))
        inputs.append('표시하지 않은 입력의 SHAP 합')
    names=[n.replace('최근 6시간 접안 유입(혼잡)','최근 6시간<br>접안 유입').replace('선박 작업량(log)','작업량(log)').replace('선박 총톤수(log GT)','총톤수(log GT)') for n in names]
    total=float(sum(contributions))
    if not np.isclose(total,explanation['pred_log']-explanation['base_log'],atol=1e-6):
        raise ValueError('SHAP display must preserve the complete additive sum')
    fig=go.Figure(go.Waterfall(
        x=['학습 기준']+names+['현재 예측'], y=[0]+contributions+[0],
        measure=['absolute']+['relative']*len(contributions)+['total'],
        text=['기여 0']+[f'{v:+.3f}' for v in contributions]+[f'{total:+.3f}'],
        textposition='outside', cliponaxis=False,
        customdata=list(zip(['SHAP 기준']+inputs+['전체 입력의 기여 합'], [0.]+contributions+[total])),
        hovertemplate='%{x}<br>%{customdata[0]}<br>기여 %{customdata[1]:+.4f} log<extra></extra>',
        connector=dict(line=dict(color='#AEBFCF',width=1.5,dash='dot')),
        increasing=dict(marker=dict(color=TEAL,line=dict(width=0))),
        decreasing=dict(marker=dict(color=CORAL,line=dict(width=0))),
        totals=dict(marker=dict(color=BLUE,line=dict(width=0))),
    ))
    cumulative=np.concatenate(([0.],np.cumsum(contributions)))
    spread=max(.1,float(cumulative.max()-cumulative.min()))
    fig.add_hline(y=0,line_color=SLATE,line_dash='dash',line_width=1)
    fig.update_layout(title=f'학습 기준 {explanation["base_rate"]:.1f}/h → 현재 예측 {explanation["pred_rate"]:.1f}/h',
        yaxis_title='학습 기준 대비 누적 SHAP · log(작업률)', xaxis_title='', showlegend=False,
        height=410,margin=dict(t=65,b=75,l=20,r=25), waterfallgap=.28,
        yaxis_range=[float(cumulative.min())-spread*.26,float(cumulative.max())+spread*.3])
    fig.update_xaxes(tickangle=0,tickfont=dict(size=11))
    return fig
