"""Offline *planning lab* for a proposed digital twin, not a real digital twin.

The only learned values passed to this module are CatBoost vessel-average
throughput estimates.  Target recovery fractions are user-supplied assumptions,
NOT estimated policy effects, verified operations, or crane-hour production.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go

@dataclass(frozen=True)
class TrainingLab:
    baseline: float
    exposed: float
    remaining_gap: float
    unit: str = '작업단위/h'

    @classmethod
    def from_model(cls, model_result: dict) -> 'TrainingLab':
        base=float(model_result['aggregate_normal'])
        storm=float(model_result['aggregate'])
        if not np.isfinite(base) or not np.isfinite(storm) or base<=0 or storm<=0:
            raise ValueError('Expected strictly positive finite model estimates')
        return cls(base,storm,max(base-storm,0.))

    def target(self,recovered_gap_pct: float) -> float:
        """Pure arithmetic of an assumption: storm + target% of modeled calm gap."""
        pct=float(recovered_gap_pct)
        if not 0<=pct<=100:raise ValueError('Assumption must be 0..100%')
        return self.exposed + self.remaining_gap*pct/100.

    def target_delta(self,recovered_gap_pct: float) -> float:
        return (self.target(recovered_gap_pct)/self.baseline-1)*100

    def table(self,reassignment_pct:float,protocol_pct:float)->pd.DataFrame:
        cases=[('기상조건별 AI 추정',0.0,'모델 계산'),
               ('가상 전략 A · 작업 재배치',float(reassignment_pct),'임의 회복 가정'),
               ('가상 전략 B · 중단/재개 훈련',float(protocol_pct),'임의 회복 가정')]
        return pd.DataFrame([dict(시나리오=name,회복_가정_pct=pct,목표_평균처리량=self.target(pct),
                                  평온대비_변화_pct=self.target_delta(pct),수치_성격=kind)
                             for name,pct,kind in cases])


def goal_plot(lab:TrainingLab,data:pd.DataFrame):
    names=['평온 기준 (ML)','악천후 조건 (ML)']+data.시나리오.iloc[1:].tolist()
    numbers=[lab.baseline,lab.exposed]+data.목표_평균처리량.iloc[1:].tolist()
    kinds=['학습모델','학습모델','가정값','가정값']
    colors=['#3289A4','#D75C59','#EDA94D','#34A48F']
    fig=go.Figure()
    for name,num,kind,color in zip(names,numbers,kinds,colors):
        fig.add_trace(go.Bar(x=[name],y=[num],marker_color=color,
            text=[f'{num:,.1f}'],textposition='outside',name=kind,
            hovertemplate=f'{name}: %{{y:,.1f}} 작업단위/h<br>수치의 성격: {kind}<extra></extra>'))
    fig.add_hline(y=lab.baseline,line_color='#5D90A2',line_dash='dash',line_width=1,
        annotation_text='평온 기준 (ML)',annotation_position='bottom right')
    fig.update_layout(height=355,showlegend=False,plot_bgcolor='white',paper_bgcolor='white',
        margin=dict(l=24,r=24,t=25,b=30),yaxis_title='선박별 평균처리율 × 가정 접안척수',
        font=dict(family='Malgun Gothic, NanumGothic, sans-serif',color='#17394B',size=12))
    fig.update_yaxes(range=[0,max(numbers)*1.24],gridcolor='#E6EDF1',zeroline=False)
    fig.update_xaxes(tickangle=0)
    return fig


def process_plot():
    """Process illustration only; no quantitative flow widths or fabricated data."""
    labels=['실제 운영 로그 수집','기상·회귀·ML 분석','가상 악천후 훈련','현장 대조검증']
    descriptions=['선박/기상/크레인/정지로그','영향 요인 통제 · 잔차 점검','작업배치/중단재개 정책 실험','실제 Moves/h · 중단시간 측정']
    fig=go.Figure()
    x=[0.15,0.383,0.616,0.85]
    colors=['#2A85A1','#368F9B','#135D72','#3F9C8B']
    for i,(xx,label,desc) in enumerate(zip(x,labels,descriptions)):
        fig.add_shape(type='rect',xref='paper',yref='paper',x0=xx-.107,x1=xx+.107,y0=.19,y1=.92,
                      line=dict(color=colors[i],width=2),fillcolor='#EFF8F9' if i!=2 else '#E4F3F4',
                      layer='below')
        fig.add_annotation(x=xx,y=.68,xref='paper',yref='paper',text=f'<b>0{i+1} {label}</b>',showarrow=False,
                           font=dict(size=16,color='#103D4F'))
        fig.add_annotation(x=xx,y=.38,xref='paper',yref='paper',text=desc,showarrow=False,
                           font=dict(size=11,color='#496F79'))
        if i<3:
            fig.add_annotation(x=(xx+x[i+1])/2,y=.54,xref='paper',yref='paper',text='➜',showarrow=False,
                               font=dict(size=24,color='#0A9694'))
    fig.update_layout(height=190,margin=dict(l=2,r=2,t=4,b=4),plot_bgcolor='white',paper_bgcolor='white',
        xaxis=dict(visible=False,range=[0,1]),yaxis=dict(visible=False,range=[0,1]))
    return fig
