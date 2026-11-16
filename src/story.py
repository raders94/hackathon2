"""Pure chart/data helpers used by the BUSAN7 BCT evidence walkthrough.

All numerical results come from the packaged regression/model assets. Historical
v4 figures are labeled separately and must not be combined with v7 estimates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

NAVY='#17364A'; AQUA='#058F9A'; CORAL='#E5656D'; MUTED='#8A9CA9'; BLUE='#467EB7'
TERM=['BCT','BNCT','HPNT','PNC','PNIT','BPTC_G','BPT_S']
COLORS={t:(CORAL if t=='BCT' else ('#D9967E' if t.startswith('BPT') else '#70A9B5')) for t in TERM}

def styled(fig,height=360):
    fig.update_layout(template='plotly_white',height=height,
      margin=dict(l=18,r=25,t=60,b=35),font=dict(family='Malgun Gothic, Noto Sans KR, Arial',size=13,color=NAVY),
      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
      hoverlabel=dict(bgcolor=NAVY,font_color='white'),
      legend=dict(orientation='h',x=0,y=1.14))
    fig.update_xaxes(gridcolor='#E4EDF2',zeroline=False)
    fig.update_yaxes(gridcolor='#E4EDF2',zeroline=False)
    return fig

def rate_box(history,year=None):
    d=history[(history.work_per_hour.notna())&(history.work_per_hour>0)].copy()
    if year is not None:d=d[d.year==year]
    if len(d)>15000:d=d.sample(15000,random_state=42)
    fig=px.box(d,x='terminal',y='work_per_hour',color='terminal',
         category_orders={'terminal':TERM},color_discrete_map=COLORS,
         points=False,title='7개 터미널의 선박별 평균 처리량 분포',
         labels={'terminal':'터미널','work_per_hour':'선박 전체 체류시간 기준 작업량/h'})
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0,float(min(200,d.work_per_hour.quantile(.99)*1.1))])
    return styled(fig,365)

def threshold_chart(th):
    d=[]
    for _,r in th.iterrows():
        for fact,lab in [('wind','풍속(m/s)'),('wave','유의파고(m)')]:
            d.append({'항만':r.harbor,'기상':lab,'P90':float(r[fact+'_P90']),'P95':float(r[fact+'_P95'])})
    x=pd.DataFrame(d)
    fig=go.Figure()
    for j,(har,fact) in enumerate([('북항','풍속(m/s)'),('신항','풍속(m/s)'),('북항','유의파고(m)'),('신항','유의파고(m)')]):
        row=x[(x['항만']==har)&(x['기상']==fact)].iloc[0]
        fig.add_trace(go.Bar(x=[row.P90,row.P95],y=[f'{har} · {fact}']*2,
         orientation='h',name='P90 / P95',marker_color=[BLUE,CORAL],
         text=[f'P90 {row.P90:g}',f'P95 {row.P95:g}'],textposition='auto',
         showlegend=False,offsetgroup=0 if j<2 else 1,
         hovertemplate='%{y}<br>%{text}<extra></extra>'))
    fig.update_layout(title='북항과 신항: 고정 P90/P95 기준 · 2020~2024',barmode='group')
    return styled(fig,330)

def dose_forest(dose):
    d=dose[(dose.scheme==3)&(dose.group=='4-6h')].copy()
    d['order']=d.terminal.map({t:i for i,t in enumerate(TERM)})
    d=d.sort_values('order',ascending=False)
    fig=go.Figure()
    for _,r in d.iterrows():
        sig=(r.FDR_q7<.05)
        col=CORAL if r.terminal=='BCT' else (AQUA if sig else MUTED)
        fig.add_trace(go.Scatter(
          x=[r.effect_pct],y=[r.terminal],mode='markers',showlegend=False,
          marker=dict(size=15,color=col,line=dict(width=1,color='#FFFFFF')),
          error_x=dict(type='data',symmetric=False,array=[max(0,r.CI_hi_pct-r.effect_pct)],arrayminus=[max(0,r.effect_pct-r.CI_lo_pct)],thickness=2,color=col),
          customdata=[[int(r.exposed_n),r.FDR_q7]],
          hovertemplate='%{y}: %{x:+.2f}%<br>노출 기항 %{customdata[0]:,}건<br>FDR q=%{customdata[1]:.4f}<extra></extra>'))
    fig.add_vline(x=0,line_dash='dot',line_color=MUTED)
    fig.update_layout(title='강풍 P90 첫 6시간 중 4–6h 노출 시 체류시간 조정 차이',xaxis_title='평온 0h 대비 체류시간 변화 (%)',yaxis_title='')
    return styled(fig,390)

def bct_five_stage(dose):
    d=dose[(dose.terminal=='BCT')&(dose.scheme==5)].copy()
    groups=['1h','2h','3-4h','5-6h'];d['grp']=pd.Categorical(d.group,groups,ordered=True);d=d.sort_values('grp')
    fig=go.Figure(go.Scatter(x=d.group,y=d.effect_pct,mode='lines+markers+text',
       line=dict(color=AQUA,width=3),marker=dict(size=13,color=[CORAL if x=='5-6h' else AQUA for x in d.group]),
       text=[f'n={int(x)}' for x in d.exposed_n],textposition='top center',
       error_y=dict(type='data',symmetric=False,array=(d.CI_hi_pct-d.effect_pct).clip(lower=0),arrayminus=(d.effect_pct-d.CI_lo_pct).clip(lower=0),thickness=2,color=AQUA),
       customdata=d[['FDR_q7','exposed_n']],
       hovertemplate='%{x}<br>체류시간: %{y:+.2f}%<br>FDR q=%{customdata[0]:.4f}<br>표본 %{customdata[1]}건<extra></extra>'))
    fig.add_hline(y=0,line_dash='dot',line_color=MUTED)
    fig.update_layout(title='BCT · 강풍 노출시간 세분화 (5단계)',xaxis_title='접안 후 첫 6시간 중 P90 강풍 노출',yaxis_title='0h 대비 조정 체류시간 변화 (%)')
    return styled(fig,390)

def wind_wave_forest(reg,factor='wind',threshold='P90'):
    d=reg[(reg.cohort=='GT_control_2020_2024')&(reg.factor==factor)&(reg.threshold==threshold)&(reg.status=='OK')].copy()
    d['order']=d.terminal.map({t:i for i,t in enumerate(TERM)})
    d=d.sort_values('order',ascending=False)
    fig=go.Figure()
    for _,r in d.iterrows():
        col=CORAL if r.terminal=='BCT' else (AQUA if r.FDR_q7<.05 else MUTED)
        fig.add_trace(go.Scatter(x=[r.effect_pct],y=[r.terminal],mode='markers',showlegend=False,
          marker=dict(size=13,color=col),error_x=dict(type='data',symmetric=False,
          array=[max(0,r.CI_hi_pct-r.effect_pct)],arrayminus=[max(0,r.effect_pct-r.CI_lo_pct)],color=col),
          customdata=[[int(r.exposed_n),r.FDR_q7]],
          hovertemplate='%{y}: %{x:+.2f}%<br>노출표본 %{customdata[0]}<br>FDR q=%{customdata[1]:.3f}<extra></extra>'))
    fig.add_vline(x=0,line_dash='dot',line_color=MUTED)
    fact={'wind':'풍속','wave':'유의파고','vis_10m':'저시정'}.get(factor,factor)
    fig.update_layout(title=f'GT·작업량·혼잡 통제 회귀 · {fact} {threshold} (접안시점)',xaxis_title='조정 체류시간 변화 (%)')
    return styled(fig,355)

def ml_year_chart(hourly,terminal='BCT',factor='wave',threshold='P95'):
    d=hourly[(hourly.terminal==terminal)&(hourly.factor==factor)&(hourly.threshold==threshold)&(hourly.status=='OK')].copy().sort_values('year')
    fig=go.Figure()
    for _,r in d.iterrows():
        col=CORAL if r.FDR_q7<.05 else BLUE
        fig.add_trace(go.Scatter(x=[r.year],y=[r.delta_residual_pp],mode='markers+text',
          marker=dict(size=15,color=col),text=[f'{r.delta_residual_pp:+.2f}%p'],textposition='top center',showlegend=False,
          error_y=dict(type='data',symmetric=False,array=[max(0,r.CI_hi_pp-r.delta_residual_pp)],arrayminus=[max(0,r.delta_residual_pp-r.CI_lo_pp)],color=col),
          customdata=[[int(r.exposed_hours),float(r.FDR_q7)]],
          hovertemplate='%{x}년<br>정상 기대 대비 차이 %{y:+.2f}%p<br>노출선박-시간 %{customdata[0]:,}<br>FDR q=%{customdata[1]:.3f}<extra></extra>'))
    fig.add_hline(y=0,line_dash='dot',line_color=MUTED)
    fig.update_xaxes(tickmode='array',tickvals=[2024,2025],range=[2023.55,2025.45])
    fig.update_layout(title=f'{terminal} · {"고파고" if factor=="wave" else "강풍"} {threshold} | 2024 vs 2025',yaxis_title='ML 정상 기대 대비 출항률 차이 (%p)',xaxis_title='독립 검증 연도')
    return styled(fig,360)

def responsiveness_bar(compare):
    d=compare.sort_values('delta_pct')
    fig=go.Figure(go.Bar(x=d['delta_pct'],y=d.terminal,orientation='h',
          marker_color=[COLORS[t] for t in d.terminal],text=[f'{v:+.1f}%' for v in d.delta_pct],textposition='outside',
          customdata=d[['per_vessel','active_ships','aggregate']],
          hovertemplate='%{y}<br>평온 대비 %{x:+.2f}%<br>선박당 %{customdata[0]:.1f}/h<br>동시접안 가정 %{customdata[1]}척<br>합산 추정 %{customdata[2]:.1f}/h<extra></extra>'))
    fig.add_vline(x=0,line_dash='dot',line_color=MUTED)
    vals=d.delta_pct.to_numpy(dtype=float); bound=max(1.5,np.nanmax(np.abs(vals))*1.4)
    fig.update_xaxes(range=[-bound,bound])
    fig.update_layout(title='동일 기상 입력에 따른 7개 터미널 모델 반응',xaxis_title='동일조건 평온 대비 추정 변화율 (%)',yaxis_title='')
    return styled(fig,370)
