"""PORTCAST Hackathon 2026 | publicly shareable aggregated-evidence dashboard.

Run `streamlit run app.py`. No live data or web service connections.
The model predicts vessel-average cargo / total berth-hour, not true intra-hour
crane/terminal Moves or an intervention's causal effect.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,date,timedelta
import io
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from logic import (ASSETS,TERMINALS,HARBOR,Scenario,category,calm_for,defaults_for,occupancy_for,
    estimate,error_range,historical_at,describe_source,load_model,metadata,read_csv,read_history,read_weather,model_row,congestion_reference,congestion_count)
from story import (styled,rate_box,threshold_chart,dose_forest,bct_five_stage,wind_wave_forest,ml_year_chart,responsiveness_bar,
    COLORS,NAVY,AQUA,CORAL,BLUE,MUTED)
from digital_twin_lab import TrainingLab,goal_plot,process_plot
from explainability import explain_prediction, compare_weather_attributions

st.set_page_config(page_title='PORTCAST | 부산항 기상영향·AI·운영학습',page_icon='⚓',layout='wide',initial_sidebar_state='collapsed')
sns.set_theme(style='whitegrid',palette='deep')
PALETTE=sns.color_palette('deep').as_hex()

st.markdown('''<style>
.main .block-container { max-width:1440px; padding-top:1.0rem; padding-bottom:3rem; }
[data-testid="stMetric"] {border:1px solid #DFE8ED;background:#FFFFFF;padding:13px 17px;border-radius:15px;}
[data-testid="stMetricLabel"] {font-size:0.88rem;color:#607788;}
[data-testid="stMetricValue"] {font-weight:800;color:#10354A;font-size:1.85rem;}
[data-testid="stMetricDelta"] {font-size:0.82rem;}
/* Executive forecast should be the first visible item in the prediction tab. */
.predict-lead{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:8px 0 9px;flex-wrap:wrap;}
.predict-lead strong{font-size:1.18rem;color:#11394C;letter-spacing:-.02em;}
.predict-lead span{font-size:.79rem;color:#5F7B89;}
.executive-kpis{display:grid;grid-template-columns:1.05fr 1fr .82fr;gap:12px;margin:0 0 10px;}
.exec-kpi{border:1px solid #D7E4EA;border-radius:16px;background:#F7FAFB;padding:17px 20px 16px;min-height:155px;box-sizing:border-box;}
.exec-kpi .kpi-label{font-size:.82rem;font-weight:750;color:#537180;margin-bottom:8px;}
.exec-kpi .kpi-value{font-size:clamp(2rem,3.25vw,3rem);font-weight:900;color:#143D51;line-height:1.15;white-space:nowrap;letter-spacing:-.055em;}
.exec-kpi .kpi-value small{font-size:.39em;letter-spacing:0;font-weight:750;margin-left:3px;}
.exec-kpi .kpi-caption{font-size:.79rem;color:#627D8B;margin-top:11px;line-height:1.5;}
.exec-kpi.focus-drop{background:linear-gradient(130deg,#FFF6F3,#FFE6E1);border:2px solid #E17D75;}
.exec-kpi.focus-drop .kpi-label{color:#8A3431;}
.exec-kpi.focus-drop .kpi-value{color:#B83731;font-size:clamp(2.7rem,4.3vw,4.1rem);}
.exec-kpi.focus-rise{background:linear-gradient(130deg,#F1F9F6,#E3F5EA);border:2px solid #66B38C;}
.exec-kpi.focus-rise .kpi-label,.exec-kpi.focus-rise .kpi-value{color:#186B4B;}
.exec-kpi.focus-flat{background:#F4F8FB;border:2px solid #95ADBD;}
.exec-kpi.focus-flat .kpi-value{color:#33586C;}
.predict-footnote{font-size:.76rem;color:#5B7583;padding:2px 0 6px;line-height:1.55;}
.selected-summary{margin:12px 0 8px;padding:12px 16px 13px;background:#F3F8FA;border:1px solid #CFDEE4;border-radius:14px;}
.selected-summary .summary-title{font-size:.88rem;font-weight:850;color:#17465B;margin-bottom:9px;}
.selected-summary .summary-grid{display:flex;flex-wrap:wrap;align-items:center;gap:7px;}
.selected-summary .summary-chip{background:#FFFFFF;border:1px solid #D7E5EB;border-radius:8px;padding:7px 10px;color:#365466;font-size:.82rem;white-space:nowrap;}
.selected-summary .summary-chip b{color:#163C51;font-weight:800;}
.selected-summary .summary-chip.risk{background:#FFF3ED;border-color:#F0C1AA;color:#9A3A24;font-weight:750;}
.selected-congestion{border:1px solid #CADFE5;border-radius:9px;background:#F2F8F9;color:#184B60;padding:9px 13px;font-size:.87rem;margin:8px 0;}
@media (max-width:960px){.executive-kpis{grid-template-columns:1fr;}.exec-kpi{min-height:118px;}.exec-kpi .kpi-value{font-size:2.7rem;}}
.port-hero{background:linear-gradient(116deg,#102C40 0%,#075D6B 100%);border-radius:19px;color:white;padding:22px 28px;margin-bottom:17px;}
.port-hero .eyebrow{font-size:.74rem;letter-spacing:2px;color:#80DCDD;font-weight:800;margin-bottom:6px;}
.port-hero h1{color:white;font-size:1.86rem;line-height:1.28;margin:0 0 7px;}
.port-hero p{font-size:.88rem;color:#D3E9EB;line-height:1.58;margin:0;}
.section-label{font-size:1.2rem;font-weight:800;color:#143B50;margin:21px 0 7px;}
.story-note{border-left:4px solid #0B949C;background:#F0F8F8;border-radius:8px;padding:13px 17px;margin:10px 0;line-height:1.65;color:#204355;}
.story-risk{border-left:4px solid #D7726D;background:#FFF6F4;border-radius:8px;padding:13px 17px;margin:10px 0;line-height:1.6;color:#6C4343;}
.muted{font-size:0.80rem;color:#6A8190;line-height:1.62;}
.pill{display:inline-block;background:#ECF5F6;border:1px solid #DBEAEC;border-radius:20px;color:#166B77;font-weight:750;font-size:0.78rem;padding:5px 11px;margin:3px 4px 6px 0;}
.evidence-card{border:1px solid #E1E9EE;border-radius:14px;padding:18px;background:white;margin:12px 0;}
.evidence-card h4{color:#123D51;margin:0 0 10px;font-size:1.08rem;}
.step-tag{background:#E1F3F3;color:#056C73;border-radius:30px;padding:5px 12px;display:inline-block;font-size:0.74rem;font-weight:800;margin-bottom:8px;}
.action-card{border:1px solid #DCE9ED;border-radius:13px;background:#FBFDFE;padding:14px 18px;min-height:134px;margin-bottom:10px;}
.action-card h4{color:#174156;font-size:1.02rem;margin:8px 0 6px;}
.action-card p{color:#587482;font-size:.83rem;line-height:1.6;}
.smalltag{color:#118F96;font-size:.72rem;font-weight:800;letter-spacing:1px;}
.dt-story-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:13px 0 19px;}
.dt-story-panel{border:1px solid #D6E8E8;border-radius:13px;background:#F7FBFB;padding:15px;min-height:140px;}
.dt-story-panel .dt-tag{font-weight:800;font-size:0.75rem;color:#23818B;margin-bottom:7px;}
.dt-story-panel h4{color:#163D4D;font-size:1rem;line-height:1.42;margin:0 0 7px;}
.dt-story-panel p{font-size:.83rem;line-height:1.57;color:#5D7581;margin:0;}
.dt-story-panel:nth-child(2){background:#FFF8F0;border-color:#EEDBC1;}
.dt-story-panel:nth-child(2) .dt-tag{color:#B26B22;}
.dt-story-panel:nth-child(3){background:#EBF6F5;border-color:#B7DDDB;}
.dt-soft-warning{background:#FFF6E8;border-left:4px solid #DC972A;border-radius:8px;padding:12px 15px;color:#7A522A;line-height:1.6;margin:7px 0 12px;}
@media (max-width:1190px){.dt-story-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:720px){.dt-story-grid{grid-template-columns:1fr;}.dt-story-panel{min-height:auto;}}

/* Four step planning roadmap: native HTML, not a Plotly chart. */
.dt-roadmap{
  display:grid;
  grid-template-columns:minmax(0,1fr) 28px minmax(0,1fr) 28px minmax(0,1fr) 28px minmax(0,1fr);
  align-items:stretch;
  column-gap:0;
  margin:14px 0 12px;
  padding:3px 0 6px;
}
.dt-road-card{
  position:relative;display:flex;flex-direction:column;min-width:0;
  border:1px solid #D8E7ED;border-radius:16px;background:#FFFFFF;
  padding:18px 17px 16px;box-sizing:border-box;
  box-shadow:0 5px 16px rgba(18,63,82,.055);
}
.dt-road-card:before{
  content:"";position:absolute;top:0;left:17px;right:17px;height:4px;
  border-radius:0 0 7px 7px;background:#2A89A8;
}
.dt-road-card.ai:before{background:#119C9F}
.dt-road-card.twin{background:linear-gradient(145deg,#133A51,#155B67);border-color:#155B67;box-shadow:0 8px 23px rgba(11,83,103,.17)}
.dt-road-card.twin:before{background:#F6BB64}
.dt-road-card.verify:before{background:#38A688}
.dt-road-top{display:flex;align-items:center;justify-content:space-between;gap:7px;margin:3px 0 15px;}
.dt-road-number{display:inline-flex;align-items:center;justify-content:center;width:37px;height:37px;flex-shrink:0;
  background:#E8F5F8;border-radius:11px;color:#146B88;font-size:1.0rem;font-weight:900;letter-spacing:.015em;}
.dt-road-card.twin .dt-road-number{background:rgba(255,255,255,.14);color:#FFFFFF}
.dt-road-phase{display:inline-block;text-align:right;font-size:.71rem;line-height:1.25;font-weight:800;color:#2F7A82;white-space:normal;}
.dt-road-card.twin .dt-road-phase{color:#F7D69F}
.dt-road-title{font-size:1.03rem;font-weight:850;color:#153B4D;line-height:1.43;word-break:keep-all;margin:0 0 8px;letter-spacing:-.025em;}
.dt-road-card.twin .dt-road-title{color:#FFFFFF}
.dt-road-desc{color:#5B7785;font-size:.82rem;line-height:1.6;word-break:keep-all;margin:0 0 15px;flex:1;}
.dt-road-card.twin .dt-road-desc{color:#D4E7ED}
.dt-road-status{align-self:flex-start;padding:6px 10px;border:1px solid #D3E8EB;border-radius:20px;
  background:#EFF8F9;color:#2E7783;font-size:.7rem;font-weight:750;line-height:1.3;}
.dt-road-card.ai .dt-road-status{background:#E3F6F2;color:#177D68;border-color:#C5E8DA}
.dt-road-card.twin .dt-road-status{background:rgba(248,186,88,.13);color:#F7D59F;border-color:rgba(248,186,88,.35)}
.dt-road-card.verify .dt-road-status{background:#EAF8F1;color:#26765A;border-color:#D2EADD}
.dt-road-arrow{display:flex;justify-content:center;align-items:center;color:#2A9B9E;font-size:1.33rem;font-weight:900;user-select:none;}
.dt-road-footnote{font-size:.79rem;color:#607B89;line-height:1.6;padding:8px 12px;
  background:#F4F9FA;border-left:3px solid #4CAAB2;border-radius:6px;margin:0 0 17px;}
@media(max-width:1150px){
  .dt-roadmap{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;}
  .dt-road-arrow{display:none}
  .dt-road-card{min-height:170px}
}
@media(max-width:680px){
  .dt-roadmap{grid-template-columns:minmax(0,1fr);gap:12px;}
  .dt-road-card{min-height:0;padding:16px 15px;}
  .dt-road-top{margin-bottom:11px}
  .dt-road-desc{margin-bottom:11px}
}

hr{border:none;border-top:1px solid #DCE6EB;margin:1.05rem 0;}
[data-baseweb="tab-list"] button{font-size:1rem;font-weight:750;}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def source_data():
    names=['annual','terminal','weather_reg','weather_first6','weather_hourly_ml','alert_hourly_ml','wind_dose',
        'weather_thresholds','calm_weather_by_month','terminal_operational_defaults','typical_active_by_month_hour',
        'model_2025_test_metrics','model_calibration','weather_coverage','congestion','shift_coverage',
        'validation_severity_2025','validation_weather_factors_2025','validation_availability_2025','validation_calibration_2025',
        'congestion_reference','history_box_annual','history_box_overall','history_weather_training_limits',
        'bct_berth_weather_distribution','weather_scatter_bins',
        'modelb_weather_residual_effects','modelb_2024_prediction_ablation','modelb_2025_prediction_ablation',
        'modelb_weather_shap','modelb_direct_ols','modelb_matched_peers','modelb_2024_holdout',
        'learning_annual_weather_penalty','learning_annual_slopes','learning_pooled_test',
        'learning_calm_2024_vs_2022','learning_sample_coverage','learning_threshold_sensitivity']
    d={n:read_csv(n) for n in names}
    # Public release intentionally excludes vessel-by-vessel and hourly observation rows.
    d['history']=None;d['weather_obs']=None
    return d

@st.cache_resource(show_spinner=False)
def model(use_gt=False):return load_model(use_gt)

D=source_data();META=metadata()
TH=D['weather_thresholds'];MT=D['model_2025_test_metrics']
CONGESTION=D['congestion_reference']


def title(x,sub=None):
    st.markdown(f'<div class="section-label">{x}</div>',unsafe_allow_html=True)
    if sub:st.caption(sub)

def chart(fig):
    st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False,'scrollZoom':True,'responsive':True})

def small_note(txt):st.markdown(f'<p class="muted">{txt}</p>',unsafe_allow_html=True)

def fmt_num(x):return f'{x:,.1f}'

def fmt_delta(x):return f'{x:+.2f}%'

def compute_spread(mod,s,attribute,vals):
    batch=pd.concat([model_row(replace(s,**{attribute:float(v)})) for v in vals],ignore_index=True)
    return np.exp(mod.predict(batch))

def preset_weather(kind,terminal,month):
    harbor=HARBOR[terminal]
    r=TH.set_index('harbor').loc[harbor]
    calm=calm_for(D['calm_weather_by_month'],terminal,month)
    w=calm['wind']; wave=calm['wave']; vis=calm['visibility_m']
    if kind=='P90 강풍':w=math.ceil(float(r.wind_P90)*10)/10
    elif kind=='P95 강풍':w=math.ceil(float(r.wind_P95)*10)/10
    elif kind=='P95 고파고':wave=math.ceil(float(r.wave_P95)*20)/20
    elif kind=='P95 저시정':vis=math.floor(float(r.vis_10m_P95*10)/100)*100
    return float(round(w,1)),float(round(wave*20)/20),int(round(vis/100)*100)

def apply_preset(kind,t,m):
    w,wa,v=preset_weather(kind,t,m)
    st.session_state['wx_wind']=w
    st.session_state['wx_wave']=wa
    st.session_state['wx_visibility']=v
    st.session_state['wx_preset']=kind

# 2027 default is a model-verified >5% decline scenario, NOT an actual forecast.
# Apply once per Streamlit session or after upgrading an older live session.
if 'initialized_default_2027_v28' not in st.session_state:
    st.session_state['wx_wind']=6.0
    st.session_state['wx_wave']=1.2
    st.session_state['wx_visibility']=5000
    st.session_state['wx_preset']='2027년 가상 악천후 기본 시나리오'
    st.session_state['initialized_default_2027_v28']=True
if 'story_step' not in st.session_state:st.session_state['story_step']='01 · 현황'

st.markdown('''<div class="port-hero"><div class="eyebrow">PORTCAST / WEATHER × OPERATIONS × LEARNING</div>
<h1>날씨를 예측하는 AI에서, 악천후 대응을 학습하는 항만 AI로</h1>
<p>① 조건별 평균 작업률 예측·SHAP → ② 작업량 보정 모델 B → ③ 악천후 대응 추세 검증 → ④ 디지털 트윈 반복훈련 제안</p></div>''',unsafe_allow_html=True)
info1,info2,info3,info4=st.columns(4)
info1.metric('7개 터미널 분석','56,791건','2020–2025 비GT 운영표본',delta_color='off')
info2.metric('2025 CatBoost 외부검증','R² 0.825','미학습 평가 7,664건',delta_color='off')
info3.metric('BCT 모델 B · 강풍 4–6회','−10.74%','조정 연관성 · 인과손실 아님',delta_color='off')
info4.metric('BCT 악천후 개선 추세','미확인','2022–24년 p=0.794',delta_color='off')
st.caption('현재 구현: 조건부 예측·SHAP·악천후별 성능 검증·모델 B·연도별 추세 탐색. 제안 단계: 디지털 트윈 학습·운영 개선 효과 실증. 기항 평균 작업률은 실제 크레인 시간당 Moves와 다릅니다.')

T1,T2,T6,T3,T4,T5=st.tabs(['01 ⚡ AI 즉시 예측','02 🔎 7터미널 기상영향','03 🧭 모델 B · 운영학습','04 🧩 디지털 트윈 제안','05 ✅ 실증 · 모델 신뢰도','06 🧪 SHAP · 악천후별 검증'])

with T1:
    st.markdown('<span class="pill">사용자 조작 즉시 재계산</span><span class="pill">학습 완료 CatBoost</span><span class="pill">P90 / P95 기준 비교</span>',unsafe_allow_html=True)
    prediction_slot=st.container()  # Fill after calculating; Streamlit renders this container above all filters.
    title('예측 조건 변경', '날짜·시간·터미널·날씨·혼잡도를 선택하면 상단 AI 예측값이 즉시 재계산됩니다. 모든 시간은 한국시간(KST) 기준입니다.')
    sel1,sel2,sel3=st.columns([1.1,1,1.05])
    with sel1:when_day=st.date_input('① 날짜',date(2027,11,20),min_value=date(2020,1,1),max_value=date(2030,12,31),key='executive_day_2027')
    with sel2:hh=st.selectbox('② 시간',list(range(24)),index=10,format_func=lambda x:f'{x:02d}:00',key='executive_hour_2027')
    with sel3:terminal=st.selectbox('③ 터미널',['BCT','BNCT','HPNT','PNC','PNIT','BPTC_G','BPT_S'],key='executive_terminal_2027')
    when=datetime.combine(when_day,datetime.min.time()).replace(hour=hh)
    harbor=HARBOR[terminal]
    th=TH.set_index('harbor').loc[harbor]
    calm=calm_for(D['calm_weather_by_month'],terminal,when.month)
    # Hourly weather lookups are disabled in the public-safe build. No timestamp-level records are redistributed.
    obs=None
    title('④ 기상조건 설정',f'{harbor} 지역 임계값: 강풍 P90 {th.wind_P90:g} m/s · P95 {th.wind_P95:g} m/s | 고파고 P90 {th.wave_P90:g} m · P95 {th.wave_P95:g} m')
    st.markdown('**원클릭 시나리오**')
    bcols=st.columns(5)
    for col,kind in zip(bcols,['평온','P90 강풍','P95 강풍','P95 고파고','P95 저시정']):
        with col:st.button(kind,use_container_width=True,key=f'btn_{kind}',on_click=apply_preset,args=(kind,terminal,when.month))
    wxmode=st.radio('기상값 출처',options=['직접 지정 / 위 예시'],horizontal=True,key='wx_mode_2027',help='공개 배포본에는 개별 날짜별 기상원시 관측을 포함하지 않습니다. 모두 사용자 입력 시나리오입니다.')
    observed_usable=wxmode=='과거 시각 관측값' and obs is not None and not obs.get('incomplete',False)
    if observed_usable:
        wind=float(obs['wind']);wave=float(obs['wave']);visibility=float(obs['visibility_m'])
        st.info('해당 시각에 기록된 기상값을 적용했습니다. '+describe_source(obs))
        ww1,ww2,ww3=st.columns(3)
        ww1.metric('풍속',f'{wind:.2f} m/s');ww2.metric('유의파고',f'{wave:.2f} m');ww3.metric('시정',f'{visibility:,.0f} m')
    else:
        if wxmode=='과거 시각 관측값':st.warning('선택 시각에 세 기상요소의 완전 관측이 없습니다. 누락값을 임의로 보간하지 않고 아래 수동 시나리오를 적용합니다.')
        wc1,wc2,wc3=st.columns(3)
        with wc1:wind=st.slider('풍속 (m/s)',0.,18.,step=.1,key='wx_wind')
        with wc2:wave=st.slider('유의파고 (m)',0.,5.,step=.05,key='wx_wave')
        with wc3:visibility=st.slider('시정 (m)',0,50000,step=100,key='wx_visibility')
    source_label=(describe_source(obs) if observed_usable else '사용자 지정 날씨 시나리오')
    dd=defaults_for(D['terminal_operational_defaults'],terminal,when.month)
    occ=occupancy_for(D['typical_active_by_month_hour'],terminal,when.month,when.hour)
    title('⑤ 혼잡도 선택','모델 입력: 직전 6시간 접안 유입 척수. 상·중·하는 2020~2024년의 터미널별 관측분포로 산정합니다.')
    crowd_choice=st.radio('혼잡도 수준',options=['하 (낮음)','중 (보통)','상 (높음)','직접 입력'],index=2,horizontal=True,key='congestion_level_2027',help='각 수준은 터미널별 역사적 Q25 / 중앙값 / Q80에 가까운 정수 유입 척수입니다. 동시 접안 선박 수와는 별개입니다.')
    crowd_key={'하 (낮음)':'하','중 (보통)':'중','상 (높음)':'상'}.get(crowd_choice)
    reference=CONGESTION.loc[CONGESTION.terminal==terminal].iloc[0]
    if crowd_key is None:
        arrivals=st.number_input('직전 6시간 실제/가정 접안 유입(척)',min_value=0,max_value=50,value=int(reference.middle),step=1,key=f'custom_congestion_{terminal}')
        crowd_name='직접 입력'
    else:
        arrivals=congestion_count(CONGESTION,terminal,crowd_key)
        crowd_name=crowd_key+' (선택)'
    st.markdown(f'<div class="selected-congestion"><b>{terminal} · 혼잡도 {crowd_name}</b> → 모델 입력 <b>직전 6시간 접안 {int(arrivals)}척</b> &nbsp;│&nbsp; 이 터미널 대표값: 하 {int(reference.low)}척 / 중 {int(reference.middle)}척 / 상 {int(reference.high)}척</div>',unsafe_allow_html=True)
    with st.expander('고급 옵션 · 대표 선박 작업량 / 동시접안 척수 / GT',expanded=False):
        u1,u2=st.columns(2)
        with u1: cargo=st.number_input('대표 선박 총작업량(원자료 단위)',min_value=10,max_value=15000,value=max(10,int(round(dd.median_cargo))),step=10,key=f'cargo_{terminal}_{when.month}')
        with u2:active=st.number_input('동시 접안 선박 수 가정',min_value=1,max_value=30,value=occ,step=1,key=f'active_{terminal}_{when.month}_{when.hour}')
        gt_allowed=2020<=when.year<=2024
        use_gt=st.checkbox('2020~2024 GT 통제 역사모형 사용',value=False,disabled=not gt_allowed,key='enable_gt',help='GT 모형은 2025년 GT 외부 검증이 불가능합니다.')
        use_gt=bool(use_gt and gt_allowed)  # Never apply GT to 2025+ even with stale UI widget state.
        gt=None
        if use_gt:
            gt=st.number_input('선박 총톤수(GT)',min_value=100,max_value=250000,value=int(round(META['gt_defaults'][terminal]/100))*100,step=100,key=f'gt_{terminal}')
        st.caption('작업량과 동시접안척수는 터미널·월·시간대의 대표값입니다. 실제 계획을 아는 경우 수정하세요. 혼잡 수준(6시간 유입)은 상단에서 바꿉니다.')
    if not use_gt:gt=None
    cur=Scenario(terminal,when,float(wind),float(wave),float(visibility),float(cargo),float(arrivals),int(active),float(gt) if gt else None)
    m=model(use_gt)
    result=estimate(m,cur,calm,use_gt)
    level,flags=category(cur,TH)
    risk_text={'HIGH':'P95 강도 도달','WATCH':'P90 강도 도달','NORMAL':'P90 미만'}[level]
    row_metric=MT[MT.terminal==terminal].iloc[0]
    low,high=error_range(result['vessel'],terminal,MT)
    change=float(result['delta_pct'])
    drop=max(0.,-change)
    actual_diff=float(result['aggregate']-result['aggregate_normal'])
    if change < -0.005:
        focus_class='focus-drop'
        trend_label='평온 대비 예상 작업률 하락'
        trend_value=f'▼ {drop:.2f}%'
        trend_sub=f'평온 대비 {abs(actual_diff):,.1f} 작업단위/h 낮게 예측'
    elif change > 0.005:
        focus_class='focus-rise'
        trend_label='평온 대비 예상 작업률 변화'
        trend_value=f'▲ +{change:.2f}%'
        trend_sub=f'평온 대비 {abs(actual_diff):,.1f} 작업단위/h 높게 예측 · 인과적 개선 아님'
    else:
        focus_class='focus-flat'
        trend_label='평온 대비 예상 작업률 변화'
        trend_value='0.00%'
        trend_sub='평온 기준 대비 모델상 차이 없음'
    with prediction_slot:
        st.markdown(f'''<div class="predict-lead"><strong>⚡ AI 예상 처리효율 · 즉시 예측 결과</strong>
         <span>{terminal} &nbsp;│&nbsp; {when.strftime('%Y-%m-%d %H:00')} &nbsp;│&nbsp; 혼잡도 {crowd_name} ({int(arrivals)}척/6h) &nbsp;│&nbsp; {risk_text}</span></div>
         <div class="executive-kpis">
          <div class="exec-kpi"><div class="kpi-label">터미널 합산 예상 시간당 처리량</div>
           <div class="kpi-value">{result['aggregate']:,.1f}<small>작업단위/h</small></div>
           <div class="kpi-caption">선박당 예측치 × 동시 접안 {result['active_ships']}척 가정</div></div>
          <div class="exec-kpi {focus_class}"><div class="kpi-label">{trend_label}</div>
           <div class="kpi-value">{trend_value}</div><div class="kpi-caption">{trend_sub}</div></div>
          <div class="exec-kpi"><div class="kpi-label">같은 운영조건의 평온 기준</div>
           <div class="kpi-value">{result['aggregate_normal']:,.1f}<small>작업단위/h</small></div>
           <div class="kpi-caption">날씨만 평온조건으로 변경한 모델 비교</div></div>
         </div>
         <div class="selected-summary"><div class="summary-title">📌 현재 선택한 예측 조건 · 변경 즉시 적용</div>
          <div class="summary-grid">
           <span class="summary-chip"><b>날짜·시간</b> {when.strftime('%Y-%m-%d %H:00')} KST</span>
           <span class="summary-chip"><b>터미널</b> {terminal}</span>
           <span class="summary-chip"><b>풍속</b> {wind:.1f} m/s</span>
           <span class="summary-chip"><b>파고</b> {wave:.2f} m</span>
           <span class="summary-chip"><b>시정</b> {visibility:,.0f} m</span>
           <span class="summary-chip"><b>혼잡도</b> {crowd_name} · 6시간 {int(arrivals)}척</span>
           <span class="summary-chip"><b>대표 작업량</b> {cargo:,.0f} 단위</span>
           <span class="summary-chip"><b>동시 접안</b> {int(active)}척</span>
           <span class="summary-chip"><b>기상 출처</b> {source_label}</span>
           <span class="summary-chip risk">{'미래 시나리오 · 실제 예보 아님' if when.year>=2026 else '조건부 예측 · 인과관계 아님'}</span>
          </div>
         </div>
         <div class="predict-footnote">이 값은 <b>선박별 총작업량 ÷ 전체 접안 후 체류시간</b>을 예측해 선박 수를 곱한 <b>시나리오 추정치</b>입니다. 실제 특정 1시간의 실적·TEU/h·크레인 Moves/h가 아니며, 변화율은 확정적 인과손실이 아닙니다.</div>''',unsafe_allow_html=True)
        p1,p2,p3=st.columns(3)
        p1.metric('선박당 평균 예측 처리량',f'{result["vessel"]:,.1f} /h',delta=f'평온 {result["normal"]:,.1f} /h',delta_color='off')
        p2.metric('동시 접안 선박 수 가정',f'{result["active_ships"]}척',delta=f'최근 6시간 유입 {int(arrivals)}척 · 혼잡 {crowd_name}',delta_color='off')
        p3.metric('2025년 검증 R²',f'{row_metric.r2:.3f}' if not use_gt else 'GT 별도',delta=f'평가 {int(row_metric.n):,}건' if not use_gt else '2025 외부검증 없음',delta_color='off')
    st.markdown('<div class="section-label">AI 예측 해석과 검증</div>',unsafe_allow_html=True)
    st.markdown('<div class="story-note"><b>어떻게 계산되나요?</b> 선박별 <b>총작업량 ÷ 전체 접안 후 체류시간</b>을 예측해 평균 작업단위/h로 표시합니다. 터미널 값은 여기에 <b>동시 접안 선박 수(가정)</b>를 곱한 값입니다. <b>실제 시간별 하역실적·TEU/h·크레인 Moves/h가 아닙니다.</b></div>',unsafe_allow_html=True)
    title('선택한 예측의 SHAP 설명 · 실제 CatBoost 계산',
          'SHAP은 예측모델의 출력값(log 평균작업률)을 각 입력변수에 배분해 설명합니다. 날씨의 인과효과나 운영 노하우의 검증 결과는 아닙니다.')
    explain=explain_prediction(m,cur,use_gt)
    scope_shap=st.radio('기여도 표시', ['전체 변수','기상·혼잡 변수 확대'],horizontal=True,key='shap_scope',
                        help='동일한 SHAP 값을 전체 또는 일부 변수로 확대해 보여줍니다. 값 자체는 바뀌지 않습니다.')
    explanation=explain['table'].copy()
    if scope_shap=='기상·혼잡 변수 확대':
        explanation=explanation[explanation.feature.isin(['wind','wave','vis_10m','arrival_prev6'])]
    explanation=explanation.sort_values('shap_log')
    left,right=st.columns([1.28,.72])
    with left:
        fig=go.Figure(go.Bar(
            x=explanation.shap_log.to_numpy(),y=explanation.label.to_numpy(),orientation='h',
            marker_color=['#CD6C5B' if v<0 else '#078E8F' for v in explanation.shap_log],
            customdata=np.column_stack((explanation.value.to_numpy(),explanation.multiplier_pct.to_numpy())),
            hovertemplate='<b>%{y}</b><br>입력값: %{customdata[0]}<br>SHAP: %{x:+.4f} log<br>해당 변수 기여의 배수 환산: %{customdata[1]:+.2f}%<extra></extra>'
        ))
        fig.add_vline(x=0,line_color='#637D8B',line_width=1)
        fig.update_layout(title='예측 증가(+청록) / 감소(−주황) 기여도',
                          xaxis_title='SHAP 값 · log(평균 작업률) 단위',yaxis_title='',
                          height=370 if scope_shap=='전체 변수' else 300,showlegend=False,
                          margin=dict(t=55,b=45,l=12,r=14))
        chart(fig)
    with right:
        st.metric('전체 학습 기준 SHAP 평균',f'{explain["base_rate"]:,.1f} /h',
                  help='모든 터미널·선박의 학습 분포를 바탕으로 한 CatBoost SHAP 기준값. 평온한 날의 기준값이 아닙니다.')
        st.metric('선택 조건의 SHAP 재구성 예측',f'{explain["pred_rate"]:,.1f} /h',
                  delta=f'모델 예측과 오차 {explain["reconstruction_error"]:.1e}',delta_color='off')
        st.caption('SHAP 합산식: 기준 log 예측 + 입력별 SHAP 합 = 현재 log 예측. ' +
                   '막대에 보인 배수 환산(%)은 서로 합산할 수 없습니다.')
    table=explain['table'][['label','value','shap_log','multiplier_pct','direction']].copy()
    st.dataframe(table.rename(columns={'label':'입력 변수','value':'선택 입력값','shap_log':'SHAP(log)','multiplier_pct':'해당 기여 배수 환산(%)','direction':'모델상 방향'}).round(4),
                 hide_index=True,use_container_width=True)
    st.caption('SHAP 기준 평균은 전체 학습분포의 평균 예측이므로 상단의 "평온조건 대비 −%"와 기준이 다릅니다. ' +
               '특히 총작업량·터미널·계절 등 입력 변수의 SHAP은 관측데이터에서 학습한 연관성입니다.')
    with st.expander('심화: 같은 운영조건에서 날씨만 평온값으로 바꾼 SHAP 차이',expanded=False):
        cmp=compare_weather_attributions(m,cur,calm,use_gt)
        st.metric('평온 → 선택 기상의 CatBoost 예측 변화',f'{cmp["scenario_delta_pct"]:+.2f}%',
                  delta=f'SHAP log 차이 검산오차 {cmp["error"]:.1e}',delta_color='off')
        diff=cmp['table'].copy()
        diff['입력 변동']=diff.changed_input.map({True:'직접 변경(기상)',False:'고정(상호작용 기여 재배분 가능)'})
        st.dataframe(diff[['label','delta_shap_log','delta_multiplier_pct','입력 변동']].rename(
            columns={'label':'변수','delta_shap_log':'SHAP 차이(log)','delta_multiplier_pct':'배수 환산(%)'}).round(4),
            use_container_width=True,hide_index=True)
        st.caption('기상 3개 입력만 바뀌어도 모델의 상호작용 때문에 고정된 다른 변수의 SHAP 기여도가 달라질 수 있습니다. ' +
                   '전체 log 차이의 합만 모델 시나리오 예측과 일치하며, 실제 기상의 인과 효과 분해가 아닙니다.')
    st.download_button('현재 예측 SHAP 설명 CSV 저장',
                       table.to_csv(index=False).encode('utf-8-sig'),
                       file_name='부산항_SHAP_선택예측_설명.csv',mime='text/csv')
    if not use_gt:small_note(f'2025년 미학습 표본의 과거 예측오차를 적용한 선박당 경험적 80% 범위: 약 <b>{low:,.1f}~{high:,.1f} 작업단위/h</b>. 정확한 예측구간이나 전체 터미널 합산의 오차범위는 아닙니다.')
    if terminal=='BCT':
        st.caption('BCT 참고: 별도 GT 통제 회귀(2020~2024)에서 첫 6시간 중 강풍 P90 4~6시간 노출과 체류시간 +8.67%의 연관성이 나타났습니다. 이 화면의 단일시점 풍속 기반 ML과 **비교조건이 다르므로 두 수치를 합산하거나 직접 대체하지 않습니다.** 「왜 BCT인가?」 탭에서 근거를 확인하세요.')
    if when.year>=2026:st.warning('2026년 이후는 학습 자료 밖의 미래 시나리오입니다. 예보 API에 연결되어 있지 않으므로 선택 날씨를 실제·예보 날씨라고 해석할 수 없습니다.')
    if use_gt:st.warning('GT 역사모형은 2020~2024년만 학습했으며 2025년 독립 검증 수치를 제공하지 않습니다.')
    q99=D['history_weather_training_limits'].set_index('harbor').loc[harbor]
    if wind>max(9.,1.25*float(q99.wind_p99)) or wave>max(2.,1.3*float(q99.wave_p99)):
        st.warning('선택한 풍속 또는 파고가 학습 관측분포의 극단부입니다. 모델의 외삽 예측은 특히 불확실합니다.')
    lc,rc=st.columns([1.08,.92])
    with lc:
        fig=go.Figure(go.Bar(x=['평온 기준','선택한 날씨'],y=[result['aggregate_normal'],result['aggregate']],
                     marker_color=[BLUE,CORAL if result['delta_pct']<0 else AQUA],
                     text=[f'{result["aggregate_normal"]:,.1f}',f'{result["aggregate"]:,.1f}'],textposition='outside',
                     hovertemplate='%{x}: %{y:,.1f} 작업단위/h<extra></extra>'))
        fig.update_layout(title='같은 작업조건에서 날씨만 바꾸면?',yaxis_title='터미널 합산 가정 · 작업단위/h',showlegend=False)
        fig.update_yaxes(range=[0,max(result['aggregate'],result['aggregate_normal'])*1.22])
        chart(styled(fig,345))
    with rc:
        st.markdown('<div class="evidence-card"><h4>이 숫자를 어떻게 읽나요?</h4>'+\
          f'<p><b>선박당 예상치</b> {result["vessel"]:,.1f}/h × <b>동시 접안</b> {result["active_ships"]}척 = <b>터미널 가정 합산</b> {result["aggregate"]:,.1f}/h</p>'+\
          f'<p>평온조건의 동일 운영 입력으로 다시 예측하면 {result["aggregate_normal"]:,.1f}/h → <b>{result["delta_pct"]:+.2f}%</b></p>'+\
          '<p class="muted">이 비교는 학습된 통계적 반응이며 확정된 기상 인과손실이 아닙니다. 작업량 단위는 원자료 단위를 따릅니다.</p></div>',unsafe_allow_html=True)
        if result['delta_pct']<0: st.info('선택 조건에서 평온 대비 낮은 예측치입니다. 운영대응 검토 대상으로 표시할 수 있지만 실제 장비중단을 뜻하지 않습니다.')
        else:st.info('선택 조건에서 평온 대비 증가 또는 유지 방향입니다. 악천후가 생산성을 높인다는 인과결론으로 해석하지 않습니다.')
    title('기상 강도를 바꾸면 예측값이 어떻게 움직이나?','각 그래프는 다른 날씨·작업조건을 고정한 채 한 변수만 바꾸는 민감도 곡선입니다. 선박당 예상치 기준입니다.')
    ss1,ss2=st.columns(2)
    with ss1:
        values=np.linspace(0,max(11.,float(th.wind_P95)*2.2),40)
        if use_gt:
            batch=pd.concat([model_row(replace(cur,wind=float(x)),True) for x in values],ignore_index=True)
            pred=np.exp(m.predict(batch))
        else:pred=compute_spread(m,cur,'wind',values)
        ff=go.Figure(go.Scatter(x=values,y=pred,mode='lines',line=dict(color=AQUA,width=3),name='ML 시나리오'))
        ff.add_vline(x=th.wind_P90,line_dash='dot',line_color=BLUE,annotation_text='P90')
        ff.add_vline(x=th.wind_P95,line_dash='dot',line_color=CORAL,annotation_text='P95')
        ff.add_vline(x=wind,line_color='#17364A',annotation_text='선택값')
        ff.update_layout(title='풍속에 따른 예측 평균 처리량',xaxis_title='풍속 (m/s)',yaxis_title='선박당 작업단위/h')
        chart(styled(ff,320))
    with ss2:
        values=np.linspace(0,max(2.5,float(th.wave_P95)*2.3),40)
        if use_gt:
            batch=pd.concat([model_row(replace(cur,wave=float(x)),True) for x in values],ignore_index=True)
            pred=np.exp(m.predict(batch))
        else:pred=compute_spread(m,cur,'wave',values)
        ff=go.Figure(go.Scatter(x=values,y=pred,mode='lines',line=dict(color=BLUE,width=3),name='ML 시나리오'))
        ff.add_vline(x=th.wave_P90,line_dash='dot',line_color=AQUA,annotation_text='P90')
        ff.add_vline(x=th.wave_P95,line_dash='dot',line_color=CORAL,annotation_text='P95')
        ff.add_vline(x=wave,line_color='#17364A',annotation_text='선택값')
        ff.update_layout(title='유의파고에 따른 예측 평균 처리량',xaxis_title='파고 (m)',yaxis_title='선박당 작업단위/h')
        chart(styled(ff,320))
    with st.expander('동일한 기상조건을 다른 6개 터미널에 적용하면? · 7터미널 동시 비교',expanded=False):
        comps=[]
        for t in TERMINALS:
            dflt=defaults_for(D['terminal_operational_defaults'],t,when.month)
            n=occupancy_for(D['typical_active_by_month_hour'],t,when.month,when.hour)
            ct=congestion_count(CONGESTION,t,crowd_key) if crowd_key is not None else int(arrivals)
            ss=Scenario(t,when,cur.wind,cur.wave,cur.visibility_m,float(dflt.median_cargo),float(ct),int(n),float(META['gt_defaults'][t]) if use_gt else None)
            r=estimate(m,ss,calm_for(D['calm_weather_by_month'],t,when.month),use_gt)
            comps.append({'terminal':t,'per_vessel':r['vessel'],'arrival_prev6':ct,'active_ships':n,'aggregate':r['aggregate'],'delta_pct':r['delta_pct']})
        comp=pd.DataFrame(comps)
        chart(responsiveness_bar(comp))
        st.dataframe(comp.rename(columns={'terminal':'터미널','per_vessel':'선박당 예상 작업량/h','arrival_prev6':'선택 혼잡 · 최근6h 유입','active_ships':'동시 접안 가정','aggregate':'터미널 합산 추정/h','delta_pct':'평온 대비 %'}).round(2),hide_index=True,use_container_width=True)
        st.caption('선택한 혼잡도 등급은 각 터미널의 2020~2024년 역사적 유입 분포로 환산했습니다(직접 입력 시 같은 유입 척수 적용). 터미널별 작업량·동시접안 대표값이 다르며, 합산 추정치는 실제 터미널 처리능력 순위가 아닙니다.')
    st.markdown('---')
    output=pd.DataFrame([{'date_time_KST':when.strftime('%Y-%m-%d %H:00'),'terminal':terminal,'weather_source':source_label,
       'wind_mps':wind,'wave_m':wave,'visibility_m':visibility,'cargo_units':cargo,'congestion_level':crowd_choice,'arrivals_prev6':arrivals,'active_vessels_assumed':active,
       'model_type':'GT_2020_2024' if use_gt else 'nonGT_2020_2024','estimated_vessel_rate_per_berth_h':result['vessel'],
       'estimated_terminal_scenario_per_h':result['aggregate'],'calm_comparison_per_h':result['aggregate_normal'],
       'relative_to_calm_pct':result['delta_pct'],'not_actual_intra_hour_moves':True}])
    st.download_button('선택 예측 결과 CSV 저장',output.to_csv(index=False).encode('utf-8-sig'),'부산항_7터미널_예측시나리오.csv',mime='text/csv')

STAGES=['01 · 현황','02 · 기상 기준','03 · 7터미널 비교','04 · BCT 심층','05 · 혼잡·GT 통제','06 · 머신러닝 검증','07 · 개선안 도출']


def change_story_step(delta):
    i=STAGES.index(st.session_state['story_step'])
    st.session_state['story_step']=STAGES[max(0,min(len(STAGES)-1,i+delta))]

with T2:
    st.markdown('<span class="pill">강풍·운영 조건 통제</span><span class="pill">2022~24 개선추세 p=0.910</span><span class="pill">자동화는 검토 요인</span>',unsafe_allow_html=True)
    title('왜 BCT를 우선 파일럿 대상으로 제안하는가?','7개 터미널 비교 → 장시간 강풍 노출 → GT·혼잡 통제 → 개선속도 비교 → 운영학습 가설과 디지털 트윈 실증 순서로 검토합니다.')
    st.markdown('<div class="story-note"><b>분석 결론을 먼저 알려드립니다.</b> BCT는 모든 기상지표에서 가장 취약한 터미널로 판정된 것이 아닙니다. BNCT와 북항에서도 강한 신호가 관측됐습니다. 다만 BCT는 장시간 강풍 관련 악화와 과거 큰 효율저하 위험 신호가 있어 <b>운영 대응 학습을 실증할 대상</b>으로 제안합니다.</div>',unsafe_allow_html=True)
    st.select_slider('근거를 순서대로 이동',options=STAGES,key='story_step')
    st.progress((STAGES.index(st.session_state['story_step'])+1)/len(STAGES),text=f'{STAGES.index(st.session_state["story_step"])+1} / {len(STAGES)} 단계')
    p1,p2,p3=st.columns([1,2,1])
    with p1:st.button('◀ 이전 근거',key='story_prev',on_click=change_story_step,args=(-1,),use_container_width=True,disabled=(st.session_state['story_step']==STAGES[0]))
    with p3:st.button('다음 근거 ▶',key='story_next',on_click=change_story_step,args=(1,),use_container_width=True,disabled=(st.session_state['story_step']==STAGES[-1]))
    stage=STAGES.index(st.session_state['story_step'])

    if stage==0:
        st.markdown('<span class="step-tag">STEP 01 · 7개 터미널을 먼저 공정하게 비교</span>',unsafe_allow_html=True)
        title('작업률 수준과 선박 구성이 다릅니다','2020~2025년 기항 기록의 단순 분포. 모든 터미널에 동일하게 접근하며, 단순 작업률 순위는 날씨의 원인이 아닙니다.')
        cols=st.columns([1.75,.8]);ytarget=None
        with cols[1]:
            target=st.selectbox('연도별 보기',['전체 기간',2020,2021,2022,2023,2024,2025],key='story_year')
            st.metric('GT 포함 분석기간','2020~2024')
            st.metric('GT 미포함 운영기간','2020~2025')
            st.info('작업규모·선박 GT·계절·혼잡 구조가 다르므로 같은 기준으로 회귀 조정한 결과가 필요합니다.')
        with cols[0]:
            rate_summary=(D['history_box_overall'] if target=='전체 기간' else D['history_box_annual'].query('year == @target'))
            fig=go.Figure(go.Bar(x=rate_summary.terminal,y=rate_summary.median_rate,
                marker_color='#278AA0',error_y=dict(type='data',symmetric=False,
                array=(rate_summary.p75-rate_summary.median_rate).to_numpy(),
                arrayminus=(rate_summary.median_rate-rate_summary.p25).to_numpy()),
                customdata=rate_summary.n,hovertemplate='%{x}<br>중앙 작업률 %{y:.1f}/h<br>표본수 %{customdata:,.0f}<extra></extra>'))
            fig.update_layout(title='터미널별 중앙 작업률 + 25~75% 범위 (집계자료)',height=385,
                yaxis_title='선박당 접안 후 처리효율 · 작업단위/h',showlegend=False)
            chart(fig)
        term=D['terminal'][['terminal','calls','median_berth_h','median_rate','median_cargo']].sort_values('calls',ascending=False)
        st.dataframe(term.rename(columns={'terminal':'터미널','calls':'기항 수','median_berth_h':'중앙 체류시간(h)','median_rate':'중앙 평균 작업량/h','median_cargo':'중앙 작업량'}).round(1),hide_index=True,use_container_width=True)
        st.markdown('<div class="story-note"><b>다음 단계의 질문:</b> 어떤 터미널에서 악천후가 어느 정도 발생했는지, 같은 기준(P90·P95)으로 정의했는가?</div>',unsafe_allow_html=True)

    elif stage==1:
        st.markdown('<span class="step-tag">STEP 02 · 순간 기상보다 강도와 지속시간</span>',unsafe_allow_html=True)
        title('북항·신항 기상분포를 각각 기준으로 설정','풍속과 파고는 각 항만의 상위 10%(P90) / 5%(P95). 시정은 작을수록 나쁘므로 하위 10% / 5%로 정의합니다.')
        a,b=st.columns([1.2,.8])
        with a:chart(threshold_chart(TH))
        with b:
            t=TH.set_index('harbor').loc['신항']
            st.markdown(f'''<div class="evidence-card"><h4>BCT는 신항 기준을 사용</h4>
            <p>풍속 P90 <b>{t.wind_P90:.2f}m/s</b> · P95 <b>{t.wind_P95:.2f}m/s</b></p>
            <p>유의파고 P90 <b>{t.wave_P90:.2f}m</b> · P95 <b>{t.wave_P95:.2f}m</b></p>
            <p>저시정 P90 <b>{t.vis_10m_P90*10:,.0f}m 이하</b> · P95 <b>{t.vis_10m_P95*10:,.0f}m 이하</b></p>
            <p class="muted">2020~2024년 시간별 관측으로 산정한 고정 기준입니다. 물리적 안전정지 기준은 아닙니다.</p></div>''',unsafe_allow_html=True)
        st.markdown('**지속시간을 따로 보는 이유**')
        st.markdown('<div class="story-note">풍속 5m/s를 한순간 경험한 선박과 접안 이후 6시간 중 5시간 경험한 선박은 같은 조건으로 볼 수 없습니다. 그래서 <b>접안 후 첫 6시간</b>에 P90 기준을 넘긴 횟수를 0h·1–3h·4–6h로 나눠 분석했습니다. 고정 관측창은 체류가 긴 선박이 기상을 더 많이 경험하는 기회편향을 완화합니다.</div>',unsafe_allow_html=True)
        st.caption('2025년 기상은 일부 선박 이벤트 시각에만 복원됐습니다. 2020~2024년처럼 매시간 연속 관측된 것으로 가정하지 않습니다.')
        with st.expander('실제 풍속·파고 관측 점 분포 보기 · Plotly 상호작용',expanded=False):
            weather_points=D['weather_scatter_bins'].copy()
            fig=px.scatter(weather_points,x='wind_bin',y='wave_bin',size='n',color='harbor',
                color_discrete_map={'신항':AQUA,'북항':CORAL},size_max=21,opacity=.55,
                title='시간별 기상의 0.5m/s × 0.1m 빈도집계 (개별 시각 미공개)',
                labels={'wind_bin':'풍속 범주 (m/s)','wave_bin':'유의파고 범주 (m)','harbor':'항만','n':'관측 수'})
            chart(styled(fig,350))

    elif stage==2:
        st.markdown('<span class="step-tag">STEP 03 · GT 통제 7터미널 정량 비교</span>',unsafe_allow_html=True)
        title('장시간 강풍 때 체류시간이 얼마나 달랐나?','2020~2024년 GT 포함 · 첫 6시간 중 P90 강풍 4~6시간 vs 0시간 · 7개 터미널 FDR 보정')
        a,b=st.columns([1.55,.65])
        with a:chart(dose_forest(D['wind_dose']))
        with b:
            st.metric('BCT · 체류시간 변화','+8.67%','FDR q=0.0024',delta_color='off')
            st.metric('BNCT · 체류시간 변화','+8.50%','같은 분석의 유사 규모',delta_color='off')
            st.markdown('<div class="story-risk"><b>이 그래프의 해석:</b> BCT만 특별히 나쁜 것이 아니라, 여러 터미널에서 조정 체류시간 증가 신호가 있습니다. 따라서 <b>BCT를 7터미널 중 유일한 피해 항만</b>이라고 표현하면 안 됩니다.</div>',unsafe_allow_html=True)
        st.caption('점은 회귀 추정치, 선은 95% 신뢰구간. 인과적 정지시간이나 선박당 실제 추가 체류시간을 뜻하지 않습니다.')
        st.markdown('<div class="story-note"><b>다음 질문:</b> BCT 내부에서 강풍 노출이 짧을 때와 길 때 반응이 어떻게 달라지는가?</div>',unsafe_allow_html=True)

    elif stage==3:
        st.markdown('<span class="step-tag">STEP 04 · BCT 장시간 강풍 심층 검증</span>',unsafe_allow_html=True)
        title('단순 악천후 여부보다 노출 지속시간에서 신호가 커집니다','첫 6시간 강풍 P90 노출 5단계. 0시간을 기준으로 GT·작업량·운영조건을 회귀 통제했습니다.')
        chart(bct_five_stage(D['wind_dose']))
        x1,x2,x3=st.columns(3)
        x1.metric('최신 분석 · 4~6h 강풍','+8.67%','조정 체류시간 변화',delta_color='off')
        x2.metric('기존 심층분석 v4','−9.76%','조정 평균 작업률 변화',delta_color='off')
        x3.metric('기존 v4 꼬리위험','2.30×','10% 이상 작업률 저하 오즈비',delta_color='off')
        st.markdown('<div class="story-risk"><b>중요:</b> 위 세 수치는 <b>분석표본·결과변수·비교조건이 서로 다른 별도 결과</b>입니다. +8.67%와 −9.76%는 동일 모형에서 나온 역수 관계로 간주하지 않으며, 2.30배는 사건 발생 <b>확률</b>이 아니라 <b>오즈비</b>입니다. 4시간이 정확한 물리적 임계점이라고 증명된 것도 아닙니다.</div>',unsafe_allow_html=True)
        st.info('강풍이 얼마나 지속됐는지를 추적하고 유사 운영사례를 확보하는 것이 먼저입니다. 수집한 사례를 향후 디지털 트윈 악천후 반복훈련에 활용하는 개선안을 제안합니다.')
        with st.expander('추가 탐색 · BCT 접안시점 강풍 유무별 중앙값과 사분위 범위 (집계)',expanded=False):
            dd=D['bct_berth_weather_distribution'].copy()
            dd['group']=dd.wind_high.map({True:'접안시점 강풍 P90',False:'접안시점 P90 미만'})
            fig=go.Figure(go.Bar(x=dd.group,y=dd.median_rate,
                marker_color=[CORAL if t else AQUA for t in dd.wind_high],
                error_y=dict(type='data',symmetric=False,array=dd.p75-dd.median_rate,
                             arrayminus=dd.median_rate-dd.p25),
                customdata=dd.n,hovertemplate='%{x}<br>중앙 작업률 %{y:.1f}/h<br>n=%{customdata:,.0f}<extra></extra>'))
            fig.update_layout(title='BCT · 접안시점 강풍별 작업률 중앙값 및 IQR',yaxis_title='작업단위/h',height=340,showlegend=False)
            chart(fig)
            st.caption('선박별 기록 없이 집계 사분위수만 사용합니다. 첫 6시간 4~6회 강풍 노출 회귀와 다른 비교조건이며 인과 검정이 아닙니다.')

    elif stage==4:
        st.markdown('<span class="step-tag">STEP 05 · 다른 원인을 가능한 범위에서 통제</span>',unsafe_allow_html=True)
        title('작업량·선박 크기·혼잡 차이를 고려했나?','선박별 GT·작업량·시기·혼잡을 포함한 회귀분석 및 FDR 비교. 관측되지 않은 원인을 모두 제거한 것은 아닙니다.')
        a,b=st.columns([1.25,.85])
        with a:
            factor=st.radio('기상요인',options=['wind','wave'],format_func=lambda x:'풍속' if x=='wind' else '유의파고',horizontal=True,key='story_factor')
            threshold=st.radio('강도',options=['P90','P95'],horizontal=True,key='story_thresh')
            chart(wind_wave_forest(D['weather_reg'],factor,threshold))
        with b:
            st.markdown('''<div class="evidence-card"><h4>통제에 포함한 핵심 요인</h4>
            <p>① 선박 규모: <b>GT</b> (2020~2024)</p><p>② 작업량 및 운영 방식</p>
            <p>③ 접안 시각·요일·월·연도</p><p>④ 최근 6시간 혼잡부담</p>
            <p>⑤ 다중검정: 7터미널 FDR</p>
            <p class="muted">Shift가 전 기간 0으로 기록된 북항 일부 표본은 실제 교대근무가 0이라는 의미가 아닙니다.</p></div>''',unsafe_allow_html=True)
        st.markdown('<div class="story-note"><b>해석:</b> 조정 이후에도 BCT 강풍 노출에 따른 체류시간 차이가 일부 사양에서 남지만, 선박의 실제 작업중단·장비고장·인력 배치 기록이 없어서 <b>자동화가 원인이라고 증명할 수는 없습니다.</b></div>',unsafe_allow_html=True)

    elif stage==5:
        st.markdown('<span class="step-tag">STEP 06 · 시간별 머신러닝을 통한 교차검증</span>',unsafe_allow_html=True)
        title('2024년과 2025년에서도 같은 신호가 유지되나?','정상 기대 출항확률 대비 실제 출항률의 차이 · 기상별 노출시간만 비교 (과거 연구의 별도 ML 모형)')
        a,b=st.columns([1.3,.7])
        with a:
            t=st.selectbox('관심 터미널',['BCT','BNCT','HPNT','PNC','PNIT','BPTC_G','BPT_S'],key='story_ml_term')
            f=st.radio('기상요인 (P95)',options=['wave','wind'],format_func=lambda x:'고파고' if x=='wave' else '강풍',horizontal=True,key='story_ml_factor')
            chart(ml_year_chart(D['weather_hourly_ml'],t,f,'P95'))
        with b:
            st.markdown('''<div class="evidence-card"><h4>왜 이 결과가 필요한가?</h4>
            <p>선박별 전체 작업률이 아닌 <b>매시간 다음 1시간 안에 출항할 확률</b>을 예측합니다.</p>
            <p>2024년과 2025년을 분리해 검증합니다.</p>
            <p><b>2025년 신항의 시간별 파고·시정 관측이 일부 누락</b>돼, 유의한 결과조차 대표성에 주의해야 합니다.</p>
            <p class="muted">시간별 출항확률 모델과 이 대시보드의 평균 처리량 회귀모델은 다른 예측 대상입니다.</p></div>''',unsafe_allow_html=True)
        st.markdown('<div class="story-risk"><b>재현성 판단:</b> 2020~2024 GT 통제 강풍 결과를 2025년 실시간 처리량의 확정 피해율로 재사용해서는 안 됩니다. 2025년 미학습 검증에서 일부 신호는 유의하지 않았으며 기상자료의 관측방식도 다릅니다.</div>',unsafe_allow_html=True)

    elif stage==6:
        st.markdown('<span class="step-tag">STEP 07 · 근거에서 실행 가능한 결재 제안으로</span>',unsafe_allow_html=True)
        title('개선속도 비교에서 디지털 트윈 검증까지','BCT가 가장 취약하다고 단정하는 것이 아니라, 강풍 취약조건과 운영학습 가능성을 함께 살펴 실증 후보로 선정했습니다.')
        p1,p2=st.columns(2)
        with p1:
            st.markdown('''<div class="evidence-card"><h4>분석이 말하는 사실</h4>
            <p>7개 터미널의 기상 관련 체류시간 변동은 다릅니다.</p>
            <p>BCT·BNCT의 <b>장시간 강풍</b>에서 큰 조정 차이가 나타납니다.</p>
            <p>BCT 기존 분석에서도 큰 효율저하 사건의 <b>꼬리위험</b>이 관측됐습니다.</p>
            <p>BCT의 연간 강풍 대응효율 개선 추세는 <b>p=0.794</b>로 확인되지 않았습니다. BCT와 다른 터미널의 기울기 차이 또한 <b>p=0.910</b>으로 유의하지 않았습니다.</p></div>''',unsafe_allow_html=True)
        with p2:
            st.markdown('''<div class="evidence-card"><h4>개선속도 해석 · 자동화는 별도 검토</h4>
            <p><b>2022~2024 p=0.910:</b> BCT와 비교 터미널 사이 강풍 대응효율 변화 기울기의 유의한 차이는 확인되지 않았습니다.</p>
            <p><b>운영 노하우 축적</b>은 개선을 설명할 수 있는 가설이며, 실제 숙련도 증가를 측정한 것은 아닙니다.</p>
            <p><b>자동화 수준</b>은 별도로 확인할 운영환경 요인입니다. 자동화 숙련 미비를 원인으로 확정하지 않습니다.</p>
            <p class="muted">비유의성(p=0.910)은 두 그룹의 개선속도가 동일하다는 증명도 아닙니다.</p></div>''',unsafe_allow_html=True)
            st.markdown('''<div class="evidence-card"><h4>결재 요청</h4>
            <p><b>범위:</b> BCT 1개 터미널, 90일 실증</p>
            <p><b>필요:</b> 선석·장비·시간당 Moves·정지/재개·근무조 로그</p>
            <p><b>방법:</b> 강풍 노출 누적 감시 → 유사기상 사례 검색 → 운영 시나리오 비교 → 현장 검증</p>
            <p><b>평가:</b> 위험 조기경보, 중단시간, 실제 작업효율, 오경보, 운영자 대응 경험 축적 여부</p>
            <p class="muted">실증 전 절감액·생산성 개선율은 확정할 수 없습니다.</p></div>''',unsafe_allow_html=True)
        st.markdown('<div class="story-note"><b>핵심 개선안:</b> 우선 90일간 BCT의 실제 시간별 작업·장비·인력·정지/재개 데이터를 수집하고, 디지털 트윈에 필요한 검증 데이터를 확보합니다. 그 뒤 가상 악천후 대응 훈련을 보정·검증하고 다른 터미널로의 확대 여부를 결정합니다. <b>④ 디지털 트윈 제안 탭</b>에서 예시 정책 비교를 확인할 수 있습니다.</div>',unsafe_allow_html=True)
        st.caption('공통 기간 2022~2024의 새 확장분석 결과입니다: BCT 연간 개선 p=0.794, BCT 대 다른 터미널 기울기 차이 p=0.910. 둘 모두 동등성·숙련효과를 입증하지 않습니다.')
    st.caption('근거 출처: 패키지 내 2020~2025년 기항자료·2020~2024년 GT 통제 회귀표·시간별 ML 결과표·기존 심층발표 v4. 별도 모델의 수치를 동일 손실률로 합산하지 않습니다.')


with T6:
    st.markdown('<span class="pill">작업량·GT·혼잡 보정</span><span class="pill">2020~24 모델 B</span><span class="pill">2022~24 동일기간 개선추세</span>',unsafe_allow_html=True)
    title('모델 B · 정상 기대효율 대비 악천후 추가 저하',
          '모델 A는 조건별 평균 작업률을 예측합니다. 모델 B는 운영조건을 먼저 통제한 정상 기대효율 잔차와 첫 6시간 P90 기상 노출을 비교합니다.')
    st.markdown('<div class="story-note"><b>핵심 해석</b> · 작업량의 SHAP 기여를 강제로 낮춘 것이 아닙니다. GT·작업량·혼잡·Shift·달력 등을 보정해 날씨와 추가 효율 저하의 <b>통계적 연관성</b>을 별도 추정했습니다. 작업효율은 선박별 총작업량 ÷ 접안 후 체류시간이므로 실제 크레인 가동 생산성이 아닙니다.</div>',unsafe_allow_html=True)
    eff=D['modelb_weather_residual_effects'].copy()
    eff['factor_ko']=eff.variable.map({'wind':'강풍','wave':'고파고','visibility':'저시정'})
    csel,esel=st.columns([1.15,.85])
    with csel:
        fac=st.selectbox('기상요인',options=['wind','wave','visibility'],format_func=lambda q:{'wind':'강풍','wave':'고파고','visibility':'저시정'}[q],key='mb_wx_factor')
    with esel:
        exp=st.radio('첫 6시간 중 P90 노출',options=['1-3','4-6'],format_func=lambda q:q+'회',horizontal=True,key='mb_exposure')
    view=eff[(eff.variable==fac)&(eff.exposure==exp)].set_index('terminal').reindex(TERMINALS).reset_index()
    fig=go.Figure(go.Scatter(x=view.adjusted_eff_change_pct,y=view.terminal,mode='markers',
            marker=dict(size=12,color=[CORAL if t=='BCT' else AQUA for t in view.terminal],line=dict(width=1,color='#fff')),
            error_x=dict(type='data',symmetric=False,array=(view.ci_high_pct-view.adjusted_eff_change_pct).to_numpy(),
                         arrayminus=(view.adjusted_eff_change_pct-view.ci_low_pct).to_numpy(),color='#78909A',thickness=1.5),
            customdata=np.stack((view.n_exposed.to_numpy(),view.fdr_q_7terminal.to_numpy()),axis=-1),
            hovertemplate='%{y}: %{x:+.2f}%<br>노출 n=%{customdata[0]}<br>FDR q=%{customdata[1]:.3f}<extra></extra>'))
    fig.add_vline(x=0,line=dict(color='#627585',dash='dash'))
    fig.update_layout(title=f'7개 터미널 · {fac} P90 {exp}회 vs 0회 조정 효율 차이',
        xaxis_title='정상 기대효율 대비 조정 차이 (%)',height=400,margin=dict(t=55,l=12,r=22,b=45),showlegend=False)
    fig.update_yaxes(autorange='reversed')
    chart(fig)
    st.dataframe(view[['terminal','n_exposed','adjusted_eff_change_pct','ci_low_pct','ci_high_pct','fdr_q_7terminal']]
        .rename(columns={'terminal':'터미널','n_exposed':'고노출 n','adjusted_eff_change_pct':'효율차이(%)',
        'ci_low_pct':'95% CI 하한','ci_high_pct':'95% CI 상한','fdr_q_7terminal':'FDR q'}).round(3),
        hide_index=True,use_container_width=True)
    st.caption('첫 6개 시간관측: 최소 5회 관측 조건. P90은 항만별 2020~2024년 고정 분위 기준입니다. 음수는 같은 운영조건을 고려한 고노출군의 낮은 처리효율 연관성을 뜻하며, 인과적 기상 손실률이 아닙니다. 다중검정은 기상요인별 7터미널 내에서 보정했습니다.')

    title('BCT 결과의 분석방법별 일관성','같은 숫자처럼 보이는 체류시간 +8.67%와 작업효율 -10.74%는 결과변수가 다릅니다. 매칭은 다른 회귀들과 동시 기상 보정 정도가 다릅니다.')
    bct=eff[(eff.terminal=='BCT')&(eff.variable=='wind')&(eff.exposure=='4-6')].iloc[0]
    ols=D['modelb_direct_ols'].query("terminal == 'BCT'").iloc[0]
    match=D['modelb_matched_peers'].query("terminal == 'BCT'").iloc[0]
    test24=D['modelb_2024_holdout'].query("terminal == 'BCT'").iloc[0]
    method=pd.DataFrame([
       {'방법':'모델 B · 교차검증 잔차','조정 효율차이(%)':bct.adjusted_eff_change_pct,'95% 하한':bct.ci_low_pct,'95% 상한':bct.ci_high_pct},
       {'방법':'직접 다중회귀','조정 효율차이(%)':ols.direct_ols_eff_pct,'95% 하한':ols.ci_low_pct,'95% 상한':ols.ci_high_pct},
       {'방법':'유사 평온조건 매칭','조정 효율차이(%)':match.matched_eff_diff_pct,'95% 하한':match.ci_low_pct,'95% 상한':match.ci_high_pct},
       {'방법':'2024 미학습 자료 검정','조정 효율차이(%)':test24.change_pct,'95% 하한':test24.ci_low_pct,'95% 상한':test24.ci_high_pct},
    ])
    st.dataframe(method.round(2),hide_index=True,use_container_width=True)
    st.info('BCT에서는 강풍 1~3회 노출 - 아닌 +0.40%(95% CI -3.60~+4.56%)로 유의한 효율 차이가 확인되지 않았지만, 4~6회 노출군에서는 감소 방향 신호가 반복됐습니다. 정확히 4시간에 작업을 멈춘다는 뜻은 아닙니다.')

    title('기상정보가 AI 예측 정확도도 개선했는가?',
          '운영조건만 사용한 모델과 날씨를 더한 모델을 미학습연도에서 직접 비교한 결과입니다.')
    eval2024=D['modelb_2024_prediction_ablation'].query("group == 'ALL'").iloc[0]
    eval2025=D['modelb_2025_prediction_ablation'].query("group == 'ALL'").iloc[0]
    comp=pd.DataFrame([
       {'검증코호트':'2024 GT·첫6시간','기본운영모델':eval2024.normal_wape_pct,
        '기상보정포함':eval2024.modelB_wape_pct,'검증 n':int(eval2024.n)},
       {'검증코호트':'2025 비GT·접안기상','기본운영모델':eval2025.operational_only_wape,
        '기상보정포함':eval2025.operational_plus_weather_wape,'검증 n':int(eval2025.n)},
    ])
    pp=comp.melt(id_vars=['검증코호트','검증 n'],value_vars=['기본운영모델','기상보정포함'],var_name='모델',value_name='WAPE (%)')
    fig=px.bar(pp,x='검증코호트',y='WAPE (%)',color='모델',barmode='group',text='WAPE (%)',
               color_discrete_map={'기본운영모델':AQUA,'기상보정포함':CORAL})
    fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside')
    fig.update_layout(title='미학습 연도 · 기상정보 추가 전/후의 예측오차',yaxis_title='WAPE (%)',height=345)
    fig.update_yaxes(range=[0,float(pp['WAPE (%)'].max())*1.25])
    chart(fig)
    st.caption('2024 GT: 전체 WAPE 14.06% → 14.10%; 2025 비GT: 15.58% → 15.66%. 현재 기상변수의 추가 예측력은 입증되지 않았습니다. 2025년 신항 기상 관측의 표본선택 가능성에도 주의하세요. 예측력 부족이 기상의 실제 영향 부재를 뜻하지는 않습니다.')
    with st.expander('모델 B 2단계 기상잔차 보정 SHAP · 예측기여도',expanded=False):
        ss=D['modelb_weather_shap'].sort_values('mean_abs_shap_log')
        fig=px.bar(ss,x='mean_abs_shap_log',y='variable',orientation='h',
            title='2024년 2단계 모델 SHAP 평균 절대기여도 (log 단위)',labels={'mean_abs_shap_log':'평균 |SHAP| (log)','variable':'입력변수'})
        chart(fig)
        st.caption('2단계 기상 보정 모형의 모델 행동 설명일 뿐 실제 기상효과의 원인 순위가 아닙니다. 전체 예측력 개선은 확인되지 않았습니다.')

    st.divider()
    title('운영학습 · 평온 작업효율 증가와 악천후 대응력 개선을 분리',
          '2022~2024년 동일기간 21,302기항(7개 터미널)을 비교합니다. BCT의 2022년 고노출 사례는 6건이므로 초기 추정의 불확실성이 큽니다.')
    lc,rc=st.columns(2)
    with lc:lc.metric('BCT 평온조건 2024 vs 2022','+18.35%','95% CI +11.34~+25.80%',delta_color='off')
    with rc:rc.metric('BCT 강풍대응 개선 추세','p=0.794','유의한 연간 개선 확인 안 됨',delta_color='off')
    peer=D['learning_pooled_test']
    contrast=peer[peer.contrast.str.contains('BCT minus equal-weight',case=False,regex=False)]
    pv=float(contrast.iloc[0].p) if not contrast.empty else float('nan')
    st.markdown(f'<div class="story-risk"><b>BCT 대 다른 6개 터미널 개선기울기 차이:</b> p={pv:.3f}. 통계적으로 차이가 유의하다는 근거가 부족하며, <b>개선속도가 같다는 증거도 아닙니다.</b> 모든 터미널에서 악천후 민감도가 줄었다는 가설도 이번 자료에서 확인되지 않았습니다.</div>',unsafe_allow_html=True)
    years=D['learning_annual_weather_penalty']
    slopes=D['learning_annual_slopes']
    target=st.selectbox('추세 확인 터미널',TERMINALS,key='learning_target_terminal')
    b=years[years.terminal==target].sort_values('year').copy()
    fig=go.Figure(go.Scatter(x=b.year,y=b.adjusted_high_eff_pct,mode='lines+markers',line=dict(color=AQUA,width=3),
        marker=dict(size=10),error_y=dict(type='data',symmetric=False,
        array=(b.ci_high_pct-b.adjusted_high_eff_pct).to_numpy(),
        arrayminus=(b.adjusted_high_eff_pct-b.ci_low_pct).to_numpy()),
        customdata=b.n_4to6,hovertemplate='%{x}: %{y:+.2f}%<br>강풍 4~6회 n=%{customdata}<extra></extra>'))
    fig.add_hline(y=0,line_dash='dash',line_color='#85939B')
    fig.update_layout(title=f'{target} · P90 강풍 4~6회와 0회 간 조정 효율차이 (연도별)',height=355,
        xaxis=dict(tickmode='array',tickvals=[2022,2023,2024]),yaxis_title='조정 효율차이 (%)')
    chart(fig)
    st.dataframe(b[['year','n_total','n_4to6','n_0','adjusted_high_eff_pct','ci_low_pct','ci_high_pct']].rename(columns={
        'year':'연도','n_total':'평가 기항','n_4to6':'고노출 4~6회','n_0':'비노출 0회','adjusted_high_eff_pct':'조정 효율차이(%)',
        'ci_low_pct':'95% 하한','ci_high_pct':'95% 상한'}).round(2),hide_index=True,use_container_width=True)
    yr=slopes.set_index('terminal').reindex(TERMINALS).reset_index()
    fig=go.Figure(go.Scatter(x=yr.annual_change_in_eff_ratio_pct,y=yr.terminal,mode='markers',
       marker=dict(size=11,color=[CORAL if q=='BCT' else AQUA for q in yr.terminal]),
       error_x=dict(type='data',symmetric=False,array=yr.annual_change_ratio_ci_high-yr.annual_change_in_eff_ratio_pct,
           arrayminus=yr.annual_change_in_eff_ratio_pct-yr.annual_change_ratio_ci_low),
       customdata=yr.p_slope,hovertemplate='%{y} 연간 변화 %{x:+.2f}%<br>추세 p=%{customdata:.3f}<extra></extra>'))
    fig.add_vline(x=0,line_dash='dash',line_color='#85939B')
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title='7개 터미널 · 장시간 강풍 대응 효율비율의 연간 변화 (95% CI)',xaxis_title='효율비율 연간 변화 (%)',height=400)
    chart(fig)
    calm_trend_df=D['learning_calm_2024_vs_2022']
    calm_trend_df=calm_trend_df[calm_trend_df.year==2024].set_index('terminal').reindex(TERMINALS).reset_index()
    fig=go.Figure(go.Bar(x=calm_trend_df.terminal,y=calm_trend_df.change_pct,marker_color=[CORAL if x=='BCT' else AQUA for x in calm_trend_df.terminal],
        error_y=dict(type='data',symmetric=False,array=calm_trend_df.ci_high_pct-calm_trend_df.change_pct,
                     arrayminus=calm_trend_df.change_pct-calm_trend_df.ci_low_pct)))
    fig.add_hline(y=0,line_dash='dash',line_color='#85939B')
    fig.update_layout(title='2024 vs 2022 · 세 기상요소 모두 평온한 조건의 조정 작업효율 변화',yaxis_title='보정 작업효율 변화 (%)',height=340)
    chart(fig)
    st.caption('평온조건 상승은 자동화·운영숙련의 인과효과가 아닙니다. 강풍 추세 모델은 2022년 BCT 고노출이 6건으로 매우 적고, 강도 통제 방식에 따라 기울기 방향이 달라질 수 있습니다.')
    with st.expander('강풍 민감도·개선속도 분석표 CSV 다운로드',expanded=False):
        q1,q2=st.columns(2)
        with q1:st.download_button('모델 B 7터미널 결과',eff.to_csv(index=False).encode('utf-8-sig'),file_name='PORTCAST_ModelB_7terminals.csv',mime='text/csv')
        with q2:st.download_button('운영학습 연도 추세 결과',years.to_csv(index=False).encode('utf-8-sig'),file_name='PORTCAST_Weather_Learning_Annual.csv',mime='text/csv')
    st.markdown('<div class="story-note"><b>확장 제안</b> · 운영기술의 자동화와 악천후 대응학습은 구분해 검증해야 합니다. BCT에서 시간별 Moves/h·작업중단/재개·작업조·장비 로그를 수집하고, 희소 악천후를 디지털 트윈에서 반복 재현하여 유사조건 비교와 운영자 승인하에 실효성을 검증하는 90일 파일럿을 제안합니다. 현재 디지털 트윈의 대응정책 효과는 미구현·미검증입니다.</div>',unsafe_allow_html=True)

with T3:
    st.markdown('<span class="pill">미래 핵심 개선안: 디지털 트윈</span><span class="pill">현재 구현: ML 조건별 예측</span><span class="pill">운영전략 효과는 아직 미검증</span>',unsafe_allow_html=True)
    title('왜 디지털 트윈인가? · 개선속도 비교와 운영학습 가설',
        '강풍 관련 차이를 통제한 뒤 강풍 대응효율 추세를 검증했습니다(BCT p=0.794; BCT-비교군 p=0.910). 운영 노하우 축적은 해석 가설이며 자동화는 별도로 고려할 운영환경 요인입니다.')
    st.markdown('''<div class="dt-story-grid">
      <div class="dt-story-panel"><div class="dt-tag">01 · 관측 결과</div>
      <h4>강풍 지속시간에서 운영 격차 확인</h4><p>BCT GT 통제 분석: 첫 6시간 중 P90 강풍 4~6시간 노출 집단의 체류시간 +8.67% 연관성. BNCT도 +8.50% 신호가 있음.</p></div>
      <div class="dt-story-panel"><div class="dt-tag">02 · 개선속도 비교</div>
      <h4>개선 추세 확인 안 됨 · p=0.910</h4><p>BCT와 비교 터미널의 개선속도 차이는 통계적으로 유의하지 않았습니다. 이 결과만으로 숙련 효과나 개선속도의 동등성을 확정하지 않습니다.</p></div>
      <div class="dt-story-panel"><div class="dt-tag">03 · 해석 가설과 별도 검토 요인</div>
      <h4>노하우 축적 가능성 / 자동화 환경 검토</h4><p>운영 경험 축적이 개선을 설명할 가능성을 검토합니다. 자동화는 별도의 운영환경 요인으로, 숙련 미비를 원인으로 단정하지 않습니다.</p></div>
      <div class="dt-story-panel"><div class="dt-tag">04 · 후속 해결책(제안)</div>
      <h4>디지털 트윈으로 희소 악천후 반복훈련</h4><p>장비·작업·중단 로그를 확보하고 악천후 대응절차를 가상훈련한 뒤 현장 효과를 검증합니다.</p></div>
    </div>''',unsafe_allow_html=True)
    st.markdown('<div class="story-risk"><b>관측 · 해석 · 제안의 구분:</b> BCT 개선 추세 p=0.794, BCT와 비교군 차이 p=0.910은 숙련 학습의 인과증거나 동등성 검정이 아닙니다. 노하우 축적은 가능한 해석, 자동화는 별도 고려 요인입니다. 아래는 <b>가상 대응 목표를 계산하는 개념 실험실</b>로, 실제 디지털 트윈·훈련 성과는 아직 구현·검증되지 않았습니다.</div>',unsafe_allow_html=True)
    title('디지털 트윈 구축 경로','현재의 선박별 평균 처리효율 모델에서 시작해, 실제 시간별 하역·장비 기록을 추가해야 운영 디지털 트윈으로 발전할 수 있습니다.')
    st.markdown('''<div class="dt-roadmap" role="list" aria-label="디지털 트윈 구축 단계">
       <article class="dt-road-card" role="listitem">
        <div class="dt-road-top"><span class="dt-road-number">01</span><span class="dt-road-phase">DATA · 실제 운영기록</span></div>
        <h4 class="dt-road-title">선박·기상·장비 로그 확보</h4>
        <p class="dt-road-desc">기존 선박·기상 자료에 크레인별 Moves/h, 작업중단·재개 시각을 연계합니다.</p>
        <span class="dt-road-status">기존 자료 있음 · 현장 로그 추가 필요</span>
       </article>
       <span class="dt-road-arrow" aria-hidden="true">→</span>
       <article class="dt-road-card ai" role="listitem">
        <div class="dt-road-top"><span class="dt-road-number">02</span><span class="dt-road-phase">ANALYSIS · 통계·AI</span></div>
        <h4 class="dt-road-title">회귀분석·ML로 영향 분석</h4>
        <p class="dt-road-desc">기상·혼잡·GT 등을 통제하고, 개선속도 검정(BCT p=0.794; 비교군 p=0.910) 및 조건별 처리효율을 검토합니다.</p>
        <span class="dt-road-status">현재 분석·예측 기능 구현</span>
       </article>
       <span class="dt-road-arrow" aria-hidden="true">→</span>
       <article class="dt-road-card twin" role="listitem">
        <div class="dt-road-top"><span class="dt-road-number">03</span><span class="dt-road-phase">TWIN · 핵심 제안</span></div>
        <h4 class="dt-road-title">악천후 가상 대응훈련</h4>
        <p class="dt-road-desc">희소 악천후를 반복 재현하고, 작업 배치·중단/재개 절차의 대응안을 실험합니다.</p>
        <span class="dt-road-status">향후 구축·보정할 기능</span>
       </article>
       <span class="dt-road-arrow" aria-hidden="true">→</span>
       <article class="dt-road-card verify" role="listitem">
        <div class="dt-road-top"><span class="dt-road-number">04</span><span class="dt-road-phase">VALIDATE · 현장 실증</span></div>
        <h4 class="dt-road-title">실제 운영 성과 검증</h4>
        <p class="dt-road-desc">BCT에서 유사조건의 Moves/h·중단시간·재개시간을 비교하고 타 터미널 확산을 판단합니다.</p>
        <span class="dt-road-status">후속 BCT 파일럿 계획</span>
       </article>
      </div><div class="dt-road-footnote"><b>현재 구현:</b> 선박·기상 기반 회귀/ML 분석 및 처리효율 예측 &nbsp; | &nbsp; <b>후속 제안:</b> 크레인·작업 로그 연계, 디지털 트윈 훈련, 현장 검증</div>''',unsafe_allow_html=True)
    title('가상 악천후 대응전략 비교 · 개념 실험실','① 선택 날씨에서 AI 추정값을 계산하고 ② 개선 목표(가정)를 입력해 ③ 어떤 현장데이터로 검증할지 계획합니다.')
    train_options=['예측 탭에서 선택한 날씨','선택 조건 + 신항/북항 P95 강풍','선택 조건 + P95 강풍·고파고','선택 조건 + P95 저시정']
    weather_case=st.selectbox('훈련 시나리오 · 날씨 조건',train_options,index=0,key='dt_training_weather')
    sim=cur
    if weather_case==train_options[1]:
        sim=replace(cur,wind=max(cur.wind,float(th.wind_P95)))
    elif weather_case==train_options[2]:
        sim=replace(cur,wind=max(cur.wind,float(th.wind_P95)),wave=max(cur.wave,float(th.wave_P95)))
    elif weather_case==train_options[3]:
        sim=replace(cur,visibility_m=min(cur.visibility_m,float(th.vis_10m_P95)*10))
    live_assessment=estimate(m,sim,calm,use_gt)
    lab=TrainingLab.from_model(live_assessment)
    desc=f'{sim.terminal} / {sim.when.strftime("%Y-%m-%d %H:00")} / 풍속 {sim.wind:.1f} m/s / 파고 {sim.wave:.2f} m / 시정 {sim.visibility_m:,.0f} m / 혼잡 {crowd_name} ({sim.arrival_prev6:.0f}척/6h)'
    st.markdown(f'<div class="selected-congestion"><b>동일 조건으로 분석 탭과 연동:</b> {desc}</div>',unsafe_allow_html=True)
    ca,cb,cc=st.columns(3)
    ca.metric('평온조건 AI 예상치',f'{lab.baseline:,.1f} 작업단위/h')
    cb.metric('악천후 AI 예상치',f'{lab.exposed:,.1f} 작업단위/h',delta=f'{live_assessment["delta_pct"]:+.2f}% (연관성)',delta_color='off')
    cc.metric('두 조건 예측 차이',f'{lab.remaining_gap:,.1f} 작업단위/h',delta='손실 확정치가 아닌 ML 조건부 차이',delta_color='off')
    st.markdown('<div class="dt-soft-warning"><b>아래 목표치는 검증된 개선효과가 아닙니다.</b> 20%·40%처럼 입력하는 수치는 사용자가 정하는 <b>가상 회복 목표</b>입니다. CatBoost는 날씨와 평균 작업률만 예측하며, 훈련·배치변경의 효과를 학습하지 않았습니다. 목표 산식은 <b>악천후 예상치 + (평온 예상치 − 악천후 예상치) × 가정 회복비율</b>입니다.</div>',unsafe_allow_html=True)
    qa,qb=st.columns(2)
    with qa:
        p_reassign=st.slider('전략 A · 작업순서·장비배치 재검토 시 회복 목표 (%)',min_value=0,max_value=100,value=20,step=5,key='dt_reassign_target')
    with qb:
        p_protocol=st.slider('전략 B · 강풍 정지/재개 대응훈련 시 회복 목표 (%)',min_value=0,max_value=100,value=40,step=5,key='dt_protocol_target')
    outcomes=lab.table(p_reassign,p_protocol)
    chart(goal_plot(lab,outcomes))
    st.dataframe(outcomes.rename(columns={'시나리오':'훈련전략','회복_가정_pct':'가정 회복비율(%)','목표_평균처리량':'가상 목표값(단위/h)','평온대비_변화_pct':'평온 대비 목표변화(%)','수치_성격':'근거 구분'}).round(2),hide_index=True,use_container_width=True)
    if lab.remaining_gap<=1e-8:
        st.info('선택한 악천후에서 ML이 평온 대비 감소를 예상하지 않습니다. 가정된 회복 여지도 0으로 표시합니다. 실제 대응전략의 효과가 없다는 뜻은 아닙니다.')
    st.caption('전략 A/B의 결과는 동일한 날씨 조건과 인위적인 목표값을 비교한 산술 예시이며, 실제 디지털 트윈 시뮬레이션·AI 정책 추천·측정된 개선율이 아닙니다. 원자료의 작업단위/h이며 실제 시간별 크레인 Moves/h와 다릅니다.')
    csv=outcomes.copy()
    csv.insert(0,'검토_기상조건',desc)
    csv['기준_평온_AI_추정']=lab.baseline
    csv['기준_악천후_AI_추정']=lab.exposed
    csv['주의사항']='가정 회복비율은 실제 효과 아님; 디지털 트윈은 기획·검증 단계'
    st.download_button('가상 대응전략 비교표 CSV 다운로드',csv.to_csv(index=False).encode('utf-8-sig'),
                       file_name='부산항_디지털트윈_가상대응전략_검토.csv',mime='text/csv')
    title('디지털 트윈을 실제로 만들기 위해 필요한 자료','현재 자료에는 장비별 매시간 Moves·정지/재개 시간·작업조 배치가 없으므로 물리·운영 디지털 트윈을 검증할 수 없습니다.')
    dc1,dc2,dc3=st.columns(3)
    with dc1:
        st.markdown('<div class="action-card"><span class="smalltag">현장 데이터</span><h4>장비·작업의 시간 해상도</h4><p>크레인 ID별 Moves/h, 중단/재개 시각, 고장코드, 선석·선박별 실제 작업 상태.</p></div>',unsafe_allow_html=True)
    with dc2:
        st.markdown('<div class="action-card"><span class="smalltag">훈련 시나리오</span><h4>악천후 대응의 반복 재현</h4><p>강풍 지속·돌풍·복합 기상 조건별 배치·순서·정지/재개 정책을 비교.</p></div>',unsafe_allow_html=True)
    with dc3:
        st.markdown('<div class="action-card"><span class="smalltag">실제 검증</span><h4>BCT 파일럿 → 단계적 확산</h4><p>유사한 선박·작업량·혼잡·Shift에서 중단시간, 재개시간, 실제 작업률을 비교·검증.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="story-note"><b>결재자가 승인할 일:</b> 운영 노하우 축적 가능성을 실증하고, 자동화 환경의 영향을 별도 검토할 수 있도록 BCT의 작업·장비·훈련 이력 연계와 가상훈련 검증 설계를 승인하는 것입니다. 실증 후 타 터미널 확산을 검토합니다.</div>',unsafe_allow_html=True)

with T4:
    title('BCT 90일 파일럿 · 결재 요청사항','분석 결과를 생산성 개선 약속으로 바꾸기보다, 실제 시간별 하역 기록을 확보하고 현장 대응체계를 검증하는 결정이 필요합니다.')
    p1,p2=st.columns(2)
    steps=[
        ('DAY 01–30','위험 감시·로그 정합성','기상 P90/P95 및 누적 노출시간 로직 검토. 선석·장비·작업조·선박 ID 결합 규칙과 이상치 기준 확정.'),
        ('DAY 31–60','시간별 현장 실적 수집','크레인별 시간당 Moves, 정지·재개, 작업 진행률, 인력·장비배치, 실제 대기시간 기록.'),
        ('DAY 61–75','디지털 트윈 훈련 설계','유사 선박·날씨·혼잡 조건을 고정하고 운영순서·장비배치·중단/재개 절차별 목표와 검증 지표 정의.'),
        ('DAY 76–90','실증 준비·확대 판단','사전 정의한 대조조건에서 경보 정확도·중단시간·실제 시간별 작업률의 관측 가능성을 점검해 실증 설계 검토.')]
    for idx,(range_,t,body) in enumerate(steps):
        with (p1 if idx%2==0 else p2):
            st.markdown(f'<div class="action-card"><span class="smalltag">{range_}</span><h4>{t}</h4><p>{body}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="story-note"><b>제안하는 승인:</b> BCT 90일 데이터 연계·훈련 설계 착수, 운영·장비·IT 담당자의 시간별 원자료 연계 협의. 이 90일은 운영 디지털 트윈 완성 약속이 아니며, 투자회수기간·절감액·실증 개선율은 검증 전에는 주장하지 않습니다.</div>',unsafe_allow_html=True)
    memo=f'''# 부산항 7개 터미널 분석 · BCT 강풍 대응 90일 파일럿 검토 메모

- 목적: 장시간 강풍 P90/P95 상황에서 실제 시간별 작업효율 변화와 대응시간을 현장에서 검증
- 근거: 최신 GT 통제 2020-2024 첫 6시간 강풍 P90 4-6h vs 0h BCT +8.67% 체류시간 연관성
- 별개 역사분석: 기존 v4 작업률 -9.76%, 큰 저하 OR 2.30배(서로 다른 결과변수/표본)
- 기간별 개선속도 차이: 2022~24 BCT 강풍 대응 추세 p=0.794 및 비교군 차이 p=0.910(둘 다 유의하지 않으며, 동등성 입증 아님)
- 1~30일: 선석·기상·장비·Shift 로그 체계화
- 31~60일: 시간별 Moves/h와 장비 정지·재개, 운영자 대응 이력 수집
- 61~75일: 디지털 트윈 모형 구축을 위한 유사조건 사례 검색·가상훈련 및 정책 비교 설계
- 76~90일: 검증 지표 및 대조군 설정; 현장 시험 준비와 데이터 품질 평가
- 해석: 노하우 축적은 운영 개선을 설명할 가설, 자동화는 별도 고려할 운영환경 요인
- 필수 주의: 관측자료 기반의 연관성이며 숙련 효과·자동화 원인·개선효과는 검증 전 미확정
'''
    st.download_button('BCT 결재 요청 메모 다운로드',memo.encode('utf-8-sig'),'부산항_BCT_개선안_90일_검토메모.md',mime='text/markdown')
    title('모형 신뢰도: 결재자가 함께 확인해야 할 숫자','여기까지는 처리량(기항 전체 체류 평균) 회귀모형의 검증 결과입니다. 위 시간별 출항확률 연구와는 결과변수가 다릅니다.')
    z=MT[MT.terminal!='전체'].copy()
    x1,x2=st.columns(2)
    with x1:
        fig=px.bar(z.sort_values('r2'),x='r2',y='terminal',orientation='h',color='terminal',color_discrete_map=COLORS,
             title='2025년 미학습 자료 · 터미널별 R²',text=z.sort_values('r2').r2.round(3),
             labels={'r2':'R²','terminal':'터미널'})
        fig.update_layout(showlegend=False)
        chart(styled(fig,340))
    with x2:
        zz=z.sort_values('coverage_2025_eligible').copy();zz['coverage_pct']=zz.coverage_2025_eligible*100
        fig=px.bar(zz,x='coverage_pct',y='terminal',orientation='h',color='terminal',color_discrete_map=COLORS,
             title='2025년 예측 평가에 포함될 수 있는 기상관측 비율',text=zz.coverage_pct.round(1),
             labels={'coverage_pct':'풍속·파고·시정 완전 관측률 (%)','terminal':'터미널'})
        fig.update_layout(showlegend=False,xaxis_range=[0,105])
        chart(styled(fig,340))
    st.dataframe(z[['terminal','n','r2','mae','rmse','coverage_2025_eligible']].rename(columns={
        'terminal':'터미널','n':'2025 평가기항','r2':'R²','mae':'MAE 작업단위/h','rmse':'RMSE 작업단위/h','coverage_2025_eligible':'관측충족 비율'}).round(3),hide_index=True,use_container_width=True)
    st.markdown('''<div class="story-risk"><b>필수 해석 제한</b><br>
    ① 실제 크레인/터미널 매시간 처리량이 아닌 선박별 <b>전체 접안체류시간 환산 평균</b>입니다.<br>
    ② 2025년 신항 기상은 선박 이벤트 시점 기록이 많아 관측 선택 편향을 배제할 수 없습니다.<br>
    ③ 평온 대비 %는 관측 연관성을 학습한 모델의 조건부 시나리오 반응으로, <b>인과효과가 아닙니다</b>.<br>
    ④ 날짜 선택만으로 미래 날씨가 입력되는 것은 아닙니다. 예보 API와 연결돼 있지 않습니다.<br>
    ⑤ GT 포함 회귀는 2020~2024년 자료에 한정됩니다.</div>''',unsafe_allow_html=True)
    with st.expander('Seaborn · 집계 중앙값 진단 차트',expanded=False):
        # Native seaborn is deliberately included for authentic statistical inspection;
        # Plotly views above are used for interactivity (zoom/hover/legend).
        if st.checkbox('Seaborn 정적 분석 차트 생성',key='sns_view',value=False):
            plt.rcParams['font.family']='NanumGothic'
            plt.rcParams['axes.unicode_minus']=False
            sample=D['history_box_overall'].copy()
            fig,ax=plt.subplots(figsize=(10,4))
            sns.barplot(data=sample,x='terminal',y='median_rate',order=TERMINALS,
                        color='#76ACBB',errorbar=None,ax=ax)
            ax.set_ylabel('중앙 접안 후 작업효율 (작업단위/h)');ax.set_xlabel('터미널')
            ax.set_title('Seaborn · 2020~2025년 터미널별 중앙 처리효율 (집계)')
            fig.tight_layout();st.pyplot(fig,use_container_width=True);plt.close(fig)
        st.caption('Seaborn/Matplotlib은 정적 도표이며, 공개본은 개별 선박의 값 대신 집계 중앙값만 표시합니다. 본문 상호작용 그래프는 Plotly입니다.')
    st.download_button('2025년 모델 성능표 CSV 저장',MT.to_csv(index=False).encode('utf-8-sig'),'부산항_모델검증_2025.csv',mime='text/csv')

with T5:
    title('2025년 악천후 조건별 CatBoost 검증',
          '2020~2024년 학습을 마친 기존 모델을 고정하고, 2025년 기상 3변수 모두 관측된 선박 7,664건에서 검증했습니다.')
    st.markdown('<div class="story-risk"><b>해석 전제</b> · 신항의 2025년 기상자료는 선박 이벤트 시점 중심으로 기록되어 관측 선택편향이 발생할 수 있습니다. ' +
                '각 강풍·고파고·저시정은 <b>접안 시점 1개 관측값</b>으로 정의하며 <b>4~6시간 노출 회귀분석과 대상이 다릅니다.</b> ' +
                'R²·MAE·WAPE는 예측 오차 지표이며 숙련도·자동화의 인과적 검정이 아닙니다.</div>',unsafe_allow_html=True)
    v_sev=D['validation_severity_2025']
    v_factors=D['validation_weather_factors_2025']
    v_avail=D['validation_availability_2025']
    v_cal=D['validation_calibration_2025']
    cohorts=['P90 미만','P90 이상·P95 미만','P95 이상']
    selected=st.selectbox('성능을 확인할 터미널', ['전체']+TERMINALS, key='validation_terminal')
    group=v_sev[v_sev.terminal==selected].copy()
    group['severity']=pd.Categorical(group.severity,categories=cohorts,ordered=True)
    group=group.sort_values('severity')
    v_coverage=v_avail[v_avail.terminal==selected].iloc[0]
    g95=group[group.severity=='P95 이상'].iloc[0]
    gbelow=group[group.severity=='P90 미만'].iloc[0]
    ka,kb,kc,kd=st.columns(4)
    ka.metric('2025년 검증 표본',f'{int(group.n.sum()):,}건',delta=f'전체 후보 {int(v_coverage.candidate_n):,}건',delta_color='off')
    kb.metric('3가지 기상 관측 충족률',f'{v_coverage.coverage_pct:.1f}%',
              delta=f'{int(v_coverage.excluded_incomplete_n):,}건 검증 제외',delta_color='off')
    kc.metric('P95 이상 WAPE',f'{g95.wape_pct:.2f}%' if pd.notna(g95.wape_pct) else '자료 없음',
              delta=f'평가 {int(g95.n):,}건',delta_color='off')
    kd.metric('P90 미만 WAPE',f'{gbelow.wape_pct:.2f}%' if pd.notna(gbelow.wape_pct) else '자료 없음',
              delta=f'평가 {int(gbelow.n):,}건',delta_color='off')
    if selected=='전체':
        st.info('전체 검증 표본 7,664건: P90 미만 5,815건 / P90~P95 사이 887건 / P95 이상 962건. ' +
                'P95 이상 WAPE 16.43%, P90 미만 15.45%로 약 0.98%p 차이입니다. ' +
                '터미널별 구성 차이와 관측 누락 때문에 날씨가 모델 오차의 원인이라고 단정할 수 없습니다.')
    gleft,gright=st.columns(2)
    with gleft:
        fig=go.Figure(go.Bar(x=group.severity.astype(str),y=group.wape_pct,
             marker_color=['#0E929A','#DBA14F','#D36B5B'],
             text=[f'{v:.2f}%<br>n={int(n):,}' for v,n in zip(group.wape_pct,group.n)],
             textposition='outside',hovertemplate='%{x}<br>WAPE %{y:.2f}%<extra></extra>'))
        fig.update_layout(title=f'{selected} · 악천후 수준별 상대 오차(WAPE)',
                          xaxis_title='',yaxis_title='WAPE (%)',height=370,showlegend=False,
                          margin=dict(t=65,b=55,l=24,r=24))
        fig.update_yaxes(range=[0,max(5,float(group.wape_pct.max())*1.3)])
        chart(fig)
    with gright:
        fig=go.Figure(go.Bar(x=group.severity.astype(str),y=group.mae,
             marker_color=['#0E929A','#DBA14F','#D36B5B'],
             text=[f'{v:.2f}/h' for v in group.mae],textposition='outside',
             hovertemplate='%{x}<br>MAE %{y:.2f} 작업단위/h<extra></extra>'))
        fig.update_layout(title=f'{selected} · 악천후 수준별 절대 오차(MAE)',
                          xaxis_title='',yaxis_title='MAE (작업단위/h)',height=370,showlegend=False,
                          margin=dict(t=65,b=55,l=24,r=24))
        fig.update_yaxes(range=[0,max(3,float(group.mae.max())*1.3)])
        chart(fig)
    st.caption('조건별 표본에서 MAE·WAPE는 실제 개별 예측과 실측 평균 작업률로 계산했습니다. ' +
               'P95 우선 구분으로 위 세 집단은 서로 겹치지 않습니다. R²는 집단 내 변동폭의 영향을 크게 받아 별도 참고 지표로 표시합니다.')
    displayed=group[['severity','train_n','n','portion_pct','mae','wape_pct','r2','bias','p90_abs_error']].copy()
    displayed['신뢰 수준']=np.where(displayed.n<30,'표본 부족 (n<30)',np.where(displayed.n<100,'표본 주의 (n<100)','표본 ≥100'))
    st.dataframe(displayed.rename(columns={'severity':'기상등급','train_n':'학습기간 사례 수','n':'2025 검증 건수',
         'portion_pct':'검증 표본 비중(%)','mae':'MAE /h','wape_pct':'WAPE (%)','r2':'R²','bias':'평균 편향(예측-실측)',
         'p90_abs_error':'오차의 P90 /h'}).round(3),hide_index=True,use_container_width=True)
    title('변수별 악천후 성능 · 풍속/파고/시정',
          '다음 지표별 범주는 서로 겹칩니다. 예: 강풍이면서 고파고인 사례는 각 조건에 모두 포함되므로 숫자를 합산하지 마세요.')
    f=v_factors[v_factors.terminal==selected].copy()
    f['flag']=np.where(f.n<30,'표본 부족',np.where(f.n<100,'주의','확인 가능'))
    fshow=f[['condition','n','train_n','mae','wape_pct','r2','bias','flag']].copy()
    st.dataframe(fshow.rename(columns={'condition':'기상조건','n':'2025 검증 건수','train_n':'학습기간 사례 수',
            'mae':'MAE /h','wape_pct':'WAPE (%)','r2':'R²','bias':'평균 편향(예측-실측)','flag':'표본수 주의'}).round(3),
            hide_index=True,use_container_width=True)
    k=f[f.n>=30].copy()
    if not k.empty:
        k=k.sort_values('wape_pct')
        colors=['#CC7060' if 'P95' in t else '#15919B' for t in k.condition]
        fig=go.Figure(go.Bar(x=k.wape_pct,y=k.condition,orientation='h',marker_color=colors,
             text=[f'{v:.1f}% · {int(n):,}건' for v,n in zip(k.wape_pct,k.n)],textposition='outside'))
        fig.update_layout(title='변수별 WAPE (검증 n ≥ 30인 조건만)',xaxis_title='WAPE (%)',
                          yaxis_title='',height=360,margin=dict(t=55,b=45,l=20,r=95),showlegend=False)
        fig.update_xaxes(range=[0,max(5,float(k.wape_pct.max())*1.35)])
        chart(fig)
    title('P90·P95 검증의 터미널별 차이','검증 표본 구성이 다른 터미널끼리 단순히 WAPE로 성과 순위를 매기면 안 됩니다.')
    heat=v_sev[v_sev.terminal!='전체']
    heat_matrix=heat.pivot(index='terminal',columns='severity',values='wape_pct').reindex(index=TERMINALS,columns=cohorts)
    count_matrix=heat.pivot(index='terminal',columns='severity',values='n').reindex(index=TERMINALS,columns=cohorts)
    heat_text=np.array([[f'{heat_matrix.loc[t,c]:.1f}%<br>n={int(count_matrix.loc[t,c])}' for c in cohorts] for t in TERMINALS],dtype=object)
    fig=go.Figure(go.Heatmap(z=heat_matrix.to_numpy(),x=cohorts,y=TERMINALS,
         text=heat_text,texttemplate='%{text}',colorscale='YlOrRd',colorbar=dict(title='WAPE (%)'),
         hovertemplate='%{y} / %{x}<br>WAPE %{z:.2f}%<extra></extra>'))
    fig.update_layout(title='7개 터미널 · 날씨 등급별 WAPE와 표본 수',height=420,margin=dict(t=60,b=38,l=22,r=22))
    chart(fig)
    title('평균 실측값 vs 평균 예측값 · 집단 보정(calibration)',
          '개별 선박 정보가 아닌, 예측값 순서에 따라 나눈 집계구간의 실제·예측 평균값입니다.')
    cal=v_cal[v_cal.terminal==selected].copy()
    if len(cal):
        fig=px.scatter(cal,x='mean_pred',y='mean_actual',size='n',color='severity',
                       category_orders={'severity':cohorts},
                       hover_data={'n':True,'bin':True,'mean_pred':':.2f','mean_actual':':.2f'},
                       labels={'mean_pred':'집계구간 평균 예측 작업률 (/h)',
                               'mean_actual':'집계구간 평균 실제 작업률 (/h)','severity':'기상등급'},
                       title='45° 기준선에 가까울수록 집단 평균 예측이 실제와 유사')
        end=float(max(cal.mean_pred.max(),cal.mean_actual.max())*1.08)
        fig.add_shape(type='line',x0=0,y0=0,x1=end,y1=end,line=dict(color='#647585',dash='dash',width=1))
        fig.update_layout(height=410,margin=dict(t=58,l=23,b=40,r=22))
        fig.update_xaxes(range=[0,end]);fig.update_yaxes(range=[0,end])
        chart(fig)
    title('검증 표본이 얼마나 대표적인가?','기상 3변수 모두 관측된 2025년 기항만 검증에 포함됐습니다.')
    st.dataframe(v_avail.rename(columns={'terminal':'터미널','candidate_n':'2025 평가 후보','evaluated_n':'기상 완전관측',
         'excluded_incomplete_n':'기상 미관측 제외','coverage_pct':'관측 충족률(%)'}).round(2),
         use_container_width=True,hide_index=True)
    st.markdown('''<div class="story-risk"><b>꼭 읽어야 할 검증 한계</b><br>
       ① P90/P95 기준은 학습기간(2020~2024년) 해역별 관측분포를 기준으로 고정했습니다.<br>
       ② 2025년 신항 기상은 전체 운항시점에 무작위로 관측되지 않았습니다. <b>신항 검증 자료의 대표성을 보장할 수 없습니다.</b><br>
       ③ 각 등급은 접안 기준 풍속·파고·시정 스냅샷으로 분류합니다. 전체 체류 중 기상 노출시간·특보 누적시간과 다릅니다.<br>
       ④ 일부 세부 집단의 표본수가 적습니다. 특히 n&lt;30일 경우 R²를 계산하지 않고 다른 지표도 참고용으로만 보세요.<br>
       ⑤ GT 포함 별도 모형은 2025년 GT 외부검증을 수행하지 않았으므로 위 성능 수치를 GT 모형에 적용하지 않습니다.<br>
       ⑥ SHAP과 조건별 모델 성능은 <b>운영 노하우나 자동화 수준의 원인 검증이 아닙니다.</b></div>''',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.download_button('등급별 검증 결과 CSV',v_sev.to_csv(index=False).encode('utf-8-sig'),
                           '부산항_2025_악천후등급별_예측검증.csv',mime='text/csv')
    with c2:
        st.download_button('기상변수별 검증 결과 CSV',v_factors.to_csv(index=False).encode('utf-8-sig'),
                           '부산항_2025_기상변수별_예측검증.csv',mime='text/csv')

st.markdown('---')
st.caption('PORTCAST | 7개 터미널 기상영향 · 모델 B · SHAP · 운영학습 추세 검증 → 디지털 트윈 확장 제안(현재 미구현) · 개인 프로젝트 · GT 포함 2020~2024 / GT 제외 2020~2025. 2025년 출항예정시각은 연구자 확인에 따라 실제 출항시각으로 취급했습니다. 실제 시간별 하역기록과 실시간 기상예보는 제공되지 않습니다.')
