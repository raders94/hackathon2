"""Dependency-light, testable business logic for the Busan7 Streamlit executive dashboard.

All throughput values are an average-per-berth-hour *proxy*, never actual hourly moves.
The conditional scenario is associational, not a causal claim or a live forecast.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

BASE_DIR=Path(__file__).resolve().parents[1]
ASSETS=BASE_DIR/'assets'
TERMINALS=['BCT','BNCT','HPNT','PNC','PNIT','BPTC_G','BPT_S']
HARBOR={t:('북항' if t.startswith('BPT') else '신항') for t in TERMINALS}
FEAT=['terminal','month','weekday','berth_hour','log_cargo','arrival_prev6','wind','wave','vis_10m']
FEAT_GT=FEAT+['log_gt']

@dataclass(frozen=True)
class Scenario:
    terminal:str
    when:datetime
    wind:float
    wave:float
    visibility_m:float
    cargo:float
    arrival_prev6:float
    active_ships:int
    gt:float|None=None


def read_csv(name:str)->pd.DataFrame:
    return pd.read_csv(ASSETS/(name+'.csv'))

def read_history()->pd.DataFrame:
    return pd.read_csv(ASSETS/'vessel_history.csv.gz')

def read_weather()->pd.DataFrame:
    return pd.read_csv(ASSETS/'weather_history.csv.gz',parse_dates=['obs_h'])

def load_model(use_gt=False)->CatBoostRegressor:
    model=CatBoostRegressor()
    model.load_model(str(ASSETS/('throughput_GT_2020_2024.cbm' if use_gt else 'throughput_nonGT_2020_2024.cbm')))
    return model

def metadata()->dict:
    return json.loads((ASSETS/'model_metadata.json').read_text())

def defaults_for(df:pd.DataFrame,terminal:str,month:int)->pd.Series:
    m=df[(df.terminal==terminal)&(df.month==month)]
    if m.empty:m=df[df.terminal==terminal]
    if m.empty:raise ValueError(f'Unsupported terminal: {terminal}')
    return m.iloc[0]

def occupancy_for(df:pd.DataFrame,terminal:str,month:int,hour:int)->int:
    q=df[(df.terminal==terminal)&(df.month==month)&(df.hour==hour)]
    if q.empty:q=df[df.terminal==terminal]
    return max(1,int(round(float(q.iloc[0].active_median))))

def congestion_reference(history:pd.DataFrame,first_year:int=2020,last_year:int=2024)->pd.DataFrame:
    """Create reproducible Low/Mid/High arrival-count scenarios per terminal.

    The model's congestion proxy is the *number of berthings in the preceding
    six hours*. Since this is a discrete count, quantile boundaries often tie.
    We choose distinct, increasing integer exemplars near the terminal's
    historical Q25, median and Q80 (2020-2024 training period). These are
    representative scenario counts, NOT real-time traffic measurements or
    an ex-post inference of a ship's actual congestion.
    """
    h=history.loc[history.year.between(first_year,last_year),['terminal','arrival_prev6']].copy()
    h['arrival_prev6']=pd.to_numeric(h['arrival_prev6'],errors='coerce')
    h=h.dropna(subset=['arrival_prev6'])
    out=[]
    for t in TERMINALS:
        g=h.loc[(h.terminal==t)&h.arrival_prev6.ge(0),'arrival_prev6']
        if g.empty:raise ValueError(f'No congestion training distribution for {t}')
        q25,q50,q80=[float(x) for x in g.quantile([.25,.50,.80]).values]
        low=max(0,int(math.floor(q25)))
        middle=max(low+1,int(round(q50)))
        high=max(middle+1,int(math.ceil(q80)))
        # Keep the representatives within observed historical support.
        max_seen=max(0,int(g.max()))
        if high>max_seen:
            high=max_seen
            middle=min(middle,max(0,high-1))
            low=min(low,max(0,middle-1))
        out.append(dict(terminal=t,low=low,middle=middle,high=high,
                        n=int(len(g)),q25=q25,q50=q50,q80=q80,
                        period=f'{first_year}-{last_year}'))
    return pd.DataFrame(out)


def congestion_count(reference:pd.DataFrame,terminal:str,level:str)->int:
    """Return model input for '하','중','상', from historical examples."""
    key={'하':'low','중':'middle','상':'high'}.get(level)
    if key is None:raise ValueError(f'Unsupported congestion category: {level}')
    selected=reference.loc[reference.terminal==terminal]
    if selected.empty:raise ValueError(f'Unsupported terminal: {terminal}')
    return int(selected.iloc[0][key])


def calm_for(df:pd.DataFrame,terminal:str,month:int)->dict:
    h=HARBOR[terminal]
    row=df[(df.harbor==h)&(df.month==month)]
    if row.empty: row=df[df.harbor==h]
    w=row.iloc[0]
    return dict(wind=float(w.wind),wave=float(w.wave),visibility_m=float(w.vis_10m*10))

def category(s:Scenario, th:pd.DataFrame)->tuple[str,list[str]]:
    r=th.set_index('harbor').loc[HARBOR[s.terminal]]
    flags=[]
    if s.wind>=r.wind_P95: flags.append('강풍 P95')
    elif s.wind>=r.wind_P90:flags.append('강풍 P90')
    if s.wave>=r.wave_P95:flags.append('고파고 P95')
    elif s.wave>=r.wave_P90:flags.append('고파고 P90')
    vis=s.visibility_m/10
    if vis<=r.vis_10m_P95:flags.append('저시정 P95')
    elif vis<=r.vis_10m_P90:flags.append('저시정 P90')
    level='HIGH' if any('P95' in f for f in flags) else ('WATCH' if flags else 'NORMAL')
    return level,flags

def model_row(s:Scenario,use_gt:bool=False)->pd.DataFrame:
    if s.terminal not in TERMINALS: raise ValueError('unknown terminal')
    d=dict(terminal=s.terminal,month=s.when.month,weekday=s.when.weekday(),berth_hour=s.when.hour,log_cargo=math.log(max(1.,s.cargo)),arrival_prev6=s.arrival_prev6,wind=s.wind,wave=s.wave,vis_10m=max(0,min(5000,s.visibility_m/10)))
    if use_gt:
        if s.gt is None or s.gt<=0:raise ValueError('GT required for GT mode')
        d['log_gt']=math.log(s.gt)
    return pd.DataFrame([d],columns=FEAT_GT if use_gt else FEAT)

def estimate(model:CatBoostRegressor,s:Scenario,calm:dict,use_gt=False)->dict:
    if min(s.wind,s.wave,s.visibility_m)<0:raise ValueError('Negative weather values invalid')
    current=np.exp(model.predict(model_row(s,use_gt))[0])
    baseline_s=Scenario(**{**s.__dict__,**calm})
    normal=np.exp(model.predict(model_row(baseline_s,use_gt))[0])
    return dict(vessel=float(current),normal=float(normal),delta_pct=float((current/normal-1)*100),aggregate=float(current*s.active_ships),aggregate_normal=float(normal*s.active_ships),active_ships=s.active_ships)

def error_range(pred:float,terminal:str,metrics:pd.DataFrame)->tuple[float,float]:
    row=metrics[metrics.terminal==terminal]
    if row.empty:row=metrics[metrics.terminal=='전체']
    r=row.iloc[0]
    return float(pred*np.exp(r.err_log_p10)),float(pred*np.exp(r.err_log_p90))

def historical_at(weather:pd.DataFrame,terminal:str,when:datetime)->dict|None:
    sel=weather.loc[(weather.harbor==HARBOR[terminal])&(weather.obs_h==pd.Timestamp(when.replace(minute=0,second=0,microsecond=0)))&(weather.observed.eq(1))]
    if sel.empty:return None
    w=sel.iloc[0]
    keys=['wind','wave','vis_10m']
    if any(pd.isna(w[k]) for k in keys):
        return dict(incomplete=True,source=str(w.source),present={k:(None if pd.isna(w[k]) else float(w[k])) for k in keys})
    return dict(incomplete=False,wind=float(w.wind),wave=float(w.wave),visibility_m=float(w.vis_10m*10),source=str(w.source),obs_h=w.obs_h)

def describe_source(obs:dict|None)->str:
    if obs is None:return '선택 시각에 관측값 없음'
    if obs['source']=='event_snapshot_2025':return '2025년 선박 이벤트 시각 복원 관측'
    return '2020~2024년 연속 시간별 관측'
