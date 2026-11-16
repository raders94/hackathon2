"""CatBoost native SHAP values for explainable vessel *berth-average* rate.

CatBoost was trained on log(work_per_berth_hour); native ShapValues are
additive on the log(rate) prediction, not on the original rate itself.
No shap package needed; this avoids a heavy cloud deployment dependency.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import numpy as np
import pandas as pd
from catboost import Pool,CatBoostRegressor
from logic import Scenario, model_row, FEAT, FEAT_GT

LABELS={
    'terminal':'터미널',
    'month':'월',
    'weekday':'요일',
    'berth_hour':'접안 시간대',
    'log_cargo':'선박 작업량(log)',
    'arrival_prev6':'최근 6시간 접안 유입(혼잡)',
    'wind':'풍속',
    'wave':'파고',
    'vis_10m':'시정',
    'log_gt':'선박 총톤수(log GT)',
}

def _input_text(feat: str, scenario:Scenario) ->str:
    if feat=='terminal':return scenario.terminal
    if feat=='month':return f'{scenario.when.month}월'
    if feat=='weekday':return ['월','화','수','목','금','토','일'][scenario.when.weekday()]+'요일'
    if feat=='berth_hour':return f'{scenario.when.hour:02d}:00'
    if feat=='log_cargo':return f'{scenario.cargo:,.0f} 작업단위'
    if feat=='arrival_prev6':return f'{scenario.arrival_prev6:g}척/6시간'
    if feat=='wind':return f'{scenario.wind:.2f} m/s'
    if feat=='wave':return f'{scenario.wave:.2f} m'
    if feat=='vis_10m':return f'{scenario.visibility_m:,.0f} m'
    if feat=='log_gt':return f'{scenario.gt:,.0f} GT' if scenario.gt is not None else '입력되지 않음'
    return ''


def _native(model:CatBoostRegressor,scenario:Scenario,use_gt:bool)->tuple[np.ndarray,float,float]:
    features=FEAT_GT if use_gt else FEAT
    row=model_row(scenario,use_gt)
    shap=np.asarray(model.get_feature_importance(Pool(row,cat_features=['terminal']),type='ShapValues'),dtype=float)
    if shap.shape!=(1,len(features)+1):
        raise ValueError(f'Unexpected CatBoost SHAP shape: {shap.shape}')
    vals=shap[0,:-1]
    base=float(shap[0,-1]); pred_log=float(model.predict(row)[0])
    if not np.isclose(base+float(vals.sum()),pred_log,rtol=1e-7,atol=1e-6):
        raise ValueError('CatBoost SHAP additivity check failed')
    return vals,base,pred_log


def explain_prediction(model:CatBoostRegressor,scenario:Scenario,use_gt:bool=False)->dict[str,Any]:
    """A native SHAP decomposition of the selected model output (log rate)."""
    vals,base,pred_log=_native(model,scenario,use_gt)
    features=FEAT_GT if use_gt else FEAT
    table=pd.DataFrame([{
        'feature':f,'label':LABELS.get(f,f),'value':_input_text(f,scenario),
        'shap_log':float(v),'multiplier_pct':float(100*np.expm1(v)),
        'direction':'증가 기여' if v>1e-9 else ('감소 기여' if v< -1e-9 else '변화 거의 없음'),
        'abs_log':abs(float(v)),
    } for f,v in zip(features,vals)])
    table=table.sort_values('abs_log',ascending=False,kind='stable').reset_index(drop=True)
    return {
        'table':table,'base_log':base,'base_rate':float(np.exp(base)),
        'pred_log':pred_log,'pred_rate':float(np.exp(pred_log)),
        'reconstruction_error':abs(base+float(vals.sum())-pred_log),
        'model_scale':'log(rate)',
    }


def compare_weather_attributions(model:CatBoostRegressor,scenario:Scenario,calm:dict,use_gt:bool=False)->dict[str,Any]:
    """Difference in two SHAP decompositions; model scenario, not causal weather effect.

    Features NOT changed can have nonzero delta SHAP because their interaction
    attributions change when weather changes. All feature deltas sum to log ratio.
    """
    changed,base1,cur_log=_native(model,scenario,use_gt)
    baseline_s=replace(scenario,**calm)
    normal,base2,normal_log=_native(model,baseline_s,use_gt)
    if abs(base1-base2)>1e-8:
        raise ValueError('Unexpected base difference for identical model')
    features=FEAT_GT if use_gt else FEAT
    diffs=changed-normal
    table=pd.DataFrame([{'feature':f,'label':LABELS.get(f,f),'delta_shap_log':float(v),
                         'delta_multiplier_pct':float(100*np.expm1(v)),
                         'changed_input': f in {'wind','wave','vis_10m'}}
                        for f,v in zip(features,diffs)])
    return {'table':table.sort_values('delta_shap_log',key=abs,ascending=False).reset_index(drop=True),
            'delta_log':cur_log-normal_log,'scenario_delta_pct':float(100*np.expm1(cur_log-normal_log)),
            'error':abs(float(diffs.sum())-(cur_log-normal_log))}
