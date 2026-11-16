"""PORTCAST release health check. Does not require Streamlit or access to raw data."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import math
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parent
A=ROOT/'assets'


def main():
    required=['app.py','requirements.txt','.streamlit/config.toml','src/logic.py',
              'src/explainability.py','src/digital_twin_lab.py','assets/throughput_nonGT_2020_2024.cbm',
              'assets/throughput_GT_2020_2024.cbm',
              'assets/modelb_weather_residual_effects.csv','assets/learning_annual_slopes.csv',
              'assets/learning_annual_weather_penalty.csv','assets/learning_pooled_test.csv',
              'assets/history_box_overall.csv','assets/congestion_reference.csv',
              'docs/해커톤_소개글_AI활용내역_수정본.md']
    for f in required:
        assert (ROOT/f).is_file(),f'Missing required file: {f}'
    for private in ['vessel_history.csv.gz','weather_history.csv.gz','vessels_GT_2020_2024.pkl.gz',
                    'vessels_all_nonGT_2020_2025.pkl.gz']:
        assert not (A/private).exists(),f'Unexpected private record file: {private}'
    sys.path.insert(0,str(ROOT/'src'))
    from logic import Scenario,load_model,calm_for,estimate,model_row,congestion_count
    from explainability import explain_prediction,compare_weather_attributions
    defaults=pd.read_csv(A/'calm_weather_by_month.csv')
    congest=pd.read_csv(A/'congestion_reference.csv')
    assert len(congest)==7 and len(set(congest.terminal))==7
    for t in congest.terminal:
        assert congestion_count(congest,t,'하')<=congestion_count(congest,t,'중')<=congestion_count(congest,t,'상')
    s=Scenario('BCT',datetime(2027,11,20,10),6.0,1.2,5000,396,2,2)
    m=load_model(False)
    calm=calm_for(defaults,s.terminal,s.when.month)
    result=estimate(m,s,calm,False)
    explanation=explain_prediction(m,s,False)
    difference=compare_weather_attributions(m,s,calm,False)
    assert math.isfinite(result['aggregate']) and result['aggregate']>0
    assert abs(result['vessel']*2-result['aggregate'])<1e-5
    assert explanation['reconstruction_error']<1e-6
    assert difference['error']<1e-6
    assert abs(result['aggregate']-58.5802)<1.0, 'Model scenario regression changed unexpectedly'
    m_gt=load_model(True)
    s_gt=Scenario('BCT',datetime(2024,11,20,10),6.0,1.2,5000,396,2,2,gt=25000)
    assert math.isfinite(estimate(m_gt,s_gt,calm_for(defaults,'BCT',11),True)['vessel'])
    assert explain_prediction(m_gt,s_gt,True)['reconstruction_error']<1e-6
    weather=pd.read_csv(A/'modelb_weather_residual_effects.csv')
    bct=weather[(weather.terminal=='BCT') & (weather.variable=='wind')&(weather.exposure=='4-6')].iloc[0]
    assert abs(bct.adjusted_eff_change_pct - (-10.73974))<.01
    trend=pd.read_csv(A/'learning_annual_slopes.csv')
    bct_t=trend.loc[trend.terminal=='BCT'].iloc[0]
    assert abs(bct_t.p_slope-.794255)<.001
    peer=pd.read_csv(A/'learning_pooled_test.csv')
    test=peer.loc[peer.contrast.str.contains('BCT minus equal-weight',regex=False)].iloc[0]
    assert abs(test.p-.909821)<.001
    non_gt=pd.read_csv(A/'validation_severity_2025.csv')
    assert int(non_gt[(non_gt.terminal=='전체')].n.sum())==7664
    print('PASS: required files, no raw history, both CatBoost models, forecasts, both SHAP checks, Model B, trend, 2025 validation')
    print(f'BCT scenario total={result["aggregate"]:.3f}/h; Model B wind={bct.adjusted_eff_change_pct:.2f}%; trend p={bct_t.p_slope:.3f}; peer p={test.p:.3f}')
    return 0

if __name__=='__main__':raise SystemExit(main())
