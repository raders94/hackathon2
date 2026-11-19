"""Inference for the separately trained PORTaI efficiency-shortfall classifier.

Runtime needs CatBoost only; sigmoid parameters and evaluation are frozen JSON.
No final cargo, duration, Shift, search outcome or 2025 evaluation result enters
the classifier. The prediction is suppressed outside the stated data support.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import math

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

RISK_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'risk'
PROXY_TERMINALS = {'BCT', 'BNCT', 'HPNT', 'PNC', 'PNIT'}


@lru_cache(maxsize=1)
def load_risk_metadata():
    return json.loads((RISK_DIR / 'metadata.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def _load_model():
    model = CatBoostClassifier()
    model.load_model(str(RISK_DIR / 'risk_classifier_2023.cbm'))
    return model


@lru_cache(maxsize=1)
def _support_reference():
    return pd.read_csv(RISK_DIR / 'support_reference_2023.csv.gz')


def load_risk_validation():
    return pd.read_csv(RISK_DIR / 'evaluation_by_terminal_2025.csv')


def calibrated_probability(raw, metadata=None):
    meta = metadata if metadata is not None else load_risk_metadata()
    c = meta['calibration']
    z = np.clip(np.asarray(raw, dtype=float) * c['slope'] + c['intercept'], -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def model_input(scenario):
    when = pd.Timestamp(scenario.when)
    if when.tzinfo is not None:
        when = when.tz_convert('Asia/Seoul').tz_localize(None)
    return pd.DataFrame([{
        'terminal': str(scenario.terminal), 'month': int(when.month), 'weekday': int(when.dayofweek),
        'berth_hour': int(when.hour), 'wind': float(scenario.wind), 'wave': float(scenario.wave),
        'vis_10m': float(scenario.visibility_m) / 10,
    }])[load_risk_metadata()['features']]


def support_diagnostics(terminal, wind, wave, vis_10m, metadata=None, history=None):
    """No outcome fields or evaluation-year cases enter this diagnostic."""
    meta = metadata if metadata is not None else load_risk_metadata()
    if terminal not in meta['support_scales']:
        return {'support_count': 0, 'within_bounds': False, 'outside_features': ['terminal']}
    config = meta['support_scales'][terminal]
    values = {'wind': float(wind), 'wave': float(wave), 'vis_10m': float(vis_10m)}
    outside = [k for k, v in values.items() if not config['bounds'][k][0] <= v <= config['bounds'][k][1]]
    history = _support_reference() if history is None else history
    g = history.loc[history.terminal.eq(terminal)]
    scales = config['scales']
    dist = np.sqrt(((g.wind.to_numpy() - wind) / scales['wind']) ** 2 / 3 +
                   ((g.wave.to_numpy() - wave) / scales['wave']) ** 2 / 3 +
                   ((np.log1p(g.vis_10m.to_numpy()) - np.log1p(vis_10m)) / scales['log_visibility']) ** 2 / 3)
    return {'support_count': int((dist <= meta['support_distance_threshold']).sum()),
            'within_bounds': not outside, 'outside_features': outside,
            'nearest_distance': float(dist.min()) if len(dist) else None}


def predict_risk(scenario):
    meta = load_risk_metadata()
    terminal = str(scenario.terminal)
    result = {
        'supported': False, 'probability': None, 'threshold': float(meta['review_threshold']),
        'overall_rate': meta['overall_rate'], 'terminal_rate': meta['terminal_rates'].get(terminal),
        'support_count': 0, 'warnings': [], 'status': '', 'metadata': meta, 'evaluation': meta['evaluation'],
    }
    warnings = result['warnings']
    if not meta['beats_frequency_baselines']:
        warnings.append('검증 단계의 참고 예측입니다. 2025년 확률오차가 단순한 과거 발생비율 기준보다 개선되지 않았습니다.')
    try:
        when = pd.Timestamp(scenario.when)
        if pd.isna(when):
            raise ValueError('missing date')
        if when.tzinfo is not None:
            when = when.tz_convert('Asia/Seoul').tz_localize(None)
        row = model_input(scenario)
        wind, wave, vis = (float(row[c].iloc[0]) for c in ['wind', 'wave', 'vis_10m'])
        if not all(math.isfinite(v) for v in [wind, wave, vis]) or wind < 0 or wave < 0 or vis <= 0:
            raise ValueError('invalid weather')
    except (TypeError, ValueError, OverflowError):
        result['status'] = '풍속·파고는 0 이상, 시정은 0보다 큰 유효한 값과 날짜가 필요합니다.'
        return result
    if when < pd.Timestamp(meta['available_from']):
        result['status'] = '이 모형은 2024년까지 학습·보정한 고정모형이므로 2025년 이전 시점의 위험 예측을 제공하지 않습니다.'
        return result
    if terminal not in meta['support_scales']:
        result['status'] = '학습에 포함된 부산항 7개 터미널만 지원합니다.'
        return result
    support = support_diagnostics(terminal, wind, wave, vis, metadata=meta)
    result.update(support)
    if not support['within_bounds']:
        names = {'wind': '풍속', 'wave': '파고', 'vis_10m': '시정'}
        result['status'] = f'{", ".join(names.get(x, x) for x in support["outside_features"])} 입력이 해당 터미널의 2023년 학습 관측범위를 벗어났습니다.'
        return result
    if support['support_count'] < meta['support_minimum']:
        result['status'] = f'가까운 학습 조건이 {support["support_count"]}건으로 최소 {meta["support_minimum"]}건 기준에 미달합니다.'
        return result
    raw = float(_load_model().predict(row, prediction_type='RawFormulaVal')[0])
    probability = float(calibrated_probability(raw, meta))
    result['supported'] = True
    result['probability'] = probability
    prefix = '검증 단계 · 참고용' if not meta['beats_frequency_baselines'] else '후향 검증 · 시나리오 참고값'
    result['status'] = prefix + (' · 검토선 이상' if probability >= result['threshold'] else ' · 검토선 아래')
    if when > pd.Timestamp(meta['data_end']) + pd.Timedelta(days=1):
        warnings.append('2025년까지의 기록으로 만든 모형입니다. 선택한 미래 날짜의 실시간 예보·운영정보는 연결되지 않습니다.')
    if terminal == 'BCT':
        n = meta['reference_calm_counts']['BCT']
        warnings.append(f'BCT의 기준 작업률은 개장 초기인 2022년 평온 표본 {n:,}건에 의존합니다. 이후 운영변화도 이 기준과의 차이에 포함됩니다.')
    return result
