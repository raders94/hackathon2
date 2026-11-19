"""Independent release checks for PORTaI case retrieval and risk inference.

Runs against packaged artifacts only.  It never fits, recalibrates or rewrites
a model.  The tests cover time/feature boundaries, units, abstention and an
independent CatBoost + sigmoid calculation, rather than UI implementation details.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from logic import Scenario, TERMINALS
from case_search import load_case_index, search_cases
from risk_engine import model_input, predict_risk, support_diagnostics, calibrated_probability


def _manual_row(scenario, features):
    when = pd.Timestamp(scenario.when)
    if when.tzinfo is not None:
        when = when.tz_convert('Asia/Seoul').tz_localize(None)
    row = {
        'terminal': scenario.terminal,
        'month': when.month,
        'weekday': when.dayofweek,
        'berth_hour': when.hour,
        'wind': scenario.wind,
        'wave': scenario.wave,
        'vis_10m': scenario.visibility_m / 10.,
    }
    return pd.DataFrame([row], columns=features)


def _manual_probability(model, row, metadata):
    raw = float(model.predict(row, prediction_type='RawFormulaVal')[0])
    z = metadata['calibration']['slope'] * raw + metadata['calibration']['intercept']
    z = max(-35., min(35., z))
    return 1. / (1. + math.exp(-z))


def _check_risk():
    path = ROOT / 'assets' / 'risk'
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    features = ['terminal', 'month', 'weekday', 'berth_hour', 'wind', 'wave', 'vis_10m']
    assert metadata['features'] == features, 'Unexpected predictor or order in risk metadata'
    classifier = CatBoostClassifier()
    classifier.load_model(str(path / 'risk_classifier_2023.cbm'))
    assert classifier.feature_names_ == features, 'Risk model contains an unexpected predictor'
    hashes = {
        filename: hashlib.sha256((path / filename).read_bytes()).hexdigest()
        for filename in ['risk_classifier_2023.cbm', 'calm_reference_2020_2022.cbm', 'metadata.json']
    }
    scenario = Scenario('BCT', datetime(2027, 11, 20, 10), 2., .3, 40000., 396., 2., 2)
    base = predict_risk(scenario)
    assert base['supported'] and base['probability'] is not None
    assert model_input(scenario).vis_10m.iloc[0] == 4000., 'Visibility m to 10m conversion failed'
    expected = _manual_row(scenario, features)
    pd.testing.assert_frame_equal(model_input(scenario), expected, check_dtype=False)
    assert math.isclose(base['probability'], _manual_probability(classifier, expected, metadata), abs_tol=1e-14)

    # These post-outcome or unsupported quantities must not enter risk inference.
    forbidden_changes = [
        {'cargo': 1.}, {'cargo': 100000.}, {'arrival_prev6': 999.},
        {'active_ships': 100}, {'gt': 200000.},
        {'cargo': float('nan'), 'arrival_prev6': -999., 'active_ships': 0, 'gt': float('nan')},
    ]
    for changes in forbidden_changes:
        alternative = replace(scenario, **changes)
        pd.testing.assert_frame_equal(model_input(alternative), model_input(scenario))
        result = predict_risk(alternative)
        assert result['probability'] == base['probability'], f'Forbidden-input dependence: {changes}'
        assert result['support_count'] == base['support_count']

    # Genuine inputs change the model row and are calculated from model + frozen
    # calibration. Chosen large contrasts also detect accidentally cached output.
    allowed_changes = [
        {'terminal': 'BPT_S'}, {'wind': 4.}, {'wave': .7}, {'visibility_m': 10000.},
        {'when': datetime(2027, 5, 20, 10)},
        {'when': datetime(2027, 11, 22, 10)},
        {'when': datetime(2027, 11, 20, 22)},
    ]
    for changes in allowed_changes:
        alternative = replace(scenario, **changes)
        row = _manual_row(alternative, features)
        assert not row.equals(expected), f'Allowed input did not reach model: {changes}'
        result = predict_risk(alternative)
        assert result['supported'], f'Chosen comparison lacks support: {changes}'
        probability = _manual_probability(classifier, row, metadata)
        assert math.isclose(result['probability'], probability, abs_tol=1e-14)
        assert not math.isclose(probability, base['probability'], abs_tol=1e-10), f'Unexpected unchanged fixture probability: {changes}'
    same_kst = replace(scenario, when=datetime(2027, 11, 20, 1, tzinfo=timezone.utc))
    pd.testing.assert_frame_equal(model_input(same_kst), model_input(scenario))
    assert predict_risk(same_kst)['probability'] == base['probability']

    # No calibrated historical probabilities before the calibration year ends.
    past = predict_risk(replace(scenario, when=datetime(2024, 12, 31, 23, 59, 59)))
    assert not past['supported'] and past['probability'] is None
    first_supported = predict_risk(replace(scenario, when=datetime(2025, 1, 1)))
    assert first_supported['supported']
    for changes in [dict(wind=100.), dict(wave=10.), dict(visibility_m=500001.)]:
        result = predict_risk(replace(scenario, **changes))
        assert not result['supported'] and result['probability'] is None
        assert not result['within_bounds'], f'Extreme input not detected: {changes}'
    for changes in [dict(visibility_m=0.), dict(wind=-1.), dict(wave=float('nan')), dict(when=pd.NaT)]:
        result = predict_risk(replace(scenario, **changes))
        assert not result['supported'] and result['probability'] is None

    # The old dashboard's deliberately severe example has too few nearby
    # classifier-training rows. The feature must abstain instead of inventing
    # confidence or returning a hidden raw probability.
    sparse = predict_risk(replace(scenario, wind=6., wave=1.2, visibility_m=5000.))
    assert sparse['support_count'] == 2
    assert not sparse['supported'] and sparse['probability'] is None

    # Independently count local support from 2023 inputs, not evaluation cases.
    reference = pd.read_csv(path / 'support_reference_2023.csv.gz')
    assert len(reference) == metadata['counts']['train']
    g = reference.loc[reference.terminal == scenario.terminal]
    scales = metadata['support_scales'][scenario.terminal]['scales']
    differences = np.column_stack([
        (g.wind.to_numpy() - scenario.wind) / scales['wind'],
        (g.wave.to_numpy() - scenario.wave) / scales['wave'],
        (np.log1p(g.vis_10m.to_numpy()) - np.log1p(scenario.visibility_m / 10.)) / scales['log_visibility'],
    ])
    distance = np.linalg.norm(differences, axis=1) / math.sqrt(3.)
    count = int((distance <= metadata['support_distance_threshold']).sum())
    diagnostic = support_diagnostics(scenario.terminal, scenario.wind, scenario.wave, scenario.visibility_m / 10.)
    assert diagnostic['support_count'] == count == base['support_count']
    assert math.isclose(diagnostic['nearest_distance'], float(distance.min()), abs_tol=1e-14)
    raw_values = [-100., -2., 0., 2., 100.]
    expected_sigmoid = []
    for raw in raw_values:
        z = max(-35., min(35., raw * metadata['calibration']['slope'] + metadata['calibration']['intercept']))
        expected_sigmoid.append(1. / (1. + math.exp(-z)))
    np.testing.assert_allclose(calibrated_probability(raw_values, metadata), expected_sigmoid, rtol=1e-14, atol=1e-15)

    evaluation = metadata['evaluation']
    assert metadata['beats_frequency_baselines'] is False
    assert evaluation['brier'] > evaluation['baseline_brier']
    assert evaluation['brier'] > evaluation['terminal_baseline_brier']
    assert any('개선되지 않았습니다' in text for text in base['warnings'])
    assert '참고' in base['status'], 'Exploratory-model status is missing'
    for name, original in hashes.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == original, 'Check unexpectedly modified a model artifact'
    return {'supported_fixture_probability': base['probability'], 'supported_fixture_n': count,
            'sparse_example_support': sparse['support_count'], 'roc_auc_2025': evaluation['roc_auc'],
            'brier_2025': evaluation['brier']}


def _check_search():
    history = load_case_index()
    s = Scenario('BCT', datetime(2027, 11, 20, 10), 6., 1.2, 5000., 396., 2., 2)
    base = search_cases(s, top_k=30, history=history)
    assert base['matched_count'] == base['returned_count'] == 6
    assert base['shortage']
    assert base['rows'].terminal.eq('BCT').all()
    assert base['rows'].end.lt(s.when).all() and base['rows'].start.lt(s.when).all()
    assert base['rows'].distance.le(1.).all() and base['rows'].distance.is_monotonic_increasing
    assert base['rows'].departure_source.eq('예정 출항 대체').all()
    assert '예정' in base['departure_basis']
    # Selecting similar conditions must not turn into selecting good outcomes.
    different_outcomes = history.copy()
    different_outcomes['work_per_hour'] *= 17
    different_outcomes['duration_h'] *= 2
    modified = search_cases(s, top_k=30, history=different_outcomes)
    assert modified['rows'].case_id.tolist() == base['rows'].case_id.tolist()
    np.testing.assert_array_equal(modified['rows'].distance, base['rows'].distance)
    assert not search_cases(replace(s, when=datetime(2022, 12, 31)))['supported']
    far = search_cases(replace(s, wind=100., wave=30., cargo=1e9, visibility_m=1.))
    assert far['returned_count'] == 0 and far['shortage']
    one = history.query("terminal == 'BCT' and start >= '2023-01-01'").head(1)
    end = one.end.iloc[0].to_pydatetime()
    assert search_cases(replace(s, when=end), history=one)['candidate_count'] == 0
    assert search_cases(replace(s, when=end + pd.Timedelta(seconds=1)), history=one)['candidate_count'] == 1
    doc = (ROOT / 'assets' / 'case_search' / 'README.md').read_text(encoding='utf-8')
    assert '기항별 시각·작업량' in doc and '예정 출항 대체' in doc
    return {'index_n': len(history), 'default_similar_cases': base['returned_count']}


def main():
    risk = _check_risk()
    search = _check_search()
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    private_notice = '비공개' in readme or 'private' in readme.lower()
    print('PASS: risk feature exclusion, input units, recalculation, time cutoff, support abstention, manual calibration, search integrity')
    print(json.dumps({'risk': risk, 'search': search, 'private_distribution_notice': private_notice}, ensure_ascii=False))
    if not private_notice:
        print('NOTICE: README must state that v7 contains call-level records and is for private review before distribution.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
