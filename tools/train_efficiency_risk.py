"""Rebuild the PORTaI risk model from the user's private, pre-P99 call dataset.

This script uses no data from 2025 to train, calibrate or choose a threshold.
Run with Python 3.12 and requirements-training.txt; input is not in this release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, roc_auc_score)
import sklearn

SEED = 271109
RISK_FEATURES = ['terminal', 'month', 'weekday', 'berth_hour', 'wind', 'wave', 'vis_10m']
REFERENCE_FEATURES = ['terminal', 'log_cargo', 'month', 'weekday', 'berth_hour']
CALENDAR_FEATURES = ['terminal', 'month', 'weekday', 'berth_hour']
FRACTION = 0.90


def probability(raw, calibration):
    z = np.clip(np.asarray(raw, dtype=float) * calibration['slope'] + calibration['intercept'], -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def calibrate(raw, y):
    fit = LogisticRegression(C=1e6, max_iter=1000, solver='lbfgs', random_state=SEED)
    fit.fit(np.asarray(raw).reshape(-1, 1), y)
    return {'method': 'sigmoid', 'slope': float(fit.coef_[0, 0]), 'intercept': float(fit.intercept_[0])}


def scores(y, p, threshold):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return {
        'n': len(y), 'events': int(y.sum()), 'event_rate': float(y.mean()),
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
        'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'threshold': float(threshold), 'review_rate': float((p >= threshold).mean()),
        'precision': float(tp / (tp + fp)) if tp + fp else 0.0,
        'recall': float(tp / (tp + fn)) if tp + fn else 0.0,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def cohort(df, first, last):
    # Purge rows whose outcome period overlaps the next training/calibration boundary.
    return df.loc[(df.start >= pd.Timestamp(f'{first}-01-01')) &
                  (df.start < pd.Timestamp(f'{last+1}-01-01')) &
                  (df.end < pd.Timestamp(f'{last+1}-01-01'))].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets' / 'risk')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(args.input)
    d = raw.copy()
    d['start'] = pd.to_datetime(d.start)
    d['end'] = pd.to_datetime(d.end)
    d['wx_time'] = pd.to_datetime(d.wx_time)
    d['weekday'] = d.start.dt.weekday.astype(int)
    d['month'] = d.start.dt.month.astype(int)
    d['berth_hour'] = d.start.dt.hour.astype(int)
    d['terminal'] = d.terminal.astype(str)
    d['log_cargo'] = np.log(d.cargo.astype(float))
    d['work_per_hour'] = d.cargo / d.duration_h
    numeric = ['wind', 'wave', 'vis_10m', 'duration_h', 'cargo', 'work_per_hour']
    valid = np.isfinite(d[numeric]).all(axis=1) & d.cargo.gt(0) & d.duration_h.gt(6)
    valid &= d.wind.ge(0) & d.wave.ge(0) & d.vis_10m.gt(0)
    valid &= d.start.notna() & d.end.notna() & d.wx_time.notna() & d.wx_time.le(d.start)
    valid &= (d.start - d.wx_time).le(pd.Timedelta(hours=1)) & d.end.gt(d.start)
    d = d.loc[valid].copy()
    # Exact duplicates are not independent calls; no broader ship-based deduplication is possible.
    dupe_keys = ['terminal', 'start', 'end', 'cargo', 'wind', 'wave', 'vis_10m']
    duplicate_count = int(d.duplicated(dupe_keys).sum())
    d = d.drop_duplicates(dupe_keys).sort_values(['start', 'terminal'], kind='stable')
    base, train, cal, test = (cohort(d, a, b) for a, b in [(2020, 2022), (2023, 2023), (2024, 2024), (2025, 2025)])
    assert base.end.max() < train.start.min() and train.end.max() < cal.start.min() and cal.end.max() < test.start.min()

    # Recompute all calm thresholds from the reference period, never reuse 2020–24 thresholds.
    weather_thresholds = {}
    calm_mask = pd.Series(False, index=base.index)
    for harbor, g in base.groupby('harbor'):
        cut = {'wind_p90': float(g.wind.quantile(.90)), 'wave_p90': float(g.wave.quantile(.90)),
               'vis_p10': float(g.vis_10m.quantile(.10)), 'n': len(g)}
        weather_thresholds[str(harbor)] = cut
        calm_mask.loc[g.index] = g.wind.lt(cut['wind_p90']) & g.wave.lt(cut['wave_p90']) & g.vis_10m.gt(cut['vis_p10'])
    calm = base.loc[calm_mask].copy()
    reference = CatBoostRegressor(iterations=350, depth=5, learning_rate=.045,
                                  loss_function='Quantile:alpha=0.5', l2_leaf_reg=8,
                                  random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False)
    reference.fit(calm[REFERENCE_FEATURES], np.log(calm.work_per_hour), cat_features=['terminal'])
    reference.save_model(str(out / 'calm_reference_2020_2022.cbm'))
    for frame in [train, cal, test]:
        frame['reference_rate'] = np.exp(reference.predict(frame[REFERENCE_FEATURES]))
        frame['efficiency_ratio'] = frame.work_per_hour / frame.reference_rate
        frame['event'] = frame.efficiency_ratio.lt(FRACTION).astype(int)

    model = CatBoostClassifier(iterations=300, depth=4, learning_rate=.04, loss_function='Logloss',
                               l2_leaf_reg=10, random_seed=SEED, thread_count=4,
                               verbose=False, allow_writing_files=False)
    model.fit(train[RISK_FEATURES], train.event, cat_features=['terminal'])
    model.save_model(str(out / 'risk_classifier_2023.cbm'))
    calibration = calibrate(model.predict(cal[RISK_FEATURES], prediction_type='RawFormulaVal'), cal.event)
    cal_p = probability(model.predict(cal[RISK_FEATURES], prediction_type='RawFormulaVal'), calibration)
    test_p = probability(model.predict(test[RISK_FEATURES], prediction_type='RawFormulaVal'), calibration)
    threshold = float(np.quantile(cal_p, .80))

    # Prespecified benchmark, not a model selected after seeing evaluation results.
    no_weather = CatBoostClassifier(iterations=300, depth=4, learning_rate=.04, loss_function='Logloss',
                                    l2_leaf_reg=10, random_seed=SEED, thread_count=4,
                                    verbose=False, allow_writing_files=False)
    no_weather.fit(train[CALENDAR_FEATURES], train.event, cat_features=['terminal'])
    nw_cal = calibrate(no_weather.predict(cal[CALENDAR_FEATURES], prediction_type='RawFormulaVal'), cal.event)
    nw_p = probability(no_weather.predict(test[CALENDAR_FEATURES], prediction_type='RawFormulaVal'), nw_cal)
    overall_cal_rate = float(cal.event.mean())
    terminal_cal_rate = cal.groupby('terminal').event.mean().to_dict()
    constant_p = np.full(len(test), overall_cal_rate)
    terminal_p = test.terminal.map(terminal_cal_rate).fillna(overall_cal_rate).to_numpy()
    result = scores(test.event, test_p, threshold)
    result.update(model='weather_calibrated', baseline_brier=float(brier_score_loss(test.event, constant_p)),
                  terminal_baseline_brier=float(brier_score_loss(test.event, terminal_p)))
    benchmark_rows = []
    for name, p in [('weather_calibrated', test_p), ('calendar_no_weather', nw_p),
                    ('constant_2024_rate', constant_p), ('terminal_2024_rate', terminal_p)]:
        benchmark_rows.append({'model': name, **scores(test.event, p, threshold)})
    pd.DataFrame(benchmark_rows).to_csv(out / 'evaluation_benchmarks_2025.csv', index=False)
    test['probability'] = test_p
    test['baseline_probability'] = constant_p
    test['terminal_baseline_probability'] = terminal_p

    validation_rows = []
    groupings = [('전체', test)] + [(t, g) for t, g in test.groupby('terminal')]
    for label, g in groupings:
        validation_rows.append({'terminal': label, **scores(g.event, g.probability, threshold),
                                'baseline_brier': float(brier_score_loss(g.event, g.baseline_probability)),
                                'terminal_baseline_brier': float(brier_score_loss(g.event, g.terminal_baseline_probability)),
                                'scheduled_departure_share': float(g['터미널_출항시각출처'].eq('예정 출항 대체').mean())})
    pd.DataFrame(validation_rows).to_csv(out / 'evaluation_by_terminal_2025.csv', index=False)
    test['probability_bin'] = pd.cut(test.probability, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration_bins = test.groupby('probability_bin', observed=True).agg(
        n=('event', 'size'), events=('event', 'sum'), mean_probability=('probability', 'mean'),
        observed_rate=('event', 'mean')).reset_index()
    calibration_bins.to_csv(out / 'calibration_2025.csv', index=False)
    source_rows = []
    for source, g in test.groupby('터미널_출항시각출처'):
        source_rows.append({'departure_source': source, **scores(g.event, g.probability, threshold)})
    pd.DataFrame(source_rows).to_csv(out / 'evaluation_by_departure_source_2025.csv', index=False)
    severity = []
    for harbor, g in test.groupby('harbor'):
        c = weather_thresholds[harbor]
        exceed = g.wind.ge(c['wind_p90']) | g.wave.ge(c['wave_p90']) | g.vis_10m.le(c['vis_p10'])
        for is_high, label in [(False, '기준 이내'), (True, '기상 1개 이상 기준 초과')]:
            f = g.loc[exceed.eq(is_high)]
            if len(f):
                severity.append({'harbor': harbor, 'weather_group': label, **scores(f.event, f.probability, threshold)})
    pd.DataFrame(severity).to_csv(out / 'evaluation_by_weather_2025.csv', index=False)

    coverage_rows = []
    for phase, frame in [('기준모형 2020–22', base), ('평온 기준학습 2020–22', calm),
                         ('분류학습 2023', train), ('확률보정 2024', cal), ('후향평가 2025', test)]:
        for terminal, g in frame.groupby('terminal'):
            coverage_rows.append({'phase': phase, 'terminal': terminal, 'n': len(g),
                                  'event_rate': float(g.event.mean()) if 'event' in g else None,
                                  'event_rate_20pct': float(g.efficiency_ratio.lt(.80).mean()) if 'efficiency_ratio' in g else None,
                                  'scheduled_departures': int(g['터미널_출항시각출처'].eq('예정 출항 대체').sum()),
                                  'original_p99_excluded_kept': int((~g.op_trim_p99).sum())})
    pd.DataFrame(coverage_rows).to_csv(out / 'cohort_coverage.csv', index=False)

    # Minimal 2023 predictor-only reference, used for support diagnostics, never for outcomes.
    support = train[['terminal', 'wind', 'wave', 'vis_10m']].copy()
    support.to_csv(out / 'support_reference_2023.csv.gz', index=False, compression='gzip')
    support_scales = {}
    for terminal, g in train.groupby('terminal'):
        transformed = pd.DataFrame({'wind': g.wind, 'wave': g.wave, 'log_visibility': np.log1p(g.vis_10m)})
        floors = {'wind': .5, 'wave': .15, 'log_visibility': .3}
        support_scales[terminal] = {
            'scales': {c: float(max(transformed[c].quantile(.75)-transformed[c].quantile(.25), floors[c])) for c in transformed},
            'bounds': {c: [float(g[c].min()), float(g[c].max())] for c in ['wind', 'wave', 'vis_10m']},
            'n': len(g),
        }
    # Evaluate the actual display policy as well as the raw model, without
    # selecting bounds/radius/minimum after observing evaluation performance.
    test['support_count'] = 0
    test['display_supported'] = False
    for terminal, g in test.groupby('terminal'):
        ref = support.loc[support.terminal.eq(terminal)]
        conf = support_scales[terminal]
        scale = np.array([conf['scales'][c] for c in ['wind', 'wave', 'log_visibility']])
        ref_x = np.column_stack([ref.wind, ref.wave, np.log1p(ref.vis_10m)]) / scale
        q_x = np.column_stack([g.wind, g.wave, np.log1p(g.vis_10m)]) / scale
        counts = []
        for offset in range(0, len(g), 128):
            distance = np.sqrt(np.mean((q_x[offset:offset+128, None, :] - ref_x[None, :, :]) ** 2, axis=2))
            counts.extend((distance <= 1.0).sum(axis=1).tolist())
        in_bounds = np.ones(len(g), dtype=bool)
        for c in ['wind', 'wave', 'vis_10m']:
            lo, hi = conf['bounds'][c]
            in_bounds &= g[c].between(lo, hi).to_numpy()
        test.loc[g.index, 'support_count'] = counts
        test.loc[g.index, 'display_supported'] = in_bounds & (np.asarray(counts) >= 20)
    display_rows = []
    for label, g in [('전체', test)] + [(t, g) for t, g in test.groupby('terminal')]:
        supported = g.loc[g.display_supported]
        row = {'terminal': label, 'all_n': len(g), 'display_coverage': len(supported)/len(g)}
        if len(supported):
            row.update(scores(supported.event, supported.probability, threshold))
            row['terminal_baseline_brier'] = float(brier_score_loss(supported.event, supported.terminal_baseline_probability))
        display_rows.append(row)
    pd.DataFrame(display_rows).to_csv(out / 'evaluation_display_supported_2025.csv', index=False)
    better = result['brier'] < min(result['baseline_brier'], result['terminal_baseline_brier'])
    validation_status = ('후향 평가에서 단순 발생률 기준보다 Brier 오차가 작습니다. 현장 전향 검증은 아직 없습니다.' if better else
                         '단순 발생률 기준 대비 확률오차 개선을 확인하지 못했습니다. 탐색적 참고 모델입니다.')
    metadata = {
        'project': 'PORTaI', 'version': '7', 'seed': SEED,
        'label_description': '2020–2022년 평온조건 기준 중앙 작업률보다 10%를 초과해 낮은 파생 작업률',
        'label_formula': 'cargo / duration_h < 0.90 × exp(calm_reference(log_cargo, terminal, calendar))',
        'label_fraction': FRACTION, 'label_threshold_is_design_choice': True,
        'data_scope': '원자료에서 체류시간이 6시간을 초과하고 작업량·접안시 기상이 유효한 기항. 신항 5개 터미널은 예정 출항시각 대체 지표.',
        'features': RISK_FEATURES, 'reference_features': REFERENCE_FEATURES,
        'excluded_predictors': ['최종 작업량', '최종 Shift 비율', '직전6시간 혼잡도', '전체 체류시간', '전체 체류 중 기상', 'GT'],
        'periods': {'reference': '2020–2022', 'classifier': '2023', 'calibration': '2024', 'evaluation': '2025'},
        'available_from': '2025-01-01', 'data_end': '2025-12-31',
        'calibration': calibration, 'review_threshold': threshold,
        'threshold_description': '2024년 보정자료 예측확률의 상위 20% 경계. 검토 우선순위를 위한 설계선이며 안전·작업중단 기준이 아닙니다.',
        'validation_status': validation_status, 'beats_frequency_baselines': better,
        'evaluation': result, 'overall_rate': float(test.event.mean()),
        'terminal_rates': test.groupby('terminal').event.mean().to_dict(),
        'terminal_calibration_rates': terminal_cal_rate, 'calibration_event_rate': overall_cal_rate,
        'calm_thresholds': weather_thresholds, 'calm_threshold_source': '2020–22 기준모형 코호트의 항만별 접안시 스냅샷 분위수. 기존 모델 B 기준과 다릅니다.',
        'support_description': '2023년 분류학습 자료 중 같은 터미널, 풍속·파고·log(1+시정)의 IQR 표준화 거리(RMS) 1 이내 사례 수. 최소20건 필요.',
        'support_minimum': 20, 'support_distance_threshold': 1.0, 'support_scales': support_scales,
        'display_support_evaluation': display_rows[0],
        'typical_weather': {t: {'wind':float(g.wind.median()), 'wave':float(g.wave.median()),
                                'visibility_m':float(g.vis_10m.median()*10)} for t, g in train.groupby('terminal')},
        'reference_calm_counts': {t:int(len(g)) for t, g in calm.groupby('terminal')},
        'counts': {'input': len(raw), 'valid_complete_before_dedup': int(valid.sum()), 'duplicate_rows_removed': duplicate_count,
                   'reference': len(base), 'calm_reference': len(calm), 'train': len(train), 'calibration': len(cal), 'test': len(test),
                   'test_original_p99_excluded_kept': int((~test.op_trim_p99).sum())},
        'temporal_bounds': {name: {'start_min': str(frame.start.min()), 'start_max': str(frame.start.max()), 'end_max': str(frame.end.max())}
                            for name, frame in [('reference', base), ('train', train), ('calibration', cal), ('test', test)]},
        'limitations': [
            '시간순 후향 평가이며 실시간 접안 전 경보의 전향 검증이 아닙니다.',
            '평온 기준은 2020–22년에 고정했습니다. 이후 숙련·선박구성·운영정책 변화와 기준모형 오차도 label에 반영될 수 있습니다.',
            'BCT 기준모형은 2022년 개장 초기의 소수 평온 표본에 의존합니다.',
            '접안시 기상 관측시각은 접안시각 이하이나 실제 자료 전달 지연은 확인하지 못했습니다.',
            '자료는 체류 6시간 초과 기항으로 이미 선별되어, 모든 접안 예정선박으로 일반화할 수 없습니다.',
            '최종 작업량은 사후 label의 기준 보정에만 사용하며 위험 예측 입력에는 사용하지 않습니다.',
            '신항 5개 터미널의 출항시간은 예정값 대체이므로 실제 작업중단·실제 효율손실 확률이 아닙니다.',
            'P99 절단을 추가하지 않았으나 원자료 이전 전처리에서 제외된 극단·오류 기항은 복원할 수 없습니다.',
            '2025년 기상 결측으로 평가집단 구성이 달라집니다. 터미널별 표본·성능을 함께 확인해야 합니다.',
        ],
        'source': {'filename': args.input.name, 'sha256': hashlib.sha256(args.input.read_bytes()).hexdigest()},
        'versions': {'python': platform.python_version(), 'catboost': catboost.__version__, 'pandas': pd.__version__,
                     'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'training_parameters': {'reference_iterations':350, 'reference_depth':5, 'classifier_iterations':300,
                                'classifier_depth':4, 'class_weights':'none', 'test_based_tuning':False},
    }
    write_json(out / 'metadata.json', metadata)
    print(json.dumps({'counts': metadata['counts'], 'calibration': calibration, 'threshold': threshold,
                      'evaluation': result, 'benchmarks': benchmark_rows, 'validation_status': validation_status}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
