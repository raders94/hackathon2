"""Reproduce the frozen PORTaI v7 case index from its original research table.

This is an extraction and distance-scale build, not a predictive-model training
job. The cohort, input transforms, 2020–2022 fit boundary and weights are fixed
to the v7 definition; no cutoff or parameter is selected on evaluation results.

Example:
    python tools/build_case_index.py --input /path/vessels_all_nonGT_2020_2025.pkl.gz --output /path/case_search

Only load a pickle obtained from the trusted research package. Pickle loading
executes Python's deserialization machinery; arbitrary external pickle files
are not a supported input format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'log_cargo': np.log(frame.cargo.astype(float)),
        'wind': frame.wind.astype(float),
        'wave': frame.wave.astype(float),
        'log_visibility': np.log1p(frame.visibility_m.astype(float)),
    })


def build_case_index(source_path: Path, output_dir: Path) -> dict:
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    data = pd.read_pickle(source_path)
    cases = pd.DataFrame({
        'terminal': data.terminal, 'start': data.start, 'end': data.end,
        'cargo': data.cargo, 'wind': data.wind, 'wave': data.wave,
        'visibility_m': data.vis_10m * 10,
        'weather_observed_at': data.wx_time,
        'duration_h': data.duration_h, 'work_per_hour': data.work_per_hour,
        'departure_source': data['터미널_출항시각출처'].fillna('출처 미상'),
    })
    numeric = cases[['cargo', 'wind', 'wave', 'visibility_m', 'duration_h', 'work_per_hour']].astype(float)
    keep = (
        np.isfinite(numeric.to_numpy()).all(axis=1)
        & cases.cargo.gt(0)
        & cases.wind.ge(0) & cases.wave.ge(0) & cases.visibility_m.gt(0)
        & cases.duration_h.gt(6) & cases.work_per_hour.gt(0)
        & cases.start.notna() & cases.end.notna() & cases.weather_observed_at.notna()
        & cases.end.gt(cases.start)
        & cases.weather_observed_at.le(cases.start)
        & cases.start.ge(pd.Timestamp('2020-01-01'))
        & cases.start.lt(pd.Timestamp('2026-01-01'))
    )
    # The extraction's own sequential ID is assigned before chronological
    # sorting. Neither good outcomes nor long/short durations select its order.
    cases = cases.loc[keep].copy().reset_index(drop=True)
    cases.insert(0, 'case_id', [f'CASE-{i:06d}' for i in range(1, len(cases) + 1)])
    cases = cases.sort_values(['start', 'terminal', 'case_id'], kind='stable').reset_index(drop=True)
    fit = cases.loc[
        cases.start.ge(pd.Timestamp('2020-01-01'))
        & cases.end.lt(pd.Timestamp('2023-01-01'))
    ].copy()
    if cases.empty or fit.empty:
        raise ValueError('No valid case records or 2020–2022 scale records.')
    pooled = _numeric_features(fit)
    pooled_iqr = pooled.quantile(.75) - pooled.quantile(.25)
    scales = {}
    for terminal, group in fit.groupby('terminal'):
        features = _numeric_features(group)
        scales[terminal] = {}
        for name in features.columns:
            q25, median, q75 = features[name].quantile([.25, .5, .75]).to_numpy()
            iqr = float(q75 - q25)
            scale = iqr
            scale_source = 'terminal_training_IQR'
            if not np.isfinite(scale) or scale <= 0:
                scale = float(pooled_iqr[name])
                scale_source = 'pooled_training_IQR_fallback'
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'No positive training scale for {name}')
            scales[terminal][name] = {
                'scale': scale, 'iqr': iqr, 'q25': float(q25),
                'median': float(median), 'q75': float(q75),
                'scale_source': scale_source, 'reference_n': len(group),
            }

    metadata = {
        'version': 1,
        'source_file': source_path.name,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'timezone': 'Asia/Seoul (source-naive timestamps interpreted as KST)',
        'supported_query_from': '2023-01-01T00:00:00',
        'index_rows': len(cases), 'input_rows': len(data),
        'excluded_rows': len(data) - len(cases),
        'filters': {
            'positive_cargo': True, 'duration_gt_hours': 6,
            'complete_nonnegative_wind_wave': True, 'positive_visibility': True,
            'positive_finite_outcomes': True,
            'weather_observation_no_later_than_start': True,
            'additional_p99_trim': False,
        },
        'rows_by_terminal': {k: int(v) for k, v in cases.terminal.value_counts().sort_index().items()},
        'departure_source_counts': {k: int(v) for k, v in cases.departure_source.value_counts().items()},
        'departure_source_counts_by_terminal': {
            str(terminal): {str(k): int(v) for k, v in group.departure_source.value_counts().items()}
            for terminal, group in cases.groupby('terminal')
        },
        'start_min': cases.start.min().isoformat(),
        'start_max': cases.start.max().isoformat(),
        'calculated_end_min': cases.end.min().isoformat(),
        'calculated_end_max': cases.end.max().isoformat(),
        'scale_fit': {
            'start_inclusive': '2020-01-01T00:00:00',
            'end_exclusive': '2023-01-01T00:00:00', 'rows': len(fit),
            'rows_by_terminal': {k: int(v) for k, v in fit.terminal.value_counts().sort_index().items()},
            'definition': 'start >= start_inclusive AND calculated end < end_exclusive; same index validity filters',
        },
        'distance': {
            'method': 'weighted_root_mean_squared_input_distance',
            'weights': {'log_cargo': .30, 'wind': .25, 'wave': .20, 'log_visibility': .15, 'month': .05, 'hour': .05},
            'numeric_transforms': {
                'log_cargo': 'natural log(cargo)', 'wind': 'wind m/s',
                'wave': 'significant wave height m',
                'log_visibility': 'natural log(1 + visibility_m)',
            },
            'numeric_scaling': 'absolute query-minus-case difference / terminal training IQR; pooled training IQR only if terminal IQR is zero',
            'cyclic_formula': 'abs(sin(pi * (case - query) / period)); period=12 for month,24 for fractional hour; one weight per cyclic group',
            'unit': 'dimensionless weighted IQR/chord distance; smaller means more similar',
            'default_max_distance': 1.0,
            'threshold_is_validated_probability': False,
            'target_fields_excluded': ['duration_h', 'work_per_hour'],
            'other_excluded_fields': ['arrival_prev6', 'shift_ratio', 'GT', 'ship_name', 'ship_id'],
        },
        'scales_by_terminal': scales,
        'departure_basis': '원자료 출항기준(예정대체포함)',
        'availability_caveat': '계산용 출항시각을 기준시각 이전으로 제한한 후향 검색입니다. 예정 출항 대체가 포함되어 실제 작업 완료나 자료의 당시 공개·확정 여부를 보증하지 않습니다.',
        'cargo_caveat': '과거 최종 작업량이 포함됩니다. 접안 전에는 계획 작업량의 확보와 그 차이에 대한 전향 검증이 필요합니다.',
        'cohort_caveat': '접안 후 체류시간 >6시간 및 접안시점 기상 완전관측 기항에 한정됩니다. 2025년 이벤트시각 기상 기록의 선택성을 포함합니다.',
        'privacy_note': '선박명·원본 선박ID를 제외한 순번 case_id만 포함하지만 기항별 시각·작업량은 남아 있습니다.',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(
        output_dir / 'case_index.csv.gz', index=False,
        compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 0},
        date_format='%Y-%m-%dT%H:%M:%S', float_format='%.12g',
    )
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', '--inputPath', dest='input_path', type=Path, required=True,
                        help='Trusted vessels_all_nonGT_2020_2025.pkl.gz research file')
    parser.add_argument('--output', '--outputDir', dest='output_dir', type=Path, required=True,
                        help='Directory for case_index.csv.gz and metadata.json')
    args = parser.parse_args()
    metadata = build_case_index(args.input_path, args.output_dir)
    print(json.dumps({
        'index_rows': metadata['index_rows'],
        'scale_fit_rows': metadata['scale_fit']['rows'],
        'output_dir': str(args.output_dir.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
