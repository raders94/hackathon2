"""Past-call retrieval for PORTaI, using input conditions rather than outcomes.

The compact index is a historical research extract, not a live operational feed.
Its ``end`` values preserve the source's calculated departure time, including
scheduled-departure substitutes. Passing an as-of cutoff does not certify actual
work completion or the original publication/availability time of each record.

Distance is a transparent weighted RMS of four robust-scaled numeric differences
and two cyclic differences. It is not a learned probability or causal estimate.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from logic import Scenario, TERMINALS


CASE_ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'case_search'
INDEX_PATH = CASE_ASSETS / 'case_index.csv.gz'
METADATA_PATH = CASE_ASSETS / 'metadata.json'
DISTANCE_FEATURES = ('log_cargo', 'wind', 'wave', 'log_visibility', 'month', 'hour')
DISTANCE_COLUMNS = tuple('distance_' + name for name in DISTANCE_FEATURES)
INDEX_COLUMNS = (
    'case_id', 'terminal', 'start', 'end', 'cargo', 'wind', 'wave',
    'visibility_m', 'weather_observed_at', 'duration_h', 'work_per_hour',
    'departure_source',
)


@lru_cache(maxsize=1)
def _read_metadata() -> dict[str, Any]:
    import json
    return json.loads(METADATA_PATH.read_text(encoding='utf-8'))


def case_search_metadata() -> dict[str, Any]:
    """Return independent metadata, including exact scale fit dates and weights."""
    return deepcopy(_read_metadata())


@lru_cache(maxsize=1)
def _read_index() -> pd.DataFrame:
    return pd.read_csv(
        INDEX_PATH,
        parse_dates=['start', 'end', 'weather_observed_at'],
        dtype={'case_id': str, 'terminal': str, 'departure_source': str},
    )


def load_case_index() -> pd.DataFrame:
    """Load the pseudonymous compact index without allowing cache mutation."""
    return _read_index().copy(deep=True)


def _korean_time(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError('검색 기준시각이 유효하지 않습니다.')
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert('Asia/Seoul').tz_localize(None)
    return timestamp


def _empty_rows(history: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = history.iloc[0:0].copy() if history is not None else pd.DataFrame(columns=INDEX_COLUMNS)
    for name in (*DISTANCE_COLUMNS, 'distance', 'age_days'):
        frame[name] = pd.Series(dtype=float)
    return frame


def _valid_index_rows(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[['cargo', 'wind', 'wave', 'visibility_m', 'duration_h', 'work_per_hour']].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    return (
        pd.Series(finite, index=frame.index)
        & numeric.cargo.gt(0)
        & numeric.wind.ge(0) & numeric.wave.ge(0) & numeric.visibility_m.gt(0)
        & numeric.duration_h.gt(6) & numeric.work_per_hour.gt(0)
        & frame.start.notna() & frame.end.notna()
        & frame.weather_observed_at.notna()
        & frame.end.gt(frame.start)
        & frame.weather_observed_at.le(frame.start)
    )


def _cyclic_distance(values: np.ndarray, query_value: float, period: float) -> np.ndarray:
    """One cyclic group, equivalent to a sin/cos chord divided by two.

    A month's/hour's sine and cosine are not counted as two weighted features.
    Opposite points have distance 1; equal points have distance 0.
    """
    return np.abs(np.sin(np.pi * (values - query_value) / period))


def search_cases(
    scenario: Scenario,
    top_k: int = 30,
    max_distance: float = 1.0,
    history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Retrieve past, same-terminal calls within a fixed condition-distance cap.

    ``candidate_count`` counts eligible same-terminal cases strictly before the
    query under the source's departure convention. ``matched_count`` counts all
    candidates within ``max_distance``, before taking at most ``top_k`` rows.
    ``shortage`` means fewer than ``top_k`` qualified cases; distant cases are
    never substituted to fill the requested number.

    Scale calibration uses records with start >= 2020-01-01 and calculated end
    < 2023-01-01. Consequently historical queries before 2023-01-01 are blocked.
    Neither duration/rate nor their ranks participate in matching or tie breaks.
    Final cargo is a retrospective input; prospective use needs planned cargo.
    """
    if scenario.terminal not in TERMINALS:
        raise ValueError('지원하지 않는 터미널입니다.')
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= 500:
        raise ValueError('top_k는 1~500 사이 정수여야 합니다.')
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError('거리 기준은 0보다 큰 유한값이어야 합니다.')
    inputs = np.asarray([scenario.cargo, scenario.wind, scenario.wave, scenario.visibility_m], dtype=float)
    if not np.isfinite(inputs).all() or scenario.cargo <= 0 or scenario.wind < 0 or scenario.wave < 0 or scenario.visibility_m <= 0:
        raise ValueError('작업량·시정은 양수, 풍속·파고는 0 이상인 유한값이어야 합니다.')

    cutoff = _korean_time(scenario.when)
    metadata = _read_metadata()
    supported_from = pd.Timestamp(metadata['supported_query_from'])
    config = {
        'method': metadata['distance']['method'],
        'weights': deepcopy(metadata['distance']['weights']),
        'scale_fit_start': metadata['scale_fit']['start_inclusive'],
        'scale_fit_end_exclusive': metadata['scale_fit']['end_exclusive'],
        'scales': deepcopy(metadata['scales_by_terminal'][scenario.terminal]),
        'cyclic_formula': metadata['distance']['cyclic_formula'],
        'distance_unit': metadata['distance']['unit'],
        'distance_is_probability': False,
        'threshold_is_validated_accuracy': False,
        'outcome_features_used': False,
        'arrival_prev6_used': False,
        'no_p99_trim': True,
    }
    result: dict[str, Any] = {
        'rows': _empty_rows(), 'candidate_count': 0, 'matched_count': 0,
        'returned_count': 0, 'shortage': True, 'supported': bool(cutoff >= supported_from),
        'message': '', 'cutoff': cutoff, 'distance_threshold': float(max_distance),
        'requested_top_k': int(top_k), 'config': config,
        'distance_components': pd.DataFrame(columns=['case_id', *DISTANCE_COLUMNS, 'distance']),
        'terminal': scenario.terminal, 'reference_start': None, 'reference_end': None,
        'reference_age_days': None, 'invalid_rows_excluded': 0,
        'departure_basis': metadata['departure_basis'],
        'availability_caveat': metadata['availability_caveat'],
        'matched_departure_source_counts': {},
    }
    if cutoff < supported_from:
        result['message'] = '검색 척도는 2020–2022년 자료로 정했습니다. 미래 자료 사용을 피하기 위해 2023년 이후 기준시각만 검색할 수 있습니다.'
        return result

    data = load_case_index() if history is None else history.copy(deep=True)
    missing = set(INDEX_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError('사례 인덱스 필수 열 누락: ' + ', '.join(sorted(missing)))
    for col in ['start', 'end', 'weather_observed_at']:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    # These are source-calculated departure timestamps, not proof of completion.
    data = data.loc[(data.terminal == scenario.terminal) & (data.start < cutoff) & (data.end < cutoff)].copy()
    valid = _valid_index_rows(data)
    result['invalid_rows_excluded'] = int((~valid).sum())
    data = data.loc[valid].copy()
    result['candidate_count'] = len(data)
    if data.empty:
        result['message'] = '해당 터미널에서 기준시각 이전 출항기준·기상 완전관측 조건을 충족한 사례가 없습니다.'
        return result
    result['reference_start'] = data.start.min()
    result['reference_end'] = data.end.max()
    result['reference_age_days'] = float((cutoff - data.end.max()).total_seconds() / 86400)

    scales = config['scales']
    vectors = {
        'log_cargo': (np.log(data.cargo.to_numpy(dtype=float)), float(np.log(scenario.cargo))),
        'wind': (data.wind.to_numpy(dtype=float), float(scenario.wind)),
        'wave': (data.wave.to_numpy(dtype=float), float(scenario.wave)),
        'log_visibility': (np.log1p(data.visibility_m.to_numpy(dtype=float)), float(np.log1p(scenario.visibility_m))),
    }
    for name, (values, query) in vectors.items():
        scale = float(scales[name]['scale'])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f'유효하지 않은 검색 척도: {name}')
        data['distance_' + name] = np.abs(values - query) / scale
    month_values = data.start.dt.month.to_numpy(dtype=float)
    data['distance_month'] = _cyclic_distance(month_values, float(cutoff.month), 12)
    hour_values = (data.start.dt.hour + data.start.dt.minute / 60 + data.start.dt.second / 3600).to_numpy(dtype=float)
    query_hour = cutoff.hour + cutoff.minute / 60 + cutoff.second / 3600
    data['distance_hour'] = _cyclic_distance(hour_values, query_hour, 24)
    weights = config['weights']
    weight_sum = float(sum(weights.values()))
    weighted_squared = np.zeros(len(data), dtype=float)
    for name in DISTANCE_FEATURES:
        weighted_squared += float(weights[name]) * data['distance_' + name].to_numpy(dtype=float) ** 2
    data['distance'] = np.sqrt(weighted_squared / weight_sum)
    data['age_days'] = (cutoff - data.end).dt.total_seconds() / 86400
    matches = data.loc[data.distance <= float(max_distance)].copy()
    result['matched_count'] = len(matches)
    # Source order and chronology cannot preferentially select good outcomes.
    matches = matches.sort_values(['distance', 'start', 'case_id'], ascending=[True, False, True], kind='stable')
    rows = matches.head(int(top_k)).reset_index(drop=True)
    result['rows'] = rows
    result['returned_count'] = len(rows)
    result['shortage'] = len(rows) < int(top_k)
    result['distance_components'] = rows[['case_id', *DISTANCE_COLUMNS, 'distance']].copy()
    result['matched_departure_source_counts'] = {str(k): int(v) for k, v in rows.departure_source.value_counts().items()}
    if rows.empty:
        result['message'] = '거리 기준 안의 유사 사례가 없습니다. 기준을 충족하지 않는 먼 사례로 채우지 않았습니다.'
    elif result['shortage']:
        result['message'] = f'거리 기준을 충족한 사례는 {len(rows):,}건입니다. 요청한 {int(top_k):,}건보다 적어 해당 사례만 표시합니다.'
    else:
        result['message'] = f'거리 기준을 충족한 {len(matches):,}건 중 가장 가까운 {len(rows):,}건을 표시합니다.'
    return result
