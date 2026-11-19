"""Case retrieval and a separately validated efficiency-shortfall risk model.

These screens share scenario inputs, not outcomes.  Retrieval ranks conditions;
risk probabilities come only from the packaged risk engine.  A recorded end
can be a planned-departure substitute, so it is never labelled an actual finish.
"""
from __future__ import annotations

from datetime import date, datetime
import math

import pandas as pd
import streamlit as st

from logic import Scenario, TERMINALS
from decision_visuals import case_raincloud, risk_donut, brier_comparison
from ui import page_header, takeaway, metrics, chart, section, source_note

DEFAULT_SCENARIO = Scenario('BCT', datetime(2027, 11, 20, 10), 6.0, 1.2, 5000, 396, 2, 2)
PROXY_TERMINALS = {'BCT', 'BNCT', 'HPNT', 'PNC', 'PNIT'}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pct(value, places=1):
    return f'{100 * float(value):.{places}f}%' if _finite(value) else '자료 없음'


def _copy_prediction():
    saved = st.session_state.get('latest_prediction')
    if not saved:
        return
    s = saved['scenario']
    st.session_state['decision_scenario'] = s
    for key, value in _widget_values(s).items():
        st.session_state[key] = value


def _apply_typical_weather(weather):
    st.session_state['decision_wind'] = round(float(weather['wind']), 1)
    st.session_state['decision_wave'] = round(round(float(weather['wave']) / .05) * .05, 2)
    st.session_state['decision_visibility'] = int(round(float(weather['visibility_m']) / 100) * 100)


def _widget_values(s):
    return {
        'decision_terminal': s.terminal,
        'decision_day': s.when.date(),
        'decision_hour': s.when.hour,
        'decision_wind': float(s.wind),
        'decision_wave': float(s.wave),
        'decision_visibility': int(s.visibility_m),
        'decision_cargo': float(s.cargo),
    }


def _inputs(kind, navigate):
    """Persist the current scenario even when Streamlit cleans hidden widgets."""
    saved = st.session_state.get('latest_prediction')
    previous = st.session_state.get('decision_scenario') or (saved['scenario'] if saved else DEFAULT_SCENARIO)
    for key, value in _widget_values(previous).items():
        if key not in st.session_state:
            st.session_state[key] = value
    from risk_engine import load_risk_metadata
    selected_terminal = st.session_state.get('decision_terminal', previous.terminal)
    typical = load_risk_metadata().get('typical_weather', {}).get(selected_terminal)
    if typical:
        button_col, note_col = st.columns([1, 2])
        with button_col:
            st.button('관측이 많은 날씨 예시', key=f'{kind}_typical_weather',
                      on_click=_apply_typical_weather, args=(typical,), width='stretch')
        with note_col:
            st.caption(f'{selected_terminal}의 2023년 풍속·파고·시정 중앙값을 화면 눈금에 맞춰 입력합니다. 날짜·시간·작업량은 유지합니다.')
    with st.expander('조건 바꾸기 · 두 새 기능에서 같은 조건을 사용합니다', expanded=False):
        left, right = st.columns([1, 2])
        with left:
            st.button('AI 예측 조건 가져오기', on_click=_copy_prediction, key=f'{kind}_copy_prediction',
                      disabled=not bool(saved), width='stretch')
        with right:
            st.caption('‘AI 예측’에서 마지막으로 계산한 조건을 불러옵니다. 이 화면에서 바꾼 조건은 두 새 기능에 함께 적용됩니다.')
        a, b, c = st.columns([1, 1.25, .8])
        with a:
            terminal = st.selectbox('조회 터미널', TERMINALS, key='decision_terminal')
        with b:
            day = st.date_input('가정 접안 날짜 · 한국시간', min_value=date(2020, 1, 1), max_value=date(2030, 12, 31), key='decision_day')
        with c:
            hour = st.selectbox('가정 접안 시각', list(range(24)), format_func=lambda h: f'{h:02d}:00', key='decision_hour')
        x, y, z = st.columns(3)
        with x:
            wind = st.slider('접안시 풍속 (m/s)', 0., 18., step=.1, key='decision_wind')
        with y:
            wave = st.slider('접안시 유의파고 (m)', 0., 5., step=.05, key='decision_wave')
        with z:
            visibility = st.slider('접안시 시정 (m)', 0, 50000, step=100, key='decision_visibility')
        if kind == 'cases':
            cargo = st.number_input('검색할 작업량 (원자료 단위)', min_value=1., max_value=15000., step=10., key='decision_cargo')
            st.caption('검색의 작업량은 시나리오 입력입니다. 과거 사례에는 최종 작업량이 기록되어 있으며, 계획 작업량과의 차이는 이 자료로 검증하지 못했습니다.')
        else:
            cargo = float(previous.cargo)
            st.caption('위험 모델 입력은 터미널·월·요일·접안시각·풍속·파고·시정입니다. 최종 작업량, Shift, 출항 후 정보와 혼잡도는 입력에 사용하지 않습니다.')
        st.caption('날씨는 직접 지정하는 조건입니다. 실시간 예보나 입항정보를 자동 수신하지 않습니다.')
    scenario = Scenario(terminal, datetime.combine(day, datetime.min.time()).replace(hour=hour),
                        float(wind), float(wave), float(visibility), float(cargo),
                        previous.arrival_prev6, previous.active_ships, previous.gt)
    st.session_state['decision_scenario'] = scenario
    return scenario


def _scenario_caption(s, cargo=False):
    detail = f'{s.terminal} · {s.when:%Y-%m-%d %H:00} KST · 풍속 {s.wind:.1f} m/s · 파고 {s.wave:.2f} m · 시정 {s.visibility_m:,.0f} m'
    if cargo:
        detail += f' · 작업량 {s.cargo:,.0f}'
    return detail


def _departure_note(terminal):
    if terminal in PROXY_TERMINALS:
        st.info(f'{terminal}의 원자료에는 실제 출항시각이 없어 예정 출항시각으로 체류시간을 계산했습니다. 아래 작업률·효율저하도 이 대체시간에 기반한 지표입니다.')
    else:
        st.caption('작업률은 총작업량을 접안 후 출항 기준까지의 체류시간으로 나눈 값입니다. 개별 사례의 출항시각 출처는 상세표에서 확인할 수 있습니다.')


def render_cases(D, META, navigate):
    from case_search import load_case_index, search_cases

    page_header('02 / 유사 기항', '비슷한 조건의 과거 기항은 어떤 결과였을까요?',
                '같은 터미널에서 작업량·날씨·계절·시간이 가까운 사례를 찾습니다. 결과가 좋거나 나쁜지는 검색 순위에 넣지 않습니다.',
                kind='과거 사례 검색')
    result_slot = st.container()
    s = _inputs('cases', navigate)
    with st.expander('검색 범위 조정', expanded=False):
        a, b = st.columns(2)
        with a:
            top_k = st.selectbox('분포에 표시할 최대 사례 수', [20, 30, 50], index=1, key='cases_topk')
        with b:
            radius_label = st.selectbox('조건 차이 허용 범위', ['기본 · 거리 1.00 이하', '엄격 · 거리 0.75 이하', '확장 · 거리 1.25 이하'], key='cases_radius')
            radius = {'기본 · 거리 1.00 이하': 1.0, '엄격 · 거리 0.75 이하': .75, '확장 · 거리 1.25 이하': 1.25}[radius_label]
        st.caption('거리는 여러 입력의 차이를 묶은 검색 기준입니다. 작을수록 조건이 가깝습니다. 거리 1이 유사도 100%를 뜻하지는 않습니다.')
    try:
        result = search_cases(s, top_k=int(top_k), max_distance=radius, history=load_case_index())
    except ValueError as exc:
        result = {'rows': pd.DataFrame(), 'supported': False, 'message': str(exc),
                  'candidate_count': 0, 'matched_count': 0, 'config': {}}
    rows = result['rows'].copy()
    with result_slot:
        if not result.get('supported', True):
            takeaway('이 조건에서는 유사 사례 검색을 제공하지 않습니다.', result.get('message', ''), tone='slate')
            st.caption(_scenario_caption(s, True))
        elif rows.empty:
            takeaway('지정한 조건을 충족하는 유사 사례가 없습니다.',
                     f'조회 가능한 과거 {int(result.get("candidate_count", 0)):,}건에서 거리 {radius:.2f} 이하 사례를 찾지 못했습니다. 조건 차이가 큰 기항을 대신 채우지 않습니다.', 'slate')
            st.caption(_scenario_caption(s, True))
        else:
            n = len(rows)
            matched = int(result.get('matched_count', n))
            median = float(rows.work_per_hour.median())
            q1, q3 = rows.work_per_hour.quantile([.25, .75])
            takeaway(f'{s.terminal} · 유사한 {n:,}건의 작업률 중앙값은 {median:.1f}/h입니다.',
                     f'과거 검색 대상 {int(result.get("candidate_count", n)):,}건 중 {matched:,}건이 거리 기준을 충족했습니다. 이 가운데 조건이 가까운 {n:,}건을 표시합니다.',
                     'blue' if n >= 20 else 'amber')
            st.caption(_scenario_caption(s, True))
            metrics([
                dict(label='분포에 표시한 유사 사례', value=f'{n:,}건', note=f'조건 차이 허용 거리 ≤ {radius:.2f}', tone='blue' if n >= 20 else 'amber'),
                dict(label='과거 작업률 중앙값', value=f'{median:.1f} /h', note=f'중간 50%는 {q1:.1f}~{q3:.1f}/h', tone='blue'),
                dict(label='과거 체류시간 중앙값', value=f'{rows.duration_h.median():.1f}시간', note='예정 출항 대체 사례가 포함될 수 있습니다.', tone='slate'),
            ])
            section('어디에 모여 있고, 얼마나 흩어졌을까요?', '가운데 박스는 중간 50%, 아래 점 하나는 과거 기록 한 건입니다. 진한 점일수록 입력 조건이 더 가깝습니다.')
            fig, density_shown = case_raincloud(rows, max_distance=radius)
            chart(fig)
            if density_shown:
                st.caption('위의 부드러운 모양은 검색된 작업률의 밀도 추정입니다. 20건 이상에서만 표시하며 관측 범위 밖으로 늘리지 않습니다. 선택 조건의 예측구간이나 신뢰구간이 아닙니다.')
            else:
                st.caption('밀도 곡선을 그리지 않고 실제 기록의 위치와 중간 50%만 보여줍니다. 점의 세로 위치는 겹침을 줄이기 위한 배치이며, 가로 위치는 기록된 작업률 그대로입니다.')
            if result.get('shortage', False) or n < 20:
                st.warning(f'조건을 충족하는 사례가 {n:,}건입니다. 이 작은 표본의 중앙값과 범위를 일반적인 결과로 확대 해석하지 마세요.')
    _departure_note(s.terminal)
    if not rows.empty:
        section('가장 가까운 사례부터 확인하기', '조회 시점 이전에 원자료상 출항 기준을 지난 기항만 검색합니다. 과거 실제 운영조치나 대응 방법은 포함되어 있지 않습니다.')
        table = rows.loc[:, ['case_id', 'start', 'cargo', 'wind', 'wave', 'visibility_m', 'work_per_hour', 'duration_h', 'departure_source', 'distance']].copy()
        table['start'] = pd.to_datetime(table['start']).dt.strftime('%Y-%m-%d %H:%M')
        table = table.rename(columns={'case_id': '사례 ID', 'start': '접안 시각', 'cargo': '최종 작업량',
                                      'wind': '풍속 m/s', 'wave': '파고 m', 'visibility_m': '시정 m',
                                      'work_per_hour': '작업률 /h', 'duration_h': '체류시간 h',
                                      'departure_source': '출항시각 출처', 'distance': '조건 거리'})
        st.dataframe(table.head(5).round(2), hide_index=True, width='stretch')
        with st.expander(f'검색된 {len(rows):,}건 전체 · 조건 차이 · CSV'):
            st.dataframe(table.round(3), hide_index=True, width='stretch')
            dims = [c for c in rows.columns if c.startswith('distance_')]
            if dims:
                st.markdown('**입력별 거리 구성**')
                st.dataframe(rows[['case_id', 'distance'] + dims].round(4), hide_index=True, width='stretch')
            st.caption('정렬은 입력 조건의 거리만 사용합니다. 작업률·체류시간·효율저하 여부는 검색 거리 계산에 포함하지 않습니다.')
            st.download_button('유사 기항 결과 CSV', table.to_csv(index=False).encode('utf-8-sig'),
                               file_name='PORTaI_similar_calls.csv', mime='text/csv', key='case_results_download')
    with st.expander('검색 방법과 적용 범위'):
        st.markdown('**같은 터미널 → 조회 시점 이전 기록 → 조건 거리 기준 → 가까운 순으로 최대 표시 건수**의 순서로 검색합니다.')
        st.markdown('작업량과 시정은 로그 변환하고, 풍속·파고·월·접안시각의 차이를 함께 비교합니다. 월·시각은 12월과 1월, 23시와 0시가 가까운 순환 구조로 처리합니다.')
        st.markdown('표준화 기준은 2020–2022년 자료에서 고정했습니다. 따라서 이 검색기는 2023년 이후 조회 시점만 지원합니다. 검색 대상은 기상 입력이 완전하고 유효 체류시간이 6시간을 초과한 기항입니다. 상위 1%라는 이유만으로 긴 체류 사례를 제거하지 않았습니다.')
        st.markdown('예정 출항 대체 시각은 실제 출항 확인과 다릅니다. 당시 데이터가 시스템에 전달된 시점도 기록되어 있지 않아 완전한 실시간 이력 재현은 아닙니다. 검색된 결과는 과거 관측이며, 높은 성과를 만든 대응책이나 인과효과를 알려주지 않습니다.')
        st.json(result.get('config', {}), expanded=False)
    st.button('같은 조건의 효율저하 위험 보기 →', key='cases_to_risk', on_click=navigate,
              args=('효율저하 위험',), width='stretch')
    source_note('case_search/case_index.csv.gz · case_search/metadata.json · 기항 단위 비공개 파생자료')


def _risk_metrics(result):
    ev = result.get('evaluation', {})
    n = ev.get('n')
    auc = ev.get('roc_auc')
    ap = ev.get('average_precision')
    brier = ev.get('brier')
    baseline = ev.get('baseline_brier')
    terminal_baseline = ev.get('terminal_baseline_brier')
    out = []
    if _finite(auc):
        out.append(dict(label='2025년 전체 판별력 · ROC-AUC', value=f'{float(auc):.3f}',
                        note='0.5는 무작위 순위 수준 · 클수록 좋음', tone='blue'))
    if _finite(brier):
        note = (f'터미널별 단순 기준 {float(terminal_baseline):.3f} · 작을수록 좋음' if _finite(terminal_baseline)
                else (f'전체 단순 기준 {float(baseline):.3f} · 작을수록 좋음' if _finite(baseline)
                      else '확률과 사건 발생의 오차 · 작을수록 좋음'))
        out.append(dict(label='2025년 전체 확률오차 · Brier', value=f'{float(brier):.3f}', note=note, tone='blue'))
    if _finite(n):
        out.append(dict(label='2025년 평가 사례', value=f'{int(n):,}건',
                        note='2023 학습 · 2024 확률 보정 · 2025 평가', tone='slate'))
    if out:
        metrics(out)
    return ev


def render_risk(D, META, navigate):
    from risk_engine import predict_risk, load_risk_metadata, load_risk_validation

    page_header('03 / 효율저하 위험', '평균값 뒤에 숨은 큰 효율저하 가능성을 봅니다',
                '평온한 운영조건의 기준 중앙 작업률보다 10% 넘게 낮아지는 사건을 별도 분류모델로 예측합니다. 화면의 확률은 날씨가 원인일 확률이나 안전사고 위험이 아닙니다.',
                kind='탐색적 위험모델')
    result_slot = st.container()
    s = _inputs('risk', navigate)
    result = predict_risk(s)
    metadata = result.get('metadata') or load_risk_metadata()
    supported = bool(result.get('supported', False))
    probability = result.get('probability')
    threshold = result.get('threshold')
    validation = load_risk_validation()
    terminal_eval = {}
    if isinstance(validation, pd.DataFrame) and not validation.empty and 'terminal' in validation:
        selected = validation.loc[validation.terminal.eq(s.terminal)]
        if not selected.empty:
            terminal_eval = selected.iloc[0].to_dict()
    with result_slot:
        quality_message = str(metadata.get('validation_status', ''))
        if _finite(terminal_eval.get('roc_auc')) and float(terminal_eval['roc_auc']) <= .5:
            quality_message += f' {s.terminal}의 ROC-AUC도 {float(terminal_eval["roc_auc"]):.3f}로, 이 터미널의 위험 순위 판별력을 확인하지 못했습니다.'
        if quality_message:
            st.warning(quality_message)
        if not supported or not _finite(probability):
            takeaway('이 조건에서는 위험 확률을 제시하지 않습니다.',
                     result.get('status', '자료지원 범위를 확인해주세요.'), 'slate')
            st.caption(_scenario_caption(s))
            metrics([
                dict(label='위험 확률', value='판단 보류', note='자료지원 조건을 충족해야 표시합니다.', tone='slate'),
                dict(label='가까운 학습 조건 사례', value=f'{int(result.get("support_count", 0)):,}건',
                     note=f'2023년 학습 · 비슷한 날씨 · 최소 {int(metadata.get("support_minimum", 20))}건', tone='amber'),
                dict(label='같은 터미널의 2025년 발생비율', value=_pct(result.get('terminal_rate')),
                     note='현재 입력의 확률을 대신하는 값은 아닙니다.', tone='slate'),
            ])
        else:
            p = min(1., max(0., float(probability)))
            above = _finite(threshold) and p >= float(threshold)
            tone = 'blue'
            takeaway(f'{s.terminal} · 큰 효율저하의 참고용 예측확률은 {p * 100:.1f}%입니다.',
                     f'{result.get("status", "시나리오 참고값")} · 관측자료로 학습한 조건부 확률입니다. 운영 중 경보로 쓰기 위한 전향 검증은 별도입니다.', tone)
            st.caption(_scenario_caption(s))
            metrics([
                dict(label='입력 조건의 효율저하 확률', value=_pct(p), note='별도 분류모델 + 보정된 확률', tone=tone),
                dict(label='같은 터미널의 2025년 발생비율', value=_pct(result.get('terminal_rate')),
                     note='전체 기상·운영조건이 섞인 관측 비율', tone='slate'),
                dict(label='가까운 학습 조건 사례', value=f'{int(result.get("support_count", 0)):,}건',
                     note=f'2023년 학습 · 비슷한 날씨 · 최소 {int(metadata.get("support_minimum", 20))}건', tone='slate'),
            ])
            if terminal_eval:
                st.caption(f'{s.terminal}의 2025년 검증: {int(terminal_eval["n"]):,}건 · ROC-AUC {float(terminal_eval["roc_auc"]):.3f} · Brier {float(terminal_eval["brier"]):.3f} · 사건 발생비율 {_pct(terminal_eval["event_rate"])}. 전체 모델 성능과 별도로 확인하세요.')
            section('원의 채워진 비율로 확률을 읽습니다', '파란색은 정의한 효율저하 사건, 회색은 그 외 결과의 모델 확률입니다.')
            donut_col, explanation_col = st.columns([1, 1.15], gap='large')
            with donut_col:
                chart(risk_donut(p))
            with explanation_col:
                st.markdown('### 탐색적 예측 · 운영 참고용')
                st.markdown('**평온조건의 기준 중앙 작업률보다 10% 넘게 낮은 결과**를 사건으로 정의했습니다.')
                if _finite(threshold):
                    position = '검토선 이상' if above else '검토선 아래'
                    st.markdown(f'**검토선 {100 * float(threshold):.1f}% · 현재 {position}**')
                    st.caption(metadata.get('threshold_description', '2024년 보정확률의 상위 20% 지점입니다. 효율 10% 저하라는 사건 정의와 다르며 안전정지 기준이 아닙니다.'))
                st.caption('실제 기항 100건을 그린 그림이 아닙니다. 선택한 조건의 모델 확률을 100% 기준으로 표현했습니다.')
        remaining_notes = [str(x) for x in dict.fromkeys(result.get('warnings', []))
                           if x and not str(x).startswith('검증 단계의 참고 예측입니다.')]
        if remaining_notes:
            st.caption(' '.join(remaining_notes))
    _departure_note(s.terminal)
    section('검증 결과 · 현재는 참고용입니다', '단순 발생률 기준보다 확률오차가 작아지지 않았습니다. 기존 평균 작업률 모델의 R²·WAPE와는 별도 평가입니다.')
    ev = _risk_metrics(result)
    display_eval = metadata.get('display_support_evaluation', {})
    if _finite(display_eval.get('n')):
        st.caption(f'위 지표는 전체 평가 자료입니다. 화면의 자료지원 조건을 충족한 {int(display_eval["n"]):,}건에서도 Brier {float(display_eval["brier"]):.4f}로, 터미널별 단순 기준 {float(display_eval["terminal_baseline_brier"]):.4f}보다 개선되지 않았습니다.')
    brier, baseline = ev.get('brier'), ev.get('baseline_brier')
    terminal_baseline = ev.get('terminal_baseline_brier')
    if _finite(brier) and _finite(baseline):
        chart(brier_comparison(ev))
        st.caption('다이아몬드는 단순 발생률 기준, 원은 새 위험모델입니다. 비교 발생률은 2024년에 고정했고 같은 2025년 기항에서 평가했습니다. 확대된 가로 눈금의 실제 수치를 함께 보세요.')
    with st.expander('무엇을 ‘큰 효율저하’라고 정의했나요?'):
        st.markdown('**기록 작업률 < 2020–2022년 평온조건 기준 중앙 작업률 × 0.90**이면 효율저하 사건으로 정했습니다. 10%는 이 기능을 위한 설계 기준이며, 기존 BCT 강풍 분석의 −10.74%를 경보 기준으로 옮긴 값이 아닙니다.')
        st.markdown('기준 중앙 작업률 모델은 2020–2022년의 평온한 기항으로 고정하고, 최종 작업량 등 운영조건을 이용해 사후 결과를 보정합니다. 그 기준모델의 오차도 사건 정의에 포함됩니다. 위험모델은 최종 작업량을 입력받지 않고 접안 시점의 조건으로 이 사건을 예측합니다.')
        st.markdown('기준 작업률 모형은 2020–2022년 자료, 위험 분류모형은 2023년 자료, 확률 보정은 2024년 자료로 만들었습니다. 2025년은 이후 연도 평가에 사용했습니다. 원자료에 포함된 유효 체류시간 6시간 초과 기항을 대상으로 하므로 모든 입항선박에 대한 검증은 아닙니다.')
        st.markdown('최종 Shift, 완료 후 체류시간, 출항시 날씨, 실제 첫 6시간 누적 기상과 최종 작업량은 위험모델의 입력에서 제외했습니다. 운영조건의 변화·선박 구성·기준모델 오차도 사건 발생에 영향을 줄 수 있어 “기상 때문에 효율이 떨어질 확률”로 해석할 수 없습니다.')
        if metadata.get('label_description'):
            st.caption(str(metadata['label_description']))
    with st.expander('2025년 검증표 · 적용 범위 · 모델 정보'):
        if isinstance(validation, pd.DataFrame) and not validation.empty:
            st.dataframe(validation.round(5), hide_index=True, width='stretch')
            st.download_button('위험모델 검증 CSV', validation.to_csv(index=False).encode('utf-8-sig'),
                               file_name='PORTaI_efficiency_risk_validation.csv', mime='text/csv', key='risk_validation_download')
        st.caption('개발 과정에서 이미 다른 분석에 사용한 연도에 대한 후향 평가입니다. 신규 현장 데이터로 관측 빈도와 확률 보정을 다시 확인해야 합니다.')
        st.json(metadata, expanded=False)
    st.button('같은 조건의 유사 기항 보기 →', key='risk_to_cases', on_click=navigate,
              args=('유사 기항',), width='stretch')
    source_note('risk/risk_classifier_2023.cbm · risk/metadata.json · risk/evaluation_by_terminal_2025.csv')
