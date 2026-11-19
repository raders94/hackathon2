"""Interactive prediction, paired visual comparison and exact native SHAP."""
from __future__ import annotations
from dataclasses import replace
from datetime import date, datetime
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from logic import (Scenario, TERMINALS, HARBOR, calm_for, defaults_for, occupancy_for,
                   congestion_count, category, model_row, estimate, error_range)
from explainability import explain_prediction, compare_weather_attributions
from prediction_visuals import prediction_pair, shap_waterfall
from ui import page_header, takeaway, metrics, chart, section, source_note, BLUE, TEAL, CORAL, SLATE


def _set_weather(w, wa, v):
    st.session_state['predict_wind'] = float(w)
    st.session_state['predict_wave'] = float(wa)
    st.session_state['predict_visibility'] = int(v)


def render_prediction(D, META, model):
    saved = st.session_state.get('latest_prediction')
    previous = saved['scenario'] if saved else Scenario('BCT', datetime(2027, 11, 20, 10), 6.0, 1.2, 5000, 396, 2, 2)
    page_header('01 / AI 예측', '이 날씨라면 작업률이 어떻게 달라질까요?',
                '같은 운영조건에서 날씨만 바꾼 두 예측값을 비교합니다. 날짜와 날씨는 사용자가 지정하는 가상 시나리오입니다.', kind='입력한 조건의 예측')
    result_slot = st.container()
    section('날씨를 바꿔 비교해보세요', '버튼을 누르면 위 결과와 아래 설명이 함께 바뀝니다. 자세한 입력은 ‘예측 조건 바꾸기’에서 조정합니다.')
    selected_terminal = st.session_state.get('predict_terminal', previous.terminal)
    selected_date = st.session_state.get('predict_day', previous.when.date())
    calm_pre = calm_for(D['calm_weather_by_month'], selected_terminal, selected_date.month)
    threshold_pre = D['weather_thresholds'].set_index('harbor').loc[HARBOR[selected_terminal]]
    for key, value in [('predict_wind', float(previous.wind)), ('predict_wave', float(previous.wave)), ('predict_visibility', int(previous.visibility_m))]:
        if key not in st.session_state:
            st.session_state[key] = value
    presets = [
        ('평온 기준 적용', calm_pre['wind'], calm_pre['wave'], calm_pre['visibility_m']),
        ('강풍 P95 적용', math.ceil(threshold_pre.wind_P95 * 10) / 10, calm_pre['wave'], calm_pre['visibility_m']),
        ('고파고 P95 적용', calm_pre['wind'], math.ceil(threshold_pre.wave_P95 * 20) / 20, calm_pre['visibility_m']),
        ('복합 악천후 예시', 6.0, 1.2, 5000),
    ]
    for i, (col, preset) in enumerate(zip(st.columns(4), presets)):
        label, w, wa, v = preset
        with col:
            st.button(label, key=f'predict_preset_{i}', on_click=_set_weather, args=(w, wa, v), width='stretch')
    with st.expander('예측 조건 바꾸기 · 터미널 / 날짜 / 날씨 / 작업량', expanded=False):
        c1, c2, c3 = st.columns([1, 1.2, .8])
        with c1:
            terminal = st.selectbox('터미널', TERMINALS, index=TERMINALS.index(previous.terminal), key='predict_terminal')
        with c2:
            day = st.date_input('가정 날짜 · 한국시간', previous.when.date(), min_value=date(2020, 1, 1), max_value=date(2030, 12, 31), key='predict_day')
        with c3:
            hour = st.selectbox('가정 접안 시간', list(range(24)), index=previous.when.hour, format_func=lambda x: f'{x:02d}:00', key='predict_hour')
        when = datetime.combine(day, datetime.min.time()).replace(hour=hour)
        w1, w2, w3 = st.columns(3)
        with w1:
            wind = st.slider('풍속 (m/s)', 0., 18., step=.1, key='predict_wind')
        with w2:
            wave = st.slider('유의파고 (m)', 0., 5., step=.05, key='predict_wave')
        with w3:
            visibility = st.slider('시정 (m)', 0, 50000, step=100, key='predict_visibility')
        dd = defaults_for(D['terminal_operational_defaults'], terminal, day.month)
        ref = D['congestion_reference'].set_index('terminal').loc[terminal]
        old_same = previous.terminal == terminal and previous.when.month == day.month
        left, right = st.columns(2)
        with left:
            cargo = st.number_input('대표 선박 작업량 (원자료 단위)', min_value=10, max_value=15000,
                                    value=int(previous.cargo) if old_same else max(10, int(round(dd.median_cargo))), step=10,
                                    key=f'predict_cargo_{terminal}_{day.month}')
        with right:
            arrivals = st.number_input('직전 6시간 접안 유입 (척)', min_value=0, max_value=50,
                                       value=int(previous.arrival_prev6) if old_same else int(ref.high), step=1,
                                       key=f'predict_arrivals_{terminal}')
        st.caption(f'{terminal} 혼잡도 참고값: 낮음 {int(ref.low)}척 / 보통 {int(ref.middle)}척 / 높음 {int(ref.high)}척. 실제 관측값을 자동 수신하지 않습니다.')
        a, b = st.columns(2)
        with a:
            active = st.number_input('합산 계산에만 쓸 동시 접안 척수', min_value=1, max_value=30,
                                     value=int(previous.active_ships) if old_same else occupancy_for(D['typical_active_by_month_hour'], terminal, day.month, hour),
                                     key=f'predict_active_{terminal}_{day.month}')
        with b:
            gt_allowed = 2020 <= day.year <= 2024
            use_gt = st.checkbox('GT 포함 역사모형 사용 (2020–2024)', value=bool(saved and saved.get('use_gt') and gt_allowed),
                                 disabled=not gt_allowed, key='predict_use_gt')
            use_gt = bool(use_gt and gt_allowed)
        gt = None
        if use_gt:
            gt = st.number_input('선박 총톤수 GT', min_value=100, max_value=250000,
                                 value=int(previous.gt) if old_same and previous.gt else int(META['gt_defaults'][terminal]), step=100,
                                 key=f'predict_gt_{terminal}')
        st.caption('학습에는 실제 최종 작업량을 사용했습니다. 접안 전 활용은 계획 작업량을 입력하는 별도 전향 검증이 필요합니다.')
    scenario = Scenario(terminal, when, float(wind), float(wave), float(visibility), float(cargo), float(arrivals), int(active), float(gt) if gt else None)
    calm = calm_for(D['calm_weather_by_month'], terminal, day.month)
    m = model(use_gt)
    result = estimate(m, scenario, calm, use_gt)
    st.session_state['latest_prediction'] = {'scenario': scenario, 'result': result, 'use_gt': use_gt, 'calm': calm}
    change = result['delta_pct']
    if change < -.005:
        headline = f'{terminal} · 선택한 날씨에서 평균 작업률이 평온 기준보다 {abs(change):.2f}% 낮게 예측됩니다.'
        tone, selected_color = 'coral', CORAL
    elif change > .005:
        headline = f'{terminal} · 선택한 날씨에서 평균 작업률이 평온 기준보다 {change:.2f}% 높게 예측됩니다.'
        tone, selected_color = 'teal', TEAL
    else:
        headline = f'{terminal} · 선택한 날씨와 평온 기준의 예측값이 같습니다.'
        tone, selected_color = 'blue', BLUE
    with result_slot:
        takeaway(headline, f'{when:%Y-%m-%d %H:00} KST · 풍속 {wind:.1f} m/s · 파고 {wave:.2f} m · 시정 {visibility:,.0f} m · 입력한 조건의 모델 비교', tone)
        metrics([
            {'label': '선택한 날씨 · 선박당 평균', 'value': f'{result["vessel"]:.1f} /h', 'note': '원자료 작업단위 ÷ 접안 후 체류시간', 'tone': tone},
            {'label': '날씨만 평온하게 바꾼 기준', 'value': f'{result["normal"]:.1f} /h', 'note': '작업량·터미널·시간·혼잡은 동일', 'tone': 'blue'},
            {'label': '평온 기준과의 차이', 'value': f'{change:+.2f}%', 'note': '확정적인 기상 원인·손실률이 아닙니다.', 'tone': tone},
        ])
        chart(prediction_pair(result, selected_color))
        level, flags = category(scenario, D['weather_thresholds'])
        label = ' · '.join(flags) if flags else '세 기상요소 모두 P90 기준 미만'
        st.caption(f'현재 기상 분류: {label}. P90/P95는 과거 관측분포 기준이며 장비 안전정지 기준이 아닙니다.')
    limits = D['history_weather_training_limits'].set_index('harbor').loc[HARBOR[terminal]]
    if wind > float(limits.wind_p99) or wave > float(limits.wave_p99):
        st.warning('선택한 풍속 또는 파고는 학습 기상분포의 상위 1%보다 큰 값입니다. 드문 조건의 예측으로 해석하세요. 이 기준은 관측 최대값이나 안전 한계가 아닙니다.')
    if use_gt:
        st.info('GT 포함 역사모형에는 2025년 독립 검증 결과가 없습니다. ‘예측 검증’ 화면은 GT 미포함 모형의 성능입니다.')

    section('이 예측값을 올리거나 낮춘 입력', '워터폴의 청록 계단은 예측을 높이고, 주황 계단은 낮춥니다. 기여를 순서대로 더한 끝점이 현재 예측입니다.')
    explanation = explain_prediction(m, scenario, use_gt)
    scope = st.radio('설명할 입력', ['주요 입력 5개', '기상·혼잡만', '전체 입력'], horizontal=True, key='prediction_shap_scope')
    full = explanation['table']
    values = full.head(5).copy() if scope == '주요 입력 5개' else (
        full.loc[full.feature.isin(['wind', 'wave', 'vis_10m', 'arrival_prev6'])].copy() if scope == '기상·혼잡만' else full.copy())
    chart(shap_waterfall(explanation, values))
    st.caption('SHAP의 기준은 학습분포이며 위 평온 기준과 다릅니다. 생략한 입력은 ‘그 외 입력’으로 합산해 예측을 정확히 재구성합니다. 세로축은 로그 기여이며 실제 손실률이 아닙니다.')
    with st.expander('정확한 SHAP 수치 · 계산 방식 · 설명 CSV'):
        st.markdown(f'**SHAP 기준값:** {explanation["base_rate"]:.2f}/h · **현재 예측 재구성:** {explanation["pred_rate"]:.2f}/h')
        table = full[['label', 'value', 'shap_log', 'multiplier_pct', 'direction']].rename(columns={
            'label': '입력', 'value': '값', 'shap_log': 'SHAP (log)', 'multiplier_pct': '기여의 배수 환산 (%)', 'direction': '방향'})
        st.dataframe(table.round(4), hide_index=True, width='stretch')
        st.caption(f'기준 log 예측 + 전체 SHAP 합 = 현재 log 예측. 합산 오차 {explanation["reconstruction_error"]:.2e}. 각 기여의 배수 환산(%)은 서로 더할 수 없습니다.')
        st.download_button('SHAP 설명 CSV', table.to_csv(index=False).encode('utf-8-sig'), 'PORTaI_SHAP.csv', 'text/csv')
        comparison = compare_weather_attributions(m, scenario, calm, use_gt)
        st.markdown('**같은 운영조건에서 날씨만 바꾼 SHAP 차이**')
        st.dataframe(comparison['table'].rename(columns={'label': '입력', 'delta_shap_log': 'SHAP 차이 (log)', 'changed_input': '직접 바뀐 입력'}), hide_index=True, width='stretch')
        st.caption('기상 입력만 바꿔도 상호작용 때문에 다른 입력의 기여가 재배분될 수 있습니다. 전체 log 차이 합만 두 예측의 차이와 일치합니다.')
    with st.expander('기상 강도를 연속해서 바꾸면? · 민감도 그림'):
        a, b = st.columns(2)
        thresholds = D['weather_thresholds'].set_index('harbor').loc[HARBOR[terminal]]
        for col, field, label, stop in [(a, 'wind', '풍속 (m/s)', 11.0), (b, 'wave', '유의파고 (m)', 2.5)]:
            with col:
                axis = np.linspace(0, stop, 40)
                batch = pd.concat([model_row(replace(scenario, **{field: float(x)}), use_gt) for x in axis], ignore_index=True)
                predicted = np.exp(m.predict(batch))
                f = go.Figure(go.Scatter(x=axis, y=predicted, mode='lines', line=dict(color=BLUE, width=3)))
                f.add_vline(x=float(getattr(scenario, field)), line_color=CORAL, annotation_text='선택값')
                f.update_layout(title=f'{label}만 변경', xaxis_title=label, yaxis_title='선박당 평균 작업률 /h', height=310, margin=dict(t=55, l=10, r=20, b=45))
                chart(f)
        st.caption('다른 입력을 고정한 모델 민감도입니다. 드문 입력 조합을 포함할 수 있으며 기상의 인과효과 곡선은 아닙니다.')
    with st.expander('7개 터미널에 같은 날씨를 입력하면?'):
        rows = []
        for t in TERMINALS:
            default = defaults_for(D['terminal_operational_defaults'], t, day.month)
            candidate = replace(scenario, terminal=t, cargo=float(default.median_cargo), gt=float(META['gt_defaults'][t]) if use_gt else None)
            r = estimate(m, candidate, calm_for(D['calm_weather_by_month'], t, day.month), use_gt)
            rows.append({'터미널': t, '선박당 예상 작업률': r['vessel'], '평온 기준 대비 (%)': r['delta_pct'], '대표 작업량': candidate.cargo})
        comparison = pd.DataFrame(rows)
        f = go.Figure(go.Bar(x=comparison['평온 기준 대비 (%)'], y=comparison['터미널'], orientation='h',
                            marker_color=[CORAL if v < 0 else TEAL for v in comparison['평온 기준 대비 (%)']],
                            text=[f'{v:+.2f}%' for v in comparison['평온 기준 대비 (%)']], textposition='outside', cliponaxis=False))
        limit = max(2., comparison['평온 기준 대비 (%)'].abs().max()*1.3)
        f.update_layout(title='같은 날씨 · 터미널별 대표 작업량', xaxis_title='각 터미널의 평온 예측 대비 (%)', xaxis_range=[-limit, limit],
                        yaxis=dict(autorange='reversed'), height=380, margin=dict(t=55, b=45, l=10, r=25), showlegend=False)
        chart(f)
        st.dataframe(comparison.round(2), hide_index=True, width='stretch')
        st.caption('날씨·시간·직전 접안 유입은 같게 두고 터미널별 대표 작업량을 적용합니다. 실제 터미널 처리능력 순위가 아닙니다.')
    with st.expander('예측값의 범위 · 합산 가정 · 결과 저장'):
        if not use_gt:
            low, high = error_range(result['vessel'], terminal, D['model_2025_test_metrics'])
            st.write(f'2025년 과거 예측오차를 적용한 경험적 80% 범위: **{low:.1f}~{high:.1f} /h**. 새 시나리오의 보장된 예측구간은 아닙니다.')
        st.write(f'동시 접안 **{active}척**을 가정하면 합산값 **{result["aggregate"]:.1f} /h**입니다. 대표 선박당 평균에 척수를 곱한 값이며 특정 시간의 실제 처리량이 아닙니다.')
        st.write('미래 날짜는 실제 예보와 연결되지 않습니다. 작업량·최근 접안 유입 등 입력이 예측 시점에 확보되는지 별도 검증해야 합니다.')
        export = pd.DataFrame([{'date_time_KST': when.isoformat(), 'terminal': terminal, 'wind_mps': wind, 'wave_m': wave,
                               'visibility_m': visibility, 'cargo_units': cargo, 'arrivals_previous_6h': arrivals,
                               'assumed_active_vessels': active, 'use_gt': use_gt, 'predicted_vessel_rate': result['vessel'],
                               'calm_vessel_rate': result['normal'], 'delta_pct': change, 'assumed_aggregate_rate': result['aggregate'],
                               'kind': 'user-specified conditional scenario; not live or causal'}])
        st.download_button('현재 예측 결과 CSV 저장', export.to_csv(index=False).encode('utf-8-sig'), 'PORTaI_prediction.csv', 'text/csv')
    source_note('throughput_nonGT_2020_2024.cbm / throughput_GT_2020_2024.cbm · explainability.py · calm_weather_by_month.csv')
