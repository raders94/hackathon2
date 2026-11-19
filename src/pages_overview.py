"""Opening screen with proportional graphics and direct navigation."""
from html import escape
import streamlit as st
from ui import section, source_note, BLUE, TEAL, CORAL, SLATE


def _bar(label, value, max_value, text, color):
    width = max(0, min(100, value / max_value * 100))
    return f'<div class="mini-label"><span>{escape(label)}</span><b>{escape(text)}</b></div><div class="bar-track"><div class="bar-fill" style="width:{width:.4f}%;background:{color}"></div></div>'


def render_overview(D, META, navigate):
    b = D['modelb_weather_residual_effects'].query("terminal == 'BCT' and variable == 'wind' and exposure == '4-6'").iloc[0]
    s = D['validation_severity_2025'].query("terminal == '전체'")
    absolute_errors = s.n * s.mae
    wape = 100 * absolute_errors.sum() / (absolute_errors / (s.wape_pct / 100)).sum()
    calm = D['learning_calm_2024_vs_2022'].query("terminal == 'BCT' and year == 2024").iloc[0]
    slope = D['learning_annual_slopes'].query("terminal == 'BCT'").iloc[0]
    st.markdown('''<div class="home-intro"><div><span class="tag">한눈에 보기 · PORTaI</span><h1>항만 작업과 날씨,<br>어떤 결과가 나왔을까요?</h1><p>핵심 결과는 그림으로 먼저 확인하고,<br>궁금한 분야에서 예측과 근거를 자세히 살펴보세요.</p></div><svg class="port-icon" viewBox="0 0 180 135" fill="none" aria-hidden="true"><path d="M7 116H173M20 126H160" stroke="#BED3DA" stroke-width="3"/><path d="M34 101V25H127L105 44H34M74 25V101M33 25L105 7V25" stroke="#527C9C" stroke-width="5" stroke-linejoin="round"/><path d="M105 44V64" stroke="#527C9C" stroke-width="3"/><rect x="89" y="64" width="33" height="20" rx="2" fill="#D66A4D"/><path d="M13 95H164L148 113H31Z" fill="#153B50"/><rect x="46" y="73" width="37" height="22" rx="2" fill="#138A81"/><rect x="87" y="87" width="33" height="8" rx="1" fill="#91BDC0"/><path d="M55 78V90M65 78V90M75 78V90M99 68V79M111 68V79" stroke="white" stroke-opacity=".45" stroke-width="2"/></svg></div>''', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap='medium')
    with c1, st.container(border=True):
        html = f'<div class="result-card" style="--tone:{CORAL}"><span class="tag">악천후 영향 · 관측 분석</span><h3>BCT 강풍 고노출군의<br>작업효율이 낮았습니다</h3><div class="big-result">{b.adjusted_eff_change_pct:+.2f}%</div><p>첫 6회 관측 중 강풍 4–6회 vs 0회<br>작업량·선박 크기·운영조건 보정</p>'
        html += _bar('강풍 0회 · 비교 기준', 100, 100, '100', BLUE)
        html += _bar('강풍 4–6회', 100 + b.adjusted_eff_change_pct, 100, f'{100+b.adjusted_eff_change_pct:.1f}', CORAL)
        html += '<div class="mini-note">기준을 100으로 둔 조정 효율비율 그림<br>기상 원인을 확정한 손실률은 아닙니다.</div></div>'
        st.markdown(html, unsafe_allow_html=True)
        st.button('악천후 영향 자세히 →', key='home_weather', on_click=navigate, args=('악천후 영향',), width='stretch')
    with c2, st.container(border=True):
        html = f'<div class="result-card" style="--tone:{BLUE}"><span class="tag">예측 검증 · 2025년 평가</span><h3>새 연도 자료에서<br>예측오차를 확인했습니다</h3><div class="big-result">{wape:.2f}%</div><p>전체 상대 예측오차 · WAPE<br>미학습 2025년 {int(s.n.sum()):,}건</p>'
        for cond, label, color in [('P90 미만', 'P90 미만', BLUE), ('P90 이상·P95 미만', 'P90~P95 사이', BLUE), ('P95 이상', '심한 기상 · P95 이상', CORAL)]:
            value = float(s.loc[s.severity == cond, 'wape_pct'].iloc[0])
            html += _bar(label, value, 25, f'{value:.2f}%', color)
        html += '<div class="mini-note">세 막대 모두 0~25% 눈금 · 짧을수록 오차가 작음<br>정확도를 뜻하는 비율이 아닙니다.</div></div>'
        st.markdown(html, unsafe_allow_html=True)
        st.button('예측 검증 자세히 →', key='home_validation', on_click=navigate, args=('예측 검증',), width='stretch')
    with c3, st.container(border=True):
        html = f'<div class="result-card" style="--tone:{SLATE}"><span class="tag">운영학습 · 연도별 분석</span><h3>평온 효율은 개선,<br>강풍 대응 개선은 미확인</h3><div class="big-result">추가 검증 필요</div><p>BCT 평온 작업효율은 {calm.change_pct:+.2f}%<br>2024년 vs 2022년 · 운영조건 보정</p>'
        html += _bar('평온 · 2022년 기준', 100, 130, '100', BLUE)
        html += _bar('평온 · 2024년', 100 + calm.change_pct, 130, f'{100+calm.change_pct:.1f}', TEAL)
        html += f'<div class="mini-note">평온 지수는 같은 0~130 눈금<br>강풍 대응의 연간 개선은 미확인 · p={slope.p_slope:.3f}</div></div>'
        st.markdown(html, unsafe_allow_html=True)
        st.button('운영학습 자세히 →', key='home_learning', on_click=navigate, args=('운영학습',), width='stretch')
    st.caption('각 카드의 막대는 해당 카드 안에서 같은 기준으로 비교합니다. 카드 간에는 지표와 눈금이 다릅니다.')
    section('직접 해보고 싶다면', '입력한 날씨의 예측과, 아직 검증하지 않은 개선 목표를 별도 화면에서 살펴볼 수 있습니다.')
    q1, q2, q3 = st.columns(3)
    for col, title, desc, target, key in [
        (q1, '01  이 날씨라면?', '터미널과 날씨를 바꾸고 예상 작업률과 SHAP 설명을 확인합니다.', 'AI 예측', 'home_predict'),
        (q2, '02  어떤 터미널을 비교했나?', '7개 터미널의 기항 수, 평균 작업률 분포와 기상 기준을 살펴봅니다.', '터미널 현황', 'home_terminals'),
        (q3, '03  얼마나 회복할 수 있다면?', '회복 비율을 가정해 목표를 계산하고 90일 검증 계획을 확인합니다.', '개선 계획', 'home_plan'),
    ]:
        with col, st.container(border=True):
            st.markdown(f'**{title}**')
            st.caption(desc)
            st.button(target + ' 열기 →', key=key, on_click=navigate, args=(target,), width='stretch')
    source_note('modelb_weather_residual_effects.csv · validation_severity_2025.csv · learning_calm_2024_vs_2022.csv · learning_annual_slopes.csv')
