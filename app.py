"""PORTaI | result-first Streamlit dashboard. Run: streamlit run app.py."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from logic import ASSETS, load_model, metadata
from ui import install_style
from pages_overview import render_overview
from pages_prediction import render_prediction
from pages_research import render_weather, render_terminals
from pages_evaluation import render_learning, render_validation
from pages_plan import render_plan
from pages_decision import render_cases, render_risk

st.set_page_config(page_title='PORTaI | 항만 작업과 날씨, 한눈에', page_icon='⚓', layout='wide', initial_sidebar_state='expanded')
install_style()


@st.cache_data(show_spinner=False)
def source_data():
    return {p.stem: pd.read_csv(p) for p in sorted(ASSETS.glob('*.csv'))}


@st.cache_resource(show_spinner=False)
def model(use_gt=False):
    return load_model(use_gt)


PAGES = ['한눈에 보기', 'AI 예측', '유사 기항', '효율저하 위험', '악천후 영향', '터미널 현황', '운영학습', '예측 검증', '개선 계획']
PAGE_DESCRIPTIONS = {
    '한눈에 보기': '핵심 결과와 다음 질문',
    'AI 예측': '입력한 조건의 예상 작업률',
    '유사 기항': '가까운 과거 사례와 결과 분포',
    '효율저하 위험': '큰 작업률 저하의 탐색적 확률',
    '악천후 영향': '운영조건을 보정한 기상 차이',
    '터미널 현황': '분석 대상의 규모와 분포',
    '운영학습': '시간이 흐르며 달라진 효율',
    '예측 검증': '2025년 예측오차와 데이터 범위',
    '개선 계획': '가정 목표와 현장 검증 준비',
}


def navigate(page):
    st.session_state['portai_page'] = page


with st.sidebar:
    st.markdown('<div class="brand">PORT<span>aI</span></div><div class="brand-sub">항만 작업과 날씨, 한눈에</div>', unsafe_allow_html=True)
    page = st.radio('분야 선택', PAGES, key='portai_page', label_visibility='collapsed')
    st.caption(PAGE_DESCRIPTIONS[page])
    st.markdown('<div class="nav-foot"><b>분석 범위</b><br>부산항 7개 터미널<br>2020–2025년 운영자료<br><br><b>화면 읽는 법</b><br>그림의 길이 = 값의 크기<br>색과 문구 = 결과의 의미<br><br>작업률 = 총작업량 ÷ 접안 후 체류시간</div>', unsafe_allow_html=True)

D = source_data()
META = metadata()
if page == '한눈에 보기':
    render_overview(D, META, navigate)
elif page == 'AI 예측':
    render_prediction(D, META, model)
elif page == '유사 기항':
    render_cases(D, META, navigate)
elif page == '효율저하 위험':
    render_risk(D, META, navigate)
elif page == '악천후 영향':
    render_weather(D, META)
elif page == '터미널 현황':
    render_terminals(D, META)
elif page == '운영학습':
    render_learning(D, META)
elif page == '예측 검증':
    render_validation(D, META)
elif page == '개선 계획':
    render_plan(D, META, model)

st.markdown('<div class="port-footer"><b>PORTaI</b> · 기항 평균 작업률을 분석합니다. 실제 크레인의 시간당 Moves와는 다른 지표입니다.<br>예측은 입력한 조건의 시나리오이며, 분석 결과는 관측자료의 연관성입니다. 개선 계획의 회복비율은 사용자 가정입니다.</div>', unsafe_allow_html=True)
