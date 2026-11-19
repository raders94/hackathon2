"""Shared, restrained visual language for the PORTaI dashboard."""
from __future__ import annotations
from html import escape
import streamlit as st

NAVY = '#183548'
BLUE = '#527C9C'
TEAL = '#138A81'
CORAL = '#D66A4D'
AMBER = '#D59A32'
SLATE = '#8B98A5'
PALETTE = {'blue': BLUE, 'teal': TEAL, 'coral': CORAL, 'amber': AMBER, 'slate': SLATE}

CSS = '''<style>
:root {--port-navy:#183548;--port-muted:#617583;--port-line:#e2e9ed;}
.stApp {background:#F7F9FB;}
.block-container {max-width:1390px;padding-top:1.6rem;padding-bottom:2rem;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e1e8ed;}
[data-testid="stSidebar"] > div:first-child {padding-top:1.6rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:9px 5px;line-height:1.4;}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {font-size:.95rem;font-weight:650;}
.brand {font-size:2rem;font-weight:900;color:#153B50;letter-spacing:-.065em;margin-bottom:0;}
.brand span {color:#138A81;}
.brand-sub {color:#647C89;font-size:.76rem;margin:0 0 22px;letter-spacing:.04em;}
.nav-foot {font-size:.76rem;line-height:1.8;color:#6e7f8c;border-top:1px solid #e2e9ed;padding-top:17px;margin-top:22px;}
.page-top {margin:4px 0 20px;}
.page-top .overline {display:flex;gap:10px;align-items:center;margin-bottom:10px;font-size:.78rem;font-weight:750;color:#637B8B;}
.tag {display:inline-block;padding:4px 9px;background:#e9f3f4;border-radius:5px;color:#16756F;font-size:.72rem;font-weight:750;letter-spacing:.02em;}
.page-top h1 {font-size:2.25rem;font-size:clamp(1.55rem,2.7vw,2.25rem);font-weight:850;line-height:1.35;letter-spacing:-.04em;color:#183548;margin:0 0 8px;word-break:keep-all;}
.page-top p {color:#627582;line-height:1.65;font-size:.91rem;margin:0;word-break:keep-all;}
.takeaway {border-left:4px solid var(--tone);border-radius:0 10px 10px 0;background:#fff;padding:16px 20px;margin:0 0 18px;border-top:1px solid #e6ecef;border-bottom:1px solid #e6ecef;border-right:1px solid #e6ecef;}
.takeaway strong {display:block;font-size:1.08rem;line-height:1.55;color:#203c4d;word-break:keep-all;}
.takeaway p {margin:6px 0 0;color:#607583;font-size:.82rem;line-height:1.65;word-break:keep-all;}
.metric-grid {display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:13px;margin:0 0 20px;}
.metric-card {background:#fff;border:1px solid #dfe8ed;border-radius:12px;padding:18px 20px;min-width:0;}
.metric-card .metric-label {font-size:.83rem;color:#5b7280;font-weight:650;margin-bottom:8px;}
.metric-card .metric-value {font-weight:850;letter-spacing:-.035em;font-size:2.3rem;font-size:clamp(1.6rem,2.4vw,2.3rem);color:var(--tone);line-height:1.28;word-break:keep-all;}
.metric-card .metric-note {font-size:.77rem;line-height:1.55;color:#617784;margin-top:8px;word-break:keep-all;}
.section-heading {font-size:1.14rem;font-weight:800;line-height:1.45;color:#1c3a4d;margin:22px 0 3px;letter-spacing:-.02em;}
.section-caption {font-size:.83rem;color:#687d89;line-height:1.65;margin:0 0 13px;word-break:keep-all;}
.source-note {font-size:.71rem;color:#758692;margin:10px 0 16px;overflow-wrap:anywhere;}
.result-card {padding:2px 1px 8px;min-height:365px;}
.result-card h3 {font-size:1.13rem;line-height:1.55;letter-spacing:-.02em;margin:12px 0 7px;color:#183548;word-break:keep-all;}
.result-card .big-result {font-size:2rem;font-size:clamp(1.5rem,2.2vw,2.25rem);letter-spacing:-.04em;font-weight:850;line-height:1.3;margin:5px 0;color:var(--tone);word-break:keep-all;}
.result-card p {font-size:.8rem;line-height:1.6;color:#637884;margin:5px 0 14px;word-break:keep-all;}
.result-card .mini-label {display:flex;justify-content:space-between;gap:8px;color:#486273;font-size:.74rem;margin:8px 0 4px;}
.bar-track {height:16px;background:#edf1f4;border-radius:4px;overflow:hidden;}
.bar-fill {height:100%;border-radius:4px;}
.result-card .mini-note {font-size:.71rem;color:#7a8993;margin-top:10px;line-height:1.55;min-height:40px;}
.home-intro {display:flex;justify-content:space-between;align-items:center;gap:20px;border-bottom:1px solid #dce7eb;padding:0 0 23px;margin:0 0 24px;}
.home-intro h1 {font-size:2.7rem;font-size:clamp(1.8rem,3vw,2.7rem);line-height:1.3;letter-spacing:-.045em;margin:5px 0 10px;color:#16394e;}
.home-intro p {font-size:.93rem;line-height:1.7;color:#677e8b;margin:0;}
.home-intro .port-icon {width:165px;flex-shrink:0;opacity:.95;}
.legend-line {display:flex;gap:15px;flex-wrap:wrap;margin:12px 0;font-size:.75rem;color:#627783;}
.legend-dot {display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;}
[data-testid="stExpander"] {background:#fff;border:1px solid #dfe8ed;border-radius:10px;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:12px;}
[data-testid="stPlotlyChart"] {border-radius:12px;overflow:hidden;}
.stButton > button {border-radius:8px;font-weight:650;}
[data-testid="stMetric"] {background:white;border:1px solid #dfe8ed;border-radius:10px;padding:14px;}
.stCaption {line-height:1.6;}
.port-footer {border-top:1px solid #e0e7eb;color:#6f8491;font-size:.74rem;line-height:1.7;margin-top:26px;padding-top:15px;}
@media(max-width:850px) {.block-container{padding-top:1rem;} .metric-grid{grid-template-columns:1fr;} .metric-card{padding:14px 17px;} .home-intro .port-icon{display:none;} .home-intro h1{font-size:1.8rem;} .page-top h1{font-size:1.65rem;}}
</style>'''


def install_style():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(section, title, description, kind='분석 결과'):
    st.markdown(f'<div class="page-top"><div class="overline">{escape(section)} <span class="tag">{escape(kind)}</span></div><h1>{escape(title)}</h1><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def takeaway(headline, detail='', tone='blue'):
    color = PALETTE.get(tone, BLUE)
    st.markdown(f'<div class="takeaway" style="--tone:{color}"><strong>{escape(headline)}</strong>' + (f'<p>{escape(detail)}</p>' if detail else '') + '</div>', unsafe_allow_html=True)


def metrics(items):
    cards = []
    for item in items:
        color = PALETTE.get(item.get('tone', 'blue'), BLUE)
        cards.append(f'<div class="metric-card" style="--tone:{color}"><div class="metric-label">{escape(str(item["label"]))}</div><div class="metric-value">{escape(str(item["value"]))}</div><div class="metric-note">{escape(str(item.get("note", "")))}</div></div>')
    st.markdown(f'<div class="metric-grid" style="--cols:{len(cards)}">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def chart(fig, height=None):
    fig.update_layout(
        template='plotly_white', paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family='Malgun Gothic, Apple SD Gothic Neo, Noto Sans CJK KR, sans-serif', size=13, color=NAVY),
        title_font=dict(size=16, color=NAVY), hoverlabel=dict(font_size=13),
        modebar=dict(bgcolor='rgba(255,255,255,0.8)'),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor='#EDF1F4', zerolinecolor='#cdd7dd')
    fig.update_yaxes(gridcolor='#EDF1F4', zerolinecolor='#cdd7dd')
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'scrollZoom': False, 'responsive': True, 'displayModeBar': 'hover'})


def section(title, caption=''):
    st.markdown(f'<div class="section-heading">{escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<p class="section-caption">{escape(caption)}</p>', unsafe_allow_html=True)


def source_note(filenames):
    st.markdown(f'<p class="source-note">근거 · {escape(filenames)}</p>', unsafe_allow_html=True)


def legend():
    st.markdown('<div class="legend-line"><span><i class="legend-dot" style="background:#527C9C"></i>비교 기준</span><span><i class="legend-dot" style="background:#138A81"></i>증가 방향</span><span><i class="legend-dot" style="background:#D66A4D"></i>감소·오차 강조</span><span><i class="legend-dot" style="background:#8B98A5"></i>차이 불확실</span><span><i class="legend-dot" style="background:#D59A32"></i>사용자 가정</span></div>', unsafe_allow_html=True)
