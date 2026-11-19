"""Result-first research and terminal overview pages for PORTaI.

Every displayed value is read from the bundled evidence tables.  These pages
describe adjusted associations or descriptive distributions, not causal losses.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ui import page_header, takeaway, metrics, chart, section, source_note
from story import dose_forest, bct_five_stage, wind_wave_forest, ml_year_chart
from research_visuals import weather_effect_heatmap, terminal_bubble_map, terminal_violin, annual_rate_heatmap
from viz_theme import render_seaborn


TERMINALS = ["BCT", "BNCT", "HPNT", "PNC", "PNIT", "BPTC_G", "BPT_S"]
FACTORS = {"wind": "강풍", "wave": "고파고", "visibility": "저시정"}
NAVY = "#17364A"
TEAL = "#138A81"
CORAL = "#D66A4D"
SLATE = "#8B98A5"
BLUE = "#527C9C"


def _pct(value: float, digits: int = 2) -> str:
    return f"{float(value):+.{digits}f}%".replace("-", "−")


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("terminal").reindex(TERMINALS).reset_index()


def _effect_forest(
    labels: list[str], values: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    significant: np.ndarray, title: str, height: int = 430,
) -> go.Figure:
    """Point = estimate, line = confidence interval; gray means uncertain."""
    vals = np.asarray(values, dtype=float)
    lows = np.asarray(lo, dtype=float)
    highs = np.asarray(hi, dtype=float)
    sig = np.asarray(significant, dtype=bool)
    colors = [SLATE if not s else (CORAL if v < 0 else TEAL)
              for v, s in zip(vals, sig)]
    labels_on_points = [f"{_pct(v)}" + (" · 불확실" if not s else "")
                      for v, s in zip(vals, sig)]
    fig = go.Figure(go.Scatter(
        x=vals, y=labels, mode="markers",
        marker=dict(size=13, color=colors, symbol=["circle" if s else "circle-open" for s in sig], line=dict(width=2)),
        cliponaxis=False,
        error_x=dict(type="data", symmetric=False,
                     array=np.maximum(highs - vals, 0),
                     arrayminus=np.maximum(vals - lows, 0),
                     color="#536676", thickness=2.4, width=5),
        customdata=np.column_stack((lows, highs)),
        hovertemplate="%{y}<br>조정 차이 %{x:+.2f}%<br>95% 신뢰구간 %{customdata[0]:+.2f}% ~ %{customdata[1]:+.2f}%<extra></extra>",
    ))
    finite = np.concatenate([lows[np.isfinite(lows)], highs[np.isfinite(highs)]])
    bound = max(3.0, float(np.max(np.abs(finite))) * 1.65) if finite.size else 5.0
    for label, val, low, high, text_label in zip(labels, vals, lows, highs, labels_on_points):
        fig.add_annotation(x=float(low - bound*.025 if val < 0 else high + bound*.025),
                           y=label, text=text_label, showarrow=False,
                           xanchor="right" if val < 0 else "left",
                           font=dict(size=12, color=NAVY))
    fig.add_vline(x=0, line_dash="dash", line_color="#768C9C", line_width=1)
    fig.update_layout(
        title=title, height=height, showlegend=False,
        margin=dict(l=20, r=60, t=60, b=50),
        xaxis_title="조정 작업효율 차이 (%) · 0 = 비교군과 차이 없음",
        yaxis_title="",
    )
    fig.update_xaxes(range=[-bound, bound], ticksuffix="%", zeroline=False)
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels)
    return fig


@st.cache_data(show_spinner=False)
def _distribution_records() -> pd.DataFrame:
    """Read the real weather-complete search cohort, not synthetic box data."""
    path = Path(__file__).resolve().parents[1] / "assets" / "case_search" / "case_index.csv.gz"
    records = pd.read_csv(path, usecols=["terminal", "start", "work_per_hour"])
    records["year"] = pd.to_datetime(records.start, errors="coerce").dt.year
    return records


def _add_point_labels(fig: go.Figure) -> go.Figure:
    """Existing research figures remain readable without hovering."""
    for trace in fig.data:
        if trace.type == "scatter" and trace.x is not None:
            try:
                trace.update(mode="markers+text", text=[_pct(x) for x in trace.x],
                             textposition="top center", cliponaxis=False)
            except (TypeError, ValueError):
                pass
    return fig


def _weather_reference(D: dict, key_prefix: str) -> None:
    st.caption("2020–2024년 시간별 관측의 항만별 고정 분위 기준입니다. 장비의 안전정지 기준과는 다릅니다.")
    option = st.radio("기상 기준을 선택하세요", ["강풍", "고파고", "저시정"],
                      horizontal=True, key=f"{key_prefix}_threshold_factor")
    variable, unit, multiplier = {
        "강풍": ("wind", "m/s", 1),
        "고파고": ("wave", "m", 1),
        "저시정": ("vis_10m", "m", 10),
    }[option]
    d = D["weather_thresholds"].copy()
    fig = go.Figure()
    for threshold, label, color in [("P90", "P90 기준", BLUE), ("P95", "P95 기준", CORAL)]:
        vals = d[f"{variable}_{threshold}"] * multiplier
        fig.add_trace(go.Bar(x=d.harbor, y=vals, name=label,
                            text=[f"{v:,.2f}".rstrip("0").rstrip(".") if unit != "m" or variable == "wave" else f"{v:,.0f}" for v in vals],
                            textposition="outside", marker_color=color,
                            hovertemplate=f"%{{x}} · {label}<br>%{{y:,.2f}} {unit}<extra></extra>"))
    fig.update_layout(title=f"{option}는 항만별 관측분포를 기준으로 구분합니다",
                      barmode="group", yaxis_title=f"{option} 기준 ({unit})", xaxis_title="",
                      legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=65, b=35))
    max_value = float(d[[f"{variable}_P90", f"{variable}_P95"]].to_numpy().max() * multiplier)
    fig.update_yaxes(range=[0, max_value * 1.23])
    chart(fig, height=330)
    if variable == "vis_10m":
        st.caption("시정은 낮을수록 나쁩니다. 저시정 P90·P95는 각각 하위 10%·5% 기준입니다.")
    else:
        st.caption(f"{option}는 높을수록 강한 조건입니다. P90·P95는 각각 상위 10%·5% 기준입니다.")
    table = d[["harbor", "wind_P90", "wind_P95", "wave_P90", "wave_P95", "vis_10m_P90", "vis_10m_P95"]].copy()
    table[["vis_10m_P90", "vis_10m_P95"]] *= 10
    st.dataframe(table.rename(columns={
        "harbor": "항만", "wind_P90": "풍속 P90 (m/s)", "wind_P95": "풍속 P95 (m/s)",
        "wave_P90": "파고 P90 (m)", "wave_P95": "파고 P95 (m)",
        "vis_10m_P90": "저시정 P90 (m)", "vis_10m_P95": "저시정 P95 (m)",
    }), hide_index=True, width='stretch')
    source_note("weather_thresholds.csv · 2020–2024년 항만별 기상 기준")


def render_weather(D: dict, META: dict) -> None:
    page_header("악천후 영향", "날씨가 나쁠 때, 작업효율은 얼마나 달랐을까요?",
                "작업량·선박 크기·운영조건을 보정한 모델 B로 7개 터미널을 같은 기준에서 비교합니다.",
                kind="분석 결과")
    controls = st.columns([1, 1.4])
    with controls[0]:
        factor = st.selectbox("분석할 날씨", list(FACTORS), format_func=FACTORS.get,
                              key="research_weather_factor")
    with controls[1]:
        exposure = st.radio("첫 6시간 중 악천후 노출", ["4-6", "1-3"],
                            format_func=lambda x: f"{x.replace('-', '–')}회 vs 0회",
                            horizontal=True, key="research_weather_exposure")
    eff = D["modelb_weather_residual_effects"]
    view = _ordered(eff[(eff.variable == factor) & (eff.exposure == exposure)].copy())
    bct = view.loc[view.terminal == "BCT"].iloc[0]
    effect = float(bct.adjusted_eff_change_pct)
    significant = bool(bct.fdr_q_7terminal < .05)
    condition = f"{FACTORS[factor]} {exposure.replace('-', '–')}회 노출"
    if significant:
        direction = "낮았습니다" if effect < 0 else "높았습니다"
        headline = f"BCT는 {condition}군에서 보정 작업효율이 {abs(effect):.2f}% {direction}."
        tone = "coral" if effect < 0 else "teal"
    else:
        headline = f"BCT의 {condition}에서는 뚜렷한 효율 차이를 확인하지 못했습니다."
        tone = "slate"
    takeaway(headline,
             "비교: 같은 기상요인의 0회 노출군 · 2020–2024년 GT 포함 자료. 관측자료의 조정 연관성이며, 인과적 손실률이 아닙니다.",
             tone=tone)
    metrics([
        {"label": "BCT · 보정 작업효율 차이", "value": _pct(effect),
         "note": f"{condition} vs 0회", "tone": tone},
        {"label": "BCT · 추정의 범위", "value": f"{_pct(bct.ci_low_pct)} ~ {_pct(bct.ci_high_pct)}",
         "note": "95% 신뢰구간", "tone": "slate"},
        {"label": "BCT · 노출군 표본", "value": f"{int(bct.n_exposed):,}건",
         "note": f"0회 비교군 {int(bct.n_calm_reference):,}건", "tone": "blue"},
    ])
    section("기상별 차이를 색으로, 추정 범위를 점과 선으로 봅니다",
            "주황: 유의한 감소 · 청록: 유의한 증가 · 회색: 차이 불확실. 히트맵의 *는 FDR q<0.05이며 테두리는 선택한 기상요인입니다.")
    overview, detail = st.columns([.95, 1.25])
    with overview:
        render_seaborn(weather_effect_heatmap(eff, exposure, factor),
                       alt="7개 터미널과 강풍·고파고·저시정의 조정 작업효율 차이. 유의한 셀만 색으로 강조하며 자료 없음은 별도로 표시합니다.")
    with detail:
        fig = _effect_forest(
            view.terminal.tolist(), view.adjusted_eff_change_pct.to_numpy(),
            view.ci_low_pct.to_numpy(), view.ci_high_pct.to_numpy(),
            (view.fdr_q_7terminal < .05).to_numpy(),
            f"선택한 {FACTORS[factor]} · 점 추정치와 95% 신뢰구간", 445,
        )
        chart(fig, height=445)
    st.caption("오른쪽 그림에서 왼쪽일수록 비교군보다 효율이 낮습니다. 회색 빈 점은 추정치가 0이라는 뜻이 아니라 차이가 불확실하다는 뜻입니다. 유의성은 기상요인별 7개 터미널 FDR q<0.05 기준입니다.")
    source_note("modelb_weather_residual_effects.csv · 총작업량 ÷ 접안 후 체류시간 기준 작업효율")

    with st.expander("선택한 결과의 표본·신뢰구간·검정값 보기", expanded=False):
        columns = ["terminal", "n_exposed", "n_calm_reference", "n_total", "adjusted_eff_change_pct", "ci_low_pct", "ci_high_pct", "fdr_q_7terminal"]
        st.dataframe(view[columns].rename(columns={
            "terminal": "터미널", "n_exposed": "노출군 n", "n_calm_reference": "0회군 n", "n_total": "전체 분석 n",
            "adjusted_eff_change_pct": "조정 효율차이 (%)", "ci_low_pct": "95% 하한 (%)",
            "ci_high_pct": "95% 상한 (%)", "fdr_q_7terminal": "FDR q",
        }).round(3), hide_index=True, width='stretch')
        st.markdown("**비교 방법** · 정상 기대효율을 먼저 추정한 뒤 기상 노출별 잔차 차이를 비교합니다. 작업량·GT·달력·혼잡·Shift 등을 보정하고 동시 기상조건을 고려했습니다.")
        st.caption("접안 후 첫 6개 시간관측 중 최소 5회 관측 조건입니다. ‘4–6회’는 고정 관측창 내 노출 횟수이며 정확한 안전정지 임계시간이 아닙니다. Shift는 최종 Shift 작업량 비율로, 근무조 인원과 다릅니다.")
        st.caption("연도 제외 교차검증에서 2020–2023년의 제외연도는 이후 연도 자료를 학습에 포함할 수 있는 후향 분석입니다. 시간순으로 분리한 2024년 검증 결과는 아래에서 따로 확인할 수 있습니다.")
        st.download_button("선택 결과 CSV 받기", view[columns].to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"PORTaI_ModelB_{factor}_{exposure}.csv", mime="text/csv",
                           key="research_weather_download")

    section("BCT의 장시간 강풍 결과는 방법을 바꿔도 감소 방향입니다",
            "아래 비교는 선택 메뉴와 별개로 ‘BCT · 강풍 4–6회 vs 0회’에 고정됩니다.")
    b = eff[(eff.terminal == "BCT") & (eff.variable == "wind") & (eff.exposure == "4-6")].iloc[0]
    o = D["modelb_direct_ols"].query("terminal == 'BCT'").iloc[0]
    m = D["modelb_matched_peers"].query("terminal == 'BCT'").iloc[0]
    h = D["modelb_2024_holdout"].query("terminal == 'BCT'").iloc[0]
    methods = pd.DataFrame([
        {"방법": "모델 B", "차이": b.adjusted_eff_change_pct, "하한": b.ci_low_pct, "상한": b.ci_high_pct},
        {"방법": "직접 다중회귀", "차이": o.direct_ols_eff_pct, "하한": o.ci_low_pct, "상한": o.ci_high_pct},
        {"방법": "유사 평온조건 매칭", "차이": m.matched_eff_diff_pct, "하한": m.ci_low_pct, "상한": m.ci_high_pct},
        {"방법": "2024년 분리 검증", "차이": h.change_pct, "하한": h.ci_low_pct, "상한": h.ci_high_pct},
    ])
    methods_sig = ((methods["상한"] < 0) | (methods["하한"] > 0)).to_numpy()
    chart(_effect_forest(methods["방법"].tolist(), methods["차이"].to_numpy(),
                       methods["하한"].to_numpy(), methods["상한"].to_numpy(), methods_sig,
                       "분석방법 4개에서 감소 방향을 확인했습니다", 310), height=310)
    st.caption("방법별 비교집단과 보정 방식이 다릅니다. 동일 자료를 활용한 민감도 비교이며 서로 독립적인 인과 증거 4개를 뜻하지 않습니다.")
    with st.expander("방법별 상세 수치와 비교조건", expanded=False):
        st.dataframe(methods.rename(columns={"차이": "효율차이 (%)", "하한": "95% 하한 (%)", "상한": "95% 상한 (%)"}).round(2),
                     hide_index=True, width='stretch')
        st.markdown("모델 B와 직접 회귀는 동시 기상조건을 보정합니다. 매칭은 유사한 평온조건을 찾는 방식으로 보정 범위가 다릅니다. 2024년 분리 검증은 이전 연도 학습 후 2024년 BCT 표본으로 확인한 결과입니다.")
        source_note("modelb_weather_residual_effects.csv · modelb_direct_ols.csv · modelb_matched_peers.csv · modelb_2024_holdout.csv")

    with st.expander("기존 체류시간 회귀 · 7터미널과 BCT 노출 단계", expanded=False):
        st.markdown("**결과변수: 작업효율이 아닌 접안 후 체류시간입니다.** 작업량이 같을 때 두 지표는 수학적으로 연결되므로 독립적인 증거로 합산하지 않습니다.")
        chart(_add_point_labels(dose_forest(D["wind_dose"])), height=400)
        dose = D["wind_dose"]
        row = dose[(dose.terminal == "BCT") & (dose.scheme == 3) & (dose.group == "4-6h")].iloc[0]
        st.caption(f"BCT: 조정 체류시간 {_pct(row.effect_pct)} · 95% 신뢰구간 {_pct(row.CI_lo_pct)} ~ {_pct(row.CI_hi_pct)}. 이 회귀의 BCT 표본은 {int(row.N):,}건, 고노출 {int(row.exposed_n):,}건으로 모델 B 표본과 다릅니다.")
        f = bct_five_stage(dose)
        if f.data:
            f.data[0].text = [_pct(v) for v in f.data[0].y]
        chart(f, height=370)
        st.caption("노출 단계는 연관성을 비교하기 위한 구간입니다. 정확히 4시간에 물리적 임계점이 생긴다는 증거는 아닙니다.")
        source_note("wind_dose.csv · GT 통제 체류시간 회귀")

    with st.expander("접안 시점의 풍속·파고 회귀 결과", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            legacy_factor = st.radio("접안시점 기상요인", ["wind", "wave"], format_func=lambda x: {"wind": "풍속", "wave": "파고"}[x],
                                     horizontal=True, key="research_legacy_factor")
        with cols[1]:
            threshold = st.radio("접안시점 기상강도", ["P90", "P95"], horizontal=True, key="research_legacy_threshold")
        chart(_add_point_labels(wind_wave_forest(D["weather_reg"], legacy_factor, threshold)), height=370)
        st.caption("GT·작업량·시기·혼잡 등을 보정한 접안시점 체류시간 회귀입니다. 첫 6시간 노출 분석과 다른 비교조건입니다.")
        source_note("weather_reg.csv · GT 포함 2020–2024년 접안시점 회귀")

    with st.expander("분석에 사용한 악천후 기준 · P90 / P95", expanded=False):
        _weather_reference(D, "research_weather")

    with st.expander("별도 연구 · 시간별 출항확률 모델의 기상별 잔차", expanded=False):
        st.caption("이 모델의 결과변수는 다음 1시간 내 출항확률입니다. 앞의 평균 작업효율 차이(%)와 이 아래 출항률 차이(%p)를 합산하지 않습니다.")
        cols = st.columns(2)
        with cols[0]:
            term = st.selectbox("출항률 결과를 볼 터미널", TERMINALS, key="research_hourly_terminal")
        with cols[1]:
            hourly_factor = st.radio("출항률 비교 기상 (P95)", ["wave", "wind"],
                                     format_func=lambda x: {"wave": "고파고", "wind": "강풍"}[x],
                                     horizontal=True, key="research_hourly_factor")
        chart(ml_year_chart(D["weather_hourly_ml"], term, hourly_factor, "P95"), height=360)
        st.caption("2025년 신항의 파고·시정 관측 누락과 이벤트 시점 중심 관측 때문에 표본 대표성에 제약이 있습니다.")
        source_note("weather_hourly_ml.csv · 2024 / 2025년 시간별 출항확률 잔차")


def render_terminals(D: dict, META: dict) -> None:
    page_header("터미널 현황", "터미널마다 작업 규모와 작업률 분포가 다릅니다",
                "버블의 면적은 기항 규모, 바이올린의 폭은 작업률이 모인 정도를 보여줍니다. 기상·선박 구성 차이를 보정하기 전의 현황입니다.",
                kind="현황 데이터")
    year = st.selectbox("살펴볼 기간", ["전체 기간", 2020, 2021, 2022, 2023, 2024, 2025],
                        key="research_terminal_year")
    selected = D["terminal"].copy() if year == "전체 기간" else D["annual"].loc[D["annual"].year == year].copy()
    data = _ordered(selected)
    available = data.calls.notna()
    present = data[available].copy()
    period = "2020–2025년" if year == "전체 기간" else f"{year}년"
    total = int(present.calls.sum())
    count = int(available.sum())
    max_terminal = present.loc[present.calls.idxmax()]
    takeaway(f"{period} 자료는 {count}개 터미널, {total:,}기항입니다.",
             "표본과 작업 규모의 차이가 크므로, 이 현황만으로 터미널 성과나 기상 취약도를 판단하지 않습니다.", tone="blue")
    metrics([
        {"label": "선택 기간의 분석 기항", "value": f"{total:,}건", "note": "유효 운영표본 · 추가 P99 필터 전", "tone": "blue"},
        {"label": "자료가 있는 터미널", "value": f"{count} / 7", "note": period, "tone": "slate"},
        {"label": "가장 많은 관측 기항", "value": str(max_terminal.terminal),
         "note": f"{int(max_terminal.calls):,}건 · 표본 규모", "tone": "teal"},
    ])
    section("운영 규모는 버블로, 실제 기록의 차이는 분포로",
            "색은 터미널 구분입니다. 왼쪽의 원 면적은 기항 수에 비례하고, 오른쪽은 실제 개별 기록으로 그린 작업률 분포입니다.")
    left, right = st.columns([1, 1.2])
    with left:
        chart(terminal_bubble_map(present, period), height=450)
        st.caption(f"유효 운영표본 {total:,}건의 집계입니다. 원 위에 마우스를 올리면 기항 수와 체류시간을 볼 수 있습니다.")
    with right:
        records = _distribution_records()
        distribution = records if year == "전체 기간" else records.loc[records.year.eq(year)]
        if distribution.empty:
            st.info("선택한 기간의 개별 기항 분포 자료가 없습니다.")
        else:
            render_seaborn(terminal_violin(distribution, period),
                           alt=f"{period} 기상 관측이 완전한 개별 기항의 터미널별 작업률 분포. 가운데 점은 중앙값이며 각 터미널의 폭은 별도로 정규화했습니다.")
            st.caption(f"풍속·파고·시정이 모두 있는 검색용 {len(distribution):,}건을 사용하므로 왼쪽 집계와 표본이 다릅니다. 체류 6시간 초과·추가 P99 필터 없음. 폭은 터미널별 정규화 밀도이며 표본 규모가 아닙니다.")
    missing = data.loc[~available, "terminal"].tolist()
    if missing:
        st.caption(f"{period} 자료 없음: {', '.join(missing)}. 관측이 없는 터미널을 0건·작업률 0으로 표시하지 않았습니다.")
    st.caption("바이올린의 내부 선은 사분위수, 가운데 점은 중앙값입니다. 20건 미만이면 밀도 곡선 대신 실제 점만 표시합니다. 작업률은 실제 크레인 Moves/h나 특정 한 시간의 터미널 실적과 다릅니다.")
    source_note(f"{'terminal.csv' if year == '전체 기간' else 'annual.csv'} · 유효 운영표본 집계 / case_search/case_index.csv.gz · 기상완전 개별 기항")

    with st.expander("연도 × 터미널 히트맵으로 변화 보기", expanded=False):
        render_seaborn(annual_rate_heatmap(D["annual"], year),
                       alt="2020년부터 2025년까지 터미널별 중앙 작업률의 히트맵. BCT의 2020년과 2021년은 자료 없음으로 표시합니다.")
        st.caption("색이 진할수록 보정 전 중앙 작업률이 높습니다. 자료가 없는 연도는 0으로 채우지 않았으며, 연도별 선박 구성과 기상 차이를 통제한 개선율은 아닙니다.")
        source_note("annual.csv · 연도별 유효 운영표본")

    with st.expander("터미널별 작업량·체류시간·표본의 상세 수치", expanded=False):
        table = data[["terminal", "calls", "median_cargo", "median_berth_h", "median_rate"]].copy()
        st.dataframe(table.rename(columns={"terminal": "터미널", "calls": "기항 수", "median_cargo": "중앙 작업량",
                                          "median_berth_h": "중앙 체류시간 (h)", "median_rate": "중앙 작업률 (/h)"}).round(2),
                     hide_index=True, width='stretch')
        st.caption("중앙 작업률은 개별 선박의 작업률을 먼저 계산한 뒤 구한 중앙값입니다. 중앙 작업량 ÷ 중앙 체류시간과 같을 필요가 없습니다.")
        st.download_button("선택 기간 현황 CSV 받기", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"PORTaI_terminal_{'all' if year == '전체 기간' else year}.csv",
                           mime="text/csv", key="research_terminal_download")

    with st.expander("작업률의 가운데 50% 범위 · P99 필터 적용 표본", expanded=False):
        summary = D["history_box_overall"].copy() if year == "전체 기간" else D["history_box_annual"].loc[D["history_box_annual"].year == year].copy()
        summary = _ordered(summary).dropna(subset=["median_rate"])
        fig = go.Figure(go.Scatter(x=summary.median_rate, y=summary.terminal, mode="markers+text",
                                  text=[f"{v:.1f}" for v in summary.median_rate], textposition="top center",
                                  marker=dict(size=10, color=TEAL),
                                  error_x=dict(type="data", symmetric=False,
                                               array=summary.p75 - summary.median_rate,
                                               arrayminus=summary.median_rate - summary.p25,
                                               color=TEAL, thickness=5, width=0),
                                  customdata=summary[["p25", "p75", "n"]].to_numpy(),
                                  hovertemplate="%{y}<br>중앙값 %{x:.1f}/h<br>25~75% %{customdata[0]:.1f}~%{customdata[1]:.1f}/h<br>n=%{customdata[2]:,.0f}<extra></extra>"))
        fig.update_layout(title="굵은 선은 기항 작업률의 가운데 50% 범위입니다", xaxis_title="선박별 평균 작업률 (/h)",
                          yaxis_title="", margin=dict(l=20, r=45, t=60, b=45))
        fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=TERMINALS)
        chart(fig, height=405)
        st.caption(f"이 그림은 P99 등 추가 분석 필터 적용 후의 {int(summary.n.sum()):,}기항을 사용하므로, 위 현황의 표본수·중앙값과 다릅니다. 굵은 선은 사분위 범위이며 추정치의 신뢰구간이 아닙니다.")
        source_note("history_box_overall.csv / history_box_annual.csv · P99 필터 적용 작업률 분포")

    with st.expander("북항·신항의 풍속과 파고 분포", expanded=False):
        bins = D["weather_scatter_bins"].copy()
        fig = px.scatter(bins, x="wind_bin", y="wave_bin", size="n", color="harbor",
                         color_discrete_map={"북항": BLUE, "신항": TEAL}, size_max=25, opacity=.55,
                         labels={"wind_bin": "풍속 (m/s)", "wave_bin": "유의파고 (m)", "harbor": "항만", "n": "관측 수"},
                         title="큰 원일수록 자주 관측된 기상조건입니다")
        fig.update_layout(legend=dict(orientation="h", y=1.1, x=0))
        chart(fig, height=400)
        st.caption("시간별 기상을 풍속 0.5 m/s × 파고 0.1 m 구간으로 집계했습니다. 원의 크기는 관측 빈도이며 피해 규모가 아닙니다.")
        source_note("weather_scatter_bins.csv · 기상 관측 빈도 집계")

    with st.expander("북항·신항의 악천후 기준과 터미널 구분", expanded=False):
        st.markdown("**신항:** PNIT · PNC · HPNT · BNCT · BCT  \n**북항:** BPT_S · BPTC_G")
        _weather_reference(D, "research_terminals")

    with st.expander("BCT 접안시점 강풍 유무에 따른 작업률 분포", expanded=False):
        b = D["bct_berth_weather_distribution"].copy()
        # CSV parsers normally infer bool; handle textual bools explicitly too.
        is_high = b.wind_high.astype(str).str.lower().eq("true")
        labels = np.where(is_high, "접안시점 강풍 P90", "접안시점 P90 미만")
        fig = go.Figure(go.Bar(x=labels, y=b.median_rate, marker_color=[CORAL if x else TEAL for x in is_high],
                              text=[f"{v:.1f}/h" for v in b.median_rate], textposition="outside",
                              error_y=dict(type="data", symmetric=False, array=b.p75-b.median_rate,
                                           arrayminus=b.median_rate-b.p25),
                              customdata=b.n, hovertemplate="%{x}<br>중앙 작업률 %{y:.1f}/h<br>n=%{customdata:,.0f}<extra></extra>"))
        fig.update_layout(title="접안시점 강풍 여부별 중앙값 · 보정 전 단순 비교", yaxis_title="작업률 (/h)",
                          showlegend=False)
        fig.update_yaxes(range=[0, float(b.p75.max())*1.15])
        chart(fig, height=350)
        st.caption("선은 25~75% 사분위 범위입니다. 첫 6시간 4–6회 노출을 다루는 모델 B와 다른 표본·비교조건입니다. 위의 연도 선택과 무관한 전체 집계입니다.")
        source_note("bct_berth_weather_distribution.csv · BCT 접안시점 강풍별 집계")
