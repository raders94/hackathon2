"""Result-first learning and validation pages for the PORTaI dashboard.

All displayed evidence comes from packaged aggregate tables.  A forecast error,
an adjusted association, and an assumed intervention result are different things.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic import TERMINALS
from ui import page_header, takeaway, metrics, chart, section, source_note


TEAL = "#138A81"
BLUE = "#527C9C"
CORAL = "#D66A4D"
AMBER = "#D59A32"
SLATE = "#8B98A5"
PALE = "#DFE6EE"
INK = "#17364D"
SEVERITIES = ["P90 미만", "P90 이상·P95 미만", "P95 이상"]
SEVERITY_NAMES = ["기준 미만<br>P90 미만", "중간 수준<br>P90 이상·P95 미만", "높은 수준<br>P95 이상"]
SEVERITY_COLORS = [BLUE, BLUE, CORAL]


def _table(df: pd.DataFrame) -> None:
    st.dataframe(df, width='stretch', hide_index=True)


def _download(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", key=key)


def _base(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Malgun Gothic, NanumGothic, sans-serif", size=13, color=INK),
                      margin=dict(l=22, r=30, t=24, b=45),
                      legend=dict(orientation="h", y=1.13, x=0))
    fig.update_xaxes(zeroline=False, gridcolor="#EDF1F5")
    fig.update_yaxes(zeroline=False, gridcolor="#EDF1F5")
    return fig


def aggregate_wape(groups: pd.DataFrame) -> float:
    """Recover pooled WAPE from disjoint groups; never average their WAPEs.

    n * MAE is the group's absolute-error sum. Dividing by WAPE/100
    recovers its actual-rate sum, which is the correct pooling denominator.
    Packaged groups all have positive n, MAE and WAPE.
    """
    g = groups.loc[groups["n"].gt(0)].copy()
    errors = g["n"].astype(float) * g["mae"].astype(float)
    rates = g["wape_pct"].astype(float)
    if g.empty or rates.isna().any() or rates.le(0).any():
        return float("nan")
    denominator = (errors / (rates / 100.0)).sum()
    return float(100.0 * errors.sum() / denominator) if denominator > 0 else float("nan")


def render_learning(D: dict, META: dict) -> None:
    page_header("04 / 운영학습", "평온할 때의 개선과 강풍 대응의 개선을 구분합니다",
                "같은 2022–2024년을 비교합니다. 작업효율은 총작업량 ÷ 전체 접안시간입니다.",
                kind="연도별 분석")
    target = st.selectbox("확인할 터미널", TERMINALS, key="learning_page_terminal")
    calm_all = D["learning_calm_2024_vs_2022"]
    calm = calm_all[(calm_all.terminal == target) & (calm_all.year == 2024)].iloc[0]
    years = D["learning_annual_weather_penalty"]
    b = years[years.terminal == target].sort_values("year").copy()
    slopes = D["learning_annual_slopes"]
    slope = slopes[slopes.terminal == target].iloc[0]
    improved = bool(calm.ci_low_pct > 0)
    declined = bool(calm.ci_high_pct < 0)
    calm_phrase = "평온 작업효율은 높아졌습니다" if improved else (
        "평온 작업효율은 낮아졌습니다" if declined else "평온 작업효율의 뚜렷한 변화는 확인되지 않았습니다")
    trend_sig = bool(slope.p_slope < .05)
    trend_phrase = ("강풍 대응 효율비율의 증가 추세가 관측됐습니다" if slope.annual_change_in_eff_ratio_pct > 0
                    else "강풍 대응 효율비율의 감소 추세가 관측됐습니다") if trend_sig else "강풍 대응력의 개선 추세는 확인되지 않았습니다"
    takeaway(f"{target}: {calm_phrase}.",
             f"{trend_phrase}. 평온조건 변화 {calm.change_pct:+.2f}%와 강풍 대응 추세는 서로 다른 질문입니다.",
             tone="teal" if improved else ("coral" if declined else "blue"))
    metrics([
        dict(label="평온 작업효율 · 2024 vs 2022", value=f"{calm.change_pct:+.2f}%",
             note=f"95% 신뢰구간 {calm.ci_low_pct:+.2f}~{calm.ci_high_pct:+.2f}%",
             tone="teal" if improved else ("coral" if declined else "slate")),
        dict(label="강풍 대응력 개선 추세", value="유의한 변화" if trend_sig else "미확인",
             note=f"연간 추세 검정 p={slope.p_slope:.3f}", tone="blue" if trend_sig else "amber"),
        dict(label="강풍 4–6회 관측 사례", value=f"{int(slope.n_high):,}건",
             note=f"2022–2024 분석 표본 {int(slope.n):,}건 중", tone="slate"),
    ])
    left, right = st.columns([1, 1.18])
    with left:
        section("① 평온조건에서는 얼마나 달라졌나?", "막대 길이는 조정 작업효율 지수입니다. 2022년을 100으로 맞췄습니다.")
        idx = 100 + float(calm.change_pct)
        fig = go.Figure(go.Bar(
            x=[100, idx], y=["2022 · 기준", "2024 · 비교"], orientation="h",
            marker_color=[PALE, TEAL if improved else (CORAL if declined else BLUE)],
            text=["100.0", f"{idx:.1f}"], textposition="outside", cliponaxis=False,
            error_x=dict(type="data", symmetric=False,
                         array=[0, float(calm.ci_high_pct-calm.change_pct)],
                         arrayminus=[0, float(calm.change_pct-calm.ci_low_pct)],
                         color=INK, thickness=1.5, width=5),
            hovertemplate="%{y}<br>조정 효율지수 %{x:.2f}<extra></extra>",
        ))
        fig.add_vline(x=100, line_dash="dot", line_color=SLATE)
        fig.update_layout(showlegend=False, bargap=.48, xaxis_title="조정 작업효율 지수 · 2022=100")
        fig.update_xaxes(range=[0, max(140, 100 + float(calm.ci_high_pct) + 15)])
        fig.update_yaxes(autorange="reversed")
        chart(_base(fig, 330), height=330)
        st.caption("가는 선은 2024년 추정치의 95% 신뢰구간입니다. 풍속·파고·시정이 모두 평온한 조건에서 비교했습니다.")
    with right:
        section("② 강풍에 따른 효율 차이는 줄어드나?", "점이 0%에 가까워질수록 강풍 고노출군과 0회군의 차이가 작습니다.")
        fig = go.Figure(go.Scatter(
            x=b.year.astype(int), y=b.adjusted_high_eff_pct, mode="lines+markers+text",
            text=[f"{v:+.1f}%<br>고노출 {int(n)}건" for v, n in zip(b.adjusted_high_eff_pct, b.n_4to6)],
            textposition="bottom center", textfont=dict(size=13),
            marker=dict(size=13, color=CORAL, line=dict(color="white", width=2)),
            line=dict(color=CORAL, width=3),
            error_y=dict(type="data", symmetric=False,
                         array=(b.ci_high_pct-b.adjusted_high_eff_pct).to_numpy(),
                         arrayminus=(b.adjusted_high_eff_pct-b.ci_low_pct).to_numpy(),
                         color="#DDA79E", thickness=2, width=6),
            customdata=np.column_stack([b.ci_low_pct, b.ci_high_pct, b.n_4to6]),
            hovertemplate="%{x}년 · %{y:+.2f}%<br>95% CI %{customdata[0]:+.2f}~%{customdata[1]:+.2f}%<br>고노출 %{customdata[2]:.0f}건<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=SLATE, line_dash="dash",
                      annotation_text="0% · 두 집단 차이 없음", annotation_position="top right")
        fig.update_layout(showlegend=False, yaxis_title="강풍 4–6회 vs 0회 · 조정 효율차이 (%)")
        fig.update_xaxes(tickmode="array", tickvals=[2022, 2023, 2024], ticktext=["2022", "2023", "2024"], range=[2021.7, 2024.3])
        fig.update_yaxes(range=[min(-25, float(years.ci_low_pct.min())-5), max(12, float(years.ci_high_pct.max())+3)])
        chart(_base(fig, 330), height=330)
        st.caption("세로선은 95% 신뢰구간입니다. 첫 6개 시간별 관측 중 강풍 기준 초과 횟수를 비교하며, 연속 노출시간과는 다릅니다.")
    if target == "BCT":
        st.info("BCT의 2022년 강풍 고노출은 6건입니다. 이 작은 초기 표본 때문에 개선 추세의 불확실성이 큽니다.")
    else:
        st.caption("평온조건의 변화와 악천후 대응 추세만으로 자동화·운영숙련의 인과효과를 판단할 수 없습니다.")

    with st.expander("다른 터미널도 강풍 대응력이 개선됐나요?"):
        peer = D["learning_pooled_test"]
        contrast = peer[peer.contrast.str.contains("BCT minus equal-weight", regex=False)].iloc[0]
        st.markdown(f"**BCT와 나머지 6개 터미널의 추세 차이도 뚜렷하지 않습니다(p={contrast.p:.3f}).** "
                    "이는 개선속도가 같다는 증명이 아닙니다. 이번 표본의 모든 터미널에서 추세의 95% 신뢰구간이 0을 포함합니다.")
        v = slopes.set_index("terminal").reindex(TERMINALS).reset_index()
        fig = go.Figure(go.Scatter(
            x=v.annual_change_in_eff_ratio_pct, y=v.terminal, mode="markers",
            marker=dict(size=12, color=[CORAL if t == target else BLUE for t in v.terminal]),
            error_x=dict(type="data", symmetric=False,
                         array=v.annual_change_ratio_ci_high-v.annual_change_in_eff_ratio_pct,
                         arrayminus=v.annual_change_in_eff_ratio_pct-v.annual_change_ratio_ci_low,
                         color=SLATE, thickness=1.5),
            customdata=np.column_stack([v.p_slope, v.n_high]),
            hovertemplate="%{y}: 연간 %{x:+.2f}%<br>p=%{customdata[0]:.3f}<br>고노출 %{customdata[1]:.0f}건<extra></extra>"))
        fig.add_vline(x=0, line_dash="dash", line_color=SLATE)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="강풍 효율비율의 연간 변화 (%) · 가로선은 95% 신뢰구간", showlegend=False)
        chart(_base(fig, 370))
        _table(v[["terminal", "n", "n_high", "annual_change_in_eff_ratio_pct", "annual_change_ratio_ci_low", "annual_change_ratio_ci_high", "p_slope"]].rename(columns={
            "terminal":"터미널", "n":"분석 표본", "n_high":"고노출 표본", "annual_change_in_eff_ratio_pct":"연간 변화(%)",
            "annual_change_ratio_ci_low":"95% 하한", "annual_change_ratio_ci_high":"95% 상한", "p_slope":"추세 p값"}).round(3))

    with st.expander("분석표 · 연도별 표본 · 민감도 확인"):
        st.markdown("**선택 터미널의 연도별 강풍 효율차이**")
        _table(b[["year", "n_total", "n_4to6", "n_0", "adjusted_high_eff_pct", "ci_low_pct", "ci_high_pct"]].rename(columns={
            "year":"연도", "n_total":"분석 표본", "n_4to6":"고노출 4–6회", "n_0":"비노출 0회",
            "adjusted_high_eff_pct":"조정 효율차이(%)", "ci_low_pct":"95% 하한", "ci_high_pct":"95% 상한"}).round(2))
        st.markdown("**7개 터미널의 평온조건 변화 · 2024 vs 2022**")
        _table(calm_all[calm_all.year == 2024][["terminal", "change_pct", "ci_low_pct", "ci_high_pct", "p"]].rename(columns={
            "terminal":"터미널", "change_pct":"조정 변화(%)", "ci_low_pct":"95% 하한", "ci_high_pct":"95% 상한", "p":"p값"}).round(3))
        sensitivity = D["learning_threshold_sensitivity"]
        st.markdown("**강풍 정의·강도 보정 변경에 따른 민감도**")
        _table(sensitivity[sensitivity.terminal == target].round(4))
        st.caption("평온조건의 상승은 자동화나 숙련도 효과의 직접 측정이 아닙니다. 추세 방향은 강풍 정의와 통제 방식에 영향을 받을 수 있습니다.")
        _download("연도별 결과 CSV", years, "PORTaI_learning_annual.csv", "learning_annual_download_v6")
        _download("터미널별 추세 CSV", slopes, "PORTaI_learning_slopes.csv", "learning_slopes_download_v6")
    source_note("출처: learning_calm_2024_vs_2022.csv · learning_annual_weather_penalty.csv · learning_annual_slopes.csv · learning_pooled_test.csv. 공통기간 2022–2024.")


def render_validation(D: dict, META: dict) -> None:
    page_header("05 / 예측 검증", "실제 2025년 자료에서 예측오차를 확인합니다",
                "2020–2024년 학습을 마친 비GT 모델을 고정해 평가했습니다. 오차 막대가 짧을수록 실제값에 가깝습니다.",
                kind="미학습 자료 평가")
    selected = st.selectbox("성능을 확인할 터미널", ["전체"] + TERMINALS, key="validation_page_terminal")
    severity = D["validation_severity_2025"]
    group = severity[severity.terminal == selected].set_index("severity").reindex(SEVERITIES).reset_index()
    avail = D["validation_availability_2025"]
    coverage = avail[avail.terminal == selected].iloc[0]
    overall = aggregate_wape(group)
    baseline, high = group.iloc[0], group.iloc[2]
    diff = float(high.wape_pct - baseline.wape_pct)
    direction = "높았습니다" if diff >= 0 else "낮았습니다"
    takeaway(f"{selected} 예측의 전체 상대오차는 {overall:.2f}%입니다.",
             f"P95 이상 조건의 오차는 기준 미만보다 {abs(diff):.2f}%p {direction}. "
             "날씨별 표본 구성도 달라, 이 차이를 날씨의 인과효과로 해석할 수는 없습니다.", tone="blue")
    metrics([
        dict(label="전체 상대오차 · WAPE", value=f"{overall:.2f}%", note=f"2025년 {int(coverage.evaluated_n):,}건 · 낮을수록 좋음", tone="blue"),
        dict(label="P95 이상 상대오차", value=f"{high.wape_pct:.2f}%", note=f"기준 미만 대비 {diff:+.2f}%p · {int(high.n):,}건", tone="coral"),
        dict(label="평가에 포함된 비율", value=f"{coverage.coverage_pct:.1f}%", note=f"후보 {int(coverage.candidate_n):,}건 중 기상 완전관측", tone="teal"),
    ])
    left, right = st.columns([1.2, 1])
    with left:
        section("① 날씨별 예측오차", "같은 0 기준에서 막대 높이를 비교하세요. 색은 기상등급을 구분합니다.")
        fig = go.Figure(go.Bar(x=SEVERITY_NAMES, y=group.wape_pct, marker_color=SEVERITY_COLORS,
            text=[f"<b>{v:.2f}%</b><br>{int(n):,}건" for v, n in zip(group.wape_pct, group.n)],
            textposition="outside", cliponaxis=False,
            customdata=np.column_stack([group.n, group.mae]),
            hovertemplate="%{x}<br>WAPE %{y:.2f}%<br>평가 %{customdata[0]:,.0f}건<br>MAE %{customdata[1]:.2f} 작업단위/h<extra></extra>"))
        fig.update_layout(showlegend=False, yaxis_title="상대 예측오차 · WAPE (%)", bargap=.42)
        fig.update_yaxes(range=[0, max(25, float(severity.wape_pct.max())*1.28)])
        chart(_base(fig, 350), height=350)
        st.caption("WAPE = 전체 절대오차 합 ÷ 실제 작업률 합 × 100. 접안 시점의 기상 1개 관측값으로 구분한 서로 겹치지 않는 세 그룹입니다.")
    with right:
        section("② 얼마나 많은 자료를 평가했나?", "막대 전체가 평가 후보입니다. 청록색만 실제 모델 검증에 포함됐습니다.")
        included = float(coverage.coverage_pct)
        excluded = 100-included
        fig = go.Figure()
        for name, pct, number, color, textcolor in [
            ("평가 포함", included, int(coverage.evaluated_n), TEAL, "white"),
            ("기상 미관측 제외", excluded, int(coverage.excluded_incomplete_n), PALE, INK),
        ]:
            text = f"<b>{pct:.1f}%</b><br>{number:,}건" if pct >= 7 else ""
            fig.add_trace(go.Bar(x=[pct], y=[selected], orientation="h", name=name,
                marker_color=color, text=[text], textposition="inside", insidetextanchor="middle",
                textfont=dict(color=textcolor, size=17),
                hovertemplate=f"{name}<br>{number:,}건 ({pct:.2f}%)<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=.58,
                          legend=dict(orientation="h", y=1.07, x=0),
                          xaxis_title="평가 후보에서 차지하는 비율 (%)")
        fig.update_xaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
        fig.update_yaxes(showticklabels=False)
        chart(_base(fig, 260), height=260)
        st.caption(f"기상 미관측으로 {int(coverage.excluded_incomplete_n):,}건이 제외됐습니다. "
                   "특히 신항의 2025년 기상 기록은 선박 이벤트 시점 중심이어서, 검증 표본의 대표성이 제한됩니다.")
    if (group.n < 100).any():
        st.info("선택한 터미널에는 평가 사례가 100건 미만인 기상등급이 있습니다. 작은 집단의 오차 차이는 조심해서 해석하세요.")

    with st.expander("기상요인별 상세 · 7개 터미널 비교"):
        factors = D["validation_weather_factors_2025"]
        f = factors[factors.terminal == selected].copy()
        st.markdown("**강풍·파고·시정 조건은 서로 겹칠 수 있습니다. 조건별 표본 수를 합산하지 않습니다.**")
        eligible = f[f.n >= 30].sort_values("wape_pct", ascending=False)
        fig = go.Figure(go.Bar(x=eligible.wape_pct, y=eligible.condition, orientation="h",
            marker_color=[CORAL if "P95" in s else BLUE for s in eligible.condition],
            text=[f"{v:.2f}% · {int(n):,}건" for v, n in zip(eligible.wape_pct, eligible.n)], textposition="outside", cliponaxis=False))
        fig.update_xaxes(range=[0, max(25, float(eligible.wape_pct.max())*1.4)])
        fig.update_layout(showlegend=False, xaxis_title="WAPE (%) · 낮을수록 좋음")
        chart(_base(fig, 370))
        s = severity[severity.terminal != "전체"]
        heat = s.pivot(index="terminal", columns="severity", values="wape_pct").reindex(index=TERMINALS, columns=SEVERITIES)
        count = s.pivot(index="terminal", columns="severity", values="n").reindex(index=TERMINALS, columns=SEVERITIES)
        texts = [[f"{heat.loc[t,c]:.1f}%<br>{int(count.loc[t,c]):,}건" for c in SEVERITIES] for t in TERMINALS]
        fig = go.Figure(go.Heatmap(z=heat.to_numpy(), x=SEVERITIES, y=TERMINALS, text=texts, texttemplate="%{text}",
            colorscale=[[0,"#EFF6FA"],[.5,"#93BBD8"],[1,"#355F91"]], zmin=0,
            colorbar=dict(title="WAPE (%)"), hovertemplate="%{y} · %{x}<br>WAPE %{z:.2f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        chart(_base(fig, 430))
        st.caption("진한 칸일수록 예측오차가 큽니다. 터미널마다 선박·기상 구성이 달라 운영성과 순위로 해석할 수 없습니다.")
        f["표본수 안내"] = np.where(f.n < 30, "30건 미만 · 참고용", np.where(f.n < 100, "100건 미만 · 주의", "100건 이상"))
        _table(f[["condition","n","train_n","mae","wape_pct","r2","표본수 안내"]].rename(columns={
            "condition":"기상조건", "n":"평가 건수", "train_n":"학습 사례", "mae":"MAE /h", "wape_pct":"WAPE (%)", "r2":"R²"}).round(3))

    with st.expander("실제값과 예측값 비교 · 상세 성능표"):
        c = D["validation_calibration_2025"]
        c = c[c.terminal == selected]
        fig = go.Figure()
        for kind, color in zip(SEVERITIES, SEVERITY_COLORS):
            p = c[c.severity == kind]
            fig.add_trace(go.Scatter(x=p.mean_pred, y=p.mean_actual, mode="markers", name=kind,
                marker=dict(size=11, color=color, opacity=.85), customdata=p.n,
                hovertemplate="평균 예측 %{x:.2f}/h<br>평균 실제 %{y:.2f}/h<br>%{customdata:,.0f}건<extra></extra>"))
        lim = max(float(c.mean_pred.max()), float(c.mean_actual.max()))*1.08
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color=SLATE, dash="dash"))
        fig.update_layout(xaxis_title="평균 예측 작업률 (/h)", yaxis_title="평균 실제 작업률 (/h)")
        fig.update_xaxes(range=[0,lim]); fig.update_yaxes(range=[0,lim])
        chart(_base(fig, 410))
        st.caption("점이 대각선에 가까울수록 집단 평균 예측과 실제가 비슷합니다. 각 점은 예측값 구간의 집계 결과입니다.")
        _table(group[["severity","train_n","n","mae","wape_pct","r2","bias","p90_abs_error"]].rename(columns={
            "severity":"기상등급", "train_n":"학습 사례", "n":"평가 건수", "mae":"MAE /h", "wape_pct":"WAPE (%)",
            "r2":"R²", "bias":"평균 편향(예측−실제)", "p90_abs_error":"절대오차 P90 /h"}).round(3))
        _table(avail.rename(columns={"terminal":"터미널", "candidate_n":"평가 후보", "evaluated_n":"평가 포함", "excluded_incomplete_n":"미관측 제외", "coverage_pct":"포함 비율(%)"}).round(2))
        st.caption("R²는 설명력 지표이며 정확도 백분율이 아닙니다. 2025년 성능은 비GT 모형만 검증했습니다. GT 모형에 이 수치를 적용하지 않습니다.")

    with st.expander("날씨를 추가하면 예측오차도 줄어드나요? · 모델 B 검증"):
        a = D["modelb_2024_prediction_ablation"].query("group == 'ALL'").iloc[0]
        b = D["modelb_2025_prediction_ablation"].query("group == 'ALL'").iloc[0]
        before = [float(a.normal_wape_pct), float(b.operational_only_wape)]
        after = [float(a.modelB_wape_pct), float(b.operational_plus_weather_wape)]
        st.markdown("**현재 비교에서는 날씨 추가에 따른 전체 예측오차 개선이 확인되지 않았습니다.** "
                    "기상과 효율 차이의 연관성, 기상변수의 추가 예측력은 별도로 판단해야 합니다.")
        fig = go.Figure()
        labels = [f"2024 GT · {int(a.n):,}건", f"2025 비GT · {int(b.n):,}건"]
        for label, values, color in [("운영조건만", before, TEAL), ("기상정보 포함", after, CORAL)]:
            fig.add_trace(go.Bar(x=labels, y=values, name=label, marker_color=color,
                text=[f"{x:.2f}%" for x in values], textposition="outside"))
        fig.update_layout(barmode="group", yaxis_title="WAPE (%) · 낮을수록 좋음")
        fig.update_yaxes(range=[0, 21])
        chart(_base(fig, 350))
        st.caption("이 비교는 모델 B의 별도 실험입니다. 상단 Model A의 전체 WAPE 15.63%와 동일한 모델 설정을 뜻하지 않습니다.")

    with st.expander("모델 B의 SHAP · 무엇을 크게 반영했나?"):
        sh = D["modelb_weather_shap"].sort_values("mean_abs_shap_log")
        names = {"terminal":"터미널", "wave_overshoot_sum":"파고 기준 초과강도 합", "wind_overshoot_sum":"풍속 기준 초과강도 합",
                 "vis_P90_hours":"저시정 P90 관측횟수", "vis_P95_hours":"저시정 P95 관측횟수",
                 "wave_P90_hours":"고파고 P90 관측횟수", "wave_P95_hours":"고파고 P95 관측횟수",
                 "wind_P90_hours":"강풍 P90 관측횟수", "wind_P95_hours":"강풍 P95 관측횟수"}
        fig = go.Figure(go.Bar(x=sh.mean_abs_shap_log, y=sh.variable.map(names).fillna(sh.variable), orientation="h",
            marker_color=BLUE, text=[f"{x:.4f}" for x in sh.mean_abs_shap_log], textposition="outside", cliponaxis=False))
        fig.update_xaxes(range=[0,float(sh.mean_abs_shap_log.max())*1.28])
        fig.update_layout(showlegend=False, xaxis_title="평균 절대 SHAP · log 작업률 단위")
        chart(_base(fig, 370))
        st.caption("2024년 2단계 잔차모델의 예측기여도입니다. 막대가 길수록 모델 예측에 크게 반영됐다는 뜻이며 실제 피해의 원인 순위가 아닙니다.")
        _table(sh.rename(columns={"variable":"모델 변수", "mean_abs_shap_log":"평균 |SHAP|", "mean_shap_log":"평균 SHAP"}).round(5))

    with st.expander("검증 기준 · 해석 범위 · 결과 다운로드"):
        st.markdown("""
- 기상등급은 **접안 시점** 관측값으로 구분합니다. 첫 6회 기상 노출을 쓰는 모델 B 영향분석과 다릅니다.
- 세 기상요소 중 하나라도 P95 수준이면 높은 등급, 그 외 P90 수준이 있으면 중간 등급입니다. 저시정은 낮은 시정값이 악천후입니다.
- 2025년 기상 완전관측 사례만 평가했습니다. 특히 신항 표본은 전체 운영상황의 대표성이 제한됩니다.
- 작업률은 **총작업량 ÷ 전체 접안시간**입니다. 실제 크레인 Moves/h나 특정 1시간 하역실적은 아닙니다.
- 성능평가는 기록된 작업량을 사용한 과거 검증입니다. 접안 전 활용 성능은 당시 알 수 있는 계획 작업량으로 별도 검증해야 합니다.
- 전체 WAPE는 그룹별 실제 작업률 합으로 가중합니다. 그룹 WAPE의 단순 평균을 쓰지 않습니다.
""")
        _download("기상등급별 성능 CSV", severity, "PORTaI_validation_severity_2025.csv", "validation_severity_download_v6")
        _download("기상요인별 성능 CSV", D["validation_weather_factors_2025"], "PORTaI_validation_factors_2025.csv", "validation_factors_download_v6")
    source_note("출처: validation_severity_2025.csv · validation_availability_2025.csv · validation_weather_factors_2025.csv · validation_calibration_2025.csv. 2020–2024 학습 / 2025 평가.")
