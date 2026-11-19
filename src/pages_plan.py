"""Planning page: explicit recovery assumptions and a proposed field pilot.

The yellow segment is a user-selected fraction of a conditional model gap.
It is not an estimated intervention effect or an operational digital twin.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from digital_twin_lab import TrainingLab
from logic import HARBOR, Scenario, calm_for, estimate
from ui import chart, metrics, page_header, section, source_note, takeaway


CURRENT = "#138A81"
ASSUMED = "#D59A32"
REMAINING = "#E7EEF3"


def _starting_case(D, model):
    """Use the last prediction, or an explicitly labelled illustrative case."""
    latest = st.session_state.get("latest_prediction")
    if isinstance(latest, dict) and isinstance(latest.get("scenario"), Scenario):
        scenario = latest["scenario"]
        use_gt = bool(latest.get("use_gt", False))
        calm = latest.get("calm") or calm_for(
            D["calm_weather_by_month"], scenario.terminal, scenario.when.month
        )
        return scenario, use_gt, calm, "AI 예측 화면에서 마지막으로 선택한 조건"
    scenario = Scenario(
        "BCT", datetime(2027, 11, 20, 10), 6.0, 1.2, 5000.0, 396.0, 2.0, 2
    )
    calm = calm_for(D["calm_weather_by_month"], "BCT", 11)
    return scenario, False, calm, "BCT 기본 가상조건 · AI 예측 화면에서 변경 가능"


def _recovery_figure(lab: TrainingLab, pct_a: int, pct_b: int):
    """Show each assumed recovery as an accurately positioned dumbbell.

    The x axis deliberately magnifies the interval between the two predictions;
    neither endpoint, segment length, nor marker area implies an observed gain.
    """
    names = [f"목표 A · {pct_a}% 회복 가정", f"목표 B · {pct_b}% 회복 가정"]
    targets = [lab.target(pct_a), lab.target(pct_b)]
    fig = go.Figure()
    for name, target in zip(names, targets):
        fig.add_trace(go.Scatter(
            x=[lab.exposed, lab.baseline], y=[name, name], mode="lines",
            line=dict(color=REMAINING, width=11), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[lab.exposed, target], y=[name, name], mode="lines",
            line=dict(color=ASSUMED, width=11), hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[lab.exposed] * 2, y=names, mode="markers", name="현재 조건의 예측",
        marker=dict(size=22, color=CURRENT, line=dict(color="white", width=2)),
        hovertemplate="현재 조건의 AI 예측<br>%{x:,.2f} 작업단위/h<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[lab.baseline] * 2, y=names, mode="markers", name="평온 기준 예측",
        marker=dict(size=24, color="white", line=dict(color="#436779", width=2.5)),
        hovertemplate="평온 조건의 AI 예측<br>%{x:,.2f} 작업단위/h<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=targets, y=names, mode="markers+text", name="사용자 가정 목표",
        marker=dict(size=18, color=ASSUMED, symbol="diamond", line=dict(color="white", width=2)),
        text=[f"<b>{target:,.2f} /h</b><br>가정 회복 +{target-lab.exposed:,.2f} /h" for target in targets],
        textposition="top center", textfont=dict(color="#9F6D16", size=13), cliponaxis=False,
        customdata=[[target-lab.exposed, pct] for target, pct in zip(targets, [pct_a, pct_b])],
        hovertemplate="%{y}<br>가정 목표 %{x:,.2f} 작업단위/h<br>가정 회복 +%{customdata[0]:,.2f} /h<extra></extra>",
    ))
    span = max(abs(lab.baseline-lab.exposed), max(lab.exposed, lab.baseline)*.035, .1)
    low = min(lab.exposed, lab.baseline) - span*.35
    high = max(lab.exposed, lab.baseline) + span*.35
    fig.add_vline(x=lab.exposed, line_color=CURRENT, line_width=1, line_dash="dot", opacity=.5)
    fig.add_vline(x=lab.baseline, line_color="#436779", line_width=1, line_dash="dot", opacity=.5)
    fig.update_layout(
        height=355, margin=dict(t=45, b=65, l=12, r=30),
        xaxis_title="선박당 평균 작업률 (/h) · 두 예측 사이의 차이를 확대해 표시",
        yaxis_title=None,
        legend=dict(orientation="h", y=-.28, x=0, font=dict(size=11)),
        hovermode="closest",
    )
    fig.update_xaxes(range=[low, high], zeroline=False, tickformat=".1f")
    fig.update_yaxes(type="category", categoryorder="array", categoryarray=names,
                     autorange="reversed", showgrid=False)
    return fig


def _roadmap():
    phases = [
        ("01", "1–30일", "데이터 기준 합의", "선석·장비·기상 로그의 연결 기준과 누락·이상치 처리 규칙을 정합니다.", "산출물 · 데이터 연계 명세"),
        ("02", "31–60일", "실제 작업 기록 수집", "크레인별 Moves/h, 작업중단·재개, 장비·작업조 배치 이력을 모읍니다.", "산출물 · 시간별 운영 기록"),
        ("03", "61–75일", "악천후 훈련 설계", "유사 선박·날씨·혼잡에서 작업순서와 중단·재개 대응안을 비교하도록 설계합니다.", "산출물 · 훈련·대조조건 설계"),
        ("04", "76–90일", "현장 실증 준비", "중단시간·재개시간·실제 작업률을 측정할 수 있는지 확인하고 다음 시험을 결정합니다.", "산출물 · 실증 준비 평가"),
    ]
    cards = []
    for num, period, title, body, output in phases:
        cards.append(
            '<article class="portai-plan-step">'
            f'<div class="plan-step-meta"><span>{escape(num)}</span><b>{escape(period)}</b></div>'
            f'<h4>{escape(title)}</h4><p>{escape(body)}</p>'
            f'<div class="plan-step-output">{escape(output)}</div></article>'
        )
    st.markdown(
        """<style>
        .portai-plan-roadmap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:12px 0 14px;}
        .portai-plan-step{border:1px solid #DDE7ED;border-top:4px solid #14889B;border-radius:13px;padding:17px;background:#fff;display:flex;flex-direction:column;}
        .plan-step-meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
        .plan-step-meta span{font-size:1.4rem;font-weight:800;color:#14889B;}
        .plan-step-meta b{font-size:.8rem;color:#527080;}
        .portai-plan-step h4{font-size:1rem;line-height:1.45;color:#193C50;margin:0 0 10px;}
        .portai-plan-step p{font-size:.83rem;line-height:1.65;color:#527080;flex:1;margin:0 0 14px;word-break:keep-all;}
        .plan-step-output{font-size:.73rem;color:#1F6B7D;border-top:1px solid #E4EDF2;padding-top:10px;}
        @media(max-width:1150px){.portai-plan-roadmap{grid-template-columns:repeat(2,minmax(0,1fr));}}
        @media(max-width:650px){.portai-plan-roadmap{grid-template-columns:1fr;}}
        </style><div class="portai-plan-roadmap">""" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_plan(D, META, model):
    page_header(
        "개선 계획", "회복 목표를 정하고, 현장에서 검증합니다",
        "노란 부분은 사용자가 정한 회복 가정입니다. 디지털 트윈 훈련과 현장 효과 검증은 후속 계획입니다.",
        kind="사용자 가정 · 후속 계획",
    )
    scenario, use_gt, calm, origin = _starting_case(D, model)
    threshold = D["weather_thresholds"].set_index("harbor").loc[HARBOR[scenario.terminal]]

    with st.expander("계산에 사용할 날씨 조건", expanded=False):
        weather_case = st.selectbox(
            "날씨 시나리오", ["선택한 예측 조건", "P95 강풍 추가", "P95 강풍·고파고 추가", "P95 저시정 추가"],
            key="plan_weather_case",
        )
        st.caption(origin + ". 기상 예보 API와 연결된 값이 아닙니다.")
    sim = scenario
    if weather_case == "P95 강풍 추가":
        sim = replace(scenario, wind=max(scenario.wind, float(threshold.wind_P95)))
    elif weather_case == "P95 강풍·고파고 추가":
        sim = replace(scenario, wind=max(scenario.wind, float(threshold.wind_P95)), wave=max(scenario.wave, float(threshold.wave_P95)))
    elif weather_case == "P95 저시정 추가":
        sim = replace(scenario, visibility_m=min(scenario.visibility_m, float(threshold.vis_10m_P95) * 10))
    assessment = estimate(model(use_gt), sim, calm, use_gt)
    # The planning page deliberately uses the vessel-average quantity. The
    # original TrainingLab's aggregate keys are populated with vessel values.
    lab = TrainingLab.from_model({
        "aggregate_normal": assessment["normal"], "aggregate": assessment["vessel"]
    })
    context = (
        f"{sim.terminal} · {sim.when:%Y-%m-%d %H:00} KST · 풍속 {sim.wind:.1f} m/s · "
        f"파고 {sim.wave:.2f} m · 시정 {sim.visibility_m:,.0f} m"
    )

    section("회복 목표 계산기", context + " · 사용자 입력 가상조건")
    controls_a, controls_b = st.columns(2)
    with controls_a:
        pct_a = st.slider("목표 A · 예측 차이의 회복 가정", 0, 100, 20, 5, format="%d%%", key="plan_recovery_a")
    with controls_b:
        pct_b = st.slider("목표 B · 예측 차이의 회복 가정", 0, 100, 40, 5, format="%d%%", key="plan_recovery_b")
    if lab.remaining_gap > 1e-8:
        takeaway(
            f"예측 차이의 {pct_b}%를 회복한다고 가정하면, 목표 B는 {lab.target(pct_b):,.1f} 작업단위/h입니다.",
            "회복 비율은 사용자 가정입니다. 작업배치·훈련으로 이 효과가 발생한다는 검증 결과는 아직 없습니다.",
            tone="amber",
        )
    else:
        takeaway(
            "현재 조건에서는 평온 기준보다 낮은 예측 차이가 없습니다.",
            "회복분을 0으로 표시합니다. 실제 운영전략의 효과가 없다는 뜻은 아닙니다.",
            tone="slate",
        )
    metrics([
        {"label": "현재 조건의 AI 예측", "value": f"{lab.exposed:,.1f} /h", "note": "선박당 평균 작업률", "tone": "blue"},
        {"label": f"목표 A · {pct_a}% 회복 가정", "value": f"{lab.target(pct_a):,.1f} /h", "note": f"현재 예측보다 +{lab.target(pct_a) - lab.exposed:,.2f} /h · 가정값", "tone": "amber"},
        {"label": f"목표 B · {pct_b}% 회복 가정", "value": f"{lab.target(pct_b):,.1f} /h", "note": f"현재 예측보다 +{lab.target(pct_b) - lab.exposed:,.2f} /h · 가정값", "tone": "amber"},
    ])
    chart(_recovery_figure(lab, pct_a, pct_b))
    st.caption(
        "두 예측 사이의 차이를 확대했습니다. 청록 원=현재 예측, 노란 마름모=가정 목표, 빈 원=평온 기준입니다. 노란 구간은 가정한 회복분입니다. "
        "회복 40%는 작업률 40% 상승이 아니라 두 예측값 사이 차이의 40%입니다."
    )

    with st.expander("산식·정확한 수치와 CSV 다운로드", expanded=False):
        st.markdown("**목표값 = 현재 예측 + max(평온 예측 − 현재 예측, 0) × 회복 가정 비율**")
        st.caption("작업률은 총작업량 ÷ 전체 접안 후 체류시간의 예측치입니다. 실제 크레인 Moves/h와 다릅니다.")
        data = pd.DataFrame({
            "항목": ["현재 조건 예측", "목표 A", "목표 B"],
            "회복 가정(%)": [0, pct_a, pct_b],
            "선박당 평균 작업률(/h)": [lab.exposed, lab.target(pct_a), lab.target(pct_b)],
            "평온 기준 대비(%)": [lab.target_delta(0), lab.target_delta(pct_a), lab.target_delta(pct_b)],
            "수치 구분": ["모델 예측", "사용자 가정", "사용자 가정"],
        })
        st.dataframe(data.round(3), hide_index=True, width='stretch')
        st.caption(
            f"보조 합산: 동시 접안 {sim.active_ships}척에 같은 대표 작업률을 적용하면 현재 {lab.exposed * sim.active_ships:,.1f}/h, "
            f"목표 B {lab.target(pct_b) * sim.active_ships:,.1f}/h입니다. 실제 터미널 시간당 실적을 뜻하지 않습니다."
        )
        data["시나리오"] = context
        data["평온 기준(/h)"] = lab.baseline
        data["주의"] = "회복비율은 사용자 가정; 실제 정책효과·디지털 트윈 성과 미검증"
        st.download_button(
            "회복 목표 계산표 다운로드", data.to_csv(index=False).encode("utf-8-sig"),
            "PORTaI_recovery_assumptions.csv", "text/csv", key="plan_export_assumptions",
        )
    source_note("계산 근거: 저장된 CatBoost 모델 · calm_weather_by_month.csv · src/digital_twin_lab.py. 회복 비율은 사용자 입력.")

    st.divider()
    section("90일 실증 준비 계획", "BCT에서 실제 운영 기록을 확보하고, 가상훈련을 검증할 조건을 만듭니다. 아래 단계는 추진 제안입니다.")
    _roadmap()
    st.caption("90일은 데이터 연계·훈련 설계·실증 준비 기간입니다. 운영 디지털 트윈 완성이나 생산성 개선율을 약속하는 일정이 아닙니다.")
    with st.expander("왜 이 계획이 필요한가? · 필요한 자료와 평가 기준", expanded=False):
        st.markdown(
            "BCT의 강풍 고노출군에서 보정 작업효율 저하 신호가 관측됐지만, "
            "강풍 대응효율이 시간에 따라 개선됐는지는 명확하지 않았습니다. "
            "운영경험·자동화 환경·배치정책을 실제 로그와 연결해 후속 검증할 필요가 있습니다."
        )
        st.caption("BCT 자체 연간 추세 p=0.794 · BCT와 6개 비교터미널의 추세 차이 p=0.910. 두 검정은 서로 다르며 동등성이나 숙련 효과를 입증하지 않습니다.")
        st.dataframe(pd.DataFrame([
            {"필요 자료": "크레인·선석별 실제 작업 기록", "확인할 지표": "실제 Moves/h, 작업중단·재개 시간"},
            {"필요 자료": "장비·작업조·작업순서 기록", "확인할 지표": "유사조건에서 대응안별 차이"},
            {"필요 자료": "기상·경보·운영자 대응 이력", "확인할 지표": "경보 정확도, 오경보, 대응시간"},
            {"필요 자료": "훈련 전후 비교와 대조조건", "확인할 지표": "훈련 효과와 다른 운영변화의 구분"},
        ]), hide_index=True, width='stretch')

    memo = """# PORTaI · BCT 90일 실증 준비 계획

## 목표
실제 시간별 작업·장비·중단 기록을 연결하고 악천후 대응훈련의 현장 검증을 준비합니다.

## 확인된 분석과 아직 확인되지 않은 효과
- Model B: 첫 6개 시간별 관측 중 강풍 P90 4–6회 vs 0회 BCT 조정 작업효율 차이 −10.74%. 관측자료의 연관성입니다.
- BCT 연간 강풍 대응 추세 p=0.794; BCT와 6개 비교터미널의 추세 차이 p=0.910.
- 유의한 개선이나 추세 차이의 미확인은 동일한 개선속도·자동화 원인·숙련 효과의 입증이 아닙니다.
- 화면의 회복 비율과 목표값은 사용자 가정이며 검증된 정책효과가 아닙니다.

## 추진 제안
1. 1–30일: 데이터 연계 기준과 누락·이상치 규칙 확정.
2. 31–60일: 크레인별 실제 Moves/h, 중단·재개, 배치 이력 수집.
3. 61–75일: 유사조건의 훈련 시나리오·대조조건·평가 지표 설계.
4. 76–90일: 관측 가능성과 데이터 품질을 점검하고 후속 현장 시험 결정.

## 결과 평가
실제 작업률, 중단·재개 시간, 대응시간, 경보 정확도와 오경보를 비교합니다.
90일은 실증 준비 기간이며 운영 디지털 트윈 완성, 절감액 또는 개선율을 보장하지 않습니다.
"""
    st.download_button(
        "90일 실증 준비 계획 다운로드", memo.encode("utf-8-sig"),
        "PORTaI_90day_pilot_plan.md", "text/markdown", key="plan_export_pilot",
    )

