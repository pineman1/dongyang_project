"""Archived reference demo: python -m streamlit run premium_reference_app.py."""

import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from premium.data import DATA_PATH, PLAN_COLUMNS, REFUNDS, SOURCE_PATH, TARGET, TERMS, load_data
from premium.model import MODEL_NAME, NOTICE, PremiumPredictor, PremiumRequest
from premium.train import evaluate


@st.cache_resource
def get_resources(dataset_hash: str):
    # Hash argument invalidates the cache if the verified dataset changes.
    data = load_data()
    return PremiumPredictor(data), evaluate(data)[0]


def main():
    st.set_page_config(page_title="동양생명 월 보험료 예측", page_icon="📊", layout="wide")
    st.caption("PREMIUM LAB · 동양생명 공개 자료 기반")
    st.title("월 보험료 예측")
    st.write("나이와 보장 조건을 바꾸며 공개 보험료 예시와 모델의 추정값을 비교해 보세요.")
    st.warning("2023년 9월 개정 「무배당 엔젤안심보험」 기준입니다. 현재 판매 보험료·가입 견적이 아닙니다.")
    try:
        source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        predictor, report = get_resources(hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"검증된 데이터를 불러올 수 없습니다: {exc}")
        st.stop()

    controls, output = st.columns([1, 2], gap="large")
    with controls:
        st.subheader("가입 조건")
        age = st.number_input("보험나이", min_value=30, max_value=50, value=40, step=1)
        st.caption("표에서 관측한 나이는 30·40·50세입니다. 중간 나이는 추정합니다.")
        sex = st.radio("성별 (원문 표 기준)", ["남자", "여자"], horizontal=True)
        coverage = st.selectbox("보장 유형", list(TERMS))
        term = st.selectbox("보험료 납입기간", TERMS[coverage], index=len(TERMS[coverage]) - 1,
                            format_func=lambda value: f"{value}년납", key=f"term_{coverage}")
        refund = st.selectbox("해약환급금 유형", REFUNDS)
        st.info("고정 조건\n\n보험가입금액 1,000만원 · 110세 만기 · 월납 · 주계약")
        st.caption("가입금액은 보장 항목별 지급 보험금과 다릅니다. 특약·심사 할증·할인은 반영하지 않습니다.")
    request = PremiumRequest(int(age), sex, coverage, refund, int(term))
    result = predictor.predict(request)
    with output:
        st.subheader("예상 월 보험료")
        predicted, published = st.columns(2)
        predicted.metric("회귀 모델 추정", f"{result['monthly_premium_krw']:,}원 / 월")
        if result["published_example_krw"] is None:
            published.metric("같은 나이의 원문 예시", "미공개")
            st.caption("선택한 나이는 원문에 없습니다. 양옆 연령의 공개 표를 학습한 추정값입니다.")
        else:
            published.metric("같은 조건의 원문 예시", f"{result['published_example_krw']:,}원 / 월")
            st.caption("원문 예시는 조회값이며 모델의 예측값과 구분해 표시합니다.")

        curve_rows = []
        for curve_age in range(30, 51):
            point = predictor.predict(PremiumRequest(curve_age, sex, coverage, refund, int(term)))
            curve_rows.append({"보험나이": curve_age, "월 보험료(원)": point["monthly_premium_krw"]})
        selected = predictor.data
        for field in PLAN_COLUMNS:
            selected = selected.loc[selected[field] == getattr(request, field)]
        observed = selected[["age", TARGET]].rename(columns={"age": "보험나이", TARGET: "월 보험료(원)"})
        line = alt.Chart(pd.DataFrame(curve_rows)).mark_line(color="#0f766e", strokeWidth=3).encode(
            x=alt.X("보험나이:Q", scale=alt.Scale(domain=[30, 50]), axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("월 보험료(원):Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=",")),
            tooltip=["보험나이:Q", alt.Tooltip("월 보험료(원):Q", format=",")],
        )
        dots = alt.Chart(observed).mark_point(color="#1d4ed8", size=100, filled=True).encode(
            x="보험나이:Q", y="월 보험료(원):Q",
            tooltip=["보험나이:Q", alt.Tooltip("월 보험료(원):Q", format=",")],
        )
        st.altair_chart((line + dots).properties(height=290), use_container_width=True)
        st.caption("초록선: 모델 추정 · 파란점: 원문 예시 · 세로축은 0에서 시작하지 않습니다.")
        st.caption(NOTICE)

    with st.expander("학습 데이터와 검증 결과", expanded=False):
        st.write("원문 표 72개를 사용했습니다. 합성 데이터와 고객 계약 정보는 없습니다.")
        st.write("30·50세 48개로 학습하고 40세 24개를 통째로 분리해 평가했습니다. "
                 "화면의 최종 모델은 평가 후 72개 전체로 다시 학습했습니다.")
        score = report["comparison"][MODEL_NAME]
        st.write(f"40세 검증 MAE: {score['mae_krw']:,.2f}원 · MAPE: {score['mape_percent']:.4f}%")
        st.caption("이 수치는 과거의 동일 상품표 안에서만 유효합니다. 현재 보험료, 다른 상품, "
                   "다른 가입금액 또는 실제 고객에 대한 정확도를 나타내지 않습니다. 통계적 신뢰구간은 제공하지 않습니다.")
        st.dataframe(pd.DataFrame(report["comparison"]).T, use_container_width=True)
        st.write("로그선형 회귀는 나이와 24개 가입 조건 조합을 학습합니다. "
                 "표 조회·보간으로도 충분히 풀 수 있는 작은 데이터이므로, 보간 기준선도 함께 비교합니다.")
        st.dataframe(predictor.data, hide_index=True, use_container_width=True)
        st.download_button("학습 CSV 다운로드", DATA_PATH.read_bytes(), "premium_examples.csv", "text/csv")
    st.markdown(f"출처: [동양생명 상품안내장 — 하나은행 공개 PDF]({source['source_url']}) · "
                "PDF 3쪽(인쇄 4~5쪽) · 2023.9 개정 · 확인일 2026.09.17")


if __name__ == "__main__":
    main()
