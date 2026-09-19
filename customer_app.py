"""Run with the existing venv: python -m streamlit run customer_app.py."""

import pandas as pd
import streamlit as st

from customer_ml.data import CATEGORIES, DATA_PATH, PRODUCT_RIDERS, classified_view, read_dataset
from customer_ml.model import NOTICE, NO_DATA, CustomerPredictor
from customer_ml.train import evaluate
import customer_ui as ui


@st.cache_resource(max_entries=4)
def get_resources(contents: bytes, filename: str):
    """Cache models with exact-profile availability checks and diagnostic metrics."""
    dataset = read_dataset(contents, filename)
    report, _ = evaluate(dataset)
    return dataset, CustomerPredictor(dataset.data), report


def load_example(customer):
    for field, value in customer.items():
        st.session_state[f"customer_{field}"] = value


def recommendations(predictor):
    with st.container(key="workspace"):
        controls, output = st.columns([1, 1.18], gap="large")
    with controls, st.container(key="customer_controls"):
        ui.section_heading("01", "고객 조건", "작은 차이에서 시작되는 가입 패턴")
        example = predictor.data.loc[predictor.data["가입상품"] == "수호천사 행복 연금보험"].iloc[0]
        st.button("학습 고객 예시 불러오기 ↗", on_click=load_example,
                  args=(example.drop(["가입상품", "가입특약"]).to_dict(),), width="stretch",
                  help="현재 학습 자료에 있는 고객 1명의 조건을 불러옵니다.")
        ui.field_group("A", "기본 정보")
        left, right = st.columns(2)
        with left:
            age = st.number_input("나이", min_value=20, max_value=65, value=40, step=1, key="customer_나이")
        with right:
            sex = st.radio("성별", CATEGORIES["성별"], horizontal=True, key="customer_성별")
        left, right = st.columns(2)
        with left:
            income = st.number_input("연소득 (만원)", min_value=3000, max_value=12000, value=5000, step=500, key="customer_연소득_만원")
        with right:
            risk = st.selectbox("직업위험등급", [1, 2, 3], key="customer_직업위험등급")
        ui.field_group("B", "가족 구성")
        left, right = st.columns(2)
        with left:
            married = st.radio("결혼 여부", CATEGORIES["결혼여부"], index=1, horizontal=True, key="customer_결혼여부")
        with right:
            children = st.selectbox("자녀 수", [0, 1, 2, 3], index=1, key="customer_자녀수")
        ui.field_group("C", "생활 · 건강")
        smoking = st.radio("흡연 여부", CATEGORIES["흡연여부"], horizontal=True, key="customer_흡연여부")
        left, right = st.columns(2)
        with left:
            chronic = st.selectbox("만성질환", CATEGORIES["만성질환"], key="customer_만성질환")
        with right:
            family = st.selectbox("가족력", CATEGORIES["가족력"], key="customer_가족력")
        st.caption("조건을 바꾸면 결과가 자동으로 반영됩니다.")
    customer = {"나이": age, "성별": sex, "연소득_만원": income, "직업위험등급": risk,
                "결혼여부": married, "자녀수": children, "흡연여부": smoking,
                "만성질환": chronic, "가족력": family}
    result = predictor.recommend(customer)
    with output, st.container(key="recommendation_output"):
        ui.section_heading("02", "상품·특약 예측", "학습 자료에서 확인한 고객 조건의 연결")
        if not result["available"]:
            ui.empty_result()
            return
        st.metric("추천 가입상품", result["product"])
        st.metric("추천 특약", result["rider"] if result["rider_available"] else NO_DATA)
        st.caption(f"상품 모델 점수 {result['product_score']:.1%}")
        if result["rider_available"]:
            st.caption(f"추천 상품 내 특약 모델 점수 {result['rider_score']:.1%}")
        else:
            st.caption("추천 상품에서 입력 조건과 일치하는 특약 학습 자료가 없습니다.")
        if result["rider"] == "특약 없음":
            st.info("이 예시 데이터의 연금상품은 특약 없음으로 생성되어 있습니다.")
        elif result["rider"] == "선택 안함":
            st.info("모델은 이 조건에서 특약을 선택하지 않은 가입 패턴을 예측했습니다.")
        ui.field_group("RANKING", "상품별 모델 점수")
        ui.ranking_bars(result["product_ranking"])
        st.caption(NOTICE)
        product = st.selectbox("특약을 확인할 상품", ["추천 상품", *PRODUCT_RIDERS])
        rider_result = result if product == "추천 상품" else predictor.recommend(customer, product=product)
        st.caption(f"특약 비교 대상: {rider_result['rider_product']}")
        if not rider_result["rider_available"]:
            st.info(NO_DATA)
            return
        riders = pd.DataFrame(rider_result["rider_ranking"]).rename(columns={"label": "가입특약", "score": "모델 점수"})
        st.dataframe(riders, hide_index=True, width="stretch",
                     column_config={"모델 점수": st.column_config.NumberColumn(format="percent")})


def data_explorer(dataset):
    view = classified_view(dataset.data)
    ui.section_heading("03", "고객 분류·정렬", "조건을 조합해 필요한 고객 그룹을 찾아보세요.")
    with st.container(key="filter_panel"):
        cols = st.columns(3)
        fields = [("연령대", ["20대", "30대", "40대", "50대", "60대"]),
                  ("흡연여부", CATEGORIES["흡연여부"]), ("결혼여부", CATEGORIES["결혼여부"]),
                  ("자녀구간", ["0명", "1명", "2명", "3명 이상"]),
                  ("가입상품", list(PRODUCT_RIDERS)), ("특약상태", ["가입", "선택 안함", "특약 없음"])]
        for i, (field, options) in enumerate(fields):
            with cols[i % 3]:
                selected = st.multiselect(field, options, key=f"filter_{field}")
            if selected:
                view = view.loc[view[field].isin(selected)]
        riders_col, sort_col = st.columns([2, 1])
        with riders_col:
            chosen_riders = st.multiselect("가입특약", sorted(dataset.data["가입특약"].unique()), key="filter_rider")
        if chosen_riders:
            view = view.loc[view["가입특약"].isin(chosen_riders)]
        with sort_col:
            sort = st.selectbox("정렬 기준", ["나이", "흡연여부", "결혼여부", "자녀수", "가입상품", "가입특약"])
        st.caption("필터는 고객 조회와 다운로드에만 적용됩니다.")
    view = view.sort_values([sort] + ([] if sort == "나이" else ["나이"]), kind="stable")
    ui.selection_summary(len(view), len(dataset.data))
    st.caption(f"선택한 고객 {len(view):,}명 / 학습 고객 {len(dataset.data):,}명 · 60대는 60~65세")
    if view.empty:
        st.info(NO_DATA)
        return
    st.dataframe(view, hide_index=True, width="stretch")
    st.download_button("선택한 고객 CSV 다운로드", view.to_csv(index=False).encode("utf-8-sig"),
                       "filtered_customers.csv", "text/csv")


def main():
    st.set_page_config(page_title="동양생명 · Data Lab", page_icon="◈", layout="wide")
    ui.apply_theme()
    with st.sidebar:
        ui.sidebar_brand()
        st.subheader("학습 데이터")
        source = st.radio("자료 선택", ["기본 고객 더미데이터", "고객 CSV 또는 정렬 ZIP 업로드"])
        upload = st.file_uploader("customer_data.csv 또는 항목별정렬_CSV.zip", type=["csv", "zip"]) if source != "기본 고객 더미데이터" else None
    try:
        if source == "기본 고객 더미데이터":
            contents, filename = DATA_PATH.read_bytes(), DATA_PATH.name
        elif upload is None:
            st.info("학습할 고객 CSV 또는 정렬 ZIP 파일을 선택하세요.")
            st.stop()
        else:
            contents, filename = upload.getvalue(), upload.name
        with st.spinner("고객 더미데이터로 모델을 준비하고 있습니다."):
            dataset, predictor, report = get_resources(contents, filename)
    except (ValueError, OSError) as exc:
        st.error(f"학습 자료를 확인해 주세요: {exc}")
        st.stop()
    with st.sidebar:
        st.write(f"학습 고객 **{len(dataset.data):,}명**")
        st.caption(filename)
        if dataset.sorted_views > 1:
            st.caption(f"정렬 파일 {dataset.sorted_views}개가 동일한 표본임을 확인했습니다. 1개 표본만 학습합니다.")
        st.html('''<div class="sidebar-note"><strong>ABOUT THIS WORKSPACE</strong>
            고객 더미데이터로 상품과 특약의 가입 패턴을 탐색합니다.<br><br>
            학습 자료에 보험료는 포함되어 있지 않습니다. 실제 가입 견적이나 상품 적합성 판단에는 사용할 수 없습니다.</div>''')
    ui.hero(dataset.data)
    predict_tab, data_tab, validation_tab = st.tabs(["상품·특약 추천", "고객 분류·정렬", "모델 검증"])
    with predict_tab:
        recommendations(predictor)
    with data_tab:
        data_explorer(dataset)
    with validation_tab:
        ui.section_heading("04", "모델 검증", "결과를 이해하는 기준, 데이터와 검증 과정")
        split = report["split"]
        ui.split_summary(split["training_rows"], split["test_rows"])
        st.write(f"동일한 고객 조건은 같은 그룹으로 묶어 학습 {split['training_rows']:,}명 / 검증 {split['test_rows']:,}명으로 분리했습니다.")
        st.caption("아래는 ‘자료가 없음’ 제한을 적용하기 전 분류기 자체의 진단 점수입니다. 화면 예측 모델은 전체 고객으로 다시 학습했습니다.")
        st.write("실제 추천 화면은 고객 조건 9개가 모두 일치해야 결과를 표시하며, 특약은 선택한 상품 안에서도 같은 조건의 자료가 있어야 합니다.")
        st.caption(f"이 검증 세트는 새로운 조건만 분리했으므로, 학습 자료 일치율은 {report['serving_policy']['holdout_profile_coverage']:.1%}입니다. 실제 표시 규칙에서는 모두 ‘자료가 없음’에 해당합니다.")
        metrics = report["metrics"]
        a, b, c = st.columns(3)
        a.metric("상품 예측 정확도", f"{metrics['product_accuracy']:.1%}")
        b.metric("상품·특약 동시 정확도", f"{metrics['product_and_rider_accuracy']:.1%}")
        c.metric("최빈 상품 기준선", f"{report['majority_baseline']['product_accuracy']:.1%}")
        st.write(f"실제 정답 상품을 알고 있을 때의 특약 정확도: {metrics['rider_accuracy_given_true_product']:.1%}")
        st.caption("위 특약 정확도는 상품 예측 오류를 포함하지 않습니다. 전체 추천 성능은 상품·특약 동시 정확도로 확인하세요.")
        st.write(f"원본 중복 {dataset.metadata['exact_duplicate_rows']}행은 보존했습니다. 학습·검증 간 동일 조건 그룹 중복은 {split['overlapping_groups']}개입니다.")
        st.write("학습에는 고객 기본 항목 9개만 사용합니다. 분석ID, 가입상품·가입특약 및 이를 가공한 분류값은 입력에서 제외합니다.")
        st.warning("생성 규칙이 포함된 더미데이터의 검증 결과입니다. 실제 고객의 가입 선택을 예측하는 정확도를 의미하지 않습니다.")
        st.download_button("현재 학습 원본 CSV 다운로드", dataset.data.to_csv(index=False).encode("utf-8-sig"),
                           "training_customers.csv", "text/csv")
    ui.footer()


if __name__ == "__main__":
    main()
