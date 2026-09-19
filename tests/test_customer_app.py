from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from customer_ml.data import FEATURES, PRODUCT_RIDERS, load_dataset
from customer_ml.model import NO_DATA

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["customer_app.py", "premium_app.py", "pages/1_Customer_Recommendation.py"])
def test_dummy_model_is_default_on_every_entrypoint(monkeypatch, entry):
    for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    app = AppTest.from_file(str(ROOT / entry)).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "고객 조건별 상품·특약 추천"
    assert any(element.value == NO_DATA for element in app.info)
    assert not any(element.label in ["추천 가입상품", "추천 특약"] for element in app.metric)
    assert any("1,000" in element.value for element in app.sidebar.markdown)
    assert not any("2023" in element.value for element in app.warning)


def set_known_customer(app):
    data = load_dataset().data
    customer = data.loc[data["가입상품"] == "수호천사 행복 연금보험"].iloc[0]
    for label, field in [("나이", "나이"), ("연소득 (만원)", "연소득_만원")]:
        next(element for element in app.number_input if element.label == label).set_value(int(customer[field]))
    for label, field in [("성별", "성별"), ("결혼 여부", "결혼여부"), ("흡연 여부", "흡연여부")]:
        next(element for element in app.radio if element.label == label).set_value(customer[field])
    for label, field in [("직업위험등급", "직업위험등급"), ("자녀 수", "자녀수"), ("만성질환", "만성질환"), ("가족력", "가족력")]:
        value = int(customer[field]) if field in ["직업위험등급", "자녀수"] else customer[field]
        next(element for element in app.selectbox if element.label == label).select(value)
    app.run()
    assert not app.exception


def test_conditions_riders_and_filters_update():
    app = AppTest.from_file(str(ROOT / "customer_app.py")).run(timeout=60)
    set_known_customer(app)
    next(element for element in app.selectbox if element.label == "특약을 확인할 상품").select("수호천사 행복 연금보험").run()
    assert not app.exception
    assert app.dataframe[0].value["가입특약"].tolist() == ["특약 없음"]
    app.multiselect(key="filter_연령대").set_value(["40대"]).run()
    app.multiselect(key="filter_결혼여부").set_value(["기혼"]).run()
    frame = next(element.value for element in app.dataframe if "나이" in element.value.columns)
    assert len(frame) == 189
    assert set(frame["연령대"]) == {"40대"} and set(frame["결혼여부"]) == {"기혼"}
    app.multiselect(key="filter_자녀구간").set_value(["3명 이상"]).run()
    assert not app.exception
    app.multiselect(key="filter_결혼여부").set_value(["미혼"]).run()
    assert not app.exception
    assert any(element.value == NO_DATA for element in app.info)
    assert not any("나이" in element.value.columns for element in app.dataframe)


def test_known_to_unknown_to_known_clears_and_restores_predictions():
    app = AppTest.from_file(str(ROOT / "customer_app.py")).run(timeout=60)
    set_known_customer(app)
    assert any(element.label == "추천 가입상품" for element in app.metric)
    next(element for element in app.number_input if element.label == "연소득 (만원)").set_value(3500).run()
    assert not app.exception
    assert any(element.value == NO_DATA for element in app.info)
    assert not any(element.label in ["추천 가입상품", "추천 특약"] for element in app.metric)
    assert not any(element.label == "특약을 확인할 상품" for element in app.selectbox)
    set_known_customer(app)
    assert any(element.label == "추천 가입상품" for element in app.metric)


def test_missing_rider_does_not_show_a_rider_score():
    app = AppTest.from_file(str(ROOT / "customer_app.py")).run(timeout=60)
    set_known_customer(app)
    next(element for element in app.selectbox if element.label == "특약을 확인할 상품").select("수호천사 간편심사(유병자)보험").run()
    assert not app.exception
    assert any(element.value == NO_DATA for element in app.info)
    assert not any("가입특약" in element.value.columns and "모델 점수" in element.value.columns for element in app.dataframe)


@pytest.mark.parametrize("available,rider_available", [(False, False), (True, False)])
def test_legacy_sales_flow_stops_before_llm_and_clears_old_design(available, rider_available):
    # Run just the recommendation handler, without booting the API-key-based app.
    import ast
    from unittest.mock import MagicMock
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_recommendation")
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    state = State(messages=[], show_tuning=True, last_recommended_product="old", last_recommended_rider="old", last_customer_info={})
    st = MagicMock(session_state=state)
    namespace = {"st": st, "get_recommendation": lambda _: {"자료있음": available, "특약자료있음": rider_available}}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), "app.py", "exec"), namespace)
    namespace["process_recommendation"]({"선호상품": "암보험"}, object(), object())
    st.info.assert_called_once_with(NO_DATA)
    assert state.messages[-1]["content"] == NO_DATA
    assert not state.show_tuning
    assert all(key not in state for key in ["last_recommended_product", "last_recommended_rider", "last_customer_info"])
