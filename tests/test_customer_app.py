from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from customer_ml.data import PRODUCT_RIDERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ["customer_app.py", "premium_app.py", "pages/1_Customer_Recommendation.py"])
def test_dummy_model_is_default_on_every_entrypoint(monkeypatch, entry):
    for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    app = AppTest.from_file(str(ROOT / entry)).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "고객 조건별 상품·특약 추천"
    assert app.metric[0].value in PRODUCT_RIDERS
    assert app.metric[1].value in PRODUCT_RIDERS[app.metric[0].value]
    assert any("1,000" in element.value for element in app.sidebar.markdown)
    assert not any("2023" in element.value for element in app.warning)


def test_conditions_riders_and_filters_update():
    app = AppTest.from_file(str(ROOT / "customer_app.py")).run(timeout=60)
    app.number_input[0].set_value(65).run()
    app.selectbox[2].select("당뇨").run()
    assert not app.exception
    app.selectbox[4].select("수호천사 행복 연금보험").run()
    assert not app.exception
    assert app.dataframe[0].value["가입특약"].tolist() == ["특약 없음"]
    app.multiselect(key="filter_연령대").set_value(["40대"]).run()
    app.multiselect(key="filter_결혼여부").set_value(["기혼"]).run()
    frame = app.dataframe[1].value
    assert len(frame) == 189
    assert set(frame["연령대"]) == {"40대"} and set(frame["결혼여부"]) == {"기혼"}
    app.multiselect(key="filter_자녀구간").set_value(["3명 이상"]).run()
    assert not app.exception
    app.multiselect(key="filter_결혼여부").set_value(["미혼"]).run()
    assert not app.exception and app.dataframe[1].value.empty
