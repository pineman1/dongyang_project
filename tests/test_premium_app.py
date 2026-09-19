from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_page_runs_without_llm_key_and_updates_conditions(monkeypatch):
    for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    app = AppTest.from_file(str(ROOT / "premium_reference_app.py")).run(timeout=30)
    assert not app.exception
    assert app.metric[1].value == "265,600원 / 월"
    app.number_input[0].set_value(35).run()
    assert not app.exception
    assert app.metric[1].value == "미공개"
    app.selectbox[0].select("생활비보장형(장해)").run()
    assert not app.exception
    assert app.selectbox[1].options == ["7년납", "10년납", "15년납"]
    app.number_input[0].set_value(50).run()
    app.radio[0].set_value("여자").run()
    app.selectbox[2].select("해약환급금일부지급형").run()
    assert not app.exception
    assert app.metric[1].value == "497,100원 / 월"
