"""Check the deployed chat handler without booting its Gemini dependencies."""

import ast
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def handler_for(result):
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "process_recommendation")
    state = State(messages=[], last_recommended_product="previous",
                  last_recommended_rider="previous", last_customer_info={})
    st = MagicMock(session_state=state)
    prompt = MagicMock()
    namespace = {"st": st, "get_recommendation": lambda _: result,
                 "PromptTemplate": prompt, "StrOutputParser": MagicMock(),
                 "format_docs": lambda docs: ""}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), "app.py", "exec"), namespace)
    return namespace["process_recommendation"], st, state, prompt


def test_unseen_valid_customer_keeps_model_result_even_with_preference():
    result = {"자료있음": True, "특약자료있음": True,
              "주계약": "수호천사 암/건강보험", "주계약_확률": 54.2,
              "추천특약": "수술비 보장 특약", "특약_확률": 73.1}
    handler, st, state, prompt = handler_for(result)
    handler({"선호상품": "암보험"}, MagicMock(), MagicMock())
    assert state.last_recommended_product == result["주계약"]
    assert state.last_recommended_rider == result["추천특약"]
    assert "수술비 보장 특약" in state.messages[0]["content"]
    assert "표적항암약물허가치료 특약" not in state.messages[0]["content"]
    prompt.from_template.assert_called_once()


def test_invalid_customer_stops_before_llm_and_clears_old_result():
    handler, st, state, prompt = handler_for({"자료있음": False, "특약자료있음": False})
    handler({"나이": 66}, MagicMock(), MagicMock())
    st.info.assert_called_once_with("자료가 없음")
    assert state.messages[-1]["content"] == "자료가 없음"
    assert "last_recommended_product" not in state
    assert "last_recommended_rider" not in state
    assert "last_customer_info" not in state
    prompt.from_template.assert_not_called()
