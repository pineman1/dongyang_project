"""Compatibility adapter used by the existing sales assistant."""

from functools import lru_cache

from customer_ml.data import DATA_PATH, read_dataset
from customer_ml.model import NO_DATA, CustomerPredictor

PREFERRED_PRODUCTS = {"암보험": "수호천사 암/건강보험", "연금보험": "수호천사 행복 연금보험",
                      "종신보험": "수호천사 우리가족 종신보험", "유병자보험": "수호천사 간편심사(유병자)보험"}


@lru_cache(maxsize=1)
def _predictor(contents: bytes) -> CustomerPredictor:
    return CustomerPredictor(read_dataset(contents, DATA_PATH.name).data)


def get_recommendation(customer_info):
    """Keep the legacy response keys; scores are synthetic-model outputs."""
    preferred = customer_info.get("선호상품")
    result = _predictor(DATA_PATH.read_bytes()).recommend(
        customer_info, product=PREFERRED_PRODUCTS.get(preferred, preferred))
    product = result["rider_product"]
    score = next((r["score"] for r in result["product_ranking"] if r["label"] == product), None)
    return {"자료있음": result["available"], "특약자료있음": result["rider_available"],
            "주계약": product if result["available"] else NO_DATA,
            "주계약_확률": round(score * 100, 1) if score is not None else None,
            "추천특약": result["rider"] if result["rider_available"] else NO_DATA,
            "특약_확률": round(result["rider_score"] * 100, 1) if result["rider_available"] else None,
            "데이터구분": "고객 더미데이터", "학습고객수": result["training_rows"],
            "안내": result["notice"]}
