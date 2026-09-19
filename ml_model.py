"""Compatibility adapter used by the existing sales assistant."""

from functools import lru_cache

from customer_ml.data import DATA_PATH, read_dataset
from customer_ml.model import CustomerPredictor


@lru_cache(maxsize=1)
def _predictor(contents: bytes) -> CustomerPredictor:
    return CustomerPredictor(read_dataset(contents, DATA_PATH.name).data)


def get_recommendation(customer_info):
    """Keep the legacy response keys; scores are synthetic-model outputs."""
    result = _predictor(DATA_PATH.read_bytes()).recommend(customer_info)
    return {"주계약": result["product"], "주계약_확률": round(result["product_score"] * 100, 1),
            "추천특약": result["rider"], "특약_확률": round(result["rider_score"] * 100, 1),
            "데이터구분": "고객 더미데이터", "학습고객수": result["training_rows"],
            "안내": result["notice"]}
