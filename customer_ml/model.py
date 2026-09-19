"""A product classifier followed by a classifier for that product's riders."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORIES, FEATURES, NUMERIC, load_dataset, validate_features, validate_training

NOTICE = "고객 더미데이터의 가입 패턴을 학습한 예시입니다. 모델 점수는 실제 가입 확률이나 보장 적합성이 아닙니다."


def make_classifier() -> Pipeline:
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", "passthrough", NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
        ])),
        ("classifier", RandomForestClassifier(n_estimators=160, max_depth=9,
                                              min_samples_leaf=4, random_state=42, n_jobs=1)),
    ])


class CustomerPredictor:
    def __init__(self, data: pd.DataFrame | None = None):
        self.data = load_dataset().data if data is None else validate_training(data)
        self.main_model = make_classifier().fit(self.data[FEATURES], self.data["가입상품"])
        self.rider_models = {
            product: make_classifier().fit(group[FEATURES], group["가입특약"])
            for product, group in self.data.groupby("가입상품", sort=True)
        }

    @staticmethod
    def _ranking(model, inputs) -> list[dict]:
        scores = model.predict_proba(inputs)[0]
        rows = [{"label": str(label), "score": float(score)} for label, score in zip(model.classes_, scores)]
        return sorted(rows, key=lambda row: (-row["score"], row["label"]))

    def recommend(self, customer: dict, product: str | None = None) -> dict:
        inputs = validate_features(pd.DataFrame([customer]))
        products = self._ranking(self.main_model, inputs)
        selected = product if product is not None else products[0]["label"]
        if selected not in self.rider_models:
            raise ValueError("특약을 확인할 상품이 학습 데이터에 없습니다.")
        riders = self._ranking(self.rider_models[selected], inputs)
        return {"product": products[0]["label"], "product_score": products[0]["score"],
                "rider_product": selected, "rider": riders[0]["label"], "rider_score": riders[0]["score"],
                "product_ranking": products, "rider_ranking": riders,
                "training_rows": len(self.data), "synthetic": True, "notice": NOTICE}

    def predict_batch(self, frame: pd.DataFrame, products=None) -> pd.DataFrame:
        inputs = validate_features(frame).reset_index(drop=True)
        chosen = self.main_model.predict(inputs) if products is None else list(products)
        if len(chosen) != len(inputs):
            raise ValueError("상품 수와 고객 수가 다릅니다.")
        result = pd.DataFrame({"product": chosen})
        result["rider"] = ""
        for product, rows in result.groupby("product").groups.items():
            if product not in self.rider_models:
                raise ValueError(f"학습하지 않은 상품입니다: {product}")
            result.loc[rows, "rider"] = self.rider_models[product].predict(inputs.loc[rows])
        return result
