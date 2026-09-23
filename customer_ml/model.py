"""A product classifier followed by a classifier for that product's riders."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORIES, FEATURES, NUMERIC, load_dataset, validate_features, validate_training

NOTICE = "고객 더미데이터의 가입 패턴을 학습한 예시입니다. 모델 점수는 실제 가입 확률이나 보장 적합성이 아닙니다."
NO_DATA = "자료가 없음"
LEGACY_MODEL_PARAMS = {"n_estimators": 160, "max_depth": 9, "min_samples_leaf": 4}
DEFAULT_MODEL_PARAMS = dict(LEGACY_MODEL_PARAMS)


def make_classifier(params: dict | None = None) -> Pipeline:
    params = DEFAULT_MODEL_PARAMS if params is None else params
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", "passthrough", NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
        ])),
        ("classifier", RandomForestClassifier(**params, random_state=42, n_jobs=1)),
    ])


class CustomerPredictor:
    def __init__(self, data: pd.DataFrame | None = None, model_params: dict | None = None):
        self.data = load_dataset().data if data is None else validate_training(data)
        self.model_params = dict(DEFAULT_MODEL_PARAMS if model_params is None else model_params)
        self.main_model = make_classifier(self.model_params).fit(self.data[FEATURES], self.data["가입상품"])
        self.rider_models = {
            product: make_classifier(self.model_params).fit(group[FEATURES], group["가입특약"])
            for product, group in self.data.groupby("가입상품", sort=True)
        }
        self.profiles = set(self.data[FEATURES].itertuples(index=False, name=None))
        self.rider_profiles = {
            product: set(group[FEATURES].itertuples(index=False, name=None))
            for product, group in self.data.groupby("가입상품", sort=True)
        }

    def _no_data(self) -> dict:
        return {"available": False, "rider_available": False, "message": NO_DATA,
                "product": None, "product_score": None, "rider_product": None,
                "rider": None, "rider_score": None, "product_ranking": [], "rider_ranking": [],
                "training_rows": len(self.data), "synthetic": True, "notice": NOTICE,
                "used_fallback": False}

    @staticmethod
    def _ranking(model, inputs) -> list[dict]:
        scores = model.predict_proba(inputs)[0]
        rows = [{"label": str(label), "score": float(score)} for label, score in zip(model.classes_, scores)]
        return sorted(rows, key=lambda row: (-row["score"], row["label"]))

    def recommend(self, customer: dict, product: str | None = None) -> dict:
        try:
            inputs = validate_features(pd.DataFrame([customer]))
        except ValueError:
            return self._no_data()
        profile = tuple(inputs.iloc[0][FEATURES])
        if product is not None and product not in self.rider_models:
            return self._no_data()
        products = self._ranking(self.main_model, inputs)
        selected = product if product is not None else products[0]["label"]
        riders = self._ranking(self.rider_models[selected], inputs)
        return {"available": True, "rider_available": True, "message": "",
                "product": products[0]["label"], "product_score": products[0]["score"],
                "rider_product": selected, "rider": riders[0]["label"],
                "rider_score": riders[0]["score"],
                "product_ranking": products, "rider_ranking": riders,
                "training_rows": len(self.data), "synthetic": True, "notice": NOTICE,
                "used_fallback": profile not in self.rider_profiles[selected]}

    def predict_batch(self, frame: pd.DataFrame, products=None) -> pd.DataFrame:
        """Apply the same validation and fallback policy as interactive recommendations."""
        selected = [None] * len(frame) if products is None else list(products)
        if len(selected) != len(frame):
            raise ValueError("상품 수와 고객 수가 다릅니다.")
        results = [self.recommend(row.to_dict(), product) for (_, row), product in zip(frame.iterrows(), selected)]
        return pd.DataFrame([{
            "product": result["rider_product"] if products is not None else result["product"],
            "rider": result["rider"], "available": result["available"],
            "rider_available": result["rider_available"], "message": result["message"],
        } for result in results], columns=["product", "rider", "available", "rider_available", "message"])

    def _predict_for_evaluation(self, frame: pd.DataFrame, products=None) -> pd.DataFrame:
        """Vectorized offline classifier diagnostics."""
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
