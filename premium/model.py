"""Small-table regression: log premium ~ age + categorical tariff profile.

No generated targets, coverage scaling, or health surcharges are used.
The model is fixed before evaluation; holdout results do not select its settings.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from premium.data import (FEATURE_COLUMNS, PLAN_COLUMNS, PRODUCT, REFUNDS,
                          TARGET, TERMS, VERSION, load_data, validate_data)

MODEL_NAME = "log_linear_tariff_regression"
NOTICE = ("2023년 9월 공개 예시를 학습한 연구용 추정값입니다. "
          "현재 가입 견적 또는 보험사의 확정 보험료가 아닙니다.")


def design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    result["age_decades"] = (data["age"].astype(float) - 40) / 10
    # Full combination preserves interactions among coverage, sex, term and refund.
    result["tariff_profile"] = data[PLAN_COLUMNS].astype(str).agg("|".join, axis=1)
    return result


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("age", "passthrough", ["age_decades"]),
        ("profile", OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False),
         ["tariff_profile"]),
    ])


def make_model() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=Pipeline([("features", preprocessor()), ("regression", LinearRegression())]),
        func=np.log, inverse_func=np.exp,
    )


@dataclass(frozen=True)
class PremiumRequest:
    age: int
    sex: str
    coverage_type: str
    refund_type: str
    payment_years: int
    sum_assured_krw: int = 10_000_000
    coverage_end_age: int = 110

    def validate(self):
        for field in ["age", "payment_years", "sum_assured_krw", "coverage_end_age"]:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{field}: 정수를 입력하세요.")
        if not 30 <= self.age <= 50:
            raise ValueError("예측 가능한 보험나이는 30~50세입니다. 범위 밖으로 외삽하지 않습니다.")
        if self.sex not in ["남자", "여자"]:
            raise ValueError("성별은 원문 표 기준 남자/여자를 선택하세요.")
        if self.coverage_type not in TERMS:
            raise ValueError("지원하지 않는 보장 유형입니다.")
        if self.refund_type not in REFUNDS:
            raise ValueError("지원하지 않는 환급 유형입니다.")
        if self.payment_years not in TERMS[self.coverage_type]:
            raise ValueError("해당 보장 유형의 공개 예시에 없는 납입기간입니다.")
        if self.sum_assured_krw != 10_000_000 or self.coverage_end_age != 110:
            raise ValueError("보험가입금액 1,000만원, 110세 만기만 지원합니다.")


class PremiumPredictor:
    def __init__(self, data: pd.DataFrame | None = None):
        self.data = load_data() if data is None else validate_data(data)
        self.model = make_model()
        self.model.fit(design_matrix(self.data), self.data[TARGET])

    def predict(self, request: PremiumRequest) -> dict:
        request.validate()
        row = {field: getattr(request, field) for field in FEATURE_COLUMNS}
        prediction = float(self.model.predict(design_matrix(pd.DataFrame([row])))[0])
        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError("유효한 보험료를 예측하지 못했습니다.")
        observed = self.data.loc[(self.data[FEATURE_COLUMNS] == pd.Series(row)).all(axis=1)]
        return {
            "monthly_premium_krw": int(round(prediction)),
            "published_example_krw": None if observed.empty else int(observed.iloc[0][TARGET]),
            "is_unobserved_age": observed.empty,
            "product_name": PRODUCT, "product_version": VERSION,
            "inputs": {**row, "sum_assured_krw": request.sum_assured_krw,
                       "coverage_end_age": request.coverage_end_age, "payment_frequency": "monthly"},
            "model": MODEL_NAME, "source_pdf_page": 3, "notice": NOTICE,
        }
