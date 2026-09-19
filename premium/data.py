"""Schema and validation for one version of a public premium example table."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "premium_examples.csv"
SOURCE_PATH = ROOT / "data" / "premium_source.json"
SOURCE_ID = "tongyang_angel_ansim_202309_hana"
PRODUCT = "무배당엔젤안심보험"
VERSION = "2023-09"
PLAN_COLUMNS = ["sex", "coverage_type", "refund_type", "payment_years"]
FEATURE_COLUMNS = ["age", *PLAN_COLUMNS]
KEY_COLUMNS = ["product_name", "product_version", *FEATURE_COLUMNS,
               "sum_assured_krw", "coverage_end_age", "payment_frequency"]
TARGET = "monthly_premium_krw"
TERMS = {"치매보장형": [5, 7, 20], "생활비보장형(장해)": [7, 10, 15]}
REFUNDS = ["표준형", "해약환급금일부지급형"]
REQUIRED = ["record_id", *KEY_COLUMNS, TARGET, "source_id", "source_pdf_page"]


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on corrupted, duplicate, unsupported or incomplete examples."""
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    data = frame.copy()
    if data[REQUIRED].isna().any().any():
        raise ValueError("학습 데이터에 결측값이 있습니다.")
    for column in ["age", "payment_years", "sum_assured_krw", "coverage_end_age",
                   TARGET, "source_pdf_page"]:
        values = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError(f"양의 정수가 필요한 열: {column}")
        data[column] = values.astype(int)
    if data.duplicated(KEY_COLUMNS).any() or data["record_id"].duplicated().any():
        raise ValueError("동일 가입 조건 또는 record_id가 중복됩니다.")
    fixed = {"product_name": PRODUCT, "product_version": VERSION,
             "sum_assured_krw": 10_000_000, "coverage_end_age": 110,
             "payment_frequency": "monthly", "source_id": SOURCE_ID,
             "source_pdf_page": 3}
    for column, value in fixed.items():
        if not data[column].eq(value).all():
            raise ValueError(f"현재 지원하지 않는 데이터 조건: {column}")
    expected = {(age, sex, coverage, refund, term)
                for age in [30, 40, 50] for sex in ["남자", "여자"]
                for coverage, terms in TERMS.items() for refund in REFUNDS
                for term in terms}
    actual = set(data[FEATURE_COLUMNS].itertuples(index=False, name=None))
    if actual != expected or len(data) != 72:
        raise ValueError("원문 표의 72개 가입 조건과 일치하지 않습니다.")
    return data


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    if Path(path).resolve() == DATA_PATH.resolve():
        metadata = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
            raise ValueError("CSV가 검증한 스냅샷과 다릅니다. 원문에서 다시 추출하세요.")
    return validate_data(pd.read_csv(path, dtype={"product_version": str}))
