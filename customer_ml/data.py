"""Read one customer population from the original CSV or its sorted ZIP views."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "customer_data.csv"
FEATURES = ["나이", "성별", "연소득_만원", "직업위험등급", "결혼여부", "자녀수", "흡연여부", "만성질환", "가족력"]
TARGETS = ["가입상품", "가입특약"]
COLUMNS = FEATURES + TARGETS
CATEGORIES = {
    "성별": ["남성", "여성"], "결혼여부": ["미혼", "기혼"], "흡연여부": ["비흡연", "흡연"],
    "만성질환": ["없음", "고혈압", "당뇨"], "가족력": ["없음", "암", "심혈관"],
}
NUMERIC = [c for c in FEATURES if c not in CATEGORIES]
PRODUCT_RIDERS = {
    "수호천사 암/건강보험": ["수술비 보장 특약", "표적항암약물허가치료 특약", "선택 안함"],
    "수호천사 간편심사(유병자)보험": ["중증질환 산정특례 보장 특약", "선택 안함"],
    "수호천사 우리가족 종신보험": ["가족 수입 보장 특약", "선택 안함"],
    "수호천사 행복 연금보험": ["특약 없음"],
}
SORTED_FILES = ["01_연령대별_정렬.csv", "02_흡연여부별_정렬.csv", "03_결혼여부별_정렬.csv",
                "04_자녀수별_정렬.csv", "05_가입상품별_정렬.csv", "06_특약별_정렬.csv"]
MAX_BYTES = 50 * 1024 * 1024


@dataclass
class CustomerDataset:
    data: pd.DataFrame
    source_name: str
    source_sha256: str
    sorted_views: int = 1

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.data.to_csv(index=False).encode("utf-8")).hexdigest()

    @property
    def metadata(self) -> dict:
        return {"source": self.source_name, "source_sha256": self.source_sha256,
                "canonical_sha256": self.fingerprint, "rows": len(self.data),
                "sorted_views": self.sorted_views,
                "exact_duplicate_rows": int(self.data.duplicated().sum()),
                "feature_groups": int(self.data[FEATURES].drop_duplicates().shape[0]),
                "synthetic": True}


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"고객 입력 항목이 없습니다: {', '.join(sorted(missing))}")
    result = frame.loc[:, FEATURES].copy()
    if result.empty or result.isna().any().any():
        raise ValueError("고객 입력값은 비어 있을 수 없습니다.")
    for col, allowed in CATEGORIES.items():
        result[col] = result[col].astype(str).str.strip()
        if not result[col].isin(allowed).all():
            raise ValueError(f"{col}: {', '.join(allowed)} 중에서 선택하세요.")
    for col in NUMERIC:
        if result[col].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise ValueError(f"{col}: 정수를 입력하세요.")
        values = pd.to_numeric(result[col], errors="coerce")
        if not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f"{col}: 유한한 정수를 입력하세요.")
        result[col] = values.astype(int)
    for col, lower, upper in [("나이", 20, 65), ("연소득_만원", 3000, 12000),
                              ("직업위험등급", 1, 3), ("자녀수", 0, 3)]:
        if not result[col].between(lower, upper).all():
            raise ValueError(f"{col}: 이 더미데이터의 지원 범위는 {lower}~{upper}입니다.")
    return result


def validate_training(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) > 50_000:
        raise ValueError("최대 50,000행까지 학습할 수 있습니다.")
    if frame.columns.duplicated().any():
        raise ValueError("중복된 열 이름이 있습니다.")
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"학습 CSV에 필요한 항목이 없습니다: {', '.join(sorted(missing))}")
    result = validate_features(frame)
    for col in TARGETS:
        if frame[col].isna().any():
            raise ValueError(f"{col}: 정답이 비어 있습니다.")
        result[col] = frame[col].astype(str).str.strip()
    if not result["가입상품"].isin(PRODUCT_RIDERS).all():
        raise ValueError("예시 데이터에서 지원하지 않는 가입상품이 있습니다.")
    for product, riders in PRODUCT_RIDERS.items():
        if not result.loc[result["가입상품"] == product, "가입특약"].isin(riders).all():
            raise ValueError(f"{product}: 상품에 맞지 않는 특약이 있습니다.")
    # Only the nine raw features and two labels survive. IDs and label-derived
    # 상품분류/특약상태/특약가입여부 must never become model inputs.
    return result[COLUMNS].sort_values(COLUMNS, kind="stable").reset_index(drop=True)


def _read_csv(contents: bytes) -> pd.DataFrame:
    try:
        return validate_training(pd.read_csv(BytesIO(contents), encoding="utf-8-sig"))
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("UTF-8 형식의 고객 CSV 파일을 사용하세요.") from exc


def read_dataset(contents: bytes, filename: str) -> CustomerDataset:
    if not contents or len(contents) > MAX_BYTES:
        raise ValueError("비어 있지 않은 50MB 이하 CSV 또는 ZIP 파일을 사용하세요.")
    suffix = Path(filename).suffix.lower()
    views = 1
    if suffix == ".csv":
        data = _read_csv(contents)
    elif suffix == ".zip":
        try:
            with zipfile.ZipFile(BytesIO(contents)) as archive:
                entries = archive.infolist()
                if sum(entry.file_size for entry in entries) > MAX_BYTES:
                    raise ValueError("압축 해제 기준 50MB 이하 ZIP 파일을 사용하세요.")
                by_name = {}
                for entry in entries:
                    name = Path(entry.filename.replace('\\', '/')).name
                    if name in SORTED_FILES:
                        if name in by_name:
                            raise ValueError("ZIP 안에 동일한 정렬 파일명이 반복됩니다.")
                        by_name[name] = entry
                if set(by_name) != set(SORTED_FILES):
                    raise ValueError("제공된 6개 정렬 CSV가 모두 포함된 ZIP 파일을 사용하세요.")
                frames = [_read_csv(archive.read(by_name[name])) for name in SORTED_FILES]
                data = frames[0]
                if any(not data.equals(other) for other in frames[1:]):
                    raise ValueError("정렬 CSV들의 고객 데이터가 서로 다릅니다. 같은 원본의 정렬 자료가 필요합니다.")
                views = len(frames)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise ValueError("정렬 ZIP 파일을 읽을 수 없습니다.") from exc
    else:
        raise ValueError("고객 CSV 또는 정렬 ZIP 파일을 선택하세요.")
    if set(data["가입상품"]) != set(PRODUCT_RIDERS):
        raise ValueError("학습 자료에는 예시 가입상품 4종이 모두 있어야 합니다.")
    return CustomerDataset(data, Path(filename).name, hashlib.sha256(contents).hexdigest(), views)


def load_dataset(path: str | Path = DATA_PATH) -> CustomerDataset:
    path = Path(path)
    return read_dataset(path.read_bytes(), path.name)


def classified_view(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["연령대"] = (result["나이"] // 10 * 10).astype(str) + "대"
    result["자녀구간"] = result["자녀수"].map({0: "0명", 1: "1명", 2: "2명", 3: "3명 이상"})
    result["특약상태"] = result["가입특약"].where(result["가입특약"].isin(["선택 안함", "특약 없음"]), "가입")
    return result
