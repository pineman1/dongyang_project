"""Separate, explicitly synthetic suitability experiment; never overwrites signup labels."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORIES, DATA_PATH, FEATURES, PRODUCT_RIDERS, ROOT, load_dataset, validate_features

GOALS = ("암·건강", "유병자 보장", "가족 생활비", "노후 자금")
NO_SECONDARY_GOAL = "없음"
NO_FIT = "예산·보장 재검토"
PRODUCTS = tuple(PRODUCT_RIDERS)
# These are arbitrary evaluation assumptions in 10,000 KRW/month, NOT premiums.
MOCK_COST_BASE = (7, 9, 13, 11)
BUDGETS = (6, 8, 10, 12, 15, 20)
EXTRA = ("우선보장목표", "추가보장목표", "월예산_만원")
INPUTS = tuple(FEATURES) + EXTRA
TARGET = "목업_적합상품"
SEED = 42
ARTIFACT_VERSION = 2


def mock_costs(frame: pd.DataFrame) -> np.ndarray:
    """Return illustrative cost assumptions; never present them as quotes."""
    age_add = (frame["나이"].to_numpy() >= 50).astype(int)
    costs = np.tile(MOCK_COST_BASE, (len(frame), 1)).astype(int)
    costs += age_add[:, None]
    costs[:, 0] += (frame["흡연여부"].to_numpy() == "흡연").astype(int)
    costs[:, 1] += (frame["만성질환"].to_numpy() != "없음").astype(int)
    return costs


def policy_scores(frame: pd.DataFrame) -> np.ndarray:
    """Transparent need-fit rubric, with unaffordable products excluded."""
    goals = frame["우선보장목표"].to_numpy()
    secondary = frame["추가보장목표"].to_numpy()
    scores = np.ones((len(frame), len(PRODUCTS)), dtype=float)
    for index, goal in enumerate(GOALS):
        scores[:, index] += 6 * (goals == goal) + 3 * (secondary == goal)
    scores[:, 0] += 2 * (frame["가족력"].to_numpy() != "없음")
    scores[:, 0] += (frame["흡연여부"].to_numpy() == "흡연")
    scores[:, 1] += 3 * (frame["만성질환"].to_numpy() != "없음")
    scores[:, 2] += 3 * ((frame["결혼여부"].to_numpy() == "기혼")
                         & (frame["자녀수"].to_numpy() > 0))
    scores[:, 3] += 2 * (frame["나이"].to_numpy() >= 50)
    scores[mock_costs(frame) > frame["월예산_만원"].to_numpy()[:, None]] = -np.inf
    return scores


def policy_labels(frame: pd.DataFrame) -> np.ndarray:
    scores = policy_scores(frame)
    best = scores.max(axis=1)
    labels = np.asarray(PRODUCTS, dtype=object)[scores.argmax(axis=1)]
    labels[best < 4] = NO_FIT
    return labels


def make_cases(data: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add declared customer needs and budgets without using signup outcomes."""
    rng = np.random.default_rng(seed)
    frame = data[FEATURES].copy().reset_index(drop=True)
    weights = np.ones((len(frame), len(GOALS)), dtype=float)
    weights[:, 0] += 2 * (frame["가족력"].to_numpy() != "없음")
    weights[:, 1] += 3 * (frame["만성질환"].to_numpy() != "없음")
    weights[:, 2] += 3 * ((frame["결혼여부"].to_numpy() == "기혼")
                         & (frame["자녀수"].to_numpy() > 0))
    weights[:, 3] += 2 * (frame["나이"].to_numpy() >= 50)
    weights /= weights.sum(axis=1, keepdims=True)
    primary = [rng.choice(GOALS, p=row) for row in weights]
    secondary = [rng.choice([g for g in GOALS if g != goal]) if rng.random() < 0.4
                 else NO_SECONDARY_GOAL for goal in primary]
    income = frame["연소득_만원"].to_numpy()
    budget_weights = np.stack([
        np.full(len(frame), 2.0), np.full(len(frame), 2.0),
        np.full(len(frame), 2.0), 1.0 + (income >= 5000),
        1.0 + (income >= 6000), 1.0 + (income >= 7500),
    ], axis=1)
    budget_weights /= budget_weights.sum(axis=1, keepdims=True)
    frame["우선보장목표"] = primary
    frame["추가보장목표"] = secondary
    frame["월예산_만원"] = [rng.choice(BUDGETS, p=row) for row in budget_weights]
    frame[TARGET] = policy_labels(frame)
    return frame


def split_cases(frame: pd.DataFrame, seed: int = SEED):
    # Group on the original customer profile: even different goals for that
    # profile cannot appear on both sides of a split.
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False)
    left, right = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                       .split(frame, groups=groups))
    return frame.iloc[left].copy(), frame.iloc[right].copy(), groups


def make_model(kind: str):
    numeric = [field for field in INPUTS if field not in CATEGORIES and field not in EXTRA[:2]]
    categories = list(CATEGORIES) + list(EXTRA[:2])
    pre = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categories),
    ])
    if kind == "forest":
        classifier = RandomForestClassifier(n_estimators=240, max_depth=12,
                                            min_samples_leaf=2, random_state=SEED, n_jobs=1)
    elif kind == "boosting":
        classifier = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
                                                    max_depth=5, l2_regularization=1,
                                                    early_stopping=False, random_state=SEED)
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline([("features", pre), ("classifier", classifier)])


def predict_safe(model, frame: pd.DataFrame) -> np.ndarray:
    """Filter products failing the stated mock need or budget rubric."""
    probability = model.predict_proba(frame[list(INPUTS)])
    classes = np.asarray(model.classes_)
    eligible = policy_scores(frame) >= 4
    for index, product in enumerate(PRODUCTS):
        probability[~eligible[:, index], np.flatnonzero(classes == product)[0]] = -1
    return classes[probability.argmax(axis=1)]


@lru_cache(maxsize=2)
def _load_artifact(path: Path, mtime_ns: int) -> dict:
    """Load once per artifact version; training stays in the explicit CLI."""
    return joblib.load(path)


def _mock_rider(product: str, customer: pd.Series, remaining_budget: int) -> tuple[str, int]:
    """Pick a product-specific rider using a stated mock extra-cost assumption."""
    allowed = PRODUCT_RIDERS[product]
    if product == PRODUCTS[0]:
        if customer["가족력"] == "암" and remaining_budget >= 2:
            return allowed[1], 2
        return (allowed[0], 1) if remaining_budget >= 1 else (allowed[2], 0)
    if product == PRODUCTS[1]:
        return (allowed[0], 1) if remaining_budget >= 1 else (allowed[1], 0)
    if product == PRODUCTS[2]:
        return (allowed[0], 1) if customer["자녀수"] > 0 and remaining_budget >= 1 else (allowed[1], 0)
    return allowed[0], 0


def evaluate(write_outputs: bool = True) -> dict:
    dataset = load_dataset()
    cases = make_cases(dataset.data)
    train, holdout, groups = split_cases(cases)
    inner_train, inner_valid, _ = split_cases(train)
    candidates = []
    for kind in ("forest", "boosting"):
        model = make_model(kind).fit(inner_train[list(INPUTS)], inner_train[TARGET])
        prediction = predict_safe(model, inner_valid)
        candidates.append({"kind": kind,
                           "inner_accuracy": float(accuracy_score(inner_valid[TARGET], prediction)),
                           "inner_macro_f1": float(f1_score(inner_valid[TARGET], prediction,
                                                            average="macro", zero_division=0))})
    selected = max(candidates, key=lambda item: (item["inner_accuracy"], item["inner_macro_f1"]))["kind"]
    model = make_model(selected).fit(train[list(INPUTS)], train[TARGET])
    raw_predicted = model.predict(holdout[list(INPUTS)])
    predicted = predict_safe(model, holdout)
    valid = holdout.reset_index(drop=True)
    correct = int((valid[TARGET].to_numpy() == predicted).sum())
    product_index = {product: index for index, product in enumerate(PRODUCTS)}
    costs = mock_costs(holdout)
    budget_violation = sum(
        predicted[row] != NO_FIT
        and costs[row, product_index[predicted[row]]] > budget
        for row, budget in enumerate(holdout["월예산_만원"].to_numpy())
    )
    two_viable = (policy_scores(holdout) >= 4).sum(axis=1) >= 2
    two_options = int(np.sum(two_viable & (predicted != NO_FIT)))
    report = {
        "scope": "mock_suitability_policy_imitation_only",
        "source_sha256": dataset.source_sha256,
        "rows": len(cases), "train_rows": len(train), "holdout_rows": len(holdout),
        "overlapping_customer_profiles": len(set(groups.iloc[train.index]) & set(groups.iloc[holdout.index])),
        "inputs": list(INPUTS), "target": TARGET,
        "mock_monthly_cost_base_10k_krw": dict(zip(PRODUCTS, MOCK_COST_BASE)),
        "cost_rule": "Base +1 for age>=50; cancer/health +1 if smoking; simplified +1 if chronic disease",
        "model_candidates": candidates, "selected_model": selected,
        "holdout_raw_model_accuracy": float(accuracy_score(valid[TARGET], raw_predicted)),
        "holdout_correct_rows": correct,
        "holdout_accuracy": float(accuracy_score(valid[TARGET], predicted)),
        "holdout_macro_f1": float(f1_score(valid[TARGET], predicted, average="macro", zero_division=0)),
        "holdout_recommendation_coverage": float(np.mean(predicted != NO_FIT)),
        "holdout_two_option_rows": two_options,
        "holdout_two_option_rate_given_recommendation": float(two_options / max(1, np.sum(predicted != NO_FIT))),
        "holdout_policy_recommendation_coverage": float(np.mean(valid[TARGET].to_numpy() != NO_FIT)),
        "holdout_budget_violation_count": int(budget_violation),
        "holdout_label_counts": valid[TARGET].value_counts().to_dict(),
        "holdout_classification": classification_report(valid[TARGET], predicted,
                                                        output_dict=True, zero_division=0),
        "majority_class_baseline": float(valid[TARGET].value_counts(normalize=True).max()),
        "limitations": [
            "The target is generated by a disclosed rule, not independently adjudicated customer suitability.",
            "Budget costs are invented mock assumptions, not insurer premiums or eligibility decisions.",
            "Accuracy is policy imitation on synthetic cases and is not comparable to historical signup accuracy.",
            "No claim about real customer outcomes can be made from this evaluation.",
        ],
    }
    if write_outputs:
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        cases.to_csv(data_dir / "mock_suitability_customers.csv", index=False,
                     encoding="utf-8-sig")
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        valid.insert(0, "canonical_row", holdout.index.to_numpy() + 1)
        valid["predicted_mock_product"] = predicted
        valid.to_csv(out / "mock_suitability_holdout.csv", index=False, encoding="utf-8-sig")
        review = (valid.groupby(TARGET, group_keys=False)
                  .apply(lambda group: group.sample(n=min(20, len(group)), random_state=SEED),
                         include_groups=False)
                  .sample(frac=1, random_state=SEED))
        review = review[["canonical_row", *INPUTS]].copy()
        review_costs = mock_costs(review)
        for index in range(len(PRODUCTS)):
            review[f"목업비용_{index + 1}_만원"] = review_costs[:, index]
        review["검토_적합상품"] = ""
        review["검토_근거"] = ""
        review.to_csv(out / "mock_suitability_review_template.csv", index=False,
                      encoding="utf-8-sig")
        (out / "mock_suitability_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifact = ROOT / "artifacts/customer_ml/mock_suitability.joblib"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": make_model(selected).fit(cases[list(INPUTS)], cases[TARGET]),
                     "report": report, "version": ARTIFACT_VERSION,
                     "scikit_learn_version": sklearn.__version__,
                     "source_sha256": dataset.source_sha256}, artifact)
    return report


def recommend_mock(customer: dict, artifact_path: Path | None = None) -> dict:
    """Serve a named mock policy, never an insurance quote or eligibility result."""
    artifact_path = artifact_path or ROOT / "artifacts/customer_ml/mock_suitability.joblib"
    if not artifact_path.is_file():
        raise FileNotFoundError("먼저 python -m customer_ml.mock_suitability 로 목업 모델을 생성하세요.")
    bundle = _load_artifact(artifact_path.resolve(), artifact_path.stat().st_mtime_ns)
    if bundle.get("version") != ARTIFACT_VERSION or bundle.get("scikit_learn_version") != sklearn.__version__:
        raise RuntimeError("목업 모델 버전이 현재 코드/환경과 다릅니다. python -m customer_ml.mock_suitability 로 다시 학습하세요.")
    if bundle.get("source_sha256") != hashlib.sha256(DATA_PATH.read_bytes()).hexdigest():
        raise RuntimeError("고객 더미데이터가 바뀌었습니다. 목업 모델을 다시 학습하세요.")
    valid = validate_features(pd.DataFrame([customer]))
    primary = customer.get("우선보장목표")
    secondary = customer.get("추가보장목표", NO_SECONDARY_GOAL)
    budget = customer.get("월예산_만원")
    if primary not in GOALS or secondary not in (*GOALS, NO_SECONDARY_GOAL) or primary == secondary:
        raise ValueError("보장 목표를 확인해 주세요.")
    if isinstance(budget, bool) or budget not in BUDGETS:
        raise ValueError("월예산_만원은 목업 예산 구간 중 하나여야 합니다.")
    valid["우선보장목표"] = primary
    valid["추가보장목표"] = secondary
    valid["월예산_만원"] = budget
    model = bundle["model"]
    selected = str(predict_safe(model, valid)[0])
    ranking = []
    if selected != NO_FIT:
        probabilities = model.predict_proba(valid[list(INPUTS)])[0]
        classes = list(model.classes_)
        costs = mock_costs(valid)[0]
        need_scores = policy_scores(valid)[0]
        candidates = [product for index, product in enumerate(PRODUCTS) if need_scores[index] >= 4]
        # The first option is the evaluated ML decision. A second option is a
        # comparison candidate ranked by the transparent need/budget policy.
        candidates.sort(key=lambda product: (-need_scores[PRODUCTS.index(product)],
                                             PRODUCTS.index(product)))
        candidates = [selected, *(p for p in candidates if p != selected)]
        for rank, product in enumerate(candidates[:2], start=1):
            index = PRODUCTS.index(product)
            rider, rider_cost = _mock_rider(product, valid.iloc[0], budget - int(costs[index]))
            ranking.append({"순위": rank, "주계약": product,
                            "모델점수": round(float(probabilities[classes.index(product)] * 100), 1),
                            "선정기준": "모델 예측" if rank == 1 else "목업 보장·예산 비교",
                            "추천특약": rider,
                            "모의_월부담_만원": int(costs[index]) + rider_cost})
    return {"목업_적합상품": selected, "추천순위": ranking, "목업": True,
            "안내": "목업 보장·예산 규칙을 학습한 예시입니다. 모델점수와 모의 월부담은 실제 가입 확률·보험료·가입 가능 여부·고객 적합성을 뜻하지 않습니다."}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
