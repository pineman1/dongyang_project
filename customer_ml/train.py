"""Train and evaluate synthetic product/rider recommendations without group leakage."""

import argparse
import json
from pathlib import Path
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

from .data import DATA_PATH, FEATURES, PRODUCT_RIDERS, ROOT, CustomerDataset, load_dataset
from .model import CustomerPredictor, LEGACY_MODEL_PARAMS, NO_DATA

TUNING_CANDIDATES = [LEGACY_MODEL_PARAMS,
                     {"n_estimators": 240, "max_depth": 12, "min_samples_leaf": 2},
                     {"n_estimators": 320, "max_depth": 16, "min_samples_leaf": 1}]


def split_dataset(data: pd.DataFrame):
    # Equal feature profiles stay together, including repeated or conflicting labels.
    groups = pd.util.hash_pandas_object(data[FEATURES], index=False)
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(data, groups=groups))
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    if set(train["가입상품"]) != set(PRODUCT_RIDERS):
        raise ValueError("분리된 학습 자료에 상품 4종이 모두 필요합니다. 고객 데이터를 추가하세요.")
    return train, test, groups


def _accuracy(predictions: pd.DataFrame, actual: pd.DataFrame) -> dict:
    product_ok = predictions["product"].to_numpy() == actual["가입상품"].to_numpy()
    rider_ok = predictions["rider"].to_numpy() == actual["가입특약"].to_numpy()
    return {"product_accuracy": float(product_ok.mean()),
            "product_and_rider_accuracy": float((product_ok & rider_ok).mean())}


def select_parameters(train: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Choose hyperparameters on a second grouped split, without seeing final holdout."""
    inner_train, inner_valid, _ = split_dataset(train)
    scores = []
    for params in TUNING_CANDIDATES:
        candidate = CustomerPredictor(inner_train, params)
        metrics = _accuracy(candidate._predict_for_evaluation(inner_valid), inner_valid)
        scores.append({"params": dict(params), **metrics})
    best = max(enumerate(scores), key=lambda item: (
        item[1]["product_and_rider_accuracy"], item[1]["product_accuracy"], -item[0]))[1]
    return dict(best["params"]), scores


def evaluate(dataset: CustomerDataset) -> tuple[dict, pd.DataFrame]:
    train, test, groups = split_dataset(dataset.data)
    selected_params, tuning_scores = select_parameters(train)
    predictor = CustomerPredictor(train, selected_params)
    predictions = predictor._predict_for_evaluation(test)
    actual = test.reset_index(drop=True)
    legacy = predictor if selected_params == LEGACY_MODEL_PARAMS else CustomerPredictor(train, LEGACY_MODEL_PARAMS)
    legacy_metrics = _accuracy(legacy._predict_for_evaluation(test), actual)
    small_sample = train.sample(n=min(1000, len(train)), random_state=42)
    small_model = CustomerPredictor(small_sample, LEGACY_MODEL_PARAMS)
    small_sample_metrics = _accuracy(small_model._predict_for_evaluation(test), actual)
    # Conditional rider quality measures the second stage using the known product.
    conditional = predictor._predict_for_evaluation(test, products=test["가입상품"])
    product = train["가입상품"].mode().iloc[0]
    rider = train.loc[train["가입상품"] == product, "가입특약"].mode().iloc[0]
    product_ok = predictions["product"] == actual["가입상품"]
    pair_ok = product_ok & (predictions["rider"] == actual["가입특약"])
    score = {
        "product_accuracy": float(product_ok.mean()),
        "product_macro_f1": float(f1_score(actual["가입상품"], predictions["product"],
                                          labels=list(PRODUCT_RIDERS), average="macro", zero_division=0)),
        "rider_accuracy_given_true_product": float(accuracy_score(actual["가입특약"], conditional["rider"])),
        "product_and_rider_accuracy": float(pair_ok.mean()),
    }
    baseline = {"product_accuracy": float((actual["가입상품"] == product).mean()),
                "product_and_rider_accuracy": float(((actual["가입상품"] == product) & (actual["가입특약"] == rider)).mean())}
    report = {
        "dataset": dataset.metadata, "features": FEATURES,
        "evaluation_scope": "serving_classifier_with_predict_proba_fallback",
        "serving_policy": {
            "customer": "validated_features_then_highest_predict_proba",
            "rider": "highest_predict_proba_within_selected_product",
            "no_data_message": NO_DATA,
            "holdout_prediction_coverage": float(predictions[["product", "rider"]].notna().all(axis=1).mean()),
            "holdout_exact_profile_coverage": float(test[FEATURES].apply(tuple, axis=1).isin(predictor.profiles).mean()),
        },
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__},
        "excluded_from_features": ["가입상품", "가입특약", "분석ID", "상품분류", "특약상태", "특약가입여부"],
        "model": "random_forest_product_then_product_specific_rider",
        "model_selection": "nested_grouped_validation_on_training_partition", "random_state": 42,
        "selected_params": selected_params, "tuning_candidates": tuning_scores,
        "split": {"method": "GroupShuffleSplit by all nine raw customer features",
                  "test_group_fraction": 0.2, "training_rows": len(train), "test_rows": len(test),
                  "training_groups": int(groups.iloc[train.index].nunique()),
                  "test_groups": int(groups.iloc[test.index].nunique()),
                  "overlapping_groups": len(set(groups.iloc[train.index]) & set(groups.iloc[test.index]))},
        "metrics": score, "legacy_params_holdout": legacy_metrics,
        "one_thousand_training_rows_comparison": {"training_rows": len(small_sample),
                                                  **small_sample_metrics},
        "majority_baseline": baseline,
        "product_classification": classification_report(actual["가입상품"], predictions["product"],
                                                         labels=list(PRODUCT_RIDERS), output_dict=True, zero_division=0),
        "limitations": ["Synthetic labels were sampled by generate_data.py, not observed customer decisions.",
                        "Scores are uncalibrated model outputs, not real purchase probabilities.",
                        "The CSV has no premium target; this model does not predict monthly premiums.",
                        "This is one grouped holdout on synthetic rows, not external validation."],
    }
    actual.insert(0, "canonical_row", test.index.to_numpy() + 1)
    actual["predicted_product"] = predictions["product"]
    actual["predicted_rider"] = predictions["rider"]
    actual["rider_given_true_product"] = conditional["rider"]
    return report, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "customer_ml")
    args = parser.parse_args()
    dataset = load_dataset(args.data)
    report, holdout = evaluate(dataset)
    predictor = CustomerPredictor(dataset.data, report["selected_params"])
    report["final_training_rows"] = len(dataset.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from .store import CACHE_VERSION
    joblib.dump({"version": CACHE_VERSION, "dataset": dataset, "predictor": predictor,
                 "report": report}, args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    holdout.to_csv(args.output_dir / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(dataset.data), "split": report["split"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
