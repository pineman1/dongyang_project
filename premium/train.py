"""Train, evaluate with whole ages held out, and save reproducible artifacts."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline

from premium.data import DATA_PATH, PLAN_COLUMNS, ROOT, TARGET, VERSION, load_data
from premium.model import MODEL_NAME, PremiumPredictor, design_matrix, make_model, preprocessor


def metrics(actual, predicted) -> dict:
    return {"mae_krw": round(float(mean_absolute_error(actual, predicted)), 2),
            "rmse_krw": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
            "mape_percent": round(float(mean_absolute_percentage_error(actual, predicted) * 100), 4)}


def split_by_age(data: pd.DataFrame, holdout_age: int = 40):
    training = data.loc[data.age != holdout_age].copy()
    holdout = data.loc[data.age == holdout_age].copy()
    if training.empty or holdout.empty:
        raise ValueError("학습 또는 평가 데이터가 비어 있습니다.")
    return training, holdout


def evaluate(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    training, holdout = split_by_age(data)
    x_train, x_test = design_matrix(training), design_matrix(holdout)
    models = {
        MODEL_NAME: make_model(),
        "dummy_median": DummyRegressor(strategy="median"),
        "random_forest": Pipeline([
            ("features", preprocessor()),
            ("regression", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ]),
    }
    comparisons = {}
    predictions = holdout.copy()
    for name, model in models.items():
        model.fit(x_train, training[TARGET])
        values = model.predict(x_test)
        comparisons[name] = metrics(holdout[TARGET], values)
        predictions[name] = values
    # A simple tariff-table interpolation is a strong non-ML comparison.
    linear_predictions = []
    log_interpolation = []
    for _, row in holdout.iterrows():
        curve = training.loc[(training[PLAN_COLUMNS] == row[PLAN_COLUMNS]).all(axis=1)].sort_values("age")
        linear_predictions.append(np.interp(row.age, curve.age, curve[TARGET]))
        log_interpolation.append(np.exp(np.interp(row.age, curve.age, np.log(curve[TARGET]))))
    comparisons["linear_table_interpolation"] = metrics(holdout[TARGET], linear_predictions)
    predictions["linear_table_interpolation"] = linear_predictions
    comparisons["log_table_interpolation"] = metrics(holdout[TARGET], log_interpolation)
    predictions["log_table_interpolation"] = log_interpolation

    age_diagnostics = []
    for age in sorted(data.age.unique()):
        fold_train, fold_test = split_by_age(data, int(age))
        model = make_model().fit(design_matrix(fold_train), fold_train[TARGET])
        values = model.predict(design_matrix(fold_test))
        age_diagnostics.append({
            "holdout_age": int(age), "train_ages": sorted(map(int, fold_train.age.unique())),
            "mode": "interpolation" if age == 40 else "extrapolation_diagnostic_only",
            "test_rows": len(fold_test), **metrics(fold_test[TARGET], values),
        })
    per_sex = {
        sex: {"rows": len(group), **metrics(group[TARGET], group[MODEL_NAME])}
        for sex, group in predictions.groupby("sex")
    }
    report = {
        "model": MODEL_NAME, "model_selection": "fixed_before_evaluation_no_hyperparameter_search",
        "dataset_rows": len(data), "product_version": VERSION,
        "independent_customer_records": 0, "synthetic_rows": 0,
        "holdout": {"train_ages": [30, 50], "test_ages": [40],
                    "train_rows": len(training), "test_rows": len(holdout),
                    "train_record_ids": training.record_id.tolist(),
                    "test_record_ids": holdout.record_id.tolist()},
        "comparison": comparisons, "holdout_by_sex": per_sex,
        "leave_one_age_out_diagnostics": age_diagnostics,
        "limitations": [
            "72 rows from one historical brochure are tariff examples, not independent customers.",
            "Only the held-out age 40 is an interpolation check; ages 31-39/41-49 have no labels.",
            "The 30/50 holdouts test extrapolation sensitivity, not supported production inputs.",
            "Small same-product holdout errors do not measure current-quote or new-product accuracy.",
            "No calibrated prediction interval is available.",
        ],
    }
    return report, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "premium")
    args = parser.parse_args()
    data = load_data()
    report, predictions = evaluate(data)
    predictor = PremiumPredictor(data)
    fingerprint = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    report.update({"dataset_sha256": fingerprint,
                   "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                   "final_training_rows": len(data)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": predictor.model, "dataset_sha256": fingerprint,
                 "feature_transform": "premium.model.design_matrix", "report": report},
                args.output_dir / "model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(args.output_dir / "holdout_predictions.csv", index=False, encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(f"Saved model, metrics and holdout predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
