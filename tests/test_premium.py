import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from premium.data import DATA_PATH, SOURCE_PATH, TARGET, load_data, validate_data
from premium.model import MODEL_NAME, PremiumPredictor, PremiumRequest
from premium.train import evaluate, split_by_age


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def predictor(data):
    return PremiumPredictor(data)


@pytest.fixture
def request_input():
    return PremiumRequest(40, "남자", "치매보장형", "표준형", 20)


def test_source_fingerprint_and_independently_reviewed_values(data):
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == source["dataset_sha256"]
    assert len(data) == 72 and source["synthetic_rows"] == 0
    # PDF page 3, left standard table / right partial-refund table.
    first = data.query("age == 40 and payment_years == 20 and sex == '남자' and refund_type == '표준형'")
    assert first[TARGET].tolist() == [265600]
    second = data.query("age == 50 and payment_years == 15 and sex == '여자' and refund_type == '해약환급금일부지급형'")
    assert second[TARGET].tolist() == [497100]


@pytest.mark.parametrize("change", ["duplicate", "missing", "negative", "nan", "wrong_version", "missing_column"])
def test_bad_training_data_rejected(data, change):
    bad = data.copy()
    if change == "duplicate":
        bad.loc[1] = bad.loc[0]
    elif change == "missing":
        bad = bad.iloc[:-1]
    elif change == "negative":
        bad.loc[0, TARGET] = -1
    elif change == "nan":
        bad.loc[0, TARGET] = np.nan
    elif change == "wrong_version":
        bad.loc[0, "product_version"] = "2026-09"
    else:
        bad = bad.drop(columns="age")
    with pytest.raises(ValueError):
        validate_data(bad)


@pytest.mark.parametrize("changes", [
    {"age": 29}, {"age": 51}, {"age": 35.5}, {"age": True},
    {"sex": "unknown"}, {"coverage_type": "암보험"}, {"refund_type": "unknown"},
    {"payment_years": 10}, {"payment_years": 7.5},
    {"sum_assured_krw": 20_000_000}, {"coverage_end_age": 100},
])
def test_unsupported_quotes_fail_closed(predictor, request_input, changes):
    with pytest.raises(ValueError):
        predictor.predict(replace(request_input, **changes))


def test_observed_example_and_unobserved_age_are_distinct(predictor, request_input):
    exact = predictor.predict(request_input)
    assert exact["published_example_krw"] == 265600
    assert exact["is_unobserved_age"] is False
    estimate = predictor.predict(replace(request_input, age=35))
    assert estimate["published_example_krw"] is None
    assert estimate["is_unobserved_age"] is True
    assert 198000 < estimate["monthly_premium_krw"] < 265600
    assert estimate["product_version"] == "2023-09"


def test_predictions_positive_and_reproducible(data, predictor):
    other = PremiumPredictor(data)
    for row in data.itertuples():
        request = PremiumRequest(row.age, row.sex, row.coverage_type, row.refund_type, row.payment_years)
        output = predictor.predict(request)
        assert output == other.predict(request)
        assert output["monthly_premium_krw"] > 0
        assert output["published_example_krw"] == row.monthly_premium_krw


def test_age_holdout_has_no_overlap_and_recomputable_metrics(data):
    train, test = split_by_age(data)
    assert set(train.age) == {30, 50} and set(test.age) == {40}
    assert len(train) == 48 and len(test) == 24
    assert not set(train.record_id) & set(test.record_id)
    report, predictions = evaluate(data)
    expected_mae = abs(predictions[TARGET] - predictions[MODEL_NAME]).mean()
    assert report["comparison"][MODEL_NAME]["mae_krw"] == round(expected_mae, 2)
    assert report["comparison"][MODEL_NAME]["mae_krw"] < report["comparison"]["dummy_median"]["mae_krw"]
    assert {fold["holdout_age"] for fold in report["leave_one_age_out_diagnostics"]} == {30, 40, 50}


def test_cli_rejects_out_of_domain_and_train_exports(tmp_path):
    invalid = subprocess.run([sys.executable, "-m", "premium.predict", "--age", "60", "--sex", "male"],
                             capture_output=True)
    assert invalid.returncode == 2
    trained = subprocess.run([sys.executable, "-m", "premium.train", "--output-dir", str(tmp_path)],
                             capture_output=True)
    assert trained.returncode == 0, trained.stderr.decode(errors="replace")
    assert (tmp_path / "model.joblib").is_file()
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["final_training_rows"] == 72
    assert (tmp_path / "holdout_predictions.csv").is_file()
