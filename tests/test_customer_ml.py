from io import BytesIO
import json
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from customer_ml.data import (COLUMNS, FEATURES, PRODUCT_RIDERS, ROOT,
                              SORTED_FILES, load_dataset, read_dataset, validate_training)
from customer_ml.model import CustomerPredictor
from customer_ml.train import evaluate, split_dataset

ZIP_PATH = ROOT / "reports/customer_segmentation/동양생명_항목별정렬_CSV.zip"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def predictor(dataset):
    return CustomerPredictor(dataset.data)


def test_original_and_six_sorted_views_are_exactly_one_population(dataset):
    archive = load_dataset(ZIP_PATH)
    pd.testing.assert_frame_equal(dataset.data, archive.data)
    assert dataset.fingerprint == archive.fingerprint
    assert archive.sorted_views == 6
    assert len(archive.data) == 1000
    assert dataset.metadata["exact_duplicate_rows"] == 15
    assert list(archive.data.columns) == COLUMNS
    assert archive.source_sha256 != dataset.source_sha256


@pytest.mark.parametrize("change", ["missing_target", "nan", "fractional", "bool", "category", "product_rider", "age"])
def test_invalid_training_data_rejected(dataset, change):
    frame = dataset.data.copy()
    if change == "missing_target":
        frame = frame.drop(columns="가입상품")
    elif change == "nan":
        frame.loc[0, "가입특약"] = None
    elif change == "fractional":
        frame["나이"] = frame["나이"].astype(float)
        frame.loc[0, "나이"] = 30.5
    elif change == "bool":
        frame["나이"] = frame["나이"].astype(object)
        frame.loc[0, "나이"] = True
    elif change == "category":
        frame.loc[0, "흡연여부"] = "모름"
    elif change == "product_rider":
        frame.loc[0, "가입상품"] = "수호천사 행복 연금보험"
        frame.loc[0, "가입특약"] = "수술비 보장 특약"
    else:
        frame.loc[0, "나이"] = 66
    with pytest.raises(ValueError):
        validate_training(frame)


def test_zip_rejects_conflicting_or_missing_view(dataset):
    for incomplete in [False, True]:
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for i, name in enumerate(SORTED_FILES[:-1] if incomplete else SORTED_FILES):
                data = dataset.data.copy()
                if not incomplete and i == 1:
                    data.loc[0, "연소득_만원"] = 9000
                archive.writestr(name, data.to_csv(index=False).encode("utf-8-sig"))
        with pytest.raises(ValueError):
            read_dataset(stream.getvalue(), "sorted.zip")


def test_no_ids_or_label_derived_fields_enter_model(dataset, predictor):
    poisoned = dataset.data.copy()
    poisoned["분석ID"] = range(len(poisoned))
    poisoned["상품분류"] = poisoned["가입상품"]
    poisoned["특약상태"] = poisoned["가입특약"]
    poisoned["특약가입여부"] = "fake label"
    pd.testing.assert_frame_equal(validate_training(poisoned), dataset.data)
    assert list(predictor.main_model.feature_names_in_) == FEATURES
    for model in predictor.rider_models.values():
        assert list(model.feature_names_in_) == FEATURES


def test_group_split_keeps_repeated_profiles_together(dataset):
    train, test, _ = split_dataset(dataset.data)
    train_profiles = set(map(tuple, train[FEATURES].to_numpy()))
    test_profiles = set(map(tuple, test[FEATURES].to_numpy()))
    assert not train_profiles & test_profiles
    assert len(train) + len(test) == 1000
    assert set(train.index).isdisjoint(test.index)


def test_recommendations_and_selected_product_use_valid_riders(dataset, predictor):
    customer = dataset.data.iloc[250][FEATURES].to_dict()
    recommended = predictor.recommend(customer)
    assert abs(sum(row["score"] for row in recommended["product_ranking"]) - 1) < 1e-12
    assert recommended["training_rows"] == 1000
    for product, allowed in PRODUCT_RIDERS.items():
        result = predictor.recommend(customer, product)
        assert result["rider"] in allowed
        assert result["rider_product"] == product
        assert result["product"] == recommended["product"]
    batch = predictor.predict_batch(dataset.data.iloc[:80])
    assert all(r.rider in PRODUCT_RIDERS[r.product] for r in batch.itertuples())
    with pytest.raises(ValueError):
        predictor.recommend({**customer, "나이": 19})


def test_metrics_recomputed_and_legacy_adapter(dataset, predictor):
    report, rows = evaluate(dataset)
    assert report["split"]["overlapping_groups"] == 0
    assert report["metrics"]["product_accuracy"] == (rows["가입상품"] == rows.predicted_product).mean()
    both = (rows["가입상품"] == rows.predicted_product) & (rows["가입특약"] == rows.predicted_rider)
    assert report["metrics"]["product_and_rider_accuracy"] == both.mean()
    from ml_model import get_recommendation
    customer = dataset.data.iloc[200][FEATURES].to_dict()
    old = get_recommendation(customer)
    new = predictor.recommend(customer)
    assert old["주계약"] == new["product"] and old["추천특약"] == new["rider"]
    assert old["학습고객수"] == 1000


def test_sorted_order_does_not_change_predictions(dataset, predictor):
    reordered = CustomerPredictor(dataset.data.sample(frac=1, random_state=7))
    customer = dataset.data.iloc[77][FEATURES].to_dict()
    assert reordered.recommend(customer) == predictor.recommend(customer)


def test_cli_zip_training_saves_only_one_population(tmp_path):
    result = subprocess.run([sys.executable, "-m", "customer_ml.train", "--data", str(ZIP_PATH),
                             "--output-dir", str(tmp_path)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["final_training_rows"] == 1000
    assert report["dataset"]["sorted_views"] == 6
    assert (tmp_path / "model.joblib").is_file()
    assert len(pd.read_csv(tmp_path / "holdout_predictions.csv")) == report["split"]["test_rows"]
