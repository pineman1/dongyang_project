from io import BytesIO
import json
import subprocess
import sys
import zipfile

import pandas as pd
import pytest

from customer_ml.data import (COLUMNS, FEATURES, PRODUCT_RIDERS, ROOT,
                              SORTED_FILES, load_dataset, read_dataset, validate_training)
from customer_ml.model import NO_DATA, CustomerPredictor
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
        matching = dataset.data.loc[(dataset.data[FEATURES] == pd.Series(customer)).all(axis=1)]
        if product in set(matching["가입상품"]):
            assert result["rider_available"] and result["rider"] in allowed
        else:
            assert not result["rider_available"] and result["rider"] is None
            assert result["rider_ranking"] == [] and result["rider_score"] is None
        assert result["rider_product"] == product
        assert result["product"] == recommended["product"]
    batch = predictor.predict_batch(dataset.data.iloc[:80])
    assert all(r.rider in PRODUCT_RIDERS[r.product] if r.rider_available else r.rider is None
               for r in batch.itertuples())
    assert predictor.recommend({**customer, "나이": 19})["message"] == NO_DATA


def test_metrics_recomputed_and_legacy_adapter(dataset, predictor):
    report, rows = evaluate(dataset)
    assert report["split"]["overlapping_groups"] == 0
    assert report["serving_policy"]["holdout_profile_coverage"] == 0
    assert report["evaluation_scope"] == "underlying_classifier_before_no_data_policy"
    assert report["metrics"]["product_accuracy"] == (rows["가입상품"] == rows.predicted_product).mean()
    both = (rows["가입상품"] == rows.predicted_product) & (rows["가입특약"] == rows.predicted_rider)
    assert report["metrics"]["product_and_rider_accuracy"] == both.mean()
    from ml_model import get_recommendation
    customer = dataset.data.iloc[200][FEATURES].to_dict()
    old = get_recommendation(customer)
    new = predictor.recommend(customer)
    assert old["주계약"] == new["product"]
    assert old["추천특약"] == (new["rider"] if new["rider_available"] else NO_DATA)
    assert old["학습고객수"] == 1000


@pytest.mark.parametrize("change", [{"연소득_만원": 3500}, {"나이": 66}, {"흡연여부": "알 수 없음"}, {"나이": None}])
def test_unlearned_inputs_do_not_call_model_or_return_scores(dataset, predictor, monkeypatch, change):
    from unittest.mock import Mock
    customer = {**dataset.data.iloc[250][FEATURES].to_dict(), **change}
    rank = Mock(side_effect=AssertionError("Unlearned inputs must not be predicted"))
    monkeypatch.setattr(predictor, "_ranking", rank)
    result = predictor.recommend(customer)
    assert result["message"] == NO_DATA and not result["available"]
    assert result["product"] is None and result["rider"] is None
    assert result["product_score"] is None and result["rider_score"] is None
    assert result["product_ranking"] == result["rider_ranking"] == []
    rank.assert_not_called()


def test_seen_individual_values_but_unseen_combination_is_no_data(dataset, predictor):
    customer = {"나이": 40, "성별": "남성", "연소득_만원": 5000, "직업위험등급": 1,
                "결혼여부": "기혼", "자녀수": 1, "흡연여부": "비흡연", "만성질환": "없음", "가족력": "없음"}
    assert all(value in set(dataset.data[field]) for field, value in customer.items())
    assert not (dataset.data[FEATURES] == pd.Series(customer)).all(axis=1).any()
    assert predictor.recommend(customer)["message"] == NO_DATA
    assert predictor.recommend(dataset.data.iloc[0][FEATURES].to_dict(), product="새 보험")["message"] == NO_DATA


def test_batch_and_legacy_no_data_do_not_leak_results(dataset, predictor):
    from ml_model import get_recommendation
    customer = dataset.data.iloc[250][FEATURES].to_dict()
    unseen = {**customer, "연소득_만원": 3500}
    results = predictor.predict_batch(pd.DataFrame([customer, unseen]))
    assert results.iloc[0]["available"]
    assert not results.iloc[1]["available"] and results.iloc[1]["message"] == NO_DATA
    assert results.iloc[1]["product"] is None and results.iloc[1]["rider"] is None
    output = get_recommendation({**unseen, "선호상품": "암보험"})
    assert not output["자료있음"] and not output["특약자료있음"]
    assert output["주계약"] == output["추천특약"] == NO_DATA
    assert output["주계약_확률"] is None and output["특약_확률"] is None


def test_preferred_product_cannot_bypass_rider_evidence(dataset):
    from ml_model import get_recommendation
    customer = dataset.data.iloc[250][FEATURES].to_dict()
    output = get_recommendation({**customer, "선호상품": "연금보험"})
    assert output["자료있음"] and not output["특약자료있음"]
    assert output["추천특약"] == NO_DATA and output["특약_확률"] is None


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
