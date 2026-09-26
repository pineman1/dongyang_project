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


def test_expanded_population_and_historical_sorted_views(dataset):
    archive = load_dataset(ZIP_PATH)
    assert len(dataset.data) == 5000
    assert dataset.fingerprint != archive.fingerprint
    assert archive.sorted_views == 6
    assert len(archive.data) == 1000
    assert dataset.metadata["exact_duplicate_rows"] >= 15
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
    assert len(train) + len(test) == len(dataset.data)
    assert set(train.index).isdisjoint(test.index)


def test_recommendations_and_selected_product_use_valid_riders(dataset, predictor):
    customer = dataset.data.iloc[250][FEATURES].to_dict()
    recommended = predictor.recommend(customer)
    assert abs(sum(row["score"] for row in recommended["product_ranking"]) - 1) < 1e-12
    assert recommended["training_rows"] == len(dataset.data)
    assert recommended["available"] and recommended["rider_available"]
    for product, allowed in PRODUCT_RIDERS.items():
        result = predictor.recommend(customer, product)
        assert result["rider_available"] and result["rider"] in allowed
        assert abs(sum(row["score"] for row in result["rider_ranking"]) - 1) < 1e-12
        assert result["rider_product"] == product
        assert result["product"] == recommended["product"]
    batch = predictor.predict_batch(dataset.data.iloc[:80])
    assert all(r.rider in PRODUCT_RIDERS[r.product] if r.rider_available else r.rider is None
               for r in batch.itertuples())
    assert predictor.recommend({**customer, "나이": 19})["message"] == NO_DATA


def test_metrics_recomputed_and_legacy_adapter(dataset, predictor):
    report, rows = evaluate(dataset)
    assert report["split"]["overlapping_groups"] == 0
    assert report["serving_policy"]["holdout_exact_profile_coverage"] == 0
    assert report["serving_policy"]["holdout_prediction_coverage"] == 1
    assert report["evaluation_scope"] == "serving_classifier_with_predict_proba_fallback"
    assert report["one_thousand_training_rows_comparison"]["training_rows"] == 1000
    assert report["metrics"]["product_accuracy"] == (rows["가입상품"] == rows.predicted_product).mean()
    both = (rows["가입상품"] == rows.predicted_product) & (rows["가입특약"] == rows.predicted_rider)
    assert report["metrics"]["product_and_rider_accuracy"] == both.mean()
    top_two = (rows["가입상품"] == rows.predicted_product) | (rows["가입상품"] == rows.second_predicted_product)
    top_two_pairs = both | ((rows["가입상품"] == rows.second_predicted_product)
                            & (rows["가입특약"] == rows.second_predicted_rider))
    assert report["metrics"]["product_top2_coverage"] == top_two.mean()
    assert report["metrics"]["product_and_rider_top2_coverage"] == top_two_pairs.mean()
    assert report["metrics"]["product_top2_coverage"] >= report["metrics"]["product_accuracy"]
    from ml_model import get_recommendation
    customer = dataset.data.iloc[200][FEATURES].to_dict()
    old = get_recommendation(customer)
    new = predictor.recommend(customer)
    assert old["주계약"] == new["product"]
    assert old["추천특약"] == (new["rider"] if new["rider_available"] else NO_DATA)
    assert old["학습고객수"] == len(dataset.data)
    assert [option["주계약"] for option in old["추천순위"]] == [
        row["label"] for row in new["product_ranking"][:2]]
    assert all(option["추천특약"] in PRODUCT_RIDERS[option["주계약"]] for option in old["추천순위"])


def test_default_joblib_loads_without_retraining(monkeypatch):
    from customer_ml import store
    from customer_ml.data import DATA_PATH
    from unittest.mock import Mock
    never_train = Mock(side_effect=AssertionError("A valid Joblib artifact must load without training"))
    monkeypatch.setattr(store, "evaluate", never_train)
    monkeypatch.setattr(store, "read_dataset", never_train)
    monkeypatch.setattr("customer_ml.model.make_classifier", never_train)
    dataset, predictor, report = store.load_or_train(DATA_PATH.read_bytes(), DATA_PATH.name)
    assert len(dataset.data) == len(predictor.data) == report["final_training_rows"]
    never_train.assert_not_called()


@pytest.mark.parametrize("change", [{"나이": 66}, {"흡연여부": "알 수 없음"}, {"나이": None}])
def test_invalid_inputs_do_not_call_model_or_return_scores(dataset, predictor, monkeypatch, change):
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


def test_unseen_combination_uses_model_probabilities(dataset, predictor):
    customer = {"나이": 40, "성별": "남성", "연소득_만원": 5000, "직업위험등급": 1,
                "결혼여부": "기혼", "자녀수": 1, "흡연여부": "비흡연", "만성질환": "없음", "가족력": "없음"}
    assert all(value in set(dataset.data[field]) for field, value in customer.items())
    assert not (dataset.data[FEATURES] == pd.Series(customer)).all(axis=1).any()
    result = predictor.recommend(customer)
    assert result["available"] and result["rider_available"] and result["used_fallback"]
    assert result["product"] == result["product_ranking"][0]["label"]
    assert result["rider"] == result["rider_ranking"][0]["label"]
    assert result["product_ranking"][0]["score"] == max(predictor.main_model.predict_proba(pd.DataFrame([customer]))[0])
    assert predictor.recommend(dataset.data.iloc[0][FEATURES].to_dict(), product="새 보험")["message"] == NO_DATA


def test_batch_and_legacy_predict_unseen_valid_profile(dataset, predictor):
    from ml_model import get_recommendation
    customer = dataset.data.iloc[250][FEATURES].to_dict()
    unseen = {**customer, "연소득_만원": 3500}
    results = predictor.predict_batch(pd.DataFrame([customer, unseen]))
    assert results.iloc[0]["available"]
    assert results.iloc[1]["available"] and results.iloc[1]["rider_available"]
    assert results.iloc[1]["rider"] in PRODUCT_RIDERS[results.iloc[1]["product"]]
    output = get_recommendation({**unseen, "선호상품": "암보험"})
    assert output["자료있음"] and output["특약자료있음"]
    assert output["주계약"] == "수호천사 암/건강보험"
    assert output["추천특약"] in PRODUCT_RIDERS[output["주계약"]]
    assert output["주계약_확률"] is not None and output["특약_확률"] is not None
    assert output["미일치조건예측"]
    assert len(output["추천순위"]) == 2
    assert output["추천순위"][0]["모델점수"] >= output["추천순위"][1]["모델점수"]


def test_preferred_product_uses_its_own_rider_model(dataset):
    from ml_model import get_recommendation
    customer = dataset.data.iloc[250][FEATURES].to_dict()
    output = get_recommendation({**customer, "선호상품": "연금보험"})
    assert output["자료있음"] and output["특약자료있음"]
    assert output["추천특약"] == "특약 없음" and output["특약_확률"] == 100


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
