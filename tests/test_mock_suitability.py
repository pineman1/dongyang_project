"""Verify the separate mock target and its budget guardrail."""

import json
from pathlib import Path

import pandas as pd
import pytest

from customer_ml.data import FEATURES, PRODUCT_RIDERS, load_dataset
from customer_ml.mock_suitability import (
    INPUTS, NO_FIT, PRODUCTS, TARGET, make_cases, mock_costs, policy_labels,
    policy_scores, recommend_mock, split_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def test_mock_target_uses_declared_inputs_and_keeps_original_labels_out():
    base = load_dataset().data
    cases = make_cases(base)
    assert len(cases) == len(base) == 5000
    exported = pd.read_csv(ROOT / "data/mock_suitability_customers.csv")
    pd.testing.assert_frame_equal(exported, cases)
    assert set(INPUTS).isdisjoint({"가입상품", "가입특약"})
    assert cases[TARGET].tolist() == policy_labels(cases).tolist()
    assert make_cases(base).equals(cases)
    train, holdout, _ = split_cases(cases)
    assert not set(map(tuple, train[FEATURES].to_numpy())) & set(map(tuple, holdout[FEATURES].to_numpy()))


def test_report_matches_holdout_and_never_shows_mock_unaffordable_product():
    report = json.loads((ROOT / "reports/mock_suitability_metrics.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(ROOT / "reports/mock_suitability_holdout.csv")
    assert len(rows) == report["holdout_rows"]
    assert int((rows[TARGET] == rows.predicted_mock_product).sum()) == report["holdout_correct_rows"]
    assert report["holdout_budget_violation_count"] == 0
    costs = mock_costs(rows)
    for i, product in enumerate(rows.predicted_mock_product):
        if product != NO_FIT:
            assert costs[i, PRODUCTS.index(product)] <= rows.loc[i, "월예산_만원"]


def test_served_mock_result_is_labeled_and_validated():
    cases = make_cases(load_dataset().data)
    two_option_cases = 0
    for _, row in cases.iloc[:100].iterrows():
        customer = row[list(INPUTS)].to_dict()
        result = recommend_mock(customer)
        assert result["목업"] is True
        assert result[TARGET] in (*PRODUCTS, NO_FIT)
        options = result["추천순위"]
        assert len(options) <= 2
        two_option_cases += len(options) == 2
        if result[TARGET] == NO_FIT:
            assert options == []
        else:
            assert options[0]["주계약"] == result[TARGET]
        for rank, option in enumerate(options, start=1):
            assert option["순위"] == rank
            assert option["주계약"] in PRODUCTS
            assert option["추천특약"] in PRODUCT_RIDERS[option["주계약"]]
            assert option["모의_월부담_만원"] <= customer["월예산_만원"]
            assert policy_scores(pd.DataFrame([customer]))[0, PRODUCTS.index(option["주계약"])] >= 4
    assert two_option_cases > 0
    customer = cases.iloc[0][list(INPUTS)].to_dict()
    with pytest.raises(ValueError):
        recommend_mock({**customer, "월예산_만원": 1})


def test_independent_review_template_hides_policy_and_model_answers():
    review = pd.read_csv(ROOT / "reports/mock_suitability_review_template.csv")
    assert len(review) == 100
    assert TARGET not in review and "predicted_mock_product" not in review
    assert review["검토_적합상품"].isna().all()
    assert review["검토_근거"].isna().all()


def test_unseen_nine_condition_combination_still_predicts():
    from ml_model import get_mock_recommendation

    base = load_dataset().data
    customer = base.iloc[250][FEATURES].to_dict()
    customer.update({"연소득_만원": 3500, "우선보장목표": "암·건강",
                     "추가보장목표": "없음", "월예산_만원": 20})
    assert not (base[FEATURES] == pd.Series({key: customer[key] for key in FEATURES})).all(axis=1).any()
    result = recommend_mock(customer)
    assert result["목업"] and result[TARGET] in (*PRODUCTS, NO_FIT)
    assert result["추천순위"]
    assert get_mock_recommendation(customer) == result
