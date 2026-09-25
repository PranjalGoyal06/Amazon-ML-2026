"""
Unit tests for the evaluation metric module.
"""

import pytest
import pandas as pd
from src.evaluation import compute_entity_f_beta, compute_macro_f05


def test_perfect_match():
    # Exactly same non-singleton matches -> F_0.5 = 1.0
    true = {"S2-100", "S3-200"}
    pred = {"S2-100", "S3-200"}
    score = compute_entity_f_beta(true, pred, beta=0.5)
    assert pytest.approx(score, 1e-6) == 1.0


def test_correct_singleton():
    # True singleton predicted as empty -> F_0.5 = 1.0
    true = set()
    pred = set()
    score = compute_entity_f_beta(true, pred, beta=0.5)
    assert pytest.approx(score, 1e-6) == 1.0


def test_singleton_false_merge():
    # True singleton predicted with false match -> F_0.5 = 0.0
    true = set()
    pred = {"S2-100"}
    score = compute_entity_f_beta(true, pred, beta=0.5)
    assert pytest.approx(score, 1e-6) == 0.0


def test_missed_match():
    # True non-singleton predicted as empty -> F_0.5 = 0.0
    true = {"S2-100"}
    pred = set()
    score = compute_entity_f_beta(true, pred, beta=0.5)
    assert pytest.approx(score, 1e-6) == 0.0


def test_partial_match_problem_statement_example():
    # Example from documentation:
    # Ground truth: [S2-00047, S3-00812] (2 items)
    # Predicted: [S2-00047, S2-00193, S3-00812] (3 items)
    # Precision = 2/3, Recall = 1.0
    # F_0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = (5/6) / (7/6) = 5/7 ~ 0.7142857
    true = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f_beta(true, pred, beta=0.5)
    assert pytest.approx(score, 1e-5) == 5.0 / 7.0


def test_false_merge_penalty():
    # Compare missed match vs false merge:
    # True = {A, B}
    # Case A: Pred = {A} (missed B: P=1.0, R=0.5)
    # Case B: Pred = {A, B, C} (false merge C: P=2/3, R=1.0)
    # Case C: Pred = {A, C} (false merge C and missed B: P=0.5, R=0.5)
    true = {"S2-A", "S3-B"}
    # Pred {A}: P=1.0, R=0.5 => (1.25 * 0.5) / (0.25*1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333
    score_missed = compute_entity_f_beta(true, {"S2-A"}, beta=0.5)
    # Pred {A, B, C}: 5/7 = 0.7143
    score_false_merge = compute_entity_f_beta(true, {"S2-A", "S3-B", "S2-C"}, beta=0.5)
    assert score_missed > score_false_merge, "F_0.5 should penalize false merge more than missed match"


def test_macro_f05_dataframe():
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"}, # score 1.0
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},           # true singleton, score 1.0
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-2"},      # true non-singleton, pred empty -> 0.0
        {"source1_entity_id": "S1-4", "matched_entity_ids": ""},           # true singleton, pred non-empty -> 0.0
    ])

    pred_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-99"},
    ])

    countries = pd.Series({"S1-1": "US", "S1-2": "US", "S1-3": "India", "S1-4": "India"})

    macro, breakdown = compute_macro_f05(gt_df, pred_df, countries)
    # scores: 1.0, 1.0, 0.0, 0.0 -> mean 0.50
    assert pytest.approx(macro, 1e-6) == 0.50
    assert breakdown["n_entities"] == 4
    assert breakdown["n_singletons"] == 2
    assert breakdown["singleton_f05"] == 0.50
    assert breakdown["by_country"]["US"]["macro_f05"] == 1.0
    assert breakdown["by_country"]["India"]["macro_f05"] == 0.0
