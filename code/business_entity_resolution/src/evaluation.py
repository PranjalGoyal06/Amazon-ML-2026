"""
Evaluation module for Amazon ML Challenge 2026: Business Entity Resolution.

Implements the official competition metric:
Macro-averaged F_0.5 per Source 1 entity, including exact singleton handling.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


def compute_entity_f_beta(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5,
) -> float:
    """Compute F_beta for a single Source 1 entity.

    Parameters
    ----------
    true_matches : Set[str]
        Set of ground-truth matched entity IDs (S2-*/S3-*). Empty set for singletons.
    pred_matches : Set[str]
        Set of predicted matched entity IDs (S2-*/S3-*). Empty set for predicted singletons.
    beta : float
        Beta parameter for F-score. Default is 0.5 (precision weighted 2x recall).

    Returns
    -------
    float
        F_beta score in [0.0, 1.0].
    """
    is_true_singleton = len(true_matches) == 0
    is_pred_empty = len(pred_matches) == 0

    if is_true_singleton:
        # Correctly predicting empty list on true singleton scores 1.0;
        # Any false match on a singleton scores 0.0.
        return 1.0 if is_pred_empty else 0.0

    # True non-singleton:
    if is_pred_empty:
        # Missed all matches
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def compute_macro_f05(
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    predictions: Union[pd.DataFrame, Dict[str, Set[str]]],
    country_series: Optional[pd.Series] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute macro-averaged F_0.5 across all Source 1 entities in ground truth.

    Parameters
    ----------
    ground_truth : DataFrame or dict
        If DataFrame, must contain columns ['source1_entity_id', 'matched_entity_ids'].
        If dict, mapping from source1_entity_id to set of true matched IDs.
    predictions : DataFrame or dict
        If DataFrame, must contain columns ['source1_entity_id', 'matched_entity_ids'].
        If dict, mapping from source1_entity_id to set of predicted matched IDs.
    country_series : pd.Series, optional
        Series indexed by source1_entity_id mapping to country name. If provided,
        per-country F_0.5 breakdown is returned in the second element.

    Returns
    -------
    overall_f05 : float
        Macro-average F_0.5 across all entities in ground_truth.
    breakdown : dict
        Dictionary of metrics including per-country scores and singleton stats.
    """
    # Normalize ground truth to dict of sets
    if isinstance(ground_truth, pd.DataFrame):
        gt_dict = {}
        for _, row in ground_truth.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matches = row.get("matched_entity_ids", "")
            if pd.isna(raw_matches) or not str(raw_matches).strip():
                gt_dict[s1_id] = set()
            else:
                gt_dict[s1_id] = {m.strip() for m in str(raw_matches).split(",") if m.strip()}
    else:
        gt_dict = ground_truth

    # Normalize predictions to dict of sets
    if isinstance(predictions, pd.DataFrame):
        pred_dict = {}
        for _, row in predictions.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matches = row.get("matched_entity_ids", "")
            if pd.isna(raw_matches) or not str(raw_matches).strip():
                pred_dict[s1_id] = set()
            else:
                pred_dict[s1_id] = {m.strip() for m in str(raw_matches).split(",") if m.strip()}
    else:
        pred_dict = predictions

    scores = []
    country_scores: Dict[str, List[float]] = {}
    singleton_scores: List[float] = []
    non_singleton_scores: List[float] = []

    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        score = compute_entity_f_beta(true_set, pred_set, beta=0.5)
        scores.append(score)

        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

        if country_series is not None and s1_id in country_series:
            cntry = country_series[s1_id]
            if cntry not in country_scores:
                country_scores[cntry] = []
            country_scores[cntry].append(score)

    overall_f05 = float(np.mean(scores)) if scores else 0.0

    breakdown = {
        "n_entities": len(scores),
        "macro_f05": overall_f05,
        "n_singletons": len(singleton_scores),
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "n_non_singletons": len(non_singleton_scores),
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
    }

    if country_scores:
        breakdown["by_country"] = {
            cntry: {
                "n_entities": len(cntry_s),
                "macro_f05": float(np.mean(cntry_s)),
            }
            for cntry, cntry_s in country_scores.items()
        }

    return overall_f05, breakdown
