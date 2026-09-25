"""
Baseline matching model module for Amazon ML Challenge 2026.
Trains a LightGBM classifier on pair-level features and sweeps decision thresholds
to maximize the exact competition F_0.5 macro metric.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Set, Optional
from tqdm import tqdm

from src.evaluation import compute_macro_f05


class BaselineMatchingModel:
    def __init__(self, random_state: int = 42):
        # Extremely lightweight tree model (well under 8B parameters, Apache 2.0 / MIT compatible)
        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
        )
        self.features = [
            "country_match",
            "name_levenshtein", "name_token_sort", "name_jaccard", "name_overlap", "name_char_3gram",
            "addr_levenshtein", "addr_token_sort", "addr_jaccard", "addr_char_3gram",
            "postal_match", "street_num_match",
            "candidate_rank", "score_gap_to_best", "embedding_cosine_sim"
        ]
        self.best_threshold = 0.5

    def fit(self, df_features: pd.DataFrame, df_labels: pd.DataFrame):
        """Train the LightGBM classifier."""
        print("Preparing training data...")
        
        # Merge labels
        # df_labels should have: source1_entity_id, candidate_entity_id, is_match (1/0)
        df_train = df_features.merge(
            df_labels, on=["source1_entity_id", "candidate_entity_id"], how="inner"
        )
        
        X = df_train[self.features]
        y = df_train["is_match"]
        
        print(f"Training LightGBM on {len(X):,} pairs (Positives: {y.sum():,})...")
        self.model.fit(X, y)
        print("Training complete.")

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        X = df_features[self.features]
        return self.model.predict_proba(X)[:, 1]

    def sweep_threshold(
        self,
        df_features: pd.DataFrame,
        gt_dict: Dict[str, Set[str]],
        country_series: pd.Series,
        thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    ) -> Tuple[float, float, Dict]:
        """Sweep decision threshold to maximize macro F_0.5 score."""
        print("Predicting probabilities for validation pairs...")
        probs = self.predict_proba(df_features)
        df_preds = df_features[["source1_entity_id", "candidate_entity_id"]].copy()
        df_preds["prob"] = probs

        best_score = -1.0
        best_thresh = 0.5
        best_breakdown = {}

        print(f"Sweeping thresholds: {thresholds}")
        for thresh in thresholds:
            # Filter matches above threshold
            matches = df_preds[df_preds["prob"] >= thresh]
            
            # Group by S1 entity to create predicted lists
            pred_dict = (
                matches.groupby("source1_entity_id")["candidate_entity_id"]
                .apply(set)
                .to_dict()
            )
            
            # Evaluate using exact competition metric
            score, breakdown = compute_macro_f05(gt_dict, pred_dict, country_series)
            
            print(f"  Threshold {thresh:.2f} -> F_0.5: {score:.5f}")
            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_breakdown = breakdown

        self.best_threshold = best_thresh
        print(f"\nBest Threshold: {self.best_threshold:.2f} with F_0.5: {best_score:.5f}")
        return best_thresh, best_score, best_breakdown

    def generate_submission_files(
        self, 
        df_features: pd.DataFrame, 
        s1_ids_list: List[str],
        output_matching_path: str,
        output_candidate_path: str,
        threshold: Optional[float] = None
    ):
        """Generate final matching_results.tsv and candidate_pairs.tsv."""
        thresh = threshold if threshold is not None else self.best_threshold
        
        # 1. Candidate Pairs TSV
        # Format: source1_entity_id \t candidate_entity_ids (comma separated)
        cands = df_features.groupby("source1_entity_id")["candidate_entity_id"].apply(list).reset_index()
        # Ensure all S1 IDs are present (even empty ones)
        cands = pd.DataFrame({"source1_entity_id": s1_ids_list}).merge(cands, on="source1_entity_id", how="left")
        cands["candidate_entity_ids"] = cands["candidate_entity_id"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
        cands[["source1_entity_id", "candidate_entity_ids"]].to_csv(output_candidate_path, sep="\t", index=False)
        
        # 2. Matching Results TSV
        # Predict probs
        df_features["prob"] = self.predict_proba(df_features)
        matches = df_features[df_features["prob"] >= thresh]
        
        matched_grouped = matches.groupby("source1_entity_id")["candidate_entity_id"].apply(list).reset_index()
        matched_grouped.rename(columns={"candidate_entity_id": "matched_entity_ids"}, inplace=True)
        
        # Ensure all S1 IDs are present
        results = pd.DataFrame({"source1_entity_id": s1_ids_list}).merge(matched_grouped, on="source1_entity_id", how="left")
        results["matched_entity_ids"] = results["matched_entity_ids"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
        results[["source1_entity_id", "matched_entity_ids"]].to_csv(output_matching_path, sep="\t", index=False)
        
        print(f"Generated {output_candidate_path} and {output_matching_path}")
