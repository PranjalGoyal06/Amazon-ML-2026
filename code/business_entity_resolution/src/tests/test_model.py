import pandas as pd
from src.model import BaselineMatchingModel

def test_model_training_and_sweep():
    # Dummy data
    df_features = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "T-1", "country_match": 1.0, "name_levenshtein": 0.9, "name_token_sort": 0.9, "name_jaccard": 0.9, "name_overlap": 0.9, "name_char_3gram": 0.9, "addr_levenshtein": 0.9, "addr_token_sort": 0.9, "addr_jaccard": 0.9, "addr_char_3gram": 0.9, "postal_match": 1.0, "street_num_match": 1.0, "candidate_rank": 1.0, "score_gap_to_best": 0.0, "embedding_cosine_sim": 0.9},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "T-2", "country_match": 1.0, "name_levenshtein": 0.2, "name_token_sort": 0.2, "name_jaccard": 0.1, "name_overlap": 0.1, "name_char_3gram": 0.1, "addr_levenshtein": 0.1, "addr_token_sort": 0.1, "addr_jaccard": 0.1, "addr_char_3gram": 0.1, "postal_match": 0.0, "street_num_match": 0.0, "candidate_rank": 2.0, "score_gap_to_best": 1.0, "embedding_cosine_sim": 0.1},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "T-3", "country_match": 1.0, "name_levenshtein": 0.95, "name_token_sort": 0.95, "name_jaccard": 1.0, "name_overlap": 1.0, "name_char_3gram": 1.0, "addr_levenshtein": 0.8, "addr_token_sort": 0.8, "addr_jaccard": 0.8, "addr_char_3gram": 0.8, "postal_match": 0.0, "street_num_match": 1.0, "candidate_rank": 1.0, "score_gap_to_best": 0.0, "embedding_cosine_sim": 0.85}
    ])
    
    df_labels = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "T-1", "is_match": 1},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "T-2", "is_match": 0},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "T-3", "is_match": 1}
    ])
    
    gt_dict = {
        "S1-1": {"T-1"},
        "S1-2": {"T-3"}
    }
    
    country_series = pd.Series({"S1-1": "US", "S1-2": "US"})
    
    model = BaselineMatchingModel()
    model.fit(df_features, df_labels)
    
    best_thresh, best_score, breakdown = model.sweep_threshold(
        df_features, gt_dict, country_series, thresholds=[0.3, 0.5, 0.7]
    )
    
    assert best_score > 0.0
    assert "by_country" in breakdown
    assert "US" in breakdown["by_country"]
    
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cand_path = os.path.join(tmpdir, "cand.tsv")
        match_path = os.path.join(tmpdir, "match.tsv")
        
        model.generate_submission_files(
            df_features, ["S1-1", "S1-2", "S1-3"], match_path, cand_path
        )
        
        res = pd.read_csv(match_path, sep="\t")
        assert len(res) == 3
        # S1-3 should be empty
        s3_row = res[res["source1_entity_id"] == "S1-3"]
        assert pd.isna(s3_row["matched_entity_ids"].iloc[0]) or s3_row["matched_entity_ids"].iloc[0] == ""
