import pandas as pd
from src.features import compute_pair_features, batch_compute_features

def test_compute_pair_features():
    feats = compute_pair_features(
        s1_addr="410 Terry Ave N", s1_country="US",
        t_addr="410 Terry Avenue North", t_country="US",
        n1_name="amazon services llc", n1_addr="410 terry ave n",
        n2_name="amazon llc", n2_addr="410 terry avenue north"
    )
    assert feats["country_match"] == 1.0
    assert feats["name_levenshtein"] > 0.6
    assert feats["addr_levenshtein"] > 0.6
    assert feats["postal_match"] == 0.0 # None found
    assert feats["street_num_match"] == 1.0 # 410 found
    
def test_batch_compute_features():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-001", "business_name": "Apple Inc.", "business_address": "1 Apple Park", "country": "US", "norm_name": "apple inc", "norm_address": "1 apple park"}
    ])
    tgt_df = pd.DataFrame([
        {"entity_id": "S2-001", "business_name": "Apple", "business_address": "1 Apple Pk Way", "country": "US", "norm_name": "apple", "norm_address": "1 apple pk way"}
    ])
    pairs = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001"}
    ])
    
    feats = batch_compute_features(pairs, s1_df, tgt_df)
    assert len(feats) == 1
    assert "source1_entity_id" in feats.columns
    assert "candidate_entity_id" in feats.columns
    assert "name_levenshtein" in feats.columns
