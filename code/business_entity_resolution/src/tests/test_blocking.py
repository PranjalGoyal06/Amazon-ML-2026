"""
Unit tests for the multi-strategy blocking module.
"""

import pytest
import pandas as pd
from src.blocking import MultiStrategyBlocker, evaluate_blocking_performance


def test_blocking_strategies_and_union():
    # Construct synthetic target records (S2 and S3)
    target_df = pd.DataFrame([
        {
            "entity_id": "S2-101",
            "business_name": "Acme Widgets Private Limited",
            "business_address": "123 Main Street, Suite 100, New York, NY 10001",
            "country": "US",
        },
        {
            "entity_id": "S2-102",
            "business_name": "Apex Dental Clinic",
            "business_address": "45 Park Avenue, Bangalore, Karnataka 560001",
            "country": "India",
        },
        {
            "entity_id": "S3-201",
            "business_name": "Akme Widgits Inc", # Typo in name, soundex match
            "business_address": "123 Main St, New York, NY",
            "country": "US",
        },
        {
            "entity_id": "S3-202",
            "business_name": "Dr Apex Dental Center", # Prefix variation
            "business_address": "Opposite Metro Pillar 45, Bangalore 560001",
            "country": "India",
        },
    ])

    blocker = MultiStrategyBlocker(
        name_token_min_len=3,
        enable_name_exact=True,
        enable_name_prefix=True,
        enable_name_tokens=True,
        enable_name_soundex=True,
        enable_addr_postal_num=True,
        enable_addr_street_num=True,
        enable_vector_tfidf=True,
    )

    blocker.index_target_records(target_df, show_progress=False)
    blocker.prune_high_frequency_keys()

    # Query 1: Exact / Token match on US Acme
    cands_1 = blocker.generate_candidates_for_record(
        raw_name="Acme Widgets LLC",
        raw_addr="123 Main St, NY 10001",
        country="US",
    )
    # Should catch S2-101 (exact token 'acme' + 'widgets' + street num 123 + postal 10001)
    # Should catch S3-201 via address (123 + main) and/or soundex of widgets/widgits
    assert "S2-101" in cands_1
    assert "S3-201" in cands_1
    # Must never include India records
    assert "S2-102" not in cands_1
    assert "S3-202" not in cands_1

    # Query 2: India record with soundex / address / token match
    cands_2 = blocker.generate_candidates_for_record(
        raw_name="Apex Dental Clinic Pvt Ltd",
        raw_addr="45 Park Ave, Bangalore 560001",
        country="India",
    )
    assert "S2-102" in cands_2
    assert "S3-202" in cands_2
    assert "S2-101" not in cands_2


def test_evaluate_blocking_performance():
    cand_dict = {
        "S1-1": {"S2-101", "S3-201"},
        "S1-2": {"S2-999"}, # false candidates
        "S1-3": set(),      # empty candidates
    }

    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-101,S3-201"}, # 2 true, both caught
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-102"},        # 1 true, missed
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},              # singleton
    ])

    s1_meta = pd.DataFrame([
        {"entity_id": "S1-1", "country": "US"},
        {"entity_id": "S1-2", "country": "India"},
        {"entity_id": "S1-3", "country": "US"},
    ])

    perf = evaluate_blocking_performance(cand_dict, gt_df, s1_meta, total_target_count=100)
    # Total true matches = 3 (2 from S1-1, 1 from S1-2). Recalled = 2.
    assert pytest.approx(perf["recall_ceiling"], 1e-5) == 2.0 / 3.0
    assert perf["total_candidate_pairs"] == 3 # 2 + 1 + 0
    assert perf["reduction_ratio"] > 0.95
