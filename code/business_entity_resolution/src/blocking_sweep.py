"""
Blocking Sweep script for Amazon ML Challenge 2026.
Sweeps multiple blocking strategies and parameters over the validation benchmark subset,
logging the recall ceiling and reduction ratio for each.
"""

import json
import os
import time
from pathlib import Path
import pandas as pd
from src.blocking import MultiStrategyBlocker, evaluate_blocking_performance


def load_target_data(base_dir: Path) -> pd.DataFrame:
    """Load S2 and S3 for indexing."""
    print("Loading Target Records (S2 and S3)...")
    s2_path = base_dir / "dataset" / "train" / "train_source2.tsv"
    s3_path = base_dir / "dataset" / "train" / "train_source3.tsv"
    
    t0 = time.time()
    s2 = pd.read_csv(s2_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s3 = pd.read_csv(s3_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    df = pd.concat([s2, s3], ignore_index=True)
    print(f"Loaded {len(df):,} target records in {time.time()-t0:.2f}s")
    return df


def load_validation_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the benchmark validation split, S1 metadata, and S1 raw data."""
    val_gt_path = base_dir / "dataset" / "splits" / "val_benchmark_20k.tsv"
    val_meta_path = base_dir / "dataset" / "splits" / "val_benchmark_20k_meta.tsv"
    s1_path = base_dir / "dataset" / "train" / "train_source1.tsv"

    val_gt = pd.read_csv(val_gt_path, sep="\t")
    val_gt["matched_entity_ids"] = val_gt["matched_entity_ids"].fillna("")
    
    val_meta = pd.read_csv(val_meta_path, sep="\t")
    val_meta.rename(columns={"source1_entity_id": "entity_id"}, inplace=True)
    
    s1_all = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1_val = s1_all[s1_all["entity_id"].isin(val_gt["source1_entity_id"])]
    
    return s1_val, val_gt, val_meta


def run_sweep():
    base_dir = Path(__file__).resolve().parents[3]
    out_dir = base_dir / "code" / "business_entity_resolution" / "experiments"
    os.makedirs(out_dir, exist_ok=True)
    
    target_df = load_target_data(base_dir)
    s1_val, val_gt, val_meta = load_validation_data(base_dir)
    
    configs = [
        # Baseline: Exact Name Only
        {
            "name": "Baseline_Exact_Name",
            "params": {
                "enable_name_exact": True, "enable_name_prefix": False, "enable_name_tokens": False, 
                "enable_name_soundex": False, "enable_addr_postal_num": False, "enable_addr_street_num": False,
                "enable_vector_tfidf": False
            }
        },
        # Strategy A: Name keys only (tokens + soundex)
        {
            "name": "Strategy_A_Name_Keys",
            "params": {
                "enable_name_exact": True, "enable_name_prefix": True, "enable_name_tokens": True, 
                "enable_name_soundex": True, "enable_addr_postal_num": False, "enable_addr_street_num": False,
                "enable_vector_tfidf": False
            }
        },
        # Strategy B: Address keys only
        {
            "name": "Strategy_B_Address_Keys",
            "params": {
                "enable_name_exact": False, "enable_name_prefix": False, "enable_name_tokens": False, 
                "enable_name_soundex": False, "enable_addr_postal_num": True, "enable_addr_street_num": True,
                "enable_vector_tfidf": False
            }
        },
        # Union A + B
        {
            "name": "Union_Name_Address",
            "params": {
                "enable_name_exact": True, "enable_name_prefix": True, "enable_name_tokens": True, 
                "enable_name_soundex": True, "enable_addr_postal_num": True, "enable_addr_street_num": True,
                "enable_vector_tfidf": False, "max_candidates_per_entity": 100, "max_key_freq": 250
            }
        },
        # Full Union A + B + C (Aggressive capping)
        {
            "name": "Full_Union_Aggressive_Cap",
            "params": {
                "enable_name_exact": True, "enable_name_prefix": True, "enable_name_tokens": True, 
                "enable_name_soundex": True, "enable_addr_postal_num": True, "enable_addr_street_num": True,
                "enable_vector_tfidf": True, "vector_top_k": 5, "max_candidates_per_entity": 50, "max_key_freq": 100
            }
        },
        # Full Union A + B + C (High Recall Focus)
        {
            "name": "Full_Union_High_Recall",
            "params": {
                "enable_name_exact": True, "enable_name_prefix": True, "enable_name_tokens": True, 
                "enable_name_soundex": True, "enable_addr_postal_num": True, "enable_addr_street_num": True,
                "enable_vector_tfidf": True, "vector_top_k": 15, "max_candidates_per_entity": 200, "max_key_freq": 500
            }
        }
    ]

    results = []
    
    for cfg in configs:
        print(f"\n==============================================")
        print(f"Running Config: {cfg['name']}")
        print(f"==============================================")
        
        t0 = time.time()
        blocker = MultiStrategyBlocker(**cfg["params"])
        
        print("Indexing targets...")
        blocker.index_target_records(target_df, show_progress=False)
        pruned = blocker.prune_high_frequency_keys()
        print(f"Pruned keys: {pruned}")
        
        print("Generating candidates...")
        t1 = time.time()
        candidates = blocker.generate_candidates_for_df(s1_val, show_progress=False)
        cand_time = time.time() - t1
        print(f"Candidates generated in {cand_time:.2f}s ({cand_time/len(s1_val)*1000:.2f} ms/query)")
        
        perf = evaluate_blocking_performance(
            candidates, val_gt, val_meta, total_target_count=len(target_df)
        )
        
        print(f"Recall Ceiling: {perf['recall_ceiling']:.4f}")
        print(f"Reduction Ratio: {perf['reduction_ratio']:.8f}")
        print(f"Avg Candidates/Entity: {perf['avg_candidates_per_entity']:.2f}")
        
        res_entry = {
            "config_name": cfg["name"],
            "recall_ceiling": perf["recall_ceiling"],
            "reduction_ratio": perf["reduction_ratio"],
            "avg_candidates": perf["avg_candidates_per_entity"],
            "index_time_s": t1 - t0,
            "query_time_s": cand_time,
            "pruned_keys": pruned,
            "params": json.dumps(cfg["params"])
        }
        results.append(res_entry)
        
        # Save incrementally
        pd.DataFrame(results).to_csv(out_dir / "blocking_sweep.csv", index=False)
        
    print("\nSweep Complete! Results saved to experiments/blocking_sweep.csv")

if __name__ == "__main__":
    from typing import Tuple
    run_sweep()
