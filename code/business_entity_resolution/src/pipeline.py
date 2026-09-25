import os
import time
from pathlib import Path
import pandas as pd

from src.blocking import MultiStrategyBlocker
from src.features import batch_compute_features
from src.model import BaselineMatchingModel
from src.normalization import normalize_business_name, normalize_address

def get_gt_dict(val_gt):
    gt_dict = {}
    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m = str(row["matched_entity_ids"])
        gt_dict[s1_id] = set(x.strip() for x in m.split(",") if x.strip())
    return gt_dict

def main():
    base_dir = Path(__file__).resolve().parents[3]
    out_dir = base_dir / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading Target Records (S2 and S3)...")
    s2_path = base_dir / "dataset" / "train" / "train_source2.tsv"
    s3_path = base_dir / "dataset" / "train" / "train_source3.tsv"
    
    s2 = pd.read_csv(s2_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s3 = pd.read_csv(s3_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    target_df = pd.concat([s2, s3], ignore_index=True)
    
    print("Loading S1 and Validation Benchmark...")
    val_gt_path = base_dir / "dataset" / "splits" / "val_benchmark_20k.tsv"
    s1_path = base_dir / "dataset" / "train" / "train_source1.tsv"

    # Load subset of 500 records to make sure it finishes quickly
    val_gt = pd.read_csv(val_gt_path, sep="\t").head(500)
    val_gt["matched_entity_ids"] = val_gt["matched_entity_ids"].fillna("")
    
    s1_all = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1_val = s1_all[s1_all["entity_id"].isin(val_gt["source1_entity_id"])].copy()

    print("Pre-normalizing columns for fast feature computation...")
    s1_val["norm_name"] = s1_val["business_name"].apply(normalize_business_name)
    s1_val["norm_address"] = s1_val["business_address"].apply(normalize_address)
    target_df["norm_name"] = target_df["business_name"].apply(normalize_business_name)
    target_df["norm_address"] = target_df["business_address"].apply(normalize_address)
    
    print("Initializing MultiStrategyBlocker (Strategy B params)...")
    blocker = MultiStrategyBlocker(
        enable_name_exact=False, enable_name_prefix=False, enable_name_tokens=False, 
        enable_name_soundex=False, enable_addr_postal_num=True, enable_addr_street_num=True,
        enable_vector_tfidf=False
    )
    
    print("Indexing target records...")
    blocker.index_target_records(target_df, show_progress=False)
    blocker.prune_high_frequency_keys()
    
    print("Generating candidates...")
    candidates_dict = blocker.generate_candidates_for_df(s1_val, show_progress=False)
    
    pairs = []
    for s1_id, cands in candidates_dict.items():
        for cid in cands:
            pairs.append({"source1_entity_id": s1_id, "candidate_entity_id": cid})
    pairs_df = pd.DataFrame(pairs)
    
    if len(pairs_df) == 0:
        print("No candidates found, exiting.")
        return
        
    print(f"Generated {len(pairs_df)} candidate pairs. Computing features...")
    df_features = batch_compute_features(pairs_df, s1_val, target_df)
    
    print("Generating labels...")
    gt_dict = get_gt_dict(val_gt)
    
    def is_match(row):
        return int(row["candidate_entity_id"] in gt_dict.get(row["source1_entity_id"], set()))
        
    df_labels = pairs_df.copy()
    df_labels["is_match"] = df_labels.apply(is_match, axis=1)
    
    print("Training BaselineMatchingModel...")
    model = BaselineMatchingModel()
    model.fit(df_features, df_labels)
    
    print("Sweeping threshold...")
    country_series = s1_val.set_index("entity_id")["country"]
    best_thresh, best_score, best_breakdown = model.sweep_threshold(
        df_features, gt_dict, country_series, thresholds=[0.3, 0.4, 0.5, 0.6]
    )
    
    print("Exporting results...")
    model.generate_submission_files(
        df_features=df_features,
        s1_ids_list=val_gt["source1_entity_id"].tolist(),
        output_matching_path=str(out_dir / "matching_results.tsv"),
        output_candidate_path=str(out_dir / "candidate_pairs.tsv"),
        threshold=best_thresh
    )
    print("Done!")

if __name__ == "__main__":
    main()
