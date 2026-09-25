import os
import sys
from pathlib import Path
import pandas as pd
import gc

# Add the parent directory of src to sys.path to allow running from anywhere
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.blocking import MultiStrategyBlocker
from src.features import batch_compute_features
from src.model import BaselineMatchingModel
from src.normalization import normalize_business_name, normalize_address
from src.pipeline import get_gt_dict

def main():
    base_dir = Path(__file__).resolve().parents[3]
    out_dir = base_dir / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Quickly train BaselineMatchingModel just like pipeline.py does
    print("Loading Train Target Records (S2 and S3)...")
    train_s2_path = base_dir / "dataset" / "train" / "train_source2.tsv"
    train_s3_path = base_dir / "dataset" / "train" / "train_source3.tsv"
    
    s2 = pd.read_csv(train_s2_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s3 = pd.read_csv(train_s3_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    train_target_df = pd.concat([s2, s3], ignore_index=True)
    
    val_gt_path = base_dir / "dataset" / "splits" / "val_benchmark_20k.tsv"
    s1_path = base_dir / "dataset" / "train" / "train_source1.tsv"

    # Load subset of 500 records to make sure it finishes quickly
    val_gt = pd.read_csv(val_gt_path, sep="\t").head(500)
    val_gt["matched_entity_ids"] = val_gt["matched_entity_ids"].fillna("")
    
    s1_all = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"])
    s1_val = s1_all[s1_all["entity_id"].isin(val_gt["source1_entity_id"])].copy()

    needed_targets = set()
    for _, row in val_gt.iterrows():
        m = str(row["matched_entity_ids"])
        needed_targets.update([x.strip() for x in m.split(",") if x.strip()])
    
    target_needed = train_target_df[train_target_df["entity_id"].isin(needed_targets)]
    target_noise = train_target_df.sample(n=min(10000, len(train_target_df)), random_state=42)
    train_target_df = pd.concat([target_needed, target_noise]).drop_duplicates(subset=["entity_id"])
    
    del s2, s3, target_needed, target_noise
    gc.collect()

    s1_val["norm_name"] = s1_val["business_name"].apply(normalize_business_name)
    s1_val["norm_address"] = s1_val["business_address"].apply(normalize_address)
    train_target_df["norm_name"] = train_target_df["business_name"].apply(normalize_business_name)
    train_target_df["norm_address"] = train_target_df["business_address"].apply(normalize_address)
    
    blocker_train = MultiStrategyBlocker(
        enable_name_exact=False, enable_name_prefix=False, enable_name_tokens=False, 
        enable_name_soundex=False, enable_addr_postal_num=True, enable_addr_street_num=True,
        enable_vector_tfidf=False
    )
    blocker_train.index_target_records(train_target_df, show_progress=False)
    blocker_train.prune_high_frequency_keys()
    candidates_dict = blocker_train.generate_candidates_for_df(s1_val, show_progress=False)
    
    pairs = []
    for s1_id, cands in candidates_dict.items():
        for cid in cands:
            pairs.append({"source1_entity_id": s1_id, "candidate_entity_id": cid})
    pairs_df = pd.DataFrame(pairs)
    
    df_features = batch_compute_features(pairs_df, s1_val, train_target_df)
    gt_dict = get_gt_dict(val_gt)
    
    def is_match(row):
        return int(row["candidate_entity_id"] in gt_dict.get(row["source1_entity_id"], set()))
        
    df_labels = pairs_df.copy()
    df_labels["is_match"] = df_labels.apply(is_match, axis=1)
    
    model = BaselineMatchingModel()
    model.fit(df_features, df_labels)
    country_series = s1_val.set_index("entity_id")["country"]
    best_thresh, _, _ = model.sweep_threshold(df_features, gt_dict, country_series, thresholds=[0.3, 0.4, 0.5, 0.6])

    # 2. Load the TEST targets
    print("\nLoading Test Target Records (S2 and S3)...")
    test_s2_path = base_dir / "dataset" / "test" / "test_source2.tsv"
    test_s3_path = base_dir / "dataset" / "test" / "test_source3.tsv"
    
    test_s2 = pd.read_csv(test_s2_path, sep="\t")
    test_s3 = pd.read_csv(test_s3_path, sep="\t")
    test_target_df = pd.concat([test_s2, test_s3], ignore_index=True)
    
    # Delete intermediate large dataframes to free RAM
    del test_s2, test_s3
    gc.collect()

    # 3. Initialize a new MultiStrategyBlocker
    print("Indexing test targets...")
    blocker_test = MultiStrategyBlocker(
        enable_name_exact=False, enable_name_prefix=False, enable_name_tokens=False, 
        enable_name_soundex=False, enable_addr_postal_num=True, enable_addr_street_num=True,
        enable_vector_tfidf=False
    )
    # This internally normalizes when building index
    blocker_test.index_target_records(test_target_df, show_progress=False)
    blocker_test.prune_high_frequency_keys()

    # 4. Process test_source1.tsv in chunks
    test_s1_path = base_dir / "dataset" / "test" / "test_source1.tsv"
    
    output_matching_path = out_dir / "test_matching_results.tsv"
    output_candidate_path = out_dir / "test_candidate_pairs.tsv"
    
    # Write headers
    with open(output_matching_path, "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(output_candidate_path, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        
    print("Processing test_source1.tsv in chunks...")
    for chunk_idx, chunk in enumerate(pd.read_csv(test_s1_path, sep="\t", chunksize=100000)):
        print(f"Processing chunk {chunk_idx}...")
        chunk["norm_name"] = chunk["business_name"].apply(normalize_business_name)
        chunk["norm_address"] = chunk["business_address"].apply(normalize_address)
        
        cands_dict = blocker_test.generate_candidates_for_df(chunk, show_progress=False)
        
        chunk_pairs = []
        needed_target_ids = set()
        for s1_id, cands in cands_dict.items():
            for cid in cands:
                chunk_pairs.append({"source1_entity_id": s1_id, "candidate_entity_id": cid})
                needed_target_ids.add(cid)
        
        chunk_pairs_df = pd.DataFrame(chunk_pairs)
        s1_ids_list = chunk["entity_id"].tolist()
        
        cands_out = pd.DataFrame({"source1_entity_id": s1_ids_list})
        matches_out = pd.DataFrame({"source1_entity_id": s1_ids_list})
        
        if len(chunk_pairs_df) > 0:
            cands_grouped = chunk_pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(list).reset_index()
            cands_grouped.rename(columns={"candidate_entity_id": "candidate_entity_ids"}, inplace=True)
            cands_out = cands_out.merge(cands_grouped, on="source1_entity_id", how="left")
            cands_out["candidate_entity_ids"] = cands_out["candidate_entity_ids"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
            
            # Sub-select target records needed for this chunk and normalize them
            chunk_target_df = test_target_df[test_target_df["entity_id"].isin(needed_target_ids)].copy()
            chunk_target_df["norm_name"] = chunk_target_df["business_name"].apply(normalize_business_name)
            chunk_target_df["norm_address"] = chunk_target_df["business_address"].apply(normalize_address)
            
            chunk_features = batch_compute_features(chunk_pairs_df, chunk, chunk_target_df)
            chunk_features["prob"] = model.predict_proba(chunk_features)
            
            matches = chunk_features[chunk_features["prob"] >= best_thresh]
            matches_grouped = matches.groupby("source1_entity_id")["candidate_entity_id"].apply(list).reset_index()
            matches_grouped.rename(columns={"candidate_entity_id": "matched_entity_ids"}, inplace=True)
            matches_out = matches_out.merge(matches_grouped, on="source1_entity_id", how="left")
            matches_out["matched_entity_ids"] = matches_out["matched_entity_ids"].apply(lambda x: ",".join(x) if isinstance(x, list) else "")
        else:
            cands_out["candidate_entity_ids"] = ""
            matches_out["matched_entity_ids"] = ""
            
        cands_out[["source1_entity_id", "candidate_entity_ids"]].to_csv(output_candidate_path, sep="\t", index=False, header=False, mode="a")
        matches_out[["source1_entity_id", "matched_entity_ids"]].to_csv(output_matching_path, sep="\t", index=False, header=False, mode="a")
        
        # Free memory aggressively inside loop
        del chunk, cands_dict, chunk_pairs, needed_target_ids, chunk_pairs_df, cands_out, matches_out
        if 'chunk_target_df' in locals():
            del chunk_target_df, chunk_features, matches, matches_grouped
        gc.collect()

    print("Done generating predictions on test set.")

if __name__ == "__main__":
    main()
