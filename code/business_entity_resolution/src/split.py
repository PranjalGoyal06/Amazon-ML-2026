"""
Dataset splitting module for Amazon ML Challenge 2026.
Splits ground truth into 80/20 train/validation stratified by country and singleton status.
Also creates a fast validation benchmark subset (e.g., 20k entities) for rapid tuning.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    s1_path: str,
    gt_path: str,
    output_dir: str,
    test_size: float = 0.20,
    benchmark_size: int = 20000,
    random_state: int = 42,
):
    """Create reproducible train/val splits stratified by country and singleton status."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Source 1 metadata (entity_id, country)...")
    s1_meta = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"])
    s1_meta.rename(columns={"entity_id": "source1_entity_id"}, inplace=True)

    print("Loading ground truth...")
    gt = pd.read_csv(gt_path, sep="\t")
    # Fill NA in matched_entity_ids with empty string
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    print("Merging metadata...")
    df = pd.merge(gt, s1_meta, on="source1_entity_id", how="left")
    df["country"] = df["country"].fillna("UNKNOWN")
    df["is_singleton"] = (df["matched_entity_ids"] == "")

    # Create stratification key
    df["strat_key"] = df["country"] + "__" + df["is_singleton"].astype(str)
    print(f"Stratification key distribution:\n{df['strat_key'].value_counts()}")

    print(f"Splitting 80/20 with random_state={random_state}...")
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["strat_key"],
    )

    print(f"Train size: {len(train_df):,} entities")
    print(f"Validation size: {len(val_df):,} entities")

    # Fast benchmark subset from validation
    val_bench_df, _ = train_test_split(
        val_df,
        train_size=min(benchmark_size, len(val_df)),
        random_state=random_state,
        stratify=val_df["strat_key"],
    )
    print(f"Benchmark validation size: {len(val_bench_df):,} entities")

    # Save TSV files (source1_entity_id, matched_entity_ids, country)
    cols_to_save = ["source1_entity_id", "matched_entity_ids"]
    train_path = os.path.join(output_dir, "train_gt.tsv")
    val_path = os.path.join(output_dir, "val_gt.tsv")
    val_bench_path = os.path.join(output_dir, "val_benchmark_20k.tsv")

    print(f"Saving to {output_dir}...")
    train_df[cols_to_save].to_csv(train_path, sep="\t", index=False)
    val_df[cols_to_save].to_csv(val_path, sep="\t", index=False)
    val_bench_df[cols_to_save].to_csv(val_bench_path, sep="\t", index=False)

    # Also save metadata for validation set (country mapping)
    val_meta_path = os.path.join(output_dir, "val_meta.tsv")
    val_df[["source1_entity_id", "country", "is_singleton"]].to_csv(val_meta_path, sep="\t", index=False)

    val_bench_meta_path = os.path.join(output_dir, "val_benchmark_20k_meta.tsv")
    val_bench_df[["source1_entity_id", "country", "is_singleton"]].to_csv(val_bench_meta_path, sep="\t", index=False)

    print("Splits successfully created!")


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parents[3]
    s1_p = base_dir / "dataset" / "train" / "train_source1.tsv"
    gt_p = base_dir / "dataset" / "train" / "train_ground_truth.tsv"
    out_d = base_dir / "dataset" / "splits"
    create_splits(str(s1_p), str(gt_p), str(out_d))
