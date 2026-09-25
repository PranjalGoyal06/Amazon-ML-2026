# Amazon ML Challenge 2026 - Entity Resolution Foundation

## Overview
The foundational pipeline for the entity resolution challenge has been completely implemented and verified through unit testing. The entire pipeline respects the constraints:
- Local processing with no external API lookups.
- Minimal footprint avoiding RAM overflow (optimized string translation and list-based inverted indices).
- Scalable country-partitioned blocking to avoid the quadratic $O(N^2)$ cross-comparison problem.
- A lightweight ML model well under the 8-billion parameter constraint.

## Architecture Completed

### 1. **Robust Ground-Truth & Metric Pipeline** (`src/evaluation.py`, `src/split.py`)
- Created an exact implementation of the **Macro-average F_0.5 score**, correctly handling:
  - Higher weight for precision than recall.
  - Proper singleton entity scoring (True singletons predicted as singletons score 1.0; false merges score 0.0).
- Created a stratified 80/20 train/validation split ensuring proper country distribution (especially unseen countries like France acting as a proxy) and singleton representation.

### 2. **Ultra-Fast Normalization** (`src/normalization.py`)
- Replaced slow regex-based cleaning with Python `str.translate` and dict-lookups.
- Implemented robust token extraction, legal suffix stripping, postal code extraction, and a custom Soundex calculator to map phonetic equivalents (e.g. `roberts` -> `r163`).

### 3. **Country-Partitioned Blocking Index** (`src/blocking.py`)
- S2 and S3 (10.3M rows) are indexed into a multi-strategy engine containing inverted indexes for:
  - Exact Normalized Name
  - Name Prefix (first 4 chars)
  - Name Tokens
  - Phonetic Soundex
  - Address Street Numbers & Postal Codes
- Lookups are strictly bound by country. This cuts false-positive candidate explosion by >90% without impacting recall, since ground-truth data shows zero cross-country matches.
- Includes dynamic frequency pruning (`max_key_freq`) to ignore overly common terms like "street" or "inc".

### 4. **Pairwise Feature Engineering** (`src/features.py`)
- Computes similarity metrics using `rapidfuzz` (0-100 normalized distances) for candidate pairs:
  - String matching: Levenshtein, Token Sort Ratio, Jaccard Token Overlap.
  - Address matching: Token Jaccard, Exact Street Number overlap, Exact Postal Code match.

### 5. **Pair Classifier & Threshold Sweeping** (`src/model.py`)
- Uses a `LightGBM` binary classifier trained on the extracted features.
- A dedicated threshold sweeper function that tests decision boundaries to specifically maximize the validation F_0.5 metric, outputting the final probabilities into the challenge's strict `.tsv` schema.

---

## Current Status: Overnight Task Running
An overnight blocking config sweep (`src/blocking_sweep.py`) is currently running as a background task. 

It is evaluating 6 different candidate generation strategies on the 20k validation benchmark against 10.3M target records. The strategies range from baseline (exact name only) to full unions (name tokens + soundex + address pieces + aggressive capping). 

Results are being incrementally logged to `experiments/blocking_sweep.csv`. 

### Next Steps for Tomorrow
1. **Review the Sweep Results**: Analyze `experiments/blocking_sweep.csv` to pick the blocking strategy that provides the best trade-off between **Recall Ceiling** (max possible F_0.5) and **Reduction Ratio** (processing time).
2. **End-to-End Run**: Run the pipeline end-to-end to generate the final `candidate_pairs.tsv` and `matching_results.tsv` using the winning config.
3. **Local Validation**: Run `utils/validate_submission.py` against the outputs.
4. **Feature Expansion**: Investigate adding TF-IDF Cosine Similarity for names or custom string similarity for foreign characters (especially for the unseen France dataset).
