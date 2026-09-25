"""
Feature engineering module for Amazon ML Challenge 2026.
Computes pair-level similarities between a Source 1 entity and a candidate (Source 2/3).
"""

from typing import Dict, Optional, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

from src.normalization import (
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_street_numbers,
)

# Global embedding model (lazy load)
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence embedding model (all-MiniLM-L6-v2)...")
            _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("sentence_transformers not found. Embedding features will be 0.0.")
            _embed_model = False
    return _embed_model


def safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def token_jaccard(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def token_overlap_coefficient(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2))


def char_ngram_jaccard(str1: str, str2: str, n: int = 3) -> float:
    if not str1 and not str2:
        return 0.0
    if not str1 or not str2:
        return 0.0
    s1 = f" {str1} "
    s2 = f" {str2} "
    set1 = {s1[i:i+n] for i in range(len(s1)-n+1)}
    set2 = {s2[i:i+n] for i in range(len(s2)-n+1)}
    return token_jaccard(set1, set2)



def compute_pair_features(
    s1_addr: str, s1_country: str,
    t_addr: str, t_country: str,
    n1_name: str, n1_addr: str,
    n2_name: str, n2_addr: str,
) -> Dict[str, float]:
    """Compute feature vector for a single pair of records."""
    features = {}

    # 1. Country Match
    features["country_match"] = 1.0 if str(s1_country).strip() == str(t_country).strip() else 0.0

    # 2. Name Similarities
    if not n1_name or not n2_name:
        features["name_levenshtein"] = 0.0
        features["name_token_sort"] = 0.0
        features["name_jaccard"] = 0.0
        features["name_overlap"] = 0.0
        features["name_char_3gram"] = 0.0
    else:
        # Rapidfuzz distances (returns 0-100 score)
        features["name_levenshtein"] = fuzz.ratio(n1_name, n2_name) / 100.0
        features["name_token_sort"] = fuzz.token_sort_ratio(n1_name, n2_name) / 100.0
        
        tok1 = set(n1_name.split())
        tok2 = set(n2_name.split())
        features["name_jaccard"] = token_jaccard(tok1, tok2)
        features["name_overlap"] = token_overlap_coefficient(tok1, tok2)
        features["name_char_3gram"] = char_ngram_jaccard(n1_name, n2_name, 3)

    # 3. Address Similarities
    if not n1_addr or not n2_addr:
        features["addr_levenshtein"] = 0.0
        features["addr_token_sort"] = 0.0
        features["addr_jaccard"] = 0.0
        features["addr_char_3gram"] = 0.0
    else:
        features["addr_levenshtein"] = fuzz.ratio(n1_addr, n2_addr) / 100.0
        features["addr_token_sort"] = fuzz.token_sort_ratio(n1_addr, n2_addr) / 100.0
        
        tok1_a = set(n1_addr.split())
        tok2_a = set(n2_addr.split())
        features["addr_jaccard"] = token_jaccard(tok1_a, tok2_a)
        features["addr_char_3gram"] = char_ngram_jaccard(n1_addr, n2_addr, 3)

    # 4. Exact Postal Code and Street Number Matches
    p1 = extract_postal_code(s1_addr, s1_country)
    p2 = extract_postal_code(t_addr, t_country)
    features["postal_match"] = 1.0 if p1 and p2 and p1 == p2 else 0.0
    
    num1 = set(extract_street_numbers(s1_addr))
    num2 = set(extract_street_numbers(t_addr))
    features["street_num_match"] = 1.0 if num1 and num2 and (num1 & num2) else 0.0

    return features

def batch_compute_features(
    pairs_df: pd.DataFrame, 
    s1_df: pd.DataFrame, 
    target_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute features for a dataframe of pairs (source1_entity_id, candidate_entity_id)."""
    # Merge S1 and Target data
    merged = pairs_df.merge(s1_df, left_on="source1_entity_id", right_on="entity_id", suffixes=("", "_drop"))
    merged = merged.merge(target_df, left_on="candidate_entity_id", right_on="entity_id", suffixes=("_s1", "_tgt"))
    
    # Compute base string/token features row by row
    records = []
    for row in merged.itertuples(index=False):
        feats = compute_pair_features(
            s1_addr=row.business_address_s1 if pd.notna(row.business_address_s1) else "",
            s1_country=row.country_s1 if pd.notna(row.country_s1) else "",
            t_addr=row.business_address_tgt if pd.notna(row.business_address_tgt) else "",
            t_country=row.country_tgt if pd.notna(row.country_tgt) else "",
            n1_name=row.norm_name_s1 if pd.notna(row.norm_name_s1) else "",
            n1_addr=row.norm_address_s1 if pd.notna(row.norm_address_s1) else "",
            n2_name=row.norm_name_tgt if pd.notna(row.norm_name_tgt) else "",
            n2_addr=row.norm_address_tgt if pd.notna(row.norm_address_tgt) else "",
        )
        feats["source1_entity_id"] = row.source1_entity_id
        feats["candidate_entity_id"] = row.candidate_entity_id
        records.append(feats)
        
    df_features = pd.DataFrame(records)
    
    # Optional: Compute Sentence Embeddings (Batched)
    embed_model = get_embed_model()
    if embed_model:
        # Prepare combined strings for S1 and Target
        s1_texts = merged["norm_name_s1"].fillna("") + " " + merged["norm_address_s1"].fillna("")
        tgt_texts = merged["norm_name_tgt"].fillna("") + " " + merged["norm_address_tgt"].fillna("")
        
        # We only need to embed unique S1 and unique Target texts to save time
        unique_s1 = pd.DataFrame({"text": s1_texts.unique()})
        unique_tgt = pd.DataFrame({"text": tgt_texts.unique()})
        
        print(f"Embedding {len(unique_s1)} unique S1 records and {len(unique_tgt)} target records...")
        s1_emb = embed_model.encode(unique_s1["text"].tolist(), normalize_embeddings=True)
        tgt_emb = embed_model.encode(unique_tgt["text"].tolist(), normalize_embeddings=True)
        
        # Map back to rows
        s1_emb_dict = dict(zip(unique_s1["text"], s1_emb))
        tgt_emb_dict = dict(zip(unique_tgt["text"], tgt_emb))
        
        cos_sims = []
        for s1_txt, tgt_txt in zip(s1_texts, tgt_texts):
            v1 = s1_emb_dict[s1_txt]
            v2 = tgt_emb_dict[tgt_txt]
            cos_sims.append(float(np.dot(v1, v2)))
        df_features["embedding_cosine_sim"] = cos_sims
    else:
        df_features["embedding_cosine_sim"] = 0.0

    # Relative Features (Rank and Score Gap)
    # We use name_char_3gram as a proxy for "base score" to compute relative ranking before the ML model
    if "name_char_3gram" in df_features.columns:
        df_features["temp_base_score"] = df_features["name_char_3gram"] + df_features["addr_char_3gram"] + df_features["embedding_cosine_sim"]
        
        # Rank features
        df_features["candidate_rank"] = df_features.groupby("source1_entity_id")["temp_base_score"].rank(method="dense", ascending=False)
        
        # Score gap to next best
        df_features["score_gap_to_best"] = df_features.groupby("source1_entity_id")["temp_base_score"].transform(lambda x: x.max() - x)
        df_features.drop(columns=["temp_base_score"], inplace=True)
    
    return df_features
