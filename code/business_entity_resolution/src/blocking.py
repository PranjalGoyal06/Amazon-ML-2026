"""
Multi-strategy candidate generation (blocking) module.
Amazon ML Challenge 2026: Business Entity Resolution.

Implements three independent blocking strategies and their union:
  Strategy A: Name token, prefix, and phonetic (Soundex) key blocking
  Strategy B: Address-based blocking (street number + postal code / street token)
  Strategy C: Vector-based / TF-IDF token overlap blocking over combined text
"""

from collections import Counter, defaultdict
import re
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.normalization import (
    clean_text,
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    extract_street_numbers,
    compute_soundex,
)


class MultiStrategyBlocker:
    """Multi-strategy candidate generation partitioned by country."""

    def __init__(
        self,
        name_token_min_len: int = 3,
        name_prefix_len: int = 4,
        max_key_freq: int = 250,
        enable_name_exact: bool = True,
        enable_name_prefix: bool = True,
        enable_name_tokens: bool = True,
        enable_name_soundex: bool = True,
        enable_addr_postal_num: bool = True,
        enable_addr_street_num: bool = True,
        enable_vector_tfidf: bool = True,
        vector_top_k: int = 15,
        vector_min_sim: float = 0.35,
        max_candidates_per_entity: Optional[int] = 100,
    ):
        self.name_token_min_len = name_token_min_len
        self.name_prefix_len = name_prefix_len
        self.max_key_freq = max_key_freq
        self.enable_name_exact = enable_name_exact
        self.enable_name_prefix = enable_name_prefix
        self.enable_name_tokens = enable_name_tokens
        self.enable_name_soundex = enable_name_soundex
        self.enable_addr_postal_num = enable_addr_postal_num
        self.enable_addr_street_num = enable_addr_street_num
        self.enable_vector_tfidf = enable_vector_tfidf
        self.vector_top_k = vector_top_k
        self.vector_min_sim = vector_min_sim
        self.max_candidates_per_entity = max_candidates_per_entity

        # Inverted index: country -> key_type -> key_val -> list of target_entity_ids
        self.indexes: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Word document frequency for TF-IDF / vector weighting: country -> token -> count
        self.doc_freq: Dict[str, Counter] = defaultdict(Counter)
        self.total_docs: Dict[str, int] = defaultdict(int)
        
        # High frequency pruned keys: country -> set(pruned_keys)
        self.pruned_keys: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        
        # Target entity count by country
        self.country_target_counts: Counter = Counter()

    def _extract_name_keys(self, norm_name: str) -> Dict[str, List[str]]:
        keys = defaultdict(list)
        if not norm_name:
            return keys

        # 1. Exact name
        if self.enable_name_exact:
            keys["exact"].append(norm_name)

        # 2. Prefix of name without spaces
        no_space = norm_name.replace(" ", "")
        if self.enable_name_prefix and len(no_space) >= self.name_prefix_len:
            keys["prefix"].append(no_space[: self.name_prefix_len])

        tokens = norm_name.split()
        for tok in tokens:
            if len(tok) >= self.name_token_min_len:
                if self.enable_name_tokens:
                    keys["token"].append(tok)
                if self.enable_name_soundex:
                    snd = compute_soundex(tok)
                    if snd:
                        keys["soundex"].append(snd)

        return keys

    def _extract_address_keys(self, address: str, country: str) -> Dict[str, List[str]]:
        keys = defaultdict(list)
        if not address:
            return keys

        numbers = extract_street_numbers(address)
        postal = extract_postal_code(address, country)
        norm_addr = normalize_address(address)
        addr_tokens = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit()]

        # Street number + postal code
        if self.enable_addr_postal_num and postal and numbers:
            for num in numbers[:2]:
                keys["num_postal"].append(f"{num}___{postal}")

        # Street number + street name token
        if self.enable_addr_street_num and numbers and addr_tokens:
            for num in numbers[:2]:
                for tok in addr_tokens[:3]:
                    keys["num_token"].append(f"{num}___{tok}")

        return keys

    def _extract_combined_tokens(self, norm_name: str, norm_addr: str) -> Set[str]:
        """Extract significant distinctive tokens from combined name and address."""
        tokens = set()
        for t in norm_name.split():
            if len(t) >= 3:
                tokens.add(t)
        for t in norm_addr.split():
            if len(t) >= 4 and not t.isdigit():
                tokens.add(t)
        return tokens

    def index_target_records(self, df: pd.DataFrame, show_progress: bool = True):
        """Build candidate inverted indexes from target DataFrame (Source 2 and/or Source 3)."""
        iterator = df.itertuples(index=False)
        if show_progress:
            iterator = tqdm(iterator, total=len(df), desc="Indexing target records")

        for row in iterator:
            # Columns: entity_id, business_name, business_address, country
            eid = row.entity_id
            cntry = str(row.country).strip() if pd.notna(row.country) else "UNKNOWN"
            self.country_target_counts[cntry] += 1
            self.total_docs[cntry] += 1

            raw_name = row.business_name if pd.notna(row.business_name) else ""
            raw_addr = row.business_address if pd.notna(row.business_address) else ""

            norm_name = normalize_business_name(raw_name)
            norm_addr = normalize_address(raw_addr)

            # Strategy A: Name keys
            name_keys = self._extract_name_keys(norm_name)
            for k_type, k_vals in name_keys.items():
                for k_val in k_vals:
                    self.indexes[cntry][k_type][k_val].append(eid)

            # Strategy B: Address keys
            addr_keys = self._extract_address_keys(raw_addr, cntry)
            for k_type, k_vals in addr_keys.items():
                for k_val in k_vals:
                    self.indexes[cntry][k_type][k_val].append(eid)

            # Strategy C: Vector tokens
            if self.enable_vector_tfidf:
                comb_tokens = self._extract_combined_tokens(norm_name, norm_addr)
                for tok in comb_tokens:
                    self.doc_freq[cntry][tok] += 1
                    self.indexes[cntry]["vector_tok"][tok].append(eid)

    def prune_high_frequency_keys(self):
        """Prune keys with posting list length > max_key_freq to prevent false merge explosion."""
        total_pruned = 0
        for cntry, type_dict in self.indexes.items():
            for k_type, key_dict in type_dict.items():
                # Do not prune high-precision keys
                if k_type in ("exact", "num_postal"):
                    continue
                for k_val, posting in list(key_dict.items()):
                    if len(posting) > self.max_key_freq:
                        # Prune or truncate
                        self.pruned_keys[cntry].add((k_type, k_val))
                        # Completely drop the key as frequent keys are noise
                        del key_dict[k_val]
                        total_pruned += 1
        return total_pruned

    def generate_candidates_for_record(
        self,
        raw_name: str,
        raw_addr: str,
        country: str,
    ) -> Set[str]:
        """Generate candidate target entity IDs for a single Source 1 entity."""
        cntry = str(country).strip()
        cntry_idx = self.indexes.get(cntry)
        if not cntry_idx:
            return set()

        candidates: Counter = Counter()
        
        # Weights for different key types to prioritize strong signals
        key_weights = {
            "exact": 100.0,
            "num_postal": 50.0,
            "prefix": 10.0,
            "num_token": 10.0,
            "token": 1.0,
            "soundex": 1.0,
        }

        norm_name = normalize_business_name(raw_name)
        norm_addr = normalize_address(raw_addr)

        # Strategy A: Name keys
        name_keys = self._extract_name_keys(norm_name)
        for k_type, k_vals in name_keys.items():
            k_map = cntry_idx.get(k_type)
            if not k_map:
                continue
            weight = key_weights.get(k_type, 1.0)
            for val in k_vals:
                postings = k_map.get(val)
                if postings:
                    for p in postings:
                        candidates[p] += weight

        # Strategy B: Address keys
        addr_keys = self._extract_address_keys(raw_addr, cntry)
        for k_type, k_vals in addr_keys.items():
            k_map = cntry_idx.get(k_type)
            if not k_map:
                continue
            weight = key_weights.get(k_type, 1.0)
            for val in k_vals:
                postings = k_map.get(val)
                if postings:
                    for p in postings:
                        candidates[p] += weight

        # Strategy C: Vector TF-IDF / Token Overlap
        if self.enable_vector_tfidf:
            comb_tokens = self._extract_combined_tokens(norm_name, norm_addr)
            if comb_tokens:
                n_docs = max(1, self.total_docs.get(cntry, 1))
                df_map = self.doc_freq.get(cntry, {})
                
                # Compute IDF weights for query tokens: idf = log((N + 1) / (df + 1)) + 1
                token_weights = {
                    tok: np.log((n_docs + 1.0) / (df_map.get(tok, 1) + 1.0)) + 1.0
                    for tok in comb_tokens
                }
                
                # Pick top 4 most informative tokens
                top_tokens = sorted(comb_tokens, key=lambda t: token_weights[t], reverse=True)[:4]
                vec_candidates: Counter = Counter()
                v_map = cntry_idx.get("vector_tok", {})
                
                for tok in top_tokens:
                    posting = v_map.get(tok)
                    if posting:
                        w = token_weights[tok]
                        for tid in posting:
                            vec_candidates[tid] += w

                # Add top vector candidates
                if vec_candidates:
                    for tid, score in vec_candidates.most_common(self.vector_top_k):
                        # Vector scores can be high (e.g. 5.0 - 15.0), scale them reasonably
                        candidates[tid] += score

        if self.max_candidates_per_entity and len(candidates) > self.max_candidates_per_entity:
            # Deterministic truncation if exceeded, top N by score
            candidates = set(c for c, _ in candidates.most_common(self.max_candidates_per_entity))
        else:
            candidates = set(candidates.keys())

        return candidates

    def generate_candidates_for_df(
        self,
        s1_df: pd.DataFrame,
        show_progress: bool = True,
    ) -> Dict[str, Set[str]]:
        """Generate candidates for all Source 1 records in DataFrame."""
        candidate_dict: Dict[str, Set[str]] = {}
        iterator = s1_df.itertuples(index=False)
        if show_progress:
            iterator = tqdm(iterator, total=len(s1_df), desc="Generating candidates")

        for row in iterator:
            s1_id = row.entity_id
            raw_name = row.business_name if pd.notna(row.business_name) else ""
            raw_addr = row.business_address if pd.notna(row.business_address) else ""
            cntry = str(row.country).strip() if pd.notna(row.country) else "UNKNOWN"

            cand_set = self.generate_candidates_for_record(raw_name, raw_addr, cntry)
            candidate_dict[s1_id] = cand_set

        return candidate_dict


def evaluate_blocking_performance(
    candidate_dict: Dict[str, Set[str]],
    ground_truth_df: pd.DataFrame,
    s1_meta_df: pd.DataFrame,
    total_target_count: int,
) -> Dict[str, any]:
    """Measure recall ceiling, candidate pair volume, and reduction ratio.

    Parameters
    ----------
    candidate_dict : Dict[str, Set[str]]
        source1_entity_id -> set of candidate entity IDs
    ground_truth_df : DataFrame
        Must contain ['source1_entity_id', 'matched_entity_ids']
    s1_meta_df : DataFrame
        Must contain ['entity_id', 'country']
    total_target_count : int
        Total number of target records in the search pool (S2 + S3)
    """
    s1_country_map = dict(zip(s1_meta_df["entity_id"], s1_meta_df["country"]))
    
    total_true_matches = 0
    recalled_matches = 0
    total_candidate_pairs = 0
    country_true: Counter = Counter()
    country_recalled: Counter = Counter()
    country_candidates: Counter = Counter()
    country_s1_count: Counter = Counter()

    for _, row in ground_truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw_matches = row.get("matched_entity_ids", "")
        if pd.isna(raw_matches) or not str(raw_matches).strip():
            true_set = set()
        else:
            true_set = {m.strip() for m in str(raw_matches).split(",") if m.strip()}

        cand_set = candidate_dict.get(s1_id, set())
        n_cands = len(cand_set)
        total_candidate_pairs += n_cands

        cntry = s1_country_map.get(s1_id, "UNKNOWN")
        country_s1_count[cntry] += 1
        country_candidates[cntry] += n_cands

        if true_set:
            total_true_matches += len(true_set)
            country_true[cntry] += len(true_set)
            hit = len(true_set & cand_set)
            recalled_matches += hit
            country_recalled[cntry] += hit

    recall_ceiling = (recalled_matches / total_true_matches) if total_true_matches > 0 else 1.0
    n_s1 = len(ground_truth_df)
    avg_candidates = total_candidate_pairs / n_s1 if n_s1 > 0 else 0.0

    # Total theoretical cross-product
    total_cross_product = n_s1 * total_target_count
    reduction_ratio = 1.0 - (total_candidate_pairs / total_cross_product) if total_cross_product > 0 else 1.0

    country_metrics = {}
    for cntry in country_s1_count:
        t_cnt = country_true[cntry]
        r_cnt = country_recalled[cntry]
        rec = (r_cnt / t_cnt) if t_cnt > 0 else 1.0
        c_s1 = country_s1_count[cntry]
        c_cand = country_candidates[cntry]
        avg_c = c_cand / c_s1 if c_s1 > 0 else 0.0
        country_metrics[cntry] = {
            "n_s1": c_s1,
            "true_matches": t_cnt,
            "recalled_matches": r_cnt,
            "recall_ceiling": rec,
            "avg_candidates": avg_c,
        }

    return {
        "n_s1_entities": n_s1,
        "total_true_matches": total_true_matches,
        "recalled_matches": recalled_matches,
        "recall_ceiling": recall_ceiling,
        "total_candidate_pairs": total_candidate_pairs,
        "avg_candidates_per_entity": avg_candidates,
        "reduction_ratio": reduction_ratio,
        "by_country": country_metrics,
    }
