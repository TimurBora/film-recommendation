import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time
import os
import inspect
from loguru import logger
from tqdm import tqdm


class RecommendationEvaluator:

    def __init__(
        self,
        models: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        relevance_threshold: float = 4.0,
        user_sample_size: Optional[int] = None,
        random_state: int = 42,
        use_gpu: bool = True,
        item_universe: Optional[List[int]] = None
    ):
        self.models = models
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state

        all_test_users = test_df['userId'].unique()
        if user_sample_size is not None and user_sample_size < len(all_test_users):
            np.random.seed(random_state)
            self.test_users = np.random.choice(all_test_users, size=user_sample_size, replace=False)
        else:
            self.test_users = all_test_users

        self.user_to_idx = {u: i for i, u in enumerate(self.test_users)}
        self.n_test_users = len(self.test_users)

        if item_universe is None:
            self.item_catalog = sorted(set(train_df['movieId'].unique()) | set(test_df['movieId'].unique()))
        else:
            self.item_catalog = sorted(set(item_universe))
        self.item_index = pd.Index(self.item_catalog)
        self.item_set = set(self.item_catalog)
        self.item_catalog_list = list(self.item_catalog)
        self.n_items = len(self.item_index)

        test_relevant = test_df[test_df['rating'] >= relevance_threshold]
        self.rel_dict = test_relevant.groupby('userId')['movieId'].apply(list).to_dict()
        self.train_watched_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()

        rel_indices = []
        for u in self.test_users:
            mids = self.rel_dict.get(u, [])
            valid_mids = [mid for mid in mids if mid in self.item_set]
            idxs = self.item_index.get_indexer(valid_mids).astype(np.int32)
            idxs.sort()
            rel_indices.append(idxs)
        self.rel_indices_sorted = rel_indices
        self.counts_arr = np.array([len(arr) for arr in rel_indices], dtype=np.float64)

        item_popularity = train_df['movieId'].value_counts().to_dict()
        total_interactions = sum(item_popularity.values())
        default_novelty = -np.log2(1.0 / total_interactions) if total_interactions > 0 else 0.0

        novelty_array = np.zeros(self.n_items, dtype=np.float64)
        for pos, mid in enumerate(self.item_catalog):
            cnt = item_popularity.get(mid, 0)
            if cnt > 0:
                novelty_array[pos] = -np.log2(cnt / total_interactions)
            else:
                novelty_array[pos] = default_novelty
        self.novelty_array = novelty_array
        self.results = None

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048
    ) -> List[Dict[str, Any]]:
        logger.info("Evaluating '{}'", model_name)
        start_time = time.time()

        n_users = self.n_test_users
        max_k = max(k_values) if k_values else max_recommendations
        max_k = min(max_k, max_recommendations)
        recs_arr = np.full((n_users, max_recommendations), -1, dtype=np.int64)

        has_batch = hasattr(model, 'get_top_k_recommendations_batch')
        supports_valid_items = False
        if has_batch:
            sig = inspect.signature(model.get_top_k_recommendations_batch)
            supports_valid_items = 'valid_items' in sig.parameters

        if has_batch and supports_valid_items:
            for start in tqdm(range(0, n_users, batch_size), desc=f"{model_name}", leave=False):
                end = min(start + batch_size, n_users)
                user_chunk = self.test_users[start:end]
                chunk_indices = [self.user_to_idx[u] for u in user_chunk]
                watched_list = [self.train_watched_items.get(u, set()) for u in user_chunk]
                try:
                    batch_recs = model.get_top_k_recommendations_batch(
                        user_ids=user_chunk,
                        watched_items_list=watched_list,
                        k=max_recommendations,
                        valid_items=self.item_catalog_list
                    )
                    for i, idx in enumerate(chunk_indices):
                        row = batch_recs[i][:max_recommendations]
                        recs_arr[idx, :len(row)] = row
                except Exception as e:
                    logger.warning("Batch error for {}: {}", model_name, e)
        elif has_batch and not supports_valid_items:
            for start in tqdm(range(0, n_users, batch_size), desc=f"{model_name}", leave=False):
                end = min(start + batch_size, n_users)
                user_chunk = self.test_users[start:end]
                chunk_indices = [self.user_to_idx[u] for u in user_chunk]
                watched_list = [self.train_watched_items.get(u, set()) for u in user_chunk]
                try:
                    batch_recs = model.get_top_k_recommendations_batch(
                        user_ids=user_chunk,
                        watched_items_list=watched_list,
                        k=max_recommendations
                    )
                    for i, idx in enumerate(chunk_indices):
                        row = [mid for mid in batch_recs[i] if mid in self.item_set][:max_recommendations]
                        recs_arr[idx, :len(row)] = row
                except Exception as e:
                    logger.warning("Batch error for {}: {}", model_name, e)
        else:
            item_catalog_list = self.item_catalog_list
            def fetch_for_user(u):
                watched = self.train_watched_items.get(u, set())
                try:
                    sig = inspect.signature(model.get_top_k_recommendations)
                    if 'valid_items' in sig.parameters:
                        recs = model.get_top_k_recommendations(
                            user_id=int(u),
                            watched_items=watched,
                            k=max_recommendations,
                            valid_items=item_catalog_list
                        )
                    else:
                        recs = model.get_top_k_recommendations(
                            user_id=int(u),
                            watched_items=watched,
                            k=max_recommendations
                        )
                        recs = [mid for mid in recs if mid in self.item_set]
                    return recs
                except Exception:
                    return []

            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
                future_to_user = {executor.submit(fetch_for_user, u): u for u in self.test_users}
                for future in tqdm(future_to_user, total=n_users, desc=f"{model_name}", leave=False):
                    u = future_to_user[future]
                    idx = self.user_to_idx[u]
                    row = future.result()[:max_recommendations]
                    recs_arr[idx, :len(row)] = row

        recs_idx = self.item_index.get_indexer(recs_arr.ravel()).reshape(recs_arr.shape)

        hits = np.zeros((n_users, max_k), dtype=bool)
        for i in range(n_users):
            rel_sorted = self.rel_indices_sorted[i]
            if len(rel_sorted) == 0:
                continue
            s = np.searchsorted(rel_sorted, recs_idx[i, :max_k])
            mask = (s < len(rel_sorted)) & (rel_sorted[s] == recs_idx[i, :max_k])
            hits[i] = mask

        safe_indices = np.where(recs_idx[:, :max_k] >= 0, recs_idx[:, :max_k], 0)
        raw_nov = np.take(self.novelty_array, safe_indices, mode='clip')
        novelty_vals = np.where(recs_idx[:, :max_k] != -1, raw_nov, 0.0)

        discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
        cum_discounts = np.zeros(max_k + 1, dtype=np.float64)
        cum_discounts[1:] = np.cumsum(discounts)
        cumsum_hits = np.cumsum(hits, axis=1).astype(np.float64)
        positions = np.arange(1, max_k + 1, dtype=np.float64)
        prec_at_j = cumsum_hits / positions
        ap_terms = prec_at_j * hits
        cumsum_ap = np.cumsum(ap_terms, axis=1)
        dcg_terms = hits * discounts
        cumsum_dcg = np.cumsum(dcg_terms, axis=1)
        first_hit_idx = np.argmax(hits, axis=1).astype(np.int32)
        any_hit = hits.any(axis=1)
        first_hit_idx[~any_hit] = -1

        min_counts_k = np.minimum(
            self.counts_arr[:, None],
            np.arange(1, max_k + 1, dtype=np.float64)[None, :]
        )
        ideal_dcg = np.where(min_counts_k > 0, cum_discounts[min_counts_k.astype(int)], 0.0)

        valid_recs = recs_arr[recs_arr != -1]
        unique_recs = np.unique(valid_recs)
        coverage = len(unique_recs) / self.n_items if self.n_items else 0.0

        results = []
        for k in k_values:
            k_idx = min(k, max_k) - 1
            prec = cumsum_hits[:, k_idx] / k
            rec = np.divide(
                cumsum_hits[:, k_idx], self.counts_arr,
                out=np.zeros_like(self.counts_arr),
                where=self.counts_arr > 0
            )
            dcg = cumsum_dcg[:, k_idx]
            ndcg = np.divide(
                dcg, ideal_dcg[:, k_idx],
                out=np.zeros_like(dcg),
                where=ideal_dcg[:, k_idx] > 0
            )
            ap = np.divide(
                cumsum_ap[:, k_idx], np.minimum(self.counts_arr, k),
                out=np.zeros_like(self.counts_arr),
                where=self.counts_arr > 0
            )
            mrr = np.where(
                (first_hit_idx >= 0) & (first_hit_idx < k),
                1.0 / (first_hit_idx + 1),
                0.0
            )
            nov = novelty_vals[:, :k].mean(axis=1)

            results.append({
                'model': model_name,
                'k': k,
                'precision': float(prec.mean()),
                'recall': float(rec.mean()),
                'ndcg': float(ndcg.mean()),
                'map': float(ap.mean()),
                'mrr': float(mrr.mean()),
                'novelty': float(nov.mean()),
                'coverage': coverage,
                'n_users': int(np.any(hits[:, :k], axis=1).sum())
            })

        elapsed = time.time() - start_time
        logger.info("'{}' done in {:.1f}s", model_name, elapsed)
        return results

    def evaluate_all_models(
        self,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048
    ) -> pd.DataFrame:
        total_start = time.time()
        all_results = []
        for model_name, model in self.models.items():
            try:
                res = self.evaluate_model(model_name, model, k_values, max_recommendations, batch_size)
                all_results.extend(res)
            except Exception as e:
                logger.error("Failed {}: {}", model_name, e)
                continue
        self.results = pd.DataFrame(all_results)
        return self.results

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("No results available. Run evaluate_all_models() first.")
        return self.results