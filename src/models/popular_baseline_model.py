import pandas as pd
import numpy as np
from typing import List, Dict, Any


class PopularityBaseline:
    def __init__(self):
        self._sorted_items = np.empty(0, dtype=np.int64)
        self._sorted_items_list = []
        self._movie_ids_lookup = np.empty(0, dtype=np.int64)
        self._pops_lookup = np.empty(0, dtype=np.float32)
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame) -> "PopularityBaseline":
        counts = ratings_df.groupby("movieId").size().sort_values(ascending=False)
        self._sorted_items = counts.index.values.astype(np.int64)
        pops_sorted = counts.values.astype(np.float32)

        sort_by_id = np.argsort(self._sorted_items)
        self._movie_ids_lookup = self._sorted_items[sort_by_id]
        self._pops_lookup = pops_sorted[sort_by_id]

        self._sorted_items_list = self._sorted_items.tolist()
        self.is_fitted = True
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        item_ids_arr = np.asarray(item_ids, dtype=np.int64)
        idx = np.searchsorted(self._movie_ids_lookup, item_ids_arr)
        mask = (idx < len(self._movie_ids_lookup)) & (
            self._movie_ids_lookup[idx] == item_ids_arr
        )
        scores = np.where(mask, self._pops_lookup[idx], 0.0)
        return scores.tolist()

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10
    ) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if not watched_items:
            return self._sorted_items_list[:k]
        result = []
        for mid in self._sorted_items_list:
            if mid not in watched_items:
                result.append(mid)
                if len(result) == k:
                    break
        return result

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int = 10,
    ) -> List[List[int]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        results = []
        for watched in watched_items_list:
            if not watched:
                results.append(self._sorted_items_list[:k])
            else:
                recs = []
                for mid in self._sorted_items_list:
                    if mid not in watched:
                        recs.append(mid)
                        if len(recs) == k:
                            break
                results.append(recs)
        return results

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        return []