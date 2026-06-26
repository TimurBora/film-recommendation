import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from typing import List, Optional, Tuple, Dict, Any
from numba import njit

MIN_RATING = np.float32(0.5)
MAX_RATING = np.float32(5.0)

@njit(nogil=True)
def _sgd_optimize(rows, cols, data, pu, qi, bu, bi, global_mean, lr_all, reg_all, n_epochs, random_state):
    np.random.seed(random_state)
    n_samples = len(data)
    n_factors = pu.shape[1]
    
    for _ in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for idx in indices:
            u = rows[idx]
            i = cols[idx]
            r = data[idx]
            
            dot_product = 0.0
            for k in range(n_factors):
                dot_product += pu[u, k] * qi[i, k]
                
            pred = global_mean + bu[u] + bi[i] + dot_product
            err = r - pred
            
            bu[u] += lr_all * (err - reg_all * bu[u])
            bi[i] += lr_all * (err - reg_all * bi[i])
            
            for k in range(n_factors):
                p_old = pu[u, k]
                pu[u, k] += lr_all * (err * qi[i, k] - reg_all * pu[u, k])
                qi[i, k] += lr_all * (err * p_old - reg_all * qi[i, k])

class CollaborativeFiltering:
    def __init__(
        self, 
        k_components: int = 50, 
        reg_all: float = 0.02, 
        lr_all: float = 0.005,
        n_epochs: int = 20,
        alpha: float = 0.2, 
        min_ratings: int = 15, 
        random_state: int = 42
    ) -> None:
        self.n_components = k_components
        self.reg_all = np.float32(reg_all)
        self.lr_all = np.float32(lr_all)
        self.n_epochs = n_epochs
        self.alpha = np.float32(alpha)
        self.min_ratings = min_ratings
        self.random_state = random_state
        
        self._raw_to_inner_user: Dict[int, int] = {}
        self._raw_to_inner_item: Dict[int, int] = {}
        self._inner_to_raw_user: np.ndarray = np.array([])
        self._inner_to_raw_item: np.ndarray = np.array([])
        
        self._pu: np.ndarray = np.array([])
        self._qi: np.ndarray = np.array([])
        self._bu: np.ndarray = np.array([])
        self._bi: np.ndarray = np.array([])
        self._qi_norms: np.ndarray = np.array([])
        
        self._item_popularity: np.ndarray = np.array([])
        self._valid_item_mask: np.ndarray = np.array([])
        self._global_mean: np.float32 = np.float32(0.0)
        self._popular_movies: List[int] = []
        self._is_fitted: bool = False

    def fit(self, df_ratings: pd.DataFrame) -> "CollaborativeFiltering":
        popularity_series = df_ratings["movieId"].value_counts()
        filtered_popularity = popularity_series[popularity_series >= self.min_ratings]
        self._popular_movies = filtered_popularity.index.tolist()
        
        unique_users = df_ratings["userId"].unique()
        unique_movies = df_ratings["movieId"].unique()
        
        n_users = len(unique_users)
        n_items = len(unique_movies)
        
        self._raw_to_inner_user = dict(zip(unique_users, range(n_users)))
        self._raw_to_inner_item = dict(zip(unique_movies, range(n_items)))
        
        self._inner_to_raw_user = unique_users
        self._inner_to_raw_item = unique_movies
        
        u_indices = df_ratings["userId"].map(self._raw_to_inner_user).values.astype(np.int32)
        i_indices = df_ratings["movieId"].map(self._raw_to_inner_item).values.astype(np.int32)
        ratings = df_ratings["rating"].values.astype(np.float32)
        
        sparse_coo = coo_matrix((ratings, (u_indices, i_indices)), shape=(n_users, n_items))
        self._global_mean = np.float32(sparse_coo.data.mean())
        
        np.random.seed(self.random_state)
        self._pu = np.random.normal(0, 0.1, (n_users, self.n_components)).astype(np.float32)
        self._qi = np.random.normal(0, 0.1, (n_items, self.n_components)).astype(np.float32)
        self._bu = np.zeros(n_users, dtype=np.float32)
        self._bi = np.zeros(n_items, dtype=np.float32)
        
        pop_array = np.zeros(n_items, dtype=np.float32)
        valid_items_indices = filtered_popularity.index.map(self._raw_to_inner_item).dropna().astype(np.int32)
        
        pop_mapped = popularity_series.rename(self._raw_to_inner_item)
        pop_mapped = pop_mapped[pop_mapped.index.isin(range(n_items))]
        pop_array[pop_mapped.index] = pop_mapped.values
        
        self._item_popularity = np.maximum(pop_array, 1.0)
        self._valid_item_mask = np.zeros(n_items, dtype=bool)
        self._valid_item_mask[valid_items_indices] = True
        
        _sgd_optimize(
            sparse_coo.row, 
            sparse_coo.col, 
            sparse_coo.data, 
            self._pu, 
            self._qi, 
            self._bu, 
            self._bi, 
            self._global_mean, 
            self.lr_all, 
            self.reg_all, 
            self.n_epochs,
            self.random_state
        )
        
        self._qi_norms = np.linalg.norm(self._qi, axis=1)
        self._qi_norms[self._qi_norms == 0] = 1e-9
        
        self._is_fitted = True
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        u_inner = self._raw_to_inner_user.get(user_id)
        i_inner = self._raw_to_inner_item.get(movie_id)
        
        if u_inner is not None and i_inner is not None:
            est = self._global_mean + self._bu[u_inner] + self._bi[i_inner] + np.dot(self._pu[u_inner], self._qi[i_inner])
        elif i_inner is not None:
            est = self._global_mean + self._bi[i_inner]
        else:
            est = self._global_mean
            
        return float(np.clip(est, MIN_RATING, MAX_RATING))

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        u_inner = self._raw_to_inner_user.get(user_id)

        if u_inner is None:
            watched_set = set(watched_movie_ids)
            unwatched_popular = [m_id for m_id in self._popular_movies if m_id not in watched_set]
            return [(m_id, float(self._global_mean)) for m_id in unwatched_popular[:top_n]]

        all_scores = self._global_mean + self._bu[u_inner] + self._bi + np.dot(self._qi, self._pu[u_inner])
        all_scores = np.clip(all_scores, MIN_RATING, MAX_RATING)

        penalized_scores = all_scores / (self._item_popularity ** self.alpha)
        mask = self._valid_item_mask.copy()
        
        watched_inners = [self._raw_to_inner_item[m] for m in watched_movie_ids if m in self._raw_to_inner_item]
        if watched_inners:
            mask[watched_inners] = False

        remaining_inners = np.nonzero(mask)[0]
        if len(remaining_inners) == 0:
            return []
            
        remaining_penalized = penalized_scores[remaining_inners]
        top_k = min(top_n, len(remaining_penalized))
        
        if top_k < len(remaining_penalized):
            partitioned_idx = np.argpartition(-remaining_penalized, top_k - 1)[:top_k]
            sorted_top_idx = partitioned_idx[np.argsort(-remaining_penalized[partitioned_idx])]
        else:
            sorted_top_idx = np.argsort(-remaining_penalized)
            
        final_inners = remaining_inners[sorted_top_idx]
        
        return [
            (self._inner_to_raw_item[idx], float(all_scores[idx]))
            for idx in final_inners
        ]

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        u_inner = self._raw_to_inner_user.get(user_id)
        i_inners = np.array([self._raw_to_inner_item.get(mid, -1) for mid in item_ids])
        valid_mask = i_inners != -1
        
        scores = np.full(len(item_ids), self._global_mean, dtype=np.float32)
        
        if not np.any(valid_mask):
            return scores.tolist()
            
        valid_i = i_inners[valid_mask]
        
        if u_inner is not None:
            est = self._global_mean + self._bu[u_inner] + self._bi[valid_i] + np.sum(self._pu[u_inner] * self._qi[valid_i], axis=1)
        else:
            est = self._global_mean + self._bi[valid_i]
            
        scores[valid_mask] = np.clip(est, MIN_RATING, MAX_RATING)
        return scores.tolist()

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10
    ) -> List[int]:
        recs = self.recommend_for_user(user_id, list(watched_items), top_n=k)
        return [mid for mid, _ in recs]

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        if not self._is_fitted:
            return []

        target_inner = self._raw_to_inner_item.get(movie_id)
        if target_inner is None:
            return []

        liked_inners = [self._raw_to_inner_item[mid] for mid in liked_items if mid in self._raw_to_inner_item]
        if not liked_inners:
            return []

        target_vector = self._qi[target_inner]
        liked_vectors = self._qi[liked_inners]

        norm_target = self._qi_norms[target_inner]
        norm_liked = self._qi_norms[liked_inners]
        
        denom = norm_liked * norm_target
        sims = np.dot(liked_vectors, target_vector) / denom
        
        valid_sim_mask = sims > 0
        valid_sims = sims[valid_sim_mask]
        valid_inners = np.array(liked_inners)[valid_sim_mask]
        
        if len(valid_sims) == 0:
            return []
            
        sort_idx = np.argsort(-valid_sims)[:top_n_reasons]
        
        return [
            {
                'movie_id': self._inner_to_raw_item[valid_inners[idx]],
                'similarity': float(valid_sims[idx])
            }
            for idx in sort_idx
        ]