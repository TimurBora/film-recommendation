import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from typing import List, Optional, Tuple, Dict, Any
from numba import njit
import time
import os
import psutil
from loguru import logger
from src.utils.logger import LoggingConfig, StepLogger
from tqdm import tqdm

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

MIN_RATING = np.float32(0.5)
MAX_RATING = np.float32(5.0)

@njit(nogil=True)
def _sgd_optimize(rows, cols, data, pu, qi, bu, bi, global_mean, lr_all, reg_all,
                  n_epochs, random_state, vu, vi, vbu, vbi, momentum):
    np.random.seed(random_state)
    n_samples = len(data)
    n_factors = pu.shape[1]
    initial_lr = lr_all
    for epoch in range(n_epochs):
        lr = initial_lr * (1.0 - float(epoch) / n_epochs)
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

            grad_bu = err - reg_all * bu[u]
            vbu[u] = momentum * vbu[u] + lr * grad_bu
            bu[u] += vbu[u]

            grad_bi = err - reg_all * bi[i]
            vbi[i] = momentum * vbi[i] + lr * grad_bi
            bi[i] += vbi[i]

            for k in range(n_factors):
                p_old = pu[u, k]
                grad_p = err * qi[i, k] - reg_all * pu[u, k]
                vu[u, k] = momentum * vu[u, k] + lr * grad_p
                pu[u, k] += vu[u, k]

                grad_q = err * p_old - reg_all * qi[i, k]
                vi[i, k] = momentum * vi[i, k] + lr * grad_q
                qi[i, k] += vi[i, k]

class CollaborativeFiltering:
    def __init__(
        self,
        k_components: int = 110,
        reg_all: float = 0.02,
        lr_all: float = 0.005,
        n_epochs: int = 20,
        alpha: float = 0.2,
        min_ratings: int = 15,
        random_state: int = 42,
        use_gpu: bool = True,
        logging_config: Optional[LoggingConfig] = None,
        progress_bar: bool = True
    ) -> None:
        self.n_components = k_components
        self.reg_all = np.float32(reg_all)
        self.lr_all = np.float32(lr_all)
        self.n_epochs = n_epochs
        self.alpha = np.float32(alpha)
        self.min_ratings = min_ratings
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar
        self._raw_to_inner_user: Dict[int, int] = {}
        self._raw_to_inner_item: Dict[int, int] = {}
        self._inner_to_raw_user: np.ndarray = np.array([])
        self._inner_to_raw_item: np.ndarray = np.array([])
        self._pu: np.ndarray = np.array([])
        self._qi: np.ndarray = np.array([])
        self._bu: np.ndarray = np.array([])
        self._bi: np.ndarray = np.array([])
        self._qi_norms: np.ndarray = np.array([])
        self._pu_gpu = None
        self._qi_gpu = None
        self._bu_gpu = None
        self._bi_gpu = None
        self._qi_norms_gpu = None
        self._item_popularity_gpu = None
        self._valid_item_mask_gpu = None
        self._item_popularity: np.ndarray = np.array([])
        self._valid_item_mask: np.ndarray = np.array([])
        self._global_mean: np.float32 = np.float32(0.0)
        self._popular_movies: List[int] = []
        self._popular_inner: np.ndarray = np.array([], dtype=np.int32)
        self._is_fitted: bool = False
        self.config = logging_config or LoggingConfig()
        self.step_logger = StepLogger(self.config)

    def fit(self, df_ratings: pd.DataFrame) -> "CollaborativeFiltering":
        total_start = time.perf_counter()
        total_cpu = time.process_time()
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
        extra_mapping = {
            "n_users": n_users,
            "n_items": n_items,
            "n_ratings": len(df_ratings),
            "n_popular_items": len(self._popular_movies)
        }
        self.step_logger.log_step("Filtering & Mapping", total_start, total_cpu, extra_mapping)

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
        pop_mapped = popularity_series.rename(self._raw_to_inner_item)
        pop_mapped = pop_mapped[pop_mapped.index.isin(range(n_items))]
        pop_array[pop_mapped.index] = pop_mapped.values
        self._item_popularity = np.maximum(pop_array, 1.0)
        self._valid_item_mask = np.ones(n_items, dtype=bool)
        self._popular_inner = np.array(
            [self._raw_to_inner_item.get(m, -1) for m in self._popular_movies], dtype=np.int32
        )
        self._popular_inner = self._popular_inner[self._popular_inner >= 0]

        extra_data = {
            "global_mean": float(self._global_mean),
            "n_factors": self.n_components,
            "sparsity": f"{sparse_coo.nnz / (n_users * n_items):.4%}"
        }
        self.step_logger.log_step("Data Preparation", total_start, total_cpu, extra_data)

        vu = np.zeros_like(self._pu)
        vi = np.zeros_like(self._qi)
        vbu = np.zeros_like(self._bu)
        vbi = np.zeros_like(self._bi)
        momentum = np.float32(0.9)

        epoch_iter = range(self.n_epochs)
        if self.progress_bar:
            epoch_iter = tqdm(epoch_iter, desc="SGD epochs", unit="epoch")

        for epoch in epoch_iter:
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
                1,
                self.random_state + epoch,
                vu,
                vi,
                vbu,
                vbi,
                momentum
            )

        extra_opt = {
            "n_epochs": self.n_epochs,
            "lr": float(self.lr_all),
            "reg": float(self.reg_all),
            "momentum": float(momentum)
        }
        self.step_logger.log_step("SGD Optimization", total_start, total_cpu, extra_opt)

        self._qi_norms = np.linalg.norm(self._qi, axis=1)
        self._qi_norms[self._qi_norms == 0] = 1e-9

        if self.use_gpu and HAS_GPU:
            self._pu_gpu = cp.asarray(self._pu)
            self._qi_gpu = cp.asarray(self._qi)
            self._bu_gpu = cp.asarray(self._bu)
            self._bi_gpu = cp.asarray(self._bi)
            self._qi_norms_gpu = cp.asarray(self._qi_norms)
            self._item_popularity_gpu = cp.asarray(self._item_popularity)
            self._valid_item_mask_gpu = cp.asarray(self._valid_item_mask)

        self._is_fitted = True
        self.step_logger.log_step("Finalization", total_start, total_cpu)
        if self.config.log_data_shapes:
            logger.info(
                f"pu shape: {self._pu.shape}, qi shape: {self._qi.shape}, "
                f"global_mean: {self._global_mean:.4f}, n_users: {n_users}, n_items: {n_items}"
            )
        total_wall = time.perf_counter() - total_start
        total_cpu_elapsed = time.process_time() - total_cpu
        mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        logger.info(
            f"Fitting complete | total_wall={total_wall:.2f}s total_cpu={total_cpu_elapsed:.2f}s final_rss={mem_final:.1f}MB"
        )
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
        watched_inners = [self._raw_to_inner_item[m] for m in watched_movie_ids if m in self._raw_to_inner_item]
        if self.use_gpu and HAS_GPU:
            all_scores = self._global_mean + self._bu_gpu[u_inner] + self._bi_gpu + cp.dot(self._qi_gpu, self._pu_gpu[u_inner])
            all_scores = cp.clip(all_scores, MIN_RATING, MAX_RATING)
            penalized_scores = all_scores / (self._item_popularity_gpu ** self.alpha)
            mask = cp.ones_like(penalized_scores, dtype=cp.bool_)
            if watched_inners:
                mask[watched_inners] = False
            remaining_inners = cp.nonzero(mask)[0]
            if len(remaining_inners) == 0:
                cp.get_default_memory_pool().free_all_blocks()
                return []
            remaining_penalized = penalized_scores[remaining_inners]
            top_k = min(top_n, len(remaining_penalized))
            if top_k < len(remaining_penalized):
                partitioned_idx = cp.argpartition(-remaining_penalized, top_k - 1)[:top_k]
                sorted_top_idx = partitioned_idx[cp.argsort(-remaining_penalized[partitioned_idx])]
            else:
                sorted_top_idx = cp.argsort(-remaining_penalized)
            final_inners = remaining_inners[sorted_top_idx].get()
            final_scores = all_scores[final_inners].get()
            cp.get_default_memory_pool().free_all_blocks()
            return [
                (self._inner_to_raw_item[idx], float(final_scores[i]))
                for i, idx in enumerate(final_inners)
            ]
        else:
            all_scores = self._global_mean + self._bu[u_inner] + self._bi + np.dot(self._qi, self._pu[u_inner])
            all_scores = np.clip(all_scores, MIN_RATING, MAX_RATING)
            penalized_scores = all_scores / (self._item_popularity ** self.alpha)
            mask = np.ones_like(penalized_scores, dtype=bool)
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
        if not np.any(valid_mask):
            return np.full(len(item_ids), self._global_mean, dtype=np.float32).tolist()
        valid_i = i_inners[valid_mask]
        if self.use_gpu and HAS_GPU:
            scores_gpu = cp.full(len(item_ids), self._global_mean, dtype=cp.float32)
            valid_i_gpu = cp.asarray(valid_i)
            if u_inner is not None:
                est = self._global_mean + self._bu_gpu[u_inner] + self._bi_gpu[valid_i_gpu] + cp.sum(self._pu_gpu[u_inner] * self._qi_gpu[valid_i_gpu], axis=1)
            else:
                est = self._global_mean + self._bi_gpu[valid_i_gpu]
            valid_mask_gpu = cp.asarray(valid_mask)
            scores_gpu[valid_mask_gpu] = cp.clip(est, MIN_RATING, MAX_RATING)
            scores = scores_gpu.get().tolist()
            cp.get_default_memory_pool().free_all_blocks()
            return scores
        else:
            scores = np.full(len(item_ids), self._global_mean, dtype=np.float32)
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

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int
    ) -> List[List[int]]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        batch_size = len(user_ids)
        n_items = len(self._inner_to_raw_item)
        xp = cp if (self.use_gpu and HAS_GPU) else np
        if xp is cp:
            bu = self._bu_gpu
            bi = self._bi_gpu
            qi = self._qi_gpu
            pu = self._pu_gpu
            pop = self._item_popularity_gpu
            popular_inner = self._popular_inner
        else:
            bu = self._bu
            bi = self._bi
            qi = self._qi
            pu = self._pu
            pop = self._item_popularity
            popular_inner = self._popular_inner

        u_inners_raw = [self._raw_to_inner_user.get(uid, -1) for uid in user_ids]
        known_flags = xp.array([ui != -1 for ui in u_inners_raw])
        known_positions = xp.nonzero(known_flags)[0]
        unknown_positions = xp.nonzero(~known_flags)[0]

        scores = xp.full((batch_size, n_items), -xp.inf, dtype=xp.float32)

        if len(known_positions) > 0:
            u_inners = xp.array([u_inners_raw[i] for i in known_positions], dtype=xp.int32)
            pu_sel = pu[u_inners]
            dot = xp.dot(pu_sel, qi.T)
            scores_known = self._global_mean + bu[u_inners, None] + bi[None, :] + dot
            scores_known = xp.clip(scores_known, MIN_RATING, MAX_RATING)
            penalized = scores_known / (pop[None, :] ** self.alpha)
            mask = xp.ones((len(known_positions), n_items), dtype=bool)
            for row_idx, batch_idx in enumerate(known_positions):
                watched_ids = watched_items_list[batch_idx]
                watched_inners = [self._raw_to_inner_item[m] for m in watched_ids if m in self._raw_to_inner_item]
                if watched_inners:
                    mask[row_idx, watched_inners] = False
            penalized[~mask] = -xp.inf
            scores[known_positions] = penalized

        if len(unknown_positions) > 0 and len(popular_inner) > 0:
            pop_scores = xp.full(n_items, -xp.inf, dtype=xp.float32)
            pop_scores[popular_inner] = self._global_mean
            for batch_idx in unknown_positions:
                watched_ids = watched_items_list[batch_idx]
                watched_inners = [self._raw_to_inner_item[m] for m in watched_ids if m in self._raw_to_inner_item]
                row_scores = pop_scores.copy()
                row_scores[watched_inners] = -xp.inf
                scores[batch_idx] = row_scores

        recs_list = []
        for row_idx in range(batch_size):
            row = scores[row_idx]
            valid_indices = xp.nonzero(row != -xp.inf)[0]
            if len(valid_indices) == 0:
                recs_list.append([])
                continue
            row_valid = row[valid_indices]
            topk = min(k, len(row_valid))
            if xp is np:
                part = np.argpartition(-row_valid, topk - 1)[:topk]
                top_idx = part[np.argsort(-row_valid[part])]
            else:
                part = cp.argpartition(-row_valid, topk - 1)[:topk]
                top_idx = part[cp.argsort(-row_valid[part])]
            inner_selected = valid_indices[top_idx]
            raw_ids = self._inner_to_raw_item[inner_selected.get() if xp is cp else inner_selected]
            recs_list.append(raw_ids.tolist())

        if xp is cp:
            cp.get_default_memory_pool().free_all_blocks()
        return recs_list

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
        if self.use_gpu and HAS_GPU:
            target_vector = self._qi_gpu[target_inner]
            liked_vectors = self._qi_gpu[liked_inners]
            norm_target = self._qi_norms_gpu[target_inner]
            norm_liked = self._qi_norms_gpu[liked_inners]
            denom = norm_liked * norm_target
            sims = cp.dot(liked_vectors, target_vector) / denom
            valid_sim_mask = sims > 0
            valid_sims = sims[valid_sim_mask]
            valid_inners_arr = cp.asarray(liked_inners)[valid_sim_mask]
            if len(valid_sims) == 0:
                cp.get_default_memory_pool().free_all_blocks()
                return []
            sort_idx = cp.argsort(-valid_sims)[:top_n_reasons]
            valid_inners_cpu = valid_inners_arr[sort_idx].get()
            valid_sims_cpu = valid_sims[sort_idx].get()
            cp.get_default_memory_pool().free_all_blocks()
            return [
                {
                    'movie_id': self._inner_to_raw_item[valid_inners_cpu[idx]],
                    'similarity': float(valid_sims_cpu[idx])
                }
                for idx in range(len(sort_idx))
            ]
        else:
            target_vector = self._qi[target_inner]
            liked_vectors = self._qi[liked_inners]
            norm_target = self._qi_norms[target_inner]
            norm_liked = self._qi_norms[liked_inners]
            denom = norm_liked * norm_target
            sims = np.dot(liked_vectors, target_vector) / denom
            valid_sim_mask = sims > 0
            valid_sims = sims[valid_sim_mask]
            valid_inners_arr = np.array(liked_inners)[valid_sim_mask]
            if len(valid_sims) == 0:
                return []
            sort_idx = np.argsort(-valid_sims)[:top_n_reasons]
            return [
                {
                    'movie_id': self._inner_to_raw_item[valid_inners_arr[idx]],
                    'similarity': float(valid_sims[idx])
                }
                for idx in sort_idx
            ]