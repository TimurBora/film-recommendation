import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz, issparse
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
import os
from loguru import logger
from tqdm import tqdm
from src.utils.logger import LoggingConfig, StepLogger

try:
    import cupy as cp
    import cupyx.scipy.sparse as cxs
    HAS_GPU_CB = True
except ImportError:
    HAS_GPU_CB = False

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@dataclass
class ContentBasedConfig:
    main_actor_weight: float = 0.15
    director_weight: float = 0.20
    cast_weight: float = 0.15
    keywords_weight: float = 0.30
    genre_weight: float = 0.10
    numerical_weight: float = 0.10
    tfidf_sublinear_tf: bool = True
    tfidf_max_features: Optional[int] = 10000
    runtime_clip_percentiles: tuple = (0.01, 0.99)
    year_n_quantiles: int = 1000
    similarity_threshold: float = 0.05
    top_k_default: int = 10
    artifacts_dir: str = "data/processed/artifacts"
    fillna_strategy: str = "median"
    use_gpu: bool = True
    pop_boost_weight: float = 0.15
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    top_k_per_item: int = 200


def _weight_cast_impl(cast_arrays, max_weight=3):
    results = []
    for lst in cast_arrays:
        weighted = []
        for i, actor in enumerate(lst):
            w = max(1, max_weight - i)
            actor_clean = str(actor).lower().replace(' ', '')
            if actor_clean:
                weighted.extend([actor_clean] * w)
        results.append(' '.join(weighted))
    return results

def _flatten_keywords_impl(keyword_lists):
    results = []
    for lst in keyword_lists:
        tokens = []
        for kw in lst:
            kw_str = str(kw).lower().replace(' ', '')
            if len(kw_str) > 2:
                tokens.append(kw_str)
        results.append(' '.join(tokens))
    return results

class ContentBasedRecommender:

    def __init__(self, config: Optional[ContentBasedConfig] = None):
        self.config = config or ContentBasedConfig()
        self.tfidf_main_actor = None
        self.tfidf_director = None
        self.tfidf_cast = None
        self.tfidf_keywords = None
        self.scaler_runtime = None
        self.scaler_year = None
        self.scaler_main_actor_rating = None
        self.scaler_director_rating = None
        self.qt_year = None
        self.similarity_matrix = None
        self.feature_matrix_norm = None
        self.movie_index = pd.Index([])
        self.movie_ids = np.array([], dtype=np.int64)
        self.user_index = pd.Index([])
        self.user_profiles = None
        self.user_liked_indices = []
        self.popularity_array = np.array([], dtype=np.float32)
        self.movie_id_to_title = {}
        self.soft_pop_arr = np.array([], dtype=np.float32)
        self.is_fitted = False
        self.step_logger = StepLogger(self.config.logging)

    @staticmethod
    def _clean_text_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.lower().str.replace(' ', '', regex=False)

    def _preprocess_numerical_features(self, df: pd.DataFrame) -> None:
        runtime_vals = df["runtime"].replace(0, np.nan)
        median_runtime = runtime_vals.median()
        df["runtime"] = runtime_vals.fillna(median_runtime)
        bounds = df["runtime"].quantile(list(self.config.runtime_clip_percentiles)).values
        df["runtime"] = df["runtime"].clip(bounds[0], bounds[1])
        self.scaler_runtime = MinMaxScaler()
        df["runtime"] = self.scaler_runtime.fit_transform(df[["runtime"]]).astype(np.float32).ravel()
        median_year = df["year"].astype(float).median()
        df["year"] = df["year"].fillna(median_year)
        self.qt_year = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=self.config.year_n_quantiles,
            random_state=42
        )
        year_transformed = self.qt_year.fit_transform(df[["year"]])
        self.scaler_year = MinMaxScaler()
        df["year"] = self.scaler_year.fit_transform(year_transformed).astype(np.float32).ravel()

    def _preprocess_actor_director_ratings(self, df: pd.DataFrame) -> None:
        if 'rating' in df.columns:
            actor_avg = df.groupby('main_actor')['rating'].transform('mean')
            median_act = actor_avg.replace(0, np.nan).median()
            actor_avg = actor_avg.replace({0: median_act, np.nan: median_act})
            df['main_actor_rating'] = np.log1p(actor_avg.values)
            self.scaler_main_actor_rating = MinMaxScaler()
            df['main_actor_rating'] = self.scaler_main_actor_rating.fit_transform(
                df[['main_actor_rating']]
            ).astype(np.float32).ravel()
        else:
            df['main_actor_rating'] = 0.0

        if 'rating' in df.columns:
            dir_avg = df.groupby('director')['rating'].transform('mean')
            median_dir = dir_avg.replace(0, np.nan).median()
            dir_avg = dir_avg.replace({0: median_dir, np.nan: median_dir})
            df['director_rating'] = np.log1p(dir_avg.values)
            self.scaler_director_rating = MinMaxScaler()
            df['director_rating'] = self.scaler_director_rating.fit_transform(
                df[['director_rating']]
            ).astype(np.float32).ravel()
        else:
            df['director_rating'] = 0.0

    def _fit_transform_tfidf(self, data: pd.Series, weight: float) -> Tuple[Any, csr_matrix]:
        vectorizer = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features,
            dtype=np.float32
        )
        matrix = vectorizer.fit_transform(data).astype(np.float32)
        return vectorizer, matrix * weight

    def _build_feature_matrix(self, df: pd.DataFrame) -> csr_matrix:
        main_actor_clean = self._clean_text_series(df["main_actor"])
        director_clean = self._clean_text_series(df["director"])

        cast_lists = df["cast"].tolist()
        cast_weighted = pd.Series(_weight_cast_impl(cast_lists), index=df.index)

        keyword_lists = df["keywords"].tolist()
        keywords_flat = pd.Series(_flatten_keywords_impl(keyword_lists), index=df.index)

        with ThreadPoolExecutor(max_workers=4) as executor:
            f_actor = executor.submit(self._fit_transform_tfidf, main_actor_clean, self.config.main_actor_weight)
            f_director = executor.submit(self._fit_transform_tfidf, director_clean, self.config.director_weight)
            f_cast = executor.submit(self._fit_transform_tfidf, cast_weighted, self.config.cast_weight)
            f_keywords = executor.submit(self._fit_transform_tfidf, keywords_flat, self.config.keywords_weight)
            self.tfidf_main_actor, main_actor_tfidf = f_actor.result()
            self.tfidf_director, director_tfidf = f_director.result()
            self.tfidf_cast, cast_tfidf = f_cast.result()
            self.tfidf_keywords, keywords_tfidf = f_keywords.result()

        numerical_features = df[[
            'runtime', 'year', 'main_actor_rating', 'director_rating'
        ]].values.astype(np.float32)
        numerical_matrix = csr_matrix(numerical_features, dtype=np.float32) * self.config.numerical_weight

        genre_cols = [col for col in df.columns if col.startswith("genre_")]
        if genre_cols:
            genre_matrix = csr_matrix(df[genre_cols].values.astype(np.float32), dtype=np.float32) * self.config.genre_weight
        else:
            genre_matrix = csr_matrix((df.shape[0], 0), dtype=np.float32)

        combined_features = hstack([
            main_actor_tfidf,
            director_tfidf,
            cast_tfidf,
            keywords_tfidf,
            genre_matrix,
            numerical_matrix
        ], dtype=np.float32).tocsr()
        return combined_features

    def _compute_similarity_matrix(self, features_norm: csr_matrix) -> csr_matrix:
        n_samples = features_norm.shape[0]
        topk = self.config.top_k_per_item
        if topk <= 0:
            return self._threshold_similarity(features_norm)

        batch_size = min(1500, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        iterator = range(0, n_samples, batch_size)
        if self.config.logging.show_progress_bars:
            iterator = tqdm(iterator, total=n_batches, desc="TopK similarity", unit="batch")

        feat_t = features_norm.T.tocsr()
        all_data = []
        all_indices = []
        indptr = [0]

        if self.config.use_gpu and HAS_GPU_CB:
            features_norm_gpu = cxs.csr_matrix(features_norm)
            feat_t_gpu = cxs.csr_matrix(feat_t)
            for i in iterator:
                end = min(i + batch_size, n_samples)
                batch_gpu = features_norm_gpu[i:end]
                sim_gpu = batch_gpu.dot(feat_t_gpu)
                sim_cpu = sim_gpu.get()
                for row_idx in range(sim_cpu.shape[0]):
                    row = np.asarray(sim_cpu[row_idx].todense()).ravel()
                    if len(row) <= topk:
                        idx = np.argsort(-row)
                    else:
                        idx = np.argpartition(-row, topk - 1)[:topk]
                        idx = idx[np.argsort(-row[idx])]
                    all_data.append(row[idx])
                    all_indices.append(idx)
                    indptr.append(indptr[-1] + len(idx))
                del batch_gpu, sim_gpu, sim_cpu
                cp.get_default_memory_pool().free_all_blocks()
            del features_norm_gpu, feat_t_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            for i in iterator:
                end = min(i + batch_size, n_samples)
                batch = features_norm[i:end]
                sim_dense = (batch @ feat_t).toarray()
                for row_idx in range(sim_dense.shape[0]):
                    row = sim_dense[row_idx]
                    if len(row) <= topk:
                        idx = np.argsort(-row)
                    else:
                        idx = np.argpartition(-row, topk - 1)[:topk]
                        idx = idx[np.argsort(-row[idx])]
                    all_data.append(row[idx])
                    all_indices.append(idx)
                    indptr.append(indptr[-1] + len(idx))
                del batch, sim_dense

        data_arr = np.concatenate(all_data).astype(np.float32)
        indices_arr = np.concatenate(all_indices).astype(np.int32)
        indptr_arr = np.array(indptr, dtype=np.int32)
        return csr_matrix((data_arr, indices_arr, indptr_arr), shape=(n_samples, n_samples))

    def _threshold_similarity(self, features_norm: csr_matrix) -> csr_matrix:
        n_samples = features_norm.shape[0]
        batch_size = 1500
        total_batches = (n_samples + batch_size - 1) // batch_size
        iterator = range(0, n_samples, batch_size)
        if self.config.logging.show_progress_bars:
            iterator = tqdm(iterator, total=total_batches, desc="Threshold similarity", unit="batch")
        all_data = []
        all_indices = []
        indptr = [0]
        feat_t = features_norm.T.tocsr()
        if self.config.use_gpu and HAS_GPU_CB:
            features_norm_gpu = cxs.csr_matrix(features_norm)
            feat_t_gpu = cxs.csr_matrix(feat_t)
            for i in iterator:
                end = min(i + batch_size, n_samples)
                batch = features_norm_gpu[i:end]
                sim_batch = batch.dot(feat_t_gpu)
                sim_batch.data[sim_batch.data < self.config.similarity_threshold] = 0
                sim_batch.eliminate_zeros()
                sim_batch_cpu = sim_batch.get()
                for row_idx in range(sim_batch_cpu.shape[0]):
                    start_ptr = sim_batch_cpu.indptr[row_idx]
                    end_ptr = sim_batch_cpu.indptr[row_idx + 1]
                    all_data.append(sim_batch_cpu.data[start_ptr:end_ptr])
                    all_indices.append(sim_batch_cpu.indices[start_ptr:end_ptr])
                    indptr.append(indptr[-1] + (end_ptr - start_ptr))
                del batch, sim_batch, sim_batch_cpu
                cp.get_default_memory_pool().free_all_blocks()
            del features_norm_gpu, feat_t_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            for i in iterator:
                end = min(i + batch_size, n_samples)
                batch = features_norm[i:end]
                sim_batch = batch.dot(feat_t)
                sim_batch.data[sim_batch.data < self.config.similarity_threshold] = 0
                sim_batch.eliminate_zeros()
                for row_idx in range(sim_batch.shape[0]):
                    start_ptr = sim_batch.indptr[row_idx]
                    end_ptr = sim_batch.indptr[row_idx + 1]
                    all_data.append(sim_batch.data[start_ptr:end_ptr])
                    all_indices.append(sim_batch.indices[start_ptr:end_ptr])
                    indptr.append(indptr[-1] + (end_ptr - start_ptr))
                del sim_batch
        data_arr = np.concatenate(all_data) if all_data else np.array([], dtype=np.float32)
        indices_arr = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
        indptr_arr = np.array(indptr, dtype=np.int32)
        return sp.csr_matrix((data_arr, indices_arr, indptr_arr), shape=(n_samples, n_samples))

    def _build_popularity_arrays(self, ratings_df: Optional[pd.DataFrame] = None):
        n_movies = len(self.movie_index)
        self.popularity_array = np.zeros(n_movies, dtype=np.float32)
        if ratings_df is not None:
            counts = ratings_df.groupby('movieId').size()
            for i, mid in enumerate(self.movie_ids):
                c = counts.get(mid, 0)
                self.popularity_array[i] = np.log1p(c)
            max_pop = self.popularity_array.max()
            if max_pop > 0:
                self.popularity_array /= max_pop
        self.soft_pop_arr = self.popularity_array.copy()

    def _predict_scores_from_items(
        self,
        item_indices: np.ndarray,
        candidate_idx: np.ndarray
    ) -> np.ndarray:
        if len(item_indices) == 0:
            return self.popularity_array[candidate_idx]
        sim_sum = self.similarity_matrix[item_indices].sum(axis=0)
        dense_scores = np.asarray(sim_sum).flatten()
        dense_scores = dense_scores / len(item_indices)
        return (
            (1.0 - self.config.pop_boost_weight) * dense_scores[candidate_idx]
            + self.config.pop_boost_weight * self.popularity_array[candidate_idx]
        )

    def _predict_scores_indexed(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        if user_idx == -1 or not self.user_liked_indices or user_idx >= len(self.user_liked_indices):
            return self.popularity_array[item_indices]
        liked_indices = self.user_liked_indices[user_idx]
        if len(liked_indices) == 0:
            return self.popularity_array[item_indices]
        sim_sum = self.similarity_matrix[liked_indices].sum(axis=0)
        dense_scores = np.asarray(sim_sum).ravel()
        if not np.any(dense_scores):
            return self.popularity_array[item_indices]
        dense_scores = dense_scores / len(liked_indices)
        scores = (
            (1.0 - self.config.pop_boost_weight) * dense_scores
            + self.config.pop_boost_weight * self.popularity_array
        )
        return scores[item_indices]

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        item_indices = self.movie_index.get_indexer(item_ids)
        valid_mask = item_indices != -1
        valid_item_indices = item_indices[valid_mask]
        user_idx = self.user_index.get_loc(user_id) if user_id in self.user_index else -1
        valid_scores = self._predict_scores_indexed(user_idx, valid_item_indices)
        final_scores = np.zeros(len(item_ids), dtype=np.float32)
        final_scores[valid_mask] = valid_scores
        return final_scores.tolist()

    def get_top_k_recommendations(
        self,
        user_id: int,
        watched_items: set,
        k: int = None,
        valid_items: Optional[List[int]] = None
    ) -> List[int]:
        if k is None:
            k = self.config.top_k_default

        watched_idx = self.movie_index.get_indexer(list(watched_items))
        watched_idx = watched_idx[watched_idx != -1]

        user_idx = self.user_index.get_loc(user_id) if user_id in self.user_index else -1
        liked_idx = self.user_liked_indices[user_idx] if user_idx != -1 and user_idx < len(self.user_liked_indices) else np.array([], dtype=np.int32)

        if valid_items is not None:
            valid_indices = self.movie_index.get_indexer(valid_items)
            valid_indices = valid_indices[valid_indices != -1]
            candidate_mask = np.zeros(len(self.movie_index), dtype=bool)
            candidate_mask[valid_indices] = True
        else:
            candidate_mask = np.ones(len(self.movie_index), dtype=bool)
        candidate_mask[watched_idx] = False
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            return []

        if len(liked_idx) > 0:
            scores = self._predict_scores_from_items(liked_idx, candidate_idx)
        else:
            scores = self.popularity_array[candidate_idx]

        if len(candidate_idx) > k:
            top_k_local_idx = np.argpartition(scores, -k)[-k:]
            top_k_local_idx = top_k_local_idx[np.argsort(scores[top_k_local_idx])[::-1]]
        else:
            top_k_local_idx = np.argsort(scores)[::-1]
        top_k_idx = candidate_idx[top_k_local_idx]
        return self.movie_ids[top_k_idx].tolist()

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int,
        valid_items: Optional[List[int]] = None
    ) -> List[List[int]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        n_movies = len(self.movie_ids)
        batch_size = len(user_ids)

        if valid_items is not None:
            valid_indices = self.movie_index.get_indexer(valid_items)
            valid_indices = valid_indices[valid_indices != -1]
            global_valid_mask = np.zeros(n_movies, dtype=bool)
            global_valid_mask[valid_indices] = True
        else:
            global_valid_mask = np.ones(n_movies, dtype=bool)

        user_to_liked_idx = {}
        for i, uid in enumerate(user_ids):
            user_idx = self.user_index.get_loc(uid) if uid in self.user_index else -1
            if user_idx != -1 and user_idx < len(self.user_liked_indices):
                user_to_liked_idx[i] = self.user_liked_indices[user_idx]
            else:
                user_to_liked_idx[i] = np.array([], dtype=np.int32)

        row_inds = []
        col_inds = []
        data_vals = []
        for i in range(batch_size):
            liked = user_to_liked_idx[i]
            if len(liked) > 0:
                weight = 1.0 / len(liked)
                row_inds.extend([i] * len(liked))
                col_inds.extend(liked)
                data_vals.extend([weight] * len(liked))

        if data_vals:
            user_item_sparse = csr_matrix(
                (data_vals, (row_inds, col_inds)),
                shape=(batch_size, n_movies),
                dtype=np.float32
            )
            dense_scores = user_item_sparse.dot(self.similarity_matrix)
            dense_scores = dense_scores.toarray() if issparse(dense_scores) else dense_scores
        else:
            dense_scores = np.zeros((batch_size, n_movies), dtype=np.float32)

        scores = (
            (1.0 - self.config.pop_boost_weight) * dense_scores
            + self.config.pop_boost_weight * self.popularity_array[None, :]
        )

        mask = np.tile(global_valid_mask, (batch_size, 1))
        for i, watched_set in enumerate(watched_items_list):
            watched_idx = self.movie_index.get_indexer(list(watched_set))
            watched_idx = watched_idx[watched_idx != -1]
            if len(watched_idx) > 0:
                mask[i, watched_idx] = False
        scores[~mask] = -np.inf

        top_k_recs = []
        for i in range(batch_size):
            row_scores = scores[i]
            valid = row_scores != -np.inf
            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                top_k_recs.append([])
                continue
            valid_scores = row_scores[valid_idx]
            if len(valid_idx) > k:
                part = np.argpartition(-valid_scores, k - 1)[:k]
                top_k_local = part[np.argsort(-valid_scores[part])]
            else:
                top_k_local = np.argsort(-valid_scores)
            top_k_recs.append(self.movie_ids[valid_idx[top_k_local]].tolist())
        return top_k_recs

    def show_user_profile_and_recommendations(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        k: int = 10,
        top_rated_count: int = 5,
        reasons_count: int = 3
    ) -> None:
        movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
        print("=" * 80)
        print(f"USER PROFILE: {user_id}")
        print("=" * 80)
        user_history = ratings_df[ratings_df['userId'] == user_id]
        if user_history.empty:
            print(f"User {user_id} not found in the database.")
            return
        print("\nGeneral Statistics:")
        print(f"   - Total ratings: {len(user_history)}")
        print(f"   - Average rating: {user_history['rating'].mean():.2f} / 5.0")
        print(f"   - Unique movies rated: {user_history['movieId'].nunique()}")
        high_rated = user_history[user_history['rating'] >= 4.0].sort_values('rating', ascending=False)
        if not high_rated.empty:
            limit = min(top_rated_count, len(high_rated))
            print(f"\nTOP {limit} FAVORITE MOVIES (Rating >= 4.0):")
            print("-" * 80)
            for i, (_, row) in enumerate(high_rated.head(limit).iterrows(), 1):
                movie_id = row['movieId']
                title = movie_titles.get(movie_id, f"Unknown (ID: {movie_id})")
                rating = row['rating']
                print(f"{i:2}. {title:<50} [Rating: {rating}/5.0]")
        else:
            print("\nNo highly rated movies (>= 4.0) found for this user.")
        print("\n" + "=" * 80)
        print(f"PERSONALIZED RECOMMENDATIONS FOR USER {user_id}")
        print("=" * 80)
        watched_items = set(user_history['movieId'].values)
        liked_items = set(user_history[user_history['rating'] >= 4.0]['movieId'].values)
        recommendations = self.get_top_k_with_titles(user_id, watched_items, k)
        if not recommendations:
            print("No recommendations could be generated.")
            return
        for i, rec in enumerate(recommendations, 1):
            rec_id = rec['movieId']
            rec_title = rec['title']
            rec_score = rec['score']
            print(f"\n{i:2}. {rec_title}")
            print(f"    Predicted Relevance Score: {rec_score:.4f}")
            reasons = self.explain_recommendation(rec_id, liked_items, top_n_reasons=reasons_count)
            if reasons:
                print("    Recommended because you liked:")
                for reason in reasons:
                    reason_id = reason['movie_id']
                    reason_title = movie_titles.get(reason_id, f"Unknown (ID: {reason_id})")
                    similarity = reason['similarity']
                    user_rating = user_history[user_history['movieId'] == reason_id]['rating'].values
                    rating_str = f" (You rated: {user_rating[0]}/5.0)" if len(user_rating) > 0 else ""
                    print(f"        - {reason_title}{rating_str} [Similarity: {similarity:.4f}]")
            else:
                print("    (No specific reasons found in your watch history)")
        print("\n" + "=" * 80)

    def get_top_k_with_titles(self, user_id: int, watched_items: set, k: int = 10) -> List[Dict[str, Any]]:
        movie_ids = self.get_top_k_recommendations(user_id, watched_items, k)
        scores = self.predict_scores(user_id, movie_ids)
        result = []
        for mid, score in zip(movie_ids, scores):
            result.append({
                'movieId': mid,
                'title': self.movie_id_to_title.get(mid, str(mid)),
                'score': score
            })
        return result

    def save_artifacts(
        self,
        similarity_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        preprocessors_path: Optional[str] = None
    ):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving artifacts.")
        start = time.perf_counter()
        cpu_start = time.process_time()
        artifacts_dir = Path(self.config.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npz")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")
        save_npz(similarity_path, self.similarity_matrix)
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'movie_index': self.movie_index,
                'movie_ids': self.movie_ids,
                'user_index': self.user_index
            }, f)
        with open(preprocessors_path, 'wb') as f:
            pickle.dump({
                'tfidf_main_actor': self.tfidf_main_actor,
                'tfidf_director': self.tfidf_director,
                'tfidf_cast': self.tfidf_cast,
                'scaler_runtime': self.scaler_runtime,
                'scaler_year': self.scaler_year,
                'scaler_main_actor_rating': self.scaler_main_actor_rating,
                'scaler_director_rating': self.scaler_director_rating,
                'qt_year': self.qt_year,
                'config': self.config,
                'feature_matrix_norm': self.feature_matrix_norm,
                'user_profiles': self.user_profiles,
                'user_liked_indices': self.user_liked_indices,
                'popularity_array': self.popularity_array,
                'soft_pop_arr': self.soft_pop_arr
            }, f)
        extra = {
            "similarity_file": str(Path(similarity_path).name),
            "mapping_file": str(Path(mapping_path).name),
            "preprocessors_file": str(Path(preprocessors_path).name)
        }
        self.step_logger.log_step("SaveArtifacts", start, cpu_start, extra)

    def load_artifacts(
        self,
        similarity_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        preprocessors_path: Optional[str] = None
    ):
        start = time.perf_counter()
        cpu_start = time.process_time()
        artifacts_dir = Path(self.config.artifacts_dir)
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npz")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")
        if similarity_path.endswith('.npy'):
            dense_matrix = np.load(similarity_path, allow_pickle=True)
            self.similarity_matrix = csr_matrix(dense_matrix)
        else:
            self.similarity_matrix = load_npz(similarity_path)
        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)
            self.movie_index = mappings['movie_index']
            self.movie_ids = mappings['movie_ids']
            self.user_index = mappings.get('user_index', pd.Index([]))
        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
            self.tfidf_main_actor = preprocessors['tfidf_main_actor']
            self.tfidf_director = preprocessors['tfidf_director']
            self.tfidf_cast = preprocessors['tfidf_cast']
            self.scaler_runtime = preprocessors['scaler_runtime']
            self.scaler_year = preprocessors['scaler_year']
            self.scaler_main_actor_rating = preprocessors['scaler_main_actor_rating']
            self.scaler_director_rating = preprocessors['scaler_director_rating']
            self.qt_year = preprocessors['qt_year']
            self.config = preprocessors['config']
            self.feature_matrix_norm = preprocessors.get('feature_matrix_norm', None)
            if self.feature_matrix_norm is None:
                raw = preprocessors.get('feature_matrix', None)
                if raw is not None:
                    self.feature_matrix_norm = normalize(raw, norm='l2', axis=1, copy=False)
            self.user_profiles = preprocessors.get('user_profiles', None)
            self.user_liked_indices = preprocessors.get('user_liked_indices', [])
            self.popularity_array = preprocessors.get('popularity_array', np.array([], dtype=np.float32))
            self.soft_pop_arr = preprocessors.get('soft_pop_arr', self.popularity_array.copy())
        self.is_fitted = True
        extra = {
            "similarity_file": str(Path(similarity_path).name),
            "mapping_file": str(Path(mapping_path).name),
            "preprocessors_file": str(Path(preprocessors_path).name)
        }
        self.step_logger.log_step("LoadArtifacts", start, cpu_start, extra)

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        movie_idx = self.movie_index.get_loc(movie_id) if movie_id in self.movie_index else -1
        if movie_idx == -1:
            return []
        row = self.similarity_matrix[movie_idx]
        cols = row.indices
        datas = row.data
        if len(cols) == 0:
            return []
        sort_order = np.argsort(datas)[::-1]
        sorted_cols = cols[sort_order]
        sorted_datas = datas[sort_order]
        results = []
        for idx, score in zip(sorted_cols, sorted_datas):
            if idx == movie_idx:
                continue
            if len(results) >= top_n:
                break
            results.append({
                'movie_id': int(self.movie_ids[idx]),
                'similarity': float(score)
            })
        return results

    def explain_recommendation(
        self,
        movie_id: int,
        liked_items: set,
        top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        movie_idx = self.movie_index.get_loc(movie_id) if movie_id in self.movie_index else -1
        if movie_idx == -1:
            return []
        liked_indices = self.movie_index.get_indexer(list(liked_items))
        liked_indices = liked_indices[liked_indices != -1]
        if len(liked_indices) == 0:
            return []
        row = self.similarity_matrix[movie_idx]
        sims = row[:, liked_indices].toarray()[0]
        valid_mask = sims > 0
        valid_sims = sims[valid_mask]
        valid_liked_idx = liked_indices[valid_mask]
        if len(valid_sims) == 0:
            return []
        sort_order = np.argsort(valid_sims)[::-1][:top_n_reasons]
        top_sims = valid_sims[sort_order]
        top_liked_idx = valid_liked_idx[sort_order]
        reasons = []
        for lid, sim in zip(top_liked_idx, top_sims):
            reasons.append({
                'movie_id': int(self.movie_ids[lid]),
                'similarity': float(sim)
            })
        return reasons

    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> 'ContentBasedRecommender':
        total_start = time.perf_counter()
        total_cpu = time.process_time()
        logger.info("Starting model fitting process")
        required_cols = ['movieId', 'main_actor', 'director', 'cast', 'runtime', 'year', 'keywords']
        missing_cols = [col for col in required_cols if col not in movies_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in movies_df: {missing_cols}")
        self.step_logger.log_step("Validation", total_start, total_cpu)

        df = movies_df.copy()
        self._preprocess_numerical_features(df)
        self._preprocess_actor_director_ratings(df)
        self.step_logger.log_step("Preprocessing", total_start, total_cpu)

        features_matrix = self._build_feature_matrix(df)
        self.feature_matrix_norm = normalize(features_matrix, norm='l2', axis=1, copy=False)
        del features_matrix
        self.step_logger.log_step("Feature Matrix", total_start, total_cpu)

        self.similarity_matrix = self._compute_similarity_matrix(self.feature_matrix_norm)
        self.step_logger.log_step("Similarity Matrix", total_start, total_cpu)

        self.movie_ids = df['movieId'].values.astype(np.int64)
        self.movie_index = pd.Index(self.movie_ids)
        self.step_logger.log_step("Index Mappings", total_start, total_cpu)

        self._build_popularity_arrays(ratings_df)
        self.user_index = pd.Index([])
        self.user_profiles = None
        self.user_liked_indices = []
        if ratings_df is not None:
            good_ratings_df = ratings_df[ratings_df['rating'] >= 4.0]
            valid_df = good_ratings_df[good_ratings_df['movieId'].isin(self.movie_ids)]
            if not valid_df.empty:
                movie_idx_map = dict(zip(self.movie_ids, range(len(self.movie_ids))))
                user_groups = valid_df.groupby('userId')['movieId'].apply(list)
                users = []
                liked_indices_list = []
                for user_id, movie_list in user_groups.items():
                    indices = [movie_idx_map[mid] for mid in movie_list if mid in movie_idx_map]
                    if indices:
                        users.append(user_id)
                        liked_indices_list.append(np.array(indices, dtype=np.int32))
                self.user_index = pd.Index(users)
                self.user_liked_indices = liked_indices_list

        if 'title' in movies_df.columns:
            self.movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
        self.step_logger.log_step("User Profiles & Metadata", total_start, total_cpu)

        self.is_fitted = True
        if self.config.logging.log_data_shapes:
            fm = self.feature_matrix_norm
            sm = self.similarity_matrix
            logger.info(
                f"Feature matrix (norm): shape={fm.shape}, nnz={fm.nnz}, density={fm.nnz/(fm.shape[0]*fm.shape[1]):.4%}"
            )
            logger.info(
                f"Similarity matrix: shape={sm.shape}, nnz={sm.nnz}, density={sm.nnz/(sm.shape[0]*sm.shape[1]):.4%}"
            )
        total_wall = time.perf_counter() - total_start
        total_cpu_elapsed = time.process_time() - total_cpu
        mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        logger.info(f"Fitting complete | total_wall={total_wall:.2f}s total_cpu={total_cpu_elapsed:.2f}s final_rss={mem_final:.1f}MB")
        return self