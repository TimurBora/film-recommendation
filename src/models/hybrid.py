import time
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Protocol
from src.utils.logger import LoggingConfig, StepLogger


class RecommenderProtocol(Protocol):
    def fit(self, *args, **kwargs) -> Any: ...
    def predict_scores(self, *args, **kwargs) -> List[float]: ...
    def get_top_k_recommendations(self, *args, **kwargs) -> List[int]: ...
    def explain_recommendation(self, movie_id: int, liked_items: set, top_n_reasons: int = 3) -> List[Dict[str, Any]]: ...


class HybridRecommender:
    def __init__(
        self,
        cf_model: RecommenderProtocol,
        cb_model: RecommenderProtocol,
        alpha: float = 0.5,
        logging_config: LoggingConfig = LoggingConfig()
    ):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha 
        self.ratings_df = None
        self.movies_df = None
        self.config = logging_config
        self.step_logger = StepLogger(self.config)

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "HybridRecommender":
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        self.cf_model.fit(ratings_df)
        self.step_logger.log_step("CF model fitted", start_time, start_cpu)

        self.cb_model.fit(movies_df, ratings_df)
        self.step_logger.log_step("CB model fitted", start_time, start_cpu)

        self.ratings_df = ratings_df
        self.movies_df = movies_df

        extra = {
            "movies_shape": str(movies_df.shape),
            "ratings_shape": str(ratings_df.shape)
        }
        self.step_logger.log_step("Hybrid fit complete", start_time, start_cpu, extra)
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        cf_scores = np.array(self.cf_model.predict_scores(user_id, item_ids))
        cb_scores = np.array(self.cb_model.predict_scores(user_id, item_ids))
        combined_scores = self.alpha * cf_scores + (1 - self.alpha) * cb_scores
        return combined_scores.tolist()

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        if hasattr(self.cb_model, 'movie_id_to_idx'):
            all_movie_ids = set(self.cb_model.movie_id_to_idx.keys())
        elif self.movies_df is not None:
            all_movie_ids = set(self.movies_df['movieId'].unique())
        else:
            return self.cf_model.get_top_k_recommendations(user_id, watched_items, k)

        candidate_ids = list(all_movie_ids - set(watched_items))
        if not candidate_ids:
            return []

        scores = self.predict_scores(user_id, candidate_ids)
        scored_items = list(zip(scores, candidate_ids))
        scored_items.sort(reverse=True, key=lambda x: x[0])

        result = [mid for score, mid in scored_items[:k]]

        if self.config.log_per_prediction:
            extra = {"candidates": len(candidate_ids), "user": user_id, "k": k}
            self.step_logger.log_step("get_top_k_recommendations", start_time, start_cpu, extra)
        return result

    def explain_recommendation(self, movie_id: int, liked_items: set, top_n_reasons: int = 3) -> List[Dict[str, Any]]:
        cb_reasons = self.cb_model.explain_recommendation(movie_id, liked_items, top_n_reasons)
        cf_reasons = self.cf_model.explain_recommendation(movie_id, liked_items, top_n_reasons)

        combined = {}
        for reason in cb_reasons + cf_reasons:
            mid = reason['movie_id']
            if mid not in combined:
                combined[mid] = []
            combined[mid].append(reason['similarity'])

        final_reasons = []
        for mid, sims in combined.items():
            final_reasons.append({
                'movie_id': mid,
                'similarity': float(np.mean(sims))
            })

        final_reasons.sort(reverse=True, key=lambda x: x['similarity'])
        return final_reasons[:top_n_reasons]