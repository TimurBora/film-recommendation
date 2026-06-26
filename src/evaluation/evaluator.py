import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    def __init__(
        self,
        models: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        relevance_threshold: float = 4.0,
        user_sample_size: Optional[int] = None,
        random_state: int = 42
    ):
        self.models = models
        self.train_df = train_df
        self.test_df = test_df
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        self.user_sample_size = user_sample_size
        
        self.test_users = test_df['userId'].unique()
        if user_sample_size is not None and user_sample_size < len(self.test_users):
            np.random.seed(random_state)
            self.test_users = np.random.choice(
                self.test_users, 
                size=user_sample_size, 
                replace=False
            )
        
        self.test_relevant_items = {}
        for user_id in self.test_users:
            user_test = test_df[test_df['userId'] == user_id]
            relevant = user_test[user_test['rating'] >= relevance_threshold]['movieId'].tolist()
            self.test_relevant_items[user_id] = relevant
        
        self.all_items = set(test_df['movieId'].unique())
        self.item_popularity = dict(train_df.groupby('movieId').size())
        
        self.results = None

    def _precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        if k == 0 or not recommended:
            return 0.0
        rec_k = recommended[:k]
        return len(set(rec_k) & set(relevant)) / k

    def _recall_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        if not relevant or not recommended:
            return 0.0
        rec_k = recommended[:k]
        return len(set(rec_k) & set(relevant)) / len(relevant)

    def _ndcg_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        if not recommended:
            return 0.0
        rec_k = recommended[:k]
        dcg = 0.0
        for i, item in enumerate(rec_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0

    def _map_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        if not recommended:
            return 0.0
        rec_k = recommended[:k]
        hits = 0
        sum_precs = 0.0
        for i, item in enumerate(rec_k):
            if item in relevant:
                hits += 1
                sum_precs += hits / (i + 1)
        return sum_precs / min(len(relevant), k) if relevant else 0.0

    def _mrr(self, recommended: List[int], relevant: List[int]) -> float:
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def _catalog_coverage(self, all_recommended: List[List[int]]) -> float:
        unique_recommended = set()
        for rec_list in all_recommended:
            unique_recommended.update(rec_list)
        return len(unique_recommended) / len(self.all_items) if self.all_items else 0.0

    def _novelty(self, recommended: List[int]) -> float:
        total_interactions = sum(self.item_popularity.values())
        if total_interactions == 0 or not recommended:
            return 0.0
        
        novelties = []
        for item in recommended:
            pop = self.item_popularity.get(item, 1)
            prob = pop / total_interactions
            novelties.append(-np.log2(prob))
        return np.mean(novelties) if novelties else 0.0

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20
    ) -> Dict[str, Any]:
        logger.info(f"Evaluating model: {model_name}")
        
        user_metrics = {k: {
            'precision': [], 'recall': [], 'ndcg': [], 
            'map': [], 'mrr': [], 'novelty': []
        } for k in k_values}
        
        all_recommendations = []
        
        for user_id in tqdm(self.test_users, desc=f"{model_name}", leave=False):
            watched_items = set(
                self.train_df[self.train_df['userId'] == user_id]['movieId'].tolist()
            )
            
            try:
                recommendations = model.get_top_k_recommendations(
                    user_id=int(user_id),
                    watched_items=watched_items,
                    k=max_recommendations
                )
            except Exception as e:
                logger.warning(f"Error for user {user_id} in {model_name}: {e}")
                continue
            
            if not recommendations:
                continue
            
            relevant = self.test_relevant_items.get(user_id, [])
            
            for k in k_values:
                user_metrics[k]['precision'].append(
                    self._precision_at_k(recommendations, relevant, k)
                )
                user_metrics[k]['recall'].append(
                    self._recall_at_k(recommendations, relevant, k)
                )
                user_metrics[k]['ndcg'].append(
                    self._ndcg_at_k(recommendations, relevant, k)
                )
                user_metrics[k]['map'].append(
                    self._map_at_k(recommendations, relevant, k)
                )
            
            user_metrics[max_recommendations]['mrr'].append(
                self._mrr(recommendations, relevant)
            )
            
            for k in k_values:
                user_metrics[k]['novelty'].append(
                    self._novelty(recommendations[:k])
                )
            
            all_recommendations.append(recommendations[:max_recommendations])
        
        coverage = self._catalog_coverage(all_recommendations)
        
        results = []
        for k in k_values:
            metrics = user_metrics[k]
            results.append({
                'model': model_name,
                'k': k,
                'precision': np.mean(metrics['precision']) if metrics['precision'] else 0.0,
                'recall': np.mean(metrics['recall']) if metrics['recall'] else 0.0,
                'ndcg': np.mean(metrics['ndcg']) if metrics['ndcg'] else 0.0,
                'map': np.mean(metrics['map']) if metrics['map'] else 0.0,
                'mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0.0,
                'novelty': np.mean(metrics['novelty']) if metrics['novelty'] else 0.0,
                'coverage': coverage,
                'n_users': len(metrics['precision'])
            })
        
        return results

    def evaluate_all_models(
        self,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20
    ) -> pd.DataFrame:
        all_results = []
        
        for model_name, model in self.models.items():
            results = self.evaluate_model(
                model_name, model, k_values, max_recommendations
            )
            all_results.extend(results)
        
        self.results = pd.DataFrame(all_results)
        return self.results

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("No results available. Run evaluate_all_models() first.")
        return self.results