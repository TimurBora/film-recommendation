import pandas as pd
import numpy as np
from typing import Dict, List


class DataSplitter:
    def __init__(self, ratings_df: pd.DataFrame, relevance_threshold: float = 3.0):
        self.relevance_threshold = relevance_threshold
        self.df_user_time = (
            ratings_df.copy()
            .sort_values(by=['userId', 'timestamp'])
            .reset_index(drop=True)
        )
        self.df_time = (
            ratings_df.copy()
            .sort_values(by='timestamp')
            .reset_index(drop=True)
        )

    def leave_one_out(self) -> Dict[str, pd.DataFrame]:
        df = self.df_user_time
        group_size = df.groupby('userId')['userId'].transform('size').values
        cumcount = df.groupby('userId').cumcount().values

        n_lt_2 = group_size < 2
        n_eq_2 = group_size == 2
        n_gt_2 = group_size > 2

        train_mask = (
            n_lt_2 |
            (n_eq_2 & (cumcount == 0)) |
            (n_gt_2 & (cumcount < group_size - 2))
        )
        val_mask = n_gt_2 & (cumcount == group_size - 2)
        test_mask = (group_size >= 2) & (
            (n_eq_2 & (cumcount == 1)) |
            (n_gt_2 & (cumcount == group_size - 1))
        )

        train = df[train_mask]
        val = df[val_mask] if val_mask.any() else pd.DataFrame()
        test = df[test_mask]
        return {'train': train, 'val': val, 'test': test}

    def global_temporal_split(self, train_ratio: float = 0.8) -> Dict[str, pd.DataFrame]:
        df = self.df_time
        split_idx = int(len(df) * train_ratio)
        return {
            'train': df.iloc[:split_idx],
            'val': pd.DataFrame(),
            'test': df.iloc[split_idx:]
        }

    def chronological_kfold(self, k: int = 5) -> List[Dict[str, pd.DataFrame]]:
        df = self.df_user_time
        group_size = df.groupby('userId')['userId'].transform('size').values
        cumcount = df.groupby('userId').cumcount().values

        valid_mask = group_size >= 2
        fold_size = group_size // k
        can_split = (fold_size > 0) & valid_mask

        fold_assignment = np.full(len(df), -1, dtype=np.int8)
        fold_assignment[can_split] = np.minimum(
            cumcount[can_split] // fold_size[can_split], k - 1
        )

        folds = []
        for fold_idx in range(k):
            test_mask = fold_assignment == fold_idx
            train_mask = ~test_mask
            folds.append({
                'train': df[train_mask],
                'val': pd.DataFrame(),
                'test': df[test_mask]
            })
        return folds