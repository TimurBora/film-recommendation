import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)


class RecommendationVisualizer:
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.models = results_df['model'].unique()
        self.k_values = sorted(results_df['k'].unique())
        
        self.colors = sns.color_palette("husl", len(self.models))
        self.model_color_map = dict(zip(self.models, self.colors))

    def plot_metric_trend(
        self,
        metric: str,
        figsize: tuple = (10, 6),
        include_std: bool = True
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        
        for model in self.models:
            model_data = self.results_df[self.results_df['model'] == model]
            k_vals = model_data['k'].values
            metric_vals = model_data[metric].values
            
            ax.plot(
                k_vals, metric_vals, 
                marker='o', 
                linewidth=2.5, 
                markersize=8,
                label=model,
                color=self.model_color_map[model]
            )
            
            if include_std and 'n_users' in model_data.columns:
                n = model_data['n_users'].values
                if metric in ['precision', 'recall', 'ndcg', 'map', 'mrr']:
                    std_err = np.sqrt(metric_vals * (1 - metric_vals) / n)
                    ax.fill_between(
                        k_vals,
                        metric_vals - std_err,
                        metric_vals + std_err,
                        alpha=0.2,
                        color=self.model_color_map[model]
                    )
        
        ax.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} vs K', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.set_xticks(self.k_values)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_model_comparison(
        self,
        k: int,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 7)
    ) -> plt.Figure:
        if metrics is None:
            metrics = ['precision', 'recall', 'ndcg', 'map', 'mrr']
        
        k_data = self.results_df[self.results_df['k'] == k].copy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.8 / len(self.models)
        
        for i, model in enumerate(self.models):
            model_data = k_data[k_data['model'] == model]
            values = [model_data[m].values[0] if len(model_data) > 0 else 0 for m in metrics]
            
            offset = x + (i - (len(self.models) - 1) / 2) * width
            bars = ax.bar(
                offset, values, width, 
                label=model, 
                color=self.model_color_map[model],
                alpha=0.85,
                edgecolor='black',
                linewidth=0.8
            )
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    rotation=90
                )
        
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison at K={k}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=15, ha='right')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.set_ylim(0, max(k_data[metrics].max().max() * 1.15, 0.1))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig

    def plot_all_metrics_heatmap(
        self,
        k: int,
        figsize: tuple = (10, 8)
    ) -> plt.Figure:
        k_data = self.results_df[self.results_df['k'] == k].copy()
        
        metrics = ['precision', 'recall', 'ndcg', 'map', 'mrr', 'novelty', 'coverage']
        available_metrics = [m for m in metrics if m in k_data.columns]
        
        pivot_data = k_data.set_index('model')[available_metrics].T
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.4f',
            cmap='YlGnBu',
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            cbar_kws={'label': 'Score', 'shrink': 0.8}
        )
        
        ax.set_title(f'Performance Heatmap at K={k}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_coverage_novelty_tradeoff(
        self,
        k: int,
        figsize: tuple = (10, 7)
    ) -> plt.Figure:
        k_data = self.results_df[self.results_df['k'] == k].copy()
        
        if 'coverage' not in k_data.columns or 'novelty' not in k_data.columns:
            raise ValueError("Coverage and/or novelty metrics not available")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            k_data['coverage'],
            k_data['novelty'],
            s=200,
            c=[self.model_color_map[m] for m in k_data['model']],
            alpha=0.7,
            edgecolors='black',
            linewidth=2,
            zorder=3
        )
        
        for idx, row in k_data.iterrows():
            ax.annotate(
                row['model'],
                (row['coverage'], row['novelty']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlabel('Catalog Coverage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Novelty', fontsize=12, fontweight='bold')
        ax.set_title(f'Coverage vs Novelty Trade-off at K={k}', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, zorder=0)
        
        plt.tight_layout()
        return fig

    def plot_radar_chart(
        self,
        k: int,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (10, 10)
    ) -> plt.Figure:
        if metrics is None:
            metrics = ['precision', 'recall', 'ndcg', 'map', 'mrr']
        
        k_data = self.results_df[self.results_df['k'] == k].copy()
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        for model in self.models:
            model_data = k_data[k_data['model'] == model]
            values = [model_data[m].values[0] if len(model_data) > 0 else 0 for m in metrics]
            values += values[:1]
            
            ax.plot(
                angles, values, 
                'o-', 
                linewidth=2.5, 
                markersize=8,
                label=model,
                color=self.model_color_map[model]
            )
            ax.fill(angles, values, alpha=0.15, color=self.model_color_map[model])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics], size=11)
        ax.set_ylim(0, 1)
        ax.set_title(
            f'Radar Chart Comparison at K={k}', 
            fontsize=14, fontweight='bold', 
            pad=30, y=1.1
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def save_all_plots(self, output_dir: str = "evaluation_plots") -> None:
        import os
        from pathlib import Path
        
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        metrics = ['precision', 'recall', 'ndcg', 'map', 'mrr']
        
        for metric in metrics:
            fig = self.plot_metric_trend(metric)
            fig.savefig(path / f"{metric}_trend.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        for k in self.k_values:
            fig = self.plot_model_comparison(k)
            fig.savefig(path / f"comparison_k{k}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            fig = self.plot_all_metrics_heatmap(k)
            fig.savefig(path / f"heatmap_k{k}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"All plots saved to {output_dir}/")