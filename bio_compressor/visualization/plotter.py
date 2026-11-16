import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("husl")

class UniversalPlotter:
    """
    Universal visualization engine that adapts to any biological system
    """
    
    def __init__(self, config=None):
        self.config = config
        self.figures = {}
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        title: str = "Prediction Performance") -> plt.Figure:
        """Plot prediction vs true values"""
        
        n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
        
        if n_outputs == 1:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            ax.set_xlabel('True Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            # Add R² text
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.legend()
            
        else:
            # Multi-output: create subplots
            n_cols = min(3, n_outputs)
            n_rows = (n_outputs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for i in range(n_outputs):
                ax = axes[i]
                
                ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15)
                
                min_val = min(y_true[:, i].min(), y_pred[:, i].min())
                max_val = max(y_true[:, i].max(), y_pred[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
                
                name = feature_names[i] if feature_names else f'Output {i+1}'
                ax.set_xlabel(f'True {name}')
                ax.set_ylabel(f'Predicted {name}')
                
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Hide unused subplots
            for i in range(n_outputs, len(axes)):
                axes[i].set_visible(False)
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> plt.Figure:
        """Plot residual analysis"""
        
        residuals = y_pred - y_true
        
        n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
        
        if n_outputs == 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Residuals vs predicted
            axes[0].scatter(y_pred, residuals, alpha=0.5, s=15)
            axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5)
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted')
            
            # Histogram of residuals
            axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1].axvline(x=0, color='r', linestyle='--', linewidth=1.5)
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residual Distribution')
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals.flatten(), dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot')
            
        else:
            fig, axes = plt.subplots(n_outputs, 3, figsize=(15, 5*n_outputs))
            
            for i in range(n_outputs):
                res = residuals[:, i]
                pred = y_pred[:, i]
                
                # Residuals vs predicted
                axes[i, 0].scatter(pred, res, alpha=0.5, s=15)
                axes[i, 0].axhline(y=0, color='r', linestyle='--')
                name = feature_names[i] if feature_names else f'Output {i+1}'
                axes[i, 0].set_ylabel(f'{name} Residuals')
                if i == n_outputs - 1:
                    axes[i, 0].set_xlabel('Predicted Values')
                
                # Histogram
                axes[i, 1].hist(res, bins=30, edgecolor='black', alpha=0.7)
                axes[i, 1].axvline(x=0, color='r', linestyle='--')
                if i == n_outputs - 1:
                    axes[i, 1].set_xlabel('Residuals')
                
                # Q-Q plot
                stats.probplot(res, dist="norm", plot=axes[i, 2])
                if i == n_outputs - 1:
                    axes[i, 2].set_xlabel('Theoretical Quantiles')
        
        plt.tight_layout()
        return fig
    
    def plot_dose_response(self, X: np.ndarray, y_true: np.ndarray,
                          y_pred: np.ndarray, dose_feature_idx: int,
                          dose_name: str, fixed_conditions: Dict = None) -> plt.Figure:
        """Plot dose-response curves"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by dose
        doses = X[:, dose_feature_idx]
        sort_idx = np.argsort(doses)
        
        doses_sorted = doses[sort_idx]
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        
        # Plot individual points
        ax.scatter(doses_sorted, y_true_sorted, alpha=0.4, s=30, 
                  label='Experimental', color='blue')
        ax.scatter(doses_sorted, y_pred_sorted, alpha=0.4, s=30,
                  label='AI Prediction', color='red')
        
        # Plot smoothed curves
        from scipy.ndimage import gaussian_filter1d
        
        y_true_smooth = gaussian_filter1d(y_true_sorted, sigma=2)
        y_pred_smooth = gaussian_filter1d(y_pred_sorted, sigma=2)
        
        ax.plot(doses_sorted, y_true_smooth, 'b-', linewidth=2, label='Experimental (smoothed)')
        ax.plot(doses_sorted, y_pred_smooth, 'r--', linewidth=2, label='AI Prediction (smoothed)')
        
        # Log scale for doses
        if np.any(doses > 0):
            ax.set_xscale('log')
        
        ax.set_xlabel(f'{dose_name} (concentration)', fontsize=12)
        ax.set_ylabel('Response', fontsize=12)
        ax.set_title('Dose-Response Curve: AI vs Experimental', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add fixed conditions text
        if fixed_conditions:
            text_str = '\n'.join([f'{k}: {v}' for k, v in fixed_conditions.items()])
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def plot_training_history(self, history: Dict[str, list]) -> plt.Figure:
        """Plot training history"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R² curve
        if 'val_metrics' in history and len(history['val_metrics']) > 0:
            r2_scores = [m.get('r2', 0) for m in history['val_metrics']]
            axes[1].plot(r2_scores, color='green', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('R² Score')
            axes[1].set_title('Validation R²')
            axes[1].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in history and len(history['learning_rates']) > 0:
            axes[2].plot(history['learning_rates'], color='purple', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importance_scores: Dict[str, float],
                               title: str = "Feature Importance") -> plt.Figure:
        """Plot feature importance"""
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_scores) * 0.3)))
        
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        # Sort by importance
        idx = np.argsort(scores)
        features = [features[i] for i in idx]
        scores = [scores[i] for i in idx]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(features)))
        
        ax.barh(features, scores, color=colors)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (feature, score) in enumerate(zip(features, scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', 
                   va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty(self, X: np.ndarray, y_pred: np.ndarray,
                        y_std: np.ndarray, feature_idx: int = 0,
                        feature_name: str = 'Feature') -> plt.Figure:
        """Plot predictions with uncertainty"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by feature
        x_vals = X[:, feature_idx]
        sort_idx = np.argsort(x_vals)
        
        x_sorted = x_vals[sort_idx]
        y_sorted = y_pred[sort_idx]
        std_sorted = y_std[sort_idx]
        
        # Plot mean prediction
        ax.plot(x_sorted, y_sorted, 'b-', linewidth=2, label='Mean Prediction')
        
        # Plot uncertainty bands
        ax.fill_between(x_sorted, 
                       y_sorted - 1.96 * std_sorted,
                       y_sorted + 1.96 * std_sorted,
                       alpha=0.2, color='blue', label='95% Confidence')
        
        ax.fill_between(x_sorted,
                       y_sorted - std_sorted,
                       y_sorted + std_sorted,
                       alpha=0.3, color='blue', label='68% Confidence')
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_title('Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame,
                               columns: List[str],
                               title: str = "Correlation Matrix") -> plt.Figure:
        """Plot correlation matrix"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr = df[columns].corr()
        
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Set ticks
        ax.set_xticks(np.arange(len(columns)))
        ax.set_yticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(columns, fontsize=10)
        
        # Add correlation values
        for i in range(len(columns)):
            for j in range(len(columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, times: np.ndarray, 
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        output_names: List[str] = None) -> plt.Figure:
        """Plot time series predictions"""
        
        n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
        
        if n_outputs == 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(times, y_true, 'b-o', label='Experimental', markersize=4)
            ax.plot(times, y_pred, 'r--s', label='AI Prediction', markersize=4)
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Response', fontsize=12)
            ax.set_title('Time Series Prediction', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            n_cols = min(2, n_outputs)
            n_rows = (n_outputs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for i in range(n_outputs):
                ax = axes[i]
                
                ax.plot(times, y_true[:, i], 'b-o', label='Experimental', 
                       markersize=3, linewidth=1.5)
                ax.plot(times, y_pred[:, i], 'r--s', label='AI Prediction',
                       markersize=3, linewidth=1.5)
                
                name = output_names[i] if output_names else f'Output {i+1}'
                ax.set_xlabel('Time')
                ax.set_ylabel(name)
                ax.set_title(f'{name} - Time Series')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_outputs, len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   dpi: int = 300, format: str = 'png'):
        """Save figure to file"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', format=format)
        print(f"✅ Figure saved to {filename}")
        return filename