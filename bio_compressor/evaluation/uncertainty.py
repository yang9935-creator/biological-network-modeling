import torch
import numpy as np
from typing import Tuple, Dict
from scipy import stats

class UncertaintyEstimator:
    """Estimate prediction uncertainty using various methods"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def mc_dropout(self, X: np.ndarray, 
                  n_iterations: int = 100,
                  dropout_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty estimation
        """
        self.model.train()  # Enable dropout
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []
        
        for _ in range(n_iterations):
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Mean and standard deviation
        pred_mean = predictions.mean(axis=0)
        pred_std = predictions.std(axis=0)
        
        return pred_mean, pred_std
    
    def ensemble_uncertainty(self, models: list, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble-based uncertainty estimation
        """
        predictions = []
        
        for model in models:
            model.eval()
            model.to(self.device)
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Ensemble mean and standard deviation
        pred_mean = predictions.mean(axis=0)
        pred_std = predictions.std(axis=0)
        
        return pred_mean, pred_std
    
    def prediction_intervals(self, X: np.ndarray,
                           confidence_level: float = 0.95,
                           n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute prediction intervals using bootstrap
        """
        from sklearn.utils import resample
        
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            indices = resample(np.arange(len(X)))
            X_boot = X[indices]
            
            X_tensor = torch.FloatTensor(X_boot).to(self.device)
            
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Percentile-based intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        median = np.percentile(predictions, 50, axis=0)
        
        return {
            'median': median,
            f'lower_{confidence_level:.0%}': lower_bound,
            f'upper_{confidence_level:.0%}': upper_bound
        }
    
    def calibration_error(self, y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_std: np.ndarray,
                         n_bins: int = 10) -> Dict[str, float]:
        """
        Compute calibration error for uncertainty estimates
        """
        z_scores = (y_true - y_pred) / (y_std + 1e-8)
        z_scores = z_scores.flatten()
        
        # Theoretical quantiles
        theoretical_quantiles = np.linspace(0, 1, n_bins + 1)
        
        # Empirical quantiles
        empirical_quantiles = np.percentile(z_scores, theoretical_quantiles * 100)
        
        # Expected quantiles for standard normal
        expected_quantiles = stats.norm.ppf(theoretical_quantiles)
        
        # Calibration error
        calibration_error = np.mean(np.abs(empirical_quantiles - expected_quantiles))
        
        # Coverage probability
        coverage = {}
        for alpha in [0.5, 0.8, 0.9, 0.95]:
            z = stats.norm.ppf(1 - (1 - alpha) / 2)
            lower = y_pred - z * y_std
            upper = y_pred + z * y_std
            coverage[f'coverage_{alpha:.0%}'] = np.mean(
                (y_true >= lower) & (y_true <= upper)
            )
        
        return {
            'calibration_error': calibration_error,
            **coverage
        }