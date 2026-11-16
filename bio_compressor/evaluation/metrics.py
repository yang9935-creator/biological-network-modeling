import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score, max_error
)

class BiologicalMetrics:
    """Domain-specific metrics for biological systems"""
    
    @staticmethod
    def dose_response_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            doses: np.ndarray) -> Dict[str, float]:
        """Metrics specific to dose-response curves"""
        metrics = {}
        
        # EC50 shift
        def estimate_ec50(y):
            half_max = (y.max() - y.min()) / 2 + y.min()
            idx = np.argmin(np.abs(y - half_max))
            return doses[idx]
        
        ec50_true = estimate_ec50(y_true)
        ec50_pred = estimate_ec50(y_pred)
        metrics['ec50_shift'] = abs(ec50_pred - ec50_true)
        metrics['ec50_ratio'] = ec50_pred / ec50_true if ec50_true > 0 else np.nan
        
        # Dynamic range
        metrics['dynamic_range_true'] = y_true.max() - y_true.min()
        metrics['dynamic_range_pred'] = y_pred.max() - y_pred.min()
        metrics['dynamic_range_ratio'] = (
            metrics['dynamic_range_pred'] / metrics['dynamic_range_true']
        )
        
        # Hill coefficient
        def estimate_hill(y):
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
            log_doses = np.log10(doses + 1e-8)
            log_ratio = np.log10(y_norm / (1 - y_norm + 1e-8))
            valid = ~(np.isinf(log_ratio) | np.isnan(log_ratio))
            if valid.sum() > 1:
                slope, _, _, _, _ = stats.linregress(
                    log_doses[valid], log_ratio[valid]
                )
                return slope
            return np.nan
        
        metrics['hill_coefficient_true'] = estimate_hill(y_true)
        metrics['hill_coefficient_pred'] = estimate_hill(y_pred)
        
        return metrics
    
    @staticmethod
    def time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           times: np.ndarray) -> Dict[str, float]:
        """Metrics for time series predictions"""
        metrics = {}
        
        # Peak time
        peak_time_true = times[np.argmax(y_true)]
        peak_time_pred = times[np.argmax(y_pred)]
        metrics['peak_time_shift'] = abs(peak_time_pred - peak_time_true)
        
        # Peak amplitude
        metrics['peak_amplitude_true'] = y_true.max()
        metrics['peak_amplitude_pred'] = y_pred.max()
        metrics['peak_amplitude_ratio'] = (
            metrics['peak_amplitude_pred'] / metrics['peak_amplitude_true']
        )
        
        # Area under curve
        metrics['auc_true'] = np.trapz(y_true, times)
        metrics['auc_pred'] = np.trapz(y_pred, times)
        metrics['auc_ratio'] = metrics['auc_pred'] / metrics['auc_true']
        
        # Rise time (10% to 90%)
        def get_rise_time(y):
            y_min, y_max = y.min(), y.max()
            y_10 = y_min + 0.1 * (y_max - y_min)
            y_90 = y_min + 0.9 * (y_max - y_min)
            
            t_10 = times[np.argmin(np.abs(y - y_10))]
            t_90 = times[np.argmin(np.abs(y - y_90))]
            return t_90 - t_10
        
        metrics['rise_time_true'] = get_rise_time(y_true)
        metrics['rise_time_pred'] = get_rise_time(y_pred)
        metrics['rise_time_ratio'] = (
            metrics['rise_time_pred'] / metrics['rise_time_true']
        )
        
        return metrics
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, float]:
        """Classification metrics for binary outcomes"""
        from sklearn.metrics import confusion_matrix, roc_auc_score
        
        y_pred_binary = (y_pred > threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp + 1e-8),
            'recall': tp / (tp + fn + 1e-8),
            'specificity': tn / (tn + fp + 1e-8),
            'f1_score': 2 * tp / (2 * tp + fp + fn + 1e-8),
            'mcc': (tp * tn - fp * fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8)
        }
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['roc_auc'] = np.nan
        
        return metrics


class ComprehensiveMetrics:
    """Complete set of evaluation metrics"""
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray,
                     feature_names: Optional[List[str]] = None) -> Dict:
        """Calculate all available metrics"""
        
        metrics = {}
        
        # Basic regression metrics
        metrics['r2'] = r2_score(y_true, y_pred, multioutput='raw_values')
        metrics['r2_mean'] = np.mean(metrics['r2'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred, multioutput='raw_values')
        metrics['max_error'] = max_error(y_true, y_pred)
        
        # Per-output metrics
        if feature_names and y_true.ndim > 1:
            metrics['per_output'] = {}
            for i, name in enumerate(feature_names):
                metrics['per_output'][name] = {
                    'r2': metrics['r2'][i],
                    'mae': metrics['mae'][i],
                    'rmse': metrics['rmse'][i],
                    'explained_variance': metrics['explained_variance'][i]
                }
        
        # Error distribution
        errors = y_pred - y_true
        metrics['error_mean'] = np.mean(errors)
        metrics['error_std'] = np.std(errors)
        metrics['error_skew'] = stats.skew(errors.flatten())
        metrics['error_kurtosis'] = stats.kurtosis(errors.flatten())
        
        # Correlation
        if y_true.ndim == 1:
            metrics['pearson'] = stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]
            metrics['spearman'] = stats.spearmanr(y_true.flatten(), y_pred.flatten())[0]
        else:
            pearson = []
            spearman = []
            for i in range(y_true.shape[1]):
                p, _ = stats.pearsonr(y_true[:, i], y_pred[:, i])
                s, _ = stats.spearmanr(y_true[:, i], y_pred[:, i])
                pearson.append(p)
                spearman.append(s)
            metrics['pearson'] = pearson
            metrics['spearman'] = spearman
        
        return metrics