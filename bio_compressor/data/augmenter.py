import pandas as pd
import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp1d

class DataAugmenter:
    """Data augmentation for biological datasets"""
    
    @staticmethod
    def add_noise(X: np.ndarray, y: np.ndarray, 
                  noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to data"""
        X_noise = X + np.random.randn(*X.shape) * noise_level * np.std(X, axis=0)
        y_noise = y + np.random.randn(*y.shape) * noise_level * np.std(y, axis=0)
        return X_noise, y_noise
    
    @staticmethod
    def interpolate_time_series(df, time_col: str, 
                               target_times: np.ndarray) -> pd.DataFrame:
        """Interpolate time series data to target time points"""
        results = []
        
        for condition, group in df.groupby([c for c in df.columns if c != time_col]):
            # Sort by time
            group = group.sort_values(time_col)
            
            # Create interpolation function
            f_interp = interp1d(group[time_col], 
                               group.drop(columns=[time_col]).values.T,
                               kind='cubic', 
                               bounds_error=False, 
                               fill_value='extrapolate')
            
            # Interpolate at target times
            interpolated = f_interp(target_times).T
            
            # Create result dataframe
            for t, vals in zip(target_times, interpolated):
                result = dict(zip(group.columns, condition))
                result[time_col] = t
                for i, col in enumerate(group.drop(columns=[time_col]).columns):
                    result[col] = vals[i]
                results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def generate_synthetic_data(model, X_template: np.ndarray, 
                               n_samples: int = 1000) -> np.ndarray:
        """Generate synthetic input data by perturbing template"""
        X_synthetic = []
        
        for _ in range(n_samples):
            # Randomly select a template sample
            idx = np.random.randint(len(X_template))
            X_base = X_template[idx].copy()
            
            # Add perturbation
            perturbation = np.random.randn(*X_base.shape) * 0.1 * np.std(X_template, axis=0)
            X_synth = X_base + perturbation
            
            # Clip to reasonable range
            X_synth = np.clip(X_synth, X_template.min(axis=0), X_template.max(axis=0))
            X_synthetic.append(X_synth)
        
        return np.array(X_synthetic)