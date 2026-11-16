import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

class DataPreprocessor:
    """Advanced data preprocessing and quality control"""
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        mask = pd.Series(True, index=df.index)
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                col_mask = z_scores < threshold
                mask.loc[df[col].dropna().index] &= col_mask
        
        return df[mask]
    
    @staticmethod
    def detect_batch_effects(df: pd.DataFrame, 
                            batch_column: str,
                            feature_columns: List[str]) -> dict:
        """Detect batch effects using ANOVA"""
        from scipy.stats import f_oneway
        
        results = {}
        batches = df[batch_column].unique()
        
        for feature in feature_columns:
            if feature not in df.columns:
                continue
            
            batch_groups = [df[df[batch_column] == b][feature].dropna() 
                          for b in batches]
            batch_groups = [g for g in batch_groups if len(g) > 1]
            
            if len(batch_groups) >= 2:
                f_stat, p_value = f_oneway(*batch_groups)
                results[feature] = {'f_stat': f_stat, 'p_value': p_value}
        
        return results
    
    @staticmethod
    def normalize_by_control(df: pd.DataFrame, 
                            control_condition: dict,
                            feature_columns: List[str]) -> pd.DataFrame:
        """Normalize data by control condition"""
        df_norm = df.copy()
        
        # Find control samples
        mask = pd.Series(True, index=df.index)
        for col, val in control_condition.items():
            mask &= (df[col] == val)
        
        control_mean = df[mask][feature_columns].mean()
        
        for feature in feature_columns:
            df_norm[feature] = df[feature] / control_mean[feature]
        
        return df_norm
    
    @staticmethod
    def compute_feature_importance(X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> pd.DataFrame:
        """Compute mutual information between inputs and outputs"""
        mi_scores = []
        
        for i in range(y.shape[1]):
            mi = mutual_info_regression(X, y[:, i], random_state=42)
            mi_scores.append(mi)
        
        mi_matrix = np.array(mi_scores).T
        
        importance_df = pd.DataFrame(
            mi_matrix,
            columns=[f'output_{i}' for i in range(y.shape[1])],
            index=feature_names
        )
        
        return importance_df