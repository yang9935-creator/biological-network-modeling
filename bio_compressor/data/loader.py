import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalDataLoader:
    """
    Universal data loader that automatically detects file format and column matching.
    Works with any biological system, any input/output features.
    """
    
    def __init__(self, config):
        self.config = config
        self.input_scaler = None
        self.output_scaler = None
        self.input_columns = []
        self.output_columns = []
        self.original_df = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load data from any supported file format"""
        data_path = self.config.data['path']
        path = Path(data_path)
        
        # Auto-detect file format
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif path.suffix == '.tsv':
            df = pd.read_csv(path, sep='\t')
        elif path.suffix == '.txt':
            df = pd.read_csv(path, sep=r'\s+')
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.feather':
            df = pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self.original_df = df
        
        # Auto-detect columns
        self._auto_detect_columns(df)
        
        # Extract features
        X = df[self.input_columns].values
        y = df[self.output_columns].values
        
        logger.info(f"✅ Loaded {len(df)} samples from {data_path}")
        logger.info(f"   Input features: {self.input_columns}")
        logger.info(f"   Output features: {self.output_columns}")
        
        return X, y, df
    
    def _auto_detect_columns(self, df: pd.DataFrame):
        """Automatically match config features to dataframe columns"""
        # Match input features
        for feature in self.config.input_features:
            matches = self._find_column_matches(feature.name, df.columns)
            if matches:
                self.input_columns.append(matches[0])
                logger.info(f"   Matched input '{feature.name}' → '{matches[0]}'")
            else:
                # Try case-insensitive match
                lower_cols = {col.lower(): col for col in df.columns}
                if feature.name.lower() in lower_cols:
                    self.input_columns.append(lower_cols[feature.name.lower()])
                    logger.info(f"   Matched input '{feature.name}' → '{lower_cols[feature.name.lower()]}'")
                else:
                    raise ValueError(f"Could not find column for input feature: {feature.name}")
        
        # Match output features
        for feature in self.config.output_features:
            matches = self._find_column_matches(feature.name, df.columns)
            if matches:
                self.output_columns.append(matches[0])
                logger.info(f"   Matched output '{feature.name}' → '{matches[0]}'")
            else:
                # Try case-insensitive match
                lower_cols = {col.lower(): col for col in df.columns}
                if feature.name.lower() in lower_cols:
                    self.output_columns.append(lower_cols[feature.name.lower()])
                    logger.info(f"   Matched output '{feature.name}' → '{lower_cols[feature.name.lower()]}'")
                else:
                    raise ValueError(f"Could not find column for output feature: {feature.name}")
    
    def _find_column_matches(self, feature_name: str, columns: list) -> list:
        """Find best matching column for a feature"""
        matches = []
        feature_lower = feature_name.lower()
        
        for col in columns:
            col_lower = col.lower()
            # Exact match
            if col_lower == feature_lower:
                matches.append(col)
            # Contains match
            elif feature_lower in col_lower:
                matches.append(col)
            # Abbreviation match
            elif feature_lower.replace('_', '') in col_lower.replace('_', ''):
                matches.append(col)
        
        return matches
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Complete data preparation pipeline"""
        
        # Apply transformations
        X_transformed = self._apply_transformations(X, is_input=True)
        y_transformed = self._apply_transformations(y, is_input=False)
        
        # Handle missing values
        X_transformed = self._handle_missing_values(X_transformed)
        y_transformed = self._handle_missing_values(y_transformed)
        
        # Scale features
        X_scaled, y_scaled = self._scale_data(X_transformed, y_transformed)
        
        # Split datasets
        split_data = self._split_data(X_scaled, y_scaled)
        
        return split_data
    
    def _apply_transformations(self, data: np.ndarray, is_input: bool = True) -> np.ndarray:
        """Apply feature-specific transformations"""
        features = self.config.input_features if is_input else self.config.output_features
        transformed = data.copy()
        
        for i, feature in enumerate(features):
            if i >= data.shape[1]:
                continue
            
            # Log transformations
            if feature.scale in ['log', 'log10']:
                transformed[:, i] = np.log10(np.maximum(data[:, i], 1e-8))
            elif feature.scale == 'log2':
                transformed[:, i] = np.log2(np.maximum(data[:, i], 1e-8))
            elif feature.scale == 'ln':
                transformed[:, i] = np.log(np.maximum(data[:, i], 1e-8))
            
            # Custom transform
            if feature.transform == 'log10':
                transformed[:, i] = np.log10(np.maximum(data[:, i], 1e-8))
            elif feature.transform == 'sqrt':
                transformed[:, i] = np.sqrt(np.maximum(data[:, i], 0))
            elif feature.transform == 'square':
                transformed[:, i] = data[:, i] ** 2
        
        return transformed
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle NaN and inf values"""
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("⚠️  Missing values detected, applying median imputation")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            return imputer.fit_transform(data)
        return data
    
    def _scale_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using appropriate scaler"""
        # Use RobustScaler for data with outliers
        from sklearn.preprocessing import RobustScaler
        
        self.input_scaler = RobustScaler(quantile_range=(5, 95))
        self.output_scaler = RobustScaler(quantile_range=(5, 95))
        
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Split into train/val/test sets"""
        test_size = self.config.training.test_size
        val_size = self.config.training.val_size
        seed = self.config.training.seed
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        
        # Second split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=seed
        )
        
        logger.info(f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def inverse_transform_output(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original units"""
        return self.output_scaler.inverse_transform(y_scaled)
    
    def inverse_transform_input(self, X_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled inputs back to original units"""
        return self.input_scaler.inverse_transform(X_scaled)
