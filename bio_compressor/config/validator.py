from typing import List, Dict, Any
from .loader import ExperimentConfig, FeatureConfig

class ConfigValidator:
    @staticmethod
    def validate(config: ExperimentConfig) -> List[str]:
        errors = []
        
        # Check input features
        if not config.input_features:
            errors.append("No input features defined")
        
        # Check output features
        if not config.output_features:
            errors.append("No output features defined")
        
        # Check data source
        if 'source' not in config.data:
            errors.append("Data source not specified")
        if 'path' not in config.data:
            errors.append("Data path not specified")
        
        # Check model configuration
        if not config.model.hidden_layers:
            errors.append("No hidden layers defined")
        
        # Check training parameters
        if config.training.epochs <= 0:
            errors.append("Epochs must be positive")
        if config.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        if config.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        return errors
    
    @staticmethod
    def suggest_fixes(config: ExperimentConfig) -> Dict[str, Any]:
        suggestions = {}
        
        # Suggest reasonable hidden layers if none provided
        if not config.model.hidden_layers:
            n_inputs = len(config.input_features)
            n_outputs = len(config.output_features)
            suggestions['hidden_layers'] = [
                max(32, n_inputs * 4),
                max(32, n_inputs * 2),
                max(16, n_outputs * 2)
            ]
        
        # Suggest log scale for features with wide range
        for i, feature in enumerate(config.input_features):
            if feature.range and feature.range[1] / feature.range[0] > 100:
                if feature.scale == 'linear':
                    suggestions[f'input_{i}_scale'] = 'log'
        
        return suggestions