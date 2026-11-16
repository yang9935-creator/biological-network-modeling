import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ScaleType(str, Enum):
    LINEAR = "linear"
    LOG = "log"
    LOG2 = "log2"
    LN = "ln"

class ArchitectureType(str, Enum):
    FEEDFORWARD = "feedforward"
    RESIDUAL = "residual"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    CNN = "cnn"

@dataclass
class FeatureConfig:
    name: str
    unit: str = "AU"
    range: Optional[list] = None
    scale: ScaleType = ScaleType.LINEAR
    transform: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            name=data['name'],
            unit=data.get('unit', 'AU'),
            range=data.get('range', None),
            scale=ScaleType(data.get('scale', 'linear')),
            transform=data.get('transform', None)
        )

@dataclass
class ModelConfig:
    architecture: ArchitectureType
    hidden_layers: list
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            architecture=ArchitectureType(data['architecture']),
            hidden_layers=data['hidden_layers'],
            activation=data.get('activation', 'relu'),
            dropout=data.get('dropout', 0.1),
            use_batch_norm=data.get('use_batch_norm', True)
        )

@dataclass
class TrainingConfig:
    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    test_size: float = 0.2
    val_size: float = 0.1
    early_stopping_patience: int = 50
    seed: int = 42
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            epochs=data.get('epochs', 1000),
            batch_size=data.get('batch_size', 32),
            learning_rate=data.get('learning_rate', 0.001),
            optimizer=data.get('optimizer', 'adam'),
            test_size=data.get('test_size', 0.2),
            val_size=data.get('val_size', 0.1),
            early_stopping_patience=data.get('early_stopping_patience', 50),
            seed=data.get('seed', 42)
        )

@dataclass
class ExperimentConfig:
    name: str
    description: str
    input_features: list[FeatureConfig]
    output_features: list[FeatureConfig]
    data: Dict
    model: ModelConfig
    training: TrainingConfig
    task_type: str = "regression"
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        exp_name = list(data.keys())[0]
        exp_data = data[exp_name]
        
        return cls(
            name=exp_name,
            description=exp_data.get('description', ''),
            input_features=[FeatureConfig.from_dict(f) for f in exp_data['input_features']],
            output_features=[FeatureConfig.from_dict(f) for f in exp_data['output_features']],
            data=exp_data['data'],
            model=ModelConfig.from_dict(exp_data['model']),
            training=TrainingConfig.from_dict(exp_data.get('training', {})),
            task_type=exp_data.get('task_type', 'regression')
        )
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump({self.name: self.to_dict()}, f, default_flow_style=False)
    
    def to_dict(self) -> Dict:
        return {
            'description': self.description,
            'input_features': [f.__dict__ for f in self.input_features],
            'output_features': [f.__dict__ for f in self.output_features],
            'data': self.data,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'task_type': self.task_type
        }