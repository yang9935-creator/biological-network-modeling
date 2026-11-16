import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from bio_compressor.config.loader import ArchitectureType
from bio_compressor.models.architectures import (
    FeedForwardNet,
    ResidualNet,
    TransformerNet,
    LSTMNet,
    CNN1DNet
)

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating appropriate neural network architectures"""
    
    @staticmethod
    def create_model(input_dim: int, 
                    output_dim: int, 
                    config) -> nn.Module:
        """Create a model based on configuration"""
        
        architecture = config.model.architecture
        hidden_layers = config.model.hidden_layers
        activation = config.model.activation
        dropout = config.model.dropout
        use_batch_norm = config.model.use_batch_norm
        
        logger.info(f"🏗️  Creating {architecture.value} model")
        logger.info(f"   Input dim: {input_dim}, Output dim: {output_dim}")
        logger.info(f"   Hidden layers: {hidden_layers}")
        
        if architecture == ArchitectureType.FEEDFORWARD:
            model = FeedForwardNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_layers,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
        
        elif architecture == ArchitectureType.RESIDUAL:
            model = ResidualNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_layers,
                activation=activation,
                dropout=dropout
            )
        
        elif architecture == ArchitectureType.TRANSFORMER:
            model = TransformerNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_layers[0],
                num_heads=4,
                num_layers=len(hidden_layers),
                dropout=dropout
            )
        
        elif architecture == ArchitectureType.LSTM:
            model = LSTMNet(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_layers[0],
                num_layers=len(hidden_layers),
                dropout=dropout
            )
        
        elif architecture == ArchitectureType.CNN:
            model = CNN1DNet(
                input_dim=input_dim,
                output_dim=output_dim,
                channel_dims=hidden_layers,
                kernel_size=3,
                dropout=dropout
            )
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"   Total parameters: {n_params:,}")
        
        return model
    
    @staticmethod
    def create_from_checkpoint(checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Recreate model
        config = checkpoint['config']
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        
        model = ModelFactory.create_model(input_dim, output_dim, config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✅ Model loaded from {checkpoint_path}")
        return model