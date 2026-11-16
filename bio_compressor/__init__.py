"""
BioCompressor: Universal AI Framework for Biological System Compression
Compress any biological signaling network from inputs to outputs using deep learning.
"""

__version__ = "0.1.0"

from bio_compressor.data.loader import UniversalDataLoader
from bio_compressor.models.factory import ModelFactory
from bio_compressor.training.trainer import UniversalTrainer
from bio_compressor.visualization.plotter import UniversalPlotter

__all__ = [
    'UniversalDataLoader',
    'ModelFactory',
    'UniversalTrainer',
    'UniversalPlotter',
    '__version__',
]
