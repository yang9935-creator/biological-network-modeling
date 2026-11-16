from pathlib import Path

import torch

from bio_compressor.config.loader import ExperimentConfig
from bio_compressor.models.factory import ModelFactory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_lac_operon_config_loads():
    config = ExperimentConfig.from_yaml(
        str(PROJECT_ROOT / "experiments" / "lac_operon" / "config.yaml")
    )
    assert config.name == "lac_operon"
    assert len(config.input_features) == 2
    assert len(config.output_features) == 1


def test_feedforward_model_shape():
    config = ExperimentConfig.from_yaml(
        str(PROJECT_ROOT / "experiments" / "lac_operon" / "config.yaml")
    )
    model = ModelFactory.create_model(2, 1, config)
    output = model(torch.zeros(4, 2))
    assert tuple(output.shape) == (4, 1)
