"""Run BioCompressor with a user-provided YAML configuration."""

import argparse
from pathlib import Path

from bio_compressor import UniversalDataLoader, ModelFactory, UniversalTrainer
from bio_compressor.config.loader import ExperimentConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to an experiment YAML file")
    parser.add_argument("--output", default="my_model.pth")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = ExperimentConfig.from_yaml(str(config_path))

    data_path = Path(config.data["path"])
    if not data_path.is_absolute():
        config.data["path"] = str((config_path.parent / data_path).resolve())

    loader = UniversalDataLoader(config)
    X, y, _ = loader.load_data()
    prepared = loader.prepare_data(X, y)

    model = ModelFactory.create_model(X.shape[1], y.shape[1], config)
    trainer = UniversalTrainer(model, config)
    trainer.train(prepared)
    trainer.save_model(args.output, loader)


if __name__ == "__main__":
    main()
