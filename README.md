# BioCompressor: Biological Network Modeling

BioCompressor is a research prototype for exploring neural-network and
parsimonious approximations of selected biological systems. The repository
contains a reusable Python package, runnable examples, configuration templates,
and representative outputs.

## Included studies

- **Lac operon:** approximate glucose/IPTG regulation of GFP expression.
- **Enzyme kinetics:** learn a Michaelis-Menten input-output relationship.
- **Glucose regulation:** compare a compartmental ODE model with a compact equation.

## Repository structure

```text
bio_compressor/          Reusable loading, modeling, training and evaluation code
examples/                Standalone demonstrations
experiments/lac_operon/  Configuration-driven BioCompressor experiment
results/                 Representative generated outputs
tests/                   Lightweight smoke tests
```

## Installation

```bash
git clone https://github.com/yang9935-creator/biological-network-modeling.git
cd biological-network-modeling
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run the examples

```bash
python examples/lac_operon/run.py
python examples/enzyme_kinetics.py
python examples/glucose_regulation/model.py
python examples/glucose_regulation/visualize.py
python experiments/lac_operon/run.py
```

Use a custom configuration and dataset:

```bash
python examples/custom_data_template.py path/to/config.yaml
```

Generated files are written under `results/`. Timestamped glucose runs are
ignored by Git, while the latest representative results can be committed.

## Configuration format

Experiments are described with YAML files containing input/output features,
the data source, model architecture and training settings. See
`experiments/lac_operon/config.yaml` for a runnable example.

## Current limitations

- This is research software, not a production or clinical system.
- EGFR and p53 configurations do not include their source datasets.
- Results can vary with library versions and available hardware.
- Biological interpretation requires validation against experimental evidence.

## Testing

```bash
python -m compileall bio_compressor examples experiments
python -m pytest
```
