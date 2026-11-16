#!/usr/bin/env python3
"""
BioCompressor - Lac Operon Experiment
Compress lactose operon dynamics from inputs (glucose, IPTG) to output (GFP)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy.integrate import odeint
import logging
from datetime import datetime

from bio_compressor.config.loader import ExperimentConfig
from bio_compressor.data.loader import UniversalDataLoader
from bio_compressor.models.factory import ModelFactory
from bio_compressor.training.trainer import UniversalTrainer
from bio_compressor.training.callbacks import EarlyStopping, ModelCheckpoint
from bio_compressor.visualization.plotter import UniversalPlotter
from bio_compressor.evaluation.metrics import ComprehensiveMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LacOperonModel:
    """Ground truth ODE model for lac operon"""
    
    def __init__(self):
        self.params = {
            'beta_m': 2.0,      # Transcription rate
            'gamma_m': 0.2,     # mRNA degradation
            'beta_p': 3.0,      # Translation rate
            'gamma_p': 0.05,    # GFP degradation
            'K_lac': 5.0,       # LacI binding constant
            'K_cap': 0.5,       # CAP binding constant
            'n': 2.5,          # Cooperativity
            'basal': 0.001     # Basal expression
        }
    
    def ode_system(self, y, t, glc, iptg):
        mRNA, GFP = y
        
        # Glucose effect: inhibits cAMP
        cAMP = 1.0 / (1.0 + (glc/0.05)**1.5)
        
        # IPTG effect: binds LacI, relieves repression
        lacI_active = 1.0 / (1.0 + (iptg/0.03)**1.8)
        
        # Transcriptional regulation
        cap_term = cAMP / (cAMP + self.params['K_cap'])
        lac_term = lacI_active**self.params['n'] / (
            lacI_active**self.params['n'] + self.params['K_lac']**self.params['n']
        )
        
        transcription = self.params['beta_m'] * cap_term * lac_term + self.params['basal']
        
        # Differential equations
        dmRNA = transcription - self.params['gamma_m'] * mRNA
        dGFP = self.params['beta_p'] * mRNA - self.params['gamma_p'] * GFP
        
        return [dmRNA, dGFP]
    
    def simulate(self, glc, iptg, t_max=600):
        """Run simulation to steady state"""
        y0 = [0.0, 0.0]
        t = np.linspace(0, t_max, 200)
        sol = odeint(self.ode_system, y0, t, args=(glc, iptg), 
                    rtol=1e-6, atol=1e-8)
        return sol[-1, 1]


def generate_simulation_data():
    """Generate training data from ODE model"""
    logger.info("Generating simulation data...")
    
    model = LacOperonModel()
    
    # Log-spaced concentrations (5 orders of magnitude)
    glc_range = np.logspace(-4, 1.5, 25)
    iptg_range = np.logspace(-4, 1.5, 25)
    
    data = []
    total = len(glc_range) * len(iptg_range)
    
    for i, glc in enumerate(glc_range):
        for j, iptg in enumerate(iptg_range):
            if (i * len(iptg_range) + j) % 50 == 0:
                progress = (i * len(iptg_range) + j) / total * 100
                logger.info(f"Progress: {progress:.1f}%")
            
            try:
                gfp = model.simulate(glc, iptg)
                # Add 2% noise
                gfp_noisy = gfp * (1 + 0.02 * np.random.randn())
                data.append([glc, iptg, max(0.001, gfp_noisy)])
            except Exception as e:
                logger.warning(f"Simulation failed for glc={glc}, iptg={iptg}: {e}")
                data.append([glc, iptg, 0.001])
    
    df = pd.DataFrame(data, columns=['glucose_mM', 'iptg_mM', 'gfp'])
    
    # Save data
    output_path = Path(__file__).parent / 'data' / 'simulation_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Generated {len(df)} data points")
    logger.info(f"   GFP range: {df['gfp'].min():.4f} - {df['gfp'].max():.4f}")
    logger.info(f"   GFP median: {df['gfp'].median():.4f}")
    
    return df


def main():
    """Main experiment pipeline"""
    
    # Create experiment directory
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 1. Generate or load data
    data_path = exp_dir / 'data' / 'simulation_data.csv'
    if not data_path.exists():
        df = generate_simulation_data()
    else:
        df = pd.read_csv(data_path)
        logger.info(f"✅ Loaded existing data from {data_path}")
    
    # 2. Load configuration
    config_path = exp_dir / 'config.yaml'
    config = ExperimentConfig.from_yaml(str(config_path))
    config.data['path'] = str(data_path)
    
    # 3. Initialize data loader
    loader = UniversalDataLoader(config)
    X, y, df = loader.load_data()
    
    # 4. Prepare data
    split_data = loader.prepare_data(X, y)
    
    # 5. Create model
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = ModelFactory.create_model(input_dim, output_dim, config)
    
    # 6. Initialize trainer
    trainer = UniversalTrainer(model, config)
    
    # 7. Setup callbacks
    callbacks = [
        EarlyStopping(patience=config.training.early_stopping_patience),
        ModelCheckpoint(
            filepath=str(results_dir / 'best_model.pt'),
            monitor='val_loss',
            mode='min'
        )
    ]
    
    # 8. Train model
    logger.info("\n" + "="*50)
    logger.info("Starting training...")
    logger.info("="*50)
    
    history = trainer.train(split_data, callbacks)
    
    # 9. Evaluate on test set
    logger.info("\n" + "="*50)
    logger.info("Evaluating on test set...")
    logger.info("="*50)
    
    test_results = trainer.evaluate(
        split_data['X_test'],
        split_data['y_test']
    )
    
    # Inverse transform to original scale
    y_test_orig = loader.inverse_transform_output(split_data['y_test'])
    y_pred_orig = loader.inverse_transform_output(test_results['predictions'])
    
    # 10. Calculate comprehensive metrics
    metrics = ComprehensiveMetrics.calculate_all(
        y_test_orig, y_pred_orig,
        feature_names=loader.output_columns
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([
        {'metric': k, 'value': v} 
        for k, v in metrics.items() 
        if not isinstance(v, dict)
    ])
    metrics_df.to_csv(results_dir / 'test_metrics.csv', index=False)
    
    # 11. Visualize results
    plotter = UniversalPlotter(config)
    
    # Prediction scatter
    fig1 = plotter.plot_predictions(
        y_test_orig, y_pred_orig,
        feature_names=loader.output_columns,
        title=f"Lac Operon - Test Set Performance (R²={metrics['r2_mean']:.4f})"
    )
    plotter.save_figure(fig1, str(results_dir / 'prediction_scatter.png'))
    
    # Residual analysis
    fig2 = plotter.plot_residuals(
        y_test_orig, y_pred_orig,
        feature_names=loader.output_columns
    )
    plotter.save_figure(fig2, str(results_dir / 'residuals.png'))
    
    # Training history
    fig3 = plotter.plot_training_history(history)
    plotter.save_figure(fig3, str(results_dir / 'training_history.png'))
    
    # 12. Save model
    trainer.save_model(
        str(results_dir / 'final_model.pth'),
        data_loader=loader
    )
    
    # 13. Save experiment summary
    summary = {
        'experiment': config.name,
        'timestamp': datetime.now().isoformat(),
        'data_points': len(df),
        'input_features': loader.input_columns,
        'output_features': loader.output_columns,
        'model_architecture': config.model.architecture.value,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'test_metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        'config': config.to_dict()
    }
    
    import json
    with open(results_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("\n" + "="*50)
    logger.info("✅ Experiment completed successfully!")
    logger.info(f"   Results saved to: {results_dir}")
    logger.info("="*50)
    
    return summary


if __name__ == '__main__':
    main()
