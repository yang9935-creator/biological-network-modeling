"""
BioCompressor - Enzyme Kinetics
Michaelis-Menten equation: v = (Vmax * [S]) / (Km + [S])
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import json
import logging
from typing import Dict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CONFIG = {
    "experiment_name": "enzyme_kinetics_optimized",
    "description": "Michaelis-Menten enzyme kinetics",
    
    "input_features": [
        {
            "name": "substrate_concentration",
            "unit": "mM",
            "min": 0.01,
            "max": 10.0,
            "scale": "log",
            "n_points": 200 
        }
    ],
    
    "output_features": [
        {
            "name": "reaction_velocity",
            "unit": "umol/min",
            "transform": "none" 
        }
    ],
    
    "data_source": "simulation",
    "data_path": "",
    
    "model": {
        "architecture": "feedforward",
        "hidden_layers": [128, 128, 64, 32], 
        "activation": "relu",
        "dropout": 0.05,  
        "use_batch_norm": True
    },
    
    "training": {
        "epochs": 2000, 
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "test_size": 0.2,
        "val_size": 0.1,
        "early_stopping_patience": 200,
        "seed": 42
    },
    
    "output_dir": "results/enzyme_kinetics"
}

class ImprovedNeuralNet(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class MichaelisMentenSimulator:
    
    def __init__(self, config: Dict):
        self.config = config
        self.input_features = config['input_features']
        self.output_features = config['output_features']
        self.output_names = [f['name'] for f in self.output_features]
        
        self.Vmax = 100.0
        self.Km = 0.5
        
    def generate_data(self) -> pd.DataFrame:
        logger.info("Generating Michaelis-Menten enzyme kinetics data...")
        
        feat = self.input_features[0]
        if feat['scale'] == 'log':
            substrate = np.logspace(
                np.log10(feat['min']), 
                np.log10(feat['max']), 
                feat['n_points']
            )
        else:
            substrate = np.linspace(feat['min'], feat['max'], feat['n_points'])
        
        data = []
        for s in substrate:
            # Michaelis-Menten equation
            v = (self.Vmax * s) / (self.Km + s)
            
            v_noisy = v * (1 + 0.005 * np.random.randn())
            v_noisy = max(0, v_noisy)
            
            data.append([s, v_noisy])
        
        df = pd.DataFrame(data, columns=[feat['name']] + self.output_names)
        
        logger.info(f"   Generated {len(df)} data points")
        logger.info(f"   Substrate range: {df[feat['name']].min():.4f} - {df[feat['name']].max():.4f}")
        logger.info(f"   Velocity range: {df[self.output_names[0]].min():.4f} - {df[self.output_names[0]].max():.4f}")
        
        return df

class OptimizedTrainer:
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
     
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )
        
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
        
    def train(self, X_train, y_train, X_val, y_val):
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']
        patience = self.config['training']['early_stopping_patience']
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
                
                y_val_np = y_val_t.cpu().numpy()
                val_pred_np = val_outputs.cpu().numpy()
                r2 = r2_score(y_val_np, val_pred_np)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['val_loss'].append(val_loss)
            self.history['val_r2'].append(r2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 200 == 0:
                logger.info(f"Epoch {epoch:4d} | Train Loss: {np.mean(train_losses):.6f} | "
                          f"Val Loss: {val_loss:.6f} | Val R²: {r2:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def evaluate(self, X_test, y_test, scaler_X, scaler_y):
        
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            y_pred_scaled = self.model(X_test_t).cpu().numpy()
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        logger.info(f"\n Final Test Metrics:")
        logger.info(f"   R²  = {r2:.6f}")
        logger.info(f"   MAE = {mae:.4f}")
        logger.info(f"   RMSE= {rmse:.4f}")
        
        return {
            'predictions': y_pred,
            'true_values': y_true,
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }

def plot_detailed_results(y_true, y_pred, metrics, save_path):
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Prediction scatter
    axes[0,0].scatter(y_true, y_pred, alpha=0.5, s=30, color='blue', edgecolors='black', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0,0].set_xlabel('True Velocity (μmol/min)', fontsize=12)
    axes[0,0].set_ylabel('Predicted Velocity (μmol/min)', fontsize=12)
    axes[0,0].set_title(f'Prediction Performance (R² = {metrics["r2"]:.4f})', fontsize=14)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_pred - y_true
    axes[0,1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0,1].set_xlabel('Prediction Error (μmol/min)', fontsize=12)
    axes[0,1].set_ylabel('Frequency', fontsize=12)
    axes[0,1].set_title(f'Residual Distribution\n(mean = {residuals.mean():.4f}, std = {residuals.std():.4f})', fontsize=14)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals.flatten(), dist="norm", plot=axes[0,2])
    axes[0,2].set_title('Q-Q Plot (Normality of Residuals)', fontsize=14)
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Error vs True Value
    axes[1,0].scatter(y_true, np.abs(residuals), alpha=0.5, s=30, color='green', edgecolors='black', linewidth=0.5)
    axes[1,0].set_xlabel('True Velocity (μmol/min)', fontsize=12)
    axes[1,0].set_ylabel('Absolute Error (μmol/min)', fontsize=12)
    axes[1,0].set_title('Error Distribution vs True Value', fontsize=14)
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Relative Error
    rel_error = np.abs(residuals) / (y_true + 1e-8) * 100
    axes[1,1].hist(rel_error, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1,1].set_xlabel('Relative Error (%)', fontsize=12)
    axes[1,1].set_ylabel('Frequency', fontsize=12)
    axes[1,1].set_title(f'Relative Error Distribution\n(mean = {rel_error.mean():.2f}%, median = {np.median(rel_error):.2f}%)', fontsize=14)
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Summary
    axes[1,2].axis('off')
    summary_text = (
        f" EXPERIMENT SUMMARY\n"
        f"{'='*25}\n\n"
        f"Michaelis-Menten Parameters:\n"
        f"  • Vmax = 100.0 μmol/min\n"
        f"  • Km   = 0.5 mM\n\n"
        f"Model Performance:\n"
        f"  • R²  = {metrics['r2']:.6f}\n"
        f"  • MAE = {metrics['mae']:.4f}\n"
        f"  • RMSE= {metrics['rmse']:.4f}\n\n"
        f"Residuals:\n"
        f"  • Mean = {residuals.mean():.4f}\n"
        f"  • Std  = {residuals.std():.4f}\n\n"
        f"Relative Error:\n"
        f"  • Mean = {rel_error.mean():.2f}%\n"
        f"  • 95%  < {np.percentile(rel_error, 95):.2f}%"
    )
    axes[1,2].text(0.05, 0.5, summary_text, transform=axes[1,2].transAxes,
                  fontsize=11, verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path / 'detailed_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_path / 'detailed_results.pdf', bbox_inches='tight')
    logger.info(f"   Saved: {save_path / 'detailed_results.png'}")

def main():
    results_path = Path(__file__).resolve().parents[1] / CONFIG['output_dir']
    results_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("OPTIMIZED MICHAELIS-MENTEN ENZYME KINETICS")
    logger.info("="*70)
    logger.info(f"Results will be saved to: {results_path}")
    
    # 1. Generate data
    simulator = MichaelisMentenSimulator(CONFIG)
    df = simulator.generate_data()
    
    # 2. Prepare data
    X = df[['substrate_concentration']].values
    y = df[['reaction_velocity']].values
    
    # 3. Scale data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 4. Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
    )
    
    logger.info(f"\n Data split:")
    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Val:   {len(X_val)} samples")
    logger.info(f"   Test:  {len(X_test)} samples")
    
    # 5. Create model
    model = ImprovedNeuralNet(1, 1)
    logger.info(f"\n Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 6. Train
    trainer = OptimizedTrainer(model, CONFIG)
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # 7. Evaluate
    results = trainer.evaluate(X_test, y_test, scaler_X, scaler_y)
    
    # 8. Save raw data
    df.to_excel(results_path / 'raw_data.xlsx', index=False)
    
    # 9. Save predictions
    pred_df = pd.DataFrame({
        'substrate_mM': scaler_X.inverse_transform(X_test).flatten(),
        'true_velocity': results['true_values'].flatten(),
        'predicted_velocity': results['predictions'].flatten(),
        'error': (results['predictions'] - results['true_values']).flatten(),
        'rel_error_percent': np.abs((results['predictions'] - results['true_values']) / 
                                   (results['true_values'] + 1e-8) * 100).flatten()
    })
    pred_df.to_excel(results_path / 'predictions.xlsx', index=False)
    
    # 10. Plot training history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(results_path / 'training_history.png', dpi=150)
    
    # 11. Plot R² history
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['val_r2'], color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.set_title('Validation R² Over Time')
    ax.grid(True, alpha=0.3)
    plt.savefig(results_path / 'r2_history.png', dpi=150)
    
    # 12. Detailed plots
    plot_detailed_results(results['true_values'], results['predictions'], results, results_path)
    
    # 13. Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'config': CONFIG,
        'final_r2': results['r2']
    }, results_path / 'model.pth')
    
    # 14. Save metrics
    with open(results_path / 'metrics.json', 'w') as f:
        json.dump({
            'r2': results['r2'],
            'mae': results['mae'],
            'rmse': results['rmse'],
            'final_val_loss': history['val_loss'][-1]
        }, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info(" EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"\n Results saved to: {results_path}")
    logger.info("\nGenerated files:")
    logger.info(f" {results_path}/detailed_results.png")
    logger.info(f" {results_path}/detailed_results.pdf")
    logger.info(f" {results_path}/predictions.xlsx")
    logger.info(f" {results_path}/raw_data.xlsx")
    logger.info(f" {results_path}/model.pth")
    logger.info(f" {results_path}/training_history.png")
    logger.info(f" {results_path}/r2_history.png")
    logger.info(f" {results_path}/metrics.json")
    logger.info(f"\n FINAL TEST R² = {results['r2']:.6f}")
    logger.info("="*70)
    
    return results

if __name__ == '__main__':
    main()
