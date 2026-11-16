"""
BioCompressor standalone lac operon demonstration.
Results are written to the repository-level results/lac_operon directory.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
results_path = project_root / "results" / "lac_operon"
results_path.mkdir(parents=True, exist_ok=True)

print("="*60)
print("BioCompressor Quick Demo")
print("="*60)
print(f"Project location: {project_root}")
print(f"Results will be saved to: {results_path}")
print("="*60)

# 1. Generate data
print("\nGenerating lac operon data...")

class LacOperonModel:
    def __init__(self):
        self.params = {
            'beta_m': 2.0, 'gamma_m': 0.2, 'beta_p': 3.0, 'gamma_p': 0.05,
            'K_lac': 5.0, 'K_cap': 0.5, 'n': 2.5, 'basal': 0.001
        }
    
    def simulate(self, glc, iptg):
        cAMP = 1.0 / (1.0 + (glc/0.05)**1.5)
        lacI_active = 1.0 / (1.0 + (iptg/0.03)**1.8)
        cap_term = cAMP / (cAMP + self.params['K_cap'])
        lac_term = lacI_active**self.params['n'] / (lacI_active**self.params['n'] + self.params['K_lac']**self.params['n'])
        transcription = self.params['beta_m'] * cap_term * lac_term + self.params['basal']
        gfp = (self.params['beta_p'] * transcription) / self.params['gamma_p']
        return gfp * (1 + 0.02 * np.random.randn())

model = LacOperonModel()

glc_range = np.logspace(-4, 1.5, 50)
iptg_range = np.logspace(-4, 1.5, 50)

data = []
for glc in glc_range:
    for iptg in iptg_range:
        gfp = model.simulate(glc, iptg)
        data.append([glc, iptg, max(0.001, gfp)])

df = pd.DataFrame(data, columns=['glucose', 'iptg', 'gfp'])
df['gfp_log'] = np.log10(df['gfp'] + 0.001)

print(f"   {len(df)} data points generated")
print(f"   GFP range: {df['gfp'].min():.4f} - {df['gfp'].max():.4f}")

# 2. Train AI model
print("\nTraining AI model...")

X = df[['glucose', 'iptg']].values
y = df['gfp_log'].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x): 
        return self.net(x)

model_ai = NeuralNet()
optimizer = torch.optim.Adam(model_ai.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)

for epoch in range(2000):
    optimizer.zero_grad()
    loss = criterion(model_ai(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"   Epoch {epoch:4d}, Loss: {loss.item():.6f}")

# 3. Evaluate
print("\nEvaluating model...")

model_ai.eval()
with torch.no_grad():
    y_pred_scaled = model_ai(torch.FloatTensor(X_test)).numpy()
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
    y_test_log = scaler_y.inverse_transform(y_test)
    y_pred = 10 ** y_pred_log
    y_true = 10 ** y_test_log

r2 = r2_score(y_true, y_pred)
print(f"\nTest R² = {r2:.4f}")

# 4. Save Excel files
print("\nSaving results...")

# Save raw data
simulation_path = results_path / 'simulation_data.xlsx'
df.to_excel(simulation_path, index=False)
print(f"   Saved: {simulation_path.name}")

# Save predictions
pred_df = pd.DataFrame({
    'glucose': scaler_X.inverse_transform(X_test)[:, 0],
    'iptg': scaler_X.inverse_transform(X_test)[:, 1],
    'true_gfp': y_true.flatten(),
    'predicted_gfp': y_pred.flatten(),
    'error': (y_pred - y_true).flatten(),
    'relative_error': (abs(y_pred - y_true) / y_true * 100).flatten()
})

predictions_path = results_path / 'predictions.xlsx'
pred_df.to_excel(predictions_path, index=False)
print(f"   Saved: {predictions_path.name}")

# 5. Generate plots
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Prediction vs True
axes[0,0].scatter(y_true, y_pred, alpha=0.3, s=10)
axes[0,0].plot([y_true.min(), y_true.max()], 
               [y_true.min(), y_true.max()], 'r--', linewidth=2)
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_xlabel('True GFP', fontsize=12)
axes[0,0].set_ylabel('Predicted GFP', fontsize=12)
axes[0,0].set_title(f'Prediction Performance (R²={r2:.4f})', fontsize=12)
axes[0,0].grid(True, alpha=0.3)

# Subplot 2: Residual distribution
residuals = y_pred - y_true
axes[0,1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0,1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0,1].set_xlabel('Prediction Error', fontsize=12)
axes[0,1].set_ylabel('Frequency', fontsize=12)
axes[0,1].set_title(f'Residual Distribution (mean={residuals.mean():.4f})', fontsize=12)
axes[0,1].grid(True, alpha=0.3)

# Subplot 3: Dose-response curve
glc_test = np.logspace(-4, 1, 50)
iptg_fixed = 0.1
gfp_true = []
gfp_pred = []

for glc in glc_test:
    gfp_true.append(model.simulate(glc, iptg_fixed))
    X_input = scaler_X.transform([[glc, iptg_fixed]])
    with torch.no_grad():
        pred = model_ai(torch.FloatTensor(X_input)).numpy()
        pred = scaler_y.inverse_transform(pred.reshape(-1, 1))
        gfp_pred.append(10 ** pred[0, 0])

axes[1,0].plot(glc_test, gfp_true, 'b-', label='True Model', linewidth=2)
axes[1,0].plot(glc_test, gfp_pred, 'r--', label='AI Prediction', linewidth=2)
axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')
axes[1,0].set_xlabel('Glucose (mM)', fontsize=12)
axes[1,0].set_ylabel('GFP', fontsize=12)
axes[1,0].set_title(f'Dose Response (IPTG={iptg_fixed} mM)', fontsize=12)
axes[1,0].legend(fontsize=10)
axes[1,0].grid(True, alpha=0.3)

# Subplot 4: Experiment Summary
summary_text = (
    f'Experiment Summary\n'
    f'{"="*25}\n\n'
    f'Model Architecture:\n'
    f'  • Input: 2 features\n'
    f'  • Hidden: 64 → 64 → 32\n'
    f'  • Output: 1 (GFP)\n'
    f'  • Parameters: 5,441\n\n'
    f'Training:\n'
    f'  • Total samples: {len(df)}\n'
    f'  • Training set: {len(X_train)}\n'
    f'  • Test set: {len(X_test)}\n\n'
    f'Performance:\n'
    f'  • R² Score: {r2:.4f}\n'
    f'  • RMSE: {np.sqrt(((y_true - y_pred)**2).mean()):.4f}\n'
    f'  • MAE: {np.mean(np.abs(y_true - y_pred)):.4f}\n\n'
    f'Results saved:\n'
    f'  • {results_path.name}/'
)

axes[1,1].text(0.1, 0.5, summary_text,
               transform=axes[1,1].transAxes, 
               fontsize=11,
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[1,1].axis('off')
axes[1,1].set_title('Experiment Summary', fontsize=12)

plt.tight_layout()

# Save figure
figure_path = results_path / 'final_results.png'
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {figure_path.name}")

# 6. Save model
model_path = results_path / 'model.pth'
torch.save({
    'model_state_dict': model_ai.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'config': {
        'input_dim': 2,
        'output_dim': 1,
        'hidden_layers': [64, 64, 32],
        'r2_score': r2
    }
}, model_path)
print(f"   Saved: {model_path.name}")

# 7. Generate HTML report
print("\nGenerating HTML report...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BioCompressor - Lac Operon Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; }}
        .metric {{ font-size: 24px; color: #27ae60; font-weight: bold; }}
        .footer {{ margin-top: 40px; color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BioCompressor: Lac Operon Compression Results</h1>
        
        <div class="summary">
            <h2>Experiment Summary</h2>
            <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Script location:</strong> {script_dir}</p>
            <p><strong>Results location:</strong> {results_path}</p>
        </div>
        
        <h2>Model Performance</h2>
        <p><strong>R² Score:</strong> <span class="metric">{r2:.4f}</span></p>
        <p><strong>RMSE:</strong> {np.sqrt(((y_true - y_pred)**2).mean()):.4f}</p>
        <p><strong>MAE:</strong> {np.mean(np.abs(y_true - y_pred)):.4f}</p>
        
        <h2>Dataset Information</h2>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Samples</th>
                <th>Features</th>
            </tr>
            <tr>
                <td>Total</td>
                <td>{len(df)}</td>
                <td>2</td>
            </tr>
            <tr>
                <td>Training</td>
                <td>{len(X_train)}</td>
                <td>2</td>
            </tr>
            <tr>
                <td>Test</td>
                <td>{len(X_test)}</td>
                <td>2</td>
            </tr>
        </table>
        
        <h2>Model Architecture</h2>
        <p><strong>Type:</strong> Feedforward Neural Network</p>
        <p><strong>Layers:</strong> 64 → 64 → 32 → 1</p>
        <p><strong>Activation:</strong> ReLU</p>
        <p><strong>Parameters:</strong> 5,441</p>
        
        <h2>Output Files</h2>
        <ul>
            <li><strong>simulation_data.xlsx</strong> - Raw simulation data (625 conditions)</li>
            <li><strong>predictions.xlsx</strong> - AI predictions vs true values</li>
            <li><strong>final_results.png</strong> - Complete visualization</li>
            <li><strong>model.pth</strong> - Trained PyTorch model</li>
        </ul>
        
        <div class="footer">
            <p>Generated by BioCompressor v1.0.0</p>
        </div>
    </div>
</body>
</html>
"""

html_path = results_path / 'report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"   Saved: {html_path.name}")

print("\n" + "="*60)
print("QUICK DEMO COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nAll results saved to: {results_path}")
print("\nGenerated files:")
print(f"   {results_path / 'simulation_data.xlsx'}  - Raw data")
print(f"   {results_path / 'predictions.xlsx'}      - AI predictions")
print(f"   {results_path / 'final_results.png'}    - Complete figure")
print(f"   {results_path / 'model.pth'}            - Trained model")
print(f"   {results_path / 'report.html'}          - HTML report")
print(f"\nTest R² = {r2:.4f}")
print("="*60)
