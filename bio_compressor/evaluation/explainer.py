import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class ModelExplainer:
    """Explain AI model predictions using various attribution methods"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def integrated_gradients(self, X: np.ndarray, 
                            baseline: Optional[np.ndarray] = None,
                            steps: int = 50) -> np.ndarray:
        """
        Compute integrated gradients for feature attribution
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(X_tensor)
        else:
            baseline = torch.FloatTensor(baseline).to(self.device)
        
        # Scale inputs and compute gradients
        scaled_inputs = [baseline + (float(i)/steps) * (X_tensor - baseline) 
                        for i in range(steps+1)]
        
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.requires_grad_()
            
            # Forward pass
            output = self.model(scaled_input)
            
            # Backward pass
            self.model.zero_grad()
            output.sum().backward()
            
            gradients.append(scaled_input.grad.clone())
        
        # Average gradients and compute integrated gradients
        avg_gradients = torch.stack(gradients).mean(0)
        integrated_gradients = (X_tensor - baseline) * avg_gradients
        
        return integrated_gradients.cpu().numpy()
    
    def gradient_shap(self, X: np.ndarray, 
                     background: np.ndarray,
                     n_samples: int = 50) -> np.ndarray:
        """
        Compute GradientSHAP values
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        background_tensor = torch.FloatTensor(background).to(self.device)
        
        shap_values = []
        
        for _ in range(n_samples):
            # Sample random baseline from background
            idx = np.random.randint(len(background))
            baseline = background_tensor[idx:idx+1].expand_as(X_tensor)
            
            # Random interpolation
            alpha = torch.rand(len(X_tensor), 1).to(self.device)
            interpolated = baseline + alpha * (X_tensor - baseline)
            interpolated.requires_grad_()
            
            # Forward and backward
            output = self.model(interpolated)
            self.model.zero_grad()
            output.sum().backward()
            
            shap_values.append(interpolated.grad.clone())
        
        shap_values = torch.stack(shap_values).mean(0)
        shap_values = shap_values * (X_tensor - background_tensor.mean(0))
        
        return shap_values.cpu().numpy()
    
    def feature_importance(self, X: np.ndarray, 
                          feature_names: List[str],
                          method: str = 'permutation') -> Dict[str, float]:
        """
        Compute feature importance scores
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Baseline prediction
        with torch.no_grad():
            baseline_pred = self.model(X_tensor).cpu().numpy()
        
        importance_scores = {}
        
        if method == 'permutation':
            for i, name in enumerate(feature_names):
                # Permute feature i
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                X_permuted_tensor = torch.FloatTensor(X_permuted).to(self.device)
                
                with torch.no_grad():
                    permuted_pred = self.model(X_permuted_tensor).cpu().numpy()
                
                # Importance = increase in error
                error_increase = np.mean((baseline_pred - permuted_pred) ** 2)
                importance_scores[name] = error_increase
        
        elif method == 'gradient':
            X_tensor = X_tensor.requires_grad_()
            output = self.model(X_tensor)
            
            self.model.zero_grad()
            output.sum().backward()
            
            gradients = X_tensor.grad.cpu().numpy()
            importance_scores = {
                name: np.abs(gradients[:, i]).mean()
                for i, name in enumerate(feature_names)
            }
        
        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        return importance_scores
    
    def sensitivity_analysis(self, X_base: np.ndarray,
                           feature_idx: int,
                           perturbation_range: np.ndarray) -> np.ndarray:
        """
        Analyze model sensitivity to specific input features
        """
        X_test = np.tile(X_base, (len(perturbation_range), 1))
        X_test[:, feature_idx] = perturbation_range
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def plot_feature_importance(self, importance_scores: Dict[str, float],
                              title: str = "Feature Importance"):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        # Sort by importance
        idx = np.argsort(scores)
        features = [features[i] for i in idx]
        scores = [scores[i] for i in idx]
        
        ax.barh(features, scores)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig