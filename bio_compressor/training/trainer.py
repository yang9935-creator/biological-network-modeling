import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from bio_compressor.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

logger = logging.getLogger(__name__)

class UniversalTrainer:
    """
    Universal trainer that works with any model, any data, any task
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: Optional[str] = None):
        
        self.model = model
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training components
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        logger.info(f"🚀 Trainer initialized on {self.device}")
        
    def _get_criterion(self) -> nn.Module:
        """Get loss function based on task type"""
        task_type = self.config.task_type
        
        if task_type == 'regression':
            return nn.MSELoss()
        elif task_type == 'regression_mae':
            return nn.L1Loss()
        elif task_type == 'regression_huber':
            return nn.HuberLoss()
        elif task_type == 'classification':
            return nn.CrossEntropyLoss()
        elif task_type == 'binary_classification':
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on configuration"""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = getattr(self.config.training, 'weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = getattr(self.config.training, 'momentum', 0.9)
            return torch.optim.SGD(self.model.parameters(), lr=lr, 
                                  momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler"""
        scheduler_name = getattr(self.config.training, 'scheduler', 'none')
        lr = self.config.training.learning_rate
        
        if scheduler_name == 'step':
            step_size = getattr(self.config.training, 'step_size', 30)
            gamma = getattr(self.config.training, 'gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        
        elif scheduler_name == 'cosine':
            T_max = self.config.training.epochs
            eta_min = getattr(self.config.training, 'eta_min', lr * 0.01)
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max, eta_min)
        
        elif scheduler_name == 'plateau':
            patience = getattr(self.config.training, 'patience', 10)
            factor = getattr(self.config.training, 'factor', 0.5)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience
            )
        
        else:
            return None
    
    def train(self, 
             data_dict: Dict[str, np.ndarray],
             callbacks: Optional[list] = None) -> Dict[str, list]:
        """
        Complete training pipeline
        
        Args:
            data_dict: Dictionary with 'X_train', 'y_train', 'X_val', 'y_val'
            callbacks: List of callbacks (EarlyStopping, ModelCheckpoint, etc.)
        
        Returns:
            Training history dictionary
        """
        
        # Convert data to tensors
        X_train = torch.FloatTensor(data_dict['X_train']).to(self.device)
        y_train = torch.FloatTensor(data_dict['y_train']).to(self.device)
        X_val = torch.FloatTensor(data_dict['X_val']).to(self.device)
        y_val = torch.FloatTensor(data_dict['y_val']).to(self.device)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Initialize callbacks
        if callbacks is None:
            callbacks = []
        
        early_stopping = None
        model_checkpoint = None
        
        for cb in callbacks:
            if isinstance(cb, EarlyStopping):
                early_stopping = cb
            if isinstance(cb, ModelCheckpoint):
                model_checkpoint = cb
        
        epochs = self.config.training.epochs
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val).item()
                
                # Calculate metrics
                val_pred = val_outputs.cpu().numpy()
                y_val_np = y_val.cpu().numpy()
                val_metrics = self._calculate_metrics(y_val_np, val_pred)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch:4d} | "
                          f"Train Loss: {avg_train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"Val R²: {val_metrics.get('r2', 0):.4f} | "
                          f"LR: {current_lr:.2e}")
            
            # Callbacks
            stop_training = False
            for cb in callbacks:
                cb_output = cb(self.model, epoch, val_loss, val_metrics)
                if cb_output == 'stop':
                    stop_training = True
            
            if stop_training:
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("✅ Training completed!")
        
        # Load best model if checkpoint exists
        if model_checkpoint and model_checkpoint.best_model_path:
            self.load_model(model_checkpoint.best_model_path)
            logger.info(f"📥 Loaded best model from {model_checkpoint.best_model_path}")
        
        return self.history
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {}
        
        # R² score
        try:
            metrics['r2'] = r2_score(y_true, y_pred, multioutput='variance_weighted')
        except:
            metrics['r2'] = 0.0
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Percentage Error
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            metrics['mape'] = mape
        
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test set"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
            test_loss = self.criterion(
                self.model(X_tensor), y_tensor
            ).item()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        metrics['loss'] = test_loss
        
        logger.info("📊 Test Set Evaluation:")
        logger.info(f"   Loss: {test_loss:.6f}")
        logger.info(f"   R²: {metrics['r2']:.4f}")
        logger.info(f"   MAE: {metrics['mae']:.4f}")
        logger.info(f"   RMSE: {metrics['rmse']:.4f}")
        
        return {
            'predictions': y_pred,
            'metrics': metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data"""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()
        
        return y_pred
    
    def save_model(self, path: str, data_loader=None):
        """Save complete model package"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'input_dim': self.model.input_dim if hasattr(self.model, 'input_dim') else None,
            'output_dim': self.model.output_dim if hasattr(self.model, 'output_dim') else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save scalers if available
        if data_loader and hasattr(data_loader, 'input_scaler'):
            save_dict['input_scaler'] = data_loader.input_scaler
            save_dict['output_scaler'] = data_loader.output_scaler
            save_dict['input_columns'] = data_loader.input_columns
            save_dict['output_columns'] = data_loader.output_columns
        
        torch.save(save_dict, path)
        logger.info(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"📥 Model loaded from {path}")
        
        return checkpoint
