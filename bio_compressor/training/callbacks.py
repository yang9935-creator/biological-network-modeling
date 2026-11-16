import torch
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = np.inf
            self.compare = lambda current, best: current < best - min_delta
        else:
            self.best_score = -np.inf
            self.compare = lambda current, best: current > best + min_delta
    
    def __call__(self, model, epoch, loss, metrics):
        score = loss if self.mode == 'min' else metrics.get('r2', 0)
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return 'stop'
        
        return None


class ModelCheckpoint:
    """Save best model during training"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best_only: bool = True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.best_model_path = None
        
        # Create directory
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model, epoch, loss, metrics):
        if self.monitor == 'val_loss':
            current_score = loss
        else:
            current_score = metrics.get(self.monitor, 0)
        
        save_model = False
        
        if self.save_best_only:
            if self.mode == 'min':
                if current_score < self.best_score:
                    self.best_score = current_score
                    save_model = True
            else:
                if current_score > self.best_score:
                    self.best_score = current_score
                    save_model = True
        else:
            save_model = True
        
        if save_model:
            save_path = self.filepath.with_name(
                f"{self.filepath.stem}_epoch{epoch}{self.filepath.suffix}"
            ) if not self.save_best_only else self.filepath
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': current_score,
                'monitor': self.monitor
            }, save_path)
            
            self.best_model_path = save_path
            logger.info(f"📥 Model checkpoint saved to {save_path}")
        
        return None


class LearningRateScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(self, schedule: dict):
        """
        schedule: {epoch: learning_rate}
        Example: {0: 0.001, 50: 0.0005, 100: 0.0001}
        """
        self.schedule = schedule
    
    def __call__(self, optimizer, epoch):
        if epoch in self.schedule:
            lr = self.schedule[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logger.info(f"📉 Learning rate updated to {lr} at epoch {epoch}")
        
        return None


class TensorBoardLogger:
    """Log metrics to TensorBoard"""
    
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    def __call__(self, model, epoch, loss, metrics):
        self.writer.add_scalar('Loss/train', loss, epoch)
        
        for name, value in metrics.items():
            self.writer.add_scalar(f'Metrics/{name}', value, epoch)
        
        # Log gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', 
                                         param.grad, epoch)
        
        return None


class CSVLogger:
    """Log metrics to CSV file"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.epochs = []
        self.metrics_history = []
    
    def __call__(self, model, epoch, loss, metrics):
        import csv
        
        self.epochs.append(epoch)
        self.metrics_history.append({
            'epoch': epoch,
            'loss': loss,
            **metrics
        })
        
        # Write to CSV
        if epoch == 0:
            mode = 'w'
        else:
            mode = 'a'
        
        with open(self.filepath, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', *metrics.keys()])
            
            if epoch == 0:
                writer.writeheader()
            
            writer.writerow({
                'epoch': epoch,
                'loss': loss,
                **metrics
            })
        
        return None