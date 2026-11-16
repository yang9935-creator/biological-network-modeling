import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention between two sequences"""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context):
        B, N, D = x.shape
        _, M, _ = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, self.dim_head)
        q = q.permute(0, 2, 1, 3)
        
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * (self.dim_head ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        
        return x


class FeatureWiseLinear(nn.Module):
    """Feature-wise linear modulation (FiLM)"""
    
    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)
        
    def forward(self, x, condition):
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        
        return gamma * x + beta


class AdaptivePooling(nn.Module):
    """Adaptive pooling that learns to weight features"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, dim) / dim)
        
    def forward(self, x):
        weights = F.softmax(self.weights, dim=-1)
        return (x * weights).sum(dim=-1, keepdim=True)


class UncertaintyHead(nn.Module):
    """Output head that predicts both mean and uncertainty"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        mean = self.mean(x)
        log_var = torch.clamp(self.log_var(x), min=-10, max=10)
        
        return mean, log_var