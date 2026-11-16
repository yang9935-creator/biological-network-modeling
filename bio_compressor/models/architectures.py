import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardNet(nn.Module):
    """Standard feedforward neural network"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            act_fn = nn.ELU()
        else:
            act_fn = nn.ReLU()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, dim: int, hidden_dim: int, 
                 activation: str = 'relu', 
                 dropout: float = 0.1):
        super().__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class ResidualNet(nn.Module):
    """Residual neural network"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list,
                 activation: str = 'relu',
                 dropout: float = 0.1):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        
        # Residual blocks
        blocks = []
        for hidden_dim in hidden_dims[1:]:
            blocks.append(ResidualBlock(hidden_dims[0], hidden_dim, activation, dropout))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[0], output_dim)
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.blocks(x)
        x = self.output_layer(x)
        
        return x


class TransformerNet(nn.Module):
    """Transformer encoder for biological data"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch, 1, features]
        
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        
        return self.output_layer(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LSTMNet(nn.Module):
    """LSTM for time series biological data"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.output_layer = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Add sequence dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        return self.output_layer(hidden)


class CNN1DNet(nn.Module):
    """1D CNN for biological data"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_dims: list,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, channel_dims[0])
        
        conv_layers = []
        in_channels = 1  # Add channel dimension
        
        for out_channels in channel_dims:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        self._to_linear = None
        self._get_conv_output((1, channel_dims[0]))
        
        self.output_layer = nn.Linear(self._to_linear, output_dim)
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.conv_layers(input)
            self._to_linear = int(torch.prod(torch.tensor(output.size())))  
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.output_layer(x)