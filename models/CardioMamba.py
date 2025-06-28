import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple
import math

class MambaBlock(nn.Module):
    """
    Mamba block implementation for sequential modeling
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        
        # Initialize A parameter (diagonal state matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Initialize D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (batch, length, dim)
        """
        batch, length, dim = x.shape
        
        # Skip connection
        residual = x
        
        # Input projection
        xz = self.in_proj(x)  # (batch, length, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # (batch, length, d_inner)
        
        # Convolution
        x = x.transpose(-1, -2)  # (batch, d_inner, length)
        x = self.conv1d(x)[..., :length]  # Remove padding
        x = x.transpose(-1, -2)  # (batch, length, d_inner)
        
        # SiLU activation
        x = F.silu(x)
        
        # SSM operation
        y = self.ssm(x)
        
        # Gating mechanism
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        # Residual connection and normalization
        return self.norm(output + residual)
    
    def ssm(self, x):
        """
        Selective State Space Model computation
        """
        batch, length, d_inner = x.shape
        
        # Get SSM parameters
        delta, B, C = self.x_proj(x).split([d_inner, self.d_state, self.d_state], dim=-1)
        
        # Apply dt projection and softplus
        delta = F.softplus(self.dt_proj(delta))
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize continuous parameters
        deltaA = torch.exp(torch.einsum('bld,nd->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln->bldn', delta, B) * x.unsqueeze(-1)
        
        # Selective scan
        last_state = None
        outputs = []
        
        for i in range(length):
            if last_state is None:
                state = deltaB_u[:, i]  # (batch, d_inner, d_state)
            else:
                state = last_state * deltaA[:, i] + deltaB_u[:, i]
            
            y = torch.einsum('bdn,bn->bd', state, C[:, i])
            outputs.append(y)
            last_state = state
        
        y = torch.stack(outputs, dim=1)  # (batch, length, d_inner)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y
    

class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class ECGMambaClassifier(nn.Module):
    """
    Mamba-based ECG classification model
    """
    def __init__(
        self,
        input_size: int = 1,  # Number of ECG leads
        d_model: int = 128,
        n_layers: int = 6,
        num_classes: int = 5,  # Number of arrhythmia classes
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input embedding
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding (optional for ECG)
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        """
        x: (batch, sequence_length, input_size)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
