import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math

class SSD(nn.Module):
    """
    Structured State Space Duality (SSD) - Core of Mamba-2
    """
    def __init__(self, d_model: int, d_state: int = 64, d_head: int = 64, num_heads: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.num_heads = num_heads
        
        # Multi-head setup
        self.d_inner = d_head * num_heads
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + 2 * d_state, bias=False)
        
        # Convolution for local processing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner
        )
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(num_heads, d_head, d_state))
        
        # Normalization
        self.norm = nn.LayerNorm(self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize A matrix for stability
        with torch.no_grad():
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
            A = A.unsqueeze(0).unsqueeze(0).repeat(self.num_heads, self.d_head, 1)
            self.A.copy_(-torch.log(A))
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        xBC = self.in_proj(x)  # (batch, seq_len, d_inner * 2 + 2 * d_state)
        
        # Split projections
        split_sizes = [self.d_inner, self.d_inner, self.d_state, self.d_state]
        x, z, B, C = torch.split(xBC, split_sizes, dim=-1)
        
        # Convolution for local context
        x_conv = x.transpose(-1, -2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Remove padding
        x = x_conv.transpose(-1, -2)  # (batch, seq_len, d_inner)
        
        # Apply activation
        x = F.silu(x)
        
        # Reshape for multi-head processing
        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        
        # State space computation
        y = self.ssd_scan(x, B, C)
        
        # Reshape back
        y = y.view(batch_size, seq_len, self.d_inner)
        
        # Gating
        y = y * F.silu(z)
        
        # Normalization and output projection
        y = self.norm(y)
        return self.out_proj(y)
    
    def ssd_scan(self, x, B, C):
        """
        Structured State Space Duality scan
        """
        batch_size, seq_len, num_heads, d_head = x.shape
        
        # Get discrete A matrix
        A = -torch.exp(self.A)  # (num_heads, d_head, d_state)
        
        # Initialize state
        h = torch.zeros(batch_size, num_heads, d_head, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Current input and projections
            x_t = x[:, t]  # (batch_size, num_heads, d_head)
            B_t = B[:, t]  # (batch_size, d_state)
            C_t = C[:, t]  # (batch_size, d_state)
            
            # Discretize A matrix
            A_discrete = torch.exp(A.unsqueeze(0))  # (1, num_heads, d_head, d_state)
            
            # Update state: h = A * h + B * x
            # Broadcast B_t and x_t for the update
            B_expanded = B_t.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, d_state)
            x_expanded = x_t.unsqueeze(-1)  # (batch_size, num_heads, d_head, 1)
            
            h = h * A_discrete + B_expanded * x_expanded
            
            # Compute output: y = C * h
            # Sum over state dimension
            C_expanded = C_t.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, 1, d_state, 1)
            y_t = (h.unsqueeze(-1) * C_expanded).sum(dim=3).squeeze(-1)  # (batch_size, num_heads, d_head)
            
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, num_heads, d_head)

class Mamba2Block(nn.Module):
    """
    Mamba-2 Block with improved architecture
    """
    def __init__(self, d_model: int, d_state: int = 64, d_head: int = 64, num_heads: int = 1, 
                 expand_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        
        # SSD layer
        self.ssd = SSD(d_model, d_state, d_head, num_heads)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        d_ff = expand_factor * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SSD with residual connection
        x = x + self.dropout(self.ssd(self.norm1(x)))
        
        # FFN with residual connection
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x

class Model(nn.Module):
    """
    Mamba-2 based ECG multi-class classifier
    Designed for ECG data shape: [batch_size, 300, 12] (12-lead ECG)
    """
    def __init__(
        self,
        configs,
        input_size: int = 12,  # 12-lead ECG
        sequence_length: int = 300,
        d_model: int = 256,
        d_state: int = 64,
        d_head: int = 64,
        num_heads: int = 4,
        num_layers: int = 8,
        num_classes: int = 10,  # Set to your number of classes
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = configs.enc_in
        self.sequence_length = configs.seq_len
        self.d_model = d_model
        self.num_classes = configs.num_class

        d_model = self.d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_length, self.d_model) * 0.02)
        
        # Mamba-2 layers
        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=self.d_model,
                d_state=d_state,
                d_head=d_head,
                num_heads=num_heads,
                expand_factor=expand_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, self.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, a, b, c):
        """
        x: (batch_size, sequence_length, input_size) = (batch_size, 300, 12)
        """
        # print("shape -> ", x.shape)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, 300, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Pass through Mamba-2 layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Global pooling (you can also try attention pooling)
        # Option 1: Mean pooling
        x = x.mean(dim=1)
        
        # Option 2: Max pooling (uncomment to use)
        # x = x.max(dim=1)[0]
        
        # Option 3: Attention pooling (uncomment to use)
        # attention_weights = torch.softmax(x.mean(dim=-1), dim=1)
        # x = (x * attention_weights.unsqueeze(-1)).sum(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits