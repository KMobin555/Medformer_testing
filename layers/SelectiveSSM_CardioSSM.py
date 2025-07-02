import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

print_all = False

class SelectiveStateSpaceModel(nn.Module):
    """
    Selective State Space Model (SSM) implementation similar to Mamba
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dt_rank=None, dt_min=0.001, dt_max=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)
        
        dt_rank = dt_rank or math.ceil(d_model / 16)
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt = torch.exp(
            torch.rand(self.d_inner, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # A parameter (diagonal)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # (B, d_inner, L)
        x = rearrange(x, 'b d l -> b l d')
        
        # Activation
        x = F.silu(x)
        
        # SSM
        y = self.ssm(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        """
        Selective State Space Model computation
        x: (B, L, d_inner)
        """
        B, L, D = x.shape
        N = self.d_state
        
        # Compute delta, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        delta, B, C = torch.split(x_dbl, [self.dt_proj.in_features, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Get A
        A = -torch.exp(self.A_log.float())  # (d_inner, N)
        
        # Discretization
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))  # (B, L, d_inner, N)
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, x)  # (B, L, d_inner, N)
        
        print("b d n -> ", B, D, N)
        # Scan
        # last_state = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        last_state = torch.zeros((B, N, D), device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            last_state = deltaA[:, i] * last_state + deltaB_u[:, i]  # (B, d_inner, N)
            y = torch.einsum('bdn,bn->bd', last_state, C[:, i])  # (B, d_inner)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Skip connection
        y = y + x * self.D
        
        return y


class MambaLayer(nn.Module):
    """
    Mamba Layer to replace attention mechanism
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.ssm = SelectiveStateSpaceModel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: (B, L, D)
        Returns: (B, L, D), None (to match attention interface)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = x + residual
        return x, None  # Return None for attention weights compatibility


class CardioformerMambaLayer(nn.Module):
    """
    Modified Cardioformer layer using Mamba instead of attention
    """
    def __init__(
        self,
        num_blocks,
        d_model,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()
        
        # Intra-block Mamba layers (replacing intra-attention)
        self.intra_mambas = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor
            )
            for _ in range(num_blocks)
        ])
        
        # Inter-block Mamba layer (replacing inter-attention)
        if no_inter or num_blocks <= 1:
            print("No inter Mamba for time")
            self.inter_mamba = None
        else:
            self.inter_mamba = MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x: List of tensors, each (B, Li, D)
        """
        if print_all:
            print(f"x len: {len(x)} and shape of x[0] = {x[0].shape}")
        
        # Intra-block processing (replacing intra-attention)
        x_intra = []
        attn_out = []  # Keep for compatibility, will be None
        
        for x_in, mamba_layer in zip(x, self.intra_mambas):
            _x_out, _attn = mamba_layer(x_in, attn_mask=None, tau=tau, delta=delta)
            x_intra.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)  # None
        
        if self.inter_mamba is not None:
            # Inter-block processing (replacing inter-attention)
            # Take the last token from each block as router
            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)  # (B, N, D)
            x_inter, attn_inter = self.inter_mamba(
                routers, attn_mask=None, tau=tau, delta=delta
            )
            
            # Combine with intra results
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)  # (B, Li, D)
                for i, x in enumerate(x_intra)
            ]
            attn_out += [attn_inter]  # None
        else:
            x_out = x_intra
            
        return x_out, attn_out