# SelfAttention_Cardio_Mamba.py
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
        
        # Scan
        last_state = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
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


# Modified Encoder Layer for Mamba
class EncoderLayerMamba(nn.Module):
    def __init__(self, mamba_layer, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayerMamba, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba_layer = mamba_layer
        
        # Keep the same ResNet blocks as original
        self.resblock1 = ResNetBlock_type1(d_model, d_ff, dropout, activation, identity=True)
        self.resblock2 = ResNetBlock_type1(d_model, d_ff, dropout, activation, identity=True)
        self.resblock3 = ResNetBlock_type1(d_model, d_ff, dropout, activation, identity=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Mamba processing (replacing attention)
        new_x, attn = self.mamba_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

        y = x = [self.norm1(_x) for _x in x]
        y = [self.resblock1(_y) for _y in y]
        y = [self.resblock2(_y) for _y in y]
        y = [self.resblock3(_y) for _y in y]
        
        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


# ResNet blocks (keeping the same as original)
class ResNetBlock_type1(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation="relu", identity=False):
        super(ResNetBlock_type1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.identity = identity
        if not self.identity:
            self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x.transpose(-1, 1)))
        x = self.dropout(x)
        x = self.conv2(x).transpose(-1, 1)
        x = self.dropout(x)

        if not self.identity:
            residual = self.conv3(residual.transpose(-1, 1))
            residual = residual.transpose(-1, 1)
            residual = self.dropout(residual)

        return self.norm(residual + x)


# Import statements for the main model file
# Add these imports to your Cardioformer.py file:
# from layers.SelfAttention_Cardio_Mamba import CardioformerMambaLayer
# from layers.Cardioformer_EncDec_Mamba import EncoderLayerMamba, Encoder

# Modified main model file - replace your Model class with this
class CardioformerMamba(nn.Module):
    """
    Cardioformer with Mamba (SSM) instead of attention mechanism
    Everything else remains the same including patch embedding
    """
    def __init__(self, configs):
        super(CardioformerMamba, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.single_channel = configs.single_channel
        
        # Embedding (keeping exactly the same)
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")

        self.enc_embedding = ListPatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len_list,
            stride_list,
            configs.dropout,
            augmentations,
            configs.single_channel,
        )
        
        # Encoder with Mamba layers instead of attention
        self.encoder = Encoder(
            [
                EncoderLayerMamba(
                    CardioformerMambaLayer(
                        len(patch_len_list),
                        configs.d_model,
                        d_state=getattr(configs, 'd_state', 16),
                        d_conv=getattr(configs, 'd_conv', 4),
                        expand_factor=getattr(configs, 'expand_factor', 2),
                        dropout=configs.dropout,
                        output_attention=configs.output_attention,
                        no_inter=configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        
        # Decoder (keeping exactly the same)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model
                * sum(patch_num_list)
                * (1 if not self.single_channel else configs.enc_in),
                configs.num_class,
            )

    def classification(self, x_enc, x_mark_enc):
        # Embedding (same as original)
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output (same as original)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification":
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None


# Encoder class (keeping the same structure)
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        print_pls = False
        if print_pls:
            print("(Encoder) layers size should be 6 and here is: ", len(self.attn_layers))
        
        attns = []
        if print_pls:
            print("input in (Encoder) class x: and len", x[0].shape, len(x))

        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            if print_pls:
                print("output in (Encoder) inside the layers loop x: and len", x[0].shape, len(x))
            attns.append(attn)

        # Concatenate all outputs
        x = torch.cat(x, dim=1)  # (batch_size, patch_num_1 + patch_num_2 + ..., d_model)

        if print_pls:
            print("output before normalization layer inside (Encoder) x: and len", x.shape)

        if self.norm is not None:
            x = self.norm(x)

        if print_pls:
            print("output after normalization layer inside (Encoder) x: and len", x.shape)

        return x, attns