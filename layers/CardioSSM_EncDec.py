import torch
import torch.nn as nn
import torch.nn.functional as F


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