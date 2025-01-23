import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

        y = x = [self.norm1(_x) for _x in x]
        y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
        y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]

        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]

        print_pls = True
        if print_pls:
            print("(Encoder) attn layers size should be 6 and here is: ", len(self.attn_layers))
        attns = []

        if print_pls:
            print("input in (Encoder) class x: ", x)

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)

            if print_pls:
                print("output in (Encoder) inside the attn_layers loop x: ", x)

            attns.append(attn)

        # concat all the outputs
        x = torch.cat(
            x, dim=1
        )  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)

        if print_pls:
            print("output before normalization layer inside (Encoder) x: ", x)

        if self.norm is not None:
            x = self.norm(x)
            
        if print_pls:
            print("output after normalization layer inside (Encoder) x: ", x)

        return x, attns
