# Import statements for the main model file
# Add these imports to your Cardioformer.py file:
# from layers.SelfAttention_Cardio_Mamba import CardioformerMambaLayer
# from layers.Cardioformer_EncDec_Mamba import EncoderLayerMamba, Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import ListPatchEmbedding
from layers.CardioSSM_EncDec import Encoder, EncoderLayerMamba
from layers.SelectiveSSM_CardioSSM import CardioMambaLayer

# Modified main model file - replace your Model class with this
class Model(nn.Module):
    """
    Cardioformer with Mamba (SSM) instead of attention mechanism
    Everything else remains the same including patch embedding
    """
    def __init__(self, configs):
        super(Model, self).__init__()
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
                    CardioMambaLayer(
                        len(patch_len_list),
                        configs.d_model,
                        d_state=getattr(configs, 'd_state', 8),
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