# ReWaS
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
import einops
from functools import partial
from encoder.transformer import MultiheadAttention, SimpleTransformer
import torch

class Phi(nn.Module):
    def __init__(self, input_dim=768, out_dim=1, proj_dims=[768, 128, 64, 16, 1]):
        super().__init__()

        self.projection1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(0.3),
        )

        self.hint_blocks = SimpleTransformer(
            attn_target = partial(
                    MultiheadAttention,
                    embed_dim=input_dim,
                    num_heads=8,
                    bias=True,
                    add_bias_kv=True,
                    dropout=0.3,
                    batch_first=True
                ),
            embed_dim = input_dim,
            num_blocks = 3,
            weight_init_style = "pytorch",  # possible values jax or pytorch
        )
        

        self.projection2 = nn.Sequential(
            nn.Linear(768,768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768,527),
            nn.ReLU()
        )

    def forward(self, x, attn_mask=None):
        x = self.projection1(x)
        x = self.hint_blocks(x, attn_mask)
        x = self.projection2(x)
        return x