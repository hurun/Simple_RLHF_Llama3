# -*- coding: utf-8 -*-
"""
@Time: 2024/7/10 9:10
@Auth: hurunlong
"""
import math
import torch
from embedding import RotoryEmbedding
from common.utils import repeat_kv, get_causal_mask


class LlamaAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(1024, 1024, bias=False)
        self.k_proj = torch.nn.Linear(1024, 256, bias=False)
        self.v_proj = torch.nn.Linear(1024, 256, bias=False)
        self.o_proj = torch.nn.Linear(1024, 1024, bias=False)
        self.rotary_emb = RotoryEmbedding()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # [batch, seq_len, dims]
        # [batch, seq_len]
        batch, seq_len, dims = hidden_states.shape

        # [batch, seq_len, dims] -> [batch, seq_len, dims] -> [batch, seq_len, 32, 32] -> [batch, 32, seq_len, 32]
        q: torch.Tensor = self.q_proj(hidden_states).reshape(batch, seq_len, 32, 32).transpose(1, 2)
        # [batch, seq_len, dims] -> [batch, seq_len, dims/4] -> [batch, seq_len, 8, 32] -> [batch, 8, seq_len, 32]
        k: torch.Tensor = self.k_proj(hidden_states).reshape(batch, seq_len, 8, 32).transpose(1, 2)
        # [batch, seq_len, dims] -> [batch, seq_len, dims/4] -> [batch, seq_len, 8, 32] -> [batch, 8, seq_len, 32]
        v: torch.Tensor = self.v_proj(hidden_states).reshape(batch, seq_len, 8, 32).transpose(1, 2)

        # [batch, 32, seq_len, 32] -> [batch, 32, seq_len, 32]
        q = self.rotary_emb(q)
        # [batch, 8, seq_len, 32] -> [batch, 8, seq_len, 32]
        k = self.rotary_emb(k)
        # [batch, 8, seq_len, 32] -> [batch, 32, seq_len, 32]
        k = repeat_kv(k)
        v = repeat_kv(v)

        # [batch, 32, seq_len, 32] * [batch, 32, 32, seq_len] -> [batch, 32, seq_len, seq_len]
        attention = q.matmul(k.transpose(2, 3))/math.sqrt(32)
        # [batch, seq_len]
        attention_mask = get_causal_mask(attention_mask)
        # [batch, 32, seq_len, seq_len]
        attention = (attention + attention_mask).softmax(dim=3)

        # [batch, 32, seq_len, seq_len] * [batch, 32, seq_len, 32] -> [batch, 32, seq_len, 32]
        score = attention.matmul(v)

        # 合并多头注意力
        score = score.transpose(1, 2).reshape(batch, seq_len, dims)
        # [batch, seq_len, dims] -> [dims, dims] -> [batch, seq_len, dims]
        output = self.o_proj(score)
        return output


if __name__ == "__main__":
    input = {
        'hidden_states': torch.randn(4, 125, 1024),
        'attention_mask': torch.ones(4, 125)
    }
    print(LlamaAttention()(**input).shape)