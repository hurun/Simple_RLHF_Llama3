import torch

from attention import LlamaAttention
from embedding import RotoryEmbedding
from mlp import LlamaMLP
from normalization import RMSNorm


class LlamaDecoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = LlamaAttention()
        self.embedding = RotoryEmbedding()
        self.mlp = LlamaMLP()
        self.norm = RMSNorm(1024)
        pass

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        #hidden_states -> [4, 125, 1024]
        #attention_mask -> [4, 125]
        norm = self.norm(hidden_states)
        attention = self.attention(hidden_states, attention_mask)
        res = attention + hidden_states

        res_norm = self.norm(res)
        mlp = self.mlp(res_norm)
        return mlp + hidden_states

if __name__ == "__main__":
    input = {
        'hidden_states': torch.randn(4, 125, 1024),
        'attention_mask': torch.ones(4, 125).long()
    }
    print(LlamaDecoder()(torch.randn(4, 125, 1024), torch.ones(4, 125).long()).shape)
