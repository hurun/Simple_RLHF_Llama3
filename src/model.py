import torch
from decoder import LlamaDecoder
from normalization import RMSNorm

class Llama3Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = torch.nn.Embedding(128256, 1024, None)
        self.layers = torch.nn.ModuleList(
            [LlamaDecoder() for _ in range(4)]
        )
        self.norm = RMSNorm(1024)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.embedding(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        norm = self.norm(hidden_states)
        return norm
    

if __name__ == "__main__":

    input = {
        'input_ids': torch.randint(100, 50000, [4, 125]),
        'attention_mask': torch.ones(4, 125).long(),
    }

    input['attention_mask'][:, 120:] = 0

    print(Llama3Model()(**input).shape)