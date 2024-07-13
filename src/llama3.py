import torch
from model import Llama3Model


class Llama3ForCausalLM(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = Llama3Model()
        self.linear = torch.nn.Linear(1024, 128256, bias=False)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.tensor, labels: torch.Tensor = None):
        logits = self.model(input_ids, attention_mask)
        logits = self.linear(logits)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, 128256)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        
        return loss, logits
    

if __name__ == "__main__":
    input = {
        'input_ids': torch.randint(100, 50000, [4, 125]),
        'attention_mask': torch.ones(4, 125).long(),
        'labels': torch.randint(100, 50000, [4, 125]),
    }

    input['attention_mask'][:, 120:] = 0

    loss, logits = Llama3ForCausalLM()(**input)
    print(loss)
    print(logits.shape)