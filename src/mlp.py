import torch

# 1024, 14336

class LlamaMLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up = torch.nn.Linear(1024, 14336, bias=False)
        self.gate = torch.nn.Linear(1024, 14336, bias=False)
        self.down = torch.nn.Linear(14336, 1024, bias=False)
        self.activate_fn = torch.nn.SiLU()

    def forward(self, x):
        #[4, 125, 1024] -> [4, 125, 14336]
        up = self.up(x)
        gate = self.gate(x)
        silu = self.activate_fn(gate)
        #[4, 125, 14336] -> [4, 125, 1024]
        down = self.down(up * silu)
        return down
    
if __name__ == "__main__":
    print(LlamaMLP()(torch.randn(4, 125, 1024)).shape)