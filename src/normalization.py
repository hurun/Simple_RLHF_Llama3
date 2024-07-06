import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = torch.nn.Parameter(torch.ones(dim))
        

    def forward(self, x):
        var = x.pow(2).mean(2, keepdim=True)
        x = x * torch.rsqrt(var + 1e-5)

        return x * self.weight

if __name__ == "__main__":
    norm = RMSNorm(1024)
    value = torch.randn(4, 125, 1024)
    print(norm(value).shape)