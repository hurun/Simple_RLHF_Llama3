import torch

class RotoryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = 500000

    def _rotary_embedding(self, dim: int, seq_len: int):
        theta = 1/(self.base ** (torch.arange(0, dim, 2)/dim))
        position_idx = torch.arange(seq_len)

        idx_theta = torch.einsum("m, d->md", position_idx, theta)
        emb = torch.concat([idx_theta, idx_theta], dim=1)
        cos_cached = emb.cos()[None, :, :]
        sin_cached = emb.sin()[None, :, :]
        return cos_cached, sin_cached

    def rotate_half(self, x):
        left = x[..., :x.shape[-1]//2]
        right = x[..., x.shape[-1]//2:]
        return torch.concat([-right, left], dim=-1) 

    def forward(self, x):
        # x -> [4, 32, 125, 32]
        _, _, seq_len, dim = x.shape
        # [1, 1, 125, 32]
        cos, sin = self._rotary_embedding(dim, seq_len)
        print(cos.shape)

        # [1, 125, 32] -> [1, 1, 125, 32]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # [4, 32, 125, 32] * [1, 1, 125, 32]
        x = x * cos + self.rotate_half(x) * sin
        return x


if __name__ == "__main__":
    input = {
    'x': torch.randn(4, 32, 125, 32),
    'sin': torch.randn(1, 125, 32),
    'cos': torch.randn(1, 125, 32)
    }
    print(RotoryEmbedding()(input["x"]).shape)