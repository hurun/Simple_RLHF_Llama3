import torch

def repeat_kv(x: torch.Tensor):
    # [4, 2, 125, 32] -> [4, 2, 4, 125, 32] -> [4, 8, 125, 32]
    shape = list(x.shape)
    shape[1] *= 4

    return x.unsqueeze(2).repeat((1, 1, 4, 1, 1)).reshape(shape)


def get_causal_mask(attention_mask: torch.Tensor, padding_token: int = 0):
    # [4, 125] -> [4, 1, 125, 125]
    batch_size, seq_len = attention_mask.shape
    min_value = -1e-5

    # [125, 125]
    causal_mask = torch.full((seq_len, seq_len), min_value).triu(diagonal=1)
    # [125, 125] -> [1, 1, 125, 125] -> [batch, 1, 125, 125]
    causal_mask = causal_mask.reshape((1, 1, seq_len, seq_len)).repeat((batch_size, 1, 1, 1))

    mask = attention_mask.reshape((batch_size, 1, 1, seq_len)) == padding_token
    # [batch, 1, 125, 125]
    causal_mask = causal_mask.masked_fill(mask, min_value)
    return causal_mask

if __name__ == "__main__":
    # print(repeat_kv(torch.randn(4, 2, 125, 32)).shape)
    print(get_causal_mask(torch.ones(4, 125).long()).shape)