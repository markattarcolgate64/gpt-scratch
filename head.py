import torch.nn as nn
import torch 
class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size, n_embed, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B,T,C = x.shape # Batch size, sequence length, channels
        q = self.query(x)
        k = self.key(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)

        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = nn.functional.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v

        return out