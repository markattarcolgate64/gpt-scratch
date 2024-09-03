from torch import nn
from multi_head import MultiHeadAttention
from feed_forward import FeedForward
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    #
    def __init__(self, n_embed, n_head, block_size):
        # n_embed: the embedding dimension or number of embeddings, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head 
        self.sa = MultiHeadAttention(n_head, head_size, block_size, n_embed)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
    