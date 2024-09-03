import torch
import torch.nn as nn
from torch.nn import functional as f 
from head import Head
from multi_head import MultiHeadAttention
from feed_forward import FeedForward
from block import Block

class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size, n_embed, block_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        #Positional embedding for each token
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )

    def forward(self,idx, targets=None):
        
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        total_emb = tok_emb + pos_emb
        total_emb = self.sa_heads(total_emb) 
        total_emb = self.ffwd(total_emb) #(B,T,C)
        logits = self.lm_head(total_emb) #(B,T,C)
        
        if targets is None:
            loss = None
        else:
            #Idx and targets are both (B,T) tensor of integers
            #Cross entropy loss function
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = f.cross_entropy(logits, targets)

        return logits,loss

    def generate(self,idx, max_new_tokens):
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size
            idx_cond = idx[:,-self.block_size:]
            #get predictions
            logits, loss = self(idx_cond)
            #focus only on the final time step
            logits = logits[:,-1,:]
            #apply softmax to get probabilities
            probs = f.softmax(logits, dim=-1)
            #apply sampling to distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            #append sampled index to output
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

