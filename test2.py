import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
context = torch.zeros((1,1), dtype=torch.long, device=device)

idx_cond = context[:,-8]

