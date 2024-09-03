#Getting the dataset, in this instance the tiny shakespeare dataset
import requests
import torch
from bigram import BigramLanguageModel
from multi_head import MultiHeadAttention
from feed_forward import FeedForward
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
torch.manual_seed(1337)


chars = sorted(list(set(text)))
vocab_size = len(chars)


#create the encoder and decoder 
stoi = { ch: i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)



data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
print(data[:1000])

#Data is split into training and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[:n]

block_size = 8

train_data[:block_size+1]
#What this says when plugged into transformer is that it will
#make a prediction for each position in the array
#Context: 18, then 47 comes next
#Context: 18,47 then 56 comes next
#And so on...
#It is predicting the next character up to block size
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     #Gets the chunks of characters up to block size
#     context = x[:t+1]
#     target = y[t]




def get_batch(split):
    #Generate as small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model = model.eval() 
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split]  = losses.mean()
    model = model.train()

    return out


    

#xb, yb = get_batch('train')

# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print("\n\n")
# #Goes through the batches
# for b in range(batch_size):
#     #Goes through the blocks
#     for t in range(block_size):
#         #All the numbers up to the target
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")


bigramModel = BigramLanguageModel(vocab_size, n_embed=32, block_size=block_size)
# logits,loss = bigramModel(xb, yb)
# print(loss)
# #Small, 1x1 tensor that is the index which we will feed
# idx = torch.zeros((1,1), dtype=torch.long)
# print(decode(bigramModel.generate(idx,max_new_tokens=100)[0].tolist()))
#[0].toList()

#Now we will train the model so that the output won't be random nonsense
optimizer = torch.optim.AdamW(bigramModel.parameters(), lr=1e-3)
#Take the gradients and update parameters based on the gradients

for iter in range(max_iters):

    #On the interval we've chosen to regularlu evaluate the loss
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss(bigramModel)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")


    xb, yb = get_batch('train')

    logits, loss = bigramModel(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(bigramModel.generate(context,max_new_tokens=500)[0].tolist()))
