# Main Project 1

## Overview

This comprehensive text generation project is meticulously crafted, leveraging PyTorch documentation and various internet resources. The project's robustness, efficiency, and versatility are indebted to the powerful combination of Python as a programming language and the advanced capabilities provided by PyTorch library. I deeply appreciate the continuous efforts of these open-source communities, which have been instrumental in the development and success of this project.

## Full Code
```python
# IMPORT MODULES
import json
import math
import tqdm
import torch
import tokenizers
import torch.nn as nn
from tqdm import auto
from torch import Tensor
from torch.nn import functional as F
from tokenizers import ByteLevelBPETokenizer

# FILL HYPERPARAMETERS
data_path = "data.txt"
batch_size = 12
block_size = 512
max_iters = 10000
eval_interval = 500
learning_rate = 5e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 12
n_head = 6
n_layer = 9
dropout = 0.2
expansion_factor = 4
vocab_size = 5000
min_frequency = 2
normalization_factor = -0.5
mean = 0.0
std = 0.2
bias = True
dtype = torch.long
split_ratio = 0.9
torch.manual_seed(1337)

# GET DATA AND MAKE CUSTOM TOKENIZER
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = ByteLevelBPETokenizer()
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
tokenizer.train(files=[data_path], vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
tokenizer.save_model("")

with open("vocab.json") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# MAKE CONFIGURATION FILE
config = {
  "block_size": block_size,
  "n_embd": n_embd,
  "n_layer": n_layer,
  "n_head": n_head,
  "vocab_size": vocab_size,
  "dropout": dropout
  }
  
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

# SPLIT DATA INTO TWO PART'S TRAIN AND VALIDATION
data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
n = int(split_ratio*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):    
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# LOSS FUNCTION
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# GELUActivation FUNCTION 
class GELUActivation(nn.Module):    
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)

# TRANSFORMER ARCHITECTURE
class SelfAttentionHead(nn.Module):
    def __init__(self, input_size, head_size, block_size, dropout, bias, dimension):
        super(SelfAttentionHead, self).__init__()
        self.key = nn.Linear(input_size, head_size, bias=bias)
        self.query = nn.Linear(input_size, head_size, bias=bias)
        self.value = nn.Linear(input_size, head_size, bias=bias)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** normalization_factor
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=dimension)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, dimension):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=dimension)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion_factor  * n_embd),
            GELUActivation(),
            nn.Linear(expansion_factor  * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super(TransformerBlock, self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, block_size=block_size, dropout=dropout):
        super(TextGenerationModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
  
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens=50, num_return_sequences=1, temperature=1.0, top_k=0, top_p=1.0, num_samples=1, dim=-1, descending=True):        
        for _ in range(max_tokens):           
            idx_cond = idx[:, -block_size:]          
            logits, loss = self(idx_cond)            
            logits = logits[:, -1, :]            
            probs = F.softmax(logits / temperature, dim=dim) 
           
            if top_k > 0:
                probs, indices = torch.topk(probs, top_k, dim=dim)
                idx_next = torch.multinomial(probs, num_samples=num_samples) 
            elif top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=descending)
                cumulative_probs = torch.cumsum(sorted_probs, dim=dim)                
                exceed_threshold = cumulative_probs > top_p
                sorted_indices[exceed_threshold] = -1
                idx_next = sorted_indices[:, 0].unsqueeze(1) 
            else:             
                idx_next = torch.multinomial(probs, num_samples=num_samples)               
            idx = torch.cat((idx, idx_next), dim=dim)                
        idx = idx.repeat(1, num_return_sequences)
        return idx

# TRAINING PROCESS
model = TextGenerationModel().to(device)
print("Your model parameters will be ", sum(p.numel() for p in model.parameters())/1e6, " Million")
sample = torch.zeros((1, 1), dtype=dtype, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in auto.tqdm(range(max_iters)):        
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss()
        torch.save(model.state_dict(), f'model{iter}.bin')
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Checkpoint model saved!") 
        print("\nProcessed Text:\n", tokenizer.decode(model.generate(sample, max_tokens=500)[0].tolist()),"\n")

    xb, yb = get_batch('train')    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```
