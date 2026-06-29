import torch
import torch.nn as nn
from torch.nn import functional as F

# Upgraded Hyperparameters for better text coherence
batch_size = 64     # Keep at 64 for stable gradients
block_size = 128     # DOUBLE the context! The model can now look back 128 characters.
max_iters = 8000    # Increase training steps so it has more time to learn
learning_rate = 3e-4 # Lower learning rate so the deeper network doesn't over-correct
n_embd = 256        # Double the vector size (Massively expands spelling memory)
n_head = 8          # 8 parallel attention heads (256 / 8 = 32 dims per head)
n_layer = 6         # Stack 6 full Transformer Blocks for serious deep learning
dropout = 0.2       # Dropout rate to prevent overfitting
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# 1. Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- TRANSFORMER COMPONENTS ---

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # causal mask (don't look into the future!)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity (giving the model time to think) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (feed-forward) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Communication with residual connection
        x = x + self.ffwd(self.ln2(x))  # Computation with residual connection
        return x

# --- THE GPT MODEL ---

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token looks up its embedding vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each position looks up its positional embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Stack our Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd)
        x = self.blocks(x) # (B,T,n_embd)
        x = self.ln_f(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so positional embeddings don't blow up
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize Upgraded Model
model = GPTLanguageModel()
m = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print("\n--- Training Upgraded Transformer ---")
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 500 == 0:
        print(f"step {iter}: loss {loss.item():.4f}")

print(f"Final training loss: {loss.item():.4f}")
print("--- Training Finished ---\n")

print("--- Generating sample text from your Transformer ---")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300)[0].tolist()))

# Save the model's brain!
torch.save(m.state_dict(), 'model_weights.pth')
print("Model weights saved to 'model_weights.pth'")