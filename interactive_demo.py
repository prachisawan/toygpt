import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# Configuration (Matches your upgraded 4.8M parameters model)
block_size = 128    
n_embd = 256        
n_head = 8          
n_layer = 6         
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 1. Load data to rebuild vocabulary map
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] 
decode = lambda l: ''.join([itos[i] for i in l])

# --- TRANSFORMER ARCHITECTURE ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        return logits, None
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- LOAD THE BRAIN ---
model = GPTLanguageModel()

# Calculate parameters seamlessly
total_params = sum(p.numel() for p in model.parameters())

if os.path.exists('model_weights.pth'):
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print("==================================================")
    print(f"🧠 MODEL ONLINE | Size: {total_params:,} parameters")
    print("✅ Pre-trained weights loaded successfully (Loss: 1.10)")
    print("==================================================")
else:
    print("❌ Error: 'model_weights.pth' not found.")
    exit()

print("\n🤖 MANN KI BAAT PERSONA EMULATOR ACTIVE 🤖")
print("Type a starting phrase in Hindi and hit Enter.")
print("Type 'exit' to quit.\n")

while True:
    user_prompt = input("Enter your seed prompt: ")
    if user_prompt.lower() == 'exit':
        break
    if not user_prompt.strip():
        continue
    
    encoded_prompt = encode(user_prompt)
    if not encoded_prompt:
        print("Error: Characters not recognized.")
        continue
        
    context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    
    print("\nCompleting speech...")
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=200)[0].tolist()
        
    print("\n" + decode(generated_tokens))
    print("\n" + "-"*50)