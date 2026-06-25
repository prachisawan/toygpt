import torch

# 1. Load the text again
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Recreate our character mapping from before
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]

# 3. Convert the ENTIRE dataset into a giant PyTorch Tensor (a mathematical array)
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Total tokens in dataset: {data.shape[0]}")

# 4. Split into Train (90%) and Validation (10%) sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train tokens: {len(train_data)} | Validation tokens: {len(val_data)}")

# 5. Define our GPT's "viewing window" (Block Size / Context Length)
block_size = 8 # The model will look at 8 characters to predict the 9th

print("\n--- Example of Inputs (x) and Targets (y) ---")
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context.tolist()}, the target next number is: {target}")