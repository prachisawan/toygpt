# 1. Read the text file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Length of dataset in characters: {len(text)}")

# 2. Find all unique characters (our 'vocabulary')
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

# 3. Create dictionaries to map characters to numbers (and vice versa)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# 4. Create encoder and decoder functions
def encode(s):
    return [stoi[c] for c in s] # Translates string to numbers

def decode(l):
    return ''.join([itos[i] for i in l]) # Translates numbers back to string

# 5. Let's test it on the first 15 characters of your dataset!
test_string = text[:15]
encoded_numbers = encode(test_string)
decoded_string = decode(encoded_numbers)

print(f"\nOriginal text snippet: {repr(test_string)}")
print(f"Encoded into numbers: {encoded_numbers}")
print(f"Decoded back to text: {repr(decoded_string)}")