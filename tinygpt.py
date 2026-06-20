"""Train a single TinyGPT on corpus.txt (BPE tokenizer) and generate text.

This is the simple, canonical script. For the multi-tokenizer comparison and
performance analysis, see experiment.py.
"""
import torch
import sentencepiece as spm

from model import TinyGPT

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__, "| device:", device)

# ----------------------------
# Tokenize with SentencePiece (BPE)
# ----------------------------
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

spm.SentencePieceTrainer.Train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=512,
    model_type="bpe",
    character_coverage=1.0,
    hard_vocab_limit=False,
)

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")
vocab_size = sp.get_piece_size()

data = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"vocab size: {vocab_size} | tokens: {len(data)}")

# ----------------------------
# Hyperparameters
# ----------------------------
block_size = 32
embedding_dim = 96
n_heads = 3
n_layers = 3
lr = 3e-3
epochs = 3000
batch_size = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


model = TinyGPT(vocab_size, block_size, embedding_dim, n_heads, n_layers, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ----------------------------
# Training loop
# ----------------------------
for step in range(epochs):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        with torch.no_grad():
            xv, yv = get_batch("val")
            _, vloss = model(xv, yv)
        print(f"step {step:4d} | train {loss.item():.3f} | val {vloss.item():.3f}")

# ----------------------------
# Generate
# ----------------------------
model.eval()
context = torch.tensor([sp.encode("the stock market")], dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=120, temperature=0.8, top_k=40)

print("\nGenerated text:\n")
print(sp.decode(out[0].tolist()))
