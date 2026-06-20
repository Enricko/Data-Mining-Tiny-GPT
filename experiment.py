"""Compare SentencePiece tokenization modes for TinyGPT.

For each tokenizer (char / word / BPE / unigram) we train the SAME model on a
90/10 train/val split of corpus.txt. Because the corpus is tiny the model
overfits quickly, so we regularize (dropout + weight decay) and use EARLY
STOPPING: we evaluate the validation loss periodically, keep the best snapshot,
and report that. This measures generalization instead of memorization.

For each mode we report:
  - actual vocab size and number of tokens (sequence length / compression)
  - train loss at the best step, best (early-stopped) val loss, and the final
    val loss after full training (to show how badly each mode overfits)
  - validation bits-per-character (BPC) -- normalized so it is COMPARABLE
    across tokenizers; per-token loss is not, because token counts differ
  - a generated sample from the best model

Results are written to RESULTS.md.
"""
import copy
import math
import time

import torch
import sentencepiece as spm

from model import TinyGPT

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

CORPUS = "corpus.txt"
with open(CORPUS, "r", encoding="utf-8") as f:
    text = f.read()
n_chars = len(text)
n_words = len(text.split())

# Shared model + training hyperparameters (identical for every tokenizer)
block_size = 32
embedding_dim = 96
n_heads = 3
n_layers = 3
dropout = 0.2
lr = 3e-3
weight_decay = 0.1
epochs = 3000
batch_size = 32
eval_interval = 250
eval_iters = 100
PROMPT = "the stock market"

CONFIGS = [
    dict(name="char",        model_type="char",    vocab_size=200,  gen=240),
    dict(name="word",        model_type="word",    vocab_size=4000, gen=45),
    dict(name="bpe-128",     model_type="bpe",     vocab_size=128,  gen=90),
    dict(name="bpe-512",     model_type="bpe",     vocab_size=512,  gen=90),
    dict(name="unigram-512", model_type="unigram", vocab_size=512,  gen=90),
]


def train_tokenizer(cfg):
    prefix = "tok_" + cfg["name"]
    # hard_vocab_limit=False lets char/word shrink to the natural vocab size
    spm.SentencePieceTrainer.Train(
        input=CORPUS,
        model_prefix=prefix,
        model_type=cfg["model_type"],
        vocab_size=cfg["vocab_size"],
        character_coverage=1.0,
        hard_vocab_limit=False,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + ".model")
    return sp


def make_batch(d):
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate(model, splits):
    model.eval()
    out = {}
    for name, d in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = make_batch(d)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


rows = []
samples = {}

for cfg in CONFIGS:
    torch.manual_seed(1337)  # same init / data order for a fair comparison
    sp = train_tokenizer(cfg)
    vocab = sp.get_piece_size()

    data = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)
    n_tok = len(data)
    n = int(0.9 * n_tok)
    splits = {"train": data[:n], "val": data[n:]}

    model = TinyGPT(vocab, block_size, embedding_dim, n_heads, n_layers, dropout=dropout).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"val": float("inf"), "step": 0, "train": None, "state": None}
    last = None
    init_val = None
    t0 = time.time()
    for step in range(epochs + 1):
        if step % eval_interval == 0 or step == epochs:
            l = estimate(model, splits)
            last = l
            if step == 0:
                init_val = l["val"]  # random-init baseline (~ ln(vocab))
            if l["val"] < best["val"]:
                best = {"val": l["val"], "step": step, "train": l["train"],
                        "state": copy.deepcopy(model.state_dict())}
        if step < epochs:
            xb, yb = make_batch(splits["train"])
            _, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    dt = time.time() - t0

    final_val = last["val"]
    # restore the best (early-stopped) weights for fair scoring + generation
    model.load_state_dict(best["state"])
    chars_per_tok = n_chars / n_tok
    val_bpc = best["val"] / math.log(2) / chars_per_tok

    model.eval()
    ctx = torch.tensor([sp.encode(PROMPT)], dtype=torch.long, device=device)
    gen = model.generate(ctx, cfg["gen"], temperature=0.8, top_k=40)
    samples[cfg["name"]] = sp.decode(gen[0].tolist())

    uniform = math.log(vocab)            # cross-entropy of a uniform guess (nats)
    learned = best["step"] > 0 and best["val"] < init_val - 0.05
    rows.append(dict(
        name=cfg["name"], mt=cfg["model_type"], req=cfg["vocab_size"], vocab=vocab,
        tokens=n_tok, cpt=chars_per_tok, params=params, uniform=uniform,
        initval=init_val, train=best["train"], bestval=best["val"], step=best["step"],
        bpc=val_bpc, finalval=final_val, learned=learned, dt=dt,
    ))
    print(f"[done] {cfg['name']:12s} vocab={vocab:4d} tokens={n_tok:6d} "
          f"init_val={init_val:.3f} best_val={best['val']:.3f}@{best['step']} "
          f"final_val={final_val:.3f} bpc={val_bpc:.3f} learned={learned} ({dt:.1f}s)")


# ----------------------------
# Write RESULTS.md (table + samples). Analysis prose is appended afterward.
# ----------------------------
md = []
md.append("# Tiny GPT - Tokenization Experiments & Analysis\n\n")
md.append(f"- **Corpus:** `corpus.txt` - {n_words} words, {n_chars} characters (topic: the stock market)\n")
md.append(f"- **Model (identical for every run):** block_size={block_size}, "
          f"embedding_dim={embedding_dim}, n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}\n")
md.append(f"- **Training:** AdamW, lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}, "
          f"{epochs} steps, 90/10 train/val split, early stopping on val loss "
          f"(eval every {eval_interval} steps), device=`{device}`, seed=1337\n\n")

md.append("## Results\n\n")
md.append("| Mode | model_type | Vocab | Tokens | Chars/Tok | Params | "
          "Random-init val | Best val (step) | Final val | Learned? | Val BPC |\n")
md.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
for r in rows:
    md.append(f"| {r['name']} | {r['mt']} | {r['req']}->{r['vocab']} | {r['tokens']} | "
              f"{r['cpt']:.2f} | {r['params']:,} | {r['initval']:.3f} | "
              f"{r['bestval']:.3f} (@{r['step']}) | {r['finalval']:.3f} | "
              f"{'**yes**' if r['learned'] else 'no'} | {r['bpc']:.3f} |\n")
md.append("\n> **Random-init val** is the validation loss before any training "
          "(approximately `ln(vocab)`); it is the score of pure guessing. A tokenizer only "
          "*learned* something that generalizes if its **best val** drops clearly below this "
          "baseline (and the best step is > 0).\n>\n"
          "> **Val BPC (bits per character)** = `best_val_loss / ln(2) / (chars per token)`. "
          "Per-token loss is not comparable across tokenizers (each cuts the text into a "
          "different number of tokens), so we normalize per character. **Lower is better -- "
          "but only meaningful for rows that actually learned.** For the `no` rows the BPC just "
          "reflects the entropy of guessing among long tokens, not real modeling.\n>\n"
          f"> **Final val** is the loss after all {epochs} steps; the gap above best val shows "
          "how hard each mode overfits this tiny corpus.\n\n")

md.append(f'## Generated samples (prompt: "{PROMPT}", best model, temp=0.8, top_k=40)\n')
for r in rows:
    md.append(f"\n**{r['name']}**\n\n```\n{samples[r['name']].strip()}\n```\n")

md.append("\n---\n\n_See **[ANALYSIS.md](ANALYSIS.md)** for the full written "
          "performance analysis._\n")

with open("RESULTS.md", "w", encoding="utf-8") as f:
    f.write("".join(md))

print("\nwrote RESULTS.md")
