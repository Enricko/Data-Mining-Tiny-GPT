# Tiny GPT

A minimal GPT-style language model, recreated from
[annaamikom/tinyGPT](https://github.com/annaamikom/tinyGPT). It trains a small
decoder-only transformer on a tiny text corpus (topic: **the stock market**) and
generates new text. It also includes an experiment comparing several tokenization
modes, with a written performance analysis.

## Files

| File                    | Purpose                                                              |
| ----------------------- | ------------------------------------------------------------------- |
| `corpus.txt`            | Training text — 2,147 words about the stock market.                 |
| `transformer_blocks.py` | Self-attention, multi-head attention, feed-forward, and Block (+dropout). |
| `model.py`              | The reusable, parameterized `TinyGPT` model.                        |
| `tinygpt.py`            | Simple single run: train one model (BPE) and generate text.        |
| `experiment.py`         | Train one model per tokenizer mode and compare them.               |
| `RESULTS.md`            | Auto-generated results table + samples (written by `experiment.py`). |
| `ANALYSIS.md`           | Written performance analysis of the experiment.                    |

## How it works

1. **Tokenize** — SentencePiece turns `corpus.txt` into integer tokens.
2. **Model** — token + position embeddings → stacked transformer `Block`s → final layer norm → linear head over the vocabulary.
3. **Train** — sample random `block_size` windows, predict the next token at each position, minimize cross-entropy with AdamW (with dropout + weight decay).
4. **Generate** — start from a prompt, repeatedly sample the next token (with temperature + top-k).

## Setup & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python tinygpt.py      # simple: train one model + generate
python experiment.py   # compare tokenizers -> writes RESULTS.md
```

> **Note (this machine):** the system Python is 3.14 and Homebrew-managed
> (externally managed), so the project uses a local `.venv`. torch 2.12 does ship
> a Python 3.14 wheel. If `pip install` ever fails on a newer Python, use 3.11/3.12.

## The tokenization experiment (assignment)

`experiment.py` trains the *same* model on a 90/10 train/val split for five
SentencePiece tokenizers — `char`, `word`, `bpe-128`, `bpe-512`, `unigram-512` —
using early stopping, and reports for each:

- actual vocab size, token count, and characters per token (compression),
- the random-init baseline, best (early-stopped) val loss, and final val loss,
- **validation bits-per-character (BPC)**, normalized so it is comparable across
  tokenizers,
- a generated sample.

**Key finding:** on a ~2000-word corpus, only the small-vocabulary tokenizers
(`char`, `bpe-128`) actually learned to generalize; `word`, `bpe-512`, and
`unigram-512` were data-starved and only memorized. Character-level gave the best
validation BPC and the most coherent text. Full discussion in
[`ANALYSIS.md`](ANALYSIS.md).

## Hyperparameters

Edit these near the top of `tinygpt.py` / `experiment.py`:

```python
block_size    = 32    # context length
embedding_dim = 96    # token vector size
n_heads       = 3     # attention heads
n_layers      = 3     # transformer blocks
dropout       = 0.2   # regularization
lr            = 3e-3
epochs        = 3000
```
