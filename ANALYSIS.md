# Performance Analysis — Tiny GPT Tokenization Experiments

This document interprets the numbers in [`RESULTS.md`](RESULTS.md) (regenerate
them with `python experiment.py`).

## 1. What was tested

The **same** model and **same** training recipe were used five times — only the
tokenizer changed — so every difference below is caused by tokenization alone.

- Model: `block_size=32, embedding_dim=96, n_heads=3, n_layers=3, dropout=0.2`
- Training: AdamW, `lr=3e-3`, `weight_decay=0.1`, 3000 steps, 90/10 train/val
  split, **early stopping** on validation loss, seed 1337.
- Corpus: `corpus.txt`, 2147 words / 11,730 characters, all about the stock market.
- Tokenizers (SentencePiece): `char`, `word`, `bpe` (vocab 128 and 512), `unigram` (512).

## 2. The core tradeoff: vocabulary size vs sequence length

| Tokenizer | Actual vocab | Tokens | Chars / token | Avg uses per token type |
|---|---|---|---|---|
| char        | 56  | 11,710 | 1.00 | ~209 |
| bpe-128     | 128 | 6,550  | 1.79 | ~51  |
| unigram-512 | 512 | 3,722  | 3.15 | ~7.3 |
| bpe-512     | 512 | 3,571  | 3.28 | ~7.0 |
| word        | 737 | 2,147  | 5.46 | ~2.9 |

A coarser tokenizer (word) packs more characters into each token, so the
sequence is shorter and cheaper to process — but the vocabulary explodes and the
embedding/output tables grow (word = 480k params vs char = 349k). The decisive
column is the **last one**: how many times the model sees each token type during
training. This is what determines whether it can learn anything.

## 3. Headline result: only the small-vocab tokenizers actually learned

A model has only *learned something that generalizes* if its validation loss
drops clearly below the **random-init baseline** (≈ `ln(vocab)`, the loss of pure
guessing). The early-stopping step tells the story:

| Tokenizer | Random-init val | Best val (step) | Learned? |
|---|---|---|---|
| char        | 4.151 | **1.702 (@1000)** | ✅ yes |
| bpe-128     | 5.025 | **3.081 (@250)**  | ✅ yes |
| word        | 6.733 | 6.733 (**@0**)    | ❌ no |
| bpe-512     | 6.417 | 6.417 (**@0**)    | ❌ no |
| unigram-512 | 6.402 | 6.402 (**@0**)    | ❌ no |

`char` and `bpe-128` improved a lot over random. The three large-vocab tokenizers
hit their best validation loss **at step 0** — i.e. *before any training* — and
then only got worse. They never generalized; they merely memorized the training
text.

**Why:** with a fixed ~2000-word corpus, a bigger vocabulary means each token
type appears only a handful of times (word: ~2.9 times; bpe-512: ~7). The
held-out 10% tail then contains tokens and contexts the model never saw enough
of, so it cannot predict them better than chance. Small vocabularies (char ~209
uses/type) give the model enough repetition to learn real structure. **Tokenizer
vocabulary size must be matched to the amount of training data.**

## 4. Reading BPC correctly (a metric trap)

Bits-per-character normalizes loss by characters so different tokenizers are
comparable. But notice that **`word` has the *lowest* BPC (1.778) while having
learned nothing.** That is a trap: a uniform guess over 737 words already costs
only `log2(737) / 5.46 ≈ 1.75` bits per character, simply because each word is
long. Low BPC there reflects the *length of the token*, not modeling skill.

So BPC is only meaningful for rows that beat their baseline. Among the genuine
learners:

| Tokenizer | Val BPC | Uniform-guess BPC | Compression achieved |
|---|---|---|---|
| char    | **2.451** | 5.81 | ~2.4× better than random |
| bpe-128 | 2.482 | 3.81 | ~1.5× better than random |

**Character-level is the best real model**, with small BPE a close second.

## 5. Overfitting: every mode overfits, big vocabularies catastrophically

Compare **best val** (early-stopped) to **final val** (after all 3000 steps):

| Tokenizer | Best val | Final val | Final perplexity | Comment |
|---|---|---|---|---|
| char        | 1.702 | 1.914  | ~6.8     | mild overfit |
| bpe-128     | 3.081 | 5.070  | ~159     | moderate |
| unigram-512 | 6.402 | 9.435  | ~12,500  | severe |
| bpe-512     | 6.417 | 10.396 | ~32,800  | severe |
| word        | 6.733 | 12.373 | ~236,000 | pathological |

`word`'s final perplexity (~236k) is **hundreds of times larger than its 737-word
vocabulary** — the model became wildly overconfident about wrong answers, the
signature of pure memorization. Dropout (0.2) and weight decay (0.1) slowed this
down, and **early stopping is what makes the comparison fair**, but no amount of
regularization rescues a vocabulary that is too large for the data.

## 6. What the generated text shows

- **char** — the most coherent: on-topic, mostly real words, shaky grammar, the
  occasional misspelling ("the stock market… stock prices… investors with… each
  exchange"). It learned spelling and local grammar from just 56 symbols.
- **word** — real words with **no grammar at all**, a "bag of words"
  ("borrower everything… ETF… volatility… mutual spread… snowball"). Consistent
  with section 3: it never learned structure, so it just samples frequent words.
  It can never misspell, but it also can never generalize.
- **bpe-128** — in between: real fragments ("the stock market", "to buy",
  "stocks", "sells") stitched together with broken subwords ("wasket", "lowlumbly").
- **bpe-512 / unigram-512** — mostly broken-subword soup
  ("futureces", "bndexseers", "companiestu") — the visual form of "did not learn".

This is the classic **char-vs-word tradeoff**: character models never run out of
vocabulary and learn morphology from few symbols, but must carry meaning across
long token sequences; word models guarantee valid words but blow up the
vocabulary and generalize poorly on small data.

## 7. BPE vs Unigram at the same size

At vocab 512 the two algorithms behave almost identically here (BPC 2.818 vs
2.931, both essentially at the random baseline). On a tiny corpus the **vocabulary
size dominates**; the merge algorithm barely matters. (On billion-token corpora
both are excellent — the difference simply isn't visible at this scale.)

## 8. Conclusion & recommendations

**Generalization ranking:** `char` > `bpe-128` ≫ `bpe-512` ≈ `unigram-512` ≈ `word`.

For a ~2000-word corpus, **smaller vocabulary / finer tokenization wins**.
Character-level generalized best; small BPE was a solid second; everything with a
512+ vocabulary was data-starved and only memorized. The general principle —
which is why production GPTs use 32k–100k-token vocabularies *and* train on
billions of tokens — is that **vocabulary size must scale with data size**.

To push this project further, in order of impact:

1. **Grow the corpus** — by far the biggest lever; 10–50× more text would let the
   512-vocab tokenizers actually learn.
2. **Keep the vocabulary small** while the corpus is small (char or BPE ≤ 256).
3. **Keep early stopping + regularization** — essential at this data scale.
4. Optionally **shrink the model** (fewer layers / smaller embedding) to reduce
   the capacity available for memorization.
