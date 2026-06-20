# Tiny GPT - Tokenization Experiments & Analysis

- **Corpus:** `corpus.txt` - 2147 words, 11730 characters (topic: the stock market)
- **Model (identical for every run):** block_size=32, embedding_dim=96, n_heads=3, n_layers=3, dropout=0.2
- **Training:** AdamW, lr=0.003, weight_decay=0.1, batch_size=32, 3000 steps, 90/10 train/val split, early stopping on val loss (eval every 250 steps), device=`cpu`, seed=1337

## Results

| Mode | model_type | Vocab | Tokens | Chars/Tok | Params | Random-init val | Best val (step) | Final val | Learned? | Val BPC |
|---|---|---|---|---|---|---|---|---|---|---|
| char | char | 200->56 | 11710 | 1.00 | 348,728 | 4.151 | 1.702 (@1000) | 1.914 | **yes** | 2.451 |
| word | word | 4000->737 | 2147 | 5.46 | 480,161 | 6.733 | 6.733 (@0) | 12.373 | no | 1.778 |
| bpe-128 | bpe | 128->128 | 6550 | 1.79 | 362,624 | 5.025 | 3.081 (@250) | 5.070 | **yes** | 2.482 |
| bpe-512 | bpe | 512->512 | 3571 | 3.28 | 436,736 | 6.417 | 6.417 (@0) | 10.396 | no | 2.818 |
| unigram-512 | unigram | 512->512 | 3722 | 3.15 | 436,736 | 6.402 | 6.402 (@0) | 9.435 | no | 2.931 |

> **Random-init val** is the validation loss before any training (approximately `ln(vocab)`); it is the score of pure guessing. A tokenizer only *learned* something that generalizes if its **best val** drops clearly below this baseline (and the best step is > 0).
>
> **Val BPC (bits per character)** = `best_val_loss / ln(2) / (chars per token)`. Per-token loss is not comparable across tokenizers (each cuts the text into a different number of tokens), so we normalize per character. **Lower is better -- but only meaningful for rows that actually learned.** For the `no` rows the BPC just reflects the entropy of guessing among long tokens, not real modeling.
>
> **Final val** is the loss after all 3000 steps; the gap above best val shows how hard each mode overfits this tiny corpus.

## Generated samples (prompt: "the stock market", best model, temp=0.8, top_k=40)

**char**

```
the stock market. Onew chosen called stocks prices of a stock that can small market called stocks and that most oney a prices and a bear it company a stock loss grow they lose market. A stock the sell investors with fatter each exchange and the stocks. In
```

**word**

```
the stock market understanding borrower everything. from ahead ETF, in if play current times where tool first movement bear filings. volatility. everything. To mutual spread. strategy, do well financially buy. helps changes secure Some whole Index weak, matter total snowball through investors. fewer be Prices often buys owns.
```

**bpe-128**

```
the stock market, hows. The price, wasket the Shis lowlumbly to buy to expe inds, the k ors stocks fud stocks a stock burow and sells. A dififill market doll The sictumbl
```

**bpe-512**

```
the stock market share trade fall futureces bndexseers quldr ag Nates mueelchangean tellas diversif people anter woree companiestu shar exchangeithim risk investor trysifop ltherth earnsfectgwnvera onproufolio hake An Investors small portcon high portcon becany asrouansorstuker sharumerouterarendrouter bearth hop isskewn volume andss earn Anomeep
```

**unigram-512**

```
the stock market adMblreateson stockolhegh tr keepug fined years weaecur B wellnow hoxchange mat perur famfor bectenBpanies re pu sur marketugsite capis Theles f diversification meable checksur agreer O make expectad differen seller expect patient seller expect look sharesment agree marketad usUten retire agreeuris recover agreeuriplarmost. rises putkerec market risesshize stocks strong
```

---

_See **[ANALYSIS.md](ANALYSIS.md)** for the full written performance analysis._
