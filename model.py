import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block


class TinyGPT(nn.Module):
    """A small decoder-only transformer (GPT-style) language model.

    All sizes are passed in so the same class can be reused across
    experiments with different vocabularies and tokenizers.
    """

    def __init__(self, vocab_size, block_size, embedding_dim=64,
                 n_heads=4, n_layers=4, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample new tokens.

        temperature < 1.0 makes the output more confident/repetitive;
        top_k keeps only the k most likely tokens at each step.
        Call model.eval() first so dropout is disabled.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # focus on the last step
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx
