"""
SLM — transformer.py

Assembla il blocco Transformer completo:
  1. PositionalEncoding — inietta informazione sulla posizione di ogni token
  2. FeedForward        — trasformazione non-lineare applicata token per token
  3. TransformerBlock   — un layer completo (attention + FF + norme + residui)

Architettura: decoder-only con pre-norm (stile GPT-2/LLaMA).
"""

import torch
import torch.nn as nn
import math
from attention import MultiHeadAttention, create_causal_mask


class PositionalEncoding(nn.Module):
    """
    Codifica sinusoidale della posizione.

    Aggiunge al vettore di ogni token un segnale che dipende dalla sua
    posizione nella sequenza, permettendo al modello di distinguere
    "gatto mangia topo" da "topo mangia gatto".

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Le frequenze decrescenti creano un "codice binario continuo":
    dimensioni basse cambiano lentamente (posizioni lontane),
    dimensioni alte cambiano rapidamente (posizioni vicine).
    """

    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Costruiamo la tabella PE una volta sola e la registriamo come buffer
        # (non è un parametro — non viene aggiornata durante il training)
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        # Divisore calcolato in log-space per stabilità numerica
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # dimensioni pari
        pe[:, 1::2] = torch.cos(position * div_term)   # dimensioni dispari

        # (max_seq_len, d_model) → (1, max_seq_len, d_model) per il broadcast sul batch
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Sommiamo la codifica posizionale all'embedding di ogni token
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Rete fully-connected applicata a ogni token separatamente.

    Formula: FFN(x) = GELU(x W1 + b1) W2 + b2

    d_ff = 4 * d_model per convenzione (proiezione in spazio più ampio
    poi riportata alla dimensione originale). Permette al modello di
    memorizzare "fatti" e trasformare le rappresentazioni raccolte dall'attention.

    GELU (Gaussian Error Linear Unit) è un'attivazione più smooth di ReLU,
    preferita nei transformer moderni.
    """

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Un singolo layer del decoder transformer (architettura pre-norm).

    Flusso:
        x → LN → MultiHeadAttention → residual → x
        x → LN → FeedForward        → residual → x

    Pre-norm (normalizza PRIMA del sublayer) è più stabile durante
    il training rispetto al post-norm dell'articolo originale.

    La causal mask viene creata internamente: ogni istanza del blocco
    garantisce da sola che il token i non veda i token j > i.
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff        = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        mask = create_causal_mask(seq_len, device=x.device)

        # --- Self-attention con pre-norm e residual ---
        # x + Attention(LN(x))
        attn_out, weights = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask=mask
        )
        x = x + self.dropout(attn_out)

        # --- Feed-forward con pre-norm e residual ---
        # x + FFN(LN(x))
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x, weights


# ---------------------------------------------------------------------------
# Test: verifica shapes e osserva come la positional encoding varia per posizione
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch     = 2
    seq_len   = 8
    d_model   = 64
    num_heads = 8
    d_ff      = 256   # 4 * d_model

    # --- Positional Encoding ---
    pe = PositionalEncoding(d_model=d_model, max_seq_len=128)
    x  = torch.zeros(batch, seq_len, d_model)   # embedding tutti zero
    x_pe = pe(x)

    print("=== Positional Encoding ===")
    print(f"Input (tutti zero):  {x.shape}")
    print(f"Output (con PE):     {x_pe.shape}")
    print("Primi 8 valori del token 0:", x_pe[0, 0, :8].detach().round(decimals=3).tolist())
    print("Primi 8 valori del token 1:", x_pe[0, 1, :8].detach().round(decimals=3).tolist())
    print("→ token diversi hanno valori diversi anche con embedding identico")

    # --- FeedForward ---
    ff = FeedForward(d_model=d_model)
    out_ff = ff(x_pe)

    print("\n=== FeedForward ===")
    print(f"Input:  {x_pe.shape}")
    print(f"Output: {out_ff.shape}")   # shape identica: trasformazione token-wise

    # --- TransformerBlock ---
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    x_rand = torch.randn(batch, seq_len, d_model)
    out_block, attn_weights = block(x_rand)

    print("\n=== TransformerBlock ===")
    print(f"Input:          {x_rand.shape}")
    print(f"Output:         {out_block.shape}")
    print(f"Attn weights:   {attn_weights.shape}   (batch, heads, seq, seq)")

    # Verifica residual: l'output non è uguale all'input (la rete trasforma)
    diff = (out_block - x_rand).abs().mean().item()
    print(f"Distanza media input/output: {diff:.4f}   (> 0 = la rete trasforma)")

    # Conta i parametri del blocco
    params = sum(p.numel() for p in block.parameters())
    print(f"\nParametri nel blocco: {params:,}")
    print(f"  d_model={d_model}, heads={num_heads}, d_ff={d_ff}")

    # --- Stack di N blocchi ---
    print("\n=== Stack di 4 blocchi ===")
    blocks = nn.ModuleList([
        TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        for _ in range(4)
    ])
    x_stack = x_rand.clone()
    for i, b in enumerate(blocks):
        x_stack, _ = b(x_stack)
        print(f"  dopo blocco {i+1}: shape {x_stack.shape}, media={x_stack.mean().item():.4f}")

    total_params = sum(p.numel() for b in blocks for p in b.parameters())
    print(f"\nParametri totali (4 blocchi): {total_params:,}")
