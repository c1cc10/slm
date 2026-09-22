"""
SLM — attention.py

Implementazione dell'attention mechanism da zero.

Concetti chiave:
  - Scaled Dot-Product Attention: ogni token "interroga" gli altri
  - Multi-Head Attention: N teste parallele, ognuna impara relazioni diverse
  - Causal mask: un token può vedere solo i token precedenti (fondamentale per generazione)
"""

import torch
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Cuore dell'attention mechanism.

    Args:
        Q: Query  (batch, heads, seq_len, d_k) — "cosa sto cercando?"
        K: Key    (batch, heads, seq_len, d_k) — "cosa offro?"
        V: Value  (batch, heads, seq_len, d_k) — "qual è il mio contenuto?"
        mask: (seq_len, seq_len) — 1 = posizione visibile, 0 = nascosta

    Returns:
        output:  (batch, heads, seq_len, d_k)
        weights: (batch, heads, seq_len, seq_len) — utili per visualizzare cosa guarda il modello
    """
    d_k = Q.size(-1)

    # Prodotto scalare Q·Kᵀ: misura la "compatibilità" tra ogni coppia di token
    # Shape: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # La causal mask impedisce ai token di guardare il futuro.
    # Sostituiamo le posizioni vietate con -inf: dopo softmax diventano 0.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax converte gli score in pesi che sommano a 1
    weights = torch.softmax(scores, dim=-1)

    # Media pesata dei Values: la rappresentazione finale arricchita dal contesto
    output = torch.matmul(weights, V)

    return output, weights


def create_causal_mask(seq_len, device):
    """
    Matrice triangolare inferiore: il token i può vedere solo le posizioni 0..i.

    Esempio per seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Senza questa mask il modello "bara" durante il training,
    guardando le risposte future per predire il token corrente.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class MultiHeadAttention(nn.Module):
    """
    Esegue l'attention in parallelo su N "teste".

    Perché più teste? Ogni testa può specializzarsi su un tipo diverso
    di relazione: sintassi, riferimenti, prossimità, ecc.
    I risultati vengono poi ricombinati.
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model:    dimensione del vettore di rappresentazione di ogni token
            num_heads:  numero di teste parallele (deve dividere d_model)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve essere divisibile per num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        # Proiezioni lineari: trasformano l'input in Q, K, V per ogni testa
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Proiezione finale: ricombina l'output di tutte le teste
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x, batch_size):
        """
        Divide d_model in num_heads teste da d_k dimensioni ciascuna.
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x, batch_size):
        """
        Operazione inversa: ricombina le teste.
        (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        In self-attention (il caso più comune) Q=K=V=x:
        ogni token si interroga su tutti gli altri token della stessa sequenza.
        """
        batch_size = Q.size(0)

        # Proietta e dividi in teste
        Q = self._split_heads(self.W_q(Q), batch_size)
        K = self._split_heads(self.W_k(K), batch_size)
        V = self._split_heads(self.W_v(V), batch_size)

        # Attention su ogni testa in parallelo
        x, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Ricombina le teste e proietta
        x = self._merge_heads(x, batch_size)
        output = self.W_o(x)

        return output, weights


# ---------------------------------------------------------------------------
# Test: verifica che le shape siano corrette e osserva i pesi di attenzione
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch     = 2
    seq_len   = 6
    d_model   = 64
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(batch, seq_len, d_model)

    # Senza mask: ogni token vede tutti gli altri
    out_no_mask, w_no_mask = mha(x, x, x)

    # Con causal mask: ogni token vede solo sé stesso e i precedenti
    mask = create_causal_mask(seq_len, device=x.device)
    out_masked, w_masked = mha(x, x, x, mask=mask)

    print("=== Shape check ===")
    print(f"Input:   {x.shape}")
    print(f"Output:  {out_masked.shape}")    # uguale all'input
    print(f"Weights: {w_masked.shape}")      # (batch, heads, seq_len, seq_len)

    print("\n=== Causal mask (6x6) ===")
    print(mask)

    print("\n=== Pesi attenzione — batch 0, testa 0 (con causal mask) ===")
    print(w_masked[0, 0].detach().round(decimals=2))
    print("Ogni riga somma a 1:", w_masked[0, 0].sum(dim=-1).detach().round(decimals=3))

    print("\n=== Confronto: token 0 senza mask vs con mask ===")
    print("Senza mask:", w_no_mask[0, 0, 0].detach().round(decimals=3))
    print("Con mask:  ", w_masked[0, 0, 0].detach().round(decimals=3))
    print("(Con mask il token 0 vede solo sé stesso → peso 1.0 su posizione 0)")
