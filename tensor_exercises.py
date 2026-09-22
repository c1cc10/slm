"""
SLM — tensor_exercises.py

Esercizi pratici sui tensori, collegati direttamente al codice
che abbiamo scritto in attention.py e transformer.py.

Esegui il file intero e leggi ogni output prima di continuare.
Dopo ogni blocco trovi una domanda: pensaci prima di scorrere.
"""

import torch

print("=" * 60)
print("BLOCCO 1 — Cos'è un tensore")
print("=" * 60)

# Scalare: 0 dimensioni — un singolo numero
scalare = torch.tensor(3.14)
print(f"\nScalare:       {scalare}")
print(f"  .shape:      {scalare.shape}")    # torch.Size([]) — vuoto, 0D
print(f"  .ndim:       {scalare.ndim}")     # 0

# Vettore: 1 dimensione — lista di numeri
# Nella nostra rete: l'embedding di UN singolo token
vettore = torch.tensor([0.1, 0.5, -0.3, 0.8])
print(f"\nVettore:       {vettore}")
print(f"  .shape:      {vettore.shape}")    # [4] — 4 elementi
print(f"  .ndim:       {vettore.ndim}")     # 1

# Matrice: 2 dimensioni — griglia di numeri
# Nella nostra rete: UNA sequenza di token (senza batch)
# 6 token, ciascuno con embedding di 4 valori
matrice = torch.randn(6, 4)
print(f"\nMatrice (6 token × 4 dimensioni):")
print(f"  .shape:      {matrice.shape}")    # [6, 4]
print(f"  .ndim:       {matrice.ndim}")     # 2
print(f"  Primo token: {matrice[0]}")       # vettore 1D da 4 elementi

# ❓ DOMANDA 1:
# matrice[2] — cosa ti aspetti? Un numero, un vettore, o una matrice?
# matrice[2, 1] — e questo?
print(f"\n  matrice[2]:    {matrice[2]}")
print(f"  matrice[2,1]:  {matrice[2, 1]:.4f}")

input("\n[ Premi INVIO per continuare ]")

print("\n" + "=" * 60)
print("BLOCCO 2 — Il batch: perché 3D")
print("=" * 60)

# Nella realtà non elaboriamo una sequenza alla volta,
# ma un BATCH (gruppo) di sequenze in parallelo.
# Questo è esattamente il tensore che entra in MultiHeadAttention.

batch_size = 2    # due frasi in parallelo
seq_len    = 6    # ogni frase ha 6 token
d_model    = 8    # ogni token ha embedding di 8 valori

x = torch.randn(batch_size, seq_len, d_model)

print(f"\nTensore x (batch di sequenze):")
print(f"  .shape:  {x.shape}    # (batch, seq_len, d_model)")
print(f"  .ndim:   {x.ndim}")

print(f"\nCome leggo le dimensioni:")
print(f"  x[0]         → prima  sequenza: shape {x[0].shape}  # (seq_len, d_model)")
print(f"  x[1]         → seconda sequenza: shape {x[1].shape}")
print(f"  x[0, 3]      → token 3 della prima seq: shape {x[0, 3].shape}  # (d_model,)")
print(f"  x[0, 3, 2]   → valore scalare: {x[0, 3, 2]:.4f}")

# ❓ DOMANDA 2:
# x.shape è (2, 6, 8). Quanto vale x[1, 0, 5]?
# È un tensore, un vettore, o un numero?
print(f"\n  x[1, 0, 5] = {x[1, 0, 5]:.4f}  ← scalare: batch 1, token 0, dim 5")

input("\n[ Premi INVIO per continuare ]")

print("\n" + "=" * 60)
print("BLOCCO 3 — transpose e matmul: operazioni sulle shape")
print("=" * 60)

# In attention.py abbiamo scritto:
#   scores = torch.matmul(Q, K.transpose(-2, -1))
# Capiamo cosa fa riga per riga.

Q = torch.randn(2, 8, 6, 4)   # (batch, heads, seq_len, d_k)
K = torch.randn(2, 8, 6, 4)   # stessa shape

print(f"\nQ.shape: {Q.shape}   # (batch=2, heads=8, seq=6, d_k=4)")
print(f"K.shape: {K.shape}")

# transpose(-2, -1) scambia le ULTIME DUE dimensioni
K_T = K.transpose(-2, -1)
print(f"\nK.transpose(-2, -1).shape: {K_T.shape}  # d_k e seq si scambiano")
print(f"  Prima:  (..., seq=6, d_k=4)")
print(f"  Dopo:   (..., d_k=4, seq=6)")

# matmul su tensori 4D: moltiplica le ultime 2 dimensioni come matrici,
# le dimensioni precedenti (batch, heads) restano invariate
scores = torch.matmul(Q, K_T)
print(f"\ntorch.matmul(Q, K_T).shape: {scores.shape}")
print(f"  (batch=2, heads=8, seq=6, seq=6)")
print(f"  → per ogni (batch, head): matrice 6×6")
print(f"    riga i = quanto il token i 'guarda' gli altri 6 token")

# ❓ DOMANDA 3:
# scores[0, 0] è una matrice. Cosa rappresenta fisicamente?
# (batch 0, head 0 — 6 righe × 6 colonne)
print(f"\n  scores[0, 0].shape: {scores[0, 0].shape}")
print(f"  scores[0, 0, 2, 4]: quanto il token 2 'guarda' il token 4")
print(f"  valore (random, non ha senso ora): {scores[0, 0, 2, 4]:.4f}")

input("\n[ Premi INVIO per continuare ]")

print("\n" + "=" * 60)
print("BLOCCO 4 — Le shape nel codice che abbiamo scritto")
print("=" * 60)

# Ripercorriamo il flusso completo di MultiHeadAttention
# con shape commentate passo per passo

import sys
sys.path.insert(0, '/Users/francescorana/Documents/Development/slm')
from attention import MultiHeadAttention, create_causal_mask

torch.manual_seed(42)

batch     = 2
seq_len   = 5
d_model   = 16
num_heads = 4
d_k       = d_model // num_heads   # 4

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

x = torch.randn(batch, seq_len, d_model)
print(f"\nInput x:             {x.shape}   # (batch=2, seq=5, d_model=16)")

# Step 1: proiezione lineare
Q_proj = mha.W_q(x)
print(f"Dopo W_q (lineare):  {Q_proj.shape}   # ancora (2, 5, 16)")

# Step 2: split in teste
# (batch, seq, d_model) → (batch, seq, heads, d_k) → (batch, heads, seq, d_k)
Q_heads = Q_proj.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
print(f"Dopo split in teste: {Q_heads.shape}  # (batch=2, heads=4, seq=5, d_k=4)")

# Step 3: attention scores
K_proj  = mha.W_k(x)
K_heads = K_proj.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
scores  = torch.matmul(Q_heads, K_heads.transpose(-2, -1))
print(f"Scores Q·Kᵀ:         {scores.shape}  # (2, 4, 5, 5) — 5×5 per testa")

# Step 4: causal mask + softmax
import math
mask    = create_causal_mask(seq_len, device=x.device)
scores  = scores.masked_fill(mask == 0, float('-inf'))
weights = torch.softmax(scores / math.sqrt(d_k), dim=-1)
print(f"Pesi (dopo softmax): {weights.shape}  # stessa shape degli scores")

# Step 5: moltiplicazione per V
V_proj  = mha.W_v(x)
V_heads = V_proj.view(batch, seq_len, num_heads, d_k).transpose(1, 2)
out_pre = torch.matmul(weights, V_heads)
print(f"Output pre-merge:    {out_pre.shape}  # (2, 4, 5, 4)")

# Step 6: merge delle teste e proiezione finale
out_merged = out_pre.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
print(f"Output finale:       {out_merged.shape}   # (2, 5, 16) — uguale all'input!")

print("\n→ Il tensore entra con shape (2,5,16) ed esce con shape (2,5,16).")
print("  Internamente è passato per uno spazio 4D (batch,heads,seq,d_k).")
print("  Ogni token ha 'visto' gli altri grazie alla matrice scores 5×5.")

input("\n[ Premi INVIO per continuare ]")

print("\n" + "=" * 60)
print("BLOCCO 5 — Esercizio finale: prevedi le shape")
print("=" * 60)

print("""
Senza eseguire il codice, prova a calcolare la shape risultante.
Poi decommenta e verifica.

  a = torch.randn(3, 7, 12)
  b = torch.randn(3, 12, 5)
  c = torch.matmul(a, b)
  # c.shape = ?

  d = torch.randn(2, 4, 8, 8)
  e = d.transpose(-2, -1)
  # e.shape = ?

  f = torch.randn(2, 6, 64)
  g = f.view(2, 6, 8, 8).transpose(1, 2)
  # g.shape = ?    ← questa è esattamente _split_heads() in MultiHeadAttention
""")

# Risposte
a = torch.randn(3, 7, 12)
b = torch.randn(3, 12, 5)
c = torch.matmul(a, b)
print(f"  c.shape = {c.shape}     # matmul: (3,7,12)×(3,12,5) → (3,7,5)")

d = torch.randn(2, 4, 8, 8)
e = d.transpose(-2, -1)
print(f"  e.shape = {e.shape}   # transpose ultime 2: (2,4,8,8) → (2,4,8,8) — quadrata, uguale!")

f = torch.randn(2, 6, 64)
g = f.view(2, 6, 8, 8).transpose(1, 2)
print(f"  g.shape = {g.shape}   # split_heads: (2,6,64) → (2,8,6,8)")

print("""
  c.shape = (3, 7, 5)
  e.shape = (2, 4, 8, 8)   ← matrice quadrata: transpose uguale!
  g.shape = (2, 8, 6, 8)   ← (batch, heads, seq, d_k)
""")

print("=" * 60)
print("Fine esercizi. Cosa hai trovato difficile da prevedere?")
print("=" * 60)
