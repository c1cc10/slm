"""
SLM — matrix_exercises.py

Dot product, moltiplicazione matriciale e split-heads
con numeri piccoli e output leggibili passo per passo.
"""

import torch
import math
torch.manual_seed(0)

print("=" * 60)
print("PARTE 1 — Dot product")
print("=" * 60)

a = torch.tensor([2.0, 1.0, 3.0])
b = torch.tensor([0.0, 4.0, 1.0])

print(f"""
a = {a.tolist()}
b = {b.tolist()}

a · b = (2×0) + (1×4) + (3×1) = {int(2*0)} + {int(1*4)} + {int(3*1)} = {int(torch.dot(a,b).item())}

PyTorch: torch.dot(a, b) = {torch.dot(a, b).item()}

Significato: misura quanto i due vettori puntano nella stessa direzione.
""")

input("[ INVIO per continuare ]")

print("\n" + "=" * 60)
print("PARTE 2 — Moltiplicazione tra matrici")
print("=" * 60)

A = torch.tensor([[1.0, 2.0],
                  [3.0, 1.0]])
B = torch.tensor([[0.0, 1.0],
                  [2.0, 0.0]])
C = torch.matmul(A, B)

print(f"""
A = [1  2]    B = [0  1]
    [3  1]        [2  0]

Regola: (m×n) × (n×p) = (m×p)
  A: ({A.shape[0]}×{A.shape[1]}) × B: ({B.shape[0]}×{B.shape[1]}) = C: ({C.shape[0]}×{C.shape[1]})

C[i,j] = riga i di A · colonna j di B

  C[0,0] = [1,2] · [0,2] = 1×0 + 2×2 = {int(1*0+2*2)}
  C[0,1] = [1,2] · [1,0] = 1×1 + 2×0 = {int(1*1+2*0)}
  C[1,0] = [3,1] · [0,2] = 3×0 + 1×2 = {int(3*0+1*2)}
  C[1,1] = [3,1] · [1,0] = 3×1 + 1×0 = {int(3*1+1*0)}

C = [{int(C[0,0].item())}  {int(C[0,1].item())}]
    [{int(C[1,0].item())}  {int(C[1,1].item())}]

PyTorch:
{C.numpy()}
""")

input("[ INVIO per continuare ]")

print("\n" + "=" * 60)
print("PARTE 3 — Trasposta e Q·Kᵀ")
print("=" * 60)

Q = torch.tensor([[1.0, 0.5, 0.2],
                  [0.3, 0.8, 0.9]])
K = torch.tensor([[1.1, 0.4, 0.1],
                  [0.2, 0.7, 1.0]])

print(f"""
2 token, d_k=3.

Q shape {list(Q.shape)}:   K shape {list(K.shape)}:
  token0: {Q[0].tolist()}    token0: {K[0].tolist()}
  token1: {Q[1].tolist()}   token1: {K[1].tolist()}
""")

try:
    torch.matmul(Q, K)
    print("Q @ K → ok (non dovrebbe succedere)")
except RuntimeError as e:
    print(f"Q @ K → ERRORE: dimensioni interne {Q.shape[1]} ≠ {K.shape[0]}")

K_T = K.transpose(0, 1)
print(f"""
Kᵀ = K.transpose(0,1) → shape {list(K_T.shape)}
  (le colonne diventano righe e viceversa)

  prima:  K[0] = {K[0].tolist()}  ← riga 0
  dopo: K_T colonna 0 = {K_T[:, 0].tolist()}  ← stessi valori, ora colonna
""")

scores = torch.matmul(Q, K_T)
print(f"scores = Q @ Kᵀ → shape {list(scores.shape)}:")
print(f"  {scores.numpy()}")
print(f"""
  scores[0,0] = Q[token0] · K[token0] = {scores[0,0]:.3f}
    (quanto il query di token0 è compatibile con la key di token0)
  scores[0,1] = Q[token0] · K[token1] = {scores[0,1]:.3f}
    (quanto il query di token0 è compatibile con la key di token1)
  scores[1,0] = Q[token1] · K[token0] = {scores[1,0]:.3f}
  scores[1,1] = Q[token1] · K[token1] = {scores[1,1]:.3f}

→ Questa è la matrice che, dopo softmax, diventa i pesi di attenzione.
""")

input("[ INVIO per continuare ]")

print("\n" + "=" * 60)
print("PARTE 4 — Split in teste, passo per passo")
print("=" * 60)

# 2 token, d_model=4, 2 teste, d_k=2
x = torch.tensor([[0.8, 0.2, 0.5, 0.3],
                  [0.1, 0.9, 0.4, 0.7]])

print(f"""
x — shape {list(x.shape)} — 2 token, d_model=4:
  gatto:  {x[0].tolist()}
  mangiò: {x[1].tolist()}

I 4 valori di ogni token verranno divisi in 2 gruppi da 2:
  gatto:  [0.8, 0.2 | 0.5, 0.3]
           testa 0    testa 1
  mangiò: [0.1, 0.9 | 0.4, 0.7]
           testa 0    testa 1
""")

# Step 1: view — riorganizza senza spostare dati
num_heads, d_k = 2, 2
x_view = x.view(2, num_heads, d_k)

print(f"Step 1: x.view(2, 2, 2) → shape {list(x_view.shape)}")
print(f"  NESSUN dato spostato — solo modo diverso di leggerli")
print(f"  x_view[0] (gatto):  {x_view[0].tolist()}")
print(f"    [0] = testa 0: {x_view[0,0].tolist()}")
print(f"    [1] = testa 1: {x_view[0,1].tolist()}")
print(f"  x_view[1] (mangiò): {x_view[1].tolist()}")

# Step 2: transpose — porta heads in prima posizione
x_heads = x_view.transpose(0, 1)

print(f"""
Step 2: .transpose(0,1) → shape {list(x_heads.shape)}  (heads, seq, d_k)
  x_heads[0] = testa 0, tutti i token:
    {x_heads[0].tolist()}
    gatto visto dalla testa 0:  {x_heads[0,0].tolist()}
    mangiò visto dalla testa 0: {x_heads[0,1].tolist()}

  x_heads[1] = testa 1, tutti i token:
    {x_heads[1].tolist()}
    gatto visto dalla testa 1:  {x_heads[1,0].tolist()}
    mangiò visto dalla testa 1: {x_heads[1,1].tolist()}

Ora ogni x_heads[i] è una matrice (seq=2, d_k=2) — pronta per Q·Kᵀ.
""")

input("[ INVIO per continuare ]")

print("\n" + "=" * 60)
print("PARTE 5 — Q·Kᵀ per ogni testa")
print("=" * 60)

print("Usiamo x_heads come Q e K (self-attention con W=identità):\n")

for h in range(num_heads):
    Q_h = x_heads[h]           # (seq=2, d_k=2)
    K_h = x_heads[h]           # stessa matrice per semplicità
    K_h_T = K_h.transpose(0,1) # (d_k=2, seq=2)

    raw = Q_h @ K_h_T          # (2, 2)
    scaled = raw / math.sqrt(d_k)
    weights = torch.softmax(scaled, dim=-1)

    print(f"  Testa {h}:")
    print(f"    Q shape {list(Q_h.shape)}: {Q_h.tolist()}")
    print(f"    Kᵀ shape {list(K_h_T.shape)}")
    print(f"    Q·Kᵀ (scores):")
    print(f"      {raw.tolist()}")
    print(f"    / √{d_k} = {math.sqrt(d_k):.3f}:")
    print(f"      {scaled.tolist()}")
    print(f"    softmax (pesi finali):")
    print(f"      gatto  guarda → gatto:{weights[0,0]:.3f}  mangiò:{weights[0,1]:.3f}")
    print(f"      mangiò guarda → gatto:{weights[1,0]:.3f}  mangiò:{weights[1,1]:.3f}")
    print()

print("""Con W_q e W_k identici i risultati sono uguali tra le teste.
Nel modello reale ogni testa ha le sue W_q, W_k con pesi diversi
→ vede sottoaspetti diversi degli stessi token
→ specializzazione emergente durante il training.
""")

print("=" * 60)
print("Esercizio finale: prova a cambiare i valori di x")
print("e osserva come cambiano i pesi di attenzione.")
print("=" * 60)
