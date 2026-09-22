"""
SLM — model.py

Il modello GPT completo: token embedding + N TransformerBlock + proiezione vocabolario.

Flusso forward:
  token IDs (interi)
    → Embedding          — ogni ID diventa un vettore d_model
    → PositionalEncoding — somma informazione sulla posizione
    → TransformerBlock × N
    → LayerNorm finale
    → Linear (lm_head)   — proietta d_model → vocab_size (logits)

Durante il training:    logits + targets → cross-entropy loss
Durante la generazione: logits → campionamento → token successivo → ripeti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from transformer import TransformerBlock, PositionalEncoding


@dataclass
class GPTConfig:
    """Iperparametri del modello. Cambia qui per scalare su o giù."""
    vocab_size:  int   = 256    # char-level: 256 caratteri Unicode base
    d_model:     int   = 256    # dimensione del vettore per ogni token
    num_heads:   int   = 8      # teste di attenzione (divide d_model)
    num_layers:  int   = 6      # blocchi Transformer impilati
    d_ff:        int   = 1024   # dimensione FFN interna (tipicamente 4 × d_model)
    max_seq_len: int   = 256    # lunghezza massima della sequenza di input
    dropout:     float = 0.1    # dropout per regolarizzazione


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Lookup table: token ID (intero) → vettore d_model
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )

        # N blocchi Transformer impilati
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # LayerNorm finale — stabilizza prima della proiezione
        self.norm_final = nn.LayerNorm(config.d_model)

        # Proietta ogni vettore d_model → vocab_size logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding e lm_head condividono gli stessi pesi.
        # Il token "gatto" come input e come output previsto ha la stessa
        # rappresentazione. Riduce i parametri e migliora la generalizzazione.
        self.lm_head.weight = self.token_embedding.weight

        # Inizializzazione stile GPT-2: pesi gaussiani con std piccola
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (batch, seq_len) — token IDs, interi
            targets: (batch, seq_len) — token IDs target per il training
                     targets[b, t] = token corretto dopo idx[b, t]

        Returns:
            logits: (batch, seq_len, vocab_size) — score non normalizzati
            loss:   scalare cross-entropy (None se targets non forniti)
        """
        # Token IDs → vettori densi
        x = self.token_embedding(idx)    # (batch, seq_len, d_model)

        # Aggiunge informazione posizionale
        x = self.pos_encoding(x)         # (batch, seq_len, d_model)

        # Passa attraverso ogni blocco Transformer
        for block in self.blocks:
            x, _ = block(x)              # (batch, seq_len, d_model)

        x = self.norm_final(x)           # (batch, seq_len, d_model)

        logits = self.lm_head(x)         # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy su tutti i token della sequenza.
            # Flatten necessario: F.cross_entropy vuole (N, C) e (N,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  # (batch×seq, vocab)
                targets.view(-1)                           # (batch×seq,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generazione autoregressiva: produce un token alla volta,
        aggiungendolo al contesto e ripetendo.

        Args:
            idx:            (1, seq_len) — contesto iniziale (token IDs)
            max_new_tokens: quanti token generare
            temperature:    >1 = distribuzione più piatta (più sorprendente)
                            <1 = distribuzione più concentrata (più prevedibile)
            top_k:          campiona solo dai top-k token più probabili
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Tronca il contesto alla lunghezza massima del modello
            ctx = idx[:, -self.config.max_seq_len:]

            logits, _ = self(ctx)

            # Considera solo il logit dell'ultimo token (il "prossimo")
            logits = logits[:, -1, :] / temperature    # (batch, vocab_size)

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Azzera tutto ciò che è sotto il k-esimo valore
                logits[logits < topk_vals[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_params(self):
        """Conta parametri totali e trainable (il peso condiviso è contato una volta)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    cfg = GPTConfig()
    model = GPT(cfg)

    n_params = model.count_params()

    print("=== SLM — configurazione ===")
    print(f"  vocab_size  : {cfg.vocab_size}  (char-level, 256 caratteri)")
    print(f"  d_model     : {cfg.d_model}")
    print(f"  num_heads   : {cfg.num_heads}   (d_k = {cfg.d_model // cfg.num_heads} per testa)")
    print(f"  num_layers  : {cfg.num_layers}")
    print(f"  d_ff        : {cfg.d_ff}")
    print(f"  max_seq_len : {cfg.max_seq_len}")
    print(f"\n  Parametri totali      : {n_params:,}")
    print(f"  Memoria pesi (fp32)   : ~{n_params * 4 / 1024**2:.1f} MB")
    print(f"  Memoria pesi (fp16)   : ~{n_params * 2 / 1024**2:.1f} MB")

    print("\n=== Forward pass ===")
    batch, seq = 4, 64
    idx     = torch.randint(0, cfg.vocab_size, (batch, seq))
    targets = torch.randint(0, cfg.vocab_size, (batch, seq))

    logits, loss = model(idx, targets)

    expected_loss = torch.log(torch.tensor(float(cfg.vocab_size))).item()
    print(f"  Input    : {list(idx.shape)}  (batch={batch}, seq={seq})")
    print(f"  Logits   : {list(logits.shape)}")
    print(f"  Loss     : {loss.item():.4f}")
    print(f"  Attesa   : {expected_loss:.4f}  (= log({cfg.vocab_size}) con pesi casuali)")
    print(f"  Diff     : {abs(loss.item() - expected_loss):.4f}  (piccola = inizializzazione ok)")

    print("\n=== Generazione (modello non addestrato) ===")
    prompt = torch.tensor([[ord('I')]])     # carattere 'I' come contesto iniziale
    out = model.generate(prompt, max_new_tokens=30, temperature=1.0, top_k=20)
    chars = ''.join(chr(t) if 32 <= t < 127 else '?' for t in out[0].tolist())
    print(f"  Prompt  : 'I'")
    print(f"  Output  : '{chars}'")
    print(f"  (rumore — il modello non è ancora addestrato)")

    print("\n=== Device ===")
    if torch.backends.mps.is_available():
        print("  Apple MPS disponibile ✓  — usa model.to('mps') per accelerare su M2")
    if torch.cuda.is_available():
        print("  CUDA disponibile ✓")
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("  CPU only")

    print("\n=== Prossimo step: train.py ===")
    print("  Tokenizer char-level + dataset italiano + training loop AdamW")
