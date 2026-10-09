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
    use_rope:    bool  = False  # True = Rotary Position Embedding (RoPE) invece di PE sinusoidale


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Lookup table: token ID (intero) → vettore d_model
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Con RoPE la posizione è codificata direttamente in Q e K dentro l'attention:
        # la PositionalEncoding sinusoidale non serve e non viene istanziata.
        if not config.use_rope:
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
                dropout=config.dropout,
                use_rope=config.use_rope,
                max_seq_len=config.max_seq_len,
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

    def forward(self, idx, targets=None, past_key_values=None, return_cache=False):
        """
        Args:
            idx:              (batch, seq_len) — token IDs, interi
            targets:          (batch, seq_len) — token IDs target per il training
            past_key_values:  lista di tuple (K_cache, V_cache) per ogni layer,
                              oppure None. Usato da generate() con KV-cache attiva.
            return_cache:     se True, restituisce anche present_key_values.
                              Usato da generate(). Il training chiama sempre con
                              return_cache=False (default) — comportamento invariato.

        Returns:
            (logits, loss)                          se return_cache=False (default)
            (logits, loss, present_key_values)      se return_cache=True
        """
        # Token IDs → vettori densi
        x = self.token_embedding(idx)    # (batch, seq_len, d_model)

        # Con RoPE l'informazione posizionale è gestita dentro l'attention su Q e K.
        if not self.config.use_rope:
            x = self.pos_encoding(x)     # (batch, seq_len, d_model)

        # Passa attraverso ogni blocco Transformer, raccogliendo lo stato KV.
        present_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, _, present_kv = block(x, past_kv=past_kv)
            present_key_values.append(present_kv)

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

        if return_cache:
            return logits, loss, present_key_values
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=1.0, repetition_penalty=1.0, use_cache=True):
        """
        Generazione autoregressiva: produce un token alla volta,
        aggiungendolo al contesto e ripetendo.

        Args:
            idx:                (1, seq_len) — contesto iniziale (token IDs)
            max_new_tokens:     quanti token generare
            temperature:        >1 = distribuzione più piatta (più sorprendente)
                                <1 = distribuzione più concentrata (più prevedibile)
            top_k:              campiona solo dai top-k token più probabili (None = disabilitato)
            top_p:              nucleus sampling — campiona dal minimo sottoinsieme di token
                                la cui probabilità cumulata supera p (1.0 = disabilitato)
            repetition_penalty: >1.0 penalizza i token già presenti nel contesto (1.0 = disabilitato)
            use_cache:          se True (default), usa KV-cache per inference O(n).
                                se False, ricalcola l'intera sequenza ad ogni passo (O(n²)).
        """
        self.eval()
        past_kv = None

        for _ in range(max_new_tokens):
            if use_cache:
                if past_kv is None:
                    # Primo passo: elabora l'intero prompt, ottieni cache iniziale.
                    ctx = idx[:, -self.config.max_seq_len:]
                    logits, _, past_kv = self(ctx, return_cache=True)
                else:
                    # Passi successivi: solo l'ultimo token generato.
                    # Il contesto completo è già nella cache.
                    logits, _, past_kv = self(idx[:, -1:],
                                              past_key_values=past_kv,
                                              return_cache=True)
            else:
                ctx = idx[:, -self.config.max_seq_len:]
                logits, _ = self(ctx)

            # Considera solo il logit dell'ultimo token (il "prossimo")
            logits = logits[:, -1, :]    # (batch, vocab_size)

            # Repetition penalty: penalizza i token già presenti nel contesto.
            # Logit positivi vengono divisi (abbassati), negativi moltiplicati (abbassati).
            # L'effetto è che i token già visti diventano meno probabili.
            if repetition_penalty != 1.0:
                for token_id in set(ctx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Azzera tutto ciò che è sotto il k-esimo valore
                logits[logits < topk_vals[:, [-1]]] = float('-inf')

            # Nucleus sampling (top-p): mantieni il minimo insieme di token la cui
            # probabilità cumulata supera p. Evita la coda lunga di token improbabili.
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Rimuove i token oltre la soglia p (shift di uno: il token che
                # la supera per primo è ancora incluso nel campionamento)
                remove_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) > top_p
                sorted_logits[remove_mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

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
