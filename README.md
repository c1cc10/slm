# SLM — Small Language Model

Modello linguistico italiano decoder-only costruito da zero in Python/PyTorch.
Progetto di studio con applicazioni professionali.

## Documentazione

Apri `docs/index.html` nel browser per accedere alla documentazione completa,
al riferimento teorico e alla visualizzazione interattiva dell'attenzione.

Con GitLab Pages attivo, la documentazione sarà disponibile all'URL del progetto.

## Struttura

```
slm/
├── attention.py          # Scaled dot-product attention, Multi-Head Attention
├── transformer.py        # PositionalEncoding, FeedForward, TransformerBlock
├── model.py              # GPTConfig, GPT (forward + generate + weight tying)
├── train.py              # Tokenizer char-level, dataset, loop AdamW
├── data/
│   └── corpus.txt        # Corpus di training (da fornire)
├── checkpoints/
│   └── best.pt           # Checkpoint migliore (generato da train.py)
└── docs/
    ├── index.html         # Indice della documentazione
    ├── slm-docs.html      # Documentazione completa
    ├── slm-teoria.html    # Riferimento teorico con formule
    └── slm-attention-viz.html  # Visualizzazione interattiva attenzione
```

## Requisiti

```bash
pip install torch
```

## Utilizzo

```bash
# Download corpus di prova
python3 train.py --download

# Training (usa automaticamente Apple MPS su M2)
python3 train.py

# Training rapido per test
python3 train.py --steps 2000 --batch 16

# Generazione testo dal checkpoint
python3 train.py --generate "Il progetto"
```

## Architettura

Decoder-only Transformer (stile GPT, pre-norm):

- `vocab_size`: determinato dal corpus (~80–120 caratteri unici)
- `d_model`: 256
- `num_heads`: 8 (d_k = 32 per testa)
- `num_layers`: 6
- `d_ff`: 1024 (4 × d_model)
- `max_seq_len`: 256
- Parametri totali: ~4.8M (~18 MB fp32)

## Roadmap

| Fase | Obiettivo | Stato |
|------|-----------|-------|
| 1 — From scratch | Char-level, training da zero | in corso |
| 2 — BPE | Tokenizer sub-word su corpus di dominio | — |
| 3 — Fine-tuning | Continued pre-training su modello italiano esistente | — |
| 4 — Multimodale | Integrazione pipeline YOLO v3 | — |
