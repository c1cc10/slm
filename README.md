# SLM — Small Language Model

Modello linguistico italiano decoder-only costruito da zero in Python/PyTorch.
Ogni componente scritto e compreso prima di passare al successivo: architettura di attenzione, tokenizer BPE-SPM, training loop AdamW, deployment via GGUF/llamafile.

**Documentazione**: [c1cc10.github.io/slm](https://c1cc10.github.io/slm/)

---

## Stato corrente

| | |
|---|---|
| Architettura | Decoder-only Transformer (stile GPT, pre-norm) |
| Parametri | 45.9M (~175 MB fp32) |
| Tokenizer | BPE-SPM, 16 000 token |
| Corpus | Wikipedia IT (~208M caratteri, 47.8M token) |
| Run corrente | #3 — step ~11 750 / 30 000, val loss 5.46 |
| Hardware | Apple M2 8GB, MPS |
| Target deployment | llamafile (Mozilla) via GGUF |

---

## Struttura

```
slm/
├── attention.py          # Scaled dot-product attention, Multi-Head Attention
├── transformer.py        # PositionalEncoding (sinusoidale → RoPE), FeedForward (GELU), TransformerBlock
├── model.py              # GPTConfig, GPT (forward + generate + weight tying)
├── train.py              # Tokenizer (char/tiktoken/BPE-SPM), dataset, AdamW, checkpoint
├── tools/
│   ├── epub_to_text.py   # Estrazione corpus da EPUB
│   └── wiki_to_corpus.py # Estrazione Wikipedia IT da dump XML
├── checkpoints/
│   ├── tokenizer.model   # Modello SentencePiece BPE (16k vocab)
│   └── tokenizer.vocab   # Vocabolario SPM
└── docs/                 # Documentazione HTML (GitHub Pages)
    ├── index.html
    ├── slm-journal.html  # Diario di sviluppo e decisioni architetturali
    ├── slm-intro.html    # Come funziona una LLM
    ├── slm-roadmap.html  # Roadmap 11 fasi
    ├── slm-docs.html     # Documentazione completa
    ├── slm-teoria.html   # Riferimento teorico con formule
    └── slm-ottimizzatore.html  # AdamW e ottimizzazione
```

---

## Requisiti

```bash
pip install torch sentencepiece
```

## Training

```bash
# Pre-training BPE-SPM su Wikipedia IT (preset medium: 45.9M parametri)
python3 train.py --tokenizer bpe-spm --data data/wiki_it.txt \
  --preset medium --steps 30000 --batch 16 --seq-len 256

# Resume da checkpoint
python3 train.py --resume --tokenizer bpe-spm --data data/wiki_it.txt

# Generazione testo da checkpoint
python3 train.py --generate "La capitale d'Italia"

# Training char-level (Fase 1, completata)
python3 train.py --steps 5000
```

## Architettura — preset medium

| Iperparametro | Valore |
|---|---|
| d_model | 512 |
| num_heads | 8 (d_k = 64) |
| num_layers | 8 |
| d_ff | 2 048 |
| max_seq_len | 256 |
| dropout | 0.1 |
| Parametri totali | 45 997 056 |

## Roadmap

| Fase | Obiettivo | Stato |
|------|-----------|-------|
| 1 | Architettura from scratch, char-level | ✅ completata |
| 2 | BPE-SPM tokenizer, Wikipedia IT | ✅ completata |
| 3 | Pre-training medium (Run #3) | 🔄 in corso |
| 4 | RoPE, Run #4, comparazione A/B | ⏳ pianificata |
| 5 | Converter GGUF (`slm_to_gguf.py`) | ⏳ pianificata |
| 6 | Quantizzazione Q4_K_M + llamafile | ⏳ pianificata |
| 7 | Fine-tuning LoRA su corpus personale | ⏳ pianificata |
