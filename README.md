# SLM — Small Language Model

Modello linguistico italiano decoder-only costruito da zero in Python/PyTorch.
Ogni componente scritto e compreso prima di passare al successivo: architettura di attenzione, tokenizer BPE-SPM, training loop AdamW, pipeline corpus, deployment via GGUF.

**Documentazione**: [c1cc10.github.io/slm](https://c1cc10.github.io/slm/)

---

## Stato corrente

| | |
|---|---|
| Architettura | Decoder-only Transformer (pre-norm, RoPE) |
| Parametri | 45.9M (~175 MB fp32) |
| Tokenizer | BPE-SPM, 16 000 token |
| Miglior checkpoint | Run #4 — val loss **3.5861** (perplexity ~36) |
| Run in corso | #5 — shard 3/7 completato · val loss min Run #5: **3.821** · shard 4/7 in training |
| Target Chinchilla | 918M token (20 × 45.9M param) — Run #5 ne processa ~917M |
| Deployment | `slm-run4.gguf` (fp16) · `slm-run4-q8_0.gguf` (Q8_0) |

---

## Struttura

```
slm/
├── attention.py              # Scaled dot-product attention, Multi-Head Attention, RoPE
├── transformer.py            # FeedForward (GELU), TransformerBlock (pre-norm)
├── model.py                  # GPTConfig, GPT (forward + generate + weight tying)
├── train.py                  # Training loop: AdamW, LR schedule, checkpoint, --resume
├── train_sft.py              # SFT: JSONL prompt/completion, masked loss
├── tools/
│   ├── filter_corpus.py      # Pipeline 4 stadi: euristico → fastText → dedup MD5 → PPL scorer
│   ├── fetch_hf_corpus.py    # Streaming generico HuggingFace (qualsiasi dataset)
│   ├── epub_to_text.py       # Estrazione corpus da EPUB
│   ├── convert_to_text.py    # Converter: PDF, DOCX, PPTX, EPUB, HTML, ODT → plain text
│   ├── eval_ppl.py           # Valutazione PPL interattiva (REPL + CLI)
│   └── export_db_corpus.py   # Export PostgreSQL/MongoDB → corpus SLM
├── data/
│   ├── wiki_it.txt           # Wikipedia IT (~200 MB, 47.8M token)
│   ├── culturax_it_ppl95.txt # CulturaX IT filtrato PPL p95 (7.5 GB, 2.44M doc)
│   └── corpus_letterario.txt # EPUB scelti, narrativa contemporanea italiana
├── checkpoints/
│   ├── best.pt               # Checkpoint corrente (Run #4: val 3.5861)
│   ├── run4_best.pt          # Backup Run #4
│   ├── tokenizer.model       # Modello SentencePiece BPE (16k vocab)
│   └── tokenizer.vocab       # Vocabolario SPM
├── slm-run4.gguf             # Run #4 convertito GGUF (fp16)
├── slm-run4-q8_0.gguf        # Run #4 quantizzato Q8_0
└── docs/                     # Documentazione HTML (GitHub Pages)
    ├── index.html
    ├── slm-journal.html      # Diario: decisioni, run, corpus, caveats
    ├── slm-intro.html        # Come funziona una LLM
    ├── slm-roadmap.html      # Roadmap 11 fasi
    ├── slm-docs.html         # Documentazione completa
    ├── slm-teoria.html       # Riferimento teorico con formule
    └── slm-ottimizzatore.html
```

---

## Requisiti

```bash
pip install torch sentencepiece fasttext
```

## Training

```bash
# Resume Run #5 (corpus espanso, da checkpoint Run #4)
python3 -u train.py --tokenizer bpe-spm --data data/corpus_run5.txt \
  --resume 2>&1 | tee /tmp/run5.log

# Pre-training da zero (preset medium: 45.9M parametri)
python3 train.py --tokenizer bpe-spm --data data/wiki_it.txt \
  --preset medium --steps 30000 --batch 32 --seq-len 512

# Generazione testo da checkpoint
python3 train.py --generate "La capitale d'Italia"
```

## Filtro corpus

```bash
# Filtro completo con PPL scorer su GPU
python3 -u tools/filter_corpus.py \
  --input data/culturax_it_raw.txt \
  --output data/culturax_it_ppl95.txt \
  --checkpoint checkpoints/best.pt \
  --device cuda --ppl-pct 95 \
  2>&1 | tee /tmp/filter.log

# Nota: python3 -u obbligatorio con tee (altrimenti log vuoto per ore)
# Nota: usare run3_best.pt come scorer, non run4_best.pt
```

## Provare il modello

I file GGUF del Run #4 sono su **HuggingFace** (non stanno nel repo perché superano il limite di 100 MB di GitHub).

**Scarica i file:**
```
https://huggingface.co/c1cc10/slm-italiano
```

Oppure da terminale con `huggingface-cli`:
```bash
pip install huggingface_hub
huggingface-cli download c1cc10/slm-italiano slm-run4-q8_0.gguf
huggingface-cli download c1cc10/slm-italiano slm-run4.gguf
```

Ci sono due modi per usarli.

---

### Opzione A — llama.cpp (consigliata, nessun Python necessario)

llama.cpp è un runtime C++ che gira su qualsiasi macchina, anche senza GPU.

**Windows**

1. Vai su https://github.com/ggerganov/llama.cpp/releases
2. Scarica l'archivio ZIP più recente per Windows — scegli la variante giusta:
   - `llama-bXXXX-bin-win-avx2-x64.zip` — CPU moderna (consigliato se non hai GPU NVIDIA)
   - `llama-bXXXX-bin-win-cuda-cu12.X.X-x64.zip` — GPU NVIDIA con CUDA
3. Estrai lo ZIP in una cartella, ad esempio `C:\llama.cpp\`
4. Apri PowerShell o il Prompt dei comandi nella stessa cartella e lancia:

```
llama-cli.exe -m C:\percorso\slm\slm-run4-q8_0.gguf -p "La capitale d'Italia è" --temp 0.8 -n 200
```

**macOS / Linux**

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --config Release -j
./build/bin/llama-cli -m /percorso/slm/slm-run4-q8_0.gguf -p "La capitale d'Italia è" --temp 0.8 -n 200
```

**Quale GGUF scegliere:**

| File | Dimensione | Quando usarlo |
|------|-----------|---------------|
| `slm-run4-q8_0.gguf` | ~46 MB | uso normale — qualità quasi identica al fp16 |
| `slm-run4.gguf` | ~88 MB | se vuoi la precisione completa fp16 |

---

### Opzione B — Python (richiede PyTorch)

Se hai già Python installato puoi usare `train.py` direttamente.

**Installare i prerequisiti**

Verifica che Python sia installato (versione 3.10 o superiore):
```
python --version
```

Se non è installato, scaricalo da https://www.python.org/downloads/ — su Windows, durante l'installazione spunta "Add Python to PATH".

Installa le dipendenze:
```
pip install torch sentencepiece
```

Su Windows con GPU NVIDIA, sostituisci il comando pip con quello specifico per CUDA che trovi su https://pytorch.org/get-started/locally/ (seleziona: Windows / Pip / CUDA).

**Generare testo dal checkpoint PyTorch:**

```
python train.py --generate "La capitale d'Italia" --temp 0.8 --max-new-tokens 200
```

Il checkpoint `checkpoints/best.pt` e il tokenizer `checkpoints/tokenizer.model` devono essere presenti nella directory.

---

**Alternativa: costruire il GGUF dal checkpoint PyTorch**

Se hai il checkpoint `checkpoints/best.pt` e llama.cpp clonato localmente, puoi convertirlo tu stesso:
```bash
# dalla directory llama.cpp
python3 convert_hf_to_gguf.py /percorso/slm --outtype f16 --outfile slm-run4.gguf
./build/bin/llama-quantize slm-run4.gguf slm-run4-q8_0.gguf Q8_0
```

> **Nota**: questo è un modello di ricerca (45.9M parametri, solo pre-training). Genera testo in italiano ma non segue istruzioni — è un completion model, non un assistente conversazionale.

---

## Architettura — preset medium (Run #4)

| Iperparametro | Valore |
|---|---|
| d_model | 512 |
| num_heads | 8 (d_k = 64) |
| num_layers | 12 |
| d_ff | 2 048 |
| max_seq_len | 512 |
| dropout | 0.1 |
| Parametri totali | 45 997 056 |

---

## Roadmap

| Fase | Obiettivo | Stato |
|------|-----------|-------|
| 1 | Architettura from scratch, char-level | ✅ val loss 1.73 |
| 2 | BPE-SPM tokenizer, Wikipedia IT | ✅ completata |
| 3 | Pre-training medium (Run #3) | ✅ val loss 4.08 — scorer PPL |
| 4 | RoPE, Run #4, export GGUF | ✅ val loss 3.59 — best checkpoint |
| 5 | Corpus espansione + Run #5 (Chinchilla-optimal) | 🔄 in corso — shard 3/7 ✓ · val min 3.821 · ~16h rimanenti |
| 6 | Run #6 su GPU, corpus mix finale | ⏳ dopo valutazione Run #5 |
| 7 | SFT su dataset istruzione-seguente italiano | ⏳ pianificata |
| 8 | LoRA adapter per dominio applicativo | ⏳ pianificata |
| 9 | Quantizzazione Q4_K_M + inferenza locale | ⏳ pianificata |
| 10 | Multimodale: YOLO v3 + LM | ⏳ pianificata |
