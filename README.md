# SLM — Small Language Model

Modello linguistico italiano decoder-only costruito da zero in Python/PyTorch.
Ogni componente scritto e compreso prima di passare al successivo: architettura di attenzione, tokenizer BPE-SPM, training loop AdamW, pipeline corpus, deployment via GGUF.

**Documentazione**: [c1cc10.github.io/slm](https://c1cc10.github.io/slm/)

---

## Stato corrente

| | |
|---|---|
| Architettura | Decoder-only Transformer (pre-norm, RoPE, KV-cache) |
| Parametri | 45,997,056 (~175 MB fp32) |
| Tokenizer | BPE-SPM, 16 000 token |
| Miglior checkpoint | Run #7 shard 3 — val loss **2.8165** (perplexity ~16.7) |
| SFT checkpoint | `sft_intent_v1.pt` — intent recognition su 6 classi (Phase 09) |
| Corpus totale | Wikipedia IT 3 shard (~210M token) su base CulturaX IT sweet spot |
| Deployment | `slm-run7-s3-q8_0.gguf` (56 MB Q8_0) |

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
│   ├── infer.py              # Inferenza CLI: prompt → generazione (REPL + arg + stdin)
│   ├── eval_ppl.py           # Valutazione PPL interattiva (REPL + CLI)
│   ├── eval.py               # Valutazione PPL su dataset con sliding window, output CSV
│   ├── gen_sft_mini.py       # Genera dataset SFT JSONL (intent recognition italiano)
│   ├── filter_corpus.py      # Pipeline 4 stadi: euristico → fastText → dedup MD5 → PPL scorer
│   ├── fetch_hf_corpus.py    # Streaming generico HuggingFace (qualsiasi dataset)
│   ├── epub_to_text.py       # Estrazione corpus da EPUB
│   ├── convert_to_text.py    # Converter: PDF, DOCX, PPTX, EPUB, HTML, ODT → plain text
│   └── export_db_corpus.py   # Export PostgreSQL/MongoDB → corpus SLM
├── data/
│   ├── wiki_it.txt           # Wikipedia IT (~200 MB, 47.8M token)
│   ├── culturax_it_ppl95.txt # CulturaX IT filtrato PPL p95 (7.5 GB, 2.44M doc)
│   ├── corpus_letterario.txt # EPUB scelti, narrativa contemporanea italiana
│   ├── sft_mini.jsonl        # Dataset SFT: 140 coppie prompt/completion, 6 intent italiani
│   └── wiki_probe.txt        # Probe set forgetting (80 KB da wiki_it.txt, fisso)
├── checkpoints/
│   ├── best.pt               # Checkpoint corrente (Run #7 shard 3: val 2.8165)
│   ├── run4_best.pt          # Backup Run #4 (val 3.5861) — scorer PPL
│   ├── sft_intent_v1.pt      # SFT Phase 09: intent recognition, 6 classi
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

## Strumenti

### `tools/infer.py` — Inferenza da checkpoint `.pt`

Esegue la generazione su un checkpoint PyTorch. Accetta prompt da argomento, stdin o REPL interattivo.

```bash
# Frase singola
python3 tools/infer.py "La capitale d'Italia"

# Checkpoint specifico
python3 tools/infer.py --checkpoint checkpoints/sft_intent_v1.pt \
  "Crea un appuntamento con Marco venerdì alle 15."

# Tronca al primo JSON valido (utile con checkpoint SFT intent)
python3 tools/infer.py --checkpoint checkpoints/sft_intent_v1.pt \
  --stop-on-json "Sposta la riunione a martedì."

# Output grezzo pipabile
python3 tools/infer.py --raw "La capitale" | head -c 200

# REPL interattivo — più prompt in sequenza
python3 tools/infer.py --checkpoint checkpoints/sft_intent_v1.pt
```

**Opzioni:**

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--checkpoint` | `checkpoints/best.pt` | Checkpoint `.pt` da usare |
| `--max-tokens` | 80 | Token da generare |
| `--temperature` | 0.3 | 0.1 = deterministico, 1.0 = standard |
| `--top-k` | 10 | Top-K campionamento (0 = disabilitato) |
| `--top-p` | 1.0 | Nucleus sampling (1.0 = disabilitato) |
| `--rep-penalty` | 1.1 | Penalità ripetizione (1.0 = disabilitato) |
| `--stop-on-json` | off | Tronca al primo `{}` JSON valido |
| `--no-cache` | off | Disabilita KV-cache (debug) |
| `--raw` | off | Output solo completion, senza formattazione |

---

### `tools/eval_ppl.py` — Perplexity su testo libero

Misura quanto il modello è "sorpreso" da un testo. Utile per valutare la qualità del corpus e per il filtro documenti.

```bash
# Testo singolo
python3 tools/eval_ppl.py "Il Parlamento europeo è un'istituzione dell'Unione Europea."

# Checkpoint specifico
python3 tools/eval_ppl.py --checkpoint checkpoints/best.pt "testo da valutare"

# REPL interattivo
python3 tools/eval_ppl.py

# Stdin
echo "testo da valutare" | python3 tools/eval_ppl.py
```

Scala PPL: `< 60` bassa (simile a Wikipedia IT) · `60–150` media · `150–400` alta · `400+` molto alta.

---

### `tools/eval.py` — Valutazione PPL su dataset con sliding window

Calcola la perplexity su un intero file di testo usando finestre sovrapposte, evitando il troncamento a `max_seq_len`. Produzione di metriche quantitative confrontabili tra run.

```bash
# PPL su corpus di test
python3 tools/eval.py --checkpoint checkpoints/best.pt --data data/wiki_it.txt

# Con output CSV per confronto tra run
python3 tools/eval.py --checkpoint checkpoints/best.pt --data data/wiki_it.txt \
  --output results/ppl_run7.csv

# Confronto diretto tra due checkpoint
python3 tools/eval.py \
  --compare checkpoints/best.pt checkpoints/run4_best.pt \
  --data data/wiki_it.txt
```

---

### `tools/gen_sft_mini.py` — Generazione dataset SFT

Genera `data/sft_mini.jsonl`: 140 coppie prompt/completion in italiano per intent recognition su 6 classi (`create_event`, `reschedule_event`, `delete_event`, `create_reminder`, `query_schedule`, `add_contact`).

```bash
python3 tools/gen_sft_mini.py
# Output: data/sft_mini.jsonl (140 righe)
```

Formato output:
```json
{"prompt": "Crea un appuntamento con Marco venerdì alle 15.", "completion": "{\"intent\": \"create_event\", \"title\": \"appuntamento\", \"date\": \"venerdì\", \"time\": \"15:00\"}"}
```

---

### `train_sft.py` — Fine-tuning supervisionato (SFT + opzione LoRA)

Fine-tuning con masked loss su dataset JSONL. Supporta due modalità:

**Modalità full SFT** (aggiorna tutti i parametri, rischio overfitting su dataset piccoli):
```bash
python3 train_sft.py \
  --checkpoint checkpoints/best.pt \
  --train data/sft_mini.jsonl \
  --epochs 5 --lr 2e-5 \
  --out checkpoints/sft_intent_v1.pt
```

**Modalità LoRA** (W frozen, solo A e B trainabili — consigliata per dataset piccoli):
```bash
python3 train_sft.py \
  --checkpoint checkpoints/best.pt \
  --train data/sft_mini.jsonl \
  --lora --lora-rank 8 --lora-alpha 16 \
  --epochs 10 --lr 2e-4 \
  --lora-out checkpoints/lora_calendar.pt \
  --out checkpoints/lora_calendar_merged.pt
```

Produce due file: l'adapter (`lora_calendar.pt`, ~1.5 MB) per il dynamic mode e il merged checkpoint per `tools/infer.py`.

**Caricare l'adapter a runtime (dynamic/unmerged mode):**
```python
from model import GPT
from lora import load_lora
import torch

ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
model = GPT(ckpt['model_config'])
model.load_state_dict(ckpt['model_state_dict'])
model = load_lora(model, 'checkpoints/lora_calendar.pt')
model.eval()
```

Il checkpoint merged conserva il tokenizer originale ed è compatibile con `tools/infer.py`.

---

## Provare il modello

I file GGUF sono su **HuggingFace** (non stanno nel repo perché superano il limite di 100 MB di GitHub).

**Scarica i file:**
```
https://huggingface.co/c1cc10/slm-italiano
```

Oppure da terminale con `huggingface-cli`:
```bash
pip install huggingface_hub
huggingface-cli download c1cc10/slm-italiano slm-run7-s3-q8_0.gguf
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
llama-cli.exe -m C:\percorso\slm-run7-s3-q8_0.gguf -p "La capitale d'Italia è" --temp 0.8 -n 200
```

**macOS / Linux**

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --config Release -j
./build/bin/llama-cli -m /percorso/slm-run7-s3-q8_0.gguf -p "La capitale d'Italia è" --temp 0.8 -n 200
```

**File disponibili:**

| File | Dimensione | Quando usarlo |
|------|-----------|---------------|
| `slm-run7-s3-q8_0.gguf` | 56 MB | uso normale — checkpoint più recente (Run #7) |

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
python3 convert_hf_to_gguf.py /percorso/slm --outtype f16 --outfile slm-run7-s3.gguf
./build/bin/llama-quantize slm-run7-s3.gguf slm-run7-s3-q8_0.gguf Q8_0
```

> **Nota**: questo è un modello di ricerca (45.9M parametri, solo pre-training). Genera testo in italiano ma non segue istruzioni — è un completion model, non un assistente conversazionale.

---

## Architettura

| Iperparametro | Valore |
|---|---|
| d_model | 512 |
| num_heads | 8 (d_k = 64) |
| num_layers | 12 |
| d_ff | 2 048 |
| max_seq_len | 512 |
| dropout | 0.1 |
| Parametri totali | **45 997 056** |

**Conto esatto dei parametri** (per chiarezza, perché il GGUF usa il namespace `gptneox.*`):

Il modello non è GPT-NeoX. L'FFN è standard a due matrici lineari, non gated:

| Componente | Per blocco | Note |
|---|---|---|
| Attention (W_q, W_k, W_v, W_o) | 4 × 512² = 1 048 576 | `bias=False` su tutte le proiezioni |
| LN1 + LN2 | 2 × 2 × 512 = 2 048 | weight + bias per ciascuna |
| FFN fc1 (d→d_ff) | 512 × 2048 + 2048 = 1 050 624 | bias=True (default PyTorch) |
| FFN fc2 (d_ff→d) | 2048 × 512 + 512 = 1 049 088 | bias=True |
| **Per blocco** | **3 150 336** | |

12 blocchi: 37 804 032  
Token embedding (tied con lm_head): 16 000 × 512 = 8 192 000  
LayerNorm finale: 1 024  
lm_head: 0 (pesi condivisi con l'embedding)  
**Totale: 45 997 056**

Il namespace `gptneox.*` nelle GGUF metadata riflette il tipo architetturale più vicino supportato da llama.cpp al momento della conversione, non indica GPT-NeoX con FFN gated (gate × up × down). `use_parallel_residual = false` è corretto.

---

## Benchmark — Run #7 shard 3 vs Minerva 350M

Benchmark su 10 prompt in italiano (stesso set per entrambi i modelli, temperatura 0.8, n_predict 300).

| Prompt | SLM s3 | Minerva 350M | Vantaggio |
|--------|--------|--------------|-----------|
| Dante Alighieri nacque | narrativa storica, nessun loop | date inventate, deriva blog | SLM |
| compilazione del kernel | loop ridotto | lista estensioni file | pari (entrambi inutili) |
| un tramonto splendido | palazzo ottomano, visivo | risacca, nuvole, sensoriale | Minerva |
| Le poesie di Ungaretti | prosa critico-letteraria fluente | blog evento, autori inventati | SLM |
| La crisi del 1929 | anno riconosciuto, contesto europeo | confonde WWI con crisi '29 | SLM |
| Il jazz nasce | jazz = K-pop anni '90 | festival jazz blog | pari |
| La fotosintesi è | geologia giapponese | lampade solari | pari |
| Prepara una pasta | loop "pourfail" | ricetta reale (spinaci, panna) | Minerva |
| La regione Puglia è | loop "Foggia" ×20 | Vieste, dettagli reali | Minerva |
| Francesco Benigni | narrativa risorgimentale, zero loop | blog fondazione parrocchiale | SLM |

**Sintesi:** SLM vince su 4 prompt (registro storico-enciclopedico e letterario), Minerva vince su 3 (registro pratico-sensoriale), 3 pari. I domini sono complementari: SLM non ha corpus culinario o turistico moderno, Minerva non ha corpus enciclopedico. Minerva 3B instruct non è utilizzabile in raw completion (il wrapper `[INST]` attiva pattern web dal corpus).

---

## Roadmap

| Fase | Obiettivo | Stato |
|------|-----------|-------|
| 1 | Architettura from scratch, char-level | ✅ val loss 1.73 |
| 2 | BPE-SPM tokenizer, Wikipedia IT | ✅ completata |
| 3 | Pre-training medium (Run #3) | ✅ val loss 4.08 — scorer PPL |
| 4 | RoPE, Run #4, export GGUF | ✅ val loss 3.59 |
| 5 | Sampling avanzato (top-p, repetition penalty) | ✅ completata |
| 6 | Valutazione quantitativa (`tools/eval.py`, sliding window) | ✅ completata |
| 7 | Efficienza training (gradient accumulation, BF16) | ✅ completata |
| 8 | KV-cache inference (4.35× speedup su M2) | ✅ completata |
| 9 | SFT intent recognition — `sft_intent_v1.pt` | ✅ completata · 6 classi · forgetting ×1.12 |
| 10 | LoRA adapter per dominio applicativo | ✅ completata |
| 11 | Multimodale: YOLO v3 + LM | ⏳ pianificata |
