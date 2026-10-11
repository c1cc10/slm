#!/usr/bin/env bash
# LoRA training — calendario intent v2 (Phase 10 — dataset espanso)
# Dataset: data/sft_v2.jsonl (4200 esempi, 21 intent, 200/intent)
# Base model: checkpoints/best.pt (run7_shard_s3, val 2.8165)
# Infra: Vast.ai RTX 4090 — Europa (Svezia preferita)
#
# Uso locale:  bash run_lora.sh [--dry-run]
# Uso remoto:  eseguito automaticamente dall'orchestrator

set -euo pipefail

DRY="${1:-}"

# ── Parametri training ────────────────────────────────────────────────────────
DEVICE="cuda"
EPOCHS=10
BATCH=4
LR="2e-4"
LORA_RANK=8
LORA_ALPHA=16
BASE_CKPT="checkpoints/best.pt"
DATASET="data/sft_v2.jsonl"
LORA_OUT="checkpoints/lora_calendar_v2.pt"
MERGED_OUT="checkpoints/lora_calendar_v2_merged.pt"
LOG_DIR="logs/lora_v2"

mkdir -p "${LOG_DIR}" checkpoints logs

echo "=== LoRA Calendar — avvio $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee "${LOG_DIR}/lora.log"
echo "  Base   : ${BASE_CKPT}" | tee -a "${LOG_DIR}/lora.log"
echo "  Dataset: ${DATASET} ($(wc -l < ${DATASET}) esempi)" | tee -a "${LOG_DIR}/lora.log"
echo "  Epoche : ${EPOCHS}  batch=${BATCH}  lr=${LR}" | tee -a "${LOG_DIR}/lora.log"
echo "  LoRA   : rank=${LORA_RANK}  alpha=${LORA_ALPHA}" | tee -a "${LOG_DIR}/lora.log"
echo "" | tee -a "${LOG_DIR}/lora.log"

if [[ "$DRY" == "--dry-run" ]]; then
  echo "[DRY RUN] Comando che verrebbe eseguito:"
  echo "  python3 train_sft.py \\"
  echo "    --checkpoint ${BASE_CKPT} \\"
  echo "    --train ${DATASET} \\"
  echo "    --lora --lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA} \\"
  echo "    --epochs ${EPOCHS} --batch ${BATCH} --lr ${LR} \\"
  echo "    --device ${DEVICE} \\"
  echo "    --lora-out ${LORA_OUT} \\"
  echo "    --out ${MERGED_OUT}"
  exit 0
fi

# ── Verifica dipendenze ───────────────────────────────────────────────────────
for f in "${BASE_CKPT}" "${DATASET}" train_sft.py lora.py model.py attention.py transformer.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERRORE: file mancante: $f" | tee -a "${LOG_DIR}/lora.log"
    exit 1
  fi
done

# ── Install dipendenze Python se necessario ───────────────────────────────────
python3 -c "import sentencepiece" 2>/dev/null || pip install sentencepiece -q
python3 -c "import torch" 2>/dev/null || { echo "PyTorch non trovato"; exit 1; }

echo "GPU disponibile:" | tee -a "${LOG_DIR}/lora.log"
python3 -c "import torch; print(f'  {torch.cuda.get_device_name(0)}  VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')" \
  2>/dev/null | tee -a "${LOG_DIR}/lora.log" || echo "  (CUDA non disponibile)" | tee -a "${LOG_DIR}/lora.log"
echo "" | tee -a "${LOG_DIR}/lora.log"

# ── Training ──────────────────────────────────────────────────────────────────
echo "--- Training avviato $(date -u '+%H:%M UTC') ---" | tee -a "${LOG_DIR}/lora.log"

python3 -u train_sft.py \
  --checkpoint "${BASE_CKPT}" \
  --train "${DATASET}" \
  --lora \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --lora-out "${LORA_OUT}" \
  --out "${MERGED_OUT}" \
  2>&1 | tee -a "${LOG_DIR}/lora_train.log"

# ── Risultati ─────────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG_DIR}/lora.log"
echo "--- Training completato $(date -u '+%H:%M UTC') ---" | tee -a "${LOG_DIR}/lora.log"
echo "" | tee -a "${LOG_DIR}/lora.log"

for f in "${LORA_OUT}" "${MERGED_OUT}"; do
  if [[ -f "$f" ]]; then
    SIZE=$(du -sh "$f" | cut -f1)
    echo "  ✓ ${f}  (${SIZE})" | tee -a "${LOG_DIR}/lora.log"
  else
    echo "  ✗ MANCANTE: ${f}" | tee -a "${LOG_DIR}/lora.log"
  fi
done

# ── Inference test ────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG_DIR}/lora.log"
echo "--- Inference test su merged checkpoint ---" | tee -a "${LOG_DIR}/lora.log"

# tools/infer_lora.py usa il separatore corretto (\n\n) e fa stop al primo JSON
if [[ -f "tools/infer_lora.py" ]]; then
  python3 tools/infer_lora.py \
    --checkpoint "${MERGED_OUT}" \
    --device cpu \
    2>&1 | tee -a "${LOG_DIR}/lora.log"
else
  # Fallback inline con separatore e JSON extraction
  python3 -c "
import torch, json, sys
sys.path.insert(0, '.')
import train  # per TrainConfig pickle
from model import GPT
import sentencepiece as spm

ckpt = torch.load('${MERGED_OUT}', map_location='cpu', weights_only=False)
model = GPT(ckpt['model_config'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(bytes(ckpt['tokenizer_state']['model_bytes']))

def extract_json(text):
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try: json.loads(text[start:i+1]); return text[start:i+1]
                except: start = None
    return None

prompts = [
  'Crea un appuntamento con Marco venerdì alle 15.',
  'Sposta la riunione di domani a giovedì pomeriggio.',
  'Cosa ho in agenda lunedì prossimo?',
  'Ricordami di chiamare il medico domani mattina.',
  'Cancella tutti gli impegni di mercoledì.',
]
for p in prompts:
    ids = sp.encode(p + '\n\n', out_type=int)
    x = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=120, temperature=0.2, top_k=10)
    raw = sp.decode(out[0][len(ids):].tolist())
    result = extract_json(raw) or '(nessun JSON valido)'
    print(f'IN : {p}')
    print(f'OUT: {result}')
    print()
" 2>&1 | tee -a "${LOG_DIR}/lora.log"
fi

echo "" | tee -a "${LOG_DIR}/lora.log"
echo "=== LoRA Calendar completato — $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee -a "${LOG_DIR}/lora.log"
echo "" | tee -a "${LOG_DIR}/lora.log"
echo "File da scaricare:"
echo "  checkpoints/lora_calendar_v2.pt        (adapter v2)"
echo "  checkpoints/lora_calendar_v2_merged.pt (merged v2)"
echo "  logs/lora_v2/lora_train.log            (loss curve)"
echo "  logs/lora_v2/lora.log                  (riepilogo)"
