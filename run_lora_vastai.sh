#!/usr/bin/env bash
# Orchestrator LoRA su Vast.ai
# 1. Crea istanza RTX 4090 Europa
# 2. Attende SSH
# 3. Carica file
# 4. Esegue run_lora.sh in remoto
# 5. Download automatico al termine
#
# Uso: bash run_lora_vastai.sh [--dry-run]

set -euo pipefail

DRY="${1:-}"
SSH_KEY="$HOME/.ssh/vastai_slm"
LOCAL_BASE="/Users/francescorana/Documents/Development/slm"
REMOTE_BASE="/root/slm"
IMAGE="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel"
DISK_GB=20

# Offerte Europa preferite (aggiornate 2026-10-11)
OFFER_EU_PRIMARY=55036744    # Germania  $0.395/hr
OFFER_EU_FALLBACK=54135765   # Danimarca $0.401/hr

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Crea istanza ───────────────────────────────────────────────────────────
log "Creazione istanza Vast.ai (offerta ${OFFER_EU_PRIMARY} — Svezia)…"

if [[ "$DRY" == "--dry-run" ]]; then
  log "[DRY RUN] vastai create instance ${OFFER_EU_PRIMARY} --image ${IMAGE} --disk ${DISK_GB} --ssh --direct"
  INSTANCE_ID="DRY_RUN_ID"
else
  # Funzione per estrarre new_contract dal risultato (supporta JSON e Python dict)
  extract_id() {
    python3 -c "
import sys, re
text = sys.stdin.read()
# Prova JSON standard
import json
try:
    d = json.loads(text)
    print(d.get('new_contract', ''))
    sys.exit(0)
except: pass
# Prova ast.literal_eval (Python dict con singoli apici)
import ast
m = re.search(r'\{[^}]+\}', text)
if m:
    try:
        d = ast.literal_eval(m.group())
        print(d.get('new_contract', ''))
        sys.exit(0)
    except: pass
# Fallback regex multipli
for pat in [r'new_contract\D+(\d+)', r'Got\s+(\d+)', r'(\d{7,9})']:
    m = re.search(pat, text)
    if m:
        print(m.group(1))
        sys.exit(0)
" <<< "$1"
  }

  RESULT=$(vastai create instance "${OFFER_EU_PRIMARY}" \
    --image "${IMAGE}" \
    --disk "${DISK_GB}" \
    --ssh --direct \
    --onstart-cmd "mkdir -p /root/slm/checkpoints /root/slm/data /root/slm/logs/lora_v2 /root/slm/tools" \
    2>&1)
  echo "${RESULT}"

  INSTANCE_ID=$(extract_id "${RESULT}")
  if [[ -z "${INSTANCE_ID}" ]]; then
    log "ERRORE: impossibile estrarre instance ID. Output:"
    echo "${RESULT}"
    log "Provo con offerta fallback (${OFFER_EU_FALLBACK})…"
    RESULT=$(vastai create instance "${OFFER_EU_FALLBACK}" \
      --image "${IMAGE}" --disk "${DISK_GB}" --ssh --direct \
      --onstart-cmd "mkdir -p /root/slm/checkpoints /root/slm/data /root/slm/logs/lora_v2 /root/slm/tools" \
      2>&1)
    echo "${RESULT}"
    INSTANCE_ID=$(extract_id "${RESULT}")
  fi
  if [[ -z "${INSTANCE_ID}" ]]; then
    log "ERRORE FATALE: instance ID non estratto. Abort."
    exit 1
  fi
  log "Instance ID: ${INSTANCE_ID}"
fi

# ── 2. Attendi SSH ────────────────────────────────────────────────────────────
if [[ "$DRY" != "--dry-run" ]]; then
  log "Attendo che l'istanza sia pronta (max 5 min)…"
  SSH_HOST=""
  SSH_PORT=""

  for attempt in $(seq 1 30); do
    sleep 10
    vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null > /tmp/vast_instance.json || true
    read STATUS SSH_HOST SSH_PORT < <(python3 -c "
import json, sys
try:
    with open('/tmp/vast_instance.json') as f:
        raw = f.read()
    d = json.loads(raw, strict=False)
    print(d.get('actual_status',''), d.get('ssh_host',''), d.get('ssh_port',''))
except Exception as e:
    print('', '', '')
" 2>/dev/null) || true

    log "  tentativo ${attempt}/30 — status=${STATUS}  host=${SSH_HOST}  port=${SSH_PORT}"

    if [[ "${STATUS}" == "running" && -n "${SSH_HOST}" && -n "${SSH_PORT}" ]]; then
      log "Istanza pronta."
      break
    fi
    if [[ $attempt -eq 30 ]]; then
      log "TIMEOUT: istanza non pronta dopo 5 min. Verifica manualmente."
      log "  vastai show instance ${INSTANCE_ID}"
      exit 1
    fi
  done

  # Attendi che SSH risponda
  log "Attendo SSH su ${SSH_HOST}:${SSH_PORT}…"
  for attempt in $(seq 1 12); do
    sleep 5
    if ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
         -p "${SSH_PORT}" "root@${SSH_HOST}" "echo ok" &>/dev/null; then
      log "SSH attivo."
      break
    fi
    if [[ $attempt -eq 12 ]]; then
      log "TIMEOUT SSH. Riprova manualmente:"
      log "  ssh -i ${SSH_KEY} -p ${SSH_PORT} root@${SSH_HOST}"
      exit 1
    fi
  done

  # Salva SSH info per download/destroy
  cat > "${LOCAL_BASE}/lora_instance.env" << EOF
INSTANCE_ID=${INSTANCE_ID}
SSH_HOST=${SSH_HOST}
SSH_PORT=${SSH_PORT}
SSH_KEY=${SSH_KEY}
REMOTE_BASE=${REMOTE_BASE}
LOCAL_BASE=${LOCAL_BASE}
EOF
  log "SSH info salvate in lora_instance.env"

# ── 3. Upload file ────────────────────────────────────────────────────────────
  log "Upload file su ${SSH_HOST}:${SSH_PORT}…"

  scp_to() {
    scp -i "${SSH_KEY}" -P "${SSH_PORT}" -q \
      -o StrictHostKeyChecking=no "$@"
  }
  ssh_run() {
    ssh -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=no \
      "root@${SSH_HOST}" "$@"
  }

  ssh_run "mkdir -p ${REMOTE_BASE}/checkpoints ${REMOTE_BASE}/data ${REMOTE_BASE}/logs/lora"

  log "  → best.pt (530 MB)…"
  scp_to "${LOCAL_BASE}/checkpoints/best.pt" \
         "root@${SSH_HOST}:${REMOTE_BASE}/checkpoints/best.pt"

  log "  → dataset sft_v2.jsonl…"
  scp_to "${LOCAL_BASE}/data/sft_v2.jsonl" \
         "root@${SSH_HOST}:${REMOTE_BASE}/data/sft_v2.jsonl"

  log "  → codice Python…"
  scp_to "${LOCAL_BASE}/train_sft.py" \
         "${LOCAL_BASE}/train.py" \
         "${LOCAL_BASE}/lora.py" \
         "${LOCAL_BASE}/model.py" \
         "${LOCAL_BASE}/attention.py" \
         "${LOCAL_BASE}/transformer.py" \
         "${LOCAL_BASE}/run_lora.sh" \
         "root@${SSH_HOST}:${REMOTE_BASE}/"

  log "  → tools/infer_lora.py…"
  ssh_run "mkdir -p ${REMOTE_BASE}/tools"
  scp_to "${LOCAL_BASE}/tools/infer_lora.py" \
         "root@${SSH_HOST}:${REMOTE_BASE}/tools/infer_lora.py"

  log "Upload completato."

# ── 4. Lancia training in background ─────────────────────────────────────────
  log "Lancio training remoto…"
  ssh_run "cd ${REMOTE_BASE} && pip install sentencepiece -q && mkdir -p logs/lora_v2 && nohup bash run_lora.sh > logs/lora_v2/stdout.log 2>&1 &"
  log "Training avviato in background sul remote."
  log ""
  log "Monitoraggio live:"
  log "  ssh -i ${SSH_KEY} -p ${SSH_PORT} root@${SSH_HOST} 'tail -f ${REMOTE_BASE}/logs/lora_v2/lora_train.log'"
  log ""
  log "Download quando completato:"
  log "  bash run_lora_download.sh"
  log ""
  log "Destroy istanza:"
  log "  vastai destroy instance ${INSTANCE_ID}"

fi
