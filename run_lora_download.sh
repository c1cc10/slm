#!/usr/bin/env bash
# Scarica i risultati del LoRA training da Vast.ai.
# Legge le SSH info da lora_instance.env (creato da run_lora_vastai.sh).
# Regola backup: NON sovrascrive file già esistenti in locale.
#
# Uso: bash run_lora_download.sh

set -euo pipefail

LOCAL_BASE="/Users/francescorana/Documents/Development/slm"
ENV_FILE="${LOCAL_BASE}/lora_instance.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERRORE: ${ENV_FILE} non trovato. Esegui prima run_lora_vastai.sh."
  exit 1
fi
source "${ENV_FILE}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

ssh_run() {
  ssh -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=no "root@${SSH_HOST}" "$@"
}
safe_scp() {
  local SRC="$1"
  local DST="$2"
  if [[ -f "${DST}" ]]; then
    log "  SKIP (esiste): $(basename ${DST})"
  else
    scp -i "${SSH_KEY}" -P "${SSH_PORT}" -q -o StrictHostKeyChecking=no \
      "root@${SSH_HOST}:${SRC}" "${DST}" && \
      log "  ✓ $(basename ${DST})  ($(du -sh ${DST} | cut -f1))"
  fi
}

log "=== Download LoRA da Vast.ai (istanza ${INSTANCE_ID}) ==="
log ""

# Verifica training completato
log "Stato training remoto:"
ssh_run "tail -5 ${REMOTE_BASE}/logs/lora/lora.log 2>/dev/null || echo '(log non trovato)'"
echo ""

log "Verifica checkpoint:"
ssh_run "ls -lh ${REMOTE_BASE}/checkpoints/lora_calendar*.pt 2>/dev/null || echo 'ATTENZIONE: checkpoint non trovati'"
echo ""

# Download checkpoint — BACKUP OBBLIGATORIO
log "--- Checkpoint ---"
mkdir -p "${LOCAL_BASE}/checkpoints" "${LOCAL_BASE}/logs/lora"
safe_scp "${REMOTE_BASE}/checkpoints/lora_calendar.pt" \
         "${LOCAL_BASE}/checkpoints/lora_calendar.pt"
safe_scp "${REMOTE_BASE}/checkpoints/lora_calendar_merged.pt" \
         "${LOCAL_BASE}/checkpoints/lora_calendar_merged.pt"

log ""
log "--- Log ---"
safe_scp "${REMOTE_BASE}/logs/lora/lora_train.log" \
         "${LOCAL_BASE}/logs/lora/lora_train.log"
safe_scp "${REMOTE_BASE}/logs/lora/lora.log" \
         "${LOCAL_BASE}/logs/lora/lora.log"

log ""
log "=== Download completato ==="
log ""
log "Prossimi step:"
log "  1. Verifica checkpoint:"
log "     ls -lh ${LOCAL_BASE}/checkpoints/lora_calendar*.pt"
log "  2. Destroy istanza (SOLO dopo aver verificato i file):"
log "     vastai destroy instance ${INSTANCE_ID}"
log "  3. Inference test locale:"
log "     python3 infer.py --checkpoint checkpoints/lora_calendar_merged.pt"
