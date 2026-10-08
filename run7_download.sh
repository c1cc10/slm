#!/usr/bin/env bash
# Scarica i risultati di Run #7 da Vast.ai prima del destroy.
# NON sovrascrive file già esistenti in locale.
#
# Uso: bash run7_download.sh
# Eseguire dopo che run7_train_only.sh ha completato tutti i 3 shard.

set -euo pipefail

SSH_KEY="$HOME/.ssh/vastai_slm"
SSH_PORT="37664"
SSH_HOST="ssh2.vast.ai"
REMOTE="root@${SSH_HOST}"
REMOTE_BASE="/root/slm"
LOCAL_BASE="/Users/francescorana/Documents/Development/slm"

safe_scp() {
  local SRC="$1"
  local DST="$2"
  if [[ -f "${DST}" ]]; then
    echo "  SKIP (esiste già): ${DST}"
  else
    scp -i "${SSH_KEY}" -P "${SSH_PORT}" "${REMOTE}:${SRC}" "${DST}" && \
      echo "  OK: $(basename ${DST})  ($(du -sh ${DST} | cut -f1))"
  fi
}

echo "=== Run #7 — download da Vast.ai ($(date '+%Y-%m-%d %H:%M')) ==="
echo ""

# Verifica che il training sia completato
echo "Verifica completamento training..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -p "${SSH_PORT}" "${REMOTE}" \
  "ls -lh ${REMOTE_BASE}/checkpoints/run7_shard_s{1,2,3}.pt 2>/dev/null || echo 'ATTENZIONE: alcuni checkpoint mancanti'"

echo ""
echo "--- Checkpoints ---"
mkdir -p "${LOCAL_BASE}/checkpoints"
safe_scp "${REMOTE_BASE}/checkpoints/run7_shard_s1.pt"  "${LOCAL_BASE}/checkpoints/run7_shard_s1.pt"
safe_scp "${REMOTE_BASE}/checkpoints/run7_shard_s2.pt"  "${LOCAL_BASE}/checkpoints/run7_shard_s2.pt"
safe_scp "${REMOTE_BASE}/checkpoints/run7_shard_s3.pt"  "${LOCAL_BASE}/checkpoints/run7_shard_s3.pt"

echo ""
echo "--- Log di training ---"
mkdir -p "${LOCAL_BASE}/logs/run7"
safe_scp "${REMOTE_BASE}/logs/run7/run7_full.log"  "${LOCAL_BASE}/logs/run7/run7_full.log"
safe_scp "${REMOTE_BASE}/logs/run7/shard1.log"     "${LOCAL_BASE}/logs/run7/shard1.log"
safe_scp "${REMOTE_BASE}/logs/run7/shard2.log"     "${LOCAL_BASE}/logs/run7/shard2.log"
safe_scp "${REMOTE_BASE}/logs/run7/shard3.log"     "${LOCAL_BASE}/logs/run7/shard3.log"

echo ""
echo "--- Benchmark baseline (eseguito sul remote prima del training) ---"
# Nome diverso per non confondere con il benchmark locale post-training
safe_scp "${REMOTE_BASE}/logs/run7/benchmark.log"  "${LOCAL_BASE}/logs/run7/benchmark_remote_baseline.log"

echo ""
echo "=== Download completato. ==="
echo ""
echo "Prossimi step:"
echo "  1. Verifica checkpoints: ls -lh ${LOCAL_BASE}/checkpoints/run7_shard_s*.pt"
echo "  2. Benchmark locale:     bash run7_benchmark_local.sh"
echo "  3. Destroy istanza:      vastai destroy instance 54681871"
