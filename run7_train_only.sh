#!/usr/bin/env bash
# Run #7 — Training sequenziale su Wikipedia IT (3 shard)
# Nessun benchmark inline: ogni shard viene salvato e il benchmark
# viene eseguito in locale su M5 dopo aver scaricato i checkpoint.
#
# Versioni da confrontare:
#   baseline   : checkpoints/run6_shard_ad.pt   (già scaricato in locale)
#   post-shard1: checkpoints/run7_shard_s1.pt
#   post-shard2: checkpoints/run7_shard_s2.pt
#   post-shard3: checkpoints/run7_shard_s3.pt
#
# Uso: bash run7_train_only.sh

set -euo pipefail

DEVICE="cuda"
STEPS=20000
BATCH=32
PRESET="medium"
BASE_CKPT="checkpoints/run6_shard_ad.pt"
LOG_DIR="logs/run7"

mkdir -p "${LOG_DIR}"

echo "=== Run #7 — avvio $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee -a "${LOG_DIR}/run7.log"
echo "Checkpoint di partenza: ${BASE_CKPT}" | tee -a "${LOG_DIR}/run7.log"
echo "Benchmark: eseguito in locale dopo download checkpoint" | tee -a "${LOG_DIR}/run7.log"

# Punto di partenza: run6_shard_ad
cp "${BASE_CKPT}" checkpoints/best.pt
echo "Copiato ${BASE_CKPT} → checkpoints/best.pt" | tee -a "${LOG_DIR}/run7.log"

SHARDS=(
  "data/wiki_run7_s1.txt"
  "data/wiki_run7_s2.txt"
  "data/wiki_run7_s3.txt"
)
SHARD_NAMES=("s1" "s2" "s3")

for i in "${!SHARDS[@]}"; do
  SHARD="${SHARDS[$i]}"
  NAME="${SHARD_NAMES[$i]}"
  SHARD_NUM=$((i + 1))

  echo "" | tee -a "${LOG_DIR}/run7.log"
  echo "━━━ Shard ${SHARD_NUM}/3 — ${SHARD} — $(date -u '+%H:%M UTC') ━━━" | tee -a "${LOG_DIR}/run7.log"

  python3 -u train.py \
    --data "${SHARD}" \
    --tokenizer bpe-spm \
    --preset "${PRESET}" \
    --rope \
    --steps "${STEPS}" \
    --batch "${BATCH}" \
    --device "${DEVICE}" \
    --resume \
    --reset-best \
    2>&1 | tee -a "${LOG_DIR}/shard${SHARD_NUM}.log"

  if [[ -f checkpoints/best.pt ]]; then
    cp checkpoints/best.pt "checkpoints/run7_shard_${NAME}.pt"
    echo "  Checkpoint salvato: checkpoints/run7_shard_${NAME}.pt" | tee -a "${LOG_DIR}/run7.log"
  else
    echo "  ATTENZIONE: checkpoints/best.pt non trovato dopo shard ${SHARD_NUM}" | tee -a "${LOG_DIR}/run7.log"
  fi

  echo "  Shard ${SHARD_NUM} completato — $(date -u '+%H:%M UTC')" | tee -a "${LOG_DIR}/run7.log"
done

echo "" | tee -a "${LOG_DIR}/run7.log"
echo "=== Run #7 completato — $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee -a "${LOG_DIR}/run7.log"
echo "Checkpoint da scaricare in locale per benchmark:" | tee -a "${LOG_DIR}/run7.log"
echo "  checkpoints/run7_shard_s1.pt" | tee -a "${LOG_DIR}/run7.log"
echo "  checkpoints/run7_shard_s2.pt" | tee -a "${LOG_DIR}/run7.log"
echo "  checkpoints/run7_shard_s3.pt" | tee -a "${LOG_DIR}/run7.log"
