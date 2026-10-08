#!/usr/bin/env bash
# Run #7 — Training sequenziale su Wikipedia IT (3 shard)
# Punto di partenza: run6_shard_ad.pt (sweet spot Run #6, val loss 3.3524)
# Benchmark obbligatorio dopo ogni shard — stessi 10 prompt, temp=0.8, top_k=20
#
# Uso: bash run7_sequential.sh [--dry-run]

set -euo pipefail

DRY="${1:-}"
DEVICE="cuda"
STEPS=8000
BATCH=32
PRESET="medium"
BASE_CKPT="checkpoints/run6_shard_ad.pt"
LOG_DIR="logs/run7"
BENCH_LOG="${LOG_DIR}/benchmark.log"

mkdir -p "${LOG_DIR}"

# 10 prompt fissi del benchmark — stessi usati per il confronto Minerva
BENCHMARK_PROMPTS=(
  "Dante Alighieri nacque"
  "compilazione del kernel"
  "un tramonto splendido"
  "Le poesie di Ungaretti"
  "La crisi del 1929"
  "Il jazz nasce"
  "La fotosintesi è"
  "Prepara una pasta"
  "La regione Puglia è"
  "Francesco Benigni"
)

run_benchmark() {
  local CKPT="$1"
  local LABEL="$2"
  echo ""
  echo "━━━ BENCHMARK: ${LABEL} ($(date -u '+%H:%M UTC')) ━━━" | tee -a "${BENCH_LOG}"
  echo "Checkpoint: ${CKPT}" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"

  for PROMPT in "${BENCHMARK_PROMPTS[@]}"; do
    echo "--- PROMPT: ${PROMPT} ---" | tee -a "${BENCH_LOG}"
    python3 train.py \
      --generate "${PROMPT}" \
      --checkpoint "${CKPT}" \
      --device cpu 2>/dev/null | tee -a "${BENCH_LOG}"
    echo "" | tee -a "${BENCH_LOG}"
  done
  echo "━━━ FINE BENCHMARK ${LABEL} ━━━" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
}

echo "=== Run #7 — avvio $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee -a "${LOG_DIR}/run7.log"
echo "Checkpoint di partenza: ${BASE_CKPT}" | tee -a "${LOG_DIR}/run7.log"

# Benchmark baseline (run6_shard_ad) prima di iniziare
echo "" | tee -a "${LOG_DIR}/run7.log"
echo "Eseguo benchmark baseline su run6_shard_ad prima del training..." | tee -a "${LOG_DIR}/run7.log"
if [[ "$DRY" != "--dry-run" ]]; then
  cp "${BASE_CKPT}" checkpoints/best.pt
  run_benchmark "checkpoints/run6_shard_ad.pt" "baseline-run6-ad"
fi

SHARDS=(
  "data/wiki_run7_s1.txt"
  "data/wiki_run7_s2.txt"
  "data/wiki_run7_s3.txt"
)
SHARD_NAMES=("s1_wiki" "s2_wiki" "s3_wiki")

for i in "${!SHARDS[@]}"; do
  SHARD="${SHARDS[$i]}"
  NAME="${SHARD_NAMES[$i]}"
  SHARD_NUM=$((i + 1))

  echo "" | tee -a "${LOG_DIR}/run7.log"
  echo "━━━ Shard ${SHARD_NUM}/3 — ${SHARD} — $(date -u '+%H:%M UTC') ━━━" | tee -a "${LOG_DIR}/run7.log"

  if [[ "$DRY" == "--dry-run" ]]; then
    echo "[DRY RUN] python3 train.py --data ${SHARD} --steps ${STEPS} ..."
    continue
  fi

  python3 train.py \
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

  # Backup checkpoint shard
  if [[ -f checkpoints/best.pt ]]; then
    cp checkpoints/best.pt "checkpoints/run7_shard_${NAME}.pt"
    echo "  Checkpoint salvato: checkpoints/run7_shard_${NAME}.pt" | tee -a "${LOG_DIR}/run7.log"
  fi

  # Benchmark dopo questo shard
  run_benchmark "checkpoints/run7_shard_${NAME}.pt" "run7-${NAME}"

  echo "  Shard ${SHARD_NUM} completato — $(date -u '+%H:%M UTC')" | tee -a "${LOG_DIR}/run7.log"
done

echo "" | tee -a "${LOG_DIR}/run7.log"
echo "=== Run #7 completato — $(date -u '+%Y-%m-%d %H:%M UTC') ===" | tee -a "${LOG_DIR}/run7.log"
echo "Log benchmark: ${BENCH_LOG}" | tee -a "${LOG_DIR}/run7.log"
echo "Checkpoints: checkpoints/run7_shard_{s1_wiki,s2_wiki,s3_wiki}.pt" | tee -a "${LOG_DIR}/run7.log"
echo "Prossimo: analisi benchmark, decisione sweet spot, avvio Run #8." | tee -a "${LOG_DIR}/run7.log"
