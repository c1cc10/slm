#!/usr/bin/env bash
# Benchmark locale Run #7 — 4 versioni × 10 prompt
# Eseguire DOPO aver scaricato i 3 checkpoint da Vast.ai.
#
# Versioni confrontate:
#   baseline: checkpoints/run6_shard_ad.pt   (CulturaX sweet spot, val 3.3524)
#   shard-s1: checkpoints/run7_shard_s1.pt   (+ wiki shard 1, 1.1GB)
#   shard-s2: checkpoints/run7_shard_s2.pt   (+ wiki shard 1+2, 2.2GB cumulative)
#   shard-s3: checkpoints/run7_shard_s3.pt   (+ wiki shard 1+2+3, 3.3GB cumulative)
#
# Uso: bash run7_benchmark_local.sh

set -euo pipefail

LOG_DIR="logs/run7"
BENCH_LOG="${LOG_DIR}/benchmark_local.log"
DEVICE="cpu"

mkdir -p "${LOG_DIR}"

CHECKPOINTS=(
  "checkpoints/run6_shard_ad.pt:baseline-run6-ad"
  "checkpoints/run7_shard_s1.pt:run7-shard-s1"
  "checkpoints/run7_shard_s2.pt:run7-shard-s2"
  "checkpoints/run7_shard_s3.pt:run7-shard-s3"
)

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

echo "=== Benchmark Run #7 locale — $(date '+%Y-%m-%d %H:%M') ===" | tee "${BENCH_LOG}"
echo "Confronto: baseline + 3 shard Wikipedia IT" | tee -a "${BENCH_LOG}"
echo "" | tee -a "${BENCH_LOG}"

for ENTRY in "${CHECKPOINTS[@]}"; do
  CKPT="${ENTRY%%:*}"
  LABEL="${ENTRY##*:}"

  if [[ ! -f "${CKPT}" ]]; then
    echo "SKIP ${LABEL}: ${CKPT} non trovato" | tee -a "${BENCH_LOG}"
    continue
  fi

  echo "━━━ ${LABEL} ($(date '+%H:%M')) ━━━" | tee -a "${BENCH_LOG}"
  echo "Checkpoint: ${CKPT}" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"

  for PROMPT in "${BENCHMARK_PROMPTS[@]}"; do
    echo "--- PROMPT: ${PROMPT} ---" | tee -a "${BENCH_LOG}"
    python3 -u train.py \
      --generate "${PROMPT}" \
      --checkpoint "${CKPT}" \
      --device "${DEVICE}" 2>/dev/null | tee -a "${BENCH_LOG}"
    echo "" | tee -a "${BENCH_LOG}"
  done

  echo "━━━ FINE ${LABEL} ━━━" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
done

echo "=== Benchmark completato — $(date '+%Y-%m-%d %H:%M') ===" | tee -a "${BENCH_LOG}"
echo "Log salvato: ${BENCH_LOG}"
