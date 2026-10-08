#!/usr/bin/env bash
# Benchmark Minerva — confronto con Run #7 shard 3
# Stessi 10 prompt del benchmark Run #7 locale.
# Usa llama-cli (llama.cpp) con i modelli GGUF già presenti.

set -euo pipefail

LOG_DIR="logs/run7"
BENCH_LOG="${LOG_DIR}/benchmark_minerva.log"
LLAMA_CLI="/opt/homebrew/bin/llama-cli"

MINERVA_350M="minerva-350m-base-v1.0.Q5_0.gguf"
MINERVA_3B="minerva-3b-instruct-v1.0.Q5_K_M.gguf"

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

mkdir -p "${LOG_DIR}"

echo "=== Benchmark Minerva — $(date '+%Y-%m-%d %H:%M') ===" | tee "${BENCH_LOG}"
echo "Confronto: Minerva-350M-base + Minerva-3B-instruct" | tee -a "${BENCH_LOG}"
echo "" | tee -a "${BENCH_LOG}"

# --- Minerva 350M base ---
echo "━━━ Minerva-350M-base ($(date '+%H:%M')) ━━━" | tee -a "${BENCH_LOG}"
echo "Modello: ${MINERVA_350M}" | tee -a "${BENCH_LOG}"
echo "" | tee -a "${BENCH_LOG}"

for PROMPT in "${BENCHMARK_PROMPTS[@]}"; do
  echo "--- PROMPT: ${PROMPT} ---" | tee -a "${BENCH_LOG}"
  echo "────────────────────────────────────────────────────────────" | tee -a "${BENCH_LOG}"
  "${LLAMA_CLI}" \
    --model "${MINERVA_350M}" \
    --prompt "${PROMPT}" \
    --n-predict 300 \
    --temp 0.8 \
    --top-k 50 \
    --top-p 0.95 \
    --repeat-penalty 1.1 \
    --no-display-prompt \
    --log-disable \
    2>/dev/null | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
  echo "────────────────────────────────────────────────────────────" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
done

echo "━━━ FINE Minerva-350M-base ━━━" | tee -a "${BENCH_LOG}"
echo "" | tee -a "${BENCH_LOG}"

# --- Minerva 3B instruct ---
echo "━━━ Minerva-3B-instruct ($(date '+%H:%M')) ━━━" | tee -a "${BENCH_LOG}"
echo "Modello: ${MINERVA_3B}" | tee -a "${BENCH_LOG}"
echo "" | tee -a "${BENCH_LOG}"

for PROMPT in "${BENCHMARK_PROMPTS[@]}"; do
  echo "--- PROMPT: ${PROMPT} ---" | tee -a "${BENCH_LOG}"
  echo "────────────────────────────────────────────────────────────" | tee -a "${BENCH_LOG}"
  # Minerva 3B è instruct → formato [INST]...[/INST]
  INST_PROMPT="[INST] Continua il seguente testo in italiano: ${PROMPT} [/INST]"
  "${LLAMA_CLI}" \
    --model "${MINERVA_3B}" \
    --prompt "${INST_PROMPT}" \
    --n-predict 300 \
    --temp 0.7 \
    --top-k 40 \
    --top-p 0.9 \
    --repeat-penalty 1.1 \
    --no-display-prompt \
    --log-disable \
    2>/dev/null | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
  echo "────────────────────────────────────────────────────────────" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
done

echo "━━━ FINE Minerva-3B-instruct ━━━" | tee -a "${BENCH_LOG}"
echo "" | tee -a "${BENCH_LOG}"

echo "=== Benchmark Minerva completato — $(date '+%Y-%m-%d %H:%M') ===" | tee -a "${BENCH_LOG}"
echo "Log salvato: ${BENCH_LOG}"
