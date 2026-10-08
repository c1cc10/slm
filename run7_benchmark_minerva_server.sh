#!/usr/bin/env bash
# Benchmark Minerva via llama-server /completion endpoint
# Stessi 10 prompt del benchmark Run #7 locale.

set -euo pipefail

LOG_DIR="logs/run7"
BENCH_LOG="${LOG_DIR}/benchmark_minerva.log"
LLAMA_SERVER="/opt/homebrew/bin/llama-server"
PORT=8765

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
echo "" | tee -a "${BENCH_LOG}"

run_model() {
  local MODEL="$1"
  local LABEL="$2"
  local TEMP="${3:-0.8}"
  local PREFIX="${4:-}"   # prefix da anteporre al prompt (per instruct)
  local SUFFIX="${5:-}"   # suffix da aggiungere al prompt

  echo "━━━ ${LABEL} ($(date '+%H:%M')) ━━━" | tee -a "${BENCH_LOG}"
  echo "Modello: ${MODEL}" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"

  # Avvia server
  "${LLAMA_SERVER}" -m "${MODEL}" --port "${PORT}" --no-jinja --log-disable -c 2048 2>/dev/null &
  SERVER_PID=$!
  sleep 8

  for PROMPT in "${BENCHMARK_PROMPTS[@]}"; do
    FULL_PROMPT="${PREFIX}${PROMPT}${SUFFIX}"
    echo "--- PROMPT: ${PROMPT} ---" | tee -a "${BENCH_LOG}"
    echo "────────────────────────────────────────────────────────────" | tee -a "${BENCH_LOG}"
    curl -s "http://localhost:${PORT}/completion" \
      -H "Content-Type: application/json" \
      -d "{\"prompt\": \"${FULL_PROMPT}\", \"n_predict\": 300, \"temperature\": ${TEMP}, \"top_k\": 50, \"top_p\": 0.95, \"repeat_penalty\": 1.1, \"stream\": false}" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('content','ERROR'))" \
      | tee -a "${BENCH_LOG}"
    echo "" | tee -a "${BENCH_LOG}"
    echo "────────────────────────────────────────────────────────────" | tee -a "${BENCH_LOG}"
    echo "" | tee -a "${BENCH_LOG}"
  done

  kill "${SERVER_PID}" 2>/dev/null
  sleep 2
  echo "━━━ FINE ${LABEL} ━━━" | tee -a "${BENCH_LOG}"
  echo "" | tee -a "${BENCH_LOG}"
}

# Minerva 350M — base model, raw completion
run_model "${MINERVA_350M}" "Minerva-350M-base" 0.8 "" ""

# Minerva 3B — instruct model, formato [INST]...[/INST]
run_model "${MINERVA_3B}" "Minerva-3B-instruct" 0.7 "[INST] Continua questo testo in italiano: " " [/INST]"

echo "=== Benchmark Minerva completato — $(date '+%Y-%m-%d %H:%M') ===" | tee -a "${BENCH_LOG}"
echo "Log salvato: ${BENCH_LOG}"
