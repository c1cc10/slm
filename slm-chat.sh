#!/usr/bin/env bash
# Avvia SLM Studio — llama-server + HTTP server per la chat UI
# Uso: bash slm-chat.sh [modello.gguf]

set -euo pipefail

cd "$(dirname "$0")"

LLAMA_SERVER="/opt/homebrew/bin/llama-server"
DEFAULT_MODEL="slm-run7-s3-q8_0.gguf"
MODEL="${1:-${DEFAULT_MODEL}}"
LLM_PORT=8080
HTTP_PORT=8081
CHAT_URL="http://localhost:${HTTP_PORT}/slm-chat.html"

# ── Controllo modello ──
if [[ ! -f "${MODEL}" ]]; then
  echo "Errore: modello non trovato — ${MODEL}"
  echo "Modelli GGUF disponibili:"
  ls *.gguf 2>/dev/null | sed 's/^/  /' || echo "  (nessuno)"
  exit 1
fi

# ── llama-server ──
if curl -sf "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
  echo "llama-server già attivo su porta ${LLM_PORT}"
else
  echo "Avvio llama-server per: ${MODEL}"
  "${LLAMA_SERVER}" \
    -m "${MODEL}" \
    --port "${LLM_PORT}" \
    --ctx-size 2048 \
    -ngl 99 \
    --log-disable \
    2>/dev/null &
  LLM_PID=$!
  echo "PID llama-server: ${LLM_PID}"

  printf "Attesa server"
  for i in $(seq 1 40); do
    sleep 1
    printf "."
    if curl -sf "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
      echo " pronto!"
      break
    fi
    if [[ $i -eq 40 ]]; then
      echo ""
      echo "Errore: server non risponde dopo 40s — controlla il modello."
      kill "${LLM_PID}" 2>/dev/null || true
      exit 1
    fi
  done
fi

# ── HTTP server per la chat (evita CORS da file://) ──
pkill -f "python3 -m http.server ${HTTP_PORT}" 2>/dev/null || true
sleep 0.3
python3 -m http.server "${HTTP_PORT}" --bind 127.0.0.1 > /dev/null 2>&1 &
HTTP_PID=$!

echo "──────────────────────────────────────────────"
echo "  Chat UI: ${CHAT_URL}"
echo "  Modello: ${MODEL}"
echo "  llama-server: http://localhost:${LLM_PORT}"
echo "──────────────────────────────────────────────"
sleep 0.5

# Apri nel browser predefinito
open "${CHAT_URL}"

echo "Premi Ctrl+C per fermare l'HTTP server."
echo "(llama-server rimane attivo, fermalo con: pkill -f llama-server)"

cleanup() {
  echo ""
  echo "Arresto HTTP server..."
  kill "${HTTP_PID}" 2>/dev/null || true
  echo "Fatto. llama-server rimane su porta ${LLM_PORT}."
  exit 0
}
trap cleanup INT TERM

wait "${HTTP_PID}"
