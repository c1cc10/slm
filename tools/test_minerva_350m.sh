#!/usr/bin/env zsh
# Test Minerva 350M — stessi prompt del confronto shard Run #6
# Parametri identici all'analisi semantica (n_predict=100, temp=0.8, top_p=0.9, repeat_penalty=1.3, seed=42)
# Avvia prima: llama-server --model minerva-350m-base-v1.0.Q5_0.gguf --port 8081 --ctx-size 2048 --no-jinja

PORT=8081
MODEL="Minerva-350M-base"
OUT_DIR="$(dirname "$0")/../private"
OUT_FILE="$OUT_DIR/test_minerva_350m_$(date +%Y%m%d_%H%M).txt"

mkdir -p "$OUT_DIR"

PROMPTS=(
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

DOMAINS=(
  "enciclopedico/letterario"
  "tecnico/informatico"
  "poetico"
  "letterario"
  "storico"
  "musicale/culturale"
  "scientifico"
  "pratico/ricette"
  "geografico/regionale"
  "nome raro (byte token test)"
)

echo "=== $MODEL — Test comparativo shard Run #6 ===" | tee "$OUT_FILE"
echo "Data: $(date)" | tee -a "$OUT_FILE"
echo "Parametri: n_predict=100, temp=0.8, top_p=0.9, repeat_penalty=1.3, seed=42" | tee -a "$OUT_FILE"
echo "" | tee -a "$OUT_FILE"

# Verifica server attivo
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
  echo "ERRORE: llama-server non raggiungibile su porta $PORT"
  echo "Avvia: llama-server --model minerva-350m-base-v1.0.Q5_0.gguf --port $PORT --ctx-size 2048 --no-jinja"
  exit 1
fi

for i in {1..${#PROMPTS[@]}}; do
  PROMPT="${PROMPTS[$i]}"
  DOMAIN="${DOMAINS[$i]}"

  echo "──────────────────────────────────────────" | tee -a "$OUT_FILE"
  echo "[$i] DOMINIO: $DOMAIN" | tee -a "$OUT_FILE"
  echo "PROMPT: \"$PROMPT\"" | tee -a "$OUT_FILE"
  echo "" | tee -a "$OUT_FILE"

  resp=$(curl -s http://localhost:$PORT/completion \
    -H "Content-Type: application/json" \
    -d "{
      \"prompt\": \"$PROMPT\",
      \"n_predict\": 100,
      \"temperature\": 0.8,
      \"top_k\": 40,
      \"top_p\": 0.9,
      \"repeat_penalty\": 1.3,
      \"repeat_last_n\": 64,
      \"seed\": 42
    }")

  OUTPUT=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('content','ERROR: '+str(d)))" 2>/dev/null)
  echo "$OUTPUT" | tee -a "$OUT_FILE"
  echo "" | tee -a "$OUT_FILE"
done

echo "══════════════════════════════════════════" | tee -a "$OUT_FILE"
echo "Output salvato in: $OUT_FILE"
