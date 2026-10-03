#!/usr/bin/env bash
# train_chunked.sh — training su corpus grande spezzato in shard
#
# Problema: corpus_run5.txt (8.29 GB) non tokenizza interamente in 8 GB RAM.
# Soluzione: split in N shard da ~1.2 GB → tokenizzazione + training sequenziale.
#            Ogni shard usa --resume: il modello accumula apprendimento shard per shard.
#
# Uso:
#   bash tools/train_chunked.sh                      # default (vedi parametri sotto)
#   bash tools/train_chunked.sh --shards 5           # forza N shard
#   bash tools/train_chunked.sh --steps-total 56000  # step totali (default)
#   bash tools/train_chunked.sh --skip-split         # usa shard già generati
#
# Log: /tmp/run5.log  (append)
# Shard: data/shards/shard_*

set -euo pipefail

# ── Parametri configurabili ────────────────────────────────────────────────────
CORPUS="data/corpus_run5.txt"
SHARD_DIR="data/shards"
N_SHARDS=7
STEPS_TOTAL=56000
LOG="/tmp/run5.log"
TOKENIZER="bpe-spm"
PRESET="medium"
SKIP_SPLIT=0

# ── Parsing argomenti ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards)       N_SHARDS="$2";      shift 2 ;;
        --steps-total)  STEPS_TOTAL="$2";   shift 2 ;;
        --skip-split)   SKIP_SPLIT=1;       shift   ;;
        *) echo "Argomento non riconosciuto: $1"; exit 1 ;;
    esac
done

STEPS_PER_SHARD=$(( STEPS_TOTAL / N_SHARDS ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "  SLM — training a shard su corpus grande"
echo "  Corpus    : $CORPUS"
echo "  Shard     : $N_SHARDS  (dir: $SHARD_DIR)"
echo "  Step/shard: $STEPS_PER_SHARD  (totale: $STEPS_TOTAL)"
echo "  Log       : $LOG"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Split del corpus ───────────────────────────────────────────────────────────
if [[ $SKIP_SPLIT -eq 0 ]]; then
    if [[ ! -f "$CORPUS" ]]; then
        echo "ERRORE: corpus non trovato: $CORPUS"
        exit 1
    fi

    echo "Pulizia shard precedenti e creazione directory..."
    rm -rf "$SHARD_DIR"
    mkdir -p "$SHARD_DIR"

    TOTAL_LINES=$(wc -l < "$CORPUS")
    LINES_PER_SHARD=$(( (TOTAL_LINES + N_SHARDS - 1) / N_SHARDS ))
    echo "Righe totali: $TOTAL_LINES  →  ~$LINES_PER_SHARD righe/shard"
    echo "Split in corso (split -l)..."

    split -l "$LINES_PER_SHARD" "$CORPUS" "$SHARD_DIR/shard_"

    # Rinomina con estensione .txt per chiarezza
    for f in "$SHARD_DIR"/shard_*; do
        [[ "$f" == *.txt ]] && continue
        mv "$f" "${f}.txt"
    done

    echo "Shard generati:"
    ls -lh "$SHARD_DIR"/shard_*.txt
    echo ""
else
    echo "--skip-split: uso shard esistenti in $SHARD_DIR"
    echo ""
fi

# ── Loop tokenizzazione + training ────────────────────────────────────────────
SHARDS=("$SHARD_DIR"/shard_*.txt)
ACTUAL_N=${#SHARDS[@]}

if [[ $ACTUAL_N -eq 0 ]]; then
    echo "ERRORE: nessun shard trovato in $SHARD_DIR"
    exit 1
fi

echo "Shard trovati: $ACTUAL_N"
echo "Step per shard: $STEPS_PER_SHARD"
echo ""

for i in "${!SHARDS[@]}"; do
    SHARD="${SHARDS[$i]}"
    SHARD_NUM=$(( i + 1 ))
    SHARD_SIZE=$(du -sh "$SHARD" | cut -f1)

    echo "════════════════════════════════════════════════════════════════"
    echo "  Shard $SHARD_NUM / $ACTUAL_N  —  $SHARD  ($SHARD_SIZE)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"

    python3 -u train.py \
        --tokenizer "$TOKENIZER" \
        --data      "$SHARD" \
        --resume \
        --preset    "$PRESET" \
        --rope \
        --batch     8 \
        --steps     "$STEPS_PER_SHARD" \
        2>&1 | tee -a "$LOG"

    EXIT_CODE=${PIPESTATUS[0]}
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "ERRORE: training su shard $SHARD_NUM fallito (exit $EXIT_CODE). Interruzione."
        exit $EXIT_CODE
    fi

    echo ""
    echo "  Shard $SHARD_NUM completato."
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "  Training completato su tutti i shard."
echo "  Step totali processati: $(( ACTUAL_N * STEPS_PER_SHARD ))"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
