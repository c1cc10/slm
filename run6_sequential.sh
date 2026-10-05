#!/bin/bash
# Run #6 — Training sequenziale su Vast.ai
# Obiettivo: Chinchilla parity (~917M token su 7 shard × 8000 step × batch 32)
#
# SETUP prima di eseguire su Vast.ai:
#   1. rsync locale → istanza:
#      rsync -avz -e "ssh -i ~/.ssh/vastai_slm" \
#        checkpoints/run4_best.pt \
#        data/shards/shard_aa.txt.bpe-spm.v16000.tokens.pt \
#        data/shards/shard_ab.txt.bpe-spm.v16000.tokens.pt \
#        data/shards/shard_ac.txt.bpe-spm.v16000.tokens.pt \
#        data/shards/shard_ad.txt.bpe-spm.v16000.tokens.pt \
#        data/shards/shard_ae.txt.bpe-spm.v16000.tokens.pt \
#        data/shards/shard_af.txt.bpe-spm.v16000.tokens.pt \
#        data/shards/shard_ag.txt.bpe-spm.v16000.tokens.pt \
#        checkpoints/tokenizer.model \
#        train.py attention.py transformer.py model.py \
#        root@<HOST>:<PORT>:/workspace/slm/
#   2. pip3 install torch sentencepiece (se non presenti)
#   3. tmux new -s run6
#   4. bash run6_sequential.sh 2>&1 | tee logs/run6_sequential.log

set -e

LOG="logs/run6_sequential.log"
mkdir -p logs checkpoints

# Punto di partenza: pesi Run #4
echo "=== Run #6 — training sequenziale ===" | tee -a "$LOG"
echo "Inizio: $(date)" | tee -a "$LOG"
cp checkpoints/run4_best.pt checkpoints/best.pt
echo "Checkpoint iniziale: run4_best.pt → best.pt (val_loss Run #4 = 3.5861)" | tee -a "$LOG"

for SHARD in aa ab ac ad ae af ag; do
    echo "" | tee -a "$LOG"
    echo "=== SHARD ${SHARD} — $(date) ===" | tee -a "$LOG"

    # I file .pt pre-tokenizzati vengono rilevati automaticamente da train.py
    # Il flag --reset-best imposta best_val_loss=inf al resume: il primo val checkpoint
    # sovrascrive best.pt con i pesi dello shard corrente, propagando la catena.
    python3 train.py \
        --data "data/shards/shard_${SHARD}.txt" \
        --steps 8000 \
        --batch 32 \
        --resume \
        --reset-best \
        --tokenizer bpe-spm \
        --preset medium \
        --rope \
        --device cuda \
        2>&1 | tee -a "$LOG"

    echo "Shard ${SHARD} completato — $(date)" | tee -a "$LOG"
    cp checkpoints/best.pt "checkpoints/run6_shard_${SHARD}.pt"
    echo "Checkpoint salvato: checkpoints/run6_shard_${SHARD}.pt" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Run #6 completato — $(date) ===" | tee -a "$LOG"
echo "Checkpoint finale: checkpoints/best.pt" | tee -a "$LOG"

# DOPO il training — download dal Vast.ai:
#   rsync -avz -e "ssh -i ~/.ssh/vastai_slm" \
#     root@<HOST>:<PORT>:/workspace/slm/checkpoints/best.pt \
#     root@<HOST>:<PORT>:/workspace/slm/checkpoints/run6_shard_*.pt \
#     root@<HOST>:<PORT>:/workspace/slm/logs/run6_sequential.log \
#     checkpoints/ logs/
