---
name: vastai-slm
description: Ciclo di vita completo istanza Vast.ai per SLM training — provisioning interattivo, trasferimento checkpoint, avvio training, monitoring, rimpatrio e cleanup.
---

# Vast.ai SLM — Skill interattiva

Sei un agente che gestisce il ciclo di vita di un'istanza Vast.ai per il progetto SLM. Ogni operazione richiede conferma esplicita prima di essere eseguita. Segui questa skill dall'inizio alla fine, passo per passo.

---

## Contesto progetto (percorsi fissi)

- Root progetto: directory di lavoro corrente
- Checkpoint: `checkpoints/best.pt`
- Corpus: `data/corpus_bpe.txt`
- Cache token: `data/corpus_bpe.txt.bpe-spm.v16000.tokens.pt`
- Comando training: `python3 train.py --tokenizer bpe-spm --data data/corpus_bpe.txt --preset medium --steps 30000 --resume`
- **File stato istanza**: `.vastai-instance` — contiene solo l'ID intero dell'istanza corrente
- **File API key**: `.vastai-apikey` — contiene solo la chiave API Vast.ai

---

## FASE 0 — Prerequisiti

### 0a. Installa vastai CLI se mancante
```bash
vastai --version 2>/dev/null || pip install vastai
```

### 0b. Gestione API key
Cerca la chiave in quest'ordine:

1. Variabile d'ambiente: `echo $VAST_API_KEY`
2. File locale: leggi `.vastai-apikey`
3. Se non trovata da nessuna parte: chiedi all'utente
   - "Inserisci la tua API key Vast.ai (la trovi su vast.ai/account):"
   - Salva la chiave in `.vastai-apikey`
   - Aggiungi `.vastai-apikey` a `.gitignore` (crea il file se non esiste, non duplicare se già presente)

Configura la chiave:
```bash
vastai set api-key <KEY>
```

Verifica che funzioni:
```bash
vastai show user
```
Se fallisce con errore di autenticazione, la chiave è sbagliata — chiedi di reinserirla.

---

## FASE 1 — Rileva stato

Leggi `.vastai-instance`.

**Se il file NON esiste** → vai al menu NESSUNA ISTANZA.

**Se il file esiste**, recupera l'ID e controlla:
```bash
vastai instances --raw
```
Cerca l'ID nel JSON risultante.

- Trovata con `status=running` o `actual_status=running` → vai al menu ISTANZA ATTIVA
- Trovata ma non running → mostra avviso "Istanza trovata ma non attiva" e chiedi se procedere con destroy o ignorare
- Non trovata → cancella `.vastai-instance`, vai al menu NESSUNA ISTANZA

---

## MENU: NESSUNA ISTANZA

Mostra:
```
─────────────────────────────────────
  SLM · Vast.ai — nessuna istanza attiva

  1)  Cerca e noleggia istanza GPU
  2)  Inserisci ID istanza esistente
  3)  Esci
─────────────────────────────────────
```

Scegli l'azione in base alla risposta dell'utente.

---

## MENU: ISTANZA ATTIVA

Recupera SSH URL:
```bash
vastai ssh-url <ID>
```

Mostra stato e menu:
```
─────────────────────────────────────
  SLM · Vast.ai — istanza #<ID>
  GPU:   <GPU_NAME>
  Costo: $<DPH>/ora
  SSH:   ssh -p <PORTA> root@<HOST>
─────────────────────────────────────

  1)  Trasferisci file e avvia training
  2)  Mostra log training (ultimi 50 righe)
  3)  Rimpatria checkpoint → Mac
  4)  Distruggi istanza
  5)  Esci
─────────────────────────────────────
```

---

## AZIONE: PROVISIONING

### P1. Cerca istanze

```bash
vastai search offers --type=on-demand "gpu_name=RTX_4090 cuda_vers>=12.0 disk_space>=29" -o dph_base --raw
```

Mostra le prime 5 in tabella formattata:
```
 ID       GPU          VRAM   RAM    $/ora   Affid.  CUDA
 ───────────────────────────────────────────────────────
 1234567  RTX 4090     24GB   64GB   0.42    98%     12.1
 ...
```

Chiedi: "Quale ID vuoi noleggiare? (o 'annulla')"

### P2. Seleziona immagine Docker

Chiedi:
```
Immagine Docker:
  1)  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime  (consigliata)
  2)  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
  3)  Inserisci URL immagine personalizzata
```

### P3. Conferma e crea

Mostra riepilogo:
```
  Istanza:  #<OFFER_ID>
  GPU:      RTX 4090 — $<PREZZO>/ora
  Immagine: <IMMAGINE>
  Disk:     30 GB

  Confermi il noleggio? (s/n)
```

Se confermato:
```bash
vastai create instance <OFFER_ID> --image <IMMAGINE> --disk 30 --raw
```

Estrai `new_contract` dal JSON — è l'ID istanza. Salvalo in `.vastai-instance`.

Mostra: "Istanza #<ID> creata. Attendo che diventi Running..."

### P4. Attendi Running

Ogni 20 secondi per massimo 5 minuti:
```bash
vastai instances --raw
```
Controlla `actual_status` per l'ID. Mostra avanzamento con `.` o percentuale. Quando Running → continua a TRANSFER.

---

## AZIONE: TRANSFER

### T1. Estrai HOST e PORTA

```bash
vastai ssh-url <ID>
```
Output formato: `ssh -p 12345 root@ssh4.vast.ai`
Estrai PORTA=`12345` e HOST=`ssh4.vast.ai` (o IP).

### T2. Verifica checkpoint locale

```bash
python3 train.py --generate "Il modello"
```
Controlla che mostri `Step:` e `val_loss:` senza errori. Se fallisce, avverti l'utente e chiedi se procedere.

### T3. Piano trasferimento

Mostra file che verranno trasferiti con dimensioni:
```bash
ls -lh train.py model.py attention.py transformer.py
ls -lh checkpoints/best.pt
ls -lh data/corpus_bpe.txt
ls -lh "data/corpus_bpe.txt.bpe-spm.v16000.tokens.pt" 2>/dev/null
```

Chiedi: "Procedo con il trasferimento? (s/n)"

### T4. Esegui trasferimento

```bash
# Directory remote
ssh -o StrictHostKeyChecking=no -p <PORTA> root@<HOST> "mkdir -p ~/slm/{data,checkpoints}"

# Codice
rsync -avz -e "ssh -p <PORTA> -o StrictHostKeyChecking=no" \
  train.py model.py attention.py transformer.py \
  root@<HOST>:~/slm/

# Checkpoint
rsync -avz -e "ssh -p <PORTA> -o StrictHostKeyChecking=no" \
  checkpoints/best.pt root@<HOST>:~/slm/checkpoints/

# Corpus
rsync -avz --progress -e "ssh -p <PORTA> -o StrictHostKeyChecking=no" \
  data/corpus_bpe.txt root@<HOST>:~/slm/data/

# Cache token (se esiste)
rsync -avz -e "ssh -p <PORTA> -o StrictHostKeyChecking=no" \
  "data/corpus_bpe.txt.bpe-spm.v16000.tokens.pt" \
  root@<HOST>:~/slm/data/ 2>/dev/null || echo "cache assente, verrà rigenerata"
```

### T5. Setup remoto

```bash
ssh -p <PORTA> root@<HOST> "pip install sentencepiece"
ssh -p <PORTA> root@<HOST> "cd ~/slm && python3 train.py --generate 'Il modello'"
```

Mostra output. Se il checkpoint viene caricato correttamente, chiedi:
"Avvio il training? (s/n)"

---

## AZIONE: AVVIO TRAINING

```bash
ssh -p <PORTA> root@<HOST> \
  "cd ~/slm && tmux new-session -d -s slm 'python3 train.py --tokenizer bpe-spm --data data/corpus_bpe.txt --preset medium --steps 30000 --resume 2>&1 | tee train.log'"
```

Attendi 15 secondi, poi mostra i primi log:
```bash
ssh -p <PORTA> root@<HOST> "tail -20 ~/slm/train.log 2>/dev/null || vastai logs <ID>"
```

Ricorda all'utente:
```
Training avviato in tmux (sessione: slm).

Per connetterti manualmente:
  ssh -p <PORTA> root@<HOST>
  tmux attach -t slm

L'istanza costa $<DPH>/ora. Ricordati di distruggerla dopo il rimpatrio.
```

---

## AZIONE: LOG

```bash
vastai logs <ID>
```
oppure, se il training scrive su file:
```bash
ssh -p <PORTA> root@<HOST> "tail -50 ~/slm/train.log"
```

---

## AZIONE: RIMPATRIO

### R1. Mostra checkpoint remoto

```bash
ssh -p <PORTA> root@<HOST> "python3 ~/slm/train.py --generate 'Il modello'"
```

Mostra step e val_loss raggiunta su EC2.

### R2. Trasferisci checkpoint → Mac

```bash
rsync -avz -e "ssh -p <PORTA> -o StrictHostKeyChecking=no" \
  root@<HOST>:~/slm/checkpoints/best.pt \
  checkpoints/best.pt
```

### R3. Verifica locale

```bash
python3 train.py --generate "Il modello"
```

Confronta step e val_loss prima/dopo. Mostra il confronto.

### R4. Chiedi cosa fare

```
Checkpoint rimpatriato con successo.

  Vuoi distruggere l'istanza Vast.ai? (s/n)
  (costo attuale: $<DPH>/ora — ogni ora in più costa $<DPH>)
```

Se sì → AZIONE: DESTROY

---

## AZIONE: DESTROY

```bash
vastai destroy instance <ID>
```

Cancella `.vastai-instance`.

Mostra:
```
Istanza #<ID> distrutta. Nessun costo in corso.
Checkpoint locale verificato a step <STEP>, val_loss <LOSS>.
```

---

## Note operative

- **StrictHostKeyChecking=no**: necessario perché ogni istanza Vast.ai ha una host key diversa. Non è un rischio in questo contesto (istanza privata appena creata).
- **tmux**: il training sopravvive alla disconnessione SSH. Per riconnetterti manualmente: `ssh -p <PORTA> root@<HOST>` poi `tmux attach -t slm`.
- **Costo idle**: Vast.ai addebita anche quando l'istanza è ferma. Dopo il rimpatrio, distruggi sempre.
- **Interruptible vs On-Demand**: questa skill usa solo On-Demand. Le istanze Interruptible costano meno ma possono essere terminate dal provider a metà training — il checkpoint viene perso se non era stato scritto nell'ultimo intervallo di eval (ogni 500 step).
