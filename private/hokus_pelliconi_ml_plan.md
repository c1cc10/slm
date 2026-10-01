# ML Applications — Hokus / Pelliconi
## Documento interno — non pubblicare, non versionare in git

**Stato:** sospeso — riprende dopo completamento modello base SLM  
**Data:** 2026-10-01

---

## Contesto

Due progetti aziendali contengono corpus testuali strutturati e già etichettati che
possono alimentare applicazioni ML leggere costruite sul modello SLM:

- **Hokus** (middleware `amel-ia`): MGA quotation manager per pratiche assicurative
- **Pelliconi / AIKnow** (repo `pelliconi1/*`): piattaforma multi-tenant di gestione documenti

---

## Use case 1 — Selettore di pertinenza email (Hokus)

### Obiettivo

Filtro pre-LLM: dato un'email in ingresso, valutare se è pertinente al dominio
Hokus prima di passarla alla pipeline classificazione (Fase 1 + Fase 2 con OpenAI).
Risparmio stimato: 20–30% delle chiamate API se il sistema filtra correttamente
OOO, spam e email fuori dominio.

### Fonte dati

**Tabella PostgreSQL:** `email_inbox` (schema `hokus`)

| Campo | Contenuto | Uso |
|---|---|---|
| `body` | Testo completo email (HTML o plain) | Corpus training |
| `stato` | Stato lavorazione (`accepted`, `rejected`, `below_threshold`, ...) | Label |
| `prodotto_accepted` | Prodotto confermato dall'UW | Ground truth positivo |
| `ramo_detected` | Ramo classificato in Fase 1 | Feature aggiuntiva |

**Email aggiuntive:** `news` / `rassegna` (articoli italiano assicurativo) — ulteriore
corpus di dominio senza etichette.

### Approccio tecnico

**Se volume > 500 email classificate:**  
SFT classificatore binario (pertinente / non pertinente) sopra encoder SLM.  
Input: testo email → score 0–1.

**Se volume < 500 email classificate:**  
PPL scorer non-supervisionato: fine-tune Run #3 solo sulle email confermate (`stato = 'accepted'`),
poi usa `eval_ppl.py` per score. PPL bassa = in-domain, PPL alta = fuori dominio.

### Limitazione nota

Il modello riconosce il vocabolario di dominio (ramo, pratica, broker, UW, premio)
ma non la semantica. Un'email pertinente in linguaggio generico ("mi serve supporto
per la mia richiesta") può avere PPL alta nonostante sia rilevante. Il filtro è
complementare, non sostitutivo.

### Posizione nel pipeline email esistente

```
[email in ingresso]
       ↓
[PPL scorer locale — ms, CPU]     ← NEW: pre-filtro economico
       ↓ (PPL < soglia)
[Fase 1 — classify_inbox_record]  ← LLM call esistente
       ↓
[Fase 2 — reclassify_prodotto]    ← LLM call esistente
```

---

## Use case 2 — Glossario di dominio (Pelliconi / Docs Manager)

### Obiettivo

Dizionario terminologico tecnico derivato dai documenti reali caricati dai tenant.
Utilizzo: riferimento interno, seed per future pipeline RAG specializzate.

### Fonte dati

**MongoDB** — `docs-manager-be`, collection `documents` (una per tenant)

Ogni documento contiene già il testo estratto (Apache POI / PDFBox lato backend).
Contenuti tipici:
- Manuali tecnici macchinari (Pelliconi produce tappi metallici e accessori beverage)
- Procedure operative
- Schede tecniche
- Report di troubleshooting da SuppOrTech

**PostgreSQL** — `supportech-be`, tabella ticket / fault reports

Descrizioni guasti, soluzioni adottate, nomenclatura macchine e impianti.

### Approccio tecnico

Non è un task generativo: il modello SLM non è adatto a generare definizioni.
L'approccio corretto è **RAG terminologico** sul corpus estratto:

1. Export corpus da MongoDB → file testo flat (via `export_db_corpus.py`)
2. Ingestione nel RAG pipeline esistente del middleware (Milvus)
3. Query: "cosa significa X?" → retrieval chunk rilevanti → risposta con LLM

In alternativa: extraction pattern-based di coppie `termine: definizione` già
presenti nei manuali tecnici (stile dizionario inline).

---

## Tool necessario — export_db_corpus.py

Vedi `tools/export_db_corpus.py`.  
Legge da PostgreSQL e/o MongoDB usando config YAML esterno (credenziali mai in chiaro).  
Output: formato corpus SLM standard (documenti separati da `\n\n`).

**Config Hokus** (da creare in `private/`):
```yaml
# private/hokus_export.yaml — NON VERSIONARE
sources:
  - type: postgresql
    url: "postgresql://user:pass@host:5432/amel_ia"
    query: >
      SELECT body FROM hokus_email_inbox
      WHERE stato = 'accepted'
        AND body IS NOT NULL
        AND length(body) > 100
    text_field: body
    min_chars: 100

  - type: postgresql
    url: "postgresql://user:pass@host:5432/amel_ia"
    query: "SELECT content FROM hokus_news WHERE lang = 'it'"
    text_field: content
```

**Config Pelliconi** (da creare in `private/`):
```yaml
# private/pelliconi_export.yaml — NON VERSIONARE
sources:
  - type: mongodb
    url: "mongodb://user:pass@host:27017"
    database: "docs_manager_tenant1"
    collection: "documents"
    text_field: "content"
    filter: {"status": "active", "language": "it"}
    min_chars: 200
```

---

## Prerequisiti prima di riprendere

1. Sistema Hokus in produzione da almeno 3–6 mesi (volume email sufficiente)
2. Accordo con il cliente Pelliconi per uso dei documenti come corpus ML
3. Completamento modello base SLM (Run #4 con RoPE, Chinchilla-optimal)
4. Valutazione volume: `SELECT COUNT(*), AVG(length(body)) FROM hokus_email_inbox WHERE stato = 'accepted'`

---

## Note privacy

- I file `private/*.yaml` contengono credenziali: mai in git, mai in backup cloud
- Il corpus estratto (`data/hokus_*.txt`, `data/pelliconi_*.txt`) contiene dati aziendali: gitignore obbligatorio
- I checkpoint fine-tuned su questi corpus sono modelli addestrati su dati proprietari: storage locale only
