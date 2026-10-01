#!/usr/bin/env python3
"""
tools/export_db_corpus.py — Esporta testo da PostgreSQL / MongoDB in formato corpus SLM

Legge la configurazione da un file YAML esterno per non esporre credenziali
nella history della shell o nei log.

Dipendenze:
    pip install psycopg2-binary pymongo pyyaml

Formato config (YAML):
    sources:
      - type: postgresql
        url: "postgresql://user:pass@host:5432/dbname"
        query: "SELECT body FROM email_inbox WHERE stato = 'accepted'"
        text_field: body        # colonna che contiene il testo
        min_chars: 100          # opzionale, default 0

      - type: mongodb
        url: "mongodb://user:pass@host:27017"
        database: docs_manager
        collection: documents
        text_field: content     # campo che contiene il testo
        filter: {"status": "active"}   # opzionale, filtro MongoDB
        min_chars: 200

Esempi:
    python3 tools/export_db_corpus.py --config private/hokus_export.yaml \\
        --out data/hokus_corpus.txt

    python3 tools/export_db_corpus.py --config private/pelliconi_export.yaml \\
        --out data/pelliconi_corpus.txt --append

    # dry-run: mostra solo statistiche senza scrivere
    python3 tools/export_db_corpus.py --config private/hokus_export.yaml --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# PULIZIA TESTO — coerente con il resto della pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ''
    text = str(text)
    text = text.replace(' ', ' ').replace('\xa0', ' ').replace('﻿', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [' '.join(l.split()) for l in text.split('\n')]
    text = '\n'.join(lines)
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text.strip()


def strip_html(text: str) -> str:
    """Rimuove tag HTML basilari — per email con body HTML."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, 'html.parser').get_text(separator='\n')
    except ImportError:
        import re
        return re.sub(r'<[^>]+>', ' ', text)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────────

def export_postgresql(source: dict, out_file, min_chars: int) -> tuple[int, int]:
    try:
        import psycopg2
    except ImportError:
        print("  ERRORE: psycopg2 non installato.  pip install psycopg2-binary")
        sys.exit(1)

    url  = source['url']
    sql  = source['query']
    col  = source.get('text_field', 'text')

    conn = psycopg2.connect(url)
    cur  = conn.cursor()
    cur.execute(sql)

    ok = skipped = 0
    for row in cur:
        text = row[0] if len(row) == 1 else row[cur.description.index((col,))]
        if text is None:
            skipped += 1
            continue

        text = str(text)
        if text.strip().startswith('<'):
            text = strip_html(text)
        text = clean_text(text)

        if len(text) < min_chars:
            skipped += 1
            continue

        if out_file:
            out_file.write(text + '\n\n')
        ok += 1

    cur.close()
    conn.close()
    return ok, skipped


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT MONGODB
# ─────────────────────────────────────────────────────────────────────────────

def export_mongodb(source: dict, out_file, min_chars: int) -> tuple[int, int]:
    try:
        from pymongo import MongoClient
    except ImportError:
        print("  ERRORE: pymongo non installato.  pip install pymongo")
        sys.exit(1)

    url        = source['url']
    db_name    = source['database']
    coll_name  = source['collection']
    field      = source.get('text_field', 'content')
    mongo_filter = source.get('filter', {})

    client = MongoClient(url)
    coll   = client[db_name][coll_name]

    ok = skipped = 0
    for doc in coll.find(mongo_filter, {field: 1}):
        text = doc.get(field)
        if text is None:
            skipped += 1
            continue

        text = str(text)
        text = clean_text(text)

        if len(text) < min_chars:
            skipped += 1
            continue

        if out_file:
            out_file.write(text + '\n\n')
        ok += 1

    client.close()
    return ok, skipped


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

_EXPORTERS = {
    'postgresql': export_postgresql,
    'postgres':   export_postgresql,
    'pg':         export_postgresql,
    'mongodb':    export_mongodb,
    'mongo':      export_mongodb,
}


def run_export(config: dict, out_path: Path, append: bool, dry_run: bool):
    sources = config.get('sources', [])
    if not sources:
        print("  ERRORE: nessuna sorgente definita nel config")
        sys.exit(1)

    mode = 'a' if append else 'w'
    t0   = time.time()

    print(f"\n  export_db_corpus.py — {len(sources)} sorgente(i)")
    print(f"  Output : {out_path}  ({'append' if append else 'overwrite'})")
    if dry_run:
        print("  Modalità: DRY-RUN (nessun file scritto)\n")
    print()

    total_ok = total_skip = 0

    with (open(out_path, mode, encoding='utf-8') if not dry_run else open(os.devnull, 'w')) as f:
        for i, src in enumerate(sources, 1):
            src_type = src.get('type', '').lower()
            fn = _EXPORTERS.get(src_type)
            if fn is None:
                print(f"  [{i}] TIPO SCONOSCIUTO: {src_type!r} — saltato")
                continue

            min_chars = src.get('min_chars', 0)
            label = f"{src_type}:{src.get('query', src.get('collection', ''))[:60]}"
            print(f"  [{i}] {label} ...")

            try:
                ok, skip = fn(src, None if dry_run else f, min_chars)
            except Exception as e:
                print(f"       ERRORE: {e}")
                continue

            total_ok   += ok
            total_skip += skip
            print(f"       {ok:,} documenti esportati  {skip:,} saltati")

    elapsed = int(time.time() - t0)
    print(f"\n  Totale : {total_ok:,} documenti  {total_skip:,} saltati")
    if not dry_run:
        size_mb = out_path.stat().st_size / 1024**2
        print(f"  File   : {out_path}  ({size_mb:.1f} MB)")
    print(f"  Durata : {elapsed}s")

    if not dry_run and total_ok > 0:
        print(f"\n  Prossimo step: filtraggio qualità")
        print(f"    python3 tools/filter_corpus.py {out_path} \\")
        print(f"        --checkpoint checkpoints/best.pt --dedup \\")
        print(f"        --out {out_path.stem}_clean.txt")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Esporta testo da PostgreSQL / MongoDB in formato corpus SLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  python3 tools/export_db_corpus.py --config private/hokus_export.yaml \\
      --out data/hokus_corpus.txt

  python3 tools/export_db_corpus.py --config private/pelliconi_export.yaml \\
      --out data/pelliconi_corpus.txt --append

  python3 tools/export_db_corpus.py --config private/hokus_export.yaml --dry-run
""")
    parser.add_argument('--config',   required=True,
                        help='file YAML con configurazione sorgenti e credenziali')
    parser.add_argument('--out',      required=True,
                        help='file corpus di output (.txt)')
    parser.add_argument('--append',   action='store_true',
                        help='aggiunge al file esistente invece di sovrascrivere')
    parser.add_argument('--dry-run',  action='store_true', dest='dry_run',
                        help='mostra statistiche senza scrivere file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"  ERRORE: config non trovato: {args.config}")
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("  ERRORE: pyyaml non installato.  pip install pyyaml")
        sys.exit(1)

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    run_export(config, out_path, args.append, args.dry_run)


if __name__ == '__main__':
    main()
