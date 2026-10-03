#!/usr/bin/env python3
"""
tools/fetch_hf_corpus.py — Streaming generico da HuggingFace verso corpus SLM

Supporta qualsiasi dataset HF con un campo testo, con filtro lingua opzionale.
Non scarica il dump completo: legge in streaming e scrive solo ciò che passa i criteri.

Dipendenze:
    pip install datasets

Esempi:
    # clean_mc4_it tiny (~10 GB sorgente, estrae fino a max-gb)
    python3 tools/fetch_hf_corpus.py \
        --dataset gsarti/clean_mc4_it --config tiny \
        --text-field text \
        --max-gb 2 --out data/mc4_it_raw.txt

    # Common Corpus — solo italiano
    python3 tools/fetch_hf_corpus.py \
        --dataset PleIAs/common_corpus \
        --text-field text --lang-field language --lang it \
        --max-gb 5 --out data/common_corpus_it_raw.txt

    # liberliber-cleaned (piccolo, scarica tutto)
    python3 tools/fetch_hf_corpus.py \
        --dataset mii-community/liberliber-cleaned \
        --text-field text \
        --out data/liberliber_it.txt

    # opus_books italiano
    python3 tools/fetch_hf_corpus.py \
        --dataset Helsinki-NLP/opus_books --config it \
        --text-field translation.it \
        --out data/opus_books_it.txt

    # dry-run: statistiche senza scrivere
    python3 tools/fetch_hf_corpus.py \
        --dataset gsarti/clean_mc4_it --config tiny \
        --text-field text --max-docs 5000 --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path


def resolve_field(doc: dict, field_path: str):
    """Legge un campo anche annidato (es. 'translation.it')."""
    parts = field_path.split('.')
    val = doc
    for p in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(p)
    return val


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


def fetch(args):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERRORE: datasets non installato.  pip install datasets")
        sys.exit(1)

    print(f"\n  fetch_hf_corpus.py")
    print(f"  Dataset : {args.dataset}" + (f"  config={args.config}" if args.config else ""))
    print(f"  Campo   : {args.text_field}" + (f"  lang_field={args.lang_field} lang={args.lang}" if args.lang else ""))
    print(f"  Limite  : {args.max_gb} GB / {args.max_docs:,} doc")
    print(f"  Output  : {args.out}" + ("  [DRY-RUN]" if args.dry_run else ""))
    print()

    load_kwargs = dict(split='train', streaming=True)
    if args.config:
        load_kwargs['name'] = args.config

    try:
        ds = load_dataset(args.dataset, **load_kwargs)
    except Exception as e:
        print(f"  ERRORE caricamento dataset: {e}")
        sys.exit(1)

    # Filtro lingua via .filter() nativo — sfrutta i metadati Parquet
    # per saltare gli shard non italiani senza scorrere tutto il dataset
    if args.lang and args.lang_field:
        lang_field = args.lang_field
        lang_val   = args.lang
        ds = ds.filter(lambda row: row.get(lang_field) == lang_val)
        args.lang = None   # disabilita il controllo manuale nel loop

    if args.out:
        os.makedirs(Path(args.out).parent, exist_ok=True)

    max_bytes = int(args.max_gb * 1024**3)
    total_bytes = 0
    total_docs = 0
    skipped_lang = 0
    skipped_empty = 0
    t_start = time.time()

    out_path = os.devnull if args.dry_run else args.out
    mode = 'a' if args.append else 'w'

    with open(out_path, mode, encoding='utf-8') as f_out:
        for row in ds:
            text = resolve_field(row, args.text_field)

            if not text:
                skipped_empty += 1
                continue

            if args.lang and args.lang_field:
                lang = resolve_field(row, args.lang_field)
                if lang != args.lang:
                    skipped_lang += 1
                    continue

            text = clean_text(str(text))
            if len(text) < args.min_chars:
                skipped_empty += 1
                continue

            encoded = text.encode('utf-8')
            f_out.write(text + '\n\n')
            total_bytes += len(encoded)
            total_docs += 1

            if total_docs % 10_000 == 0:
                elapsed = int(time.time() - t_start)
                gb = total_bytes / 1024**3
                speed = total_docs / max(elapsed, 1) / 1000
                print(f"\r  {total_docs:>8,} doc  {gb:.3f}/{args.max_gb} GB"
                      f"  {speed:.1f}k doc/s  {elapsed}s   ",
                      end='', flush=True)

            if total_bytes >= max_bytes:
                break
            if total_docs >= args.max_docs:
                break

    elapsed = int(time.time() - t_start)
    gb = total_bytes / 1024**3

    print(f"\n\n  Completato:")
    print(f"    Documenti scritti : {total_docs:,}")
    print(f"    Saltati (vuoti)   : {skipped_empty:,}")
    if args.lang:
        print(f"    Saltati (lingua)  : {skipped_lang:,}")
    print(f"    Testo estratto    : {gb:.3f} GB")
    print(f"    Durata            : {elapsed // 60}m {elapsed % 60}s")

    if not args.dry_run and args.out and total_docs > 0:
        size_mb = Path(args.out).stat().st_size / 1024**2
        print(f"    File output       : {args.out}  ({size_mb:.0f} MB)")
        print(f"\n  Prossimo step: filtraggio qualità")
        print(f"    python3 tools/filter_corpus.py {args.out} \\")
        print(f"        --checkpoint checkpoints/run4_best.pt \\")
        print(f"        --dedup --ppl-cut 0.80 --out {Path(args.out).stem}_clean.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Streaming generico da HuggingFace verso corpus SLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  python3 tools/fetch_hf_corpus.py \\
      --dataset gsarti/clean_mc4_it --config tiny \\
      --text-field text --max-gb 2 --out data/mc4_it_raw.txt

  python3 tools/fetch_hf_corpus.py \\
      --dataset PleIAs/common_corpus \\
      --text-field text --lang-field language --lang it \\
      --max-gb 5 --out data/common_it_raw.txt

  python3 tools/fetch_hf_corpus.py \\
      --dataset gsarti/clean_mc4_it --config tiny \\
      --text-field text --max-docs 5000 --dry-run
""")
    parser.add_argument('--dataset',    required=True,
                        help='nome dataset HuggingFace (es. gsarti/clean_mc4_it)')
    parser.add_argument('--config',     default=None,
                        help='configurazione dataset (es. tiny, small, it)')
    parser.add_argument('--text-field', default='text', dest='text_field',
                        help='campo testo nel dataset (default: text). Supporta annidamento: translation.it')
    parser.add_argument('--lang-field', default=None, dest='lang_field',
                        help='campo lingua per filtraggio (es. language)')
    parser.add_argument('--lang',       default=None,
                        help='valore lingua da mantenere (es. it)')
    parser.add_argument('--out',        required=False, default=None,
                        help='file corpus output (obbligatorio senza --dry-run)')
    parser.add_argument('--max-gb',     type=float, default=10.0, dest='max_gb',
                        help='GB massimi da estrarre (default: 10)')
    parser.add_argument('--max-docs',   type=int, default=10_000_000, dest='max_docs',
                        help='documenti massimi (default: 10 000 000)')
    parser.add_argument('--min-chars',  type=int, default=50, dest='min_chars',
                        help='caratteri minimi per documento (default: 50)')
    parser.add_argument('--append',     action='store_true',
                        help='aggiunge al file esistente')
    parser.add_argument('--dry-run',    action='store_true', dest='dry_run',
                        help='statistiche senza scrivere file')

    args = parser.parse_args()
    if not args.dry_run and args.out is None:
        parser.error("--out è obbligatorio senza --dry-run")
    fetch(args)


if __name__ == '__main__':
    main()
