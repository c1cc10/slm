#!/usr/bin/env python3
"""
tools/fetch_cc100.py — Scarica corpus CC-100 via HuggingFace streaming

CC-100: Common Crawl 2018, filtrato per lingua, disponibile su HuggingFace
senza account né accettazione termini (a differenza di OSCAR/CulturaX).

Dipendenze:
    pip install datasets

Esempi:
    python3 tools/fetch_cc100.py --lang it --max-gb 10 --out data/cc100_it_raw.txt
    python3 tools/fetch_cc100.py --lang it --max-docs 500000 --out data/cc100_it_raw.txt

Note:
    - Il dataset è scaricato in streaming: nessun download completo sul disco.
    - La lingua italiana (--lang it) pesa ~72 GB compressa; con streaming si
      scarica solo la quota specificata.
    - CulturaX IT (uonlp/CulturaX) è di qualità superiore ma richiede HF account.
"""

import argparse
import os
import sys
import time


def fetch_cc100(lang: str, out_path: str, max_gb: float, max_docs: int):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERRORE: datasets non installato.  pip install datasets")
        sys.exit(1)

    print(f"\n  fetch_cc100.py — lingua: {lang}  max: {max_gb} GB / {max_docs} doc")
    print(f"  Output : {out_path}")
    print(f"  Caricamento stream HuggingFace (cc100, {lang})...\n")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    try:
        ds = load_dataset('cc100', lang=lang, split='train', streaming=True,
                          trust_remote_code=True)
    except Exception as e:
        print(f"  ERRORE caricamento dataset: {e}")
        print(f"  Alternativa: usa CulturaX (uonlp/CulturaX, lang=it) con HF account.")
        sys.exit(1)

    max_bytes  = int(max_gb * 1024**3)
    total_bytes = 0
    total_docs  = 0
    t_start     = time.time()

    with open(out_path, 'w', encoding='utf-8') as f_out:
        for example in ds:
            text = example.get('text', '').strip()
            if not text:
                continue

            f_out.write(text)
            f_out.write('\n\n')
            total_bytes += len(text.encode('utf-8'))
            total_docs  += 1

            if total_docs % 10_000 == 0:
                gb_done  = total_bytes / 1024**3
                elapsed  = int(time.time() - t_start)
                speed_k  = total_docs / max(elapsed, 1) / 1000
                print(f"\r  {total_docs:>8,} doc  {gb_done:.3f}/{max_gb} GB  "
                      f"{speed_k:.1f}k doc/s  {elapsed}s   ",
                      end='', flush=True)

            if total_bytes >= max_bytes:
                break
            if total_docs >= max_docs:
                break

    elapsed = int(time.time() - t_start)
    gb_done = total_bytes / 1024**3
    mb_file = os.path.getsize(out_path) / 1024**2
    print(f"\n\n  Completato: {total_docs:,} documenti  {gb_done:.3f} GB testo")
    print(f"  File output: {out_path}  ({mb_file:.0f} MB)")
    print(f"  Durata: {elapsed // 60}m {elapsed % 60}s")
    print(f"\n  Prossimo step: filtraggio qualità")
    print(f"    python3 tools/filter_corpus.py {out_path} \\")
    print(f"        --checkpoint checkpoints/best.pt --lang it \\")
    print(f"        --ppl-cut 0.80 --dedup --out data/cc100_it_clean.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Scarica CC-100 via HuggingFace streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  python3 tools/fetch_cc100.py --lang it --max-gb 5 --out data/cc100_it_raw.txt
  python3 tools/fetch_cc100.py --lang it --max-docs 200000 --out data/cc100_it_raw.txt
""")
    parser.add_argument('--lang',      default='it',
                        help='codice lingua ISO 639-1 (default: it)')
    parser.add_argument('--out',       required=True,
                        help='file di output .txt')
    parser.add_argument('--max-gb',    type=float, default=10.0, dest='max_gb',
                        help='GB massimi di testo da scaricare (default: 10)')
    parser.add_argument('--max-docs',  type=int, default=10_000_000, dest='max_docs',
                        help='numero massimo documenti (default: 10 000 000)')

    args = parser.parse_args()
    fetch_cc100(args.lang, args.out, args.max_gb, args.max_docs)


if __name__ == '__main__':
    main()
