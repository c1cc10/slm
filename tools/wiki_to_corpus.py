"""
SLM — tools/wiki_to_corpus.py

Estrae testo pulito dal dump XML di Wikipedia IT e lo prepara
come corpus per il training BPE.

Dipendenze:
    pip install wikiextractor

────────────────────────────────────────────────────────────────
Flusso di lavoro completo
────────────────────────────────────────────────────────────────

Step 1 — scarica il dump (una sola volta, ~1.5 GB):

    curl -L "https://dumps.wikimedia.org/itwiki/latest/itwiki-latest-pages-articles.xml.bz2" \\
         -o data/itwiki.xml.bz2

Step 2 — estrai e pulisci (5-15 min su M2, a seconda del --max-mb):

    python3 tools/wiki_to_corpus.py data/itwiki.xml.bz2

Step 3 — combina con corpus narrativo esistente:

    cat data/wiki_it.txt data/corpus.txt > data/corpus_bpe.txt

Step 4 — addestra tokenizer SPM e avvia training:

    python3 train.py --tokenizer bpe-spm \\
                     --bpe-corpus data/corpus_bpe.txt \\
                     --bpe-vocab 16000 \\
                     --data data/corpus_bpe.txt

────────────────────────────────────────────────────────────────
Opzioni
────────────────────────────────────────────────────────────────

--max-mb 200    Dimensione massima output (default 200 MB)
                Per un modello da 50M parametri, 200 MB sono sufficienti.
                Aumenta a 500+ per modelli più grandi.

--min-len 500   Filtra articoli più corti di N caratteri.
                Rimuove stub, liste di nomi, disambiguazioni brevi.

--processes 4   Processi paralleli per wikiextractor.
                Su M2 puoi alzare a 8 senza problemi.
"""

import sys, os, re, unicodedata, argparse, shutil, tempfile, subprocess
from pathlib import Path


def check_deps():
    try:
        import wikiextractor  # noqa: F401
    except ImportError:
        print("Dipendenza mancante. Installa con:")
        print("  pip install wikiextractor")
        sys.exit(1)


def run_extractor(dump_path: str, out_dir: str, processes: int):
    cmd = [
        sys.executable, '-m', 'wikiextractor.WikiExtractor',
        dump_path,
        '--output', out_dir,
        '--processes', str(processes),
        '--bytes', '20M',
        '--no-templates',
        '--quiet',
    ]
    print("Estrazione dump XML (può richiedere 5–15 minuti)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Errore wikiextractor:")
        print(result.stderr[-3000:])
        sys.exit(1)
    print("  Estrazione completata.")


_DOC_RE    = re.compile(r'<doc[^>]*>\n(.*?)\n</doc>', re.DOTALL)
_MULTI_NL  = re.compile(r'\n{3,}')
_WIKI_HDR  = re.compile(r'^={1,6}.{0,120}={1,6}\s*$', re.MULTILINE)


def clean_article(text: str) -> str:
    """Pulizia di un singolo articolo estratto da wikiextractor."""
    text = unicodedata.normalize('NFC', text)
    text = text.replace(' ', ' ')   # narrow no-break space (Word/FR tipografia)
    text = text.replace('\xa0',   ' ')   # non-breaking space
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _WIKI_HDR.sub('', text)       # rimuove titoli di sezione (==Sezione==)
    text = _MULTI_NL.sub('\n\n', text)
    return text.strip()


def iter_articles(extracted_dir: str, min_len: int):
    """Itera su tutti gli articoli estratti dal dump, yield testo pulito."""
    root  = Path(extracted_dir)
    files = sorted(root.rglob('wiki_*'))
    for fpath in files:
        try:
            raw = fpath.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        for m in _DOC_RE.finditer(raw):
            body = m.group(1).strip()
            if len(body) < min_len:
                continue
            body = clean_article(body)
            if body:
                yield body


def report_chars(path: str, sample_bytes: int = 300_000):
    """Riporta i caratteri non-ASCII presenti nel corpus (campione)."""
    sample = open(path, encoding='utf-8').read(sample_bytes)
    exotic = sorted(set(c for c in sample if ord(c) > 127))
    print(f"  Caratteri non-ASCII ({len(exotic)}): {repr(''.join(exotic[:50]))}"
          + ('...' if len(exotic) > 50 else ''))


def main():
    parser = argparse.ArgumentParser(
        description='Estrae e pulisce Wikipedia IT per il corpus SLM-BPE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('dump',
        help='Path al dump scaricato (.xml.bz2)')
    parser.add_argument('--out', default='data/wiki_it.txt',
        help='File di output (default: data/wiki_it.txt)')
    parser.add_argument('--max-mb', type=int, default=200,
        help='Dimensione massima output in MB (default: 200)')
    parser.add_argument('--min-len', type=int, default=500,
        help='Lunghezza minima articolo in caratteri (default: 500)')
    parser.add_argument('--processes', type=int, default=4,
        help='Processi paralleli per wikiextractor (default: 4)')
    args = parser.parse_args()

    check_deps()

    if not os.path.exists(args.dump):
        print(f"File non trovato: {args.dump}")
        print()
        print("Scarica il dump con:")
        print('  curl -L "https://dumps.wikimedia.org/itwiki/latest/'
              'itwiki-latest-pages-articles.xml.bz2" \\')
        print('       -o data/itwiki.xml.bz2')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    max_bytes     = args.max_mb * 1024 * 1024
    tmp_dir       = tempfile.mkdtemp(prefix='slm_wiki_')
    written_bytes = 0
    article_count = 0

    try:
        run_extractor(args.dump, tmp_dir, args.processes)

        print(f"Scrittura corpus → {args.out}  (limite: {args.max_mb} MB)")
        with open(args.out, 'w', encoding='utf-8') as f:
            for text in iter_articles(tmp_dir, args.min_len):
                chunk        = text + '\n\n'
                chunk_bytes  = len(chunk.encode('utf-8'))
                if written_bytes + chunk_bytes > max_bytes:
                    break
                f.write(chunk)
                written_bytes += chunk_bytes
                article_count += 1
                if article_count % 10_000 == 0:
                    mb = written_bytes / 1024 / 1024
                    print(f"  {article_count:,} articoli  |  {mb:.1f} MB")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\nCorpus salvato: {args.out}")
    print(f"  {article_count:,} articoli  |  {size_mb:.1f} MB su disco")
    report_chars(args.out)

    print(f"""
Prossimi passi:

  # 1. Combina con corpus narrativo esistente
  cat {args.out} data/corpus.txt > data/corpus_bpe.txt

  # 2. Training con tokenizer SPM
  python3 train.py --tokenizer bpe-spm \\
                   --bpe-corpus data/corpus_bpe.txt \\
                   --bpe-vocab 16000 \\
                   --data data/corpus_bpe.txt \\
                   --steps 10000 --batch 32
""")


if __name__ == '__main__':
    main()
