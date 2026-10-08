"""
SLM — tools/wiki_create_shards.py

Estrae il dump Wikipedia UNA SOLA VOLTA e distribuisce gli articoli
in N shard sequenziali non sovrapposti, senza ri-estrarre N volte.

Uso:
    python3 tools/wiki_create_shards.py data/itwiki.xml.bz2 \\
        --shard-mb 1100 --skip-mb 200 --n 3 --prefix data/wiki_run7 --processes 8

Produce:
    data/wiki_run7_s1.txt  (~1100 MB)
    data/wiki_run7_s2.txt  (~1100 MB)
    data/wiki_run7_s3.txt  (~1100 MB)

Logica di suddivisione:
    - Salta i primi --skip-mb MB di testo estratto
    - Poi riempie shard 1 fino a --shard-mb MB
    - Poi riempie shard 2, shard 3, ...
    - Gli articoli non si sovrappongono tra shard

Stima tempi (M5 8 processi): estrazione ~8-12 min, scrittura ~2-3 min.
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
    print("Estrazione dump XML (una sola volta — può richiedere 8-15 minuti)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Errore wikiextractor:")
        print(result.stderr[-3000:])
        sys.exit(1)
    print("  Estrazione completata.")


_DOC_RE   = re.compile(r'<doc[^>]*>\n(.*?)\n</doc>', re.DOTALL)
_MULTI_NL = re.compile(r'\n{3,}')
_WIKI_HDR = re.compile(r'^={1,6}.{0,120}={1,6}\s*$', re.MULTILINE)


def clean_article(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.replace(' ', ' ')
    text = text.replace('\xa0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _WIKI_HDR.sub('', text)
    text = _MULTI_NL.sub('\n\n', text)
    return text.strip()


def iter_articles(extracted_dir: str, min_len: int):
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
    sample = open(path, encoding='utf-8').read(sample_bytes)
    exotic = sorted(set(c for c in sample if ord(c) > 127))
    print(f"  Caratteri non-ASCII ({len(exotic)}): {repr(''.join(exotic[:40]))}"
          + ('...' if len(exotic) > 40 else ''))


def main():
    parser = argparse.ArgumentParser(
        description='Crea N shard sequenziali da Wikipedia IT in una sola estrazione',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('dump', help='Path al dump scaricato (.xml.bz2)')
    parser.add_argument('--prefix', default='data/wiki_run7',
        help='Prefisso output: genera <prefix>_s1.txt, _s2.txt, ... (default: data/wiki_run7)')
    parser.add_argument('--n', type=int, default=3,
        help='Numero di shard da creare (default: 3)')
    parser.add_argument('--shard-mb', type=int, default=1100,
        help='Dimensione massima di ogni shard in MB (default: 1100)')
    parser.add_argument('--skip-mb', type=int, default=0,
        help='Salta i primi N MB di testo estratto prima di iniziare (default: 0)')
    parser.add_argument('--min-len', type=int, default=500,
        help='Lunghezza minima articolo in caratteri (default: 500)')
    parser.add_argument('--processes', type=int, default=4,
        help='Processi paralleli per wikiextractor (default: 4)')
    args = parser.parse_args()

    check_deps()

    if not os.path.exists(args.dump):
        print(f"File non trovato: {args.dump}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.prefix) or '.', exist_ok=True)

    skip_bytes  = args.skip_mb * 1024 * 1024
    shard_bytes = args.shard_mb * 1024 * 1024
    tmp_dir     = tempfile.mkdtemp(prefix='slm_wiki_')

    try:
        run_extractor(args.dump, tmp_dir, args.processes)

        # Apri tutti gli shard contemporaneamente
        out_paths = [f"{args.prefix}_s{i+1}.txt" for i in range(args.n)]
        handles   = [open(p, 'w', encoding='utf-8') for p in out_paths]
        counts    = [0] * args.n
        sizes     = [0] * args.n

        skipped_bytes    = 0
        skip_done        = (skip_bytes == 0)
        current_shard    = 0
        total_written    = 0
        article_count    = 0

        if skip_bytes:
            print(f"Skip fase: salto i primi {args.skip_mb} MB di testo...")
        print(f"Scrittura {args.n} shard da ~{args.shard_mb} MB ciascuno...")

        for text in iter_articles(tmp_dir, args.min_len):
            chunk       = text + '\n\n'
            chunk_bytes = len(chunk.encode('utf-8'))

            # Fase di skip
            if not skip_done:
                skipped_bytes += chunk_bytes
                if skipped_bytes >= skip_bytes:
                    skip_done = True
                    print(f"  Skip completato ({skipped_bytes/1024/1024:.0f} MB saltati).")
                continue

            # Shard corrente pieno?
            if sizes[current_shard] + chunk_bytes > shard_bytes:
                current_shard += 1
                if current_shard >= args.n:
                    break   # tutti gli shard pieni

            handles[current_shard].write(chunk)
            sizes[current_shard]  += chunk_bytes
            counts[current_shard] += 1
            total_written         += chunk_bytes
            article_count         += 1

            if article_count % 10_000 == 0:
                mb = total_written / 1024 / 1024
                status = ' | '.join(
                    f"s{i+1}:{sizes[i]/1024/1024:.0f}MB" for i in range(args.n)
                )
                print(f"  {article_count:,} articoli | {status}")

    finally:
        for h in handles:
            h.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nShard creati:")
    for i, path in enumerate(out_paths):
        if os.path.exists(path):
            mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {path}  {mb:.1f} MB  ({counts[i]:,} articoli)")
            report_chars(path)
        else:
            print(f"  {path}  (non creato — corpus esaurito)")

    print(f"""
Prossimo passo: tokenizza ogni shard con il tokenizer esistente e avvia training.
    python3 train.py --data data/wiki_run7_s1.txt --tokenizer bpe-spm \\
                     --steps 1 --preset medium --rope --device cpu
    # (il primo run crea il cache .pt; poi train reale su GPU)
""")


if __name__ == '__main__':
    main()
