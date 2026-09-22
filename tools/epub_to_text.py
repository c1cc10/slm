"""
SLM — tools/epub_to_text.py

Estrae testo da uno o più file EPUB per creare un corpus di training.

Dipendenze:
    pip install ebooklib beautifulsoup4

Uso:
    # Singolo file
    python3 tools/epub_to_text.py libro.epub

    # Più file → corpus unico
    python3 tools/epub_to_text.py libro1.epub libro2.epub --out data/corpus.txt

    # Tutti gli epub in una cartella
    python3 tools/epub_to_text.py --dir ~/Libri --out data/corpus.txt
"""

import sys, os, argparse, unicodedata
from pathlib import Path

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    print("Dipendenze mancanti. Installa con:")
    print("  pip install ebooklib beautifulsoup4")
    sys.exit(1)


# Tag HTML da rimuovere prima dell'estrazione del testo.
# Includono note a piè di pagina, riferimenti, script, stili.
_STRIP_TAGS = ['script', 'style', 'sup', 'sub', 'aside', 'figure',
               'figcaption', 'nav', 'header', 'footer']


def extract_epub(path: str) -> str:
    """
    Estrae il testo da un EPUB rispettando l'ordine dei capitoli (spine).
    Ritorna una stringa UTF-8 pulita.
    """
    book  = epub.read_epub(path, options={'ignore_ncx': True})
    parts = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
        except Exception:
            continue

        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        raw = soup.get_text(separator='\n')
        lines = [l.strip() for l in raw.splitlines()]

        # Elimina righe vuote consecutive (max una riga vuota tra paragrafi)
        cleaned = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned.append('')
                prev_empty = True

        parts.append('\n'.join(cleaned).strip())

    return '\n\n'.join(p for p in parts if p)


def clean_text(text: str) -> str:
    """
    Pulizia del testo estratto.

    Caratteri rimossi o sostituiti:
      U+202F  narrow no-break space  -> spazio  (artefatto Word/tipografia FR)
      U+00A0  non-breaking space     -> spazio  (HTML &nbsp;)
      U+FEFF  BOM / zero-width NBSP  -> rimosso (header UTF-8, artefatto Windows)
      U+00A9  copyright sign         -> rimosso (pagine copyright degli ebook)
      U+2026  ellipsis               -> ...     (tre punti ASCII)
      U+FFFD  replacement character  -> rimosso (byte non decodificabile)
    """
    text = unicodedata.normalize('NFC', text)
    text = text.replace(' ', ' ')   # U+202F narrow no-break space
    text = text.replace(' ', ' ')        # U+00A0 non-breaking space
    text = text.replace('﻿', '')     # U+FEFF BOM
    text = text.replace('©', '')          # U+00A9 copyright sign
    text = text.replace('…', '...')  # U+2026 ellipsis
    text = text.replace('�', '')     # U+FFFD replacement character
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


def report_chars(text: str, label: str):
    exotic = sorted(set(c for c in text if ord(c) > 127))
    sample = ''.join(exotic[:50])
    print(f"  Caratteri non-ASCII ({len(exotic)}): {repr(sample)}"
          + ('...' if len(exotic) > 50 else ''))


def main():
    parser = argparse.ArgumentParser(
        description='Estrae testo da EPUB per il corpus SLM'
    )
    parser.add_argument('files', nargs='*', help='File .epub da processare')
    parser.add_argument('--dir', type=str, default=None,
                        help='Cartella da cui raccogliere tutti gli .epub')
    parser.add_argument('--out', type=str, default=None,
                        help='File di output (default: stesso nome del primo epub, .txt)')
    parser.add_argument('--append', action='store_true',
                        help='Aggiunge al file di output invece di sovrascriverlo')
    args = parser.parse_args()

    # Raccoglie i file epub
    epub_files = list(args.files)
    if args.dir:
        epub_files += sorted(str(p) for p in Path(args.dir).glob('**/*.epub'))

    if not epub_files:
        parser.print_help()
        sys.exit(1)

    # File di output
    if args.out:
        out_path = args.out
    elif len(epub_files) == 1:
        out_path = epub_files[0].replace('.epub', '.txt')
    else:
        out_path = 'data/corpus.txt'

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Estrazione
    all_text = []
    total_chars = 0

    for path in epub_files:
        if not os.path.exists(path):
            print(f"  SKIP (non trovato): {path}")
            continue

        print(f"Estrazione: {path}")
        try:
            text = extract_epub(path)
            text = clean_text(text)
        except Exception as e:
            print(f"  ERRORE: {e}")
            continue

        chars = len(text)
        total_chars += chars
        print(f"  {chars:,} caratteri estratti")
        report_chars(text, path)
        all_text.append(text)

    if not all_text:
        print("Nessun testo estratto.")
        sys.exit(1)

    corpus = '\n\n'.join(all_text)

    mode = 'a' if args.append else 'w'
    with open(out_path, mode, encoding='utf-8') as f:
        f.write(corpus)

    size_kb = os.path.getsize(out_path) // 1024
    print(f"\nCorpus salvato: {out_path}")
    print(f"  Totale: {total_chars:,} caratteri  |  {size_kb} KB su disco")
    print(f"\nProssimo step:")
    print(f"  python3 train.py --data {out_path}")


if __name__ == '__main__':
    main()
