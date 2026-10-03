#!/usr/bin/env python3
"""
tools/filter_corpus.py — Pipeline di filtraggio qualità per corpus SLM

Implementa la pipeline multi-stadio ispirata a Qwen2.5 / FineWeb:
  1. Filtro euristico (lunghezza, rapporto caratteri, linee duplicate)
  2. Filtro lingua (fastText, opzionale)
  3. Deduplicazione (exact-hash su paragrafi, opzionale)
  4. Quality score PPL (usa un checkpoint SLM come scorer, opzionale)

Dipendenze:
    pip install fasttext   # per --lang
    # Il checkpoint SLM è già disponibile nel progetto

Esempi:
    # Solo euristico + dedup (veloce, nessuna dipendenza extra)
    python3 tools/filter_corpus.py data/cc100_it_raw.txt \
        --dedup --out data/cc100_it_clean.txt

    # Pipeline completa (euristico + lingua + dedup + PPL scorer)
    python3 tools/filter_corpus.py data/cc100_it_raw.txt \
        --checkpoint checkpoints/best.pt \
        --lang it --min-words 50 --ppl-cut 0.80 --dedup \
        --out data/cc100_it_clean.txt
"""

import argparse
import hashlib
import math
import os
import re
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# FILTRO EURISTICO
# ─────────────────────────────────────────────────────────────────────────────

def passes_heuristic(doc: str, min_words: int) -> bool:
    """Controlla condizioni necessarie di qualità minima."""
    words = doc.split()
    if len(words) < min_words:
        return False

    chars_total = len(doc)
    if chars_total == 0:
        return False

    # Rapporto caratteri non-alfanumerici (simboli, punteggiatura anomala)
    non_alpha = sum(1 for c in doc if not (c.isalpha() or c.isspace() or c.isdigit()))
    if non_alpha / chars_total > 0.35:
        return False

    # Densità righe duplicate interne (spam, boilerplate)
    lines = [l.strip() for l in doc.split('\n') if l.strip()]
    if len(lines) > 5:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.5:
            return False

    # Almeno 3 lettere alfabetiche ogni 10 caratteri
    alpha_ratio = sum(1 for c in doc if c.isalpha()) / chars_total
    if alpha_ratio < 0.30:
        return False

    # Token alfanumerici misti (es: 'v4sta', 'c4n', username, codici prodotto)
    # Segnale di testo non-linguistico: rumore da web, forum, codice inline
    mixed_alnum = sum(1 for w in words if re.search(r'[a-zA-Z]\d|\d[a-zA-Z]', w))
    if mixed_alnum / len(words) > 0.05:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# FILTRO LINGUA — fastText
# ─────────────────────────────────────────────────────────────────────────────

_FT_MODEL = None
_FT_MODEL_PATH = os.path.expanduser('~/.cache/fasttext/lid.176.bin')

def _ensure_fasttext_model():
    global _FT_MODEL
    if _FT_MODEL is not None:
        return _FT_MODEL

    try:
        import fasttext as _ft
    except ImportError:
        print("  ERRORE: fasttext non installato.  pip install fasttext")
        sys.exit(1)

    if not os.path.exists(_FT_MODEL_PATH):
        import urllib.request
        os.makedirs(os.path.dirname(_FT_MODEL_PATH), exist_ok=True)
        url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
        print(f"  Download modello fastText langdetect... ({url})")
        urllib.request.urlretrieve(url, _FT_MODEL_PATH)
        print(f"  Salvato: {_FT_MODEL_PATH}")

    _FT_MODEL = _ft.load_model(_FT_MODEL_PATH)
    return _FT_MODEL


def is_language(doc: str, lang: str, threshold: float = 0.85) -> bool:
    model = _ensure_fasttext_model()
    # fastText vuole il testo su una riga
    line  = doc[:500].replace('\n', ' ').strip()
    preds = model.predict(line, k=1)
    label = preds[0][0].replace('__label__', '')
    score = preds[1][0]
    return label == lang and score >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICAZIONE — hash per paragrafo
# ─────────────────────────────────────────────────────────────────────────────

def dedup_doc(doc: str, seen_hashes: set) -> str:
    """Rimuove paragrafi già visti (exact hash). Restituisce doc ripulito."""
    paragraphs = [p.strip() for p in doc.split('\n\n') if p.strip()]
    unique = []
    for para in paragraphs:
        h = hashlib.md5(para.encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(para)
    return '\n\n'.join(unique)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY SCORE — PPL con checkpoint SLM esistente
# ─────────────────────────────────────────────────────────────────────────────

def load_ppl_scorer(checkpoint_path: str, device: str = None):
    """
    Carica il modello SLM dal checkpoint e restituisce (scorer, device).
    device: 'cuda' | 'mps' | 'cpu' — auto-detect se None.
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = ('cuda' if torch.cuda.is_available() else
                  'mps'  if torch.backends.mps.is_available() else
                  'cpu')

    # Aggiunge il parent dir al path per importare model.py
    project_root = str(Path(checkpoint_path).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from model import GPT

    # Funzione per caricare il tokenizer (copia da train.py)
    def _load_tok(state):
        t = state.get('type', 'char')
        if t == 'char':
            class _C:
                def __init__(self, s):
                    self._stoi = s['stoi']
                def encode(self, text):
                    return [self._stoi[c] for c in text if c in self._stoi]
            return _C(state)
        elif t == 'bpe-tiktoken':
            import tiktoken as _tt
            enc = _tt.get_encoding(state['encoding'])
            class _T:
                def encode(self, text): return enc.encode(text, disallowed_special=())
            return _T()
        elif t == 'bpe-spm':
            import sentencepiece as _spm
            import tempfile
            sp = _spm.SentencePieceProcessor()
            with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
                f.write(state['model_bytes']); tmp = f.name
            sp.load(tmp); os.unlink(tmp)
            class _S:
                def encode(self, text): return sp.encode(text, out_type=int)
            return _S()
        raise ValueError(f"Tokenizer sconosciuto: {t}")

    # TrainConfig è serializzato nel checkpoint come __main__.TrainConfig
    # Pickle cerca in sys.modules['__main__'], quindi lo iniettiamo lì
    import importlib, __main__
    try:
        train_mod = importlib.import_module('train')
        __main__.TrainConfig = train_mod.TrainConfig
    except Exception:
        pass

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tok_state = ckpt.get('tokenizer_state', ckpt.get('vocab', {}))
    if 'type' not in tok_state and 'stoi' in tok_state:
        tok_state = {'type': 'char', **tok_state}
    tok = _load_tok(tok_state)

    model = GPT(ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    max_seq = ckpt['model_config'].max_seq_len

    @torch.no_grad()
    def scorer(text: str) -> float:
        ids = tok.encode(text[:10_000])
        if len(ids) < 8:
            return float('inf')
        ids = ids[:max_seq + 1]
        x   = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y   = torch.tensor([ids[1:]],  dtype=torch.long, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size), y.view(-1)
        )
        return math.exp(loss.item())

    return scorer, device


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def filter_corpus(args):
    in_path  = Path(args.input)
    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    rej_path = Path(args.rejected) if args.rejected else None
    if rej_path:
        os.makedirs(rej_path.parent, exist_ok=True)

    print(f"\n  filter_corpus.py")
    print(f"  Input  : {in_path}  ({in_path.stat().st_size / 1024**2:.0f} MB)")
    print(f"  Output : {out_path}")
    if rej_path:
        print(f"  Rejected: {rej_path}  (campione doc scartati da PPL con score)")
    active = []
    if args.lang:        active.append(f"lingua={args.lang}")
    if args.dedup:       active.append("dedup")
    if args.checkpoint:  active.append(f"PPL-scorer ppl-cut={args.ppl_cut:.0%}")
    print(f"  Filtri : euristico  {' | '.join(active) if active else ''}")
    print()

    # Carica PPL scorer se richiesto
    scorer    = None
    ppl_scores = []
    if args.checkpoint:
        print(f"  Caricamento checkpoint per PPL scorer...")
        scorer, ppl_device = load_ppl_scorer(args.checkpoint, getattr(args, 'device', None))
        print(f"  Scorer pronto  [device: {ppl_device}]")
        if ppl_device == 'cpu':
            est_h = int(2_600_000 / 7 / 3600)
            print(f"  AVVISO: PPL scorer su CPU — stima ~{est_h}h per 2.6M doc. Usa --device cuda su Vast.ai.")

    # Soglia di log: più frequente quando PPL scorer attivo (lento)
    LOG_EVERY = 5_000 if scorer is not None else 100_000

    # Prima passata PPL (calcola distribuzione per determinare cut)
    ppl_cut_abs = None
    if scorer is not None:
        print(f"  Prima passata — calcolo distribuzione PPL su campione (5 000 doc)...")
        sample_scores = []
        with open(in_path, encoding='utf-8', errors='replace') as f:
            doc = []
            for line in f:
                line = line.rstrip('\n')
                if line == '':
                    text = '\n'.join(doc).strip()
                    doc = []
                    if text and passes_heuristic(text, args.min_words):
                        ppl = scorer(text)
                        if ppl != float('inf'):
                            sample_scores.append(ppl)
                        if len(sample_scores) >= 5000:
                            break
                else:
                    doc.append(line)

        if sample_scores:
            sample_scores.sort()
            idx = int(len(sample_scores) * args.ppl_cut)
            ppl_cut_abs = sample_scores[min(idx, len(sample_scores) - 1)]
            pct_label = f"p{int(args.ppl_cut * 100)}"
            print(f"  Distribuzione PPL: p10={sample_scores[len(sample_scores)//10]:.1f}"
                  f"  mediana={sample_scores[len(sample_scores)//2]:.1f}"
                  f"  {pct_label}={ppl_cut_abs:.1f}  (soglia al {args.ppl_cut:.0%})")
        else:
            print(f"  AVVISO: impossibile calcolare distribuzione PPL — PPL scorer disabilitato")
            scorer = None

    # Seconda passata — filtraggio effettivo
    seen_hashes = set() if args.dedup else None
    counts = {'in': 0, 'heuristic': 0, 'lang': 0, 'dedup': 0, 'ppl': 0, 'out': 0}
    t_start = time.time()
    rej_written = 0
    REJ_MAX = 500   # max esempi nel file rejected
    file_size = in_path.stat().st_size

    with open(in_path, encoding='utf-8', errors='replace') as f_in, \
         open(out_path, 'w', encoding='utf-8') as f_out, \
         (open(rej_path, 'w', encoding='utf-8') if rej_path else open(os.devnull, 'w')) as f_rej:

        doc = []
        while True:
            line = f_in.readline()
            if not line:
                break
            line = line.rstrip('\n')
            if line == '':
                text = '\n'.join(doc).strip()
                doc = []
                if not text:
                    continue
                counts['in'] += 1

                if not passes_heuristic(text, args.min_words):
                    counts['heuristic'] += 1
                    continue

                if args.lang and not is_language(text, args.lang):
                    counts['lang'] += 1
                    continue

                if seen_hashes is not None:
                    text = dedup_doc(text, seen_hashes)
                    if not text:
                        counts['dedup'] += 1
                        continue

                if scorer is not None and ppl_cut_abs is not None:
                    ppl = scorer(text)
                    if ppl > ppl_cut_abs:
                        counts['ppl'] += 1
                        if rej_written < REJ_MAX:
                            f_rej.write(f"=== PPL {ppl:.1f} ===\n{text[:600]}\n\n")
                            rej_written += 1
                        continue

                f_out.write(text)
                f_out.write('\n\n')
                counts['out'] += 1

                if counts['in'] % LOG_EVERY == 0:
                    elapsed  = time.time() - t_start
                    kept_pct = counts['out'] / max(counts['in'], 1) * 100
                    drop_h   = counts['heuristic'] / max(counts['in'], 1) * 100
                    drop_d   = counts['dedup']     / max(counts['in'], 1) * 100
                    drop_p   = counts['ppl']       / max(counts['in'], 1) * 100
                    speed    = counts['in'] / max(elapsed, 1) / 1000
                    pos      = f_in.tell()
                    pct_done = pos / file_size * 100 if file_size > 0 else 0
                    if pct_done > 0.1:
                        eta_sec = elapsed / (pct_done / 100) - elapsed
                        eta_str = f"  ETA {int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m"
                    else:
                        eta_str = ""
                    print(f"  {counts['in']:>9,} in  {counts['out']:>9,} kept ({kept_pct:.1f}%)"
                          f"  -heur {drop_h:.1f}%  -dedup {drop_d:.1f}%  -ppl {drop_p:.1f}%"
                          f"  {speed:.1f}k doc/s  {int(elapsed)//60}m{int(elapsed)%60:02d}s"
                          f"  {pct_done:.1f}%{eta_str}")
            else:
                doc.append(line)

    elapsed = int(time.time() - t_start)
    kept    = counts['out']
    total   = max(counts['in'], 1)
    print(f"\n\n  Risultati:")
    print(f"    Totale in  : {counts['in']:>8,}")
    print(f"    Euristico  : -{counts['heuristic']:>7,}  ({counts['heuristic']/total:.1%})")
    if args.lang:
        print(f"    Lingua     : -{counts['lang']:>7,}  ({counts['lang']/total:.1%})")
    if args.dedup:
        print(f"    Dedup      : -{counts['dedup']:>7,}  ({counts['dedup']/total:.1%})")
    if scorer:
        print(f"    PPL filter : -{counts['ppl']:>7,}  ({counts['ppl']/total:.1%})")
    print(f"    Output     :  {kept:>8,}  ({kept/total:.1%} mantenuto)")
    print(f"    File       : {out_path}  ({out_path.stat().st_size/1024**2:.0f} MB)")
    print(f"    Durata     : {elapsed // 60}m {elapsed % 60}s")


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline di filtraggio qualità per corpus SLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  # Rapido — solo euristico e dedup
  python3 tools/filter_corpus.py data/cc100_raw.txt --dedup --out data/cc100_clean.txt

  # Completo — euristico + lingua + dedup + PPL scorer
  python3 tools/filter_corpus.py data/cc100_raw.txt \\
      --checkpoint checkpoints/best.pt \\
      --lang it --min-words 50 --ppl-cut 0.80 --dedup \\
      --out data/cc100_clean.txt
""")
    parser.add_argument('input',
                        help='file corpus da filtrare (.txt, documenti separati da \\n\\n)')
    parser.add_argument('--out',        required=True,
                        help='file corpus filtrato in output')
    parser.add_argument('--min-words',  type=int, default=50, dest='min_words',
                        help='parole minime per documento (default: 50)')
    parser.add_argument('--lang',       default=None,
                        help='filtra per lingua ISO 639-1 (es. it) — richiede fasttext')
    parser.add_argument('--rejected',   default=None,
                        help='file dove salvare un campione (max 500) dei doc scartati dal PPL scorer con il loro score')
    parser.add_argument('--dedup',      action='store_true',
                        help='rimuove paragrafi duplicati via hash MD5')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint SLM per quality scoring PPL')
    parser.add_argument('--ppl-cut',    type=float, default=0.80, dest='ppl_cut',
                        help='percentile PPL da scartare (default: 0.80 = top 20%%)')
    parser.add_argument('--device',     default=None,
                        help='device per PPL scorer: cuda | mps | cpu (default: auto-detect)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"  ERRORE: file non trovato: {args.input}")
        sys.exit(1)

    filter_corpus(args)


if __name__ == '__main__':
    main()
