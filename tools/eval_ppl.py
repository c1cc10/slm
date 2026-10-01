#!/usr/bin/env python3
"""
tools/eval_ppl.py — Valuta la perplexity di un testo con il modello addestrato

La perplexity misura quanto il modello è "sorpreso" dal testo:
  bassa  (30–60)   → testo simile a Wikipedia IT — alta qualità per il modello
  media  (60–150)  → italiano buono, registro diverso da enciclopedico
  alta   (150–400) → probabilmente non-italiano o degradato
  molto alta (400+) → spazzatura, lingua straniera, HTML frammentato

Uso:
    python3 tools/eval_ppl.py                          # interattivo (REPL)
    python3 tools/eval_ppl.py "Il testo da valutare"   # argomento diretto
    echo "testo multiriga" | python3 tools/eval_ppl.py # stdin pipe
    python3 tools/eval_ppl.py --checkpoint checkpoints/best.pt
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO MODELLO
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    root = str(Path(checkpoint_path).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    # TrainConfig è stato serializzato con __module__='__main__' (train.py girava
    # come script). pickle lo cerca in __main__ dell'interprete corrente — che è
    # eval_ppl.py. Iniettarlo qui risolve l'AttributeError senza modificare train.py.
    import importlib, __main__
    _train = importlib.import_module('train')
    for _name in ('TrainConfig', 'GPTConfig'):
        if hasattr(_train, _name) and not hasattr(__main__, _name):
            setattr(__main__, _name, getattr(_train, _name))

    from model import GPT

    if torch.backends.mps.is_available():   device = 'mps'
    elif torch.cuda.is_available():         device = 'cuda'
    else:                                   device = 'cpu'

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Tokenizer — stessa logica di train.py
    tok_state = ckpt.get('tokenizer_state', ckpt.get('vocab', {}))
    if 'type' not in tok_state and 'stoi' in tok_state:
        tok_state = {'type': 'char', **tok_state}
    tok = _make_tokenizer(tok_state)

    model = GPT(ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    info = {
        'step':     ckpt.get('step', '?'),
        'val_loss': ckpt.get('val_loss', float('nan')),
        'vocab':    tok_state.get('type', 'char'),
        'device':   device,
        'max_seq':  ckpt['model_config'].max_seq_len,
    }
    return model, tok, device, info


def _make_tokenizer(state):
    t = state.get('type', 'char')

    if t == 'char':
        stoi = state['stoi']
        class _C:
            def encode(self, text): return [stoi[c] for c in text if c in stoi]
        return _C()

    if t == 'bpe-tiktoken':
        import tiktoken as _tt
        enc = _tt.get_encoding(state['encoding'])
        class _T:
            def encode(self, text): return enc.encode(text, disallowed_special=())
        return _T()

    if t == 'bpe-spm':
        import sentencepiece as _spm, tempfile
        sp = _spm.SentencePieceProcessor()
        with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
            f.write(state['model_bytes']); tmp = f.name
        sp.load(tmp); os.unlink(tmp)
        class _S:
            def encode(self, text): return sp.encode(text, out_type=int)
        return _S()

    raise ValueError(f"Tokenizer sconosciuto: {t}")


# ─────────────────────────────────────────────────────────────────────────────
# CALCOLO PPL
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(text: str, model, tok, device: str, max_seq: int) -> dict:
    ids = tok.encode(text)
    n_tokens = len(ids)

    if n_tokens < 2:
        return {'ppl': None, 'tokens': n_tokens, 'label': '—', 'note': 'testo troppo corto'}

    ids_trunc = ids[:max_seq + 1]
    x = torch.tensor([ids_trunc[:-1]], dtype=torch.long).to(device)
    y = torch.tensor([ids_trunc[1:]],  dtype=torch.long).to(device)

    logits, _ = model(x)
    loss = F.cross_entropy(
        logits.view(-1, model.config.vocab_size), y.view(-1)
    )
    ppl = math.exp(loss.item())

    if ppl < 60:
        label, color = 'BASSA   ✓', '\033[32m'   # verde
    elif ppl < 150:
        label, color = 'MEDIA    ', '\033[33m'   # giallo
    elif ppl < 400:
        label, color = 'ALTA    ↓', '\033[91m'   # arancio
    else:
        label, color = 'MOLTO ALTA ✗', '\033[31m'  # rosso
    reset = '\033[0m'

    note = ''
    if n_tokens > max_seq:
        note = f' (troncato a {max_seq} token su {n_tokens})'

    return {
        'ppl':    ppl,
        'tokens': n_tokens,
        'label':  label,
        'color':  color,
        'reset':  reset,
        'note':   note,
    }


def print_result(text: str, r: dict):
    if r['ppl'] is None:
        print(f"  PPL —         {r['note']}")
        return
    bar_len = min(int(math.log10(max(r['ppl'], 1)) * 8), 40)
    bar = '█' * bar_len
    print(f"  PPL {r['color']}{r['ppl']:8.1f}  {r['label']}{r['reset']}"
          f"  {bar}  [{r['tokens']} token{r['note']}]")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Valuta perplexity di un testo con il modello SLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  python3 tools/eval_ppl.py
  python3 tools/eval_ppl.py "Il Parlamento europeo è un'istituzione dell'UE."
  echo "The European Parliament is an EU institution." | python3 tools/eval_ppl.py
""")
    parser.add_argument('text',        nargs='?', default=None,
                        help='testo da valutare (ometti per modalità interattiva)')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt',
                        help='checkpoint da usare (default: checkpoints/best.pt)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"  ERRORE: checkpoint non trovato: {args.checkpoint}")
        sys.exit(1)

    print(f"  Caricamento {args.checkpoint}...", end='', flush=True)
    model, tok, device, info = load_model(args.checkpoint)
    print(f" ok  [{info['vocab']}  step {info['step']}  val_loss {info['val_loss']:.4f}"
          f"  max_seq {info['max_seq']}  {info['device']}]")
    print()
    print("  Scala:  < 60 BASSA (simile a Wikipedia IT)"
          "  60–150 MEDIA  150–400 ALTA  400+ MOLTO ALTA")
    print()

    # Argomento diretto
    if args.text:
        r = compute_ppl(args.text, model, tok, device, info['max_seq'])
        print(f"  ▷ {repr(args.text[:80])}")
        print_result(args.text, r)
        return

    # Stdin pipe
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            r = compute_ppl(text, model, tok, device, info['max_seq'])
            print(f"  ▷ {repr(text[:80])}")
            print_result(text, r)
        return

    # REPL interattivo
    print("  Inserisci testo (invio per valutare, riga vuota per finire, Ctrl+C per uscire):")
    print("  ─" * 30)
    try:
        while True:
            lines = []
            while True:
                try:
                    line = input('  ▷ ' if not lines else '    ')
                except EOFError:
                    break
                if line == '' and lines:
                    break
                if line == '' and not lines:
                    continue
                lines.append(line)

            if not lines:
                break

            text = '\n'.join(lines)
            r = compute_ppl(text, model, tok, device, info['max_seq'])
            print_result(text, r)
            print()

    except KeyboardInterrupt:
        print('\n  Uscita.')


if __name__ == '__main__':
    main()
