#!/usr/bin/env python3
"""
tools/eval.py — Valutazione PPL su dataset con sliding window

Calcola la perplexity su un intero file di testo usando finestre
sovrapposte. Ogni token viene scorato esattamente una volta, con
la massima quantità di contesto disponibile.

Differenza da eval_ppl.py: non valuta un singolo testo ma itera
su un corpus intero con finestre sovrapposte, aggrega la NLL su
tutti i token, e opzionalmente salva i risultati per chunk in CSV.

Uso:
    python3 tools/eval.py --checkpoint checkpoints/best.pt --data data/wiki_it.txt
    python3 tools/eval.py --checkpoint checkpoints/best.pt --data data/wiki_it.txt --stride 128
    python3 tools/eval.py --checkpoint ckpt_a.pt --compare ckpt_b.pt --data data/wiki_it.txt
    python3 tools/eval.py --checkpoint checkpoints/best.pt --data data/wiki_it.txt --output risultati.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO MODELLO — stessa logica di eval_ppl.py
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str):
    root = str(Path(checkpoint_path).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

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
        'device':   device,
        'max_seq':  ckpt['model_config'].max_seq_len,
    }
    return model, tok, device, ckpt['model_config'].max_seq_len, info


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
# EVAL CON SLIDING WINDOW
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_corpus(ids, model, device, max_seq, stride, verbose=True):
    """
    Valuta PPL su una lista di token ID con sliding window.

    Strategia:
      - Prima finestra [0 → max_seq]: scorata interamente.
      - Finestre successive [i → i+max_seq]: solo gli ultimi 'stride' token
        vengono scorati (quelli nuovi rispetto alla finestra precedente).
    Ogni token è scorato esattamente una volta, con il massimo contesto disponibile.

    Returns:
        (total_ppl, total_tokens, chunks)
        chunks = lista di dict {start_token, n_tokens, loss, ppl}
    """
    total_nll = 0.0
    total_tokens = 0
    chunks = []

    n = len(ids)
    positions = list(range(0, n - 1, stride))

    for idx_pos, begin in enumerate(positions):
        end = min(begin + max_seq, n)
        seq_len = end - begin
        if seq_len < 2:
            break

        x = torch.tensor([ids[begin:end - 1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[begin + 1:end]], dtype=torch.long, device=device)

        logits, _ = model(x)

        # Prima finestra: scorare tutti i token.
        # Finestre successive: solo gli ultimi 'stride' token (quelli non
        # ancora scorati dalla finestra precedente).
        if begin == 0:
            logits_eval = logits[0]     # (seq_len-1, vocab)
            y_eval = y[0]               # (seq_len-1,)
        else:
            n_new = min(stride, seq_len - 1)
            logits_eval = logits[0, -n_new:]
            y_eval = y[0, -n_new:]

        n_score = y_eval.size(0)
        loss_sum = F.cross_entropy(logits_eval, y_eval, reduction='sum').item()

        total_nll += loss_sum
        total_tokens += n_score
        chunk_loss = loss_sum / n_score
        chunks.append({
            'start_token': begin,
            'n_tokens':    n_score,
            'loss':        chunk_loss,
            'ppl':         math.exp(chunk_loss),
        })

        if verbose and len(chunks) % 50 == 0:
            print(f"  chunk {len(chunks):4d}  token {begin:8d}  "
                  f"ppl {math.exp(chunk_loss):7.1f}", flush=True)

    total_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    return total_ppl, total_tokens, chunks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _run(checkpoint_path, data_path, stride_override=None, output_csv=None, label=None):
    print(f"\n{'─'*60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Corpus     : {data_path}")

    model, tok, device, max_seq, info = _load_model(checkpoint_path)
    stride = stride_override if stride_override else max_seq // 2

    print(f"  Device     : {device}")
    print(f"  max_seq    : {max_seq}  stride: {stride}")
    print(f"  Val loss (training) : {info['val_loss']:.4f}  "
          f"(step {info['step']})")

    text = Path(data_path).read_text(encoding='utf-8')
    print(f"  Corpus     : {len(text):,} caratteri — tokenizzazione in corso...")
    ids = tok.encode(text)
    print(f"               {len(ids):,} token")

    print(f"  Avvio eval ({math.ceil((len(ids) - 1) / stride)} finestre attese)...")
    total_ppl, total_tokens, chunks = eval_corpus(ids, model, device, max_seq, stride)

    ppl_val = info['val_loss']
    print(f"\n  {'─'*40}")
    if label:
        print(f"  [{label}]")
    print(f"  PPL sul corpus   : {total_ppl:8.2f}")
    print(f"  Val loss training: {ppl_val:.4f}  (PPL {math.exp(ppl_val):.2f})")
    print(f"  Token scorati    : {total_tokens:,}")

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['start_token', 'n_tokens', 'loss', 'ppl'])
            w.writeheader()
            w.writerows(chunks)
        print(f"  CSV salvato      : {output_csv}")

    return total_ppl, total_tokens, chunks


def main():
    parser = argparse.ArgumentParser(description='Eval PPL su dataset con sliding window')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint .pt da valutare')
    parser.add_argument('--data', required=True,
                        help='File di testo da usare come corpus di test')
    parser.add_argument('--stride', type=int, default=None,
                        help='Passo dello sliding window (default: max_seq//2)')
    parser.add_argument('--output', type=str, default=None,
                        help='CSV di output con PPL per chunk (opzionale)')
    parser.add_argument('--compare', type=str, default=None,
                        help='Secondo checkpoint per confronto tabellare')
    args = parser.parse_args()

    ppl_a, tokens_a, _ = _run(args.checkpoint, args.data, args.stride, args.output, label='A')

    if args.compare:
        ppl_b, tokens_b, _ = _run(args.compare, args.data, args.stride, label='B')
        print(f"\n  {'─'*40}")
        print(f"  Confronto:")
        print(f"  A  {args.checkpoint:<40}  PPL {ppl_a:.2f}")
        print(f"  B  {args.compare:<40}  PPL {ppl_b:.2f}")
        delta = ppl_a - ppl_b
        if delta > 0:
            print(f"  → B è migliore di {delta:.2f} PPL")
        elif delta < 0:
            print(f"  → A è migliore di {-delta:.2f} PPL")
        else:
            print(f"  → PPL identica")

    print()


if __name__ == '__main__':
    main()
