#!/usr/bin/env python3
"""
tools/infer.py — Genera testo dal modello con prompt libero

Uso:
    python3 tools/infer.py "Crea un appuntamento con Marco venerdì alle 15."
    python3 tools/infer.py --checkpoint checkpoints/sft_intent_v1.pt "Sposta la riunione"
    echo "Aggiungi Marco ai contatti" | python3 tools/infer.py
    python3 tools/infer.py                         # REPL interattivo

Opzioni utili:
    --checkpoint   checkpoint .pt da usare (default: checkpoints/best.pt)
    --max-tokens   token da generare (default: 80)
    --temperature  0.1 = deterministico, 1.0 = standard, >1 = creativo (default: 0.3)
    --top-k        campiona dai top-K token più probabili, 0 = disabilitato (default: 10)
    --top-p        nucleus sampling 0-1, 1.0 = disabilitato (default: 1.0)
    --rep-penalty  penalità ripetizione, 1.0 = disabilitato (default: 1.1)
    --no-cache     disabilita KV-cache (più lento, per debug)
    --raw          mostra solo la completion (senza ripetere il prompt)
"""

import argparse
import os
import sys
from pathlib import Path

import torch


# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO MODELLO
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
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
        'vocab':    tok_state.get('type', 'char'),
        'device':   device,
        'max_seq':  ckpt['model_config'].max_seq_len,
    }
    return model, tok, device, info


def _make_tokenizer(state):
    t = state.get('type', 'char')

    if t == 'char':
        stoi = state['stoi']
        itos = {v: k for k, v in stoi.items()}
        class _C:
            def encode(self, text): return [stoi[c] for c in text if c in stoi]
            def decode(self, ids):  return ''.join(itos.get(i, '?') for i in ids)
        return _C()

    if t == 'bpe-tiktoken':
        import tiktoken as _tt
        enc = _tt.get_encoding(state['encoding'])
        class _T:
            def encode(self, text): return enc.encode(text, disallowed_special=())
            def decode(self, ids):  return enc.decode(ids)
        return _T()

    if t == 'bpe-spm':
        import sentencepiece as _spm, tempfile
        sp = _spm.SentencePieceProcessor()
        with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
            f.write(state['model_bytes']); tmp = f.name
        sp.load(tmp); os.unlink(tmp)
        class _S:
            def encode(self, text): return sp.encode(text, out_type=int)
            def decode(self, ids):  return sp.decode(ids)
        return _S()

    raise ValueError(f"Tokenizer sconosciuto: {t}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERAZIONE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(prompt: str, model, tok, device: str, max_seq: int,
             max_new_tokens: int, temperature: float, top_k: int,
             top_p: float, rep_penalty: float, use_cache: bool) -> str:

    ids = tok.encode(prompt)
    if not ids:
        return ''

    idx = torch.tensor([ids], dtype=torch.long).to(device)
    prompt_len = idx.shape[1]

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        use_cache=use_cache,
    )

    new_ids = out[0, prompt_len:].tolist()
    return tok.decode(new_ids)


def _first_json(text: str) -> str:
    """Ritorna il primo oggetto JSON completo trovato in text, o text invariato."""
    import json
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
    return text


def _print_generation(prompt: str, completion: str, raw: bool):
    BOLD  = '\033[1m'
    DIM   = '\033[2m'
    CYAN  = '\033[36m'
    GREEN = '\033[32m'
    RESET = '\033[0m'

    if raw:
        print(completion)
        return

    print()
    print(f"  {DIM}prompt   ▷{RESET}  {prompt}")
    print(f"  {CYAN}output   ▷{RESET}  {GREEN}{completion}{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Genera testo con il modello SLM da prompt CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  python3 tools/infer.py "Crea un appuntamento con Marco venerdì alle 15."
  python3 tools/infer.py --checkpoint checkpoints/sft_intent_v1.pt "Sposta la riunione"
  python3 tools/infer.py --max-tokens 120 --temperature 0.1 "Eliminare evento di domani"
  python3 tools/infer.py --raw "Aggiungi Marco ai contatti" | python3 -m json.tool
""")
    parser.add_argument('prompt',        nargs='?', default=None,
                        help='prompt di input (ometti per REPL interattivo)')
    parser.add_argument('--checkpoint',  default='checkpoints/best.pt',
                        help='checkpoint .pt (default: checkpoints/best.pt)')
    parser.add_argument('--max-tokens',  type=int,   default=80,
                        help='token da generare (default: 80)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='temperatura campionamento (default: 0.3)')
    parser.add_argument('--top-k',       type=int,   default=10,
                        help='top-k campionamento, 0=disabilitato (default: 10)')
    parser.add_argument('--top-p',       type=float, default=1.0,
                        help='nucleus sampling 0-1 (default: 1.0=disabilitato)')
    parser.add_argument('--rep-penalty', type=float, default=1.1,
                        help='penalità ripetizione (default: 1.1)')
    parser.add_argument('--no-cache',      action='store_true',
                        help='disabilita KV-cache')
    parser.add_argument('--raw',           action='store_true',
                        help='output solo completion, senza formattazione')
    parser.add_argument('--stop-on-json', action='store_true',
                        help='tronca la completion al primo oggetto JSON valido')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"  ERRORE: checkpoint non trovato: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    if not args.raw:
        print(f"  Caricamento {args.checkpoint}...", end='', flush=True)

    model, tok, device, info = load_model(args.checkpoint)

    if not args.raw:
        print(f" ok  [{info['vocab']}  step {info['step']}  val_loss {info['val_loss']:.4f}"
              f"  {info['device']}]")
        print(f"  temp={args.temperature}  top_k={args.top_k}  top_p={args.top_p}"
              f"  rep={args.rep_penalty}  max_tokens={args.max_tokens}")
        print()

    gen_kwargs = dict(
        model=model, tok=tok, device=device,
        max_seq=info['max_seq'],
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        use_cache=not args.no_cache,
    )

    def _run(prompt: str):
        completion = generate(prompt, **gen_kwargs)
        if args.stop_on_json:
            completion = _first_json(completion)
        _print_generation(prompt, completion, args.raw)

    # Argomento diretto
    if args.prompt:
        _run(args.prompt)
        return

    # Stdin pipe
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if prompt:
            _run(prompt)
        return

    # REPL interattivo
    print("  Inserisci prompt (Invio per generare, riga vuota per finire, Ctrl+C per uscire):")
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

            _run('\n'.join(lines))

    except KeyboardInterrupt:
        print('\n  Uscita.')


if __name__ == '__main__':
    main()
