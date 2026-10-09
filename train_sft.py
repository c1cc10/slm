#!/usr/bin/env python3
"""
train_sft.py — Supervised Fine-Tuning per SLM

Formato dati: JSONL {"prompt": "...", "completion": "..."}
Separatore default: "\\n\\n" tra prompt e completion

Loss mascherata: cross-entropy solo sui token di completion.
I token di prompt ricevono label=-100 → F.cross_entropy li ignora.

Esempi:
    python3 train_sft.py --train data/sft_cond.jsonl \\
        --out checkpoints/run5a_sft_cond.pt
    python3 train_sft.py --train data/sft_moto.jsonl --val data/sft_moto_val.jsonl \\
        --probe data/wiki_probe.txt --out checkpoints/run5b_sft_moto.pt
    python3 train_sft.py --train data/sft_scrit.jsonl --epochs 5 --lr 1e-5 \\
        --out checkpoints/run5c_sft_scrit.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import GPT, GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER — versioni minimal, state-only (dati dal checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class _CharTok:
    def __init__(self, state):
        self._stoi = state['stoi']
        self._itos = {int(k): v for k, v in state['itos'].items()}

    @property
    def vocab_size(self): return len(self._stoi)

    def encode(self, text):
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids):
        return ''.join(self._itos.get(i, '?') for i in ids)


class _TiktokenTok:
    def __init__(self, state):
        try:
            import tiktoken as _tt
        except ImportError:
            print("  ERRORE: tiktoken non installato.  pip install tiktoken")
            sys.exit(1)
        self._enc = _tt.get_encoding(state['encoding'])

    @property
    def vocab_size(self): return self._enc.n_vocab

    def encode(self, text):
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids):
        try:
            return self._enc.decode(ids)
        except Exception:
            return self._enc.decode([i for i in ids if i < self.vocab_size])


class _SPMTok:
    def __init__(self, state):
        try:
            import sentencepiece as _spm
        except ImportError:
            print("  ERRORE: sentencepiece non installato.  pip install sentencepiece")
            sys.exit(1)
        import tempfile
        self._sp = _spm.SentencePieceProcessor()
        with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
            f.write(state['model_bytes'])
            tmp = f.name
        self._sp.load(tmp)
        os.unlink(tmp)

    @property
    def vocab_size(self): return self._sp.vocab_size()

    def encode(self, text):
        return self._sp.encode(text, out_type=int)

    def decode(self, ids):
        return self._sp.decode(ids)


def _get_tok_state(ckpt):
    """Legge stato tokenizer dal checkpoint; supporta vecchio formato 'vocab'."""
    if 'tokenizer_state' in ckpt:
        return ckpt['tokenizer_state']
    return {'type': 'char', **ckpt['vocab']}


def load_tokenizer(state):
    t = state.get('type', 'char')
    if t == 'char':         return _CharTok(state)
    if t == 'bpe-tiktoken': return _TiktokenTok(state)
    if t == 'bpe-spm':      return _SPMTok(state)
    raise ValueError(f"Tipo tokenizer sconosciuto nel checkpoint: {t}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET SFT
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Carica un file JSONL {"prompt": "...", "completion": "..."}.

    Per ogni coppia costruisce:
      input_ids : [p0, ..., pm, c0, ..., cn]          lunghezza N
      targets   : [-100]*(m-1) + [c0,...,cn] + [-100]  lunghezza N

    Dove targets[i] è il token da predire alla posizione i, cioè input_ids[i+1].
    Le posizioni prompt ricevono -100 → il gradiente non viene calcolato su di esse.
    """

    def __init__(self, path, tokenizer, max_seq_len, sep="\n\n"):
        self.pairs   = []
        self.tok     = tokenizer
        self.max_len = max_seq_len
        skipped      = 0

        with open(path, encoding='utf-8') as f:
            for lineno, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(f"  [SFTDataset] riga {lineno}: JSON non valido ({e}), saltata")
                    skipped += 1
                    continue

                if 'prompt' not in item or 'completion' not in item:
                    print(f"  [SFTDataset] riga {lineno}: campo mancante, saltata")
                    skipped += 1
                    continue

                prompt_ids = tokenizer.encode(item['prompt'] + sep)
                comp_ids   = tokenizer.encode(item['completion'])

                if not comp_ids:
                    skipped += 1
                    continue

                # Tronca la completion se l'esempio supera max_seq_len
                total = len(prompt_ids) + len(comp_ids)
                if total > max_seq_len:
                    budget = max_seq_len - len(prompt_ids)
                    if budget < 4:
                        # Prompt già troppo lungo: esempio inutilizzabile
                        skipped += 1
                        continue
                    comp_ids = comp_ids[:budget]

                n_prompt = len(prompt_ids)
                n_comp   = len(comp_ids)

                input_ids = prompt_ids + comp_ids

                # targets[i] = input_ids[i+1] dove i è in posizione completion,
                # altrimenti -100.
                # Layout: [-100]*(n_prompt-1) + comp_ids + [-100]
                # Lunghezza: (n_prompt-1) + n_comp + 1 = n_prompt + n_comp = len(input_ids) ✓
                targets = [-100] * (n_prompt - 1) + comp_ids + [-100]

                self.pairs.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(targets,   dtype=torch.long),
                ))

        if skipped:
            print(f"  [SFTDataset] {skipped} esempi saltati")
        print(f"  [SFTDataset] {len(self.pairs)} esempi caricati da {path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _collate_pad(batch):
    """Padding a destra: input_ids → pad 0, targets → pad -100."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    padded_in  = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_tgt = torch.full((len(targets), max_len), -100, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = inp.size(0)
        padded_in[i,  :L] = inp
        padded_tgt[i, :L] = tgt

    return padded_in, padded_tgt


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE — warmup lineare + cosine decay (LR ridotto per SFT)
# ─────────────────────────────────────────────────────────────────────────────

def get_sft_lr(step, max_steps, lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# PROBE PERPLEXITY — rilevamento catastrophic forgetting
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_probe_ppl(model, probe_ids, max_seq_len, device, n_windows=50):
    """
    Stima PPL su n_windows finestre casuali del probe corpus.
    Un ratio probe_ppl/baseline > threshold segnala forgetting.
    """
    model.eval()
    n = len(probe_ids) - max_seq_len - 1
    if n <= 0:
        model.train()
        return None

    losses = []
    idxs   = torch.randint(n, (n_windows,))

    for i in idxs:
        x = probe_ids[i : i + max_seq_len].unsqueeze(0).to(device)
        y = probe_ids[i + 1 : i + max_seq_len + 1].unsqueeze(0).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size), y.view(-1)
        )
        losses.append(loss.item())

    model.train()
    return math.exp(sum(losses) / len(losses))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SFT
# ─────────────────────────────────────────────────────────────────────────────

def train_sft(args):

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        if torch.backends.mps.is_available():   device = 'mps'
        elif torch.cuda.is_available():         device = 'cuda'
        else:                                   device = 'cpu'
    else:
        device = args.device

    print(f"\n{'═'*60}")
    print(f"  SLM — Supervised Fine-Tuning")
    print(f"{'═'*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Train      : {args.train}")
    if args.val:
        print(f"  Val        : {args.val}")
    if args.probe:
        print(f"  Probe      : {args.probe}")
    print(f"  Output     : {args.out}")

    # ── Carica checkpoint core ────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"\n  ERRORE: checkpoint non trovato: {args.checkpoint}")
        sys.exit(1)

    # Registra TrainConfig in __main__ per la deserializzazione pickle del checkpoint.
    import importlib as _il, __main__ as _m
    _tr = _il.import_module('train')
    for _n in ('TrainConfig', 'GPTConfig'):
        if hasattr(_tr, _n) and not hasattr(_m, _n):
            setattr(_m, _n, getattr(_tr, _n))

    print(f"\n  Caricamento checkpoint...", end='', flush=True)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    tok  = load_tokenizer(_get_tok_state(ckpt))

    model_cfg   = ckpt['model_config']
    max_seq_len = model_cfg.max_seq_len

    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f" ok")
    print(f"  Step base  : {ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    print(f"  Parametri  : {model.count_params():,}")
    print(f"  Vocab      : {tok.vocab_size:,}  max_seq={max_seq_len}")
    tok_type = _get_tok_state(ckpt).get('type', 'char')
    print(f"  Tokenizer  : {tok_type}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n  Caricamento dataset...")
    train_ds = SFTDataset(args.train, tok, max_seq_len, sep=args.sep)

    if len(train_ds) == 0:
        print("  ERRORE: dataset di training vuoto dopo il filtraggio.")
        sys.exit(1)

    val_ds = None
    if args.val and os.path.exists(args.val):
        val_ds = SFTDataset(args.val, tok, max_seq_len, sep=args.sep)

    # ── Probe corpus ──────────────────────────────────────────────────────────
    probe_ids    = None
    probe_ppl_0  = None
    if args.probe and os.path.exists(args.probe):
        probe_text = Path(args.probe).read_text(encoding='utf-8')
        raw_ids    = tok.encode(probe_text[:500_000])
        probe_ids  = torch.tensor(raw_ids, dtype=torch.long)
        probe_ppl_0 = compute_probe_ppl(model, probe_ids, max_seq_len, device)
        if probe_ppl_0 is not None:
            print(f"  Probe PPL baseline : {probe_ppl_0:.2f}")
        else:
            print(f"  Probe : corpus troppo corto (serve > {max_seq_len} token)")
            probe_ids = None

    # ── AdamW ─────────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': 0.01},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # ── Pianificazione step ────────────────────────────────────────────────────
    steps_epoch = math.ceil(len(train_ds) / args.batch)
    total_steps = args.epochs * steps_epoch

    print(f"\n  Esempi     : {len(train_ds)}  batch={args.batch}")
    print(f"  Epoche     : {args.epochs}  ({steps_epoch} step/epoca, {total_steps} totali)")
    print(f"  LR         : {args.lr:.1e} → {args.min_lr:.1e}  warmup={args.warmup_steps}")
    print(f"{'─'*60}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=_collate_pad,
        drop_last=False,
    )

    best_val_loss = float('inf')
    global_step   = 0
    t_start       = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for batch_in, batch_tgt in loader:
            batch_in  = batch_in.to(device)
            batch_tgt = batch_tgt.to(device)

            # logits: (B, T, V) — targets passati None, loss calcolata qui
            logits, _ = model(batch_in)
            loss = F.cross_entropy(
                logits.view(-1, model_cfg.vocab_size),
                batch_tgt.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            lr = get_sft_lr(global_step, total_steps, args.lr, args.min_lr,
                             args.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr

            optimizer.step()
            global_step  += 1
            epoch_losses.append(loss.item())

            if global_step % args.log_interval == 0:
                elapsed = int(time.time() - t_start)
                print(f"  ep {epoch}  step {global_step:4d}/{total_steps}"
                      f"  loss {loss.item():.4f}  lr {lr:.2e}  {elapsed}s")

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        # ── Valutazione fine epoca ────────────────────────────────────────────
        if val_ds is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vi, vt in DataLoader(val_ds, batch_size=args.batch,
                                         collate_fn=_collate_pad, shuffle=False):
                    vi, vt = vi.to(device), vt.to(device)
                    vlogits, _ = model(vi)
                    vl = F.cross_entropy(
                        vlogits.view(-1, model_cfg.vocab_size),
                        vt.view(-1),
                        ignore_index=-100,
                    )
                    val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)
            model.train()
            val_str = f"val {val_loss:.4f}"
        else:
            val_loss = avg_train_loss
            val_str  = f"train {avg_train_loss:.4f}  (no val set)"

        # ── Probe PPL ─────────────────────────────────────────────────────────
        probe_str = ""
        if probe_ids is not None:
            ppl = compute_probe_ppl(model, probe_ids, max_seq_len, device)
            if ppl is not None and probe_ppl_0 is not None:
                ratio = ppl / probe_ppl_0
                flag  = "  ⚠ FORGETTING RILEVATO" if ratio > args.forgetting_threshold else ""
                probe_str = f"  probe_ppl {ppl:.1f} (×{ratio:.3f}){flag}"

        mins, secs = divmod(int(time.time() - t_start), 60)
        print(f"\n  Epoca {epoch}/{args.epochs}  {val_str}{probe_str}  │  {mins:02d}:{secs:02d}")

        # ── Checkpoint best ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_out = {
                'step':                 ckpt['step'],
                'sft_epoch':            epoch,
                'sft_step':             global_step,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             best_val_loss,
                'model_config':         model_cfg,
                'tokenizer_state':      _get_tok_state(ckpt),
                'sft_args':             vars(args),
            }
            torch.save(ckpt_out, args.out)
            print(f"  ✓ Checkpoint → {args.out}  (val={best_val_loss:.4f})")

    mins, secs = divmod(int(time.time() - t_start), 60)
    print(f"\n{'═'*60}")
    print(f"  SFT completato in {mins}m {secs}s")
    print(f"  Best val loss : {best_val_loss:.4f}  (ppl ≈ {math.exp(best_val_loss):.1f})")
    print(f"  Checkpoint    : {args.out}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SLM — Supervised Fine-Tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  # Branch A — condominio
  python3 train_sft.py --train data/sft_cond.jsonl \\
      --out checkpoints/run5a_sft_cond.pt

  # Branch B — moto, con validazione e probe forgetting
  python3 train_sft.py --train data/sft_moto.jsonl --val data/sft_moto_val.jsonl \\
      --probe data/wiki_probe.txt --out checkpoints/run5b_sft_moto.pt

  # Branch C — scrittura, più epoche e LR più basso
  python3 train_sft.py --train data/sft_scrit.jsonl --epochs 5 --lr 1e-5 \\
      --out checkpoints/run5c_sft_scrit.pt
""")

    parser.add_argument('--train',      required=True,
                        help='file JSONL training {"prompt":..., "completion":...}')
    parser.add_argument('--val',        default=None,
                        help='file JSONL validazione (opzionale; se assente, usa loss train)')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt',
                        help='checkpoint core da fine-tune (default: checkpoints/best.pt)')
    parser.add_argument('--probe',      default=None,
                        help='file .txt per PPL probe anti-forgetting')
    parser.add_argument('--out',        default='checkpoints/run5a_sft.pt',
                        help='path checkpoint output (default: checkpoints/run5a_sft.pt)')
    parser.add_argument('--epochs',     type=int,   default=3,
                        help='epoche di fine-tuning (default: 3)')
    parser.add_argument('--batch',      type=int,   default=4,
                        help='batch size (default: 4)')
    parser.add_argument('--lr',         type=float, default=2e-5,
                        help='learning rate massimo (default: 2e-5)')
    parser.add_argument('--min-lr',     type=float, default=2e-6,
                        dest='min_lr',
                        help='learning rate minimo fine cosine (default: 2e-6)')
    parser.add_argument('--warmup-steps', type=int, default=20,
                        dest='warmup_steps',
                        help='step di warmup lineare (default: 20)')
    parser.add_argument('--grad-clip',  type=float, default=1.0,
                        dest='grad_clip',
                        help='gradient clipping norm (default: 1.0)')
    parser.add_argument('--sep',        default='\n\n',
                        help='separatore prompt/completion (default: \\n\\n)')
    parser.add_argument('--log-interval', type=int, default=10,
                        dest='log_interval',
                        help='step tra log successivi (default: 10)')
    parser.add_argument('--forgetting-threshold', type=float, default=1.15,
                        dest='forgetting_threshold',
                        help='ratio probe_ppl/baseline oltre cui avvisare (default: 1.15)')
    parser.add_argument('--device',     default='auto',
                        help='device: auto | mps | cuda | cpu (default: auto)')

    args = parser.parse_args()
    train_sft(args)
