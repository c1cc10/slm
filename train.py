"""
SLM — train.py

Pipeline completo: corpus → tokenizer → dataset → training loop AdamW.

Tokenizer supportati:
  char        — char-level, nessuna dipendenza (default, Fase 1)
  bpe-tiktoken — BPE pre-addestrato via tiktoken  (pip install tiktoken)
  bpe-spm      — BPE personalizzato via SentencePiece (pip install sentencepiece)

Esempi:
    python3 train.py                                   # char-level
    python3 train.py --tokenizer bpe-tiktoken          # tiktoken cl100k_base
    python3 train.py --tokenizer bpe-spm \\
        --bpe-corpus data/corpus.txt --bpe-vocab 8000  # SPM: allena + usa
    python3 train.py --tokenizer bpe-spm               # SPM: riusa tokenizer esistente
    python3 train.py --resume                          # riprende da checkpoint
    python3 train.py --generate "Il progetto"          # genera testo
"""

import os, sys, math, time, argparse, urllib.request
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

from model import GPT, GPTConfig


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Dati
    data_path:     str   = 'data/corpus.txt'
    out_dir:       str   = 'checkpoints'
    val_split:     float = 0.1

    # Sequenze
    batch_size:    int   = 32
    seq_len:       int   = 256

    # Durata training
    max_steps:     int   = 5000

    # AdamW
    lr:            float = 3e-4
    min_lr:        float = 3e-5
    warmup_steps:  int   = 200
    weight_decay:  float = 0.1
    betas:         tuple = (0.9, 0.95)

    # Gradient clipping
    grad_clip:     float = 1.0

    # Log / checkpoint
    eval_interval: int   = 500
    eval_steps:    int   = 100
    log_interval:  int   = 50
    sample_len:    int   = 200

    # Device
    device:        str   = 'auto'    # 'auto' | 'mps' | 'cuda' | 'cpu'

    # Continued pre-training
    resume:        bool  = False

    # Tokenizer
    tokenizer:     str   = 'char'          # 'char' | 'bpe-tiktoken' | 'bpe-spm'
    bpe_encoding:  str   = 'cl100k_base'   # solo per bpe-tiktoken
    bpe_corpus:    str   = ''              # corpus per addestrare SPM (prima corsa)
    bpe_vocab:     int   = 8000            # vocab size per SPM

    # Architettura modello (default = small ~5M)
    # Usa --preset medium per il config da 46M parametri
    d_model:       int   = 256
    num_layers:    int   = 6
    num_heads:     int   = 8
    d_ff:          int   = 1024

    # Cache token pre-codificati su disco (evita re-encoding ad ogni avvio)
    no_cache:      bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER — interfaccia comune, tre implementazioni
# ─────────────────────────────────────────────────────────────────────────────
#
# Tutti e tre espongono:
#   tok.encode(text)    → List[int]
#   tok.decode(ids)     → str
#   tok.vocab_size      → int
#   tok.sample_token()  → int   (token iniziale per la generazione)
#   tok.print_info()            (stampa dettagli vocabolario)
#   tok.save_state()    → dict  (salvato nel checkpoint, usato da load_tokenizer)
#   tok.compat_key()    → hashable (per il controllo di compatibilità al resume)
#
# Il training loop non sa quale backend sta usando.


class CharTokenizer:
    """
    Vocabolario dai caratteri unici nel corpus.
    vocab_size tipico: 80–120. Zero dipendenze. Backend di default (Fase 1).
    """
    def __init__(self, text=None, state=None):
        if state is not None:
            self._stoi = state['stoi']
            self._itos = {int(k): v for k, v in state['itos'].items()}
        else:
            chars = sorted(set(text))
            self._stoi = {c: i for i, c in enumerate(chars)}
            self._itos = {i: c for c, i in self._stoi.items()}

    @property
    def vocab_size(self): return len(self._stoi)

    def encode(self, text):
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids):
        return ''.join(self._itos.get(i, '?') for i in ids)

    def sample_token(self):
        return 0

    def print_info(self):
        sample = ''.join(list(self._stoi.keys())[:40])
        ellipsis = '...' if self.vocab_size > 40 else ''
        print(f"  Chars     : {repr(sample)}{ellipsis}")

    def save_state(self):
        return {'type': 'char', 'stoi': self._stoi, 'itos': self._itos}

    def compat_key(self):
        # due tokenizer char sono compatibili se hanno lo stesso set di caratteri
        return frozenset(self._stoi.keys())


class TiktokenTokenizer:
    """
    BPE via tiktoken — vocabolari pre-addestrati da OpenAI.
    Non richiede corpus di training. vocab_size = 100,277 (cl100k_base).

    pip install tiktoken
    """
    def __init__(self, encoding='cl100k_base', state=None):
        try:
            import tiktoken as _tt
        except ImportError:
            print("  ERRORE: tiktoken non installato.  pip install tiktoken")
            sys.exit(1)
        enc_name   = state['encoding'] if state else encoding
        self._enc  = _tt.get_encoding(enc_name)
        self._name = enc_name

    @property
    def vocab_size(self): return self._enc.n_vocab

    def encode(self, text):
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids):
        try:
            return self._enc.decode(ids)
        except Exception:
            return self._enc.decode([i for i in ids if i < self.vocab_size])

    def sample_token(self):
        return self._enc.encode(' ')[0]

    def print_info(self):
        print(f"  Encoding  : {self._name}  ({self.vocab_size:,} token)")

    def save_state(self):
        return {'type': 'bpe-tiktoken', 'encoding': self._name}

    def compat_key(self):
        return ('tiktoken', self._name)


class SPMTokenizer:
    """
    BPE via SentencePiece — addestrato sul tuo corpus.
    vocab_size configurabile (tipico: 4.000–32.000).

    pip install sentencepiece

    Prima corsa:  --bpe-corpus data/corpus.txt --bpe-vocab 8000
                  → allena e salva checkpoints/tokenizer.model
    Corse successive: carica automaticamente il modello esistente.
    """
    def __init__(self, model_path=None, state=None):
        try:
            import sentencepiece as _spm
        except ImportError:
            print("  ERRORE: sentencepiece non installato.  pip install sentencepiece")
            sys.exit(1)
        self._sp = _spm.SentencePieceProcessor()
        if state is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
                f.write(state['model_bytes'])
                tmp = f.name
            self._sp.load(tmp)
            os.unlink(tmp)
            self._path  = None
            self._bytes = state['model_bytes']
        else:
            self._sp.load(model_path)
            self._path = model_path
            with open(model_path, 'rb') as f:
                self._bytes = f.read()

    @property
    def vocab_size(self): return self._sp.vocab_size()

    def encode(self, text):
        return self._sp.encode(text, out_type=int)

    def decode(self, ids):
        return self._sp.decode(ids)

    def sample_token(self):
        bos = self._sp.bos_id()
        return bos if bos >= 0 else 0

    def print_info(self):
        path = self._path or '(da checkpoint)'
        print(f"  SPM model : {path}  ({self.vocab_size:,} token)")

    def save_state(self):
        return {'type': 'bpe-spm', 'model_bytes': self._bytes}

    def compat_key(self):
        # I primi 64 byte del modello come firma: modelli diversi hanno byte diversi
        return ('spm', self._bytes[:64])

    @classmethod
    def train(cls, corpus_path, vocab_size, out_prefix):
        try:
            import sentencepiece as _spm
        except ImportError:
            print("  ERRORE: sentencepiece non installato.  pip install sentencepiece")
            sys.exit(1)
        os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
        print(f"  Training tokenizer BPE: vocab_size={vocab_size}, corpus={corpus_path}")
        _spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=out_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9999,
            model_type='bpe',
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        model_path = out_prefix + '.model'
        print(f"  Tokenizer salvato: {model_path}")
        return cls(model_path=model_path)


def make_tokenizer(cfg: TrainConfig, text=None):
    """Crea il tokenizer corretto in base a cfg.tokenizer."""
    if cfg.tokenizer == 'char':
        if text is None:
            raise ValueError("CharTokenizer richiede il testo del corpus")
        return CharTokenizer(text=text)

    elif cfg.tokenizer == 'bpe-tiktoken':
        return TiktokenTokenizer(encoding=cfg.bpe_encoding)

    elif cfg.tokenizer == 'bpe-spm':
        model_path = os.path.join(cfg.out_dir, 'tokenizer.model')
        if os.path.exists(model_path):
            print(f"  Tokenizer esistente: {model_path}")
            return SPMTokenizer(model_path=model_path)
        elif cfg.bpe_corpus:
            return SPMTokenizer.train(
                corpus_path=cfg.bpe_corpus,
                vocab_size=cfg.bpe_vocab,
                out_prefix=os.path.join(cfg.out_dir, 'tokenizer'),
            )
        else:
            print(f"  ERRORE: nessun tokenizer SPM in {model_path}")
            print(f"  Prima corsa: aggiungi  --bpe-corpus data/corpus.txt --bpe-vocab 8000")
            sys.exit(1)

    else:
        print(f"  ERRORE: tokenizer '{cfg.tokenizer}' non riconosciuto.")
        print(f"  Opzioni: char | bpe-tiktoken | bpe-spm")
        sys.exit(1)


def load_tokenizer(state: dict):
    """Ricostruisce il tokenizer dallo stato salvato nel checkpoint."""
    t = state.get('type', 'char')
    if t == 'char':
        return CharTokenizer(state=state)
    elif t == 'bpe-tiktoken':
        return TiktokenTokenizer(state=state)
    elif t == 'bpe-spm':
        return SPMTokenizer(state=state)
    else:
        raise ValueError(f"Tipo tokenizer sconosciuto nel checkpoint: {t}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — FINESTRA SCORREVOLE (SLIDING WINDOW)
# ─────────────────────────────────────────────────────────────────────────────
# La finestra di seq_len+1 token produce:
#   input  = [t_i,   ..., t_i+seq_len-1]
#   target = [t_i+1, ..., t_i+seq_len  ]   (spostato di 1)
#
# Il modello vede input[t] e deve prevedere target[t] = input[t+1].

class TextDataset:
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data    = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, batch_size: int, device: str):
        n = len(self)
        if n == 0:
            raise ValueError(
                f"Dataset troppo piccolo: {len(self.data)} token con seq_len={self.seq_len}. "
                f"Servono almeno {self.seq_len + 2} token. Usa un corpus più grande."
            )
        ix = torch.randint(n, (batch_size,))
        x  = torch.stack([self.data[i   : i + self.seq_len    ] for i in ix])
        y  = torch.stack([self.data[i+1 : i + self.seq_len + 1] for i in ix])
        return x.to(device), y.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE: WARMUP LINEARE + COSINE DECAY
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# STIMA LOSS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, cfg: TrainConfig, device: str) -> dict:
    model.eval()
    losses = {}
    for name, ds in [('train', train_ds), ('val', val_ds)]:
        L = [model(*ds.get_batch(cfg.batch_size, device))[1].item()
             for _ in range(cfg.eval_steps)]
        losses[name] = sum(L) / len(L)
    model.train()
    return losses


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD CORPUS DI PROVA
# ─────────────────────────────────────────────────────────────────────────────

def download_sample_corpus(out_path: str = 'data/corpus.txt'):
    url = "https://www.gutenberg.org/files/25790/25790-0.txt"
    print(f"Download corpus di prova da Project Gutenberg...")
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  Errore: {e}\n  Metti un file .txt in {out_path}")
        sys.exit(1)
    start = raw.find("*** START OF THE PROJECT")
    end   = raw.find("*** END OF THE PROJECT")
    if start != -1: raw = raw[raw.find('\n', start)+1:]
    if end   != -1: raw = raw[:end]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(raw.strip())
    print(f"  Salvato: {out_path}  ({os.path.getsize(out_path)//1024} KB)")
    print(f"  Ora esegui:  python3 train.py\n")


# ─────────────────────────────────────────────────────────────────────────────
# CACHE TOKEN
# ─────────────────────────────────────────────────────────────────────────────

def _token_cache_path(data_path: str, vocab_size: int, tokenizer_type: str) -> str:
    """Percorso del file .pt con i token pre-codificati.
    Convenzione: <corpus>.bpe-spm.v16000.tokens.pt
    """
    return f"{data_path}.{tokenizer_type}.v{vocab_size}.tokens.pt"


def _cache_is_valid(cache_path: str, data_path: str, spm_model_path) -> bool:
    """True se la cache esiste ed è più recente sia del corpus che del tokenizer SPM."""
    if not os.path.exists(cache_path):
        return False
    t_cache = os.path.getmtime(cache_path)
    if os.path.getmtime(data_path) > t_cache:
        return False
    if spm_model_path and os.path.exists(spm_model_path):
        if os.path.getmtime(spm_model_path) > t_cache:
            return False
    return True


def _encode_with_progress(tok, text: str):
    """Encoding a chunk con percentuale e ETA stampati a schermo.

    Divide il testo in blocchi da 5MB, codifica ciascuno e stampa il progresso.
    Più lento di una singola chiamata encode() ma mostra avanzamento in tempo reale.
    """
    CHUNK   = 5_000_000   # 5 MB di testo per iterazione
    total   = len(text)
    all_ids = []
    t0      = time.time()

    for start in range(0, total, CHUNK):
        chunk = text[start : start + CHUNK]
        all_ids.extend(tok.encode(chunk))

        done    = min(start + CHUNK, total)
        pct     = done / total * 100
        elapsed = time.time() - t0
        eta     = (elapsed / max(pct, 0.01)) * (100 - pct)
        print(
            f"\r  Encoding  {pct:5.1f}%  |  {len(all_ids):>12,} token  |  ETA {eta:5.0f}s   ",
            end='', flush=True,
        )

    elapsed_total = time.time() - t0
    print(f"\r  Encoding  100.0%  |  {len(all_ids):>12,} token  |  {elapsed_total:.0f}s totali          ")
    return all_ids


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig):

    # ── Device ────────────────────────────────────────────────────────────────
    if cfg.device == 'auto':
        if torch.backends.mps.is_available():   device = 'mps'
        elif torch.cuda.is_available():         device = 'cuda'
        else:                                   device = 'cpu'
    else:
        device = cfg.device

    print(f"\n{'═'*60}")
    print(f"  SLM — Training  [{cfg.tokenizer}]")
    print(f"{'═'*60}")
    print(f"  Device : {device}")

    # ── Corpus ────────────────────────────────────────────────────────────────
    if not os.path.exists(cfg.data_path):
        print(f"\n  ERRORE: corpus non trovato in '{cfg.data_path}'")
        print(f"    python3 train.py --download   scarica corpus di prova")
        sys.exit(1)

    print(f"\n  Corpus : {cfg.data_path}")
    text = Path(cfg.data_path).read_text(encoding='utf-8')
    print(f"  Testo  : {len(text):,} caratteri")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok        = make_tokenizer(cfg, text)
    vocab_size = tok.vocab_size
    print(f"  Vocab  : {vocab_size:,}  (tokenizer={cfg.tokenizer})")
    tok.print_info()

    # ── Encoding con cache ────────────────────────────────────────────────────
    spm_path   = getattr(tok, '_path', None)   # None per char/tiktoken
    cache_path = _token_cache_path(cfg.data_path, vocab_size, cfg.tokenizer)
    if not cfg.no_cache and _cache_is_valid(cache_path, cfg.data_path, spm_path):
        print(f"  Cache   : {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        print(f"  Token   : {len(data):,}  (da cache — avvio istantaneo)")
    else:
        reason = ' (--no-cache)' if cfg.no_cache else ''
        print(f"  Encoding corpus{reason}  ({len(text):,} caratteri → token)...")
        ids  = _encode_with_progress(tok, text)
        data = torch.tensor(ids, dtype=torch.long)
        torch.save(data, cache_path)
        mb = os.path.getsize(cache_path) / 1024**2
        print(f"  Cache   : {cache_path}  ({mb:.0f} MB)")

    # ── Split train / val ─────────────────────────────────────────────────────
    n_val   = int(len(data) * cfg.val_split)
    tr_data = data[:-n_val]
    vl_data = data[-n_val:]
    print(f"  Train  : {len(tr_data):,} token")
    print(f"  Val    : {len(vl_data):,} token")

    train_ds = TextDataset(tr_data, cfg.seq_len)
    val_ds   = TextDataset(vl_data, cfg.seq_len)

    min_tokens = cfg.seq_len + 2
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"\n  ERRORE: corpus troppo piccolo per seq_len={cfg.seq_len}.")
        print(f"  Train: {len(tr_data)} token  Val: {len(vl_data)} token  (serve ≥{min_tokens})")
        sys.exit(1)

    # ── Modello ───────────────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size  = vocab_size,
        d_model     = cfg.d_model,
        num_heads   = cfg.num_heads,
        num_layers  = cfg.num_layers,
        d_ff        = cfg.d_ff,
        max_seq_len = cfg.seq_len,
        dropout     = 0.1,
    )
    print(f"\n  Caricamento modello su {device}...", end='', flush=True)
    t_mps = time.time()
    model    = GPT(model_cfg).to(device)
    n_params = model.count_params()
    print(f" {time.time()-t_mps:.1f}s")
    print(f"  Parametri : {n_params:,}  ({n_params*4/1024**2:.1f} MB fp32)")

    # ── AdamW ────────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params,    'weight_decay': cfg.weight_decay},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=cfg.lr, betas=cfg.betas,
    )
    print(f"  AdamW     : {sum(p.numel() for p in decay_params):,} param con decay, "
          f"{sum(p.numel() for p in no_decay_params):,} senza")

    os.makedirs(cfg.out_dir, exist_ok=True)

    # ── Resume da checkpoint ──────────────────────────────────────────────────
    # VINCOLO COMPATIBILITÀ: due tokenizer sono compatibili se producono lo
    # stesso vocabolario — compat_key() cattura questa identità in modo
    # specifico per ogni tipo (set di caratteri, nome encoding, firma SPM).
    best_val_loss = float('inf')
    resumed_step  = 0

    if cfg.resume:
        ckpt_path = os.path.join(cfg.out_dir, 'best.pt')
        if not os.path.exists(ckpt_path):
            print(f"\n  AVVISO: --resume attivo ma {ckpt_path} non trovato. Training da zero.")
        else:
            ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
            saved_tok = load_tokenizer(_get_tok_state(ckpt))

            if saved_tok.compat_key() != tok.compat_key():
                print(f"\n  AVVISO: tokenizer del checkpoint incompatibile con quello corrente.")
                print(f"  Checkpoint : {saved_tok.save_state()['type']}  "
                      f"vocab={saved_tok.vocab_size}")
                print(f"  Corrente   : {tok.save_state()['type']}  "
                      f"vocab={tok.vocab_size}")
                print(f"  Training da zero.")
            else:
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                best_val_loss    = ckpt['val_loss']
                resumed_step     = ckpt['step']
                cfg.warmup_steps = min(100, cfg.max_steps // 20)
                print(f"\n  Ripreso da step {resumed_step}  val_loss={best_val_loss:.4f}")
                print(f"  Re-warmup: {cfg.warmup_steps} step")

    # ── Loop di training ──────────────────────────────────────────────────────
    total_tokens = cfg.max_steps * cfg.batch_size * cfg.seq_len
    print(f"\n  Steps  : {cfg.max_steps}  batch={cfg.batch_size}  seq={cfg.seq_len}")
    print(f"  Token totali training : {total_tokens:,}")
    print(f"  Loss attesa (inizio)  : {math.log(vocab_size):.3f}  (= log({vocab_size}))")
    print(f"\n{'─'*60}")

    t_start = time.time()

    for step in range(cfg.max_steps + 1):

        # ── Valutazione periodica ─────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            losses     = estimate_loss(model, train_ds, val_ds, cfg, device)
            mins, secs = divmod(int(time.time() - t_start), 60)
            print(f"\nStep {step:5d}/{cfg.max_steps}"
                  f"  │  train {losses['train']:.4f}"
                  f"  val {losses['val']:.4f}"
                  f"  │  {mins:02d}:{secs:02d}")

            if step > 0:
                model.eval()
                seed = torch.tensor([[tok.sample_token()]], dtype=torch.long, device=device)
                out  = model.generate(seed, max_new_tokens=cfg.sample_len,
                                      temperature=0.8, top_k=20)
                sample    = tok.decode(out[0].tolist())
                printable = ''.join(c if c.isprintable() else '·' for c in sample)
                print(f"  ▶ {repr(printable[:120])}")
                model.train()

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                ckpt = {
                    'step':                 step,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':             best_val_loss,
                    'model_config':         model_cfg,
                    'tokenizer_state':      tok.save_state(),
                    'train_config':         cfg,
                }
                ckpt_path = os.path.join(cfg.out_dir, 'best.pt')
                torch.save(ckpt, ckpt_path)
                print(f"  ✓ Checkpoint → {ckpt_path}  (val={best_val_loss:.4f})")

        if step == cfg.max_steps:
            break

        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr

        x, y = train_ds.get_batch(cfg.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0 and step > 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}")

    mins, secs = divmod(int(time.time() - t_start), 60)
    print(f"\n{'═'*60}")
    print(f"  Completato in {mins}m {secs}s  |  best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {cfg.out_dir}/best.pt")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _get_tok_state(ckpt: dict) -> dict:
    """Legge lo stato del tokenizer dal checkpoint, supportando il vecchio formato."""
    if 'tokenizer_state' in ckpt:
        return ckpt['tokenizer_state']
    # backward compat: checkpoint pre-refactoring usava 'vocab'
    return {'type': 'char', **ckpt['vocab']}


def generate_from_checkpoint(ckpt_path: str, prompt: str = '',
                              max_tokens: int = 300, temperature: float = 0.8,
                              top_k: int = 20):
    print(f"Caricamento: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tok  = load_tokenizer(_get_tok_state(ckpt))

    if torch.backends.mps.is_available():   device = 'mps'
    elif torch.cuda.is_available():         device = 'cuda'
    else:                                   device = 'cpu'

    model = GPT(ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tok_type = ckpt.get('tokenizer_state', {}).get('type', 'char')
    print(f"  Tokenizer : {tok_type}  vocab={tok.vocab_size:,}")
    print(f"  Step      : {ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    print(f"  Device    : {device}")

    ids = tok.encode(prompt) if prompt else [tok.sample_token()]
    if not ids:
        print("  Attenzione: prompt non codificabile, uso token iniziale.")
        ids = [tok.sample_token()]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k)
    result = tok.decode(out[0].tolist())
    print(f"\n{'─'*60}\n{result}\n{'─'*60}\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SLM — training  (char | bpe-tiktoken | bpe-spm)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""esempi:
  python3 train.py
  python3 train.py --tokenizer bpe-tiktoken
  python3 train.py --tokenizer bpe-spm --bpe-corpus data/corpus.txt --bpe-vocab 8000
  python3 train.py --tokenizer bpe-spm --resume
  python3 train.py --generate "Il progetto"
""")
    parser.add_argument('--download',     action='store_true',
                        help='scarica corpus di prova (I Promessi Sposi)')
    parser.add_argument('--generate',    type=str, default=None,
                        help='genera testo dal checkpoint best.pt')
    parser.add_argument('--data',        type=str, default='data/corpus.txt')
    parser.add_argument('--steps',       type=int, default=5000)
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--device',      type=str, default='auto')
    parser.add_argument('--resume',      action='store_true',
                        help='riprende da checkpoints/best.pt')
    parser.add_argument('--tokenizer',   type=str, default='char',
                        choices=['char', 'bpe-tiktoken', 'bpe-spm'],
                        help='backend tokenizer (default: char)')
    parser.add_argument('--bpe-encoding', type=str, default='cl100k_base',
                        help='encoding tiktoken (default: cl100k_base)')
    parser.add_argument('--bpe-corpus',  type=str, default='',
                        help='corpus per addestrare tokenizer SPM (prima corsa)')
    parser.add_argument('--bpe-vocab',   type=int, default=8000,
                        help='vocab size per SPM (default: 8000)')
    parser.add_argument('--no-cache',    action='store_true',
                        help='forza re-encoding del corpus ignorando la cache token')
    parser.add_argument('--preset',      type=str, default='small',
                        choices=['small', 'medium'],
                        help='preset architettura: small=~5M (default), medium=~46M')
    parser.add_argument('--d-model',     type=int, default=None,
                        help='dimensione embedding (override preset)')
    parser.add_argument('--num-layers',  type=int, default=None,
                        help='numero di blocchi Transformer (override preset)')
    parser.add_argument('--num-heads',   type=int, default=None,
                        help='teste di attenzione (override preset)')
    parser.add_argument('--d-ff',        type=int, default=None,
                        help='dimensione FFN interna (override preset)')
    parser.add_argument('--seq-len',     type=int, default=None,
                        help='lunghezza sequenza in token (override preset)')
    args = parser.parse_args()

    # ── Preset architettura ────────────────────────────────────────────────────
    _PRESETS = {
        'small':  dict(d_model=256, num_layers=6,  num_heads=8, d_ff=1024, seq_len=256),
        'medium': dict(d_model=512, num_layers=12, num_heads=8, d_ff=2048, seq_len=512),
    }
    arch = dict(_PRESETS[args.preset])
    if args.d_model    is not None: arch['d_model']    = args.d_model
    if args.num_layers is not None: arch['num_layers'] = args.num_layers
    if args.num_heads  is not None: arch['num_heads']  = args.num_heads
    if args.d_ff       is not None: arch['d_ff']       = args.d_ff
    if args.seq_len    is not None: arch['seq_len']    = args.seq_len

    if args.download:
        download_sample_corpus()
        sys.exit(0)

    if args.generate is not None:
        generate_from_checkpoint('checkpoints/best.pt', prompt=args.generate)
        sys.exit(0)

    cfg = TrainConfig(
        data_path    = args.data,
        max_steps    = args.steps,
        batch_size   = args.batch,
        seq_len      = arch['seq_len'],
        device       = args.device,
        resume       = args.resume,
        tokenizer    = args.tokenizer,
        bpe_encoding = args.bpe_encoding,
        bpe_corpus   = args.bpe_corpus,
        bpe_vocab    = args.bpe_vocab,
        no_cache     = args.no_cache,
        d_model      = arch['d_model'],
        num_layers   = arch['num_layers'],
        num_heads    = arch['num_heads'],
        d_ff         = arch['d_ff'],
    )
    train(cfg)
