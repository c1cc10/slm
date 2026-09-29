#!/usr/bin/env python3
"""
Converte il checkpoint SLM Run #4 in formato GGUF (architettura GPT-NeoX).

Mapping architettura:
  - Sequential residual (use_parallel_residual=False)
  - RoPE su d_k interi (rotation_pct=1.0)
  - GELU activation
  - No attention biases, FFN con bias

Pesi saltati:
  - blocks.{i}.attention.rope_cos / rope_sin  (buffers, llama.cpp li ricalcola)

Uso:
    python3 tools/convert_to_gguf.py \\
        --checkpoint checkpoints/run4_best.pt \\
        --tokenizer checkpoints/tokenizer.model \\
        --output slm-run4.gguf

Poi quantizzare:
    llama-quantize slm-run4.gguf slm-run4-q4_k_m.gguf Q4_K_M
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Importa le classi del modello dal progetto
sys.path.insert(0, str(Path(__file__).parent.parent))
from train import GPTConfig, GPT, TrainConfig  # noqa: F401  (serve per torch.load)


def main():
    parser = argparse.ArgumentParser(description="SLM → GGUF converter (GPT-NeoX)")
    parser.add_argument("--checkpoint", default="checkpoints/run4_best.pt",
                        help="Percorso checkpoint .pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.model",
                        help="Percorso modello SentencePiece .model")
    parser.add_argument("--output", default="slm-run4.gguf",
                        help="File GGUF di output (fp32)")
    args = parser.parse_args()

    import gguf
    import sentencepiece as spm

    # ── 1. Carica checkpoint ───────────────────────────────────────────────
    print(f"[1/4] Caricamento checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg: GPTConfig = ckpt["model_config"]
    sd: dict = ckpt["model_state_dict"]

    d_k = cfg.d_model // cfg.num_heads
    print(f"      vocab={cfg.vocab_size}  d_model={cfg.d_model}  "
          f"layers={cfg.num_layers}  heads={cfg.num_heads}  "
          f"d_ff={cfg.d_ff}  seq={cfg.max_seq_len}  d_k={d_k}  RoPE={cfg.use_rope}")
    print(f"      step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")

    # ── 2. Carica tokenizer ────────────────────────────────────────────────
    print(f"\n[2/4] Caricamento tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.vocab_size()
    print(f"      vocab_size={vocab_size}")

    tokens, scores, token_types = [], [], []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        if sp.IsUnknown(i):    t = gguf.TokenType.UNKNOWN
        elif sp.IsControl(i):  t = gguf.TokenType.CONTROL
        elif sp.IsUnused(i):   t = gguf.TokenType.UNUSED
        elif sp.IsByte(i):     t = gguf.TokenType.BYTE
        else:                  t = gguf.TokenType.NORMAL
        tokens.append(piece)
        scores.append(score)
        token_types.append(t)

    # llama.cpp cerca un token '\n' via .at() senza try-catch nel path gptneox.
    # Il corpus Wikipedia non aveva newline dopo il parsing → token assente.
    # Aggiungiamo '\n' sintetico come ultimo token (ID=vocab_size).
    # Le tabelle di embedding vengono estese con una riga zero.
    nl_id = vocab_size
    tokens.append("\n")
    scores.append(0.0)
    token_types.append(gguf.TokenType.NORMAL)
    print(f"      token '\\n' aggiunto come ID {nl_id} (vocab esteso a {vocab_size + 1})")

    # ── 3. Scrive header GGUF + metadati + tensori ─────────────────────────
    print(f"\n[3/4] Scrittura GGUF: {args.output}")
    writer = gguf.GGUFWriter(args.output, "gptneox")

    # Iperparametri modello
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.d_model)
    writer.add_block_count(cfg.num_layers)
    writer.add_feed_forward_length(cfg.d_ff)
    writer.add_head_count(cfg.num_heads)
    writer.add_layer_norm_eps(1e-5)
    writer.add_parallel_residual(False)       # sequential (non Pythia-style)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(d_k)      # rotazione 100% di d_k

    # Tokenizer (SentencePiece BPE)
    writer.add_tokenizer_model("llama")       # llama.cpp usa "llama" per SPM
    writer.add_tokenizer_pre("spm")           # pre-tokenizer SentencePiece (▁ come marcatore spazio)
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())

    # Helper: estrai tensore come float32 numpy
    def t(key: str) -> np.ndarray:
        return sd[key].float().numpy()

    # Embedding di input — esteso con riga zero per il token '\n' sintetico
    emb = t("token_embedding.weight")                            # [vocab_size, d_model]
    emb_ext = np.vstack([emb, np.zeros((1, cfg.d_model), dtype=np.float32)])
    writer.add_tensor("token_embd.weight", emb_ext)

    # Blocchi transformer
    skipped_buffers = 0
    for i in range(cfg.num_layers):
        p = f"blocks.{i}"

        # rope_cos / rope_sin sono buffer (non parametri) → saltati
        # llama.cpp ricalcola internamente i valori RoPE
        skipped_buffers += 2

        # Attenzione: llama.cpp si aspetta QKV fusi [3*d_model, d_model]
        # Il nostro modello non ha attention biases → aggiungiamo zero bias
        qkv = np.concatenate([
            t(f"{p}.attention.W_q.weight"),
            t(f"{p}.attention.W_k.weight"),
            t(f"{p}.attention.W_v.weight"),
        ], axis=0)
        qkv_bias = np.zeros(3 * cfg.d_model, dtype=np.float32)
        out_bias = np.zeros(cfg.d_model, dtype=np.float32)
        writer.add_tensor(f"blk.{i}.attn_qkv.weight",    qkv)
        writer.add_tensor(f"blk.{i}.attn_qkv.bias",      qkv_bias)
        writer.add_tensor(f"blk.{i}.attn_output.weight", t(f"{p}.attention.W_o.weight"))
        writer.add_tensor(f"blk.{i}.attn_output.bias",   out_bias)

        # LayerNorm prima dell'attenzione
        writer.add_tensor(f"blk.{i}.attn_norm.weight",   t(f"{p}.norm1.weight"))
        writer.add_tensor(f"blk.{i}.attn_norm.bias",     t(f"{p}.norm1.bias"))

        # FFN (con bias)
        writer.add_tensor(f"blk.{i}.ffn_up.weight",      t(f"{p}.ff.net.0.weight"))
        writer.add_tensor(f"blk.{i}.ffn_up.bias",        t(f"{p}.ff.net.0.bias"))
        writer.add_tensor(f"blk.{i}.ffn_down.weight",    t(f"{p}.ff.net.2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_down.bias",      t(f"{p}.ff.net.2.bias"))

        # LayerNorm prima della FFN
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",    t(f"{p}.norm2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias",      t(f"{p}.norm2.bias"))

    # Norma finale + testa LM — output.weight esteso con riga zero per '\n' sintetico
    lm_head = t("lm_head.weight")                                # [vocab_size, d_model]
    lm_ext  = np.vstack([lm_head, np.zeros((1, cfg.d_model), dtype=np.float32)])
    writer.add_tensor("output_norm.weight", t("norm_final.weight"))
    writer.add_tensor("output_norm.bias",   t("norm_final.bias"))
    writer.add_tensor("output.weight",      lm_ext)

    # Scrittura su disco
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ── 4. Riepilogo ──────────────────────────────────────────────────────
    out_size = Path(args.output).stat().st_size
    print(f"\n[4/4] GGUF scritto:")
    print(f"      {args.output}  ({out_size / 1e6:.1f} MB fp32)")
    print(f"      Tensor scritti: {3 + cfg.num_layers * 12}")
    print(f"      Buffer saltati: {skipped_buffers} (rope_cos/rope_sin)")
    print()
    print("Per quantizzare a Q4_K_M (~35 MB):")
    print(f"  llama-quantize {args.output} slm-run4-q4_k_m.gguf Q4_K_M")
    print()
    print("Test inferenza (fp32):")
    print(f"  llama-cli --model {args.output} -p 'Il Parlamento europeo' -n 50 --no-mmap")
    print()
    print("Test inferenza (quantizzato):")
    print("  llama-cli --model slm-run4-q4_k_m.gguf -p 'Il Parlamento europeo' -n 50")


if __name__ == "__main__":
    main()
