#!/usr/bin/env python3
"""
Converte un checkpoint SLM in formato GGUF (architettura GPT-NeoX).

Mapping architettura:
  - Sequential residual (use_parallel_residual=False)
  - RoPE su d_k interi (rotation_pct=1.0)
  - GELU activation
  - No attention biases, FFN con bias

Nota tokenizer:
  Il modello SPM-BPE non include byte-fallback token (<0xXX>).
  Vengono aggiunti 256 byte token sintetici (indici vocab_size..vocab_size+255)
  con embedding zero per consentire a llama.cpp di codificare qualsiasi testo
  (inclusi caratteri speciali inseriti dal chat template) senza crash.

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

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import GPTConfig, GPT, TrainConfig  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="SLM → GGUF converter (GPT-NeoX)")
    parser.add_argument("--checkpoint", default="checkpoints/run4_best.pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.model")
    parser.add_argument("--output", default="slm-run4.gguf")
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
          f"d_ff={cfg.d_ff}  seq={cfg.max_seq_len}  d_k={d_k}")
    print(f"      step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")

    # ── 2. Carica tokenizer ────────────────────────────────────────────────
    print(f"\n[2/4] Caricamento tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab_size = sp.vocab_size()
    print(f"      vocab_size SPM={vocab_size}")

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

    # Aggiunge 256 byte-fallback token sintetici <0x00>..<0xFF>.
    # Il nostro SPM non include byte token (modello BPE senza byte_fall_back=True).
    # llama.cpp chiama byte_to_token(ch) per ogni carattere non coperto da pezzi;
    # senza questi token la funzione lancia unordered_map::at → crash immediato.
    # I token vengono aggiunti con score molto negativo per scoraggiare la
    # generazione; le righe embedding corrispondenti sono zero.
    n_byte_tokens = 256
    for byte_val in range(n_byte_tokens):
        tokens.append(f"<0x{byte_val:02X}>")
        scores.append(-10000.0)
        token_types.append(gguf.TokenType.BYTE)

    total_vocab = vocab_size + n_byte_tokens
    print(f"      vocab totale GGUF={total_vocab}  "
          f"(SPM {vocab_size} + byte-fallback {n_byte_tokens})")

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
    writer.add_parallel_residual(False)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(d_k)

    # Tokenizer SPM ("llama" = code path SPM in llama.cpp)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())

    # Helper: estrai tensore come float32 numpy
    def t(key: str) -> np.ndarray:
        return sd[key].float().numpy()

    # Embedding di input: estendi con n_byte_tokens righe zero per i byte-fallback
    emb = t("token_embedding.weight")   # [vocab_size, d_model]
    byte_pad = np.zeros((n_byte_tokens, cfg.d_model), dtype=np.float32)
    emb_extended = np.concatenate([emb, byte_pad], axis=0)  # [total_vocab, d_model]
    writer.add_tensor("token_embd.weight", emb_extended)

    # Blocchi transformer
    skipped_buffers = 0
    for i in range(cfg.num_layers):
        p = f"blocks.{i}"
        skipped_buffers += 2  # rope_cos, rope_sin

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
        writer.add_tensor(f"blk.{i}.attn_norm.weight",   t(f"{p}.norm1.weight"))
        writer.add_tensor(f"blk.{i}.attn_norm.bias",     t(f"{p}.norm1.bias"))
        writer.add_tensor(f"blk.{i}.ffn_up.weight",      t(f"{p}.ff.net.0.weight"))
        writer.add_tensor(f"blk.{i}.ffn_up.bias",        t(f"{p}.ff.net.0.bias"))
        writer.add_tensor(f"blk.{i}.ffn_down.weight",    t(f"{p}.ff.net.2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_down.bias",      t(f"{p}.ff.net.2.bias"))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",    t(f"{p}.norm2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias",      t(f"{p}.norm2.bias"))

    # Norma finale + testa LM (estesa con zero rows come l'embedding)
    writer.add_tensor("output_norm.weight", t("norm_final.weight"))
    writer.add_tensor("output_norm.bias",   t("norm_final.bias"))
    lm_head = t("lm_head.weight")          # [vocab_size, d_model]
    lm_head_extended = np.concatenate([lm_head, byte_pad], axis=0)  # [total_vocab, d_model]
    writer.add_tensor("output.weight", lm_head_extended)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ── 4. Riepilogo ──────────────────────────────────────────────────────
    out_size = Path(args.output).stat().st_size
    print(f"\n[4/4] GGUF scritto:")
    print(f"      {args.output}  ({out_size / 1e6:.1f} MB fp32)")
    print(f"      Vocab: {total_vocab} token ({vocab_size} SPM + {n_byte_tokens} byte-fallback)")
    print(f"      Tensori scritti: {2 + cfg.num_layers * 12}")
    print(f"      Buffer saltati: {skipped_buffers} (rope_cos/rope_sin)")
    print()
    print("Test inferenza (fp32):")
    print(f"  llama-cli --model {args.output} -p 'Il Parlamento europeo' -n 50")
    print()
    print("Per quantizzare a Q8_0:")
    print(f"  llama-quantize {args.output} slm-q8_0.gguf Q8_0")


if __name__ == "__main__":
    main()
