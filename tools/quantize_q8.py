#!/usr/bin/env python3
"""
Quantizzazione Q8_0 in Python puro per GGUF SLM.

Perché questo script esiste
---------------------------
llama-quantize v0.5.0 ha rimosso GPT-NeoX dalla whitelist di architetture
supportate per la quantizzazione. Questo script produce lo stesso file Q8_0
che produrrebbe llama-quantize se GPT-NeoX fosse nella whitelist.

Il file prodotto è un normale GGUF Q8_0 caricabile da qualsiasi client
llama.cpp (llama-cli, llama-simple, llama-server): la dequantizzazione
a runtime è architecture-agnostic.

Formato Q8_0
------------
Ogni blocco di 32 float32 viene rappresentato come:
  [scale_f16 (2 byte)] + [32 × int8]  = 34 byte
con  int8 = clip(round(x / scale), -127, 127)
     scale = max(|x_i|) / 127

Un tensore [768, 512] fp32 occupa 1.5 MB → Q8_0 occupa ~0.52 MB (34.7%).

Uso
---
    python3 tools/quantize_q8.py --checkpoint checkpoints/run4_best.pt \
                                  --tokenizer checkpoints/tokenizer.model \
                                  --output slm-run4-q8_0.gguf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import GPTConfig, GPT, TrainConfig  # noqa: F401

Q8_BLOCK = 32


def quantize_q8_0(arr: np.ndarray) -> np.ndarray:
    """
    Quantizza un array float32 in formato GGUF Q8_0.
    Restituisce i byte raw: per ogni blocco di 32 elementi → [f16 scale][32×int8].
    """
    flat = arr.astype(np.float32).flatten()
    pad = (-len(flat)) % Q8_BLOCK
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])

    blocks  = flat.reshape(-1, Q8_BLOCK)
    scales  = np.max(np.abs(blocks), axis=1) / 127.0
    scales  = np.where(scales == 0, 1.0, scales)
    quant   = np.clip(np.round(blocks / scales[:, None]), -127, 127).astype(np.int8)

    out = bytearray()
    for s, q in zip(scales.astype(np.float16), quant):
        out += s.tobytes()   # 2 byte
        out += q.tobytes()   # 32 byte
    return np.frombuffer(bytes(out), dtype=np.uint8)


def q8_byte_shape(numpy_shape: tuple) -> list:
    """
    Converte shape numpy [R, C] in GGUF Q8_0 byte-shape per raw_shape.

    La dimensione INTERNA è C (colonne, contigue in memoria per array C-contiguous).
    Q8_0 quantizza blocchi di 32 elementi contigui, quindi lungo C.

    raw_shape atteso da GGUFWriter: shape in NUMPY ORDER (il writer la inverte in GGUF):
      numpy [R, C]  →  raw_shape [R, C_bytes]  dove C_bytes = C//32×34
      writer inverte quando scrive → file GGUF contiene [C, R] ✓

    C deve essere divisibile per Q8_BLOCK=32.
    """
    R, C = numpy_shape
    assert C % Q8_BLOCK == 0, f"dim interna {C} non divisibile per {Q8_BLOCK}"
    bytes_inner = (C // Q8_BLOCK) * 34
    return [int(R), int(bytes_inner)]  # Python int, non numpy (struct.pack richiede int)


def should_quantize(name: str, shape: tuple) -> bool:
    """
    Quantizza solo tensori 2D dove la dimensione interna (C, colonne) è divisibile per 32.
    Esclude LayerNorm, bias, vettori 1D.
    """
    if len(shape) != 2:
        return False
    R, C = shape
    if R * C < 1024:
        return False
    return C % Q8_BLOCK == 0  # inner dim divisibile per 32


def main():
    parser = argparse.ArgumentParser(description="SLM GGUF fp32 → Q8_0")
    parser.add_argument("--checkpoint", default="checkpoints/run4_best.pt")
    parser.add_argument("--tokenizer",  default="checkpoints/tokenizer.model")
    parser.add_argument("--output",     default="slm-run4-q8_0.gguf")
    args = parser.parse_args()

    import gguf
    import sentencepiece as spm

    # ── 1. Carica checkpoint ──────────────────────────────────────────────────
    print(f"[1/4] Caricamento checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg: GPTConfig = ckpt["model_config"]
    sd: dict       = ckpt["model_state_dict"]
    d_k = cfg.d_model // cfg.num_heads
    print(f"      vocab={cfg.vocab_size}  d_model={cfg.d_model}  layers={cfg.num_layers}  "
          f"heads={cfg.num_heads}  d_ff={cfg.d_ff}  d_k={d_k}")

    # ── 2. Carica tokenizer ───────────────────────────────────────────────────
    print(f"[2/4] Caricamento tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor(); sp.load(args.tokenizer)
    vocab_size = sp.vocab_size()

    tokens, scores, token_types = [], [], []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i); score = sp.get_score(i)
        if sp.IsUnknown(i):    t = gguf.TokenType.UNKNOWN
        elif sp.IsControl(i):  t = gguf.TokenType.CONTROL
        elif sp.IsUnused(i):   t = gguf.TokenType.UNUSED
        elif sp.IsByte(i):     t = gguf.TokenType.BYTE
        else:                  t = gguf.TokenType.NORMAL
        tokens.append(piece); scores.append(score); token_types.append(t)
    tokens.append("\n"); scores.append(0.0); token_types.append(gguf.TokenType.NORMAL)

    # ── 3. Scrivi GGUF Q8_0 ──────────────────────────────────────────────────
    print(f"[3/4] Scrittura {args.output}")
    writer = gguf.GGUFWriter(args.output, "gptneox")

    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.d_model)
    writer.add_block_count(cfg.num_layers)
    writer.add_feed_forward_length(cfg.d_ff)
    writer.add_head_count(cfg.num_heads)
    writer.add_layer_norm_eps(1e-5)
    writer.add_parallel_residual(False)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_dimension_count(d_k)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)

    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("spm")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())

    def fp(key) -> np.ndarray:
        return sd[key].float().numpy()

    def add(name: str, arr: np.ndarray):
        if should_quantize(name, arr.shape):
            q = quantize_q8_0(arr)
            writer.add_tensor(name, q,
                              raw_shape=q8_byte_shape(arr.shape),
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, arr)

    emb = fp("token_embedding.weight")
    emb_ext = np.vstack([emb, np.zeros((1, cfg.d_model), np.float32)])
    add("token_embd.weight", emb_ext)

    q_count = fp32_count = 0
    for i in range(cfg.num_layers):
        p = f"blocks.{i}"
        qkv = np.concatenate([fp(f"{p}.attention.W_q.weight"),
                               fp(f"{p}.attention.W_k.weight"),
                               fp(f"{p}.attention.W_v.weight")], axis=0)
        qkv_bias = np.zeros(3 * cfg.d_model, np.float32)
        out_bias  = np.zeros(cfg.d_model, np.float32)
        add(f"blk.{i}.attn_qkv.weight",    qkv)
        writer.add_tensor(f"blk.{i}.attn_qkv.bias",      qkv_bias)
        add(f"blk.{i}.attn_output.weight", fp(f"{p}.attention.W_o.weight"))
        writer.add_tensor(f"blk.{i}.attn_output.bias",   out_bias)
        writer.add_tensor(f"blk.{i}.attn_norm.weight",   fp(f"{p}.norm1.weight"))
        writer.add_tensor(f"blk.{i}.attn_norm.bias",     fp(f"{p}.norm1.bias"))
        add(f"blk.{i}.ffn_up.weight",      fp(f"{p}.ff.net.0.weight"))
        writer.add_tensor(f"blk.{i}.ffn_up.bias",        fp(f"{p}.ff.net.0.bias"))
        add(f"blk.{i}.ffn_down.weight",    fp(f"{p}.ff.net.2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_down.bias",      fp(f"{p}.ff.net.2.bias"))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",    fp(f"{p}.norm2.weight"))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias",      fp(f"{p}.norm2.bias"))

    lm = fp("lm_head.weight")
    lm_ext = np.vstack([lm, np.zeros((1, cfg.d_model), np.float32)])
    writer.add_tensor("output_norm.weight", fp("norm_final.weight"))
    writer.add_tensor("output_norm.bias",   fp("norm_final.bias"))
    add("output.weight", lm_ext)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    in_size  = Path(args.checkpoint).stat().st_size
    out_size = Path(args.output).stat().st_size
    print(f"\n[4/4] Completato:")
    print(f"      Input (checkpoint): {in_size  / 1e6:.1f} MB")
    print(f"      Output (GGUF Q8_0): {out_size / 1e6:.1f} MB  "
          f"({out_size / in_size * 100:.0f}% del checkpoint)")
    print(f"\n      Testa:")
    print(f"      llama-simple --model {args.output} -p 'Il Parlamento europeo' -n 50")


if __name__ == "__main__":
    main()
