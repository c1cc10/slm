#!/usr/bin/env python3
"""
Inferenza SFT/LoRA — estrae JSON dal checkpoint merged.

Il modello è stato addestrato con separatore '\n\n' tra prompt e completion.
Questo script aggiunge il separatore, genera fino a max_new_tokens, poi
estrae il primo JSON completo dall'output (stop al bilanciamento delle {}).

Uso:
    python3 tools/infer_lora.py --checkpoint checkpoints/lora_calendar_merged.pt
    python3 tools/infer_lora.py --checkpoint checkpoints/lora_calendar_merged.pt \
        --prompt "Crea un appuntamento con Luca domenica alle 10."
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import train  # noqa: F401  — registra TrainConfig per il pickle del checkpoint


def load_model(ckpt_path: str, device: str = "cpu"):
    from model import GPT
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = GPT(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(bytes(ckpt["tokenizer_state"]["model_bytes"]))

    return model, sp, device


def extract_first_json(text: str):
    """Ritorna il primo oggetto JSON ben formato trovato in text, o None."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    start = None
    return None


def infer(model, sp, prompt: str, device: str,
          sep: str = "\n\n", max_new_tokens: int = 120,
          temperature: float = 0.2, top_k: int = 10):

    full_prompt = prompt + sep
    ids = sp.encode(full_prompt, out_type=int)
    x = torch.tensor([ids], device=device)

    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)

    gen_ids = out[0][len(ids):].tolist()
    raw = sp.decode(gen_ids)
    extracted = extract_first_json(raw)
    return raw, extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/lora_calendar_merged.pt")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print(f"Caricamento {args.checkpoint}...", end=" ", flush=True)
    model, sp, device = load_model(args.checkpoint, args.device)
    print("pronto.\n")

    default_prompts = [
        "Crea un appuntamento con Marco venerdì alle 15.",
        "Sposta la riunione di domani a giovedì pomeriggio.",
        "Cosa ho in agenda lunedì prossimo?",
        "Ricordami di chiamare il medico domani mattina.",
        "Cancella tutti gli impegni di mercoledì.",
        "Aggiungi Giulia Bianchi, cell 347-1234567.",
        "Elimina la riunione del martedì.",
    ]

    prompts = [args.prompt] if args.prompt else default_prompts

    for p in prompts:
        _, extracted = infer(model, sp, p, device, temperature=args.temp, top_k=args.top_k)
        if extracted:
            try:
                pretty = json.dumps(json.loads(extracted), ensure_ascii=False)
            except Exception:
                pretty = extracted
        else:
            pretty = "(nessun JSON valido estratto)"
        print(f"IN : {p}")
        print(f"OUT: {pretty}")
        print()


if __name__ == "__main__":
    main()
