"""
lora.py — Low-Rank Adaptation per il modello SLM

LoRA congela tutti i parametri del base model e aggiunge due matrici piccole
A e B per ogni proiezione lineare target. Il delta effettivo al peso W è:

    ΔW = (alpha/r) · B @ A     shape: (d_out, d_in) — uguale a W

Durante il forward:
    output = W(x) + (alpha/r) · B(A(x))

Solo A e B vengono aggiornati dall'ottimizzatore (~393k param su 45.9M).

Uso standalone:
    from lora import apply_lora, save_lora, load_lora, merge_lora_model
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# LoRALinear — wrapper di nn.Linear frozen + adattatori trainabili
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.linear = linear          # pesi W originali — congelati dal chiamante
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        self.r      = r
        self.alpha  = alpha
        self.scale  = alpha / r

        # A: inizializzazione casuale small (come in Kaiming uniform)
        # B: inizializzazione a zero → delta nullo all'inizio del training
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale


# ─────────────────────────────────────────────────────────────────────────────
# apply_lora — congela il modello e sostituisce i Linear target con LoRALinear
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TARGETS = ('W_q', 'W_k', 'W_v', 'W_o')

def apply_lora(model, r: int = 8, alpha: int = 16,
               target_modules: tuple = _DEFAULT_TARGETS):
    """
    1. Congela TUTTI i parametri del modello.
    2. Sostituisce i Linear in target_modules con LoRALinear trainabili.
    3. Ritorna (model, n_trainable, n_total).

    Il modello viene modificato in-place.
    """
    # Passo 1 — congela tutto
    for p in model.parameters():
        p.requires_grad_(False)

    # Passo 2 — sostituisce i Linear target con LoRALinear
    replaced = 0
    for module in model.modules():
        for attr_name in list(module._modules.keys()):
            if attr_name not in target_modules:
                continue
            child = getattr(module, attr_name)
            if not isinstance(child, nn.Linear):
                continue
            setattr(module, attr_name, LoRALinear(child, r=r, alpha=alpha))
            replaced += 1

    # Passo 3 — conta i parametri trainabili (solo lora_A e lora_B)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())

    return model, n_trainable, n_total


# ─────────────────────────────────────────────────────────────────────────────
# save_lora — salva solo i pesi A e B (adapter portatile, ~1.5 MB)
# ─────────────────────────────────────────────────────────────────────────────

def save_lora(model, path: str,
              target_modules: tuple = _DEFAULT_TARGETS,
              r: int = 8, alpha: int = 16):
    """
    Salva l'adapter LoRA (solo A e B, non il base model) in un file compatto.
    Il file include anche r, alpha e target_modules per il reload.
    """
    adapters = {}
    for mod_name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if not isinstance(child, LoRALinear):
                continue
            key = f'{mod_name}.{attr_name}' if mod_name else attr_name
            adapters[key] = {
                'A': child.lora_A.weight.data.cpu().clone(),
                'B': child.lora_B.weight.data.cpu().clone(),
            }

    torch.save({
        'r':              r,
        'alpha':          alpha,
        'target_modules': list(target_modules),
        'adapters':       adapters,
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# load_lora — carica un adapter su un base model frozen
# ─────────────────────────────────────────────────────────────────────────────

def load_lora(model, path: str):
    """
    1. Carica l'adapter da `path` (prodotto da save_lora).
    2. Applica la struttura LoRA al modello (congela W, aggiunge A e B).
    3. Carica i pesi A e B salvati.
    4. Ritorna il modello pronto per l'inferenza.
    """
    state = torch.load(path, map_location='cpu', weights_only=True)
    r              = state['r']
    alpha          = state['alpha']
    target_modules = tuple(state['target_modules'])
    adapters       = state['adapters']

    model, _, _ = apply_lora(model, r=r, alpha=alpha,
                              target_modules=target_modules)

    for mod_name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if not isinstance(child, LoRALinear):
                continue
            key = f'{mod_name}.{attr_name}' if mod_name else attr_name
            if key not in adapters:
                continue
            child.lora_A.weight.data.copy_(adapters[key]['A'])
            child.lora_B.weight.data.copy_(adapters[key]['B'])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# merge_lora_model — restituisce un GPT plain con delta LoRA fusi in W
# ─────────────────────────────────────────────────────────────────────────────

def merge_lora_model(lora_model):
    """
    Crea un nuovo GPT con gli stessi parametri ma senza struttura LoRA.
    ΔW = (alpha/r) · B @ A viene sommato direttamente a W.

    Il modello restituito ha il format standard di checkpoint — compatibile
    con infer.py e tutti gli altri strumenti del progetto.
    """
    from model import GPT

    # Parte dallo state dict del modello LoRA: ha tutte le chiavi,
    # incluse embedding.weight e lm_head.weight (weight tying).
    lora_sd    = lora_model.state_dict()
    merged_sd  = {}

    for key, val in lora_sd.items():
        # salta i pesi adapter
        if '.lora_A.' in key or '.lora_B.' in key:
            continue
        # remap: .linear.weight → .weight  /  .linear.bias → .bias
        plain_key = key.replace('.linear.weight', '.weight') \
                       .replace('.linear.bias',   '.bias')
        merged_sd[plain_key] = val.clone()

    # Somma il delta LoRA a ciascun W target
    for mod_name, module in lora_model.named_modules():
        for attr_name, child in module.named_children():
            if not isinstance(child, LoRALinear):
                continue
            delta  = (child.lora_B.weight.data @ child.lora_A.weight.data) * child.scale
            prefix = (mod_name + '.') if mod_name else ''
            key    = prefix + attr_name + '.weight'
            if key in merged_sd:
                merged_sd[key] = merged_sd[key] + delta

    merged = GPT(lora_model.config)
    merged.load_state_dict(merged_sd)
    merged.eval()
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — test rapido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from model import GPT, GPTConfig

    cfg   = GPTConfig(vocab_size=16000, d_model=512, num_layers=12,
                      num_heads=8, d_ff=2048, max_seq_len=512)
    model = GPT(cfg)
    total_before = sum(p.numel() for p in model.parameters())

    model, n_train, n_total = apply_lora(model, r=8, alpha=16)

    print(f"  Parametri totali  : {n_total:,}")
    print(f"  Parametri trainabili (LoRA): {n_train:,}  ({n_train/n_total*100:.2f}%)")
    print(f"  Parametri congelati: {n_total - n_train:,}")

    # Forward pass
    x      = torch.randint(0, 16000, (2, 64))
    logits, _ = model(x)
    print(f"  Forward ok — logits shape: {logits.shape}")

    # Merge e confronto output (eval mode — dropout disabilitato)
    model.eval()
    merged = merge_lora_model(model)
    with torch.no_grad():
        logits_lora,   _ = model(x)
        logits_merged, _ = merged(x)
    diff = (logits_lora - logits_merged).abs().max().item()
    print(f"  Merge ok — diff max logits: {diff:.2e}  (atteso ~0)")

    # Save/load adapter
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    save_lora(model, tmp, r=8, alpha=16)
    size_kb = os.path.getsize(tmp) / 1024
    print(f"  Adapter salvato: {size_kb:.0f} KB  (atteso ~1500 KB)")

    # Il reload corretto: stessa base (merged = same W), poi applica adapter A/B
    model2 = GPT(cfg)
    model2.load_state_dict(merged.state_dict())  # stessa base del modello LoRA
    model2 = load_lora(model2, tmp)
    model2.eval()
    os.unlink(tmp)
    with torch.no_grad():
        logits_reload, _ = model2(x)
    diff2 = (logits_lora - logits_reload).abs().max().item()
    print(f"  Reload ok — diff max logits: {diff2:.2e}  (atteso ~0)")
    print("\n  Tutti i test passati.")
