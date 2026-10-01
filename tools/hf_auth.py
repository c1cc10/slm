#!/usr/bin/env python3
"""
tools/hf_auth.py — Gestione autenticazione HuggingFace

Comandi:
    python3 tools/hf_auth.py login     # login interattivo (richiede token HF)
    python3 tools/hf_auth.py whoami    # mostra utente autenticato
    python3 tools/hf_auth.py logout    # rimuove credenziali
    python3 tools/hf_auth.py check     # verifica se autenticato

Token HF: https://huggingface.co/settings/tokens
    → "New token" → tipo "Read" è sufficiente per scaricare dataset gated
"""

import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'

    try:
        from huggingface_hub import HfApi, login, logout, whoami
    except ImportError:
        print("ERRORE: huggingface_hub non installato.  pip install huggingface_hub")
        sys.exit(1)

    if cmd == 'login':
        token = sys.argv[2] if len(sys.argv) > 2 else None
        if not token:
            print("Uso: python3 tools/hf_auth.py login <token>")
            print("Token: https://huggingface.co/settings/tokens  (tipo Read)")
            sys.exit(1)
        login(token=token)
        try:
            info = whoami()
            print(f"Autenticato come: {info['name']} ({info.get('email', 'email non disponibile')})")
        except Exception:
            pass

    elif cmd == 'whoami':
        try:
            info = whoami()
            print(f"Utente: {info['name']}")
            print(f"Email : {info.get('email', 'non disponibile')}")
            print(f"Tipo  : {info.get('type', 'user')}")
        except Exception:
            print("Non autenticato. Esegui: python3 tools/hf_auth.py login")
            sys.exit(1)

    elif cmd == 'logout':
        logout()
        print("Logout completato.")

    elif cmd == 'check':
        try:
            info = whoami()
            print(f"Autenticato come: {info['name']}")
        except Exception:
            print("Non autenticato.")
            print("Esegui: python3 tools/hf_auth.py login")
            print("Token: https://huggingface.co/settings/tokens")
            sys.exit(1)

    elif cmd == 'test-culturax':
        print("Test accesso CulturaX (gated dataset)...")
        try:
            from datasets import load_dataset
            ds = load_dataset('uonlp/CulturaX', 'it', split='train', streaming=True)
            row = next(iter(ds))
            print(f"Accesso OK. Campi: {list(row.keys())}")
            print(f"Testo: {str(row.get('text', ''))[:200]}")
        except Exception as e:
            print(f"Accesso negato: {e}")
            sys.exit(1)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
