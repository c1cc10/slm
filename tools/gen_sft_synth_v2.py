"""
Dataset SFT v2 — 21 intent + unknown, 200 esempi/intent = 4400 totali.

Rispetto alla v1:
- EVENTI_GEN: tuple (parola, genere m/f, iniziale v/c) → articoli sempre corretti
- 21 intent: CRUD eventi, promemoria, contatti, compleanni, temporali, festività, stato
- unknown: ~200 domande fuori dominio

Output: data/sft_v2_raw.jsonl
"""
import json
import random

random.seed(42)

# ── Vocabolario ────────────────────────────────────────────────────────────────

NOMI_M = ["Marco", "Luca", "Alessandro", "Mattia", "Kevin", "Andrea", "Stefano",
          "Davide", "Riccardo", "Filippo", "Roberto", "Nicola", "Pietro",
          "Fabio", "Giorgio", "Simone", "Emanuele", "Tommaso", "Enrico",
          "Alberto", "Lorenzo", "Francesco", "Giovanni", "Antonio", "Massimo"]

NOMI_F = ["Giulia", "Sara", "Laura", "Maria", "Chiara", "Francesca", "Elena",
          "Monica", "Valentina", "Alessia", "Irene", "Claudia", "Sofia",
          "Anna", "Marta", "Roberta", "Paola", "Barbara", "Lorena", "Silvia",
          "Cristina", "Raffaella", "Beatrice", "Camilla", "Elisa"]

NOMI = NOMI_M + NOMI_F

COGNOMI = ["Rossi", "Bianchi", "Ferrari", "Esposito", "Ricci", "Conti", "Moro",
           "Neri", "Russo", "Verdi", "Greco", "Mancini", "Romano", "Colombo",
           "Gallo", "Barbieri", "Martini", "Marini", "Costa", "Giordano"]

RUOLI = ["cliente", "fornitore", "collega", "responsabile", "manager",
         "direttore", "avvocato", "commercialista", "medico", "notaio",
         "tecnico", "assistente", "coordinatore", "project manager", "consulente",
         "ingegnere", "responsabile HR", "responsabile vendite", "referente"]

# (parola, genere m/f, iniziale v/c)
EVENTI_GEN = [
    ("riunione", "f", "c"),
    ("meeting", "m", "c"),
    ("call", "f", "c"),
    ("videochiamata", "f", "c"),
    ("appuntamento", "m", "v"),
    ("colloquio", "m", "c"),
    ("pranzo", "m", "c"),
    ("cena", "f", "c"),
    ("presentazione", "f", "c"),
    ("revisione", "f", "c"),
    ("briefing", "m", "c"),
    ("workshop", "m", "c"),
    ("sessione", "f", "c"),
    ("incontro", "m", "v"),
    ("consulenza", "f", "c"),
    ("aggiornamento", "m", "v"),
    ("demo", "f", "c"),
    ("formazione", "f", "c"),
    ("intervista", "f", "v"),
    ("sopralluogo", "m", "c"),
    ("stand-up", "m", "c"),
    ("check-in", "m", "c"),
]

GIORNI_REL = ["domani", "dopodomani", "oggi pomeriggio", "lunedì", "martedì",
              "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]

GIORNI_PROSS = ["lunedì prossimo", "martedì prossimo", "mercoledì prossimo",
                "giovedì prossimo", "venerdì prossimo", "la settimana prossima",
                "il mese prossimo", "dopodomani", "tra due settimane"]

ORE = ["08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
       "12:00", "13:00", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30",
       "17:00", "17:30", "18:00"]

PERIODI = ["mattina", "pomeriggio", "sera", "tarda mattinata", "primo pomeriggio",
           "tardo pomeriggio"]

DATE_ABS = ["3 novembre", "5 novembre", "10 novembre", "15 novembre", "20 novembre",
            "25 novembre", "1 dicembre", "8 dicembre", "20 ottobre", "25 ottobre",
            "28 ottobre", "2 dicembre", "12 novembre", "7 novembre", "18 novembre",
            "22 ottobre", "4 dicembre", "14 febbraio", "8 marzo", "1 maggio"]

DURATE = ["30 minuti", "1 ora", "2 ore", "3 ore", "mezz'ora", "un'ora e mezza",
          "45 minuti", "20 minuti"]

MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]

RICORRENZE_BREVI = ["ogni lunedì", "ogni martedì", "ogni mercoledì", "ogni giovedì",
                    "ogni venerdì", "ogni settimana", "ogni mese"]

FESTIVITÀ = [
    "Capodanno", "Epifania", "Pasqua", "Pasquetta", "Festa della Liberazione",
    "Festa dei Lavoratori", "Festa della Repubblica", "Ferragosto",
    "Ognissanti", "Immacolata Concezione", "Natale", "Santo Stefano",
    "Carnevale", "San Valentino", "Halloween",
]

CAMPI_CONTATTO = ["numero", "telefono", "email", "ruolo", "indirizzo", "recapito"]

STATI_STATUS = [
    "fuori ufficio", "assente", "non disponibile", "in ferie",
    "in malattia", "in trasferta", "non raggiungibile",
]

TARGET_DURATA = [
    "Natale", "Capodanno", "la fine del mese", "la scadenza del contratto",
    "la riunione di domani", "il weekend", "le vacanze", "la prossima settimana",
    "il mio compleanno", "la fine dell'anno", "la presentazione di venerdì",
    "il kick-off del progetto", "la deadline",
]

NOTE_POOL = [
    "chiamare {nome}", "rispondere alla mail di {nome}", "mandare il report",
    "pagare la bolletta", "rinnovare l'abbonamento", "prenotare il volo",
    "comprare il regalo per {nome}", "fare il backup", "mandare la fattura",
    "preparare la presentazione", "firmare il contratto", "leggere il verbale",
    "confermare la partecipazione", "aggiornare la documentazione",
    "verificare lo stato del progetto", "chiamare il {ruolo}", "rispondere a {nome}",
    "mandare i documenti", "controllare le email", "ricontattare {nome}",
    "portare i materiali", "revisionare il report", "contattare il {ruolo}",
    "inviare la proposta", "preparare i dati", "controllare i numeri",
    "scrivere le note", "organizzare i file", "verificare la scadenza",
    "richiamare {nome}", "aggiornare il team", "preparare l'ordine del giorno",
    "inviare il resoconto", "prenotare la sala", "confermare l'ordine",
    "leggere il contratto", "approvare le spese", "aggiornare il CRM",
    "inviare le credenziali a {nome}", "pagare l'abbonamento",
]

TITOLI_EVENTO = ["revisione budget", "review codice", "allineamento settimanale",
                 "presentazione progetto", "brainstorming", "onboarding",
                 "check-in mensile", "retrospettiva", "planning sprint",
                 "riunione di team", "aggiornamento stato", "demo prodotto",
                 "colloquio candidato", "sessione formativa", "call cliente",
                 "kick-off progetto", "stand-up mattutino", "analisi risultati",
                 "verifica avanzamento", "meeting commerciale"]

DOMINI = ["gmail.com", "outlook.com", "azienda.it", "lavoro.com", "studio.it",
          "email.it", "impresa.it", "consulting.it", "company.com"]

PREFISSI_CELL = ["333", "347", "328", "339", "340", "346", "380", "393", "348", "366"]
PREFISSI_FISSO = ["02", "06", "011", "081", "055", "051", "049", "045"]

# ── Articoli italiani ──────────────────────────────────────────────────────────
# Regole:
#  m+v → "un X"   (mai "un'X"), "l'X", "dell'X", "all'X"
#  m+c → "un X",  "il X",  "del X",  "al X"
#  f+v → "un'X",  "l'X",  "dell'X", "all'X"
#  f+c → "una X", "la X", "della X", "alla X"

def art_indef(eg):
    word, g, i = eg
    if g == "f" and i == "v":
        return "un'" + word
    elif g == "f":
        return "una " + word
    else:
        return "un " + word   # maschile: mai elision anche prima di vocale

def art_def(eg):
    word, g, i = eg
    if i == "v":
        return "l'" + word
    return ("la " if g == "f" else "il ") + word

def art_prep_del(eg):
    word, g, i = eg
    if i == "v":
        return "dell'" + word
    return ("della " if g == "f" else "del ") + word

def art_prep_al(eg):
    word, g, i = eg
    if i == "v":
        return "all'" + word
    return ("alla " if g == "f" else "al ") + word

# ── Helper casuali ─────────────────────────────────────────────────────────────

def rn():     return random.choice(NOMI)
def rn_m():   return random.choice(NOMI_M)
def rn_f():   return random.choice(NOMI_F)
def rcog():   return random.choice(COGNOMI)
def rd():     return random.choice(GIORNI_REL)
def rdp():    return random.choice(GIORNI_PROSS)
def rda():    return random.choice(DATE_ABS)
def ro():     return random.choice(ORE)
def rp():     return random.choice(PERIODI)
def rev():    return random.choice(EVENTI_GEN)
def rte():    return random.choice(TITOLI_EVENTO)
def rdu():    return random.choice(DURATE)
def rru():    return random.choice(RUOLI)
def rric():   return random.choice(RICORRENZE_BREVI)
def rmese():  return random.choice(MESI)
def rfest():  return random.choice(FESTIVITÀ)

def rphone(fisso=False):
    pre = random.choice(PREFISSI_FISSO if fisso else PREFISSI_CELL)
    return f"{pre}-{random.randint(1000000, 9999999)}"

def remail(nome):
    suf = random.choice(["", "m.", "r.", "a."])
    return f"{suf}{nome.lower()}@{random.choice(DOMINI)}"

def rnota(nome=None, ruolo=None):
    t = random.choice(NOTE_POOL)
    return t.format(nome=nome or rn(), ruolo=ruolo or rru())

# ── Output ─────────────────────────────────────────────────────────────────────

out = []

def add(prompt, completion):
    c = {k: v for k, v in completion.items() if v is not None}
    out.append({"prompt": prompt.strip(), "completion": json.dumps(c, ensure_ascii=False)})

def count_intent(intent):
    return sum(1 for e in out if json.loads(e["completion"]).get("intent") == intent)

def fill_intent(gen_fn, intent, n=200):
    used = set()
    attempts = 0
    while count_intent(intent) < n and attempts < n * 60:
        attempts += 1
        p, c = gen_fn()
        k = p[:60]
        if k not in used:
            used.add(k)
            add(p, c)
    got = count_intent(intent)
    if got < n:
        print(f"  WARN {intent}: {got}/{n}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. CREATE_EVENT
# ═════════════════════════════════════════════════════════════════════════════

def gen_create_event():
    eg = rev()
    ev = eg[0]; ei = art_indef(eg); ed = art_def(eg)
    s = random.randint(0, 19)
    if s == 0:
        n, d, o = rn(), rd(), ro()
        return (f"Fissa {ei} con {n} {d} alle {o}.",
                {"intent": "create_event", "title": ev, "contact": n, "date": d, "time": o})
    elif s == 1:
        d, p = rd(), rp()
        return (f"Prenota {ei} per {d} {p}.",
                {"intent": "create_event", "title": ev, "date": d, "time": p})
    elif s == 2:
        te, da, o = rte(), rda(), ro()
        return (f"Crea un evento: {te} il {da} alle {o}.",
                {"intent": "create_event", "title": te, "date": da, "time": o})
    elif s == 3:
        ru, dp = rru(), rdp()
        return (f"Metti in agenda {ei} con il {ru} per {dp}.",
                {"intent": "create_event", "title": ev, "contact": ru, "date": dp})
    elif s == 4:
        n, d, o = rn(), rd(), ro()
        return (f"Ho bisogno di {ei} con {n} {d} alle {o}.",
                {"intent": "create_event", "title": ev, "contact": n, "date": d, "time": o})
    elif s == 5:
        te, da, o = rte(), rda(), ro()
        return (f"Aggiungi al calendario: {te}, {da}, ore {o}.",
                {"intent": "create_event", "title": te, "date": da, "time": o})
    elif s == 6:
        n, d, du = rn(), rd(), rdu()
        return (f"Organizza {ei} con {n} {d} per {du}.",
                {"intent": "create_event", "title": ev, "contact": n, "date": d, "duration": du})
    elif s == 7:
        n, dp = rn(), rdp()
        return (f"Devo incontrare {n} {dp}. Mettilo in agenda.",
                {"intent": "create_event", "contact": n, "date": dp})
    elif s == 8:
        n, da, o = rn(), rda(), ro()
        return (f"Crea appuntamento con {n}: {da} ore {o}.",
                {"intent": "create_event", "contact": n, "date": da, "time": o})
    elif s == 9:
        n1, n2, d, o = rn(), rn(), rd(), ro()
        while n2 == n1: n2 = rn()
        return (f"Aggiungi: {ei} con {n1} e {n2} {d} alle {o}.",
                {"intent": "create_event", "title": ev, "contact": [n1, n2], "date": d, "time": o})
    elif s == 10:
        te, n, d, o, du = rte(), rn(), rd(), ro(), rdu()
        return (f"Metti in calendario: {te} con {n}, {d}, dalle {o} per {du}.",
                {"intent": "create_event", "title": te, "contact": n, "date": d, "time": o, "duration": du})
    elif s == 11:
        n, da, p = rn(), rda(), rp()
        return (f"Pianifica {ei} con {n} il {da} di {p}.",
                {"intent": "create_event", "title": ev, "contact": n, "date": da, "time": p})
    elif s == 12:
        ru, dp, o = rru(), rdp(), ro()
        return (f"Fissa un appuntamento con il {ru} {dp} alle {o}.",
                {"intent": "create_event", "title": "appuntamento", "contact": ru, "date": dp, "time": o})
    elif s == 13:
        n, d = rn(), rd()
        return (f"C'è {ei} con {n} {d} mattina. Aggiungilo al calendario.",
                {"intent": "create_event", "title": ev, "contact": n, "date": d, "time": "mattina"})
    elif s == 14:
        da, o, te = rda(), ro(), rte()
        return (f"Crea evento il {da} alle {o}: {te}.",
                {"intent": "create_event", "title": te, "date": da, "time": o})
    elif s == 15:
        n, d, p = rn(), rd(), rp()
        return (f"Segna {ei} con {n} per {d} {p}.",
                {"intent": "create_event", "title": ev, "contact": n, "date": d, "time": p})
    elif s == 16:
        n1, n2, n3, dp = rn(), rn(), rn(), rdp()
        while n2 == n1: n2 = rn()
        while n3 in (n1, n2): n3 = rn()
        return (f"Aggiungi: {ei} con {n1}, {n2} e {n3} per {dp}.",
                {"intent": "create_event", "title": ev, "contact": [n1, n2, n3], "date": dp})
    elif s == 17:
        te, d, o = rte(), rd(), ro()
        return (f"Inserisci in agenda il {te} per {d} alle {o}.",
                {"intent": "create_event", "title": te, "date": d, "time": o})
    elif s == 18:
        n, d, o = rn(), rd(), ro()
        return (f"Ho {ei} con {n} {d} alle {o}. Salvala in agenda.",
                {"intent": "create_event", "title": ev, "contact": n, "date": d, "time": o})
    else:
        ru, d, o, du = rru(), rd(), ro(), rdu()
        return (f"Metti {ei} con il {ru} {d} alle {o}. Durata {du}.",
                {"intent": "create_event", "title": ev, "contact": ru, "date": d, "time": o, "duration": du})

fill_intent(gen_create_event, "create_event")

# ═════════════════════════════════════════════════════════════════════════════
# 2. UPDATE_EVENT
# ═════════════════════════════════════════════════════════════════════════════

def gen_update_event():
    eg = rev()
    ev = eg[0]; ed = art_def(eg); edel = art_prep_del(eg)
    s = random.randint(0, 17)
    if s == 0:
        n, d, dp = rn(), rd(), rdp()
        return (f"Sposta {ed} con {n} da {d} a {dp}.",
                {"intent": "update_event", "event_ref": f"{ev} {d}", "contact": n, "new_date": dp})
    elif s == 1:
        ev2 = rev(); ev2w = ev2[0]; ev2i = art_indef(ev2)
        d, dp, o = rd(), rdp(), ro()
        return (f"Devo posticipare {art_def(ev2)} di {d} a {dp} alle {o}.",
                {"intent": "update_event", "event_ref": f"{ev2w} {d}", "new_date": dp, "new_time": o})
    elif s == 2:
        n, dp, o = rn(), rdp(), ro()
        return (f"La riunione con {n} è stata anticipata a {dp} alle {o}.",
                {"intent": "update_event", "event_ref": f"riunione", "contact": n, "new_date": dp, "new_time": o})
    elif s == 3:
        d, dp = rd(), rdp()
        return (f"{ed.capitalize()} di {d} va spostato a {dp}.",
                {"intent": "update_event", "event_ref": f"{ev} {d}", "new_date": dp})
    elif s == 4:
        d, dp, o = rd(), rdp(), ro()
        return (f"Posticipa {ed} di {d} a {dp} alle {o}.",
                {"intent": "update_event", "event_ref": f"{ev} {d}", "new_date": dp, "new_time": o})
    elif s == 5:
        te, da, dp = rte(), rda(), rda()
        while dp == da: dp = rda()
        return (f"La {te} è slittata dal {da} al {dp}.",
                {"intent": "update_event", "event_ref": te, "new_date": dp})
    elif s == 6:
        n, dp, o = rn(), rdp(), ro()
        return (f"Aggiorna il meeting con {n}: non più domani ma {dp} alle {o}.",
                {"intent": "update_event", "contact": n, "event_ref": "meeting", "new_date": dp, "new_time": o})
    elif s == 7:
        o1, o2, d = ro(), ro(), rd()
        while o2 == o1: o2 = ro()
        return (f"Sposta {ed} delle {o1} di {d} alle {o2}.",
                {"intent": "update_event", "event_ref": f"{ev} {o1} {d}", "new_date": d, "new_time": o2})
    elif s == 8:
        n, d, dp = rn(), rd(), rdp()
        return (f"Devo rimandare {ed} con {n} di {d}. Può fare {dp}?",
                {"intent": "update_event", "contact": n, "event_ref": f"{ev} {d}", "new_date": dp})
    elif s == 9:
        te, n, dp = rte(), rn(), rdp()
        return (f"Rinomina la riunione con {n} in '{te}'.",
                {"intent": "update_event", "contact": n, "event_ref": "riunione", "new_title": te})
    elif s == 10:
        n1, n2, d, dp = rn(), rn(), rd(), rdp()
        while n2 == n1: n2 = rn()
        return (f"Sposta il meeting con {n1} e {n2} da {d} a {dp}.",
                {"intent": "update_event", "contact": [n1, n2], "event_ref": f"meeting {d}", "new_date": dp})
    elif s == 11:
        d, dp, o = rd(), rdp(), ro()
        return (f"Aggiorna: {ev} di {d} spostato a {dp} alle {o}.",
                {"intent": "update_event", "event_ref": f"{ev} {d}", "new_date": dp, "new_time": o})
    elif s == 12:
        n, d, dp = rn(), rd(), rdp()
        return (f"Ho dovuto spostare il pranzo con {n} da {d} a {dp}.",
                {"intent": "update_event", "contact": n, "event_ref": f"pranzo {d}", "new_date": dp})
    elif s == 13:
        n1, n2, d = rn(), rn(), rd()
        while n2 == n1: n2 = rn()
        return (f"Aggiungi {n2} alla riunione con {n1} di {d}.",
                {"intent": "update_event", "event_ref": f"riunione {d}", "contact": n1, "new_contact": n2})
    elif s == 14:
        te, da = rte(), rda()
        return (f"Cambia il titolo {art_prep_del(eg)} del {da} in '{te}'.",
                {"intent": "update_event", "event_ref": f"{ev} {da}", "new_title": te})
    elif s == 15:
        da, dp = rda(), rda()
        while dp == da: dp = rda()
        return (f"Aggiorna il calendario: {ed} non è più il {da} ma il {dp}.",
                {"intent": "update_event", "event_ref": ev, "new_date": dp})
    elif s == 16:
        n, d, o = rn(), rd(), ro()
        return (f"Sposta il check-in con {n} di {d} alle {o}.",
                {"intent": "update_event", "event_ref": f"check-in {d}", "contact": n, "new_time": o})
    else:
        d, o = rd(), ro()
        return (f"Anticipa {ed} di {d} di un'ora: ora alle {o}.",
                {"intent": "update_event", "event_ref": f"{ev} {d}", "new_time": o})

fill_intent(gen_update_event, "update_event")

# ═════════════════════════════════════════════════════════════════════════════
# 3. DELETE_EVENT
# ═════════════════════════════════════════════════════════════════════════════

DEL_VERBS = ["Cancella", "Elimina", "Rimuovi", "Togli", "Disdici", "Annulla"]

def gen_delete_event():
    eg = rev()
    ev = eg[0]; ed = art_def(eg); edel = art_prep_del(eg)
    v = random.choice(DEL_VERBS)
    s = random.randint(0, 17)
    if s == 0:
        n, d = rn(), rd()
        return (f"{v} {ed} con {n} di {d}.",
                {"intent": "delete_event", "contact": n, "event_ref": f"{ev} {d}", "date": d})
    elif s == 1:
        da = rda()
        return (f"{v} {ed} del {da}.",
                {"intent": "delete_event", "event_ref": ev, "date": da})
    elif s == 2:
        d, p = rd(), rp()
        return (f"{v} tutto quello che ho {d} {p}.",
                {"intent": "delete_event", "date": d, "time": p, "scope": "all"})
    elif s == 3:
        d, o = rd(), ro()
        return (f"{v} {ed} delle {o} di {d}.",
                {"intent": "delete_event", "event_ref": ev, "date": d, "time": o})
    elif s == 4:
        n = rn()
        return (f"Non ho più bisogno {edel} con {n}. {v}lo dal calendario.",
                {"intent": "delete_event", "contact": n, "event_ref": ev})
    elif s == 5:
        n = rn()
        return (f"{v} tutti gli appuntamenti con {n} questa settimana.",
                {"intent": "delete_event", "contact": n, "date": "questa settimana", "scope": "all"})
    elif s == 6:
        ru = rru()
        return (f"{ed.capitalize()} con il {ru} è annullata. Toglila dal calendario.",
                {"intent": "delete_event", "contact": ru, "event_ref": ev})
    elif s == 7:
        te, da = rte(), rda()
        return (f"{v} il {te} del {da}.",
                {"intent": "delete_event", "event_ref": te, "date": da})
    elif s == 8:
        n, d = rn(), rd()
        return (f"Non viene più {n}, {v.lower()} {ed} di {d}.",
                {"intent": "delete_event", "contact": n, "event_ref": ev, "date": d})
    elif s == 9:
        dp = rdp()
        return (f"{v} tutti gli impegni di {dp}.",
                {"intent": "delete_event", "date": dp, "scope": "all"})
    elif s == 10:
        d, o = rd(), ro()
        return (f"{v} {ed} alle {o} di {d}.",
                {"intent": "delete_event", "event_ref": ev, "date": d, "time": o})
    elif s == 11:
        da = rda()
        return (f"{ed.capitalize()} del {da} è saltato. Toglilo.",
                {"intent": "delete_event", "event_ref": ev, "date": da})
    elif s == 12:
        d = rd()
        return (f"È saltato tutto per {d}. Pulisci il calendario.",
                {"intent": "delete_event", "date": d, "scope": "all"})
    elif s == 13:
        n1, n2, dp = rn(), rn(), rdp()
        while n2 == n1: n2 = rn()
        return (f"{v} {ed} con {n1} e {n2} di {dp}.",
                {"intent": "delete_event", "contact": [n1, n2], "event_ref": ev, "date": dp})
    elif s == 14:
        te = rte()
        return (f"Cancella il {te} dalla prossima settimana.",
                {"intent": "delete_event", "event_ref": te, "date": "la prossima settimana"})
    elif s == 15:
        ru, d = rru(), rd()
        return (f"Il {ru} ha disdetto. Togli {ed} di {d}.",
                {"intent": "delete_event", "contact": ru, "event_ref": ev, "date": d})
    elif s == 16:
        d, p = rd(), rp()
        return (f"{v} il blocco del {p} di {d}.",
                {"intent": "delete_event", "date": d, "time": p})
    else:
        n, da = rn(), rda()
        return (f"Non c'è più bisogno del colloquio con {n} del {da}. Cancellalo.",
                {"intent": "delete_event", "contact": n, "event_ref": "colloquio", "date": da})

fill_intent(gen_delete_event, "delete_event")

# ═════════════════════════════════════════════════════════════════════════════
# 4. CREATE_RECURRING
# ═════════════════════════════════════════════════════════════════════════════

def gen_create_recurring():
    eg = rev()
    ev = eg[0]; ei = art_indef(eg)
    s = random.randint(0, 14)
    ric = rric(); o = ro()
    if s == 0:
        return (f"Crea {ei} ricorrente {ric} alle {o}.",
                {"intent": "create_recurring", "title": ev, "recurrence": ric, "time": o})
    elif s == 1:
        return (f"Aggiungi {ei} fisso {ric} alle {o}.",
                {"intent": "create_recurring", "title": ev, "recurrence": ric, "time": o})
    elif s == 2:
        te = rte()
        return (f"Imposta {ric}: {te} alle {o}.",
                {"intent": "create_recurring", "title": te, "recurrence": ric, "time": o})
    elif s == 3:
        n = rn()
        return (f"Ho bisogno di {ei} con {n} {ric} alle {o}.",
                {"intent": "create_recurring", "title": ev, "recurrence": ric, "time": o, "contact": n})
    elif s == 4:
        da = rda()
        return (f"A partire dal {da}, {ei} ricorrente {ric}.",
                {"intent": "create_recurring", "title": ev, "recurrence": ric, "date_start": da})
    elif s == 5:
        return (f"Imposta {ei} settimanale ogni lunedì alle {o}.",
                {"intent": "create_recurring", "title": ev, "recurrence": "ogni lunedì", "time": o})
    elif s == 6:
        te = rte()
        return (f"Crea evento mensile: {te} il primo del mese alle {o}.",
                {"intent": "create_recurring", "title": te, "recurrence": "mensile", "time": o})
    elif s == 7:
        n = rn()
        return (f"Con {n} facciamo {ei} fisso {ric}. Aggiungilo al calendario.",
                {"intent": "create_recurring", "title": ev, "recurrence": ric, "contact": n})
    elif s == 8:
        te = rte()
        return (f"Pianifica il {te} come appuntamento ricorrente {ric} alle {o}.",
                {"intent": "create_recurring", "title": te, "recurrence": ric, "time": o})
    elif s == 9:
        da = rda()
        return (f"Dal {da} ogni venerdì alle {o}: {ei}.",
                {"intent": "create_recurring", "title": ev, "recurrence": "ogni venerdì", "time": o, "date_start": da})
    elif s == 10:
        return (f"Aggiungi lo stand-up giornaliero alle {o}.",
                {"intent": "create_recurring", "title": "stand-up", "recurrence": "giornaliero", "time": o})
    elif s == 11:
        te, n = rte(), rn()
        return (f"Metti in agenda {ei} bisettimanale con {n}: {te}.",
                {"intent": "create_recurring", "title": te, "recurrence": "bisettimanale", "contact": n})
    elif s == 12:
        return (f"Ogni settimana alle {o}: {ei} con il team.",
                {"intent": "create_recurring", "title": ev, "recurrence": "ogni settimana", "time": o, "contact": "team"})
    elif s == 13:
        te = rte()
        return (f"Voglio {art_indef(random.choice(EVENTI_GEN))} mensile per il {te}.",
                {"intent": "create_recurring", "title": te, "recurrence": "mensile"})
    else:
        n = rn(); da = rda()
        return (f"Crea {ei} fisso con {n}: {ric} alle {o} a partire dal {da}.",
                {"intent": "create_recurring", "title": ev, "recurrence": ric, "time": o,
                 "contact": n, "date_start": da})

fill_intent(gen_create_recurring, "create_recurring")

# ═════════════════════════════════════════════════════════════════════════════
# 5. DELETE_RECURRING
# ═════════════════════════════════════════════════════════════════════════════

def gen_delete_recurring():
    eg = rev()
    ev = eg[0]; ed = art_def(eg); ei = art_indef(eg)
    v = random.choice(DEL_VERBS)
    s = random.randint(0, 13)
    ric = rric()
    if s == 0:
        return (f"{v} {ed} ricorrente {ric}.",
                {"intent": "delete_recurring", "event_ref": ev, "recurrence": ric})
    elif s == 1:
        return (f"Togli la serie ricorrente {art_prep_del(eg)}.",
                {"intent": "delete_recurring", "event_ref": ev})
    elif s == 2:
        te = rte()
        return (f"Annulla il {te} ricorrente {ric}.",
                {"intent": "delete_recurring", "event_ref": te, "recurrence": ric})
    elif s == 3:
        return (f"Non voglio più {ei} fisso {ric}. Cancellalo.",
                {"intent": "delete_recurring", "event_ref": ev, "recurrence": ric})
    elif s == 4:
        n = rn()
        return (f"{v} tutti gli appuntamenti ricorrenti con {n}.",
                {"intent": "delete_recurring", "event_ref": "appuntamenti", "contact": n})
    elif s == 5:
        return (f"Rimuovi lo stand-up giornaliero.",
                {"intent": "delete_recurring", "event_ref": "stand-up", "recurrence": "giornaliero"})
    elif s == 6:
        te = rte()
        return (f"Cancella la serie di {te} {ric}.",
                {"intent": "delete_recurring", "event_ref": te, "recurrence": ric})
    elif s == 7:
        return (f"Non faremo più {ed} mensile. Toglilo.",
                {"intent": "delete_recurring", "event_ref": ev, "recurrence": "mensile"})
    elif s == 8:
        n = rn()
        return (f"Ho smesso di lavorare con {n}. Cancella tutti gli appuntamenti fissi con loro.",
                {"intent": "delete_recurring", "contact": n})
    elif s == 9:
        return (f"Sospendi tutti gli eventi ricorrenti {ric}.",
                {"intent": "delete_recurring", "recurrence": ric})
    elif s == 10:
        te = rte()
        return (f"Il {te} settimanale non serve più. Eliminalo.",
                {"intent": "delete_recurring", "event_ref": te, "recurrence": "settimanale"})
    elif s == 11:
        return (f"Togli la {ev} del lunedì.",
                {"intent": "delete_recurring", "event_ref": ev, "recurrence": "ogni lunedì"})
    elif s == 12:
        n = rn()
        return (f"{v} il check-in ricorrente con {n}.",
                {"intent": "delete_recurring", "event_ref": "check-in", "contact": n})
    else:
        return (f"Non faremo più la riunione settimanale. Rimuovila.",
                {"intent": "delete_recurring", "event_ref": "riunione", "recurrence": "settimanale"})

fill_intent(gen_delete_recurring, "delete_recurring")

# ═════════════════════════════════════════════════════════════════════════════
# 6. QUERY_SCHEDULE
# ═════════════════════════════════════════════════════════════════════════════

def gen_query_schedule():
    s = random.randint(0, 21)
    if s == 0:
        d = rd()
        return (f"Cosa ho in agenda {d}?",
                {"intent": "query_schedule", "date": d})
    elif s == 1:
        d, o = rd(), ro()
        return (f"Sono libero {d} alle {o}?",
                {"intent": "query_schedule", "date": d, "time": o, "query_type": "availability"})
    elif s == 2:
        n, d = rn(), rd()
        return (f"Ho qualcosa con {n} {d}?",
                {"intent": "query_schedule", "contact": n, "date": d})
    elif s == 3:
        dp = rdp()
        return (f"Mostrami tutti gli eventi di {dp}.",
                {"intent": "query_schedule", "date": dp})
    elif s == 4:
        eg = rev(); ev = eg[0]; d = rd()
        return (f"Quante {ev} ho {d}?",
                {"intent": "query_schedule", "date": d, "query_type": "count"})
    elif s == 5:
        n = rn()
        return (f"A che ora è la riunione con {n}?",
                {"intent": "query_schedule", "contact": n, "query_type": "time_of_event"})
    elif s == 6:
        ru = rru()
        return (f"Quando è il prossimo appuntamento con il {ru}?",
                {"intent": "query_schedule", "contact": ru, "query_type": "next_occurrence"})
    elif s == 7:
        d, p = rd(), rp()
        return (f"Ho slot liberi {d} {p}?",
                {"intent": "query_schedule", "date": d, "time": p, "query_type": "availability"})
    elif s == 8:
        dp, p = rdp(), rp()
        return (f"Cosa ho in programma {dp} nel {p}?",
                {"intent": "query_schedule", "date": dp, "time": p})
    elif s == 9:
        d, o = rd(), ro()
        return (f"Ci sono conflitti {d} alle {o}?",
                {"intent": "query_schedule", "date": d, "time": o, "query_type": "conflicts"})
    elif s == 10:
        return (f"Dammi un'overview della settimana.",
                {"intent": "query_schedule", "date": "questa settimana", "query_type": "overview"})
    elif s == 11:
        n = rn()
        return (f"Quali impegni ho con {n} nei prossimi giorni?",
                {"intent": "query_schedule", "contact": n, "query_type": "upcoming"})
    elif s == 12:
        da = rda()
        return (f"Cosa ho fissato il {da}?",
                {"intent": "query_schedule", "date": da})
    elif s == 13:
        d, p = rd(), rp()
        return (f"Sono impegnato {d} {p}?",
                {"intent": "query_schedule", "date": d, "time": p, "query_type": "availability"})
    elif s == 14:
        n = rn()
        return (f"Quando rivedo {n} la prossima volta?",
                {"intent": "query_schedule", "contact": n, "query_type": "next_occurrence"})
    elif s == 15:
        o1, o2, d = ro(), ro(), rd()
        while o2 == o1: o2 = ro()
        return (f"Cosa ho tra le {o1} e le {o2} di {d}?",
                {"intent": "query_schedule", "date": d, "time_range": f"{o1}-{o2}"})
    elif s == 16:
        dp = rdp()
        return (f"Mostrami il calendario di {dp}.",
                {"intent": "query_schedule", "date": dp})
    elif s == 17:
        da, o = rda(), ro()
        return (f"Ho già qualcosa in agenda il {da} alle {o}?",
                {"intent": "query_schedule", "date": da, "time": o, "query_type": "availability"})
    elif s == 18:
        d = rd()
        return (f"Qual è il mio primo impegno di {d}?",
                {"intent": "query_schedule", "date": d, "query_type": "first_event"})
    elif s == 19:
        d = rd()
        return (f"Riepilogami la giornata di {d}.",
                {"intent": "query_schedule", "date": d, "query_type": "overview"})
    elif s == 20:
        n = rn()
        return (f"Ho riunioni con {n} questo mese?",
                {"intent": "query_schedule", "contact": n, "date": "questo mese"})
    else:
        d, o = rd(), ro()
        return (f"Cosa ho alle {o} di {d}?",
                {"intent": "query_schedule", "date": d, "time": o})

fill_intent(gen_query_schedule, "query_schedule")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FIND_FREE_SLOT
# ═════════════════════════════════════════════════════════════════════════════

def gen_find_free_slot():
    s = random.randint(0, 13)
    d = rd(); du = rdu(); p = rp(); dp = rdp(); o = ro()
    if s == 0:
        return (f"Trovami uno slot libero {d} per {du}.",
                {"intent": "find_free_slot", "date": d, "duration": du})
    elif s == 1:
        return (f"Quando sono libero {dp}?",
                {"intent": "find_free_slot", "date": dp})
    elif s == 2:
        return (f"Ho {du} liberi {d} {p}?",
                {"intent": "find_free_slot", "date": d, "duration": du, "time_pref": p})
    elif s == 3:
        n = rn()
        return (f"Trova uno spazio {d} per incontrare {n}.",
                {"intent": "find_free_slot", "date": d, "contact": n})
    elif s == 4:
        return (f"Quando posso inserire {du} di riunione {dp}?",
                {"intent": "find_free_slot", "date": dp, "duration": du})
    elif s == 5:
        return (f"Dammi il primo slot libero di {d} {p}.",
                {"intent": "find_free_slot", "date": d, "time_pref": p})
    elif s == 6:
        return (f"Sono disponibile {d} mattina?",
                {"intent": "find_free_slot", "date": d, "time_pref": "mattina"})
    elif s == 7:
        return (f"Trova {du} libere questa settimana nel pomeriggio.",
                {"intent": "find_free_slot", "date": "questa settimana", "duration": du, "time_pref": "pomeriggio"})
    elif s == 8:
        return (f"Ho uno slot di {du} libero {dp}?",
                {"intent": "find_free_slot", "date": dp, "duration": du})
    elif s == 9:
        n = rn()
        return (f"Quando sono libero per {du} per parlare con {n}?",
                {"intent": "find_free_slot", "duration": du, "contact": n})
    elif s == 10:
        return (f"Trovami un buco in agenda {d} pomeriggio.",
                {"intent": "find_free_slot", "date": d, "time_pref": "pomeriggio"})
    elif s == 11:
        return (f"Quali ore ho libere {dp}?",
                {"intent": "find_free_slot", "date": dp, "query_type": "list"})
    elif s == 12:
        return (f"Cerca uno slot di almeno {du} {d}.",
                {"intent": "find_free_slot", "date": d, "duration": du})
    else:
        return (f"Quando posso aggiungere una call {dp} senza conflitti?",
                {"intent": "find_free_slot", "date": dp})

fill_intent(gen_find_free_slot, "find_free_slot")

# ═════════════════════════════════════════════════════════════════════════════
# 8. ADD_REMINDER
# ═════════════════════════════════════════════════════════════════════════════

REM_VERBS = ["Ricordami", "Metti un reminder", "Aggiungi un promemoria",
             "Metti un promemoria", "Avvisami", "Metti un avviso"]

def gen_add_reminder():
    s = random.randint(0, 15)
    v = random.choice(REM_VERBS)
    n = rn(); ru = rru()
    nota = rnota(n, ru)
    if s == 0:
        d, o = rd(), ro()
        return (f"{v} di {nota} {d} alle {o}.",
                {"intent": "add_reminder", "note": nota, "date": d, "time": o})
    elif s == 1:
        d = rd()
        return (f"{v} di {nota} entro {d}.",
                {"intent": "add_reminder", "note": nota, "deadline": d})
    elif s == 2:
        dp = rdp()
        return (f"{v} di {nota} {dp}.",
                {"intent": "add_reminder", "note": nota, "date": dp})
    elif s == 3:
        da, o = rda(), ro()
        return (f"Imposta un reminder per il {da} alle {o}: {nota}.",
                {"intent": "add_reminder", "note": nota, "date": da, "time": o})
    elif s == 4:
        da = rda()
        return (f"{v} di {nota} il {da}.",
                {"intent": "add_reminder", "note": nota, "date": da})
    elif s == 5:
        d, o = rd(), ro()
        eg = rev()
        return (f"{v} di {nota} prima {art_prep_del(eg)} di {d}.",
                {"intent": "add_reminder", "note": nota, "date": d})
    elif s == 6:
        da = rda()
        return (f"Aggiungi promemoria: {nota}, scadenza {da}.",
                {"intent": "add_reminder", "note": nota, "deadline": da})
    elif s == 7:
        return (f"{v} di {nota} domani mattina.",
                {"intent": "add_reminder", "note": nota, "date": "domani", "time": "mattina"})
    elif s == 8:
        dp = rdp()
        return (f"Non voglio dimenticare di {nota}. Metti un reminder per {dp}.",
                {"intent": "add_reminder", "note": nota, "date": dp})
    elif s == 9:
        d, o = rd(), ro()
        return (f"Avvisami {d} alle {o}: {nota}.",
                {"intent": "add_reminder", "note": nota, "date": d, "time": o})
    elif s == 10:
        da = rda()
        return (f"Devo {nota} entro il {da}. Ricordamelo.",
                {"intent": "add_reminder", "note": nota, "deadline": da})
    elif s == 11:
        d, p = rd(), rp()
        return (f"Metti avviso per {d} {p}: {nota}.",
                {"intent": "add_reminder", "note": nota, "date": d, "time": p})
    elif s == 12:
        o = ro()
        return (f"Reminder: {nota}, oggi entro le {o}.",
                {"intent": "add_reminder", "note": nota, "deadline": "oggi", "time": o})
    elif s == 13:
        dp = rdp()
        return (f"Crea promemoria per {dp}: {nota}.",
                {"intent": "add_reminder", "note": nota, "date": dp})
    elif s == 14:
        d = rd()
        return (f"Segna di {nota} per {d}.",
                {"intent": "add_reminder", "note": nota, "date": d})
    else:
        return (f"Aggiungimi un promemoria di {nota} per stamattina.",
                {"intent": "add_reminder", "note": nota, "date": "oggi", "time": "mattina"})

fill_intent(gen_add_reminder, "add_reminder")

# ═════════════════════════════════════════════════════════════════════════════
# 9. UPDATE_REMINDER
# ═════════════════════════════════════════════════════════════════════════════

def gen_update_reminder():
    s = random.randint(0, 13)
    n = rn(); ru = rru()
    nota = rnota(n, ru)
    nota2 = rnota(rn(), rru())
    d = rd(); dp = rdp(); o = ro(); da = rda()
    if s == 0:
        return (f"Sposta il reminder di {nota} a {dp}.",
                {"intent": "update_reminder", "note_ref": nota, "new_date": dp})
    elif s == 1:
        return (f"Aggiorna il promemoria di {nota}: ora alle {o}.",
                {"intent": "update_reminder", "note_ref": nota, "new_time": o})
    elif s == 2:
        return (f"Cambia il promemoria '{nota}' in '{nota2}'.",
                {"intent": "update_reminder", "note_ref": nota, "new_note": nota2})
    elif s == 3:
        return (f"Posticipa il reminder di {nota} a {da}.",
                {"intent": "update_reminder", "note_ref": nota, "new_date": da})
    elif s == 4:
        return (f"Aggiorna il promemoria di domani: spostalo a {dp} alle {o}.",
                {"intent": "update_reminder", "note_ref": "promemoria domani", "new_date": dp, "new_time": o})
    elif s == 5:
        return (f"Modifica il reminder di {nota}: cambia l'orario alle {o}.",
                {"intent": "update_reminder", "note_ref": nota, "new_time": o})
    elif s == 6:
        return (f"Il promemoria di {nota} va spostato a {d}.",
                {"intent": "update_reminder", "note_ref": nota, "new_date": d})
    elif s == 7:
        return (f"Rinvia il reminder di {nota} di un giorno.",
                {"intent": "update_reminder", "note_ref": nota, "new_date": "domani"})
    elif s == 8:
        return (f"Aggiorna il promemoria: {nota} → {nota2}.",
                {"intent": "update_reminder", "note_ref": nota, "new_note": nota2})
    elif s == 9:
        return (f"Sposta il reminder delle {o} di {d} alle {ro()}.",
                {"intent": "update_reminder", "note_ref": f"promemoria {o} {d}", "new_time": o})
    elif s == 10:
        return (f"Anticipa il promemoria di {nota} a {d} mattina.",
                {"intent": "update_reminder", "note_ref": nota, "new_date": d, "new_time": "mattina"})
    elif s == 11:
        return (f"Cambia il testo del promemoria su {nota}: ora scrivi '{nota2}'.",
                {"intent": "update_reminder", "note_ref": nota, "new_note": nota2})
    elif s == 12:
        return (f"Posticipa di una settimana il reminder di {nota}.",
                {"intent": "update_reminder", "note_ref": nota, "new_date": "tra una settimana"})
    else:
        return (f"Aggiorna il reminder di domani mattina: spostalo a {dp}.",
                {"intent": "update_reminder", "note_ref": "promemoria domani mattina", "new_date": dp})

fill_intent(gen_update_reminder, "update_reminder")

# ═════════════════════════════════════════════════════════════════════════════
# 10. DELETE_REMINDER
# ═════════════════════════════════════════════════════════════════════════════

def gen_delete_reminder():
    s = random.randint(0, 12)
    v = random.choice(DEL_VERBS)
    n = rn(); ru = rru()
    nota = rnota(n, ru)
    d = rd(); dp = rdp(); da = rda(); o = ro()
    if s == 0:
        return (f"{v} il promemoria di {nota}.",
                {"intent": "delete_reminder", "note_ref": nota})
    elif s == 1:
        return (f"Togli il reminder di {d}.",
                {"intent": "delete_reminder", "date": d})
    elif s == 2:
        return (f"Non mi serve più il reminder di {nota}. Cancellalo.",
                {"intent": "delete_reminder", "note_ref": nota})
    elif s == 3:
        return (f"{v} tutti i promemoria di {dp}.",
                {"intent": "delete_reminder", "date": dp})
    elif s == 4:
        return (f"Elimina il promemoria delle {o} di {d}.",
                {"intent": "delete_reminder", "date": d, "time": o})
    elif s == 5:
        return (f"Ho già fatto {nota}. Togli il reminder.",
                {"intent": "delete_reminder", "note_ref": nota})
    elif s == 6:
        return (f"Cancella il reminder di domani mattina.",
                {"intent": "delete_reminder", "date": "domani", "time": "mattina"})
    elif s == 7:
        return (f"{v} tutti i promemoria della settimana.",
                {"intent": "delete_reminder", "date": "questa settimana"})
    elif s == 8:
        return (f"Rimuovi il promemoria sul {nota}.",
                {"intent": "delete_reminder", "note_ref": nota})
    elif s == 9:
        return (f"{v} il reminder delle {o}.",
                {"intent": "delete_reminder", "time": o})
    elif s == 10:
        return (f"Svuota tutti i promemoria di {da}.",
                {"intent": "delete_reminder", "date": da})
    elif s == 11:
        return (f"Togli il promemoria di oggi.",
                {"intent": "delete_reminder", "date": "oggi"})
    else:
        return (f"Non serve più ricordarmi di {nota}. Elimina il reminder.",
                {"intent": "delete_reminder", "note_ref": nota})

fill_intent(gen_delete_reminder, "delete_reminder")

# ═════════════════════════════════════════════════════════════════════════════
# 11. ADD_CONTACT
# ═════════════════════════════════════════════════════════════════════════════

def gen_add_contact():
    s = random.randint(0, 14)
    n = rn(); cog = rcog()
    nome_c = f"{n} {cog}"
    phone = rphone(); email = remail(n); ru = rru()
    if s == 0:
        return (f"Aggiungi {nome_c} con numero {phone}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone})
    elif s == 1:
        return (f"Salva il contatto di {nome_c}: {email}.",
                {"intent": "add_contact", "name": nome_c, "email": email})
    elif s == 2:
        return (f"Nuovo contatto: {nome_c}, {ru}, {phone}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "role": ru})
    elif s == 3:
        return (f"Aggiungi alla rubrica {nome_c}: {email}, {phone}.",
                {"intent": "add_contact", "name": nome_c, "email": email, "phone": phone})
    elif s == 4:
        return (f"Crea contatto per il {ru}: {nome_c}, {phone}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "role": ru})
    elif s == 5:
        return (f"Salva questo numero come {nome_c}: {phone}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone})
    elif s == 6:
        return (f"Aggiungi il {ru} {nome_c}: {email}.",
                {"intent": "add_contact", "name": nome_c, "email": email, "role": ru})
    elif s == 7:
        return (f"Aggiungi contatto: {nome_c}, ruolo {ru}, cell. {phone}, email {email}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "email": email, "role": ru})
    elif s == 8:
        return (f"Salva il recapito di {nome_c}: {phone}. È il nostro {ru}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "role": ru})
    elif s == 9:
        return (f"Inserisci nella rubrica: {nome_c} ({ru}), {phone}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "role": ru})
    elif s == 10:
        return (f"Nuovo fornitore: {nome_c}, {phone}, {email}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "email": email, "role": "fornitore"})
    elif s == 11:
        phone_fisso = rphone(fisso=True)
        return (f"Salva il numero fisso di {nome_c}: {phone_fisso}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone_fisso})
    elif s == 12:
        return (f"Crea contatto per il nuovo cliente: {nome_c}, {email}.",
                {"intent": "add_contact", "name": nome_c, "email": email, "role": "cliente"})
    elif s == 13:
        return (f"Aggiungi {nome_c} come contatto professionale: {email}.",
                {"intent": "add_contact", "name": nome_c, "email": email})
    else:
        return (f"Salva {nome_c}: {phone}, {email}. Ruolo: {ru}.",
                {"intent": "add_contact", "name": nome_c, "phone": phone, "email": email, "role": ru})

fill_intent(gen_add_contact, "add_contact")

# ═════════════════════════════════════════════════════════════════════════════
# 12. QUERY_CONTACT
# ═════════════════════════════════════════════════════════════════════════════

def gen_query_contact():
    s = random.randint(0, 14)
    n = rn(); ru = rru(); campo = random.choice(CAMPI_CONTATTO)
    if s == 0:
        return (f"Dammi il numero di {n}.",
                {"intent": "query_contact", "name": n, "field": "numero"})
    elif s == 1:
        return (f"Qual è l'email di {n}?",
                {"intent": "query_contact", "name": n, "field": "email"})
    elif s == 2:
        return (f"Hai i recapiti di {n}?",
                {"intent": "query_contact", "name": n})
    elif s == 3:
        return (f"Dimmi il telefono del {ru}.",
                {"intent": "query_contact", "name": ru, "field": "telefono"})
    elif s == 4:
        return (f"Come posso contattare {n}?",
                {"intent": "query_contact", "name": n})
    elif s == 5:
        return (f"Qual è il ruolo di {n}?",
                {"intent": "query_contact", "name": n, "field": "ruolo"})
    elif s == 6:
        cog = rcog()
        return (f"Cerca il contatto {n} {cog}.",
                {"intent": "query_contact", "name": f"{n} {cog}"})
    elif s == 7:
        return (f"Trova il {campo} di {n}.",
                {"intent": "query_contact", "name": n, "field": campo})
    elif s == 8:
        return (f"Dammi tutti i dati di {n}.",
                {"intent": "query_contact", "name": n})
    elif s == 9:
        return (f"Che numero ha {n}?",
                {"intent": "query_contact", "name": n, "field": "numero"})
    elif s == 10:
        return (f"Hai l'email del {ru}?",
                {"intent": "query_contact", "name": ru, "field": "email"})
    elif s == 11:
        return (f"Mostrami il contatto di {n}.",
                {"intent": "query_contact", "name": n})
    elif s == 12:
        return (f"Qual è il numero di cellulare di {n}?",
                {"intent": "query_contact", "name": n, "field": "telefono"})
    elif s == 13:
        return (f"Dove lavora {n}? Qual è il suo ruolo?",
                {"intent": "query_contact", "name": n, "field": "ruolo"})
    else:
        return (f"Hai salvato il contatto di {n} {rcog()}?",
                {"intent": "query_contact", "name": f"{n} {rcog()}"})

fill_intent(gen_query_contact, "query_contact")

# ═════════════════════════════════════════════════════════════════════════════
# 13. UPDATE_CONTACT
# ═════════════════════════════════════════════════════════════════════════════

def gen_update_contact():
    s = random.randint(0, 14)
    n = rn(); ru = rru()
    new_phone = rphone(); new_email = remail(n)
    campo = random.choice(CAMPI_CONTATTO)
    if s == 0:
        return (f"Aggiorna il numero di {n}: ora è {new_phone}.",
                {"intent": "update_contact", "name": n, "field": "numero", "new_value": new_phone})
    elif s == 1:
        return (f"Cambia l'email di {n} in {new_email}.",
                {"intent": "update_contact", "name": n, "field": "email", "new_value": new_email})
    elif s == 2:
        return (f"Aggiorna il ruolo di {n}: adesso è {ru}.",
                {"intent": "update_contact", "name": n, "field": "ruolo", "new_value": ru})
    elif s == 3:
        return (f"Il numero di {n} è cambiato: {new_phone}.",
                {"intent": "update_contact", "name": n, "field": "numero", "new_value": new_phone})
    elif s == 4:
        return (f"Modifica il contatto di {n}: nuovo numero {new_phone}.",
                {"intent": "update_contact", "name": n, "field": "numero", "new_value": new_phone})
    elif s == 5:
        return (f"Aggiorna l'email del {ru} con {new_email}.",
                {"intent": "update_contact", "name": ru, "field": "email", "new_value": new_email})
    elif s == 6:
        cog = rcog(); new_cog = rcog()
        while new_cog == cog: new_cog = rcog()
        return (f"{n} {cog} si è sposata. Ora si chiama {n} {new_cog}.",
                {"intent": "update_contact", "name": f"{n} {cog}", "field": "nome", "new_value": f"{n} {new_cog}"})
    elif s == 7:
        return (f"Salva il nuovo numero di {n}: {new_phone}. Il vecchio non è valido.",
                {"intent": "update_contact", "name": n, "field": "numero", "new_value": new_phone})
    elif s == 8:
        return (f"Aggiorna il {campo} di {n}.",
                {"intent": "update_contact", "name": n, "field": campo, "new_value": None})
    elif s == 9:
        return (f"Il {ru} ora ha un nuovo telefono: {new_phone}.",
                {"intent": "update_contact", "name": ru, "field": "telefono", "new_value": new_phone})
    elif s == 10:
        return (f"Modifica l'indirizzo email di {n}: {new_email}.",
                {"intent": "update_contact", "name": n, "field": "email", "new_value": new_email})
    elif s == 11:
        return (f"Aggiorna il profilo di {n}: è diventato/a {ru}.",
                {"intent": "update_contact", "name": n, "field": "ruolo", "new_value": ru})
    elif s == 12:
        return (f"Correggi il numero di {n}: è {new_phone}, non quello vecchio.",
                {"intent": "update_contact", "name": n, "field": "numero", "new_value": new_phone})
    elif s == 13:
        return (f"Il {ru} ha cambiato email: {new_email}.",
                {"intent": "update_contact", "name": ru, "field": "email", "new_value": new_email})
    else:
        return (f"Aggiorna i dati di {n}: nuovo numero {new_phone}, nuova email {new_email}.",
                {"intent": "update_contact", "name": n, "field": "numero", "new_value": new_phone})

fill_intent(gen_update_contact, "update_contact")

# ═════════════════════════════════════════════════════════════════════════════
# 14. DELETE_CONTACT
# ═════════════════════════════════════════════════════════════════════════════

def gen_delete_contact():
    s = random.randint(0, 11)
    v = random.choice(DEL_VERBS)
    n = rn(); ru = rru(); cog = rcog()
    if s == 0:
        return (f"{v} il contatto di {n}.",
                {"intent": "delete_contact", "name": n})
    elif s == 1:
        return (f"Rimuovi {n} {cog} dalla rubrica.",
                {"intent": "delete_contact", "name": f"{n} {cog}"})
    elif s == 2:
        return (f"Non lavoro più con {n}. {v} il suo contatto.",
                {"intent": "delete_contact", "name": n})
    elif s == 3:
        return (f"Togli il {ru} {n} dai contatti.",
                {"intent": "delete_contact", "name": n, "role": ru})
    elif s == 4:
        return (f"{v} il numero di {n} {cog}.",
                {"intent": "delete_contact", "name": f"{n} {cog}"})
    elif s == 5:
        return (f"{n} non fa più parte del team. {v}lo dalla rubrica.",
                {"intent": "delete_contact", "name": n})
    elif s == 6:
        return (f"Rimuovi dalla rubrica: {n}.",
                {"intent": "delete_contact", "name": n})
    elif s == 7:
        return (f"Cancella il contatto del {ru}.",
                {"intent": "delete_contact", "name": ru})
    elif s == 8:
        return (f"Ho sbagliato a salvare {n}. {v}lo.",
                {"intent": "delete_contact", "name": n})
    elif s == 9:
        return (f"Non ho più bisogno del contatto di {n} {cog}. Toglilo.",
                {"intent": "delete_contact", "name": f"{n} {cog}"})
    elif s == 10:
        return (f"Elimina {n} dalla lista contatti.",
                {"intent": "delete_contact", "name": n})
    else:
        return (f"Rimuovi il vecchio {ru}: {n} {cog}.",
                {"intent": "delete_contact", "name": f"{n} {cog}", "role": ru})

fill_intent(gen_delete_contact, "delete_contact")

# ═════════════════════════════════════════════════════════════════════════════
# 15. ADD_BIRTHDAY
# ═════════════════════════════════════════════════════════════════════════════

def gen_add_birthday():
    s = random.randint(0, 13)
    n = rn(); da = rda(); mese = rmese(); giorno = random.randint(1, 28)
    data_comp = f"{giorno} {mese}"
    if s == 0:
        return (f"Il compleanno di {n} è il {data_comp}. Salvalo.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 1:
        return (f"Aggiungi il compleanno di {n}: {da}.",
                {"intent": "add_birthday", "name": n, "date": da})
    elif s == 2:
        return (f"Segna che {n} compie gli anni il {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 3:
        return (f"Salva il compleanno di {n}: {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 4:
        cog = rcog()
        return (f"Il {data_comp} è il compleanno di {n} {cog}. Aggiungilo.",
                {"intent": "add_birthday", "name": f"{n} {cog}", "date": data_comp})
    elif s == 5:
        return (f"Inserisci nella rubrica: {n} compie gli anni il {da}.",
                {"intent": "add_birthday", "name": n, "date": da})
    elif s == 6:
        return (f"Ricordami il compleanno di {n} il {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 7:
        return (f"Aggiungi: {n} - compleanno {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 8:
        return (f"Salva che {n} festeggia il {da}.",
                {"intent": "add_birthday", "name": n, "date": da})
    elif s == 9:
        return (f"Segna il compleanno di {n}: il {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 10:
        ru = rru()
        return (f"Il {ru} {n} compie gli anni il {data_comp}. Aggiungilo al calendario.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 11:
        return (f"Memorizza che {n} nasce il {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    elif s == 12:
        return (f"Aggiungi evento annuale: compleanno di {n} il {data_comp}.",
                {"intent": "add_birthday", "name": n, "date": data_comp})
    else:
        return (f"Crea promemoria annuale per il compleanno di {n}: {da}.",
                {"intent": "add_birthday", "name": n, "date": da})

fill_intent(gen_add_birthday, "add_birthday")

# ═════════════════════════════════════════════════════════════════════════════
# 16. QUERY_BIRTHDAY
# ═════════════════════════════════════════════════════════════════════════════

def gen_query_birthday():
    s = random.randint(0, 13)
    n = rn(); mese = rmese()
    if s == 0:
        return (f"Quando è il compleanno di {n}?",
                {"intent": "query_birthday", "name": n})
    elif s == 1:
        return (f"Ci sono compleanni a {mese}?",
                {"intent": "query_birthday", "month": mese})
    elif s == 2:
        return (f"Dammi i compleanni di questo mese.",
                {"intent": "query_birthday", "month": "questo mese"})
    elif s == 3:
        return (f"Quando compie gli anni {n}?",
                {"intent": "query_birthday", "name": n})
    elif s == 4:
        return (f"Ci sono compleanni questa settimana?",
                {"intent": "query_birthday", "month": "questa settimana"})
    elif s == 5:
        return (f"Ho compleanni da ricordare a {mese}?",
                {"intent": "query_birthday", "month": mese})
    elif s == 6:
        return (f"Qual è la data di nascita di {n}?",
                {"intent": "query_birthday", "name": n})
    elif s == 7:
        return (f"Mostrami tutti i compleanni salvati.",
                {"intent": "query_birthday"})
    elif s == 8:
        return (f"Quando festeggia {n}?",
                {"intent": "query_birthday", "name": n})
    elif s == 9:
        return (f"Quanti compleanni ho a {mese}?",
                {"intent": "query_birthday", "month": mese})
    elif s == 10:
        return (f"Chi compie gli anni questa settimana?",
                {"intent": "query_birthday", "month": "questa settimana"})
    elif s == 11:
        return (f"Il compleanno di {n} è questo mese?",
                {"intent": "query_birthday", "name": n})
    elif s == 12:
        return (f"Quali compleanni ho segnato per {mese}?",
                {"intent": "query_birthday", "month": mese})
    else:
        return (f"Dammi la data del compleanno di {n}.",
                {"intent": "query_birthday", "name": n})

fill_intent(gen_query_birthday, "query_birthday")

# ═════════════════════════════════════════════════════════════════════════════
# 17. QUERY_DATETIME
# ═════════════════════════════════════════════════════════════════════════════

_DT_TIME = [
    "Che ora è?", "Che ore sono?", "Dimmi l'ora.", "Che ora è adesso?",
    "Che ora è in questo momento?", "Quante ore mancano a mezzanotte?",
    "È mattina o pomeriggio?", "Abbiamo superato mezzogiorno?", "Siamo al mattino?",
    "Sai che ora è?", "Mi dici che ore sono?", "Hai l'ora?", "Quanto sono le?",
    "L'ora esatta?", "Quant'è tardi?", "Sono già le 9?", "Sono passate le 17?",
    "Siamo ancora in orario di lavoro?", "Quanto manca all'ora di pranzo?",
    "Dimmi l'orario corrente.", "Che orario segna l'orologio?",
    "È già tarda mattinata?", "Sono passate le 8?", "Mancano pochi minuti a mezzogiorno?",
    "A che ora siamo adesso?", "Quante ore sono trascorse dalla mezzanotte?",
    "L'orologio dice che ore sono?", "Sono le 15 circa?",
    "Dimmi solo l'ora.", "Che ora è precisa?",
    "È ora di pranzo?", "È tarda sera?", "Siamo in orario di ufficio?",
    "Quanti minuti mancano alle 12?", "Siamo oltre le 18?",
]

_DT_DATE = [
    "Che giorno è oggi?", "Che giorno è?", "Qual è la data di oggi?",
    "Dimmi la data.", "Che data è?", "Qual è la data corrente?",
    "Che settimana è?", "Siamo all'inizio o alla fine del mese?",
    "Quanti giorni mancano alla fine del mese?", "Che data abbiamo oggi?",
    "Oggi quanti ne abbiamo?", "Qual è il giorno corrente?", "Dimmi oggi che giorno è.",
    "Che giorno è domani?", "La data di oggi qual è?", "Siamo al 10 del mese?",
    "Siamo nella prima o seconda metà del mese?", "Quanti giorni ha questo mese?",
    "Siamo verso la fine del mese?", "Che data è oggi esattamente?",
    "Qual è la data odierna?", "Dimmi il giorno di oggi.", "Oggi è il quindici?",
    "Qual è il numero del giorno corrente?", "La data precisa?",
    "Siamo all'inizio del mese?", "Quanti giorni ci sono stati questo mese?",
    "Quanti ne abbiamo oggi?", "Dimmi la data completa.", "È il primo del mese?",
    "Oggi è l'ultimo del mese?", "Quante settimane abbiamo fatto questo mese?",
    "Siamo a metà mese?", "Che giorno è oggi, lo sai?", "In che data siamo?",
    "Qual è la data precisa di oggi?", "Oggi è che giorno?",
    "Dimmi il giorno e la data.", "Quanti giorni sono passati dall'inizio del mese?",
    "Siamo al ventesimo del mese?", "È il fine mese?", "Siamo nella seconda settimana del mese?",
]

_DT_DOW = [
    "Che giorno della settimana è?", "Siamo di lunedì o di martedì?", "Oggi è lunedì?",
    "In che giorno della settimana siamo?", "Oggi è venerdì?", "Siamo a metà settimana?",
    "È già giovedì?", "Che giorno della settimana cade oggi?", "Oggi è mercoledì?",
    "Siamo a martedì?", "In che giorno feriale siamo?", "È il primo giorno lavorativo?",
    "Oggi è un giorno festivo?", "È weekend?", "Siamo in settimana lavorativa?",
    "Che weekday è?", "Oggi è giovedì o venerdì?", "Siamo a inizio settimana?",
    "È già fine settimana?", "Oggi cade di che giorno?",
    "In quale giorno della settimana siamo?", "Oggi è martedì?",
    "Il giorno della settimana corrente?", "Siamo di mercoledì?",
    "È il penultimo giorno lavorativo?", "Oggi è il primo giorno lavorativo della settimana?",
    "È già metà settimana?", "Siamo al quarto giorno lavorativo?",
    "Che giorno è questa settimana?", "Oggi è lunedì o martedì?",
    "In che giorno ci troviamo oggi?", "È sabato?", "Oggi è domenica?",
    "Siamo ancora a inizio settimana?", "Domani è l'ultimo giorno lavorativo?",
    "Siamo nel mezzo della settimana?", "Oggi è un giorno lavorativo?",
    "Quanti giorni lavorativi restano questa settimana?",
    "Siamo nella seconda metà della settimana?",
    "Oggi è venerdì?", "Il fine settimana è vicino?",
    "Domani è sabato?", "Quante giornate lavorative ha questa settimana?",
]

_DT_MONTH = [
    "In che mese siamo?", "Qual è il mese corrente?", "Siamo a settembre?",
    "In che periodo dell'anno siamo?", "Qual è il mese di oggi?", "Siamo a ottobre?",
    "In che mese ci troviamo?", "Il mese corrente qual è?", "Siamo ancora a novembre?",
    "Qual è il quinto mese dell'anno?", "Siamo nel mese di gennaio?",
    "In quale mese siamo?", "Qual è il mese attuale?", "Siamo a dicembre?",
    "Quanti mesi mancano alla fine dell'anno?", "Siamo nel terzo trimestre?",
    "È un mese pari o dispari?", "Siamo a metà anno?", "Qual è il numero del mese corrente?",
    "Siamo in primavera o autunno?", "Qual è il mese in corso?",
    "Siamo ancora in estate?", "Quanti mesi abbiamo già percorso?",
    "Siamo nell'ultimo trimestre dell'anno?", "In che stagione siamo?",
    "Il mese attuale è estivo?", "Siamo nel secondo semestre?",
    "In quale mese dell'anno ci troviamo?", "Siamo a febbraio?",
    "Questo è l'undicesimo mese?", "Siamo ancora in inverno?",
    "Qual è il mese presente?", "In che mese ci troviamo adesso?",
    "Siamo in autunno?", "È il mese di agosto?",
]

_DT_YEAR = [
    "In che anno siamo?", "Qual è l'anno in corso?", "Siamo nel 2026?",
    "Dimmi l'anno attuale.", "Qual è l'anno corrente?", "In quale anno ci troviamo?",
    "Siamo ancora nel 2026?", "L'anno attuale qual è?", "Qual è l'anno presente?",
    "In che anno viviamo?", "Siamo nel ventunesimo secolo?",
    "In quale anno siamo adesso?", "Il 2026 è l'anno corrente?",
    "Quanti anni sono passati dal 2020?", "In che decennio siamo?",
    "Siamo ancora agli anni '20?", "L'anno corrente è un anno bisestile?",
    "Qual è il numero dell'anno in corso?", "Siamo oltre il 2025?",
    "In che anno ci troviamo oggi?", "L'anno è il 2026?",
    "Qual è l'anno presente esatto?", "Siamo nel terzo millennio?",
    "Quanti anni sono trascorsi dal 2000?", "In che anno siamo entrati?",
    "Dimmi solo l'anno.", "Qual è l'annata corrente?",
    "Siamo ancora nei primi anni 2020?", "L'anno solare attuale?",
    "In quale anno solare ci troviamo?",
    "Quanti anni mancano al 2030?", "L'anno corrente è pari o dispari?",
    "Siamo nell'anno parigino o olimpico?", "Qual è l'ultimo anno completato?",
    "In quale anno del decennio siamo?",
    "Quante ore precise mancano alla fine dell'anno?",
    "Sono quasi le 16?", "È già mezzogiorno passato?",
    "Quanti minuti mancano alle 18?", "Siamo prima delle 10 di mattina?",
    "L'orario attuale supera le 12?", "Sono le 9 di mattina circa?",
    "Quanto manca alle 15?", "È già notte?", "Siamo nel pomeriggio tardi?",
    "La giornata lavorativa è finita?",
]

_dt_all = (
    [("time", p) for p in _DT_TIME] +
    [("date", p) for p in _DT_DATE] +
    [("day_of_week", p) for p in _DT_DOW] +
    [("month", p) for p in _DT_MONTH] +
    [("year", p) for p in _DT_YEAR]
)
random.shuffle(_dt_all)
_dt_idx = [0]

def gen_query_datetime():
    idx = _dt_idx[0] % len(_dt_all)
    _dt_idx[0] += 1
    field, prompt = _dt_all[idx]
    return (prompt, {"intent": "query_datetime", "field": field})

fill_intent(gen_query_datetime, "query_datetime")

# ═════════════════════════════════════════════════════════════════════════════
# 18. QUERY_DURATION
# ═════════════════════════════════════════════════════════════════════════════

DURATION_TEMPLATES_UNTIL = [
    "Quanti giorni mancano a {target}?",
    "Quanto manca a {target}?",
    "Fra quanto è {target}?",
    "Quanto tempo manca a {target}?",
    "Quante settimane mancano a {target}?",
    "Quando arriva {target}? Quanti giorni mancano?",
    "Tra quanti giorni è {target}?",
    "Quanto ci vuole ad arrivare a {target}?",
    "Fra quanti giorni è {target}?",
    "Quanto tempo ho prima di {target}?",
]

DURATION_TEMPLATES_SINCE = [
    "Quanto tempo è passato da {target}?",
    "Quanti giorni sono passati da {target}?",
    "Da quanto tempo siamo dopo {target}?",
    "Quante settimane sono passate da {target}?",
    "Quanto è trascorso da {target}?",
    "Da quanti giorni è passato {target}?",
    "Quanto fa da {target} a oggi?",
]

_dur_until = [(t, "until") for t in DURATION_TEMPLATES_UNTIL]
_dur_since = [(t, "since") for t in DURATION_TEMPLATES_SINCE]
_dur_all = _dur_until * 8 + _dur_since * 5  # until più frequente
random.shuffle(_dur_all)
_dur_idx = [0]

def gen_query_duration():
    idx = _dur_idx[0] % len(_dur_all)
    _dur_idx[0] += 1
    tmpl, direction = _dur_all[idx]
    target = random.choice(TARGET_DURATA)
    prompt = tmpl.format(target=target)
    return (prompt, {"intent": "query_duration", "target": target, "direction": direction})

fill_intent(gen_query_duration, "query_duration")

# ═════════════════════════════════════════════════════════════════════════════
# 19. QUERY_HOLIDAY
# ═════════════════════════════════════════════════════════════════════════════

def gen_query_holiday():
    s = random.randint(0, 13)
    fest = rfest(); mese = rmese()
    if s == 0:
        return (f"Quando cade {fest} quest'anno?",
                {"intent": "query_holiday", "name": fest, "query_type": "date"})
    elif s == 1:
        return (f"Ci sono festività a {mese}?",
                {"intent": "query_holiday", "month": mese, "query_type": "list"})
    elif s == 2:
        return (f"Quante festività ci sono a {mese}?",
                {"intent": "query_holiday", "month": mese, "query_type": "count"})
    elif s == 3:
        return (f"Che giorno è {fest} quest'anno?",
                {"intent": "query_holiday", "name": fest, "query_type": "date"})
    elif s == 4:
        return (f"Quali festività ci sono in questo mese?",
                {"intent": "query_holiday", "month": "questo mese", "query_type": "list"})
    elif s == 5:
        return (f"È {fest} un giorno festivo in Italia?",
                {"intent": "query_holiday", "name": fest, "query_type": "is_holiday"})
    elif s == 6:
        return (f"Dammi le festività nazionali di {mese}.",
                {"intent": "query_holiday", "month": mese, "query_type": "list"})
    elif s == 7:
        return (f"Quando è {fest}?",
                {"intent": "query_holiday", "name": fest, "query_type": "date"})
    elif s == 8:
        return (f"Quali sono i giorni festivi di {mese}?",
                {"intent": "query_holiday", "month": mese, "query_type": "list"})
    elif s == 9:
        return (f"Il {random.randint(1, 28)} {mese} è festivo?",
                {"intent": "query_holiday", "month": mese, "query_type": "is_holiday"})
    elif s == 10:
        return (f"Mostrami le feste nazionali italiane di quest'anno.",
                {"intent": "query_holiday", "query_type": "list"})
    elif s == 11:
        return (f"Quanti ponti ci sono a {mese}?",
                {"intent": "query_holiday", "month": mese, "query_type": "bridges"})
    elif s == 12:
        return (f"C'è qualche giorno di chiusura a {mese}?",
                {"intent": "query_holiday", "month": mese, "query_type": "list"})
    else:
        return (f"Quando festeggiamo {fest}?",
                {"intent": "query_holiday", "name": fest, "query_type": "date"})

fill_intent(gen_query_holiday, "query_holiday")

# ═════════════════════════════════════════════════════════════════════════════
# 20. SET_STATUS
# ═════════════════════════════════════════════════════════════════════════════

def gen_set_status():
    s = random.randint(0, 17)
    status = random.choice(STATI_STATUS)
    d = rd(); dp = rdp(); da = rda()
    da2 = rda()
    while da2 == da: da2 = rda()
    mese = rmese(); giorno = random.randint(1, 28)
    data_c = f"{giorno} {mese}"
    if s == 0:
        return (f"Sono {status} {d}.",
                {"intent": "set_status", "status": status, "date_start": d})
    elif s == 1:
        return (f"Segna che sono {status} da {d} a {dp}.",
                {"intent": "set_status", "status": status, "date_start": d, "date_end": dp})
    elif s == 2:
        return (f"Non sarò disponibile {d}.",
                {"intent": "set_status", "status": "non disponibile", "date_start": d})
    elif s == 3:
        return (f"Sono fuori ufficio dal {da} al {da2}.",
                {"intent": "set_status", "status": "fuori ufficio", "date_start": da, "date_end": da2})
    elif s == 4:
        return (f"Imposta il mio stato come '{status}' per oggi.",
                {"intent": "set_status", "status": status, "date_start": "oggi"})
    elif s == 5:
        return (f"Non sono in ufficio {dp}. Segna che sono {status}.",
                {"intent": "set_status", "status": status, "date_start": dp})
    elif s == 6:
        return (f"Sono in ferie dal {da} al {da2}. Aggiorna il calendario.",
                {"intent": "set_status", "status": "in ferie", "date_start": da, "date_end": da2})
    elif s == 7:
        o = ro()
        return (f"Sono in malattia da stamattina. Non sarò disponibile fino alle {o}.",
                {"intent": "set_status", "status": "in malattia", "date_start": "oggi", "time": o})
    elif s == 8:
        return (f"Imposta: {status} per tutta la settimana.",
                {"intent": "set_status", "status": status, "date_start": "questa settimana"})
    elif s == 9:
        return (f"Avvisa tutti che sono {status} {d} e {dp}.",
                {"intent": "set_status", "status": status, "date_start": d, "date_end": dp})
    elif s == 10:
        return (f"Sono assente {d}. Blocca l'agenda.",
                {"intent": "set_status", "status": "assente", "date_start": d})
    elif s == 11:
        return (f"Sono in trasferta a {mese}: dal {data_c} in poi.",
                {"intent": "set_status", "status": "in trasferta", "date_start": data_c})
    elif s == 12:
        return (f"Non raggiungermi {dp}: sono {status}.",
                {"intent": "set_status", "status": status, "date_start": dp})
    elif s == 13:
        return (f"Metti fuori ufficio per oggi e domani.",
                {"intent": "set_status", "status": "fuori ufficio", "date_start": "oggi", "date_end": "domani"})
    elif s == 14:
        return (f"Sono non disponibile {d} {rp()}.",
                {"intent": "set_status", "status": "non disponibile", "date_start": d})
    elif s == 15:
        return (f"Blocca la settimana di {dp}: sono in formazione.",
                {"intent": "set_status", "status": "in formazione", "date_start": dp})
    elif s == 16:
        return (f"Non ci sono {d}: sono {status}. Segna tutto come occupato.",
                {"intent": "set_status", "status": status, "date_start": d})
    else:
        return (f"Imposta il mio stato su '{status}' per {dp}.",
                {"intent": "set_status", "status": status, "date_start": dp})

fill_intent(gen_set_status, "set_status")

# ═════════════════════════════════════════════════════════════════════════════
# 21. UNKNOWN
# ═════════════════════════════════════════════════════════════════════════════

UNKNOWN_POOL = [
    "Qual è la capitale della Francia?",
    "Come si fa la pasta al pomodoro?",
    "Quanto fa 15 per 23?",
    "Chi ha scritto la Divina Commedia?",
    "Che tempo farà domani a Milano?",
    "Traduci 'hello' in italiano.",
    "Qual è la montagna più alta del mondo?",
    "Come si chiama il presidente degli Stati Uniti?",
    "Qual è la formula dell'acqua?",
    "Quando è nata la Repubblica Italiana?",
    "Quante regioni ha l'Italia?",
    "Come si dice 'grazie' in giapponese?",
    "Qual è la velocità della luce?",
    "Chi ha dipinto la Gioconda?",
    "Quanto costa un biglietto per Roma?",
    "Qual è la capitale della Germania?",
    "Come funziona un motore a combustione interna?",
    "Qual è il pianeta più grande del sistema solare?",
    "Chi ha inventato il telefono?",
    "Quante ore ci vogliono per volare a New York?",
    "Qual è la distanza tra la Terra e la Luna?",
    "Come si prepara il tiramisù?",
    "Qual è il film più visto di tutti i tempi?",
    "Dove si trova il Colosseo?",
    "Qual è la valuta giapponese?",
    "Chi ha vinto il mondiale del 2006?",
    "Cosa significa RNA?",
    "Come funziona il machine learning?",
    "Qual è la differenza tra virus e batterio?",
    "Quante stelle ha la bandiera europea?",
    "Qual è la popolazione mondiale?",
    "Cosa è il PIL?",
    "Come si calcola la media aritmetica?",
    "Qual è la densità dell'acqua?",
    "Chi ha scritto 'Guerra e Pace'?",
    "Qual è il simbolo chimico dell'oro?",
    "Quanti continenti ci sono?",
    "Qual è la lingua più parlata nel mondo?",
    "Chi ha fondato Apple?",
    "Qual è la distanza da Roma a Napoli in km?",
    "Come si chiama la moneta della Svizzera?",
    "Quanti anni dura un mandato presidenziale in USA?",
    "Qual è la differenza tra HTTP e HTTPS?",
    "Chi ha scritto 'Il Signore degli Anelli'?",
    "Come funziona un frigorifero?",
    "Quante corde ha una chitarra standard?",
    "Qual è il simbolo di pi greco?",
    "Dove si trovano le piramidi di Giza?",
    "Cosa è la fotosintesi?",
    "Qual è il numero di Avogadro?",
    "Chi ha composto la Quinta Sinfonia?",
    "Qual è il paese più grande del mondo per superficie?",
    "Cosa significa CPU?",
    "Quante note musicali ci sono?",
    "Qual è la differenza tra statistica e probabilità?",
    "Come si chiama il re di Spagna?",
    "Qual è la capitale del Brasile?",
    "Chi ha vinto il premio Nobel per la fisica nel 2023?",
    "Cosa è il teorema di Pitagora?",
    "Dimmi una barzelletta.",
    "Qual è il libro più venduto di sempre?",
    "Chi ha inventato internet?",
    "Come funziona la blockchain?",
    "Qual è la temperatura di ebollizione dell'acqua?",
    "Quanti cromosomi ha l'essere umano?",
    "Dove si trova il Monte Everest?",
    "Qual è la densità del ferro?",
    "Chi era Napoleone Bonaparte?",
    "Cosa è l'intelligenza artificiale?",
    "Quanti secondi ci sono in un anno?",
    "Qual è il paese più popolato del mondo?",
    "Come si chiama il presidente della BCE?",
    "Cosa è il metaverso?",
    "Qual è la velocità del suono nell'aria?",
    "Chi ha scritto 'Cent'anni di solitudine'?",
    "Quali sono i pianeti del sistema solare?",
    "Come si calcola l'area di un cerchio?",
    "Cosa è Bitcoin?",
    "Qual è la differenza tra IA debole e IA forte?",
    "Chi era Julius Caesar?",
    "Cosa è la gravità?",
    "Come funziona un pannello solare?",
    "Quante lingue parlano i polglotti mediamente?",
    "Dove si trova il Vaticano?",
    "Cosa è il codice binario?",
    "Qual è la differenza tra introversione ed estroversione?",
    "Chi ha fondato Microsoft?",
    "Qual è la formula della forza in fisica?",
    "Cosa è il quantum computing?",
    "Quanti gradi ha un angolo retto?",
    "Chi era Leonardo da Vinci?",
    "Cosa è il big data?",
    "Qual è il ruolo del DNA?",
    "Come funziona un satellite?",
    "Quante parole ha la lingua italiana circa?",
    "Qual è la differenza tra fusione e fissione nucleare?",
    "Chi è l'autore di '1984'?",
    "Cosa è il protocollo TCP/IP?",
    "Qual è la radice quadrata di 144?",
    "Come si chiama la valuta cinese?",
    "Cosa è un algoritmo?",
    "Qual è la specie animale più veloce?",
    "Chi ha inventato la stampa?",
    "Cosa è il metodo scientifico?",
    "Qual è la differenza tra Nord e Sud Italia?",
    "Come si fa il risotto alla milanese?",
    "Cosa è la coscienza?",
    "Chi ha scritto 'I Promessi Sposi'?",
    "Qual è la lunghezza dell'equatore?",
    "Come funziona la memoria umana?",
    "Quanti milioni di italiani vivono all'estero?",
    "Cosa è la crittografia?",
    "Qual è la differenza tra rete LAN e WAN?",
    "Chi è stato il primo uomo sulla Luna?",
    "Cosa è il paradosso del gatto di Schrödinger?",
    "Quanti litri d'acqua dovrei bere al giorno?",
    "Qual è la differenza tra carboidrati e proteine?",
    "Come si chiama il primo ministro italiano?",
    "Cosa è l'entropia in termodinamica?",
    "Quanti millimetri ci sono in un metro?",
    "Qual è la differenza tra sincrono e asincrono?",
    "Come funziona un motore elettrico?",
    "Chi ha dipinto la Cappella Sistina?",
    "Qual è la differenza tra romanzo e racconto?",
    "Come si coltiva il grano?",
    "Qual è il peso specifico dell'oro?",
    "Chi era Galileo Galilei?",
    "Cosa è la deriva dei continenti?",
    "Quante lingue esistono nel mondo?",
    "Qual è la differenza tra filosofia e scienza?",
    "Come si chiama il fiume più lungo del mondo?",
    "Cosa è il paradosso di Zenone?",
    "Qual è la struttura di un atomo?",
    "Chi ha fondato Roma?",
    "Come funziona un aereo?",
    "Qual è la differenza tra etica e morale?",
    "Quante stagioni ha l'anno?",
    "Cosa è la pressione atmosferica?",
    "Chi era Platone?",
    "Qual è la differenza tra microeconomia e macroeconomia?",
    "Come si chiama la valuta della Svezia?",
    "Cosa è l'indice di massa corporea?",
    "Qual è la distanza tra il Sole e la Terra?",
    "Chi ha scritto 'Ulisse'?",
    "Come funziona il ciclo dell'acqua?",
    "Qual è la differenza tra CPU e GPU?",
    "Quante ossa ha il corpo umano?",
    "Cosa è il sistema immunitario?",
    "Chi era Sigmund Freud?",
    "Qual è la differenza tra intelligenza emotiva e QI?",
    "Come si calcola la velocità media?",
    "Cosa è il teorema di Bayes?",
    "Chi ha vinto la Seconda Guerra Mondiale?",
    "Qual è la differenza tra democrazia e repubblica?",
    "Come si chiama il monte più alto d'Italia?",
    "Cosa è il paradosso di Russell?",
    "Qual è la lunghezza media di una vita umana?",
    "Chi era Charles Darwin?",
    "Come funziona la rete elettrica?",
    "Qual è la differenza tra statica e dinamica?",
    "Quante vertebre ha la colonna vertebrale umana?",
    "Cosa è il fotone?",
    "Chi ha inventato la radio?",
    "Qual è la differenza tra reddito e ricchezza?",
    "Come si chiama il lago più profondo del mondo?",
    "Cosa è la legge di Ohm?",
    "Qual è la differenza tra virus informatico e malware?",
    "Chi era Albert Camus?",
    "Come funziona un termometro?",
    "Qual è la differenza tra prosa e poesia?",
    "Quanti pianeti ha il sistema solare?",
    "Cosa è la biodiversità?",
    "Chi ha vinto il Nobel per la pace nel 2022?",
    "Qual è la differenza tra liberalismo e conservatorismo?",
    "Come si fa la pizza napoletana?",
    "Cosa è il paradosso del sorite?",
    "Qual è la distanza tra Milano e Roma?",
    "Chi era Karl Marx?",
    "Come funziona un generatore elettrico?",
    "Qual è la differenza tra romanticismo e illuminismo?",
    "Quanti milioni di neuroni ha il cervello umano?",
    "Cosa è la meccanica quantistica?",
    "Chi ha inventato il vaccino?",
    "Qual è la differenza tra sincronia e diacronia in linguistica?",
    "Come si chiama il confine tra Italia e Austria?",
    "Cosa è l'ipotesi di Riemann?",
    "Qual è la formula dell'area del triangolo?",
    "Chi era Hannah Arendt?",
    "Come funziona un hard disk?",
    "Qual è la differenza tra astrofisica e cosmologia?",
    "Quante lettere ha l'alfabeto italiano?",
    "Cosa è il principio di indeterminazione di Heisenberg?",
    "Chi ha costruito il Pantheon a Roma?",
    "Qual è la differenza tra analogico e digitale?",
    "Come si prepara la carbonara originale?",
    "Cosa è la legge di gravitazione universale?",
    "Chi era Niccolò Machiavelli?",
    "Come funziona un computer quantistico?",
    "Qual è la differenza tra saturno e giove?",
    "Quanti senatori ha il parlamento italiano?",
]

random.shuffle(UNKNOWN_POOL)
_unk_idx = [0]

def gen_unknown():
    idx = _unk_idx[0] % len(UNKNOWN_POOL)
    _unk_idx[0] += 1
    p = UNKNOWN_POOL[idx]
    return (p, {"intent": "unknown"})

fill_intent(gen_unknown, "unknown")

# ═════════════════════════════════════════════════════════════════════════════
# Shuffle, statistiche, salvataggio
# ═════════════════════════════════════════════════════════════════════════════

random.shuffle(out)

counts = {}
for e in out:
    i = json.loads(e["completion"])["intent"]
    counts[i] = counts.get(i, 0) + 1

print("Esempi per intent:")
for k in sorted(counts):
    bar = "█" * (counts[k] // 10)
    print(f"  {k:25s} {counts[k]:4d}  {bar}")
print(f"\nTOTALE: {len(out)}")

out_path = "data/sft_v2_raw.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"\nSalvato: {out_path}")
