"""
Genera esempi SFT sintetici per il dataset calendario/agenda.
Output: data/sft_synth_raw.jsonl  (~1060 nuovi esempi, ~175-180 per intent)

Fix: tutti i valori casuali vengono estratti PRIMA di costruire prompt e completion,
garantendo coerenza tra testo e JSON.
"""
import json, random

random.seed(42)

NOMI = ["Marco","Sara","Luca","Giulia","Andrea","Valentina","Mattia","Kevin",
        "Francesca","Roberto","Elena","Davide","Chiara","Simone","Alessia",
        "Paolo","Monica","Fabio","Irene","Giorgio","Marta","Stefano","Laura",
        "Riccardo","Anna","Pietro","Sofia","Filippo","Claudia","Nicola"]

COGNOMI = ["Rossi","Bianchi","Ferrari","Esposito","Ricci","Conti","Moro","Neri",
           "Russo","Verdi","Greco","Mancini","Romano","Colombo","Gallo"]

RUOLI = ["cliente","fornitore","team","collega","responsabile","manager",
         "direttore","avvocato","commercialista","medico","notaio",
         "tecnico","assistente","coordinatore","project manager","consulente",
         "ingegnere","responsabile HR","responsabile vendite","referente"]

GIORNI_REL = ["domani","dopodomani","oggi","lunedì","martedì","mercoledì",
              "giovedì","venerdì","sabato","domenica"]

GIORNI_PROSS = ["lunedì prossimo","martedì prossimo","mercoledì prossimo",
                "giovedì prossimo","venerdì prossimo","la settimana prossima",
                "la prossima settimana","il mese prossimo","dopodomani"]

ORE = ["08:00","08:30","09:00","09:30","10:00","10:30","11:00","11:30",
       "12:00","13:00","14:00","14:30","15:00","15:30","16:00","16:30",
       "17:00","17:30","18:00","19:00","20:00"]

PERIODI = ["mattina","pomeriggio","sera","tarda mattinata","primo pomeriggio",
           "tardo pomeriggio"]

DATE_ABS = ["3 novembre","5 novembre","10 novembre","15 novembre","20 novembre",
            "25 novembre","1 dicembre","8 dicembre","15 dicembre","20 ottobre",
            "25 ottobre","28 ottobre","31 ottobre","2 dicembre","12 novembre",
            "7 novembre","18 novembre","22 ottobre","4 dicembre","9 dicembre"]

DURATE = ["30 minuti","1 ora","2 ore","3 ore","mezz'ora","un'ora e mezza"]

EVENTI = ["riunione","meeting","call","videochiamata","appuntamento","colloquio",
          "pranzo","cena","presentazione","revisione","briefing","workshop",
          "sessione","incontro","consulenza","aggiornamento","demo","formazione"]

TITOLI_EVENTO = ["revisione budget","review codice","allineamento settimanale",
                 "presentazione progetto","brainstorming","onboarding",
                 "check-in mensile","retrospettiva","planning sprint",
                 "riunione di team","aggiornamento stato","demo prodotto",
                 "colloquio candidato","sessione formativa","call cliente",
                 "kick-off progetto","stand-up mattutino","analisi risultati"]

DOMINI = ["gmail.com","outlook.com","azienda.it","lavoro.com","studio.it",
          "email.it","pec.it","impresa.it","consulting.it"]

PREFISSI_CELL = ["333","347","328","339","340","346","380","393","348","366"]
PREFISSI_FISSO = ["02","06","011","081","055","051","049","045","010","090"]

NOTE_POOL = [
    "chiamare {nome}","rispondere alla mail di {nome}","mandare il report",
    "pagare la bolletta","rinnovare l'abbonamento","prenotare il volo",
    "comprare il regalo per {nome}","fare il backup","mandare la fattura",
    "preparare la presentazione","firmare il contratto","leggere il verbale",
    "confermare la partecipazione","aggiornare la documentazione",
    "verificare lo stato del progetto","chiamare il {ruolo}","rispondere a {nome}",
    "mandare i documenti","controllare le email","fare la spesa",
    "portare i materiali","revisionare il report","contattare il {ruolo}",
    "inviare la proposta","preparare i dati","controllare i numeri",
    "scrivere le note","organizzare i file","verificare la scadenza",
    "richiamare {nome}","aggiornare il team","preparare l'ordine del giorno",
    "inviare il resoconto","prenotare la sala","confermare l'ordine",
    "leggere il contratto","approvare le spese","aggiornare il CRM",
]

out = []

def rn():  return random.choice(NOMI)
def rcog(): return random.choice(COGNOMI)
def rd():  return random.choice(GIORNI_REL)
def rdp(): return random.choice(GIORNI_PROSS)
def rda(): return random.choice(DATE_ABS)
def ro():  return random.choice(ORE)
def rp():  return random.choice(PERIODI)
def rev(): return random.choice(EVENTI)
def rte(): return random.choice(TITOLI_EVENTO)
def rdu(): return random.choice(DURATE)
def rru(): return random.choice(RUOLI)

def rphone(fisso=False):
    pre = random.choice(PREFISSI_FISSO if fisso else PREFISSI_CELL)
    num = random.randint(1000000, 9999999)
    return f"{pre}-{num}"

def remail(nome):
    suf = random.choice(["","m.","r.","a.","g.","s."])
    return f"{suf}{nome.lower()}@{random.choice(DOMINI)}"

def rnota(nome=None, ruolo=None):
    n = random.choice(NOTE_POOL)
    nome = nome or rn()
    ruolo = ruolo or rru()
    return n.format(nome=nome, ruolo=ruolo)

def add(prompt, completion):
    # Rimuovi chiavi con valore None
    c = {k: v for k, v in completion.items() if v is not None}
    out.append({"prompt": prompt.strip(), "completion": json.dumps(c, ensure_ascii=False)})

def count_intent(intent):
    return sum(1 for e in out if json.loads(e["completion"]).get("intent") == intent)

# ─────────────────────────────────────────────────────────────────────────────
# CREATE_EVENT (175)
# ─────────────────────────────────────────────────────────────────────────────

def gen_create_event():
    style = random.randint(0, 24)

    if style == 0:
        n, d, o, ev = rn(), rd(), ro(), rev()
        return (f"Fissa {ev} con {n} {d} alle {o}.",
                {"intent":"create_event","title":ev,"contact":n,"date":d,"time":o})
    elif style == 1:
        d, p, ev = rd(), rp(), rev()
        return (f"Prenota una {ev} per {d} {p}.",
                {"intent":"create_event","title":ev,"date":d,"time":p})
    elif style == 2:
        te, da, o = rte(), rda(), ro()
        return (f"Crea un evento: {te} il {da} alle {o}.",
                {"intent":"create_event","title":te,"date":da,"time":o})
    elif style == 3:
        ev, ru, dp = rev(), rru(), rdp()
        return (f"Metti in agenda {ev} con il {ru} per {dp}.",
                {"intent":"create_event","title":ev,"contact":ru,"date":dp})
    elif style == 4:
        n1, n2, d, o, ev = rn(), rn(), rd(), ro(), rev()
        while n2 == n1: n2 = rn()
        return (f"Ho bisogno di {ev} con {n1} e {n2} {d} alle {o}.",
                {"intent":"create_event","title":ev,"contact":[n1,n2],"date":d,"time":o})
    elif style == 5:
        te, da, o = rte(), rda(), ro()
        return (f"Aggiungi al calendario: {te}, {da}, ore {o}.",
                {"intent":"create_event","title":te,"date":da,"time":o})
    elif style == 6:
        ev, n, d, du = rev(), rn(), rd(), rdu()
        return (f"Organizza un {ev} con {n} {d} per {du}.",
                {"intent":"create_event","title":ev,"contact":n,"date":d,"duration":du})
    elif style == 7:
        ev, d, o, du = rev(), rd(), ro(), rdu()
        return (f"Segna una {ev} di {du} {d} alle {o}.",
                {"intent":"create_event","title":ev,"date":d,"time":o,"duration":du})
    elif style == 8:
        n, da, o = rn(), rda(), ro()
        return (f"Crea appuntamento con {n}: {da} ore {o}.",
                {"intent":"create_event","contact":n,"date":da,"time":o})
    elif style == 9:
        ev, dp, o = rev(), rdp(), ro()
        return (f"Inserisci in calendario la {ev} con il team {dp} alle {o}.",
                {"intent":"create_event","title":ev,"contact":"team","date":dp,"time":o})
    elif style == 10:
        te, o = rte(), ro()
        return (f"Aggiungi evento: {te} ogni lunedì alle {o}.",
                {"intent":"create_event","title":te,"date":"ogni lunedì","time":o,"recurrence":"settimanale"})
    elif style == 11:
        n, d, o = rn(), rd(), ro()
        return (f"Devo incontrare {n} {d}. Mettilo in agenda alle {o}.",
                {"intent":"create_event","contact":n,"date":d,"time":o})
    elif style == 12:
        ev, o = rev(), ro()
        return (f"Crea una {ev} ricorrente ogni venerdì alle {o}.",
                {"intent":"create_event","title":ev,"date":"ogni venerdì","time":o,"recurrence":"settimanale"})
    elif style == 13:
        ev, n, da, p = rev(), rn(), rda(), rp()
        return (f"Pianifica un {ev} con {n} il {da} di {p}.",
                {"intent":"create_event","title":ev,"contact":n,"date":da,"time":p})
    elif style == 14:
        ru, dp, o = rru(), rdp(), ro()
        return (f"Fissa un appuntamento con il {ru} {dp} alle {o}.",
                {"intent":"create_event","title":f"appuntamento {ru}","contact":ru,"date":dp,"time":o})
    elif style == 15:
        te, n, d, o, du = rte(), rn(), rd(), ro(), rdu()
        return (f"Metti in calendario: {te} con {n}, {d}, dalle {o} per {du}.",
                {"intent":"create_event","title":te,"contact":n,"date":d,"time":o,"duration":du})
    elif style == 16:
        ev, n, d = rev(), rn(), rd()
        return (f"C'è un {ev} con {n} {d} mattina. Aggiungilo.",
                {"intent":"create_event","title":ev,"contact":n,"date":d,"time":"mattina"})
    elif style == 17:
        da, o, te = rda(), ro(), rte()
        return (f"Crea evento il {da} alle {o}: {te}.",
                {"intent":"create_event","title":te,"date":da,"time":o})
    elif style == 18:
        n1, n2, n3, ev, dp = rn(), rn(), rn(), rev(), rdp()
        while n2 == n1: n2 = rn()
        while n3 in (n1, n2): n3 = rn()
        return (f"Aggiungi: {ev} con {n1}, {n2} e {n3}, {dp}.",
                {"intent":"create_event","title":ev,"contact":[n1,n2,n3],"date":dp})
    elif style == 19:
        ev, d = rev(), rd()
        return (f"Inserisci in agenda il {ev} mensile il primo {d} del mese.",
                {"intent":"create_event","title":ev,"date":f"primo {d} del mese","recurrence":"mensile"})
    elif style == 20:
        ev, n, d = rev(), rn(), rd()
        return (f"Ho un {ev} con {n} {d} sera. Salvalo.",
                {"intent":"create_event","title":ev,"contact":n,"date":d,"time":"sera"})
    elif style == 21:
        te, da, o = rte(), rda(), ro()
        return (f"Pianifica {te} per {da} alle {o}.",
                {"intent":"create_event","title":te,"date":da,"time":o})
    elif style == 22:
        te, d, p = rte(), rd(), rp()
        return (f"Crea un blocco in calendario per {d} {p}: {te}.",
                {"intent":"create_event","title":te,"date":d,"time":p})
    elif style == 23:
        n, da, p = rn(), rda(), rp()
        return (f"Aggiungi un appuntamento con {n} il {da} di {p}.",
                {"intent":"create_event","contact":n,"date":da,"time":p})
    else:
        ev, ru, d, o, du = rev(), rru(), rd(), ro(), rdu()
        return (f"Metti una {ev} con il {ru} {d} alle {o}. Durata {du}.",
                {"intent":"create_event","title":ev,"contact":ru,"date":d,"time":o,"duration":du})

used = set()
while count_intent("create_event") < 175:
    p, c = gen_create_event()
    k = p[:50]
    if k not in used:
        used.add(k)
        add(p, c)

# ─────────────────────────────────────────────────────────────────────────────
# RESCHEDULE_EVENT (175)
# ─────────────────────────────────────────────────────────────────────────────

SPOSTA = ["Sposta","Posticipa","Anticipa","Aggiorna","Rinvia","Ripianifica"]

def gen_reschedule():
    style = random.randint(0, 19)

    if style == 0:
        n, d_old, dp = rn(), rd(), rdp()
        return (f"Sposta la riunione con {n} da {d_old} a {dp}.",
                {"intent":"reschedule_event","contact":n,"event_ref":f"riunione {d_old}","new_date":dp})
    elif style == 1:
        ev, d, dp, o = rev(), rd(), rdp(), ro()
        return (f"Devo posticipare il {ev} di {d} a {dp} alle {o}.",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":dp,"new_time":o})
    elif style == 2:
        n, dp, o = rn(), rdp(), ro()
        return (f"La riunione con {n} è stata anticipata a {dp} alle {o}.",
                {"intent":"reschedule_event","contact":n,"event_ref":"riunione","new_date":dp,"new_time":o})
    elif style == 3:
        ev, d, dp, o = rev(), rd(), rdp(), ro()
        return (f"Sposta la {ev} con il cliente di {d} a {dp} alle {o}.",
                {"intent":"reschedule_event","event_ref":f"{ev} cliente {d}","new_date":dp,"new_time":o})
    elif style == 4:
        ev, d, dp = rev(), rd(), rdp()
        return (f"Il {ev} di {d} va spostato a {dp}.",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":dp})
    elif style == 5:
        ev, d, dp, o = rev(), rd(), rdp(), ro()
        return (f"Posticipa l'{ev} di {d} a {dp} alle {o}.",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":dp,"new_time":o})
    elif style == 6:
        ev, d, dp = rev(), rd(), rdp()
        return (f"Dobbiamo spostare la {ev} di {d}, possiamo fare {dp}?",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":dp})
    elif style == 7:
        te, da, dp = rte(), rda(), rda()
        while dp == da: dp = rda()
        return (f"La {te} è slittata dal {da} al {dp}.",
                {"intent":"reschedule_event","event_ref":te,"new_date":dp})
    elif style == 8:
        n, dp, o = rn(), rdp(), ro()
        return (f"Aggiorna il meeting con {n}: non più domani ma {dp} alle {o}.",
                {"intent":"reschedule_event","contact":n,"event_ref":"meeting","new_date":dp,"new_time":o})
    elif style == 9:
        ev, o1, o2, d = rev(), ro(), ro(), rd()
        while o2 == o1: o2 = ro()
        return (f"Ho bisogno di spostare la {ev} delle {o1} di {d} alle {o2}.",
                {"intent":"reschedule_event","event_ref":f"{ev} {o1} {d}","new_date":d,"new_time":o2})
    elif style == 10:
        ev, d, dp = rev(), rd(), rdp()
        return (f"Sposta il {ev} con il team da {d} a {dp}.",
                {"intent":"reschedule_event","contact":"team","event_ref":f"{ev} {d}","new_date":dp})
    elif style == 11:
        ev, d, dp, o = rev(), rd(), rdp(), ro()
        return (f"La {ev} con il responsabile di {d} va anticipata a {dp} alle {o}.",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":dp,"new_time":o})
    elif style == 12:
        te, da, dp = rte(), rda(), rda()
        while dp == da: dp = rda()
        return (f"Aggiorna il calendario: {te} non è più il {da} ma il {dp}.",
                {"intent":"reschedule_event","event_ref":te,"new_date":dp})
    elif style == 13:
        n, ev, dp = rn(), rev(), rdp()
        return (f"Devo rimandare il {ev} con {n}. Può fare {dp}?",
                {"intent":"reschedule_event","contact":n,"event_ref":ev,"new_date":dp})
    elif style == 14:
        ev, d, dp, o = rev(), rd(), rdp(), ro()
        return (f"Sposta la {ev} delle {o} di {d} a {dp} stessa ora.",
                {"intent":"reschedule_event","event_ref":f"{ev} {o} {d}","new_date":dp,"new_time":o})
    elif style == 15:
        ev, da, dp = rev(), rda(), rda()
        while dp == da: dp = rda()
        return (f"Rinvia l'{ev} del {da} di una settimana.",
                {"intent":"reschedule_event","event_ref":f"{ev} {da}","new_date":f"una settimana dopo {da}"})
    elif style == 16:
        ev, d, dp, o = rev(), rd(), rdp(), ro()
        return (f"Aggiorna: {ev} di {d} spostato a {dp} alle {o}.",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":dp,"new_time":o})
    elif style == 17:
        n, d, dp = rn(), rd(), rdp()
        return (f"Ho dovuto spostare il pranzo con {n} da {d} a {dp}.",
                {"intent":"reschedule_event","contact":n,"event_ref":f"pranzo {d}","new_date":dp})
    elif style == 18:
        ev, d, o = rev(), rd(), ro()
        return (f"La {ev} di {d} è stata spostata alle {o} dello stesso giorno.",
                {"intent":"reschedule_event","event_ref":f"{ev} {d}","new_date":d,"new_time":o})
    else:
        n1, n2, d, dp = rn(), rn(), rd(), rdp()
        while n2 == n1: n2 = rn()
        return (f"Sposta il meeting con {n1} e {n2} da {d} a {dp}.",
                {"intent":"reschedule_event","contact":[n1,n2],"event_ref":f"meeting {d}","new_date":dp})

used = set()
while count_intent("reschedule_event") < 175:
    p, c = gen_reschedule()
    k = p[:50]
    if k not in used:
        used.add(k)
        add(p, c)

# ─────────────────────────────────────────────────────────────────────────────
# DELETE_EVENT (180)
# ─────────────────────────────────────────────────────────────────────────────

DEL = ["Cancella","Elimina","Rimuovi","Togli","Disdici","Annulla"]

def gen_delete():
    style = random.randint(0, 17)
    v = random.choice(DEL)

    if style == 0:
        n, d, ev = rn(), rd(), rev()
        return (f"{v} la {ev} con {n} di {d}.",
                {"intent":"delete_event","contact":n,"event_ref":f"{ev} {d}","date":d})
    elif style == 1:
        ev, da = rev(), rda()
        return (f"{v} il {ev} del {da}.",
                {"intent":"delete_event","event_ref":ev,"date":da})
    elif style == 2:
        d, p = rd(), rp()
        return (f"{v} tutto quello che ho {d} {p}.",
                {"intent":"delete_event","date":d,"time":p,"scope":"all"})
    elif style == 3:
        ev, d, o = rev(), rd(), ro()
        return (f"{v} la {ev} delle {o} di {d}.",
                {"intent":"delete_event","event_ref":ev,"date":d,"time":o})
    elif style == 4:
        n, ev = rn(), rev()
        return (f"Non ho più bisogno della {ev} con {n}. {v}la.",
                {"intent":"delete_event","contact":n,"event_ref":ev})
    elif style == 5:
        n, ev = rn(), rev()
        return (f"{v} tutti gli appuntamenti con {n} questa settimana.",
                {"intent":"delete_event","contact":n,"date":"questa settimana","scope":"all"})
    elif style == 6:
        ru, ev = rru(), rev()
        return (f"La {ev} con il {ru} è annullata. Toglila dal calendario.",
                {"intent":"delete_event","contact":ru,"event_ref":ev})
    elif style == 7:
        te, da = rte(), rda()
        return (f"{v} il {te} del {da}.",
                {"intent":"delete_event","event_ref":te,"date":da})
    elif style == 8:
        ev, d = rev(), rd()
        return (f"{v} tutte le {ev} ricorrenti del {d}.",
                {"intent":"delete_event","event_ref":ev,"recurrence":"settimanale","scope":"all"})
    elif style == 9:
        n, ev, d = rn(), rev(), rd()
        return (f"Hai messo in agenda {ev} con {n} {d}, ma non serve più.",
                {"intent":"delete_event","contact":n,"event_ref":ev,"date":d})
    elif style == 10:
        n1, n2, ev, dp = rn(), rn(), rev(), rdp()
        while n2 == n1: n2 = rn()
        return (f"{v} la {ev} con {n1} e {n2} di {dp}.",
                {"intent":"delete_event","contact":[n1,n2],"event_ref":ev,"date":dp})
    elif style == 11:
        ev, d, o = rev(), rd(), ro()
        return (f"{v} la {ev} alle {o} di {d}.",
                {"intent":"delete_event","event_ref":ev,"date":d,"time":o})
    elif style == 12:
        ev, da = rev(), rda()
        return (f"Il {ev} del {da} è saltato. Toglilo.",
                {"intent":"delete_event","event_ref":ev,"date":da})
    elif style == 13:
        dp = rdp()
        return (f"{v} tutti gli impegni di {dp}.",
                {"intent":"delete_event","date":dp,"scope":"all"})
    elif style == 14:
        n, ev, d = rn(), rev(), rd()
        return (f"Non viene più {n}, {v.lower()} la {ev} di {d}.",
                {"intent":"delete_event","contact":n,"event_ref":ev,"date":d})
    elif style == 15:
        d, p = rd(), rp()
        return (f"{v} il blocco del {p} di {d}.",
                {"intent":"delete_event","date":d,"time":p})
    elif style == 16:
        d = rd()
        return (f"È saltato tutto per {d}. Pulisci il calendario.",
                {"intent":"delete_event","date":d,"scope":"all"})
    else:
        ev, d = rev(), rd()
        return (f"{v} la serie ricorrente di {ev} ogni {d}.",
                {"intent":"delete_event","event_ref":ev,"recurrence":"settimanale","scope":"all"})

used = set()
while count_intent("delete_event") < 180:
    p, c = gen_delete()
    k = p[:50]
    if k not in used:
        used.add(k)
        add(p, c)

# ─────────────────────────────────────────────────────────────────────────────
# CREATE_REMINDER (175)
# ─────────────────────────────────────────────────────────────────────────────

REM = ["Ricordami","Metti un reminder","Aggiungi un promemoria",
       "Metti un promemoria","Aggiungi un avviso","Avvisami"]

def gen_reminder():
    style = random.randint(0, 15)
    v = random.choice(REM)
    n = rn()
    ru = rru()
    nota = rnota(n, ru)

    if style == 0:
        d, o = rd(), ro()
        return (f"{v} di {nota} {d} alle {o}.",
                {"intent":"create_reminder","note":nota,"date":d,"time":o})
    elif style == 1:
        d = rd()
        return (f"{v} di {nota} entro {d}.",
                {"intent":"create_reminder","note":nota,"deadline":d})
    elif style == 2:
        dp = rdp()
        return (f"{v} di {nota} {dp}.",
                {"intent":"create_reminder","note":nota,"date":dp})
    elif style == 3:
        da, o = rda(), ro()
        return (f"Imposta un reminder per il {da} alle {o}: {nota}.",
                {"intent":"create_reminder","note":nota,"date":da,"time":o})
    elif style == 4:
        da = rda()
        return (f"{v} di {nota} il {da}.",
                {"intent":"create_reminder","note":nota,"date":da})
    elif style == 5:
        d, o = rd(), ro()
        return (f"Metti un avviso ogni {d} alle {o}: {nota}.",
                {"intent":"create_reminder","note":nota,"date":f"ogni {d}","time":o,"recurrence":"settimanale"})
    elif style == 6:
        ev, d = rev(), rd()
        return (f"{v} di {nota} prima della {ev} di {d}.",
                {"intent":"create_reminder","note":nota,"date":d,"context":f"prima della {ev}"})
    elif style == 7:
        da = rda()
        return (f"Aggiungi promemoria: {nota}, scadenza {da}.",
                {"intent":"create_reminder","note":nota,"deadline":da})
    elif style == 8:
        return (f"{v} di {nota} domani mattina appena arrivo.",
                {"intent":"create_reminder","note":nota,"date":"domani","time":"mattina"})
    elif style == 9:
        dp = rdp()
        return (f"Non voglio dimenticare di {nota}. Metti un reminder per {dp}.",
                {"intent":"create_reminder","note":nota,"date":dp})
    elif style == 10:
        d, o = rd(), ro()
        return (f"Avvisami {d} alle {o} di {nota}.",
                {"intent":"create_reminder","note":nota,"date":d,"time":o})
    elif style == 11:
        return (f"Crea promemoria mensile: {nota}, il primo del mese.",
                {"intent":"create_reminder","note":nota,"date":"primo del mese","recurrence":"mensile"})
    elif style == 12:
        da = rda()
        return (f"Devo {nota} entro il {da}. Ricordamelo.",
                {"intent":"create_reminder","note":nota,"deadline":da})
    elif style == 13:
        d, p = rd(), rp()
        return (f"Metti avviso per {d} {p}: {nota}.",
                {"intent":"create_reminder","note":nota,"date":d,"time":p})
    elif style == 14:
        o = ro()
        return (f"Reminder: {nota}, oggi entro le {o}.",
                {"intent":"create_reminder","note":nota,"deadline":"oggi","time":o})
    else:
        ev, d = rev(), rd()
        return (f"{v} di {nota} subito dopo la {ev} di {d}.",
                {"intent":"create_reminder","note":nota,"date":d,"context":f"dopo la {ev}"})

used = set()
while count_intent("create_reminder") < 175:
    p, c = gen_reminder()
    k = p[:50]
    if k not in used:
        used.add(k)
        add(p, c)

# ─────────────────────────────────────────────────────────────────────────────
# QUERY_SCHEDULE (175)
# ─────────────────────────────────────────────────────────────────────────────

def gen_query():
    style = random.randint(0, 22)

    if style == 0:
        d = rd()
        return (f"Cosa ho in agenda {d}?",
                {"intent":"query_schedule","date":d})
    elif style == 1:
        d, o = rd(), ro()
        return (f"Sono libero {d} alle {o}?",
                {"intent":"query_schedule","date":d,"time":o,"query_type":"availability"})
    elif style == 2:
        n, d = rn(), rd()
        return (f"Ho qualcosa con {n} {d}?",
                {"intent":"query_schedule","contact":n,"date":d})
    elif style == 3:
        dp = rdp()
        return (f"Mostrami tutti gli eventi di {dp}.",
                {"intent":"query_schedule","date":dp})
    elif style == 4:
        ev, d = rev(), rd()
        return (f"Quante {ev} ho {d}?",
                {"intent":"query_schedule","date":d,"query_type":"count"})
    elif style == 5:
        n, ev = rn(), rev()
        return (f"A che ora è la {ev} con {n}?",
                {"intent":"query_schedule","contact":n,"query_type":"time_of_event"})
    elif style == 6:
        ru = rru()
        return (f"Quando è il prossimo appuntamento con il {ru}?",
                {"intent":"query_schedule","contact":ru,"query_type":"next_occurrence"})
    elif style == 7:
        d, p = rd(), rp()
        return (f"Ho slot liberi {d} {p}?",
                {"intent":"query_schedule","date":d,"time":p,"query_type":"availability"})
    elif style == 8:
        dp, p = rdp(), rp()
        return (f"Cosa ho in programma {dp} nel {p}?",
                {"intent":"query_schedule","date":dp,"time":p})
    elif style == 9:
        d, o = rd(), ro()
        return (f"Ci sono conflitti {d} alle {o}?",
                {"intent":"query_schedule","date":d,"time":o,"query_type":"conflicts"})
    elif style == 10:
        return (f"Dammi un'overview della settimana.",
                {"intent":"query_schedule","date":"questa settimana","query_type":"overview"})
    elif style == 11:
        n = rn()
        return (f"Quali impegni ho con {n} nei prossimi giorni?",
                {"intent":"query_schedule","contact":n,"query_type":"upcoming"})
    elif style == 12:
        da = rda()
        return (f"Cosa ho fissato il {da}?",
                {"intent":"query_schedule","date":da})
    elif style == 13:
        ru = rru()
        return (f"Quanti appuntamenti ho questa settimana con il {ru}?",
                {"intent":"query_schedule","contact":ru,"date":"questa settimana","query_type":"count"})
    elif style == 14:
        d, p = rd(), rp()
        return (f"Sono impegnato {d} {p}?",
                {"intent":"query_schedule","date":d,"time":p,"query_type":"availability"})
    elif style == 15:
        n = rn()
        return (f"Quando rivedo {n} la prossima volta?",
                {"intent":"query_schedule","contact":n,"query_type":"next_occurrence"})
    elif style == 16:
        o1, o2, d = ro(), ro(), rd()
        while o2 == o1: o2 = ro()
        return (f"Cosa ho tra le {o1} e le {o2} di {d}?",
                {"intent":"query_schedule","date":d,"time_range":f"{o1}-{o2}"})
    elif style == 17:
        dp = rdp()
        return (f"Mostrami il calendario di {dp}.",
                {"intent":"query_schedule","date":dp})
    elif style == 18:
        da, o = rda(), ro()
        return (f"Ho già qualcosa in agenda il {da} alle {o}?",
                {"intent":"query_schedule","date":da,"time":o,"query_type":"availability"})
    elif style == 19:
        d = rd()
        return (f"Qual è il mio primo impegno di {d}?",
                {"intent":"query_schedule","date":d,"query_type":"first_event"})
    elif style == 20:
        d = rd()
        return (f"Riepilogami la giornata di {d}.",
                {"intent":"query_schedule","date":d,"query_type":"overview"})
    elif style == 21:
        n = rn()
        return (f"Ho riunioni con {n} questo mese?",
                {"intent":"query_schedule","contact":n,"date":"questo mese"})
    else:
        d, o = rd(), ro()
        return (f"Cosa ho alle {o} di {d}?",
                {"intent":"query_schedule","date":d,"time":o})

used = set()
while count_intent("query_schedule") < 175:
    p, c = gen_query()
    k = p[:50]
    if k not in used:
        used.add(k)
        add(p, c)

# ─────────────────────────────────────────────────────────────────────────────
# ADD_CONTACT (180)
# ─────────────────────────────────────────────────────────────────────────────

def gen_contact():
    style = random.randint(0, 14)
    n = rn()
    cog = rcog()
    nome_completo = f"{n} {cog}"
    phone = rphone()
    phone_fisso = rphone(fisso=True)
    email = remail(n)
    ru = rru()

    if style == 0:
        return (f"Aggiungi {nome_completo} con numero {phone}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone})
    elif style == 1:
        return (f"Salva il contatto di {nome_completo}: {email}.",
                {"intent":"add_contact","name":nome_completo,"email":email})
    elif style == 2:
        return (f"Nuovo contatto: {nome_completo}, {ru}, {phone}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"role":ru})
    elif style == 3:
        return (f"Aggiungi alla rubrica {nome_completo}: {email}, {phone}.",
                {"intent":"add_contact","name":nome_completo,"email":email,"phone":phone})
    elif style == 4:
        return (f"Crea contatto per il {ru}: {nome_completo}, {phone}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"role":ru})
    elif style == 5:
        return (f"Salva questo numero come {nome_completo}: {phone}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone})
    elif style == 6:
        return (f"Aggiungi il {ru} {nome_completo}: {email}.",
                {"intent":"add_contact","name":nome_completo,"email":email,"role":ru})
    elif style == 7:
        return (f"Aggiungi contatto: {nome_completo}, ruolo {ru}, cell. {phone}, email {email}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"email":email,"role":ru})
    elif style == 8:
        return (f"Salva il recapito di {nome_completo}: {phone}. È il nostro {ru}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"role":ru})
    elif style == 9:
        return (f"Aggiungi {nome_completo} come contatto professionale: {email}.",
                {"intent":"add_contact","name":nome_completo,"email":email,"role":ru})
    elif style == 10:
        return (f"Inserisci nella rubrica: {nome_completo} ({ru}), {phone}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"role":ru})
    elif style == 11:
        return (f"Nuovo fornitore: {nome_completo}, {phone}, {email}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"email":email,"role":"fornitore"})
    elif style == 12:
        return (f"Salva il numero fisso di {nome_completo}: {phone_fisso}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone_fisso})
    elif style == 13:
        return (f"Crea contatto per il nuovo cliente: {nome_completo}, {email}.",
                {"intent":"add_contact","name":nome_completo,"email":email,"role":"cliente"})
    else:
        return (f"Aggiungi il {ru} {nome_completo}: {phone}, {email}.",
                {"intent":"add_contact","name":nome_completo,"phone":phone,"email":email,"role":ru})

used = set()
while count_intent("add_contact") < 180:
    p, c = gen_contact()
    k = p[:50]
    if k not in used:
        used.add(k)
        add(p, c)

# ─────────────────────────────────────────────────────────────────────────────
# Shuffle e salva
# ─────────────────────────────────────────────────────────────────────────────
random.shuffle(out)

counts = {}
for e in out:
    i = json.loads(e["completion"])["intent"]
    counts[i] = counts.get(i, 0) + 1

print("Esempi generati per intent:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
print(f"  TOTALE: {len(out)}")

with open("data/sft_synth_raw.jsonl", "w", encoding="utf-8") as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"\nSalvato: data/sft_synth_raw.jsonl")
