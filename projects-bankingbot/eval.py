import sys
import time
from rag_core import initialize_rag_system
import config

EVAL_SET = [
    {
        "question": "Care este diferența dintre 'Mov eax, c' și 'Mov edx, [c]'?",
        "expected_keywords": ["adresa", "offset", "eax", "valoarea", "conținutul", "[c]", "edx"]
    },
    {
        "question": "De ce instrucțiunile 'mov edx, [CS:c]' și 'mov edx, [DS:c]' au același efect?",
        "expected_keywords": ["modelului", "flat", "memory", "4GiB", "adresa 0", "selectorii", "segment"]
    },
    {
        "question": "Selectorii FS și GS respectă și ei modelul flat?",
        "expected_keywords": ["nu", "speciale", "sistemul de operare", "thread", "FS", "GS", "tls"]
    },
    {
        "question": "Unde încep de obicei segmentele CODE şi DATA în OllyDbg?",
        "expected_keywords": ["00401000", "00402000", "offset", "linker-ului", "code", "data"]
    },
    {
        "question": "Ce se întâmplă dacă definesc 'db 300'?",
        "expected_keywords": ["avertisment", "warning", "depăşeşte", "255", "trunchiată", "byte", "2Ch"]
    },
    {
        "question": "De ce o definire ca 'a15 dd eax' produce o eroare de sintaxă?",
        "expected_keywords": ["eax", "registru", "valoarea", "constantă", "cunoscută", "asamblării", "static"]
    },
    {
        "question": "Care este diferența dintre directivele DB şi RESB?",
        "expected_keywords": ["inițializează", "valori", "db", "rezervă", "spațiu", "neinițializați", "resb"]
    },
    {
        "question": "Care sunt directivele pentru definirea datelor inițializate?",
        "expected_keywords": ["db", "dw", "dd", "dq", "dt", "octet", "cuvinte"]
    },
    {
        "question": "Cum se rezervă un word neinițializat în NASM?",
        "expected_keywords": ["resw", "nu", "dw ?", "neinițializat", "resw 1"]
    },
    {
        "question": "Ce face directiva EQU?",
        "expected_keywords": ["atribuie", "valoare", "numerică", "etichete", "constantă", "asamblării", "equ"]
    },
    {
        "question": "De ce instrucţiunea 'mov [v], 0' produce eroarea 'operation size not specified'?",
        "expected_keywords": ["operation size not specified", "eroare", "dimensiunea", "ambiguitate", "octet", "cuvânt", "dublucuvânt"]
    },
    {
        "question": "Cum se corectează eroarea 'operation size not specified'?",
        "expected_keywords": ["explicită", "dimensiunii", "byte [v]", "word [v]", "dword [v]", "operator de tip"]
    },
    {
        "question": "Ce se întâmplă cu instrucţiunea 'push 15'?",
        "expected_keywords": ["inconsistenţă", "nasm", "implicit", "tratat", "push dword 15"]
    },
    {
        "question": "Care este diferența dintre limbajul maşină şi limbajul de asamblare?",
        "expected_keywords": ["maşină", "biți", "procesorul", "asamblare", "simbolic", "mnemonice", "etichete"]
    },
    {
        "question": "Ce este o 'instrucţiune' și ce este o 'directivă'?",
        "expected_keywords": ["instrucţiune", "procesorul", "execuției", "directivă", "asamblorul", "asamblării", "mov", "db"]
    },
    {
        "question": "Este NASM case-sensitive?",
        "expected_keywords": ["da", "identificatorii", "etichete", "variabile", "nu", "mnemonicele", "regiştrilor"]
    },
    {
        "question": "Ce este un 'contor de locaţii' (location counter)?",
        "expected_keywords": ["număr", "asamblor", "deplasamentul", "curent", "segmentului", "octeți", "$"]
    },
    {
        "question": "Ce reprezintă simbolul '$' în NASM?",
        "expected_keywords": ["simbolului", "special", "$", "contor", "locații", "adresa", "curent"]
    },
    {
        "question": "Care este regula fundamentală pentru a accesa adresa vs. conținutul unei variabile în NASM?",
        "expected_keywords": ["numele", "variabilei", "adresa", "offset", "paranteze", "drepte", "[p]", "conținutul", "valoarea"]
    },
    {
        "question": "Ce face instrucţiunea 'lea eax, [v]'?",
        "expected_keywords": ["lea", "load effective address", "încarcă", "adresa", "offset-ul", "variabilei v", "eax"]
    },
    {
        "question": "Ce regulă specială se aplică numerelor hexazecimale care folosesc sufixul H?",
        "expected_keywords": ["obligațiu", "cifră", "0-9", "0abch", "valid", "abch", "simbol"]
    },
    {
        "question": "Ce registru de segment este folosit implicit când se folosește EBP ca bază?",
        "expected_keywords": ["ss", "stack segment", "implicit", "ebp", "esp", "bază"]
    },
    {
        "question": "Cine decide adresa de început a segmentului de cod (de exemplu, 00401000)?",
        "expected_keywords": ["linkeditorul", "the linker", "adresă", "început", "segment", "cod"]
    },
    {
        "question": "De ce secțiunile (cod, date) încep la adrese care sunt multipli de 0x1000?",
        "expected_keywords": ["aliniată", "pagină", "memorie", "4KiB", "0x1000", "drepturi", "acces", "sistemului de operare"]
    },
    {
        "question": "Ce reprezintă simbolurile speciale '$' și '$$' în NASM?",
        "expected_keywords": ["$", "adresa", "liniei curente", "$$", "început", "secțiunii", "segmentului"]
    }
]


def run_benchmark():
    print(f"--- RULARE EVALUARE ---")
    print(f"Mod: {'HYBRID (Optimized)' if config.USE_HYBRID_SEARCH else 'VECTOR (Baseline)'}")
    print(f"Model LLM: {config.LLM_MODEL}")

    start_time = time.time()
    rag_chain = initialize_rag_system("ASC")  # Testam pe ASC

    if not rag_chain:
        return

    correct = 0
    total = len(EVAL_SET)

    for item in EVAL_SET:
        q = item["question"]
        kws = item["expected_keywords"]

        try:
            resp = rag_chain.invoke(q)
            resp_lower = resp.lower()

            # Verificare simpla keywords
            found = sum(1 for kw in kws if kw.lower() in resp_lower)
            if found >= (len(kws) + 1) // 2:  # Prag 50% keywords
                correct += 1

        except Exception as e:
            print(f"Eroare la intrebarea '{q}': {e}")

    end_time = time.time()
    duration = end_time - start_time

    accuracy = (correct / total) * 100
    avg_time = duration / total

    print("\n" + "=" * 30)
    print(f"REZULTATE PENTRU RAPORT ({'OPTIMIZAT' if config.USE_HYBRID_SEARCH else 'BASELINE'}):")
    print(f"1. Acuratete: {accuracy:.2f}%")
    print(f"2. Timp total: {duration:.2f} sec")
    print(f"3. Viteza medie: {avg_time:.2f} sec/intrebare")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    run_benchmark()