import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_ROOT_PATH = os.path.join(BASE_DIR, "DOCS")
DOCS2_PATH = os.path.join(BASE_DIR, "DOCS2")

DB_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

# --- HIPERPARAMETRI OPTIMIZABILI ---
RETRIEVER_K = 10                  # Cati vecini cautam initial
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Setari pentru Hybrid Search (Ensembling)
USE_HYBRID_SEARCH = True          # <--- Seteaza False pentru varianta "Basic", True pentru "Optimized"
HYBRID_WEIGHTS = [0.5, 0.5]       # 50% Vectorial, 50% Cuvinte cheie

QA_SPLIT_REGEX = r'(?=\nQ:)'