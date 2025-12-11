import os
import shutil
import re
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def process_subject_folder(subject_name, folder_path, is_qa_mode=False):
    """
    Incarca PDF-urile dintr-un folder specific unei materii.

    Args:
        subject_name (str): Numele materiei (ex: ASC, LC).
        folder_path (str): Calea catre folder.
        is_qa_mode (bool): Daca True, foloseste split pe baza de Regex (pentru DOCS2).
                           Daca False, foloseste split standard (pentru DOCS).
    """
    type_label = "Q&A" if is_qa_mode else "CURS"
    print(f"--- Procesare {type_label}: {subject_name} din {folder_path} ---")

    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=True
    )

    try:
        documents = loader.load()
    except Exception as e:
        print(f"  [!] Eroare la incarcarea folderului {folder_path}: {e}")
        return []

    if not documents:
        print(f"  [!] Nu s-au gasit documente in {folder_path}.")
        return []

    print(f"  -> S-au incarcat {len(documents)} fisiere.")

    final_chunks = []

    for doc in documents:
        doc.metadata["subject"] = subject_name
        doc.metadata["type"] = "qa" if is_qa_mode else "course"

        if is_qa_mode:
            splits = re.split(config.QA_SPLIT_REGEX, doc.page_content)
            for split in splits:
                if split.strip():
                    new_doc = Document(page_content=split.strip(), metadata=doc.metadata)
                    final_chunks.append(new_doc)
        else:
            pass

    if not is_qa_mode:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        final_chunks = text_splitter.split_documents(documents)

    print(f"  -> Rezultat: {len(final_chunks)} chunk-uri generate pentru {subject_name} ({type_label}).")
    return final_chunks


def scan_and_process_root(root_path, is_qa_mode):
    """
    Scaneaza un folder radacina (DOCS sau DOCS2) si itereaza prin subfolderele materiilor.
    """
    chunks_accumulator = []

    if not os.path.exists(root_path):
        print(f"Atentie: Folderul radacina {root_path} nu exista. Se sare.")
        return []

    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    for subject in subfolders:
        subject_path = os.path.join(root_path, subject)

        chunks = process_subject_folder(subject, subject_path, is_qa_mode=is_qa_mode)
        chunks_accumulator.extend(chunks)

    return chunks_accumulator


def main():
    all_chunks = []

    print("=== START PROCESARE DOCS (Cursuri/Materiale) ===")
    docs_chunks = scan_and_process_root(config.DOCS_ROOT_PATH, is_qa_mode=False)
    all_chunks.extend(docs_chunks)

    print("\n=== START PROCESARE DOCS2 (Q&A) ===")
    qa_chunks = scan_and_process_root(config.DOCS2_PATH, is_qa_mode=True)
    all_chunks.extend(qa_chunks)

    if not all_chunks:
        print("\n[!] Nu s-au gasit documente de procesat in niciun folder.")
        return

    print(f"\n=== GENERARE DATABASE ({len(all_chunks)} chunk-uri totale) ===")

    if os.path.exists(config.DB_PATH):
        print(f"Se sterge baza de date veche: {config.DB_PATH}")
        shutil.rmtree(config.DB_PATH)

    print(f"Se genereaza embeddings si se salveaza in ChromaDB...")
    try:
        embeddings_model = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

        batch_size = 166
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            Chroma.from_documents(
                batch,
                embeddings_model,
                persist_directory=config.DB_PATH
            )
            current_batch = (i // batch_size) + 1
            print(f"  -> Salvat batch {current_batch}/{total_batches} ({len(batch)} chunk-uri)")

        print("\nFINALIZAT! Baza de date a fost actualizata cu succes pentru toate materiile.")

    except Exception as e:
        print(f"Eroare critica la crearea DB: {e}")


if __name__ == "__main__":
    main()