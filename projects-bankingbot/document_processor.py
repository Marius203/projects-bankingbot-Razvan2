"""
Helper module for incremental document processing.
Processes single PDFs and adds them to ChromaDB without rebuilding the entire database.
"""
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def process_single_document(pdf_path, subject, progress_callback=None):
    """
    Process a single PDF and add it to ChromaDB incrementally.

    Args:
        pdf_path: Full path to the PDF file
        subject: Subject name (ASC, LC, SDA, etc.)
        progress_callback: Optional function to call with progress updates
                          callback(progress_percent, message)

    Returns:
        dict: Result with success status and chunk count
    """

    def update_progress(percent, message):
        if progress_callback:
            progress_callback(percent, message)

    try:
        # Step 1: Load PDF (0-30%)
        update_progress(10, "Încărcare PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            return {"success": False, "error": "PDF-ul nu conține text sau este corupt"}

        update_progress(20, f"Încărcat {len(documents)} pagini")

        # Step 2: Add metadata (30-40%)
        update_progress(30, "Adăugare metadata...")
        for doc in documents:
            doc.metadata["subject"] = subject
            doc.metadata["type"] = "course"

        # Step 3: Split into chunks (40-60%)
        update_progress(40, "Împărțire în fragmente...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        update_progress(50, f"Generat {len(chunks)} fragmente")

        # Step 4: Connect to ChromaDB (60-70%)
        update_progress(60, "Conectare la baza de date...")
        if not os.path.exists(config.DB_PATH):
            return {"success": False, "error": "Baza de date nu există. Rulați rebuild_db.bat"}

        embeddings_model = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
        db = Chroma(persist_directory=config.DB_PATH, embedding_function=embeddings_model)

        # Step 5: Generate embeddings and add to database (70-100%)
        update_progress(70, "Generare embeddings...")

        # Process in batches to show progress
        batch_size = 10
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            db.add_documents(batch)

            current_batch = (i // batch_size) + 1
            progress = 70 + int((current_batch / total_batches) * 25)
            update_progress(progress, f"Procesare batch {current_batch}/{total_batches}")

        update_progress(95, "Finalizare...")

        # Success
        update_progress(100, "Document adăugat cu succes!")
        return {
            "success": True,
            "chunks_added": len(chunks),
            "pages": len(documents)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_document_info(pdf_path):
    """
    Get basic info about a PDF without processing it fully.

    Args:
        pdf_path: Path to PDF file

    Returns:
        dict: Info about the PDF (page count, size, etc.)
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        file_size = os.path.getsize(pdf_path)
        file_size_mb = file_size / (1024 * 1024)

        return {
            "success": True,
            "pages": len(documents),
            "size_mb": round(file_size_mb, 2),
            "filename": os.path.basename(pdf_path)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
