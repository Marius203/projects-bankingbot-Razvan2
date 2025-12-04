import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

# --- PROMPT OPTIMIZAT PENTRU EXPLAINABILITY ---
PROMPT_TEMPLATE = """
Esti un asistent AI expert in materia: {subject}.
Foloseste DOAR contextul de mai jos pentru a raspunde.
Daca informatia nu exista in context, spune "Nu am informatii in documentele furnizate."

Reguli:
1. Raspunde concis si la obiect.
2. La finalul raspunsului, enumera sursele folosite (nume fisier si pagina daca exista) intr-un format clar.

Context:
{context}

Intrebare:
{question}

Raspuns:
"""


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "Niciun context relevant gasit."

    formatted_text = ""
    for doc in docs:
        source = doc.metadata.get("source", "Necunoscut")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.replace("\n", " ")
        formatted_text += f"[Sursa: {os.path.basename(source)}, Pagina: {page}] Content: {content}\n\n"
    return formatted_text


def initialize_rag_system(selected_subject="ASC"):
    print(
        f"Se initializeaza RAG ({'HIBRID' if config.USE_HYBRID_SEARCH else 'STANDARD'}) pentru: {selected_subject}...")

    if not os.path.exists(config.DB_PATH):
        return None

    try:
        embeddings_model = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
        db = Chroma(persist_directory=config.DB_PATH, embedding_function=embeddings_model)
    except Exception as e:
        print(f"Eroare DB: {e}")
        return None

    # 1. RETRIEVER VECTORIAL (Semantic)
    search_kwargs = {"k": config.RETRIEVER_K}
    if selected_subject:
        search_kwargs["filter"] = {"subject": selected_subject}

    vector_retriever = db.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    final_retriever = vector_retriever

    # 2. RETRIEVER KEYWORD (BM25) - DOAR DACA ESTE ACTIVAT OPTIMIZAREA
    if config.USE_HYBRID_SEARCH:
        # Trebuie sa tragem documentele din DB pentru a construi indexul BM25 in memorie
        # Nota: Asta poate dura cateva secunde la initializare
        try:
            # Luam toate documentele pentru materia respectiva
            # (Limitam la get daca ai f multe documente, dar pt proiect e ok)
            collection_data = db.get(where={"subject": selected_subject})

            if collection_data['documents']:
                # Reconstruim obiecte Document pentru BM25
                bm25_docs = []
                for i, text in enumerate(collection_data['documents']):
                    meta = collection_data['metadatas'][i] if collection_data['metadatas'] else {}
                    bm25_docs.append(Document(page_content=text, metadata=meta))

                bm25_retriever = BM25Retriever.from_documents(bm25_docs)
                bm25_retriever.k = config.RETRIEVER_K

                # ENSEMBLING: Combinam Vector + Keyword
                final_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=config.HYBRID_WEIGHTS  # 50/50
                )
            else:
                print("Atentie: Nu s-au gasit documente pentru BM25. Se foloseste doar Vector.")
        except Exception as e:
            print(f"Eroare la initializarea BM25: {e}. Se continua doar cu Vector.")

    # 3. LLM Chain
    llm = OllamaLLM(model=config.LLM_MODEL, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).partial(subject=selected_subject)

    rag_chain = (
            RunnableParallel({
                "context": final_retriever | format_docs,
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain