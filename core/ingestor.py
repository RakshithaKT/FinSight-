"""
core/ingestor.py
Handles everything related to PDF → FAISS:
  1. Load PDFs from the /data directory
  2. Split into overlapping chunks (so context isn't lost at boundaries)
  3. Embed each chunk using Ollama's nomic-embed-text model
  4. Store in a FAISS index saved to /vectorstore

Run this file directly to ingest PDFs:
    python -m core.ingestor
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(exist_ok=True)

# ── Embedding model (Ollama, runs 100% locally, free) ─────────────────────────
EMBED_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text


def get_embeddings() -> OllamaEmbeddings:
    """Return the Ollama embedding model object."""
    return OllamaEmbeddings(model=EMBED_MODEL)


def load_and_split_pdfs(data_dir: Path = DATA_DIR) -> list:
    """
    Load every PDF in data_dir and split into overlapping chunks.
    Returns a list of LangChain Document objects.
    """
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. "
            "Download SEC 10-K filings or annual reports and place them there."
        )

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # ~250 words per chunk
        chunk_overlap=200,     # 200-char overlap so context isn't cut at boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        # Tag each chunk with its source filename so we can cite it later
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path.name
        all_docs.extend(chunks)
        print(f"    → {len(chunks)} chunks created")

    print(f"\nTotal chunks across all PDFs: {len(all_docs)}")
    return all_docs


def build_vectorstore(docs: list, save: bool = True) -> FAISS:
    """
    Embed all document chunks and store in a FAISS index.
    Saves to disk so you don't re-embed on every run.
    """
    print("\nEmbedding chunks with Ollama nomic-embed-text (this takes a minute)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    if save:
        vectorstore.save_local(str(VECTORSTORE_DIR))
        print(f"FAISS index saved to {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    Raises FileNotFoundError if you haven't run ingestion yet.
    """
    index_file = VECTORSTORE_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            "No FAISS index found. Run: python -m core.ingestor"
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,   # safe — our own files
    )


# ── Run directly to ingest ─────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_and_split_pdfs()
    build_vectorstore(docs)
    print("\n✅ Ingestion complete. You can now run main.py or app.py.")