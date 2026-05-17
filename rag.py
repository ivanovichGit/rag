# You might need the following imports. Feel free to change it if you opt for different libraries.

import os
import glob as globmod
from typing import Any
from typing import Optional
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    resolved = {
        "api_key": os.environ.get('OPENAI_KEY'),
        "base_url": os.environ.get('OPENAI_BASE_URL'),
        "model": os.environ.get('MODEL'),
        "embedding_model": os.environ.get('DEFAULT_EMBEDDING_MODEL'),
        "top_k": _parse_int_setting(
            "TOP_K",
            os.environ.get('DEFAULT_TOP_K'),
        ),
        "chunk_size": _parse_int_setting(
            "CHUNK_SIZE",
            os.environ.get('DEFAULT_CHUNK_SIZE'),
        ),
        "chunk_overlap": _parse_int_setting(
            "CHUNK_OVERLAP",
            os.environ.get('DEFAULT_CHUNK_OVERLAP')
        ),
    }

    if resolved["top_k"] <= 0:
        raise ValueError("TOP_K must be > 0")
    if resolved["chunk_size"] <= 0:
        raise ValueError("CHUNK_SIZE must be > 0")
    if resolved["chunk_overlap"] < 0:
        raise ValueError("CHUNK_OVERLAP must be >= 0")
    if resolved["chunk_overlap"] >= resolved["chunk_size"]:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return resolved

# --- Paso 1: Preparación de documentos ---
# Carga de documentos desde las carpetas data/emails, data/notes, data/sms y data/calendar.
def load_documents(data_dir: str = DEFAULT_DATA_DIR) -> list[Document]:
    """Loads documents from the personal data folders.
    The collection contains one LangChain Document per `.txt` file in the
    emails, notes, SMS, and calendar folders. Each document stores the file text
    as `page_content` and includes metadata for the source file path and
    document type.
    """
    documents = []

    doc_all_types = ["calendar", "emails", "notes", "sms"]

    # Cada folder 
    for doc_type in doc_all_types:

        # Todos los txt del folder
        search_pattern = os.path.join(data_dir, doc_type, "*.txt")
        file_all_paths = globmod.glob(search_pattern)

        # Cada archivo txt file path en el path del folder 
        for file_path in file_all_paths:
            
            with open(file_path, "r", encoding="utf-8") as f:
               raw_text = f.read()

            # Cada archivo .txt deberá convertirse en un Document
            doc = Document(
                page_content=raw_text, 
                # Los metadatos deberán incluir al menos la ruta del archivo y el tipo de documento.
                # Ejemplo: .../01_surprise_party.txt, calendar
                metadata={
                    "source": file_path,
                    "type" : doc_type
                    }
                )
            
            documents.append(doc)

    return documents

# División de documentos en chunks utilizando RecursiveCharacterTextSplitter.
def split_documents(docs: list[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,) -> list[Document]:
    """Splits documents into overlapping chunks.
    The resulting chunked Document objects use the configured chunk size and
    overlap while preserving the original document metadata.
    """

    # El tamaño de chunk y el overlap deberán tomarse de la configuración recibida por el pipeline.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Los metadatos de los documentos originales deberán preservarse en los chunks.
    chunks = splitter.split_documents(docs)

    return chunks

# --- Paso 2: Generar embeddings e indexar con FAISS ---
# Construcción de un índice FAISS a partir de embeddings generados con Sentence Transformers.
def build_index(chunks: list[Document],embedding_model: SentenceTransformer,) -> faiss.IndexFlatIP:
    """Creates a FAISS inner-product index for embedded document chunks.
    The index contains normalized float32 embeddings generated from each
    chunk's text with the provided embedding model.
    """

    texts = [chunk.page_content for chunk in chunks]
    
    embeddings = embedding_model.encode(texts, normalize_embeddings=True).astype(np.float32)
    
    # Creacion indice 
    index = faiss.IndexFlatIP(embeddings.shape[1])
    
    # Embedding al indice 
    index.add(embeddings)

    return index

# --- Paso 3: Recuperar documentos relevantes ---
def retrieve(query: str, index: faiss.IndexFlatIP, model: SentenceTransformer, chunks: list[Document], k: int = DEFAULT_TOP_K,) -> list[dict]:
    """Gets the most relevant chunks for a query.
    Results are ordered by similarity and include the chunk text, similarity
    score, and metadata for each matching chunk.
    """

    query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
    
    scores, indices = index.search(query_embedding, k)
    
    results = []
    
    # Recorre resultados encontrados
    for score, idx in zip(scores[0], indices[0]):

        chunk = chunks[idx]

        results.append({
            "text": chunk.page_content, 
            "score": float(score),
            "metadata": chunk.metadata
        })
    
    return results


# --- Paso 4: Construir el prompt y generar respuesta ---
# El asistente deberá recuperar contexto relevante antes de llamar al modelo de lenguaje.
# Las respuestas deberán estar fundamentadas en el contexto proporcionado, tanto en el mensaje actual como en los anteriores en el historial.
# Si no hay documentos relevantes, el asistente deberá abstenerse de responder.
SYSTEM_PROMPT = """You are a technical assistant. Answer the user's question using ONLY the provided context. Follow these rules:
- If the context doesn't contain the answer, say "I don't have enough information to answer this question."
- Be concise and precise.
- Do not use prior knowledge outside of the context."""


class Assistant:
    """Stateful RAG assistant.
    The assistant owns the pipeline components, resolved configuration, and
    conversation history. Questions are answered with retrieved document context
    and the configured chat model."""

    def __init__(
            self,
            index: faiss.IndexFlatIP,
            model: SentenceTransformer,
            chunks: list[Document],
            client: OpenAI,
            config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.index = index
        self.model = model
        self.chunks = chunks
        self.client = client
        self.config = resolve_config(config)
        self.llm_model = self.config["model"]
        self.top_k = self.config["top_k"]
        self.history: list[dict[str, str]] = []

    def ask(self, question: str, k: Optional[int] = None) -> str:
        """Generates an answer from the retrieved context and conversation history.

        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """
        pass

    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]] = None) -> "Assistant":
        """Initializes the components required by the assistant and instantiates it

        The pipeline includes resolved configuration, loaded documents, chunked
        documents, an embedding model, a FAISS index, and an OpenAI-compatible
        client.
        """
        resolved_config = resolve_config(config)

        print("Loading documents...")
        docs = load_documents()
        print(f"  Loaded {len(docs)} documents")

        print("Splitting into chunks...")
        chunks = split_documents(
            docs,
            chunk_size=resolved_config["chunk_size"],
            chunk_overlap=resolved_config["chunk_overlap"],
        )
        print(f"  Created {len(chunks)} chunks")

        embedding_model = SentenceTransformer(resolved_config["embedding_model"])

        print("Building FAISS index...")
        index = build_index(chunks, embedding_model)
        print(f"  Indexed {index.ntotal} vectors (dim={index.d})")

        client_kwargs = {}
        if resolved_config["api_key"]:
            client_kwargs["api_key"] = resolved_config["api_key"]
        if resolved_config["base_url"]:
            client_kwargs["base_url"] = resolved_config["base_url"]
        client = OpenAI(**client_kwargs)

        print("Ready!\n")
        return cls(index, embedding_model, chunks, client, resolved_config)


# Testing 
if __name__ == "__main__":

    docs = load_documents()
    chunks = split_documents(docs)
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    index = build_index(chunks, model)

    results = retrieve(
        "Where is Laura's party?",
        index,
        model,
        chunks,
        k=3
    )

    for result in results:

        print("SCORE:")
        print(result["score"])

        print("\nMETADATA:")
        print(result["metadata"])

        print("\nTEXT:")
        print(result["text"])

        print("\n" + "-" * 50 + "\n")