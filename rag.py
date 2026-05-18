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

# Default configs
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_TOP_K = 4

def _parse_int_setting(name: str, value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer; got {value!r}") from exc
    return parsed


def resolve_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolves runtime configuration with defaults and typed settings."""
    config = config or {}

    api_key = config.get("api_key") or os.environ.get('OPENAI_API_KEY')
    base_url = config.get("base_url") or os.environ.get('OPENAI_BASE_URL')
    model = config.get("model") or os.environ.get('MODEL')
    embedding_model = config.get("embedding_model") or os.environ.get('EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL
    top_k = config.get("top_k") or os.environ.get('TOP_K') or DEFAULT_TOP_K 
    chunk_size = config.get("chunk_size") or os.environ.get('CHUNK_SIZE') or DEFAULT_CHUNK_SIZE
    chunk_overlap = config.get("chunk_overlap") or os.environ.get('CHUNK_OVERLAP') or DEFAULT_CHUNK_OVERLAP

    resolved = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "embedding_model": embedding_model,
        "top_k": _parse_int_setting("TOP_K", top_k),
        "chunk_size": _parse_int_setting("CHUNK_SIZE", chunk_size),
        "chunk_overlap": _parse_int_setting("CHUNK_OVERLAP", chunk_overlap),
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

    def ask(self, question: str, k: int | None = None) -> str:
        """Generates an answer from the retrieved context and conversation history.
        The current question is combined with relevant document chunks, previous
        conversation messages, and the system prompt. The assistant response is
        appended to history alongside the user message.
        """

        # Asistente puede permitir filtrar la búsqueda por tipo de documento utilizando etiquetas en la pregunta
        filter_type = None

        all_filters = {
            "/notes": "notes",
            "/sms": "sms",
            "/calendar": "calendar",
            "/email": "emails"
        }

        # Quita el comando de busqueda en la pregunta 
        for command, doc_type in all_filters.items():
            if command in question:
                filter_type = doc_type
                question = question.replace(command, "").strip()

                break

        results = retrieve(
            query=question,
            index=self.index,
            model=self.model,
            chunks=self.chunks, 
            k=(k or self.top_k) * 5 # overfetching - recuperar un mayor número de documentos del vector store
        )

        # Filtrarlos por metadata posteriormente
        if filter_type:
            filtered_results = []

            for result in results:
                if result["metadata"]["type"] == filter_type:
                    filtered_results.append(result)

            # Solo top_k finales
            results = filtered_results[:self.top_k]     

        if not results:
            return "I don't have enough information to answer this question."
        
        # El asistente deberá recuperar contexto relevante antes de llamar al modelo de lenguaje.
        # Las respuestas deberán estar fundamentadas en el contexto proporcionado,
        # tanto en el mensaje actual como en los anteriores en el historial.
        context = "\n\n---\n\n".join(r["text"] for r in results)

        history_txt = ""

        for message in self.history:
            history_txt += f"{message['role']}: {message['content']}\n"
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT
                 },
                {
                    "role": "user",
                    "content": f"""
                        Conversation history: {history_txt}
                        Context:{context}
                        Question: {question}"""
                 },
            ],
        )

        answer = response.choices[0].message.content
        
        # Referencias de la respuesta 
        all_sources = []

        for result in results:
            source = result["metadata"]["source"]

            # No repetir fuentes
            if source not in all_sources:
                all_sources.append(source)

        answer += "\n\nReference:\n"

        # Impresion de donde vienen
        for source in all_sources:
            answer += f"- {source}\n"
        
        # Se guarda historial para futuras preguntas  
        self.history.append({
            "role": "user",
            "content": question
        })

        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer 


    def clear_history(self) -> None:
        """Empties the conversation history."""
        self.history.clear()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> 'Assistant':
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