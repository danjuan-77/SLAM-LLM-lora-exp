from typing import List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStore

def load_vectorstore(store_type: str, path: str, embedding_model) -> VectorStore:
    """Load vector store based on the specified type."""
    store_type = store_type.lower()
    if store_type == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    elif store_type == "chroma":
        from langchain_community.vectorstores import Chroma
        return Chroma(persist_directory=path, embedding_function=embedding_model)
    else:
        raise ValueError(f"Unsupported vector store type: '{store_type}'. Supported types: 'faiss', 'chroma'.")

def search_vectorstore(
    query: str,
    top_k: int,
    store_type: str,
    store_path: str,
    embedding_model_name: str,
) -> List[str]:
    """Perform similarity search in the specified vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = load_vectorstore(store_type, store_path, embedding_model)
    results = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

def run_retrieval(query: str, decode_config) -> str:
    """
    Main entry point for vector store retrieval using config.

    Args:
        query (str): The user query.
        decode_config: Configuration object with nested `rag_config`.

    Returns:
        str: Formatted retrieval result.
    """
    rag = decode_config.rag_config

    if rag.framework.lower() != "langchain":
        raise ValueError(f"Unsupported framework '{rag.framework}'. Only 'langchain' is supported.")

    if rag.vector_store_type.lower() not in {"faiss", "chroma"}:
        raise ValueError(f"Unsupported vector store type '{rag.vector_store_type}'. Only 'faiss' and 'chroma' supported.")

    results = search_vectorstore(
        query=query,
        top_k=rag.top_k,
        store_type=rag.vector_store_type,
        store_path=rag.vector_store_path,
        embedding_model_name=rag.embedding_model_name
    )

    return "<Retrieved Documents>\n" + "\n".join(f"- {content}" for content in results)
