# rag/retrieval.py
import logging
from typing import List, Tuple, Any

from rag.vector_store import VectorStore, Document

logger = logging.getLogger("RAGApp.Retrieval")
logger.setLevel(logging.INFO)

# BaseRetriever import with fallback identical to app.py
try:
    from langchain.schema import BaseRetriever
except ImportError:
    try:
        from langchain_core.schema import BaseRetriever
    except ImportError:
        BaseRetriever = object

class VectorStoreRetrieverAdapter(BaseRetriever):
    """
    Adapter class that wraps our custom VectorStore interface to be compatible 
    with LangChain's RetrievalQA chains and retriever APIs.
    """
    model_config = {"extra": "allow"}

    def __init__(self, vector_store: VectorStore, k: int = 8, filter_source: str = None):
        # We call object.__setattr__ because Pydantic-based BaseRetriever might block direct attributes
        object.__setattr__(self, "vector_store", vector_store)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "filter_source", filter_source)
        object.__setattr__(self, "tags", [])
        object.__setattr__(self, "metadata", {})

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from local FAISS vector store.
        """
        logger.info(f"Retrieving relevant documents for query: '{query[:50]}...' with filter: {self.filter_source}")
        return self.vector_store.search(query, k=self.k, filter_source=self.filter_source)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Asynchronous document retrieval (falls back to synchronous execution).
        """
        return self.get_relevant_documents(query)

    def get_relevant_documents_with_score(self, query: str) -> List[Tuple[Document, float]]:
        """
        Returns document search results with similarity scores.
        """
        docs = self.vector_store.search(query, k=self.k, filter_source=self.filter_source)
        return [(d, d.metadata.get("score", 1.0)) for d in docs]
