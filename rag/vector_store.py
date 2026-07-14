# rag/vector_store.py
from abc import ABC, abstractmethod
from typing import List

# Safe Document import mapping
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        try:
            from langchain.schema import Document
        except ImportError:
            class Document:
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

class VectorStore(ABC):
    """
    Abstract VectorStore interface representing indexing and search capabilities.
    Other databases can implement this class to be drop-in replacements.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[int]:
        """
        Encodes and inserts documents into the vector store.
        
        Args:
            documents: List of LangChain Document objects.
            
        Returns:
            List[int]: Unique integer IDs assigned to the newly added document chunks.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 8) -> List[Document]:
        """
        Performs search against the vector index.
        
        Args:
            query: The search query string.
            k: Number of nearest neighbors to retrieve.
            
        Returns:
            List[Document]: Relevant Document objects sorted by similarity.
        """
        pass

    @abstractmethod
    def save(self, folder_path: str) -> None:
        """
        Persists the index and metadata to local storage.
        
        Args:
            folder_path: The local directory path where the index should be saved.
        """
        pass

    @abstractmethod
    def load(self, folder_path: str) -> None:
        """
        Loads the index and metadata from local storage.
        
        Args:
            folder_path: The local directory path from which the index should be loaded.
        """
        pass
