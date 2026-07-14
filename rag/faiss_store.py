# rag/faiss_store.py
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import faiss

from config import settings
from rag.vector_store import VectorStore, Document

logger = logging.getLogger("RAGApp.FAISSStore")
logger.setLevel(logging.INFO)
import streamlit as st

# Embeddings loader helpers
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

@st.cache_resource
def get_embeddings_model():
    """
    Load the HuggingFace embedding model.
    """
    print(f"🚀 [FAISSStore] Loading HuggingFace embeddings model: {settings.EMBED_MODEL}...", flush=True)
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("HuggingFaceEmbeddings is not available in the environment.")
    model = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)
    print("🚀 [FAISSStore] HuggingFace embeddings model loaded successfully.", flush=True)
    return model

class FAISSVectorStore(VectorStore):
    """
    FAISS implementation of the VectorStore interface.
    Uses faiss.IndexIDMap to wrap IndexFlatIP for exact cosine similarity searches.
    """
    
    def __init__(self, dimension: int = 384):
        self.embeddings = get_embeddings_model()
        self.dimension = dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        self.docs: Dict[int, Document] = {}

    def add_documents(self, documents: List[Document]) -> List[int]:
        """
        Generates embeddings and adds documents with incremental unique IDs.
        """
        if not documents:
            return []
            
        logger.info(f"Generating embeddings for {len(documents)} document chunks...")
        texts = [d.page_content for d in documents]
        
        # Generate embeddings
        embeddings_list = self.embeddings.embed_documents(texts)
        embeddings_np = np.array(embeddings_list, dtype=np.float32)
        
        # Normalize vectors in-place (L2 normalization) so that flat Inner Product (IP) acts as Cosine Similarity
        faiss.normalize_L2(embeddings_np)
        
        # Generate unique incremental IDs
        next_id = max(self.docs.keys()) + 1 if self.docs else 0
        ids = list(range(next_id, next_id + len(documents)))
        ids_np = np.array(ids, dtype=np.int64)
        
        logger.info(f"Adding vectors to FAISS IndexIDMap with IDs starting at {next_id}...")
        self.index.add_with_ids(embeddings_np, ids_np)
        
        # Update our docs dict
        for id_, doc in zip(ids, documents):
            self.docs[id_] = doc
            
        return ids

    def search(self, query: str, k: int = 8) -> List[Document]:
        """
        Performs inner product (cosine similarity) search on the normalized vectors.
        """
        if not self.docs or self.index.ntotal == 0:
            return []
            
        # Get query embedding
        logger.info(f"Embedding search query: '{query[:50]}...'")
        qvec = self.embeddings.embed_query(query)
        qvec_np = np.array([qvec], dtype=np.float32)
        
        # Normalize query vector
        faiss.normalize_L2(qvec_np)
        
        # Cap k at current total vectors in index
        k_search = min(k, self.index.ntotal)
        if k_search <= 0:
            return []
            
        # Search index
        scores, indices = self.index.search(qvec_np, k_search)
        
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            # -1 signifies no match/index empty slots
            if idx != -1 and idx in self.docs:
                doc = self.docs[idx]
                # Embed the score into metadata if needed, copy to prevent modifying stored doc
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy()
                )
                new_doc.metadata["score"] = float(scores[0][i])
                new_doc.metadata["chunk_id"] = int(idx)
                retrieved_docs.append(new_doc)
                
        return retrieved_docs

    def save(self, folder_path: str) -> None:
        """
        Saves the FAISS index (index.faiss) and chunks dict (metadata.pkl) locally.
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
        
        index_path = folder / settings.FAISS_INDEX_FILE
        metadata_path = folder / settings.METADATA_PKL_FILE
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata mapping
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.docs, f)
            
        logger.info(f"FAISS vector store saved successfully to {folder_path} (Vectors count: {self.index.ntotal})")

    def load(self, folder_path: str) -> None:
        """
        Loads FAISS index and chunks dict from files, keeping backward compatibility.
        """
        folder = Path(folder_path)
        index_path = folder / settings.FAISS_INDEX_FILE
        metadata_path = folder / settings.METADATA_PKL_FILE
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"FAISS index files not found in {folder_path}")
            
        # Read FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Read metadata map
        with open(metadata_path, 'rb') as f:
            raw_docs = pickle.load(f)
            
        # Add backward-compatibility check: if it is saved as a list, convert to dict
        if isinstance(raw_docs, list):
            self.docs = {}
            for i, item in enumerate(raw_docs):
                if isinstance(item, dict):
                    self.docs[i] = Document(
                        page_content=item.get("page_content", ""),
                        metadata=item.get("metadata", {})
                    )
                else:
                    self.docs[i] = item
        elif isinstance(raw_docs, dict):
            self.docs = raw_docs
        else:
            raise ValueError(f"Unknown metadata structure type: {type(raw_docs)}")
            
        logger.info(f"FAISS vector store loaded successfully from {folder_path} (Vectors count: {self.index.ntotal})")
