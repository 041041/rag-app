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

from functools import lru_cache

# Embeddings loader helpers
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

from langchain_core.embeddings import Embeddings

class GoogleGenAIEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings class wrapping the modern google-genai SDK.
    Bypasses legacy REST and model-prefix translation layers.
    """
    def __init__(self, model: str = "text-embedding-004", google_api_key: Optional[str] = None):
        from google import genai
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        
        # 1. Query the SDK to list available models for the current API key
        print("🚀 [STARTUP LOG] Querying Google GenAI SDK for available models list...", flush=True)
        available_models = []
        try:
            model_list = self.client.models.list()
            for m in model_list:
                # 2. Identify models that support embeddings (checking supported_methods / supportedMethods)
                methods = getattr(m, 'supported_methods', None) or getattr(m, 'supportedMethods', None) or []
                if 'embedContent' in methods or any('embed' in method.lower() for method in methods):
                    available_models.append(m.name)
            print(f"🚀 [STARTUP LOG] All supported embedding models for this API key: {available_models}", flush=True)
        except Exception as e:
            print(f"🚀 [STARTUP LOG] Failed to query models list from SDK: {e}", flush=True)

        # 5. Verify whether this API key supports the requested model
        supported_model = None
        for m_name in available_models:
            # Check exact match or basename match
            if m_name == self.model or m_name.split('/')[-1] == self.model.split('/')[-1]:
                supported_model = m_name
                break
                
        if supported_model:
            print(f"🚀 [STARTUP LOG] Verified requested model '{self.model}' is available on this API key.", flush=True)
            self.model = supported_model
        else:
            print(f"🚀 [STARTUP LOG] Requested embedding model '{self.model}' is NOT supported or not found for this API key.", flush=True)
            # 6. Automatically use the first supported embedding model returned by the SDK
            if available_models:
                self.model = available_models[0]
                print(f"🚀 [STARTUP LOG] Falling back dynamically to first supported model: '{self.model}'", flush=True)
            else:
                print(f"🚀 [STARTUP LOG] No supported embedding models found. Keeping default: '{self.model}'", flush=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 3. Print the exact request details and 4. Print endpoint URL
        print(f"🚀 [EMBED REQUEST] API Endpoint: https://generativelanguage.googleapis.com/v1beta", flush=True)
        print(f"🚀 [EMBED REQUEST] Method: Client.models.embed_content. Model: '{self.model}'", flush=True)
        print(f"🚀 [EMBED REQUEST] Parameters: Batch of {len(texts)} chunks", flush=True)
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )
        return [emb.values for emb in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        # 3. Print the exact request details and 4. Print endpoint URL
        print(f"🚀 [EMBED REQUEST] API Endpoint: https://generativelanguage.googleapis.com/v1beta", flush=True)
        print(f"🚀 [EMBED REQUEST] Method: Client.models.embed_content. Model: '{self.model}'", flush=True)
        print(f"🚀 [EMBED REQUEST] Parameters: Single query", flush=True)
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

@lru_cache(maxsize=1)
def get_embeddings_model():
    """
    Load the appropriate embedding model based on credentials.
    """
    # Prefer Google Gemini Embeddings (runs via API, bypassing PyTorch segfaults on Python 3.14)
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        try:
            import google.genai
            import importlib.metadata
            genai_version = importlib.metadata.version("google-genai")
        except Exception:
            genai_version = "Unknown/Not Installed"
            
        print(f"🚀 [STARTUP LOG] google-genai SDK Version: {genai_version}", flush=True)
        print("🚀 [STARTUP LOG] Instantiating custom GoogleGenAIEmbeddings wrapper...", flush=True)
        print(f"🚀 [STARTUP LOG] Embedding Class: {GoogleGenAIEmbeddings.__name__}", flush=True)
        
        try:
            model = GoogleGenAIEmbeddings(model="text-embedding-004", google_api_key=google_key)
            # Test it immediately
            model.embed_query("test")
            print("🚀 [FAISSStore] Google Gemini embeddings initialized and verified successfully.", flush=True)
            return model
        except Exception as e:
            print(f"🚀 [FAISSStore] Google Gemini embeddings test failed: {e}", flush=True)
            print("🚀 [FAISSStore] Falling back to local HuggingFace embeddings...", flush=True)
        
    print("🚀 [FAISSStore] Loading HuggingFace embeddings model (all-MiniLM-L6-v2) (Local CPU Mode)...", flush=True)
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
    
    def __init__(self, dimension: Optional[int] = None):
        print("🔧 [FAISSStore] Inside __init__", flush=True)
        print("🔧 [FAISSStore] Call get_embeddings_model()", flush=True)
        self.embeddings = get_embeddings_model()
        print("🔧 [FAISSStore] get_embeddings_model() completed", flush=True)
        
        if dimension is not None:
            self.dimension = dimension
        else:
            print("🔧 [FAISSStore] Detecting embedding dimension dynamically...", flush=True)
            test_vector = self.embeddings.embed_query("test")
            self.dimension = len(test_vector)
            print(f"🔧 [FAISSStore] Dimension detected: {self.dimension}", flush=True)
            
        print("🔧 [FAISSStore] Call faiss.IndexFlatIP()", flush=True)
        flat_index = faiss.IndexFlatIP(self.dimension)
        print("🔧 [FAISSStore] faiss.IndexFlatIP() completed", flush=True)
        
        print("🔧 [FAISSStore] Call faiss.IndexIDMap()", flush=True)
        self.index = faiss.IndexIDMap(flat_index)
        print("🔧 [FAISSStore] faiss.IndexIDMap() completed", flush=True)
        
        self.docs: Dict[int, Document] = {}
        print("🔧 [FAISSStore] __init__ completed successfully", flush=True)

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
