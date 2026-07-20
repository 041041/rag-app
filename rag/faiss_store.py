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
    Load the HuggingFace embedding model with offline support.
    Preferred order:
      1. models/bge-small-en-v1.5
      2. BAAI/bge-small-en-v1.5 (download and save locally if missing)
      3. models/all-MiniLM-L6-v2 (local offline fallback)
    """
    import os
    from pathlib import Path
    
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Try BAAI/bge-small-en-v1.5 local path
    local_bge_path = models_dir / "bge-small-en-v1.5"
    if local_bge_path.exists():
        try:
            logger.info("Loading BAAI/bge-small-en-v1.5 from local disk...")
            model = HuggingFaceEmbeddings(
                model_name=str(local_bge_path),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Successfully loaded local BAAI/bge-small-en-v1.5 model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load local BAAI/bge-small-en-v1.5: {e}")
            
    # 2. Try downloading BAAI/bge-small-en-v1.5 if missing (and we are online)
    try:
        logger.info("Local BAAI/bge-small-en-v1.5 not found. Downloading from Hugging Face Hub...")
        model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Saving downloaded BAAI/bge-small-en-v1.5 model to {local_bge_path} for offline use...")
        client = getattr(model, "_client", None)
        if client is not None and hasattr(client, "save"):
            client.save(str(local_bge_path))
        logger.info("Successfully downloaded and saved BAAI/bge-small-en-v1.5 model.")
        return model
    except Exception as e:
        logger.warning(f"Failed to download BAAI/bge-small-en-v1.5: {e}. Falling back to models/all-MiniLM-L6-v2...")
        
    # 3. Try fallback models/all-MiniLM-L6-v2
    local_minilm_path = models_dir / "all-MiniLM-L6-v2"
    if local_minilm_path.exists():
        try:
            logger.info("Loading all-MiniLM-L6-v2 from local disk...")
            model = HuggingFaceEmbeddings(
                model_name=str(local_minilm_path),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Successfully loaded local all-MiniLM-L6-v2 model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load local all-MiniLM-L6-v2: {e}")
            
    raise RuntimeError("❌ [FAISSStore] No embedding models could be loaded or downloaded.")

class FAISSVectorStore(VectorStore):
    """
    FAISS implementation of the VectorStore interface.
    Uses faiss.IndexIDMap to wrap IndexFlatIP for exact cosine similarity searches.
    """
    
    def __init__(self, dimension: int = 384):
        import time
        import traceback
        
        print("🚀 [INIT 1] Constructor entered", flush=True)
        start_time = time.time()
        
        # Step 2: Creating embeddings
        print("🚀 [INIT 2] Creating embeddings...", flush=True)
        try:
            t0 = time.time()
            self.embeddings = get_embeddings_model()
            dt = time.time() - t0
            print(f"🚀 [INIT 3] Embeddings ready. Time taken: {dt:.2f}s", flush=True)
            if dt > 5.0:
                print(f"⚠️ [WARNING] Embedding model load took longer than 5 seconds: {dt:.2f}s", flush=True)
        except Exception as e:
            print("❌ [INIT ERROR] Failed to load embeddings:", flush=True)
            traceback.print_exc()
            raise e
            
        # Set dimension
        self.dimension = dimension
        
        # Step 4: Creating flat FAISS index
        print("🚀 [INIT 4] Creating FAISS IndexFlatIP...", flush=True)
        try:
            t0 = time.time()
            self.flat_index = faiss.IndexFlatIP(self.dimension)
            dt = time.time() - t0
            print(f"🚀 [INIT 5] FAISS IndexFlatIP ready. Time taken: {dt:.2f}s", flush=True)
            if dt > 5.0:
                print(f"⚠️ [WARNING] IndexFlatIP creation took longer than 5 seconds: {dt:.2f}s", flush=True)
        except Exception as e:
            print("❌ [INIT ERROR] Failed to create IndexFlatIP:", flush=True)
            traceback.print_exc()
            raise e
            
        # Step 6: Creating IndexIDMap wrapper
        print("🚀 [INIT 6] Wrapping index with IndexIDMap...", flush=True)
        try:
            t0 = time.time()
            self.index = faiss.IndexIDMap(self.flat_index)
            dt = time.time() - t0
            print(f"🚀 [INIT 7] FAISS IndexIDMap ready. Time taken: {dt:.2f}s", flush=True)
            if dt > 5.0:
                print(f"⚠️ [WARNING] IndexIDMap wrapping took longer than 5 seconds: {dt:.2f}s", flush=True)
        except Exception as e:
            print("❌ [INIT ERROR] Failed to wrap with IndexIDMap:", flush=True)
            traceback.print_exc()
            raise e
            
        # Step 8: Document mapping dict initialization
        print("🚀 [INIT 8] Initializing document mapping dictionary...", flush=True)
        self.docs: Dict[int, Document] = {}
        
        total_dt = time.time() - start_time
        print(f"🚀 [INIT 9] Initialization complete. Total time: {total_dt:.2f}s", flush=True)

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

    def search(self, query: str, k: int = 8, filter_sources: List[str] = None) -> List[Document]:
        """
        Performs inner product (cosine similarity) search on the normalized vectors.
        Optionally filters results to a specific set of source files, applies query expansion,
        document-aware boosting, keyword-overlap reranking, and similarity thresholding.
        """
        if not self.docs or self.index.ntotal == 0:
            return []
            
        import re
        
        def compute_keyword_overlap(q_str: str, text: str) -> float:
            stopwords = {"what", "is", "the", "and", "of", "in", "to", "for", "with", "a", "an", "on", "at", "by", "from", "as", "about"}
            query_words = set(re.findall(r'\b[a-zA-Z0-9_]{3,}\b', q_str.lower()))
            query_words = query_words - stopwords
            if not query_words:
                return 0.0
            text_lower = text.lower()
            matches = sum(1 for word in query_words if word in text_lower)
            return matches / len(query_words)

        # Task 1 & 2: Query Expansion
        expanded_query = query
        query_lower = query.lower()
        if "sdtm" in query_lower:
            expanded_query += " Study Data Tabulation Model CDISC standard definition purpose domains structure"
        if "adam" in query_lower:
            expanded_query += " Analysis Data Model CDISC standard definition datasets ADSL"

        # Get query embedding
        logger.info(f"Embedding search query: '{query[:50]}...' (Expanded: '{expanded_query[:50]}...')")
        qvec = self.embeddings.embed_query(expanded_query)
        qvec_np = np.array([qvec], dtype=np.float32)
        
        # Normalize query vector
        faiss.normalize_L2(qvec_np)
        
        # Fetch candidate pool (larger than k to allow reranking)
        fetch_k = min(k * 5, self.index.ntotal)
        if fetch_k <= 0:
            return []
            
        scores, indices = self.index.search(qvec_np, fetch_k)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.docs:
                doc = self.docs[idx]
                # Apply hard document source filter
                if filter_sources and doc.metadata.get("source") not in filter_sources:
                    continue
                
                base_score = float(scores[0][i])
                
                # Task 3: Domain-aware boosting and penalties
                source_lower = str(doc.metadata.get("source", "")).lower()
                boost_or_penalty = 0.0
                
                adam_kws = {"adam", "avisit", "aval", "adsl", "shift"}
                sdtm_kws = {"sdtm", "domains", "dm", "ae", "lb"}
                
                has_adam_kw = any(kw in query_lower for kw in adam_kws)
                has_sdtm_kw = any(kw in query_lower for kw in sdtm_kws)
                
                is_adam_doc = "adam" in source_lower or "adamig" in source_lower
                is_sdtm_doc = "sdtm" in source_lower or "sdtmig" in source_lower
                is_standard_doc = is_adam_doc or is_sdtm_doc
                
                if has_adam_kw:
                    if is_adam_doc:
                        boost_or_penalty = 0.30
                    elif not is_standard_doc:
                        boost_or_penalty = -0.40
                elif has_sdtm_kw:
                    if is_sdtm_doc:
                        boost_or_penalty = 0.30
                    elif not is_standard_doc:
                        boost_or_penalty = -0.40
                else:
                    if "sdtm" in query_lower and is_sdtm_doc:
                        boost_or_penalty = 0.15
                    elif "adam" in query_lower and is_adam_doc:
                        boost_or_penalty = 0.15
                        
                boosted_score = base_score + boost_or_penalty
                
                # Task 1: Filter by similarity score threshold
                if boosted_score < settings.MIN_SIMILARITY_SCORE:
                    continue
                
                # Task 4: Semantic relevance reranking (keyword density overlap)
                overlap = compute_keyword_overlap(query, doc.page_content)
                final_score = boosted_score + 0.2 * overlap
                
                candidates.append((doc, idx, base_score, boosted_score, final_score))

        # Sort candidates by final score descending
        candidates.sort(key=lambda x: x[4], reverse=True)
        
        retrieved_docs = []
        for doc, idx, base_score, boosted_score, final_score in candidates[:k]:
            new_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            new_doc.metadata["score"] = base_score
            new_doc.metadata["boosted_score"] = boosted_score
            new_doc.metadata["final_score"] = final_score
            new_doc.metadata["chunk_id"] = int(idx)
            retrieved_docs.append(new_doc)

        # Task 6: Add debug logs
        logger.info(f"--- RAG RETRIEVAL DEBUG LOG ---")
        logger.info(f"Question: '{query}'")
        logger.info("Retrieved candidate chunks:")
        for doc, idx, base_score, boosted_score, final_score in candidates:
            logger.info(f"  - Document: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} | FAISS Score: {base_score:.4f} | Boosted: {boosted_score:.4f} | Final Score: {final_score:.4f}")
            
        logger.info("Reranked final chunks sent to LLM:")
        for doc in retrieved_docs:
            logger.info(f"  - Document: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} | Final Score: {doc.metadata.get('final_score'):.4f}")
        logger.info(f"-------------------------------")

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
