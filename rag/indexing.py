# rag/indexing.py
import hashlib
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from config import settings
from storage import r2_storage
from rag.vector_store import VectorStore, Document

logger = logging.getLogger("RAGApp.Indexing")
logger.setLevel(logging.INFO)

# --- Safe Imports for Splitters & Loaders ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

# Document loaders
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
        Docx2txtLoader,
        UnstructuredHTMLLoader,
    )
except ImportError:
    try:
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            CSVLoader,
            Docx2txtLoader,
            UnstructuredHTMLLoader,
        )
    except ImportError:
        PyPDFLoader = TextLoader = CSVLoader = Docx2txtLoader = UnstructuredHTMLLoader = None

def _get_loader_for_path(fp: Path):
    ext = fp.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader
    if ext in [".txt", ".md"]:
        return TextLoader
    if ext == ".csv":
        return CSVLoader
    if ext == ".docx":
        return Docx2txtLoader
    if ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader
    return None

def compute_file_hash(file_bytes: bytes) -> str:
    """
    Computes SHA-256 hash of file bytes for duplicate detection.
    """
    return hashlib.sha256(file_bytes).hexdigest()

def get_document_metadata() -> Dict[str, Any]:
    """
    Loads document metadata from the local cache folder.
    Returns default schema if the file does not exist or is corrupted.
    """
    metadata_path = settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading document metadata: {e}. Re-initializing metadata.")
            
    return {
        "version": 0,
        "last_updated": "",
        "documents": {}
    }

def save_document_metadata(metadata: Dict[str, Any]) -> None:
    """
    Saves document metadata back to the local cache folder.
    """
    metadata_path = settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def process_and_index_file(
    filename: str,
    file_bytes: bytes,
    vector_store: VectorStore
) -> Dict[str, Any]:
    """
    Full document ingestion workflow:
    1. Check for duplicates (filename & hash).
    2. Upload original file to Cloudflare R2 bucket.
    3. Extract text.
    4. Chunk document.
    5. Generate embeddings and append to FAISS (no rebuilding).
    6. Update and version metadata.
    7. Sync new index files back to R2.
    
    Returns:
        A dict containing status ('skipped' or 'success'), timing logs, and errors if any.
    """
    results = {
        "status": "success",
        "timings": {},
        "message": ""
    }
    
    # Step 1: Duplicate check by computing file hash
    file_hash = compute_file_hash(file_bytes)
    metadata = get_document_metadata()
    
    # Check if duplicate by hash or by filename
    for doc_id, doc_info in metadata.get("documents", {}).items():
        if doc_info.get("hash") == file_hash:
            results["status"] = "skipped"
            results["message"] = f"Document already exists (hash match: {doc_info.get('filename')}). Skipping."
            logger.info(results["message"])
            return results
        if doc_info.get("filename") == filename:
            results["status"] = "skipped"
            results["message"] = f"Document already exists (filename match: {filename}). Skipping."
            logger.info(results["message"])
            return results

    # Save file locally under data directory
    local_doc_path = settings.DATA_DIR / filename
    local_doc_path.write_bytes(file_bytes)
    
    try:
        # Step 2: Upload original document to Cloudflare R2 (documents/ prefix)
        t_start = time.time()
        r2_key = f"{settings.R2_DOCUMENTS_PREFIX}{filename}"
        r2_uploaded = r2_storage.upload_file(local_doc_path, r2_key)
        if not r2_uploaded:
            raise RuntimeError("Cloudflare R2 document upload failed.")
        results["timings"]["document_upload"] = time.time() - t_start
        
        # Step 3: Extract document text
        t_start = time.time()
        logger.info(f"Extracting text from {filename}...")
        Loader = _get_loader_for_path(local_doc_path)
        if not Loader:
            raise ValueError(f"No supported loader found for file: {filename}")
            
        if Loader in (CSVLoader, TextLoader):
            loader = Loader(str(local_doc_path), encoding="utf-8")
        else:
            loader = Loader(str(local_doc_path))
            
        extracted_docs = loader.load()
        for d in extracted_docs:
            d.metadata = getattr(d, "metadata", {}) or {}
            d.metadata["source"] = filename
        results["timings"]["text_extraction"] = time.time() - t_start
        
        # Step 4: Chunk document
        t_start = time.time()
        logger.info(f"Chunking {filename} (chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("LangChain RecursiveCharacterTextSplitter is not available.")
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(extracted_docs)
        results["timings"]["chunking"] = time.time() - t_start
        logger.info(f"Split {filename} into {len(chunks)} chunks.")
        
        # Step 5: Generate embeddings and append ONLY new chunks to existing index
        t_start = time.time()
        logger.info(f"Generating embeddings and appending {len(chunks)} chunks into FAISS index...")
        chunk_ids = vector_store.add_documents(chunks)
        results["timings"]["embedding_generation"] = time.time() - t_start
        results["timings"]["faiss_update"] = results["timings"]["embedding_generation"]
        
        # Step 6: Update Metadata with versioning
        new_version = metadata.get("version", 0) + 1
        doc_id = f"doc_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]}"
        
        metadata["documents"][doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "hash": file_hash,
            "timestamp": datetime.now().isoformat(),
            "r2_path": r2_key,
            "chunk_count": len(chunks),
            "chunk_ids": chunk_ids
        }
        metadata["version"] = new_version
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Step 7: Save updated index and metadata files locally
        vector_store.save(str(settings.INDEXES_DIR))
        save_document_metadata(metadata)
        
        # Step 8: Automatically upload updated index files back to Cloudflare R2
        t_start = time.time()
        logger.info("Uploading updated index files to R2...")
        
        # S3 server-side backup first
        r2_storage.backup_indexes()
        
        # Upload index files
        r2_storage.upload_file(settings.INDEXES_DIR / settings.FAISS_INDEX_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}")
        r2_storage.upload_file(settings.INDEXES_DIR / settings.METADATA_PKL_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}")
        r2_storage.upload_file(settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
        
        results["timings"]["uploading_index_to_r2"] = time.time() - t_start
        results["message"] = f"Successfully processed and indexed {filename}."
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        results["status"] = "error"
        results["message"] = f"Failed to process file: {str(e)}"
        
    return results
