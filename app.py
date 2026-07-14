# app.py - Refactored for Cloudflare R2 and FAISS Integration
import os
import asyncio
import time
import json
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGApp.Main")

# Suppress verbose third-party HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Import modular components
from config import settings
from storage import r2_storage
from rag.faiss_store import FAISSVectorStore
from rag.retrieval import VectorStoreRetrieverAdapter
from rag.indexing import process_and_index_file, get_document_metadata, save_document_metadata

# Ensure an asyncio event loop exists for the current thread (Streamlit-related fix)
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

# --- LangChain imports and fallbacks ---
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        try:
            from langchain.schema import Document
        except Exception:
            class Document:
                def __init__(self, page_content="", metadata=None):
                    self.page_content = page_content
                    self.metadata = metadata or {}

# PromptTemplate import
try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain.prompt import PromptTemplate
    except Exception:
        class PromptTemplate:
            def __init__(self, input_variables=None, template=""):
                self.input_variables = input_variables or []
                self.template = template

# RetrievalQA import
RetrievalQA = None
try:
    from langchain.chains.retrieval_qa.base import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        try:
            from langchain.chains import RetrievalQA
        except Exception:
            RetrievalQA = None

# Google Gemini integration
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


# ========================================
# METRICS & CACHE
# ========================================
class MetricsTracker:
    def __init__(self):
        self.queries = []
        self.total_queries = 0
        self.cache_hits = 0
        self.total_tokens = 0
        self.errors = 0

    def log_query(self, query: str, response_time: float, cached: bool = False, tokens: int = 0, error: bool = False):
        self.total_queries += 1
        if cached:
            self.cache_hits += 1
        if error:
            self.errors += 1
        self.total_tokens += tokens
        self.queries.append({
            "query": query[:100],
            "response_time": response_time,
            "cached": cached,
            "tokens": tokens,
            "timestamp": datetime.now(),
            "error": error
        })

    def get_stats(self):
        if self.total_queries == 0:
            return {
                "total_queries": 0,
                "cache_hit_rate": "0%",
                "avg_response_time": 0,
                "error_rate": "0%",
                "total_tokens": 0,
                "estimated_cost": "$0.00"
            }
        cache_rate = (self.cache_hits / self.total_queries) * 100
        error_rate = (self.errors / self.total_queries) * 100
        non_cached = [q for q in self.queries if not q["cached"]]
        avg_time = sum(q["response_time"] for q in non_cached) / len(non_cached) if non_cached else 0
        estimated_cost = (self.total_tokens / 1_000_000) * 0.35
        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "avg_response_time": f"{avg_time:.2f}s",
            "error_rate": f"{error_rate:.1f}%",
            "total_tokens": self.total_tokens,
            "estimated_cost": f"${estimated_cost:.4f}"
        }

if 'metrics' not in st.session_state:
    st.session_state.metrics = MetricsTracker()

class QueryCache:
    def __init__(self, max_size=50, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def _get_key(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str):
        key = self._get_key(query)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None

    def set(self, query: str, result):
        key = self._get_key(query)
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (result, time.time())

    def get_stats(self):
        return {"size": len(self.cache), "max_size": self.max_size, "usage": f"{(len(self.cache) / self.max_size) * 100:.1f}%"}

if 'query_cache' not in st.session_state:
    st.session_state.query_cache = QueryCache(max_size=50, ttl_seconds=3600)


# -------------------------
# Robust extractor for LLM responses
# -------------------------
def _extract_text_from_llm_response(raw):
    try:
        if raw is None:
            return "", raw
        if isinstance(raw, str):
            return raw, raw
        if isinstance(raw, dict):
            for k in ("result", "text", "content", "message", "output", "response"):
                if k in raw and isinstance(raw[k], str):
                    return raw[k], raw
            if "candidates" in raw and isinstance(raw["candidates"], (list, tuple)) and raw["candidates"]:
                c = raw["candidates"][0]
                if isinstance(c, str):
                    return c, raw
                if isinstance(c, dict) and "content" in c and isinstance(c["content"], str):
                    return c["content"], raw
        if hasattr(raw, "generations"):
            try:
                gens = getattr(raw, "generations")
                if gens:
                    first = gens[0]
                    if isinstance(first, list) and first:
                        g = first[0]
                    else:
                        g = first
                    for attr in ("text", "content", "message"):
                        if hasattr(g, attr):
                            val = getattr(g, attr)
                            if isinstance(val, str) and val:
                                return val, raw
                            if hasattr(val, "content") and isinstance(val.content, str):
                                return val.content, raw
                    if hasattr(g, "text") and isinstance(g.text, str):
                        return g.text, raw
            except Exception:
                pass
        for attr in ("content", "text", "message", "result"):
            if hasattr(raw, attr):
                val = getattr(raw, attr)
                if isinstance(val, str):
                    return val, raw
                if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
                    return val.content, raw
        return str(raw), raw
    except Exception as e:
        return f"<unextractable response: {e}>", raw


# -------------------------
# Robust fallback QA wrapper
# -------------------------
class SimpleQAWrapper:
    """
    Robust fallback QA wrapper that prefers ChatGoogleGenerativeAI.invoke(str).
    """
    def __init__(self, llm, retriever, prompt_template):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt_template

    def _build_input(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
        if hasattr(self.prompt, "template"):
            prompt_text = self.prompt.template.format(question=query, context=context)
        else:
            prompt_text = f"Question: {query}\nContext:\n{context}"
        return prompt_text, docs

    def _call_llm_variants(self, prompt_text: str):
        last_err = None
        last_raw = None

        # Prefer invoking with a plain string
        inv_fn = getattr(self.llm, "invoke", None)
        if callable(inv_fn):
            try:
                raw = inv_fn(prompt_text)
                last_raw = raw
                if hasattr(raw, "content") and isinstance(getattr(raw, "content"), str):
                    return getattr(raw, "content"), raw, getattr(raw, "source_documents", None) or []
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                last_err = e

        # Try generate with list-of-dicts
        gen_fn = getattr(self.llm, "generate", None)
        if callable(gen_fn):
            try:
                raw = gen_fn([{"content": prompt_text}])
                last_raw = raw
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                last_err = e

        # Other callables
        for name in ("predict", "create", "chat", "respond", "answer"):
            fn = getattr(self.llm, name, None)
            if not callable(fn):
                continue
            try:
                raw = fn(prompt_text)
                last_raw = raw
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                last_err = e
                continue

        raw_type = type(last_raw).__name__ if last_raw is not None else "None"
        raw_preview = repr(last_raw)[:1000] if last_raw is not None else "<no raw captured>"
        raise RuntimeError(f"No callable LLM methods succeeded. last_err: {last_err} | last_raw_type: {raw_type} | last_raw_preview: {raw_preview}")

    def run(self, query: str):
        prompt_text, docs = self._build_input(query)
        text, raw, src_docs = self._call_llm_variants(prompt_text)
        source_documents = src_docs or docs
        return {"result": text, "source_documents": source_documents, "raw": raw}


# -------------------------
# LLM prompt / QA builder
# -------------------------
PROMPT_TEMPLATE_STR = (
    "You are an expert assistant for clinical trial data standards. Be concise and use bullet points when helpful.\n\n"
    "Question: {question}\nContext:\n{context}\n\nAnswer:"
)
prompt_template = PromptTemplate(input_variables=["question", "context"], template=PROMPT_TEMPLATE_STR)

def create_qa_from_retriever(retriever):
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("ChatGoogleGenerativeAI (langchain-google-genai) not available; install it.")
    llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0.2, max_output_tokens=1024)
    try:
        if RetrievalQA is None:
            raise ImportError("RetrievalQA not importable in this environment.")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
        )
        return qa
    except Exception as e:
        logger.warning(f"⚠️ Could not create RetrievalQA chain (falling back to internal wrapper): {e}")
        return SimpleQAWrapper(llm=llm, retriever=retriever, prompt_template=prompt_template)


# -------------------------
# Chain-safe invoker
# -------------------------
def _call_chain_safe(qa_chain, query: str):
    """
    Try common invocation methods and normalize output:
    returns {"result": str, "source_documents": [Document, ...], "raw": raw}
    Accepts a LangChain RetrievalQA-like object OR our SimpleQAWrapper instance.
    """
    last_err = None
    last_raw = None

    # If it's our fallback wrapper, call its run(query)
    if hasattr(qa_chain, "run") and callable(getattr(qa_chain, "run")) and type(qa_chain).__name__ == "SimpleQAWrapper":
        try:
            return qa_chain.run(query)
        except Exception as e:
            raise RuntimeError(f"Could not invoke QA chain - last error: {e}")

    # Otherwise try common langchain chain call patterns
    call_methods = [
        ("run", lambda fn, q: fn(q)),
        ("invoke", lambda fn, q: fn(q)),
        ("__call__", lambda fn, q: fn(q)),
        ("predict", lambda fn, q: fn(q)),
        ("generate", lambda fn, q: fn([q]))
    ]
    for name, caller in call_methods:
        fn = getattr(qa_chain, name, None)
        if not callable(fn):
            continue
        try:
            raw = caller(fn, query)
            last_raw = raw
            # if raw is dict
            if isinstance(raw, dict):
                for k in ("result", "text", "content", "message", "output", "response"):
                    if k in raw and isinstance(raw[k], str):
                        return {"result": raw[k], "source_documents": raw.get("source_documents", []) or raw.get("documents", []), "raw": raw}
                if "candidates" in raw and isinstance(raw["candidates"], (list, tuple)) and raw["candidates"]:
                    c0 = raw["candidates"][0]
                    if isinstance(c0, str):
                        return {"result": c0, "source_documents": raw.get("source_documents", []) or raw.get("documents", []), "raw": raw}
                    if isinstance(c0, dict) and "content" in c0 and isinstance(c0["content"], str):
                        return {"result": c0["content"], "source_documents": raw.get("source_documents", []) or raw.get("documents", []), "raw": raw}
            if isinstance(raw, str):
                return {"result": raw, "source_documents": [], "raw": raw}
            # langchain-style generations
            try:
                if hasattr(raw, "generations"):
                    gens = getattr(raw, "generations")
                    if gens:
                        g0 = gens[0]
                        candidate = g0[0] if isinstance(g0, list) and g0 else g0
                        for attr in ("text", "content", "message"):
                            if hasattr(candidate, attr):
                                val = getattr(candidate, attr)
                                if isinstance(val, str):
                                    return {"result": val, "source_documents": getattr(raw, "source_documents", []) or [], "raw": raw}
                                if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
                                    return {"result": getattr(val, "content"), "source_documents": getattr(raw, "source_documents", []) or [], "raw": raw}
            except Exception:
                pass
            # final best-effort by checking attributes
            source_docs = getattr(raw, "source_documents", None) or (raw.get("source_documents") if isinstance(raw, dict) else None) or []
            for attr in ("text", "content", "message", "result"):
                if hasattr(raw, attr):
                    val = getattr(raw, attr)
                    if isinstance(val, str):
                        return {"result": val, "source_documents": source_docs or [], "raw": raw}
                    if hasattr(val, "content") and isinstance(getattr(val, "content"), str):
                        return {"result": getattr(val, "content"), "source_documents": source_docs or [], "raw": raw}
            return {"result": str(raw), "source_documents": source_docs or [], "raw": raw}
        except Exception as e:
            last_err = e
            continue

    debug_raw_repr = repr(last_raw) if last_raw is not None else "<no raw captured>"
    debug_raw_type = type(last_raw).__name__ if last_raw is not None else "None"
    raise RuntimeError(f"Could not invoke QA chain - last error: {last_err} | last_raw_type: {debug_raw_type} | last_raw_preview: {debug_raw_repr[:1000]}")


# -------------------------
# Query wrapper with caching & metrics
# -------------------------
def query_with_features(qa_chain, query: str):
    start_time = time.time()
    cache = st.session_state.query_cache
    metrics = st.session_state.metrics

    cached_result = cache.get(query)
    if cached_result:
        elapsed = time.time() - start_time
        metrics.log_query(query, elapsed, cached=True)
        st.success("🎯 Cache hit! Instant response.")
        return cached_result, True, elapsed

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info("Search started...")
            result = _call_chain_safe(qa_chain, query)
            elapsed = time.time() - start_time
            logger.info(f"Search completed in {elapsed:.2f}s.")
            estimated_tokens = int(len(query.split()) + len(result.get("result", "").split()) * 1.3)
            metrics.log_query(query, elapsed, cached=False, tokens=int(estimated_tokens))
            cache.set(query, result)
            return result, False, elapsed
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                elapsed = time.time() - start_time
                metrics.log_query(query, elapsed, error=True)
                err_str = str(e)
                debug_hint = ""
                try:
                    if "last_raw_type" in err_str or "last_raw_preview" in err_str:
                        debug_hint = "\n\nDebug info from chain:\n" + err_str
                except Exception:
                    debug_hint = f"\n\nException repr: {repr(e)}"
                st.error(f"❌ All retries failed: {err_str}{debug_hint}")
                st.write("----\n**Debug (copy this and paste in chat):**")
                st.write("Exception:", repr(e))
                try:
                    if "last_raw_preview" in err_str:
                        preview = err_str.split("last_raw_preview: ", 1)[-1]
                        st.code(preview[:5000])
                except Exception:
                    pass
                return None, False, elapsed

    return None, False, time.time() - start_time


# ========================================
# APPLICATION STARTUP & INITIALIZATION
# ========================================
def initialize_app():
    """
    Startup Health Check & Synchronization workflow.
    - Connects to R2
    - Checks index files availability
    - Downloads index files if present or creates a new empty FAISS index.
    """
    if "vector_store" not in st.session_state:
        st.session_state.health_status = {
            "r2_connected": False,
            "index_loaded": False,
            "version": 0,
            "last_updated": "Never",
            "message": ""
        }
        
        # Initialize default vector store structure
        store = FAISSVectorStore()
        
        # Health Check: Connect to R2
        r2_connected = r2_storage.verify_connection()
        st.session_state.health_status["r2_connected"] = r2_connected
        
        if r2_connected:
            # Health Check: Verify if Index files exist on Cloudflare R2
            logger.info("Checking index availability on R2...")
            index_files_exist = False
            try:
                index_files_exist = (
                    r2_storage.check_file_exists(f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}") and
                    r2_storage.check_file_exists(f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}") and
                    r2_storage.check_file_exists(f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
                )
            except Exception as e:
                logger.error(f"R2 verification error: {e}")
                
            if index_files_exist:
                logger.info("Downloading index from R2...")
                try:
                    # Synchronize Local Cache
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}",
                        settings.INDEXES_DIR / settings.FAISS_INDEX_FILE
                    )
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}",
                        settings.INDEXES_DIR / settings.METADATA_PKL_FILE
                    )
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}",
                        settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE
                    )
                    
                    # Load FAISS index and metadata
                    store.load(str(settings.INDEXES_DIR))
                    st.session_state.health_status["index_loaded"] = True
                    
                    # Extract version details
                    doc_meta = get_document_metadata()
                    st.session_state.health_status["version"] = doc_meta.get("version", 0)
                    st.session_state.health_status["last_updated"] = doc_meta.get("last_updated", "Unknown")
                    st.session_state.health_status["message"] = f"Synchronized successfully with R2 index (v{st.session_state.health_status['version']})."
                    
                except Exception as e:
                    logger.error(f"Failed to synchronize index from R2: {e}")
                    st.session_state.health_status["message"] = f"Failed to load R2 index files: {e}. Falling back to empty index."
                    # Fallback to local empty
                    store = FAISSVectorStore()
                    store.save(str(settings.INDEXES_DIR))
                    empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
                    save_document_metadata(empty_meta)
            else:
                logger.info("No index files found on R2. Initializing empty index...")
                st.session_state.health_status["message"] = "Cloudflare R2 empty. Initialized empty local index."
                # Save empty local files
                store.save(str(settings.INDEXES_DIR))
                empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
                save_document_metadata(empty_meta)
        else:
            logger.warning("R2 Connection failed on startup. Checking if local cache exists offline...")
            st.session_state.health_status["message"] = "Offline mode: R2 connection failed."
            
            # Check if local files exist offline
            local_exists = (
                (settings.INDEXES_DIR / settings.FAISS_INDEX_FILE).exists() and
                (settings.INDEXES_DIR / settings.METADATA_PKL_FILE).exists() and
                (settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE).exists()
            )
            if local_exists:
                try:
                    store.load(str(settings.INDEXES_DIR))
                    st.session_state.health_status["index_loaded"] = True
                    doc_meta = get_document_metadata()
                    st.session_state.health_status["version"] = doc_meta.get("version", 0)
                    st.session_state.health_status["last_updated"] = doc_meta.get("last_updated", "Unknown")
                except Exception as e:
                    logger.error(f"Error loading offline cache: {e}")
                    store = FAISSVectorStore()
                    store.save(str(settings.INDEXES_DIR))
                    empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
                    save_document_metadata(empty_meta)
            else:
                store.save(str(settings.INDEXES_DIR))
                empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
                save_document_metadata(empty_meta)
                
        st.session_state.vector_store = store

# Initialize state
initialize_app()


# ========================================
# STREAMLIT UI - RICH DESIGN AESTHETICS
# ========================================
st.set_page_config(page_title="Clinical Docs Search (Cloudflare R2 RAG)", layout="wide")

# Inject Custom Elegant Styling for Premium Aesthetics
st.markdown("""
<style>
    /* Styling headers & cards */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 25, 41) 0%, rgb(10, 12, 18) 90%);
        color: #f0f3f9;
    }
    .css-1d391kg {
        background-color: rgba(18, 22, 33, 0.9) !important;
    }
    h1 {
        background: linear-gradient(135deg, #a5f3fc 0%, #38bdf8 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 Clinical Docs Search & Assistant")
    st.caption("🚀 Migrated to Cloudflare R2 Persistent Storage & FAISS Incremental Indexing")
with col2:
    stats = st.session_state.metrics.get_stats()
    st.metric("Total Queries", stats["total_queries"])

# Sidebar Stats & Details
st.sidebar.header("📊 System Metrics")
stats = st.session_state.metrics.get_stats()
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Cache Hit Rate", stats["cache_hit_rate"])
col_s2.metric("Error Rate", stats["error_rate"])
col_s1.metric("Avg Response", stats["avg_response_time"])
col_s2.metric("Total Tokens", f"{stats['total_tokens']:,}")
st.sidebar.metric("Estimated Cost", stats["estimated_cost"])

cache_stats = st.session_state.query_cache.get_stats()
st.sidebar.metric("Cache Usage", f"{cache_stats['size']}/{cache_stats['max_size']}")

st.sidebar.markdown("---")

# Health Status Panel
st.sidebar.header("🌐 R2 Storage Node Status")
status = st.session_state.health_status
if status["r2_connected"]:
    st.sidebar.success("✅ Connected to Cloudflare R2")
    st.sidebar.markdown(f"**Index Version**: v{status['version']}")
    st.sidebar.markdown(f"**Last Sync**: `{status['last_updated'][:19]}`")
else:
    st.sidebar.warning("⚠️ R2 Disconnected (Offline Mode)")
    
with st.sidebar.expander("ℹ️ Connection Details"):
    st.write(f"Endpoint: `{settings.R2_ENDPOINT or 'None'}`")
    st.write(f"Bucket: `{settings.R2_BUCKET_NAME or 'None'}`")
    st.caption(status["message"])

if st.sidebar.button("🔄 Sync with Cloudflare R2"):
    st.session_state.pop("vector_store", None)
    st.rerun()

st.sidebar.markdown("---")

# Sidebar file uploader
st.sidebar.header("📁 Document Ingestion")
uploaded = st.sidebar.file_uploader(
    "Upload clinical trial files (PDF, DOCX, TXT, CSV, MD, HTML)",
    type=["pdf", "docx", "txt", "csv", "md", "html"],
    accept_multiple_files=True,
)

# Set of processed file names in this session to prevent duplicate spams on rerun
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

if uploaded:
    new_files = [f for f in uploaded if f.name not in st.session_state.processed_files]
    if new_files:
        for f in new_files:
            # Generate unique owner session id for distributed optimistic lock
            owner_id = f"session_{int(time.time())}_{f.name.replace(' ', '_')}"
            
            with st.sidebar.spinner(f"🔒 Acquiring R2 Lock for {f.name}..."):
                # Acquire Lock
                lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=45)
                
            if not lock_acquired:
                st.sidebar.error(f"❌ Locking Conflict: another operation is writing to the index. Skip {f.name}.")
                continue
                
            try:
                # 1. Pull the latest index from R2 to avoid overwriting newer changes (Optimistic Concurrency Control)
                if st.session_state.health_status["r2_connected"]:
                    with st.sidebar.spinner("Syncing latest index..."):
                        try:
                            r2_storage.download_file(
                                f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}",
                                settings.INDEXES_DIR / settings.FAISS_INDEX_FILE
                            )
                            r2_storage.download_file(
                                f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}",
                                settings.INDEXES_DIR / settings.METADATA_PKL_FILE
                            )
                            r2_storage.download_file(
                                f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}",
                                settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE
                            )
                            st.session_state.vector_store.load(str(settings.INDEXES_DIR))
                        except Exception as e:
                            logger.info(f"No active remote index to pull: {e}")
                
                # 2. Process, chunk, embed, append and upload updated index
                with st.sidebar.spinner(f"Ingesting & Embedding {f.name}..."):
                    res = process_and_index_file(f.name, f.getvalue(), st.session_state.vector_store)
                    
                if res["status"] == "success":
                    st.sidebar.success(f"✅ Successfully Indexed {f.name}")
                    # Log execution times to screen
                    timings = res.get("timings", {})
                    with st.sidebar.expander(f"⏱️ Profiling Logs: {f.name}"):
                        for step, duration in timings.items():
                            st.write(f"**{step.replace('_', ' ').title()}**: `{duration:.2f}s`")
                    # Update status
                    doc_meta = get_document_metadata()
                    st.session_state.health_status["version"] = doc_meta.get("version", 0)
                    st.session_state.health_status["last_updated"] = doc_meta.get("last_updated", "Just now")
                elif res["status"] == "skipped":
                    st.sidebar.info(res["message"])
                else:
                    st.sidebar.error(res["message"])
                    
                st.session_state.processed_files.add(f.name)
            finally:
                # Release Distributed Lock
                r2_storage.release_lock(owner_id)

st.sidebar.markdown("### 📄 Active Index Files")
metadata = get_document_metadata()
indexed_docs = metadata.get("documents", {})
if indexed_docs:
    for doc_id, doc in indexed_docs.items():
        st.sidebar.markdown(f"📄 **{doc.get('filename')}**")
        st.sidebar.caption(f"Chunks: {doc.get('chunk_count')} | Hash: `{doc.get('hash')[:8]}...`")
else:
    st.sidebar.caption("No files indexed yet. Upload clinical data above.")

# Search UI Main Section
st.header("🔎 Ask Assistant")
q = st.text_area("Ask a question about your clinical trials / study data standards", height=120, placeholder="What are the inclusion criteria for the studies?")

with st.expander("💡 Example Questions"):
    st.markdown("""
    - What are the main findings of the study?
    - What are the inclusion criteria?
    - What adverse events were reported?
    - Summarize the methodology
    """)

if st.button("🔍 Search Database", type="primary"):
    if not q.strip():
        st.warning("⚠️ Please enter a question first.")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("❌ Missing GOOGLE_API_KEY environment variable. Please set it to connect to Gemini LLM.")
    else:
        # Check if local index has vectors
        if st.session_state.vector_store.index.ntotal == 0:
            st.error("❌ Index is empty. Please upload documents in the sidebar first.")
        else:
            # Wrap local FAISS VectorStore into Retriever Adapter
            retriever = VectorStoreRetrieverAdapter(st.session_state.vector_store, k=settings.RETRIEVER_K)
            try:
                qa = create_qa_from_retriever(retriever)
            except Exception as e:
                st.error(f"❌ QA chain initialization failed: {e}")
                st.stop()
                
            with st.spinner("🧠 Retrieving clinical context and generating answer..."):
                t_search_start = time.time()
                result, was_cached, elapsed = query_with_features(qa, q)
                search_execution_time = time.time() - t_search_start
                logger.info(f"Search execution completed in {search_execution_time:.2f}s")
                
            if result:
                col_r1, col_r2, col_r3 = st.columns([3, 1, 1])
                with col_r1:
                    st.subheader("✨ Response")
                with col_r2:
                    st.metric("Query Execution", f"{elapsed:.2f}s")
                with col_r3:
                    st.metric("Source Node", "🎯 Cache" if was_cached else "🤖 Gemini LLM")
                    
                st.markdown(result.get("result", "").strip() if result.get("result") else "")
                
                st.subheader("📚 Sources Cited")
                uniq = list({d.metadata.get("source", "unknown") for d in result.get("source_documents", [])})
                if uniq:
                    for s in uniq:
                        st.write("📄", s)
                else:
                    st.caption("No explicit sources cited.")
                    
                with st.expander("🔍 View Context Evidence Snippets"):
                    for i, d in enumerate(result.get("source_documents", [])[:6], 1):
                        st.markdown(f"**{i}. {d.metadata.get('source','unknown')}**")
                        # Display chunk score if available
                        score = d.metadata.get("score")
                        if score is not None:
                            st.caption(f"Cosine Similarity Score: `{score:.4f}` | Chunk ID: `{d.metadata.get('chunk_id')}`")
                        st.text(d.page_content[:400].replace('\n', ' '))
                        st.markdown("---")
                        
                st.markdown("### Was this helpful?")
                col_f1, col_f2, col_f3 = st.columns([1, 1, 4])
                with col_f1:
                    if st.button("👍 Yes"):
                        st.success("Thanks for your feedback!")
                with col_f2:
                    if st.button("👎 No"):
                        feedback = st.text_input("What could be improved?")
                        if feedback:
                            st.info("Feedback recorded. Thank you!")
