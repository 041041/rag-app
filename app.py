# app.py - Refactored for Cloudflare R2 and FAISS Integration
import os
import asyncio
import time
import json
import logging
import zipfile
import io
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
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# Import modular components
from config import settings
from storage import r2_storage
from rag.faiss_store import FAISSVectorStore
from rag.retrieval import VectorStoreRetrieverAdapter
from rag.indexing import process_and_index_file, get_document_metadata, save_document_metadata

try:
    from langchain_core.messages import HumanMessage
except ImportError:
    try:
        from langchain.schema import HumanMessage
    except ImportError:
        HumanMessage = None

# Map GEMINI_API_KEY to GOOGLE_API_KEY (LangChain defaults to GOOGLE_API_KEY)
# We prioritize GEMINI_API_KEY if present to avoid conflicts with default system keys
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Ensure an asyncio event loop exists for the current thread (Streamlit-related fix)
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

ensure_event_loop()

# Set page config as the absolute first Streamlit command
st.set_page_config(page_title="Clinical Docs Search (Cloudflare R2 RAG)", layout="wide")

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

# Groq integration fallback
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None


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
        last_raw = None
        errs = []

        # 1. Try invoke method
        inv_fn = getattr(self.llm, "invoke", None)
        if callable(inv_fn):
            # A. Try invoking with a list of HumanMessage (LangChain standard)
            if HumanMessage is not None:
                try:
                    raw = inv_fn([HumanMessage(content=prompt_text)])
                    last_raw = raw
                    if hasattr(raw, "content") and isinstance(getattr(raw, "content"), str):
                        return getattr(raw, "content"), raw, getattr(raw, "source_documents", None) or []
                    text, raw_saved = _extract_text_from_llm_response(raw)
                    return text, raw_saved, getattr(raw, "source_documents", None) or []
                except Exception as e:
                    errs.append(f"invoke(HumanMessage) failed: {e}")
            else:
                errs.append("invoke(HumanMessage) skipped: HumanMessage is None")

            # B. Try invoking with a list of message dicts (fallback)
            try:
                raw = inv_fn([{"role": "user", "content": prompt_text}])
                last_raw = raw
                if hasattr(raw, "content") and isinstance(getattr(raw, "content"), str):
                    return getattr(raw, "content"), raw, getattr(raw, "source_documents", None) or []
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                errs.append(f"invoke(DictList) failed: {e}")

            # C. Try invoking with raw string (fallback)
            try:
                raw = inv_fn(prompt_text)
                last_raw = raw
                if hasattr(raw, "content") and isinstance(getattr(raw, "content"), str):
                    return getattr(raw, "content"), raw, getattr(raw, "source_documents", None) or []
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                errs.append(f"invoke(RawString) failed: {e}")

        # 2. Try generate method with list-of-dicts
        gen_fn = getattr(self.llm, "generate", None)
        if callable(gen_fn):
            try:
                raw = gen_fn([{"content": prompt_text}])
                last_raw = raw
                text, raw_saved = _extract_text_from_llm_response(raw)
                return text, raw_saved, getattr(raw, "source_documents", None) or []
            except Exception as e:
                errs.append(f"generate() failed: {e}")

        # 3. Other callables
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
                errs.append(f"llm.{name}() failed: {e}")
                continue

        raw_type = type(last_raw).__name__ if last_raw is not None else "None"
        raw_preview = repr(last_raw)[:1000] if last_raw is not None else "<no raw captured>"
        detailed_errors = " | ".join(errs)
        raise RuntimeError(f"No LLM invocation succeeded. Errors: {detailed_errors} | raw_type: {raw_type} | raw_preview: {raw_preview}")

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
    has_gemini = bool(os.environ.get("GOOGLE_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    
    if not has_gemini and has_groq:
        if ChatGroq is None:
            raise RuntimeError("ChatGroq (langchain-groq) is not available; install it.")
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-specdec")
        llm = ChatGroq(model_name=model_name, temperature=0.2)
        logger.info(f"Initialized Groq LLM with model: {model_name}")
    else:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("ChatGoogleGenerativeAI (langchain-google-genai) not available; install it.")
        llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0.2, max_output_tokens=1024)
        logger.info(f"Initialized Gemini LLM with model: {settings.LLM_MODEL}")
        
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
                raise RuntimeError(f"All retries failed: {err_str}{debug_hint}")

    return None, False, time.time() - start_time


# ========================================
# APPLICATION STARTUP & INITIALIZATION
# ========================================
@st.cache_resource
def initialize_rag():
    """
    Cached Startup Health Check & Synchronization workflow.
    Guarantees execution exactly once globally across all sessions.
    """
    print("🚀 [STARTUP LOG] [STEP 1] Running initialize_rag()...", flush=True)
    
    # 1. Initialize FAISSVectorStore
    print("🚀 [STARTUP LOG] [STEP 2] Creating FAISSVectorStore...", flush=True)
    store = FAISSVectorStore()
    print("🚀 [STARTUP LOG] [STEP 3] FAISSVectorStore created successfully.", flush=True)
    
    health_status = {
        "r2_connected": False,
        "index_loaded": False,
        "version": 0,
        "last_updated": "Never",
        "message": ""
    }
    
    # 2. Verify R2 connection
    print("🚀 [STARTUP LOG] [STEP 4] Verifying Cloudflare R2 connection...", flush=True)
    r2_connected = r2_storage.verify_connection()
    print(f"🚀 [STARTUP LOG] [STEP 5] R2 verification completed. Result: {r2_connected}", flush=True)
    health_status["r2_connected"] = r2_connected
    
    if r2_connected:
        print("🚀 [STARTUP LOG] [STEP 6] Checking if index files exist on Cloudflare R2...", flush=True)
        index_files_exist = False
        try:
            index_files_exist = (
                r2_storage.check_file_exists(f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}") and
                r2_storage.check_file_exists(f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}") and
                r2_storage.check_file_exists(f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
            )
            print(f"🚀 [STARTUP LOG] [STEP 7] R2 index check result: {index_files_exist}", flush=True)
        except Exception as e:
            print(f"🚀 [STARTUP LOG] [STEP 7 ERROR] Failed to check R2 index availability: {e}", flush=True)
            
        if index_files_exist:
            print("🚀 [STARTUP LOG] [STEP 8] Synchronizing index files from R2 to local container...", flush=True)
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
                print("🚀 [STARTUP LOG] [STEP 9] Loading downloaded index files into FAISSVectorStore...", flush=True)
                store.load(str(settings.INDEXES_DIR))
                health_status["index_loaded"] = True
                
                doc_meta = get_document_metadata()
                health_status["version"] = doc_meta.get("version", 0)
                health_status["last_updated"] = doc_meta.get("last_updated", "Unknown")
                health_status["message"] = f"Synchronized successfully with R2 index (v{health_status['version']})."
                print(f"🚀 [STARTUP LOG] [STEP 10] Successfully loaded index (v{health_status['version']}).", flush=True)
            except Exception as e:
                print(f"🚀 [STARTUP LOG] [STEP 10 ERROR] Failed to synchronize R2 index: {e}. Reinitializing empty index.", flush=True)
                health_status["message"] = f"Failed to load R2 index: {e}."
                store = FAISSVectorStore()
                store.save(str(settings.INDEXES_DIR))
                empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
                save_document_metadata(empty_meta)
        else:
            print("🚀 [STARTUP LOG] [STEP 8] R2 is empty. Initializing brand new local empty index...", flush=True)
            health_status["message"] = "Cloudflare R2 empty. Initialized empty local index."
            store.save(str(settings.INDEXES_DIR))
            empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
            save_document_metadata(empty_meta)
            print("🚀 [STARTUP LOG] [STEP 9] Saved new empty index locally.", flush=True)
    else:
        print("🚀 [STARTUP LOG] [STEP 6] R2 is offline. Checking if local cache files are available for offline fallback...", flush=True)
        health_status["message"] = "Offline mode: R2 connection failed."
        local_exists = (
            (settings.INDEXES_DIR / settings.FAISS_INDEX_FILE).exists() and
            (settings.INDEXES_DIR / settings.METADATA_PKL_FILE).exists() and
            (settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE).exists()
        )
        print(f"🚀 [STARTUP LOG] [STEP 7] Local cache files exist offline: {local_exists}", flush=True)
        if local_exists:
            try:
                store.load(str(settings.INDEXES_DIR))
                health_status["index_loaded"] = True
                doc_meta = get_document_metadata()
                health_status["version"] = doc_meta.get("version", 0)
                health_status["last_updated"] = doc_meta.get("last_updated", "Unknown")
                print(f"🚀 [STARTUP LOG] [STEP 8] Loaded local offline cache (v{health_status['version']}).", flush=True)
            except Exception as e:
                print(f"🚀 [STARTUP LOG] [STEP 8 ERROR] Failed to load offline cache: {e}. Reinitializing empty index.", flush=True)
                store = FAISSVectorStore()
                store.save(str(settings.INDEXES_DIR))
                empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
                save_document_metadata(empty_meta)
        else:
            print("🚀 [STARTUP LOG] [STEP 8] Local files do not exist either. Saving new empty index...", flush=True)
            store.save(str(settings.INDEXES_DIR))
            empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
            save_document_metadata(empty_meta)
            print("🚀 [STARTUP LOG] [STEP 9] Saved new local empty index.", flush=True)
            
    print("🚀 [STARTUP LOG] [STEP 11] initialize_rag() completed.", flush=True)
    return store, health_status

# Startup initialization will run lazily in the main UI flow below

@st.cache_resource
def get_file_bytes(filename: str) -> bytes:
    """
    Downloads file from Cloudflare R2 (if missing locally) and returns file bytes.
    """
    local_path = settings.DATA_DIR / filename
    if not local_path.exists():
        r2_key = f"{settings.R2_DOCUMENTS_PREFIX}{filename}"
        r2_storage.download_file(r2_key, local_path)
    if local_path.exists():
        return local_path.read_bytes()
    return b""

@st.dialog("🗑️ Delete Documents Confirmation")
def confirm_bulk_delete_dialog(doc_ids: List[str], filenames: List[str]):
    """
    Modal confirmation popup for bulk document deletion (Requirement 8).
    """
    count = len(doc_ids)
    st.write(f"Are you sure you want to permanently delete **{count}** selected documents from the index and Cloudflare R2?")
    st.warning("⚠️ This action cannot be undone and will remove all corresponding vector chunks.")
    
    with st.expander("Show files being deleted"):
        for f in filenames:
            st.write(f"- {f}")
            
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Cancel", key="cancel_bulk_delete_btn", use_container_width=True):
            st.rerun()
    with col_c2:
        if st.button("Delete Permanent", key="confirm_bulk_delete_btn", type="primary", use_container_width=True):
            for d_id in doc_ids:
                delete_document_workflow(d_id)
            st.session_state.selected_docs = set()
            st.session_state.selected_detail_doc = None
            st.rerun()

@st.dialog("🗑️ Delete Document Confirmation")
def confirm_delete_dialog(doc_id: str, filename: str):
    """
    Modal confirmation popup for single document deletion (Requirement 8).
    """
    st.write(f"Are you sure you want to permanently delete **{filename}** from the index and Cloudflare R2?")
    st.warning("⚠️ This action cannot be undone and will remove all corresponding vector chunks.")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Cancel", key="cancel_delete_btn", use_container_width=True):
            st.rerun()
    with col_c2:
        if st.button("Delete Permanent", key="confirm_delete_btn", type="primary", use_container_width=True):
            delete_document_workflow(doc_id)
            st.session_state.selected_docs.discard(filename)
            if st.session_state.selected_detail_doc and st.session_state.selected_detail_doc.get("filename") == filename:
                st.session_state.selected_detail_do@st.dialog("🗄️ Document Management Portal", width="large")
def document_management_dialog():
    """
    Redesigned enterprise Document Management Portal modal dialog (Requirement 5).
    """
    # Initialize temporary selection caching (Google Drive style picker)
    if st.session_state.get("dialog_init_needed", True) or "temp_selected_docs" not in st.session_state:
        st.session_state.temp_selected_docs = set(st.session_state.selected_docs)
        st.session_state.dialog_init_needed = False
        
    # Read metadata database
    metadata = get_document_metadata()
    indexed_docs = metadata.get("documents", {})
    
    # Inline Bulk Delete Confirmation (Requirement 8 - no nesting dialogs)
    if st.session_state.get("show_bulk_delete_confirm"):
        selected_ids = []
        selected_names = []
        for doc_id, doc in indexed_docs.items():
            if doc.get("filename") in st.session_state.temp_selected_docs:
                selected_ids.append(doc_id)
                selected_names.append(doc.get("filename"))
                
        st.markdown("### 🗑️ Bulk Delete Confirmation")
        count = len(selected_ids)
        st.write(f"Are you sure you want to permanently delete **{count}** selected documents from the index and Cloudflare R2?")
        st.warning("⚠️ This action cannot be undone and will remove all corresponding vector chunks.")
        
        with st.expander("Show files being deleted"):
            for f in selected_names:
                st.write(f"- {f}")
                
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Cancel Deletion", key="cancel_bulk_delete_btn", use_container_width=True):
                st.session_state.show_bulk_delete_confirm = False
                st.rerun()
        with col_c2:
            if st.button("Delete Permanent", key="confirm_bulk_delete_btn", type="primary", use_container_width=True):
                for d_id in selected_ids:
                    delete_document_workflow(d_id)
                # Clear from both temporary and main selection lists
                st.session_state.temp_selected_docs = set()
                st.session_state.selected_docs = set()
                # Clear checkbox keys
                for k in list(st.session_state.keys()):
                    if k.startswith("chk_"):
                        st.session_state[k] = False
                st.session_state.selected_detail_doc = None
                st.session_state.show_bulk_delete_confirm = False
                st.rerun()
        return

    st.markdown(
        "<p style='color: #CBD5E1 !important; font-size: 16px !important; font-weight: 400 !important; opacity: 1 !important; margin-bottom: 20px;'>Redesigned clinical document repository for enterprise scaling (1,500+ files).</p>",
        unsafe_allow_html=True
    )
    
    # Render compact filter controls in a single columns block (Requirement 2)
    col_f1, col_f2, col_f3, col_f4 = st.columns([45, 18, 18, 19])
    with col_f1:
        st.markdown("<p style='font-size: 0.8em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; opacity: 1;'>Search Documents</p>", unsafe_allow_html=True)
        doc_search = st.text_input("Search filename", placeholder="Search documents by filename...", value=st.session_state.get("doc_search_filter", ""), key="doc_search_modal_input", label_visibility="collapsed")
    with col_f2:
        st.markdown("<p style='font-size: 0.8em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; opacity: 1;'>Status</p>", unsafe_allow_html=True)
        status_filter = st.selectbox("Status", ["All", "Indexed", "Processing", "Failed"], index=["All", "Indexed", "Processing", "Failed"].index(st.session_state.get("doc_status_filter", "All")), key="doc_status_modal_sel", label_visibility="collapsed")
    with col_f3:
        st.markdown("<p style='font-size: 0.8em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; opacity: 1;'>File Type</p>", unsafe_allow_html=True)
        type_filter = st.selectbox("File Type", ["All", "PDF", "DOCX", "TXT", "CSV"], index=["All", "PDF", "DOCX", "TXT", "CSV"].index(st.session_state.get("doc_type_filter", "All")), key="doc_type_modal_sel", label_visibility="collapsed")
    with col_f4:
        st.markdown("<p style='font-size: 0.8em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px; opacity: 1;'>Sort By</p>", unsafe_allow_html=True)
        sort_filter = st.selectbox("Sort", ["Newest", "Oldest", "A-Z", "Z-A"], index=["Newest", "Oldest", "A-Z", "Z-A"].index(st.session_state.get("doc_sort_filter", "Newest")), key="doc_sort_modal_sel", label_visibility="collapsed")
        
    st.session_state.doc_search_filter = doc_search
    st.session_state.doc_status_filter = status_filter
    st.session_state.doc_type_filter = type_filter
    st.session_state.doc_sort_filter = sort_filter
    
    # Filter documents
    filtered_docs_list = []
    if indexed_docs:
        for doc_id, doc in indexed_docs.items():
            fname = doc.get("filename", "")
            # 1. Search text filter
            if doc_search.strip() and doc_search.lower() not in fname.lower():
                continue
                
            # 2. Status filter
            if status_filter != "All":
                status = doc.get("status", "Indexed")
                if status.lower() != status_filter.lower():
                    continue
                    
            # 3. File Type filter
            if type_filter != "All":
                ext = fname.split(".")[-1].lower()
                if type_filter.lower() != ext:
                    continue
                    
            filtered_docs_list.append((doc_id, doc))
            
    # Sort documents
    if sort_filter == "Newest":
        filtered_docs_list.sort(key=lambda x: x[1].get("timestamp", ""), reverse=True)
    elif sort_filter == "Oldest":
        filtered_docs_list.sort(key=lambda x: x[1].get("timestamp", ""), reverse=False)
    elif sort_filter == "A-Z":
        filtered_docs_list.sort(key=lambda x: x[1].get("filename", "").lower(), reverse=False)
    elif sort_filter == "Z-A":
        filtered_docs_list.sort(key=lambda x: x[1].get("filename", "").lower(), reverse=True)
        
    st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'/>", unsafe_allow_html=True)
    
    total_documents = len(filtered_docs_list)
    
    # Pagination calculations (50 documents per page)
    num_pages = max(1, (total_documents - 1) // 50 + 1)
    current_page = min(st.session_state.get("doc_page", 1), num_pages)
    st.session_state.doc_page = current_page
    start_index = (current_page - 1) * 50
    end_index = min(start_index + 50, total_documents)
    page_docs = filtered_docs_list[start_index:end_index]
    
    total_docs_count = len(indexed_docs)
    filtered_docs_count = len(filtered_docs_list)
    selected_docs_count = len(st.session_state.temp_selected_docs)
    
    st.markdown(f"""
    <div style='font-size: 0.85em; opacity: 0.85; margin-bottom: 8px; font-weight: 500;'>
        📊 Repository Stats: 
        <span style='color: #cbd5e1; font-weight: 600;'>Total Documents:</span> {total_docs_count} | 
        <span style='color: #cbd5e1; font-weight: 600;'>Filtered Results:</span> {filtered_docs_count} | 
        <span style='color: #38bdf8; font-weight: 700;'>Selected Documents:</span> {selected_docs_count}
    </div>
    """, unsafe_allow_html=True)
    
    col_table, col_details = st.columns([7, 5])
    with col_table:
        # Floating Bulk Action Toolbar (Requirement 5, 6, 7)
        if len(st.session_state.temp_selected_docs) > 0:
            selected_count = len(st.session_state.temp_selected_docs)
            sel_label = f"📄 {selected_count} Document Selected" if selected_count == 1 else f"📄 {selected_count} Documents Selected"
            
            # Render compact toolbar in a single horizontal row (Requirement 5)
            col_b_sel, col_b1, col_b2, col_b3, col_b4 = st.columns([2.8, 1.2, 1.2, 1.2, 1.2])
            with col_b_sel:
                st.markdown(f"<p style='font-size: 0.9em; font-weight: 700; color: #38bdf8; margin-top: 6px; margin-bottom: 0;'>{sel_label}</p>", unsafe_allow_html=True)
            with col_b1:
                if st.button("🗑 Delete", type="primary", key="bulk_delete_action_btn", use_container_width=True):
                    st.session_state.show_bulk_delete_confirm = True
            with col_b2:
                # Zip and Download selected documents
                selected_names_list = list(st.session_state.temp_selected_docs)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for doc_name in selected_names_list:
                        file_bytes = get_file_bytes(doc_name)
                        if file_bytes:
                            zip_file.writestr(doc_name, file_bytes)
                zip_data = zip_buffer.getvalue()
                
                st.download_button(
                    label="⬇ Download",
                    data=zip_data,
                    file_name="selected_clinical_documents.zip",
                    key="bulk_download_action_btn",
                    use_container_width=True
                )
            with col_b3:
                if st.button("🔄 Re-index", key="bulk_reindex_action_btn", use_container_width=True):
                    with st.spinner("Re-indexing..."):
                        for doc_name in st.session_state.temp_selected_docs:
                            file_bytes = get_file_bytes(doc_name)
                            if file_bytes:
                                process_and_index_file(doc_name, file_bytes, st.session_state.vector_store)
                    st.success("Re-indexed!")
            with col_b4:
                if st.button("✖ Clear Selection", key="bulk_clear_action_btn", use_container_width=True):
                    st.session_state.temp_selected_docs.clear()
                    for k in list(st.session_state.keys()):
                        if k.startswith("chk_"):
                            st.session_state[k] = False
            st.markdown("<hr style='margin: 8px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'/>", unsafe_allow_html=True)
                    
        # Table Headers with Select All (Requirement 8)
        page_docs_filenames = [doc.get("filename") for doc_id, doc in page_docs]
        all_selected = all(f in st.session_state.temp_selected_docs for f in page_docs_filenames) if page_docs_filenames else False
        
        col_hdr_chk, col_hdr_status, col_hdr_size, col_hdr_chunks = st.columns([6, 2, 2, 2])
        with col_hdr_chk:
            # Removed standard widget key to prevent state caching conflicts
            select_all = st.checkbox("Select All", value=all_selected)
            if select_all != all_selected:
                if select_all:
                    for f in page_docs_filenames:
                        st.session_state.temp_selected_docs.add(f)
                else:
                    for f in page_docs_filenames:
                        st.session_state.temp_selected_docs.discard(f)
                
        st.markdown("""
        <div style='display: flex; font-weight: bold; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 4px; margin-bottom: 6px; font-size: 0.8em; opacity: 0.9;'>
            <div style='flex: 6;'>Document</div>
            <div style='flex: 2; text-align: center;'>Status</div>
            <div style='flex: 2; text-align: center;'>Size</div>
            <div style='flex: 2; text-align: center; padding-right: 5px;'>Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
        if page_docs:
            for doc_id, doc in page_docs:
                doc_name = doc.get("filename")
                size = doc.get("file_size_kb")
                if size is not None:
                    size_str = f"{size/1024:.1f} MB" if size > 1024 else f"{size:.0f} KB"
                else:
                    size_str = "Unknown"
                    
                status = doc.get("status", "Indexed")
                if status == "Indexed":
                    status_str = "🟢 Indexed"
                elif status == "Processing":
                    status_str = "🟡 Processing"
                else:
                    status_str = "🔴 Failed"
                    
                chunks_count = doc.get("chunk_count", 0)
                chunks_str = f"{chunks_count}"
                
                # Highlight active row in table via filename pointer (Requirement 10)
                is_active = False
                selected_detail_doc = st.session_state.get("selected_detail_doc")
                if selected_detail_doc and selected_detail_doc.get("filename") == doc_name:
                    is_active = True
                doc_display_name = f"👉 📄 {doc_name}" if is_active else f"📄 {doc_name}"
                
                col_row_chk, col_row_name, col_row_status, col_row_size, col_row_chunks = st.columns([1, 5, 2, 2, 2])
                with col_row_chk:
                    doc_checked = st.checkbox("", value=(doc_name in st.session_state.temp_selected_docs), key=f"chk_{doc_id}", label_visibility="collapsed")
                    if doc_checked != (doc_name in st.session_state.temp_selected_docs):
                        if doc_checked:
                            st.session_state.temp_selected_docs.add(doc_name)
                        else:
                            st.session_state.temp_selected_docs.discard(doc_name)
                with col_row_name:
                    if st.button(doc_display_name, key=f"btn_detail_name_{doc_id}", use_container_width=True):
                        st.session_state.selected_detail_doc = doc
                with col_row_status:
                    if st.button(f"{status_str}", key=f"btn_detail_status_{doc_id}", use_container_width=True):
                        st.session_state.selected_detail_doc = doc
                with col_row_size:
                    if st.button(f"{size_str}", key=f"btn_detail_size_{doc_id}", use_container_width=True):
                        st.session_state.selected_detail_doc = doc
                with col_row_chunks:
                    if st.button(f"{chunks_str}", key=f"btn_detail_chunks_{doc_id}", use_container_width=True):
                        st.session_state.selected_detail_doc = doc
                    
                st.markdown("<hr style='margin: 2px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.03);'/>", unsafe_allow_html=True)
        else:
            st.info("No matching documents found.")
            
    with col_details:
        detail_doc = st.session_state.get("selected_detail_doc")
        if detail_doc:
            dfname = detail_doc.get("filename")
            st.markdown(f"### 📋 File Details")
            st.markdown(f"**Document Name**: `{dfname}`")
            
            status = detail_doc.get("status", "Indexed")
            if status == "Indexed":
                status_str = "🟢 Indexed"
            elif status == "Processing":
                status_str = "🟡 Processing"
            else:
                status_str = "🔴 Failed"
            st.markdown(f"**Status**: `{status_str}`")
            
            dsize = detail_doc.get("file_size_kb")
            if dsize is not None:
                dsize_str = f"{dsize/1024:.1f} MB" if dsize > 1024 else f"{dsize:.0f} KB"
            else:
                dsize_str = "Unknown"
            st.markdown(f"**Size**: `{dsize_str}`")
            
            dext = dfname.split(".")[-1].upper()
            st.markdown(f"**File Type**: `{dext}`")
            
            pages_val = detail_doc.get("pages") or detail_doc.get("metadata", {}).get("pages") or "N/A"
            st.markdown(f"**Pages**: `{pages_val}`")
            st.markdown(f"**Chunks**: `{detail_doc.get('chunk_count', 0)} Chunks`")
            
            st.markdown(f"**Uploaded Date**: `{detail_doc.get('timestamp', 'Unknown')[:16].replace('T', ' ')}`")
            st.markdown(f"**Last Indexed**: `{detail_doc.get('timestamp', 'Unknown')[:16].replace('T', ' ')}`")
            
            # Additional metadata dictionary
            st.markdown("**Metadata**:")
            st.json(detail_doc.get("metadata", {}))
            
            st.markdown("---")
            # Inline Single Delete Confirmation (Requirement 8 - no nesting dialogs)
            if st.session_state.get("show_single_delete_confirm") == dfname:
                st.markdown("#### 🗑️ Confirm File Deletion")
                st.write(f"Are you sure you want to permanently delete **{dfname}**?")
                st.warning("⚠️ This action cannot be undone.")
                col_sd1, col_sd2 = st.columns(2)
                with col_sd1:
                    if st.button("Cancel", key="cancel_single_delete_btn", use_container_width=True):
                        st.session_state.show_single_delete_confirm = None
                with col_sd2:
                    if st.button("Delete Permanent", key="confirm_single_delete_btn", type="primary", use_container_width=True):
                        # Find doc_id
                        ddoc_id = None
                        for d_id, d in indexed_docs.items():
                            if d.get("filename") == dfname:
                                ddoc_id = d_id
                                break
                        if ddoc_id:
                            delete_document_workflow(ddoc_id)
                            st.session_state.temp_selected_docs.discard(dfname)
                            st.session_state.selected_docs.discard(dfname)
                            st.session_state[f"chk_{ddoc_id}"] = False
                            st.session_state.selected_detail_doc = None
                        st.session_state.show_single_delete_confirm = None
            else:
                # Action Buttons inside detail panel
                file_bytes = get_file_bytes(dfname)
                st.download_button(
                    label="📥 Download Original",
                    data=file_bytes,
                    file_name=dfname,
                    key="detail_download_action_btn",
                    use_container_width=True
                )
                
                if st.button("🔄 Re-index File", key="detail_reindex_action_btn", use_container_width=True):
                    with st.spinner("Re-indexing..."):
                        process_and_index_file(dfname, file_bytes, st.session_state.vector_store)
                    st.success("Re-indexed successfully!")
                    
                if st.button("🗑️ Delete File", key="detail_delete_action_btn", type="primary", use_container_width=True):
                    st.session_state.show_single_delete_confirm = dfname
        else:
            st.info("ℹ️ Select a document row to view detailed metadata and file actions.")
            
    # Admin Controls inside modal
    with st.expander("🛠️ Administrative Controls"):
        st.warning("⚠️ Warning: Rebuilding the index will permanently clear all vectors and delete files from persistent storage.")
        if st.button("⚙️ Rebuild / Clear Index", type="secondary", key="admin_rebuild_modal_btn", use_container_width=True):
            rebuild_empty_index_workflow()
            
    # Dialog Footer Controls (Requirement 7)
    st.markdown("<hr style='margin: 10px 0 15px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'/>", unsafe_allow_html=True)
    col_foot_left, col_foot_right = st.columns([7, 5])
    
    with col_foot_left:
        col_p_text, col_p_prev, col_p_next = st.columns([5, 2.5, 2.5])
        with col_p_text:
            st.markdown(f"<p style='font-size: 0.85em; opacity: 0.85; margin-top: 8px;'>Showing <b>{start_index + 1}–{end_index}</b> of <b>{total_documents}</b> Documents</p>", unsafe_allow_html=True)
        with col_p_prev:
            if st.button("Previous", disabled=(current_page == 1), key="btn_page_prev", use_container_width=True):
                st.session_state.doc_page = max(1, current_page - 1)
        with col_p_next:
            if st.button("Next", disabled=(current_page >= num_pages), key="btn_page_next", use_container_width=True):
                st.session_state.doc_page = min(num_pages, current_page + 1)
                
    with col_foot_right:
        col_f_cancel, col_f_done = st.columns([1, 1])
        with col_f_cancel:
            if st.button("Cancel", key="btn_portal_cancel", use_container_width=True):
                # Discard temporary selection state
                st.session_state.temp_selected_docs = set(st.session_state.selected_docs)
                st.session_state.show_doc_manager_dialog = False
                st.rerun() # Closes the dialog modal
        with col_f_done:
            if st.button("Done", key="btn_portal_done", type="primary", use_container_width=True):
                # Commit temporary selection to main state
                st.session_state.selected_docs = set(st.session_state.temp_selected_docs)
                st.session_state.show_doc_manager_dialog = False
                st.rerun() # Closes the dialog modal and triggers parent page redraw

def sync_index_if_version_changed():
    """
    Checks if the remote R2 index version is newer than the local loaded version.
    If newer, downloads and reloads the index.
    """
    if "health_status" not in st.session_state or not st.session_state.health_status.get("r2_connected"):
        return
        
    import time
    curr_time = time.time()
    last_check = st.session_state.get("last_r2_sync_check_time", 0.0)
    if curr_time - last_check < 60.0:
        return
    st.session_state.last_r2_sync_check_time = curr_time
        
    print("🔄 Checking if remote R2 index version changed...", flush=True)
    try:
        temp_meta_path = settings.INDEXES_DIR / "temp_document_metadata.json"
        # Download document_metadata.json from R2 to get the remote version
        download_success = r2_storage.download_file(
            f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}",
            temp_meta_path
        )
        if not download_success:
            print("🔄 Remote metadata not found on R2.", flush=True)
            return
            
        import json
        if temp_meta_path.exists():
            with open(temp_meta_path, 'r', encoding='utf-8') as f:
                remote_meta = json.load(f)
                
            remote_version = remote_meta.get("version", 0)
            local_version = st.session_state.health_status["version"]
            
            if remote_version > local_version:
                print(f"🔄 Remote index version (v{remote_version}) is newer than local version (v{local_version}). Syncing...", flush=True)
                with st.spinner(f"🔄 Syncing newer R2 index version (v{remote_version})..."):
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}",
                        settings.INDEXES_DIR / settings.FAISS_INDEX_FILE
                    )
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}",
                        settings.INDEXES_DIR / settings.METADATA_PKL_FILE
                    )
                    import shutil
                    shutil.copy2(temp_meta_path, settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE)
                    
                    st.session_state.vector_store.load(str(settings.INDEXES_DIR))
                    st.session_state.health_status["version"] = remote_version
                    st.session_state.health_status["last_updated"] = remote_meta.get("last_updated", "Just now")
                    st.session_state.health_status["index_loaded"] = True
                    print(f"✅ Synced index successfully to v{remote_version}", flush=True)
            else:
                print(f"🔄 Index is up to date (local: v{local_version}, remote: v{remote_version})", flush=True)
    except Exception as e:
        print(f"⚠️ Error checking R2 version: {e}", flush=True)

def delete_document_workflow(doc_id: str):
    owner_id = f"delete_session_{int(time.time())}_{doc_id}"
    with st.spinner("🗑️ Acquiring R2 Lock & deleting document..."):
        lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=45)
        if not lock_acquired:
            st.error("❌ Locking Conflict: another operation is writing. Please try again.")
            return
            
        try:
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
                logger.info(f"No index to pull: {e}")
                
            metadata = get_document_metadata()
            if doc_id not in metadata.get("documents", {}):
                st.error("❌ Document not found in index.")
                return
                
            doc_info = metadata["documents"][doc_id]
            filename = doc_info["filename"]
            chunk_ids = doc_info.get("chunk_ids", [])
            
            if chunk_ids:
                import numpy as np
                ids_to_remove = np.array(chunk_ids, dtype=np.int64)
                removed_count = st.session_state.vector_store.index.remove_ids(ids_to_remove)
                print(f"🗑️ Removed {removed_count} vectors from FAISS index.", flush=True)
                
            del metadata["documents"][doc_id]
            metadata["version"] = metadata.get("version", 0) + 1
            metadata["last_updated"] = datetime.now().isoformat()
            
            st.session_state.vector_store.save(str(settings.INDEXES_DIR))
            save_document_metadata(metadata)
            
            r2_storage.backup_indexes()
            r2_storage.upload_file(settings.INDEXES_DIR / settings.FAISS_INDEX_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.METADATA_PKL_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
            
            r2_key = doc_info.get("r2_path", f"{settings.R2_DOCUMENTS_PREFIX}{filename}")
            r2_storage.delete_file(r2_key)
            
            if filename in st.session_state.processed_files:
                st.session_state.processed_files.remove(filename)
                
            st.session_state.health_status["version"] = metadata["version"]
            st.session_state.health_status["last_updated"] = metadata["last_updated"]
            
            st.success(f"🗑️ Successfully deleted {filename} from index and Cloudflare R2.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to delete document: {e}")
        finally:
            r2_storage.release_lock(owner_id)

def rebuild_empty_index_workflow():
    owner_id = f"rebuild_session_{int(time.time())}"
    with st.spinner("⚙️ Rebuilding empty index on Cloudflare R2..."):
        lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=45)
        if not lock_acquired:
            st.error("❌ Locking Conflict. Please try again.")
            return
        try:
            store = FAISSVectorStore()
            store.save(str(settings.INDEXES_DIR))
            
            empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
            save_document_metadata(empty_meta)
            
            r2_storage.upload_file(settings.INDEXES_DIR / settings.FAISS_INDEX_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.METADATA_PKL_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
            
            st.session_state.vector_store = store
            st.session_state.health_status["version"] = 0
            st.session_state.health_status["last_updated"] = empty_meta["last_updated"]
            st.session_state.processed_files.clear()
            
            st.success("⚙️ Successfully rebuilt index! All indices cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Rebuild failed: {e}")
        finally:
            r2_storage.release_lock(owner_id)

# Startup initialization will run lazily in the main UI flow below

def sync_index_if_version_changed():
    """
    Checks if the remote R2 index version is newer than the local loaded version.
    If newer, downloads and reloads the index.
    """
    if "health_status" not in st.session_state or not st.session_state.health_status.get("r2_connected"):
        return
        
    print("🔄 Checking if remote R2 index version changed...", flush=True)
    try:
        temp_meta_path = settings.INDEXES_DIR / "temp_document_metadata.json"
        # Download document_metadata.json from R2 to get the remote version
        download_success = r2_storage.download_file(
            f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}",
            temp_meta_path
        )
        if not download_success:
            print("🔄 Remote metadata not found on R2.", flush=True)
            return
            
        import json
        if temp_meta_path.exists():
            with open(temp_meta_path, 'r', encoding='utf-8') as f:
                remote_meta = json.load(f)
                
            remote_version = remote_meta.get("version", 0)
            local_version = st.session_state.health_status["version"]
            
            if remote_version > local_version:
                print(f"🔄 Remote index version (v{remote_version}) is newer than local version (v{local_version}). Syncing...", flush=True)
                with st.spinner(f"🔄 Syncing newer R2 index version (v{remote_version})..."):
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}",
                        settings.INDEXES_DIR / settings.FAISS_INDEX_FILE
                    )
                    r2_storage.download_file(
                        f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}",
                        settings.INDEXES_DIR / settings.METADATA_PKL_FILE
                    )
                    import shutil
                    shutil.copy2(temp_meta_path, settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE)
                    
                    st.session_state.vector_store.load(str(settings.INDEXES_DIR))
                    st.session_state.health_status["version"] = remote_version
                    st.session_state.health_status["last_updated"] = remote_meta.get("last_updated", "Just now")
                    st.session_state.health_status["index_loaded"] = True
                    print(f"✅ Synced index successfully to v{remote_version}", flush=True)
            else:
                print(f"🔄 Index is up to date (local: v{local_version}, remote: v{remote_version})", flush=True)
    except Exception as e:
        print(f"⚠️ Error checking R2 version: {e}", flush=True)

def delete_document_workflow(doc_id: str):
    owner_id = f"delete_session_{int(time.time())}_{doc_id}"
    with st.spinner("🗑️ Acquiring R2 Lock & deleting document..."):
        lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=45)
        if not lock_acquired:
            st.error("❌ Locking Conflict: another operation is writing. Please try again.")
            return
            
        try:
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
                logger.info(f"No index to pull: {e}")
                
            metadata = get_document_metadata()
            if doc_id not in metadata.get("documents", {}):
                st.error("❌ Document not found in index.")
                return
                
            doc_info = metadata["documents"][doc_id]
            filename = doc_info["filename"]
            chunk_ids = doc_info.get("chunk_ids", [])
            
            if chunk_ids:
                import numpy as np
                ids_to_remove = np.array(chunk_ids, dtype=np.int64)
                removed_count = st.session_state.vector_store.index.remove_ids(ids_to_remove)
                print(f"🗑️ Removed {removed_count} vectors from FAISS index.", flush=True)
                
            del metadata["documents"][doc_id]
            metadata["version"] = metadata.get("version", 0) + 1
            metadata["last_updated"] = datetime.now().isoformat()
            
            st.session_state.vector_store.save(str(settings.INDEXES_DIR))
            save_document_metadata(metadata)
            
            r2_storage.backup_indexes()
            r2_storage.upload_file(settings.INDEXES_DIR / settings.FAISS_INDEX_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.METADATA_PKL_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
            
            r2_key = doc_info.get("r2_path", f"{settings.R2_DOCUMENTS_PREFIX}{filename}")
            r2_storage.delete_file(r2_key)
            
            if filename in st.session_state.processed_files:
                st.session_state.processed_files.remove(filename)
                
            st.session_state.health_status["version"] = metadata["version"]
            st.session_state.health_status["last_updated"] = metadata["last_updated"]
            
            st.success(f"🗑️ Successfully deleted {filename} from index and Cloudflare R2.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to delete document: {e}")
        finally:
            r2_storage.release_lock(owner_id)

def rebuild_empty_index_workflow():
    owner_id = f"rebuild_session_{int(time.time())}"
    with st.spinner("⚙️ Rebuilding empty index on Cloudflare R2..."):
        lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=45)
        if not lock_acquired:
            st.error("❌ Locking Conflict. Please try again.")
            return
        try:
            store = FAISSVectorStore()
            store.save(str(settings.INDEXES_DIR))
            
            empty_meta = {"version": 0, "last_updated": datetime.now().isoformat(), "documents": {}}
            save_document_metadata(empty_meta)
            
            r2_storage.upload_file(settings.INDEXES_DIR / settings.FAISS_INDEX_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.METADATA_PKL_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
            
            st.session_state.vector_store = store
            st.session_state.health_status["version"] = 0
            st.session_state.health_status["last_updated"] = empty_meta["last_updated"]
            st.session_state.processed_files.clear()
            
            st.success("⚙️ Successfully rebuilt index! All indices cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Rebuild failed: {e}")
        finally:
            r2_storage.release_lock(owner_id)


# ========================================
# STREAMLIT UI - RICH DESIGN AESTHETICS
# ========================================

# Inject Custom Elegant Styling for Premium Aesthetics
st.markdown("""
<style>
    /* Styling headers & cards */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 25, 41) 0%, rgb(10, 12, 18) 90%);
    }
    header, [data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgb(15, 23, 42) 0%, rgb(10, 12, 18) 100%) !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stMarkdown {
        color: #f0f3f9;
    }
    .stApp textarea, .stApp input, .stApp select {
        color: #0f172a !important;
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
    .stButton>button, .stDownloadButton>button, div[data-testid="stDownloadButton"]>button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover, div[data-testid="stDownloadButton"]>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        color: white !important;
    }
    .stAlert {
        border-radius: 8px;
    }
    /* Drag-and-drop uploader modern card style */
    div[data-testid="stFileUploader"] {
        background-color: rgba(30, 41, 59, 0.45) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    /* Ask AI centered button size */
    .stApp div.row-widget.stButton > button[kind="primary"] {
        width: 240px !important;
        height: 50px !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 12px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stApp div.row-widget.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    /* Style the sidebar secondary buttons (Manage Documents) */
    [data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        color: #f0f3f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    /* Theme-aware Modal Dialogs / Backdrops */
    [data-testid="stDialog"] {
        background-color: rgba(10, 12, 18, 0.75) !important;
        backdrop-filter: blur(6px) !important;
    }
    div[role="dialog"], [data-testid="stDialog"] [role="dialog"], [data-testid="stModal"] > div {
        background-color: rgb(15, 23, 42) !important;
        background: rgb(15, 23, 42) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5) !important;
    }
    div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] h4, 
    div[role="dialog"] p, div[role="dialog"] label, div[role="dialog"] span, div[role="dialog"] div, 
    div[role="dialog"] small, div[role="dialog"] th, div[role="dialog"] td {
        color: #f0f3f9 !important;
    }
    div[role="dialog"] button {
        color: white !important;
    }
    /* Expander summary text color inside the modal (Requirement 2) */
    div[role="dialog"] div[data-testid="stExpander"] summary p {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    div[role="dialog"] div[data-testid="stExpander"] summary svg {
        fill: #F8FAFC !important;
    }
    /* Form inputs inside modal aligned to premium dark theme (Notion/Sharepoint mode) */
    div[role="dialog"] input, 
    div[role="dialog"] textarea, 
    div[role="dialog"] select,
    div[role="dialog"] div[data-basewidget="select"] > div {
        color: #f8fafc !important;
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px !important;
    }
    div[data-testid="stPopoverBody"] {
        background-color: rgb(30, 41, 59) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stPopoverBody"] h1, div[data-testid="stPopoverBody"] h2, div[data-testid="stPopoverBody"] h3, div[data-testid="stPopoverBody"] p, div[data-testid="stPopoverBody"] label, div[data-testid="stPopoverBody"] span, div[data-testid="stPopoverBody"] div, div[data-testid="stPopoverBody"] button {
        color: #f0f3f9 !important;
        fill: #f0f3f9 !important;
    }
    /* Style buttons inside st.expander to look like clickable list items */
    div[data-testid="stExpander"] button {
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        padding: 0 !important;
        text-align: left !important;
        font-size: 0.95em !important;
        margin-bottom: 8px !important;
        display: block !important;
    }
    div[data-testid="stExpander"] button:hover {
        color: #38bdf8 !important;
        text-decoration: underline !important;
    }
    /* Increase visibility and typography hierarchy of dialog title */
    div[role="dialog"] [data-testid="stHeading"] h2, 
    div[role="dialog"] h2 {
        color: #f8fafc !important;
        font-size: 2.2em !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px !important;
        margin-bottom: 5px !important;
    }
    
    /* Style all secondary buttons inside the document portal table as clean row links */
    div[role="dialog"] button[data-basebuttonstyle="secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #f1f5f9 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 2px 6px !important;
        font-size: 0.85em !important;
        line-height: 1.2 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        height: 26px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    div[role="dialog"] button[data-basebuttonstyle="secondary"]:hover {
        color: #38bdf8 !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Clean centered text for Status, Size, and Chunks columns (Google Drive style) */
    div[role="dialog"] div[data-testid="column"]:nth-of-type(3) button,
    div[role="dialog"] div[data-testid="column"]:nth-of-type(4) button,
    div[role="dialog"] div[data-testid="column"]:nth-of-type(5) button {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        text-align: center !important;
        justify-content: center !important;
        font-weight: 400 !important;
        box-shadow: none !important;
    }
    div[role="dialog"] div[data-testid="column"]:nth-of-type(3) button:hover,
    div[role="dialog"] div[data-testid="column"]:nth-of-type(4) button:hover,
    div[role="dialog"] div[data-testid="column"]:nth-of-type(5) button:hover {
        color: #38bdf8 !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        text-decoration: underline !important;
    }
    
    /* Style all dialog buttons to have uniform premium heights and widths */
    div[role="dialog"] button {
        height: 32px !important;
        line-height: 32px !important;
        padding: 0 10px !important;
        font-size: 0.85em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Make document column alignment compact like a list directory (Google Drive style) */
    div[role="dialog"] div[data-testid="column"] {
        padding: 0px !important;
        margin: 0px !important;
    }
    
    /* Vertically center checkbox column contents */
    div[role="dialog"] div[data-testid="column"]:nth-of-type(1) {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
        padding-top: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Startup check trigger at top-level to prevent AttributeError
if "vector_store" not in st.session_state:
    with st.spinner("🔄 Loading RAG index and connecting to Cloudflare R2..."):
        store, health_status = initialize_rag()
        st.session_state.vector_store = store
        st.session_state.health_status = health_status
else:
    sync_index_if_version_changed()

# Initialize document manager state (Requirement 5)
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = set()
if 'selected_detail_doc' not in st.session_state:
    st.session_state.selected_detail_doc = None
if 'doc_page' not in st.session_state:
    st.session_state.doc_page = 1
if 'doc_status_filter' not in st.session_state:
    st.session_state.doc_status_filter = "All"
if 'doc_type_filter' not in st.session_state:
    st.session_state.doc_type_filter = "All"
if 'doc_sort_filter' not in st.session_state:
    st.session_state.doc_sort_filter = "Newest"
if 'show_bulk_delete_confirm' not in st.session_state:
    st.session_state.show_bulk_delete_confirm = False
if 'show_single_delete_confirm' not in st.session_state:
    st.session_state.show_single_delete_confirm = None
if 'show_doc_manager_dialog' not in st.session_state:
    st.session_state.show_doc_manager_dialog = False

st.title("📚 ClinicalDocs AI")
st.caption("Clinical Knowledge Assistant")

# Left Sidebar Configurations
# 1. System Health Panel (Requirement 4 & 8)
st.sidebar.header("📋 System Health")
status = st.session_state.health_status
st.sidebar.markdown(f"""
- 🟢 **Cloud Storage Connected**
- 🟢 **Embedding Model Ready**
- 🟢 **Knowledge Base Version**: `v{status['version']}`
- 🟢 **Knowledge Base Synced**
- 🟢 **System Ready**
""")

# 2. Knowledge Base Stats (Requirement 5)
metadata = get_document_metadata()
indexed_docs = metadata.get("documents", {})
total_docs = len(indexed_docs)
total_chunks = sum(doc.get("chunk_count", 0) for doc in indexed_docs.values())
total_size_kb = sum(doc.get("file_size_kb", 0) for doc in indexed_docs.values())
total_size_mb = total_size_kb / 1024

last_sync_raw = status.get("last_updated", "Never")
if last_sync_raw != "Never" and len(last_sync_raw) > 16:
    try:
        dt = datetime.fromisoformat(last_sync_raw)
        if dt.date() == datetime.now().date():
            last_sync_str = f"Today {dt.strftime('%I:%M %p')}"
        else:
            last_sync_str = dt.strftime("%b %d, %I:%M %p")
    except Exception:
        last_sync_str = last_sync_raw[:16].replace("T", " ")
else:
    last_sync_str = last_sync_raw

with st.sidebar.expander("📊 Knowledge Base Stats", expanded=False):
    st.markdown(f"""
    <div style='font-size: 0.85em; line-height: 1.5; opacity: 0.9;'>
        📁 Docs: <strong>{total_docs}</strong> | 🧩 Chunks: <strong>{total_chunks:,}</strong><br/>
        💾 Size: <strong>{total_size_mb:.2f} MB</strong> | 🏷️ Version: <strong>v{status['version']}</strong><br/>
        ⏱️ Sync: <strong>{last_sync_str}</strong><br/>
        🤖 Model: <code>all-MiniLM-L6-v2</code>
    </div>
    """, unsafe_allow_html=True)

# 3. Connection & Sync state (Requirement 6)
st.sidebar.caption(f"✓ Synced ({last_sync_str})")

st.sidebar.markdown("---")

# 4. Document Ingestion sidebar uploader
st.sidebar.header("📁 Document Ingestion")
uploaded = st.sidebar.file_uploader(
    "Upload clinical trial files (PDF, DOCX, TXT, CSV, MD, HTML)",
    type=["pdf", "docx", "txt", "csv", "md", "html"],
    accept_multiple_files=True,
)

# Set of processed file names in this session to prevent duplicate spams on rerun
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# 5. Manage Documents Trigger (Requirement 2 & 1)
st.sidebar.markdown("""
<div style='background-color: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px; margin-top: 15px; margin-bottom: 5px;'>
    <p style='font-size: 0.85em; font-weight: 600; margin: 0; color: #f8fafc;'>📁 Manage Documents</p>
    <p style='font-size: 0.75em; opacity: 0.8; margin: 2px 0 8px 0; line-height: 1.3;'>View, search and organize uploaded documents.</p>
</div>
""", unsafe_allow_html=True)
if st.sidebar.button("Open Document Manager", key="btn_open_portal_sidebar", use_container_width=True):
    st.session_state.dialog_init_needed = True
    st.session_state.show_doc_manager_dialog = True
    st.rerun()

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

# Startup check complete. Rendering centered search interface (Requirement 1 & 5)

q = st.text_area(
    "Ask anything about your clinical documents...",
    height=110,
    placeholder="Ask anything about your clinical documents...",
    value=st.session_state.get("query_input", ""),
    key="query_input_box",
    label_visibility="collapsed"
)

# Focus & Enter-Key Submission Javascript helper (Requirement 3)
st.markdown("""
<script>
function setupSearchTextarea() {
    const textareas = window.parent.document.querySelectorAll('textarea');
    for (const t of textareas) {
        if (t.placeholder && t.placeholder.includes("Ask anything")) {
            // Set focus
            if (window.parent.document.activeElement !== t) {
                t.focus();
                t.setSelectionRange(t.value.length, t.value.length);
            }
            
            // Add Enter Key listener (prevent newline, trigger submit button)
            if (!t.dataset.enterListenerAdded) {
                t.dataset.enterListenerAdded = "true";
                t.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        // Find the Primary Ask AI button
                        const buttons = window.parent.document.querySelectorAll('button');
                        for (const b of buttons) {
                            if (b.innerText && b.innerText.includes("Ask AI")) {
                                b.click();
                                break;
                            }
                        }
                    }
                });
            }
        }
    }
    
    // Prevent Enter key from triggering submit on anything except the main search textarea
    if (!window.parent.document.dataset.formPreventAdded) {
        window.parent.document.dataset.formPreventAdded = "true";
        window.parent.document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                const active = window.parent.document.activeElement;
                if (active && active.tagName !== 'TEXTAREA') {
                    e.stopPropagation();
                }
            }
        }, true);
    }
}
setTimeout(setupSearchTextarea, 200);
</script>
""", unsafe_allow_html=True)

# Search Scope display above search textbox (Search Box section)
selected_docs_list = list(st.session_state.get("selected_docs", set()))
if len(selected_docs_list) == 0:
    st.markdown("""
    <div style='font-size: 0.85em; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 4px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 8px;'>
        🌍 Searching All Documents
    </div>
    """, unsafe_allow_html=True)
else:
    col_scope_b, col_scope_c = st.columns([5, 2.5])
    with col_scope_b:
        st.markdown(f"""
        <div style='font-size: 0.85em; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 4px; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); color: #38bdf8; margin-bottom: 8px;'>
            📄 Searching {len(selected_docs_list)} Selected Documents
        </div>
        """, unsafe_allow_html=True)
    with col_scope_c:
        if st.button("❌ Clear Selection", key="clear_selection_search_scope_btn", use_container_width=True):
            st.session_state.selected_docs.clear()
            st.rerun()

q_text = q

# Clean expandable Example Questions panel (Requirement 2)
with st.expander("💡 Example Questions", expanded=False):
    if st.button("• What is ADaM?", key="ex_adam", use_container_width=False):
        st.session_state.query_input = "What is ADaM?"
        st.session_state.search_executed = False
        st.session_state.last_error = None
        st.rerun()
    if st.button("• Explain SDTM.", key="ex_sdtm", use_container_width=False):
        st.session_state.query_input = "Explain SDTM."
        st.session_state.search_executed = False
        st.session_state.last_error = None
        st.rerun()
    if st.button("• Show inclusion criteria.", key="ex_criteria", use_container_width=False):
        st.session_state.query_input = "Show inclusion criteria."
        st.session_state.search_executed = False
        st.session_state.last_error = None
        st.rerun()
    if st.button("• Summarize methodology.", key="ex_methodology", use_container_width=False):
        st.session_state.query_input = "Summarize methodology."
        st.session_state.search_executed = False
        st.session_state.last_error = None
        st.rerun()

# Initialize search states
if "searching" not in st.session_state:
    st.session_state.searching = False
if "search_executed" not in st.session_state:
    st.session_state.search_executed = False
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# Run search synchronously on rerun if searching state is triggered (Requirement 4)
if st.session_state.searching:
    st.session_state.last_error = None
    if st.session_state.vector_store.index.ntotal == 0:
        st.session_state.last_error = "❌ Index is empty. Please upload documents in the sidebar first."
        st.session_state.searching = False
        st.rerun()
    else:
        sel_docs = list(st.session_state.selected_docs) if st.session_state.get("selected_docs") else None
        retriever = VectorStoreRetrieverAdapter(
            st.session_state.vector_store,
            k=settings.RETRIEVER_K,
            filter_sources=sel_docs
        )
        try:
            qa = create_qa_from_retriever(retriever)
        except Exception as e:
            st.session_state.last_error = f"❌ QA chain initialization failed: {e}"
            st.session_state.searching = False
            st.rerun()
            
        try:
            with st.spinner("🧠 Generating Answer..."):
                t_search_start = time.time()
                search_q = q_text
                if sel_docs:
                    search_q = f"In documents {sel_docs}: {q_text}"
                res_tuple = query_with_features(qa, search_q)
                
                if res_tuple:
                    result, was_cached, elapsed = res_tuple
                    if result is not None:
                        st.session_state.search_executed = True
                        st.session_state.last_result = result
                        st.session_state.last_was_cached = was_cached
                        st.session_state.last_elapsed = elapsed
                        st.session_state.last_error = None
                    else:
                        st.session_state.search_executed = False
                        st.session_state.last_error = "All API attempts failed. Please verify your Gemini or Groq API keys."
                else:
                    st.session_state.search_executed = False
                    st.session_state.last_error = "All API attempts failed. No response generated."
        except Exception as e:
            st.session_state.search_executed = False
            st.session_state.last_error = f"Error during generation: {e}"
            
        st.session_state.searching = False
        st.rerun()

# Render errors permanently if search failed (prevents disappearing on rerun)
if st.session_state.get("last_error"):
    st.error(st.session_state.last_error)

# Focus Javascript helper on chip click (Requirement 3)
if st.session_state.get("query_input"):
    st.markdown("""
    <script>
    setTimeout(() => {
        const textareas = window.parent.document.querySelectorAll('textarea');
        for (const t of textareas) {
            if (t.placeholder && t.placeholder.includes("Ask anything")) {
                t.focus();
                t.setSelectionRange(t.value.length, t.value.length);
            }
        }
    }, 100);
    </script>
    """, unsafe_allow_html=True)

# Primary action button rendering (Requirement 1 & 4)
btn_label = "⏳ Generating Answer..." if st.session_state.searching else "✨ Ask AI"
if st.button(btn_label, type="primary", use_container_width=False, disabled=st.session_state.searching, key="primary_ask_ai_btn"):
    if not q_text.strip():
        st.warning("⚠️ Please enter a question first.")
    elif not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        st.error("❌ Missing GOOGLE_API_KEY, GEMINI_API_KEY, or GROQ_API_KEY. Please set at least one in Streamlit secrets.")
    else:
        st.session_state.searching = True
        st.rerun()

# Render Search Results if present
if st.session_state.search_executed and st.session_state.get("last_result"):
    result = st.session_state.last_result
    was_cached = st.session_state.get("last_was_cached", False)
    elapsed = st.session_state.get("last_elapsed", 0.0)
    
    # Render results inside a modern container card (Requirement 5 & 7 & 9)
    with st.container(border=True):
        st.subheader("✨ AI Answer")
        st.markdown(result.get("result", "").strip() if result.get("result") else "")
        
        # Group and format sources
        from collections import defaultdict
        sources_group = defaultdict(list)
        for d in result.get("source_documents", []):
            source_name = d.metadata.get("source", "Unknown Document")
            page = d.metadata.get("page", None)
            chunk_id = d.metadata.get("chunk_id", None)
            sources_group[source_name].append((page, chunk_id))
            
        if sources_group:
            st.markdown("---")
            st.markdown("#### 📄 Sources")
            for source_name, chunks_info in sources_group.items():
                pages = [p for p, c in chunks_info if p is not None]
                chunks = [c for p, c in chunks_info if c is not None]
                
                pages_str = ""
                if pages:
                    unique_pages = sorted(list(set(pages)))
                    if len(unique_pages) == 1:
                        pages_str = f"Page {unique_pages[0] + 1}"
                    elif unique_pages[-1] - unique_pages[0] == len(unique_pages) - 1:
                        pages_str = f"Pages {unique_pages[0] + 1}-{unique_pages[-1] + 1}"
                    else:
                        pages_str = f"Pages " + ", ".join(str(p + 1) for p in unique_pages)
                        
                chunks_str = ""
                if chunks:
                    unique_chunks = sorted(list(set(chunks)))
                    if len(unique_chunks) == 1:
                        chunks_str = f"Chunk {unique_chunks[0]}"
                    elif unique_chunks[-1] - unique_chunks[0] == len(unique_chunks) - 1:
                        chunks_str = f"Chunks {unique_chunks[0]}-{unique_chunks[-1]}"
                    else:
                        chunks_str = f"Chunks " + ", ".join(str(c) for c in unique_chunks)
                        
                details = []
                if pages_str:
                    details.append(pages_str)
                if chunks_str:
                    details.append(chunks_str)
                    
                details_str = f" ({', '.join(details)})" if details else ""
                st.write(f"📄 **{source_name}**{details_str}")
        
        # Render compact horizontal metadata card (Requirement 6)
        num_docs_used = len(sources_group)
        num_chunks_retrieved = len(result.get("source_documents", []))
        
        st.markdown(f"""
        <div style='background-color: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 10px 15px; margin-top: 15px; font-size: 0.9em; opacity: 0.9; display: flex; justify-content: space-around; align-items: center;'>
            <span>⏱️ <strong>{elapsed:.2f} sec</strong></span>
            <span>📄 <strong>{num_docs_used} Sources</strong></span>
            <span>🧩 <strong>{num_chunks_retrieved} Chunks</strong></span>
            <span>📚 <strong>{num_docs_used} Documents</strong></span>
        </div>
        """, unsafe_allow_html=True)
            
    # Collapsible Context Evidence block
    with st.expander("🔍 View Context Evidence Snippets"):
        for i, d in enumerate(result.get("source_documents", [])[:6], 1):
            st.markdown(f"**{i}. {d.metadata.get('source','unknown')}**")
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

# Render the dialog modal if active (stay open across native reruns)
if st.session_state.get("show_doc_manager_dialog", False):
    document_management_dialog()
