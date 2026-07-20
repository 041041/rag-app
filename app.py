# app.py - Refactored for Cloudflare R2 and FAISS Integration
import os
import sys

# Ensure torchvision and its common submodules are pre-populated with valid ModuleSpec objects
# to prevent importlib.util.find_spec from throwing a ValueError during parallel hot-reloads.
import types
import importlib.machinery

for name in ["torchvision", "torchvision.io", "torchvision.transforms", "torchvision.transforms.v2"]:
    if name not in sys.modules or not hasattr(sys.modules[name], "__spec__") or sys.modules[name].__spec__ is None:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        mod = types.ModuleType(name)
        mod.__spec__ = spec
        mod.__path__ = []
        mod.__file__ = __file__
        sys.modules[name] = mod

# Prevent torchvision import errors from optional Hugging Face transformers vision models
# by registering a custom meta-path finder that dynamically resolves nested submodules.
class DummyModuleLoader:
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__spec__ = spec
        mod.__path__ = []
        mod.__file__ = __file__
        return mod
    def exec_module(self, module):
        pass

class TorchvisionMockFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(
                name=fullname,
                loader=DummyModuleLoader(),
                is_package=True
            )
        return None

sys.meta_path.insert(0, TorchvisionMockFinder())

# Diagnostic check for sentence-transformers import
import logging
logger = logging.getLogger("RAGApp.Diagnostic")
try:
    import sentence_transformers
    logger.info("👉 Diagnostic: sentence_transformers imported successfully in app.py!")
except Exception as e:
    logger.error("❌ Diagnostic: sentence_transformers import failed in app.py!", exc_info=True)

# Standardize on GOOGLE_API_KEY only: delete GEMINI_API_KEY if present in environment
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

# Monkey-patch google.genai._api_client.BaseApiClient._request to bypass all SDK-level retry loops
# This guarantees that the Fallback Manager receives the first 429 immediately.
try:
    from google.genai._api_client import BaseApiClient
    def custom_request(self, http_request, http_options=None, stream=False):
        return self._request_once(http_request, stream=stream)
    BaseApiClient._request = custom_request
except Exception:
    pass
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

# Consistent usage of GOOGLE_API_KEY is enforced throughout the app.

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

from rag.llm import ClinicalRAGLLM


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
    Decoupled QA wrapper that uses ClinicalRAGLLM to handle all prompt invocations.
    """
    def __init__(self, llm, retriever, prompt_template):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt_template

    def _build_input(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        context_parts = []
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", 0)
            page_num = page + 1 if isinstance(page, int) else page
            context_parts.append(
                f"[Document {i}]: Source: {source}, Page: {page_num}\n"
                f"Content:\n{d.page_content}"
            )
        context = "\n\n".join(context_parts)
        if hasattr(self.prompt, "template"):
            prompt_text = self.prompt.template.format(question=query, context=context)
        else:
            prompt_text = f"Question: {query}\nContext:\n{context}"
        return prompt_text, docs

    def run(self, query: str):
        prompt_text, docs = self._build_input(query)
        logger.info("Invoking ClinicalRAGLLM unified abstraction...")
        response = self.llm.invoke(prompt_text)
        result_text = response.content if hasattr(response, "content") else str(response)
        return {"result": result_text, "source_documents": docs, "raw": response}


# -------------------------
# LLM prompt / QA builder
# -------------------------
PROMPT_TEMPLATE_STR = (
    "You are an expert assistant for clinical trial data standards. Respond to the user's question using the following format:\n\n"
    "1. Short definition: A brief, 1-2 sentence definition. Include an inline citation at the end of the definition in the format (Source: <filename>, Page: <page_num>).\n"
    "2. Key points: Key details as separate bullet items. Start each bullet on a new line. Do not combine bullets into paragraphs. Include inline citations at the end of bullet points in the format (Source: <filename>, Page: <page_num>).\n"
    "3. Sources: List of source document names and page references actually cited in the sections above.\n\n"
    "Formatting rules:\n"
    "- Start the response directly with the definition. Do not use introductory phrases like 'Based on the provided context' or 'According to the context'.\n"
    "- Each bullet point must be on its own line. Do not combine bullets into paragraphs.\n"
    "- Use only real metadata from the context documents below. Never invent page numbers.\n"
    "- Never mention the phrase 'provided context' or 'provided text' in your answer.\n\n"
    "Question: {question}\nContext:\n{context}\n\nAnswer:"
)
prompt_template = PromptTemplate(input_variables=["question", "context"], template=PROMPT_TEMPLATE_STR)

@st.cache_resource
def get_clinical_llm():
    return ClinicalRAGLLM()

def create_qa_from_retriever(retriever):
    llm = get_clinical_llm()
    return SimpleQAWrapper(llm=llm, retriever=retriever, prompt_template=prompt_template)


# -------------------------
# Chain-safe invoker
# -------------------------
def _call_chain_safe(qa_chain, query: str):
    res = _call_chain_safe_raw(qa_chain, query)
    if res and isinstance(res, dict):
        import re
        
        raw_source_docs = list(res.get("source_documents", []))
        
        # 1. Clean thinking tags and introductory context phrases
        result_text = res.get("result", "")
        if isinstance(result_text, str):
            result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()
            
            # Remove common introductory patterns
            phrases = [
                r"^\s*based\s+on\s+the\s+provided\s+context,?\s*",
                r"^\s*based\s+on\s+clinical\s+trial\s+standards\s+and\s+the\s+provided\s+context,?\s*",
                r"^\s*according\s+to\s+the\s+context,?\s*",
                r"^\s*based\s+on\s+the\s+context\s+provided,?\s*",
                r"^\s*according\s+to\s+the\s+provided\s+context,?\s*"
            ]
            for pattern in phrases:
                result_text = re.sub(pattern, "", result_text, flags=re.IGNORECASE)
            
            # Capitalize first letter if it starts lowercase after stripping
            if result_text:
                result_text = result_text[0].upper() + result_text[1:]
                
            res["result"] = result_text

        after_validation_docs = list(res.get("source_documents", []))

        # 2. Filter source_documents to only those actually cited in the result text
        source_docs = res.get("source_documents", [])
        if source_docs and isinstance(result_text, str):
            citations = re.findall(
                r"\(\s*Source:\s*([^,)]+?),\s*Page:?\s*(\d+)\s*\)",
                result_text,
                flags=re.IGNORECASE
            )
            if citations:
                def normalize_source_name(name: str) -> str:
                    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
                    return "".join(c for c in name.lower() if c.isalnum())
                
                cited_pairs = set()
                for c_src, c_pg in citations:
                    try:
                        cited_pairs.add((normalize_source_name(c_src), int(c_pg)))
                    except ValueError:
                        pass
                
                cited_docs = []
                for d in source_docs:
                    doc_src = d.metadata.get("source", "")
                    doc_pg = d.metadata.get("page")
                    if doc_pg is not None:
                        doc_pg_1 = doc_pg + 1
                        norm_doc_src = normalize_source_name(doc_src)
                        if (norm_doc_src, doc_pg_1) in cited_pairs:
                            cited_docs.append(d)
                
                # Only restrict if we actually matched some of them to avoid blanking
                if cited_docs:
                    res["source_documents"] = cited_docs
                    
        # 3. Log debug information (Task 5 logs)
        retrieved_list = [f"{d.metadata.get('source')} Page {d.metadata.get('page', 0) + 1}" for d in raw_source_docs]
        validation_list = [f"{d.metadata.get('source')} Page {d.metadata.get('page', 0) + 1}" for d in after_validation_docs]
        displayed_list = [f"{d.metadata.get('source')} Page {d.metadata.get('page', 0) + 1}" for d in res.get("source_documents", [])]
        
        logger.info(f"--- CITATION SOURCE VALIDATION LOG ---")
        logger.info(f"Question: '{query}'")
        logger.info(f"Before validation:\n{type(res)}")
        logger.info("Retrieved sources:")
        for s in retrieved_list:
            logger.info(f"  - {s}")
        logger.info(f"After validation:\n{type(res)}")
        logger.info("After validation:")
        for s in validation_list:
            logger.info(f"  - {s}")
        logger.info("Displayed sources:")
        for s in displayed_list:
            logger.info(f"  - {s}")
        logger.info(f"Final returned object:\n{res.keys()}")
        logger.info(f"--------------------------------------")
    return res

def _call_chain_safe_raw(qa_chain, query: str):
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
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                elapsed = time.time() - start_time
                metrics.log_query(query, elapsed, error=True)
                raise RuntimeError(f"All retries failed: {last_error}")

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
    
    # 1. Initialize FAISSVectorStore (verifies FAISS loaded and embedding model loaded)
    print("🚀 [STARTUP LOG] [STEP 2] Creating FAISSVectorStore...", flush=True)
    print("Loading embedding model...", flush=True)
    print("Embedding provider: sentence-transformers", flush=True)
    print("Model: BAAI/bge-small-en-v1.5", flush=True)
    
    try:
        store = FAISSVectorStore()
        dimension = getattr(store, "dimension", 384)
        index_size = store.index.ntotal if (getattr(store, "index", None) is not None) else 0
        
        print(f"Embedding dimension: {dimension}", flush=True)
        print("FAISS loaded successfully", flush=True)
        print(f"Index size: {index_size} documents", flush=True)
        print("Application ready", flush=True)
        embeddings_loaded = True
        print("🚀 [STARTUP LOG] [STEP 3] FAISSVectorStore created successfully.", flush=True)
    except Exception as e:
        print(f"🚀 [STARTUP LOG] [STEP 3 ERROR] FAISSVectorStore creation failed: {e}", flush=True)
        raise RuntimeError("Embedding model initialization failed. Application cannot start.") from e
        
    health_status = {
        "r2_connected": False,
        "index_loaded": False,
        "version": 0,
        "last_updated": "Never",
        "message": "",
        "embeddings_loaded": embeddings_loaded,
        "gemini_reachable": None,
        "groq_reachable": None
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

    # 3. Check Gemini reachability if credentials exist
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            print("🚀 [STARTUP LOG] [STEP 12] Verifying Gemini model reachability...", flush=True)
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                temperature=0.2,
                google_api_key=gemini_key
            )
            # A lightweight call to verify connection
            llm.invoke([HumanMessage(content="Hello")])
            health_status["gemini_reachable"] = True
            print("🚀 [STARTUP LOG] [STEP 13] Gemini is online and reachable.", flush=True)
        except Exception as e:
            print(f"🚀 [STARTUP LOG] [STEP 13 WARNING] Gemini reachability check failed: {e}", flush=True)
            health_status["gemini_reachable"] = False
    else:
        health_status["gemini_reachable"] = None

    # 4. Check Groq reachability if credentials exist
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            print("🚀 [STARTUP LOG] [STEP 14] Verifying Groq model reachability...", flush=True)
            from langchain_groq import ChatGroq
            groq_model = os.getenv("GROQ_PRIMARY_MODEL", os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"))
            if "llama-3.3-70b" in groq_model or "versatile" in groq_model:
                groq_model = "qwen/qwen3.6-27b"
            llm = ChatGroq(
                model_name=groq_model,
                temperature=0.2,
                groq_api_key=groq_key
            )
            llm.invoke([HumanMessage(content="Hello")])
            health_status["groq_reachable"] = True
            print("🚀 [STARTUP LOG] [STEP 15] Groq is online and reachable.", flush=True)
        except Exception as e:
            # Try secondary model as check fallback
            try:
                groq_sec = os.getenv("GROQ_SECONDARY_MODEL", "openai/gpt-oss-20b")
                if "llama-3.3-70b" in groq_sec or "versatile" in groq_sec:
                    groq_sec = "openai/gpt-oss-20b"
                print(f"🚀 [STARTUP LOG] Groq primary failed. Testing secondary model '{groq_sec}'...", flush=True)
                llm = ChatGroq(
                    model_name=groq_sec,
                    temperature=0.2,
                    groq_api_key=groq_key
                )
                llm.invoke([HumanMessage(content="Hello")])
                health_status["groq_reachable"] = True
                print("🚀 [STARTUP LOG] Groq secondary is online and reachable.", flush=True)
            except Exception as e2:
                print(f"🚀 [STARTUP LOG] Groq reachability check failed for all models: {e2}", flush=True)
                health_status["groq_reachable"] = False
    else:
        health_status["groq_reachable"] = None

    # 5. Startup Validation and Diagnostics
    try:
        print("\n🚀 [STARTUP LOG] [STEP 16] Running FAISS Index Startup Validation...", flush=True)
        index_path = settings.INDEXES_DIR / settings.FAISS_INDEX_FILE
        total_vectors = store.index.ntotal if (getattr(store, "index", None) is not None) else 0
        metadata_records = len(store.docs) if (getattr(store, "docs", None) is not None) else 0
        
        # Load document_metadata.json to see if we expect documents
        doc_meta = get_document_metadata()
        expected_docs = list(doc_meta.get("documents", {}).keys())
        num_expected_docs = len(expected_docs)
        
        # Diagnostics
        print(f"FAISS index path: {index_path}", flush=True)
        print(f"Total vectors: {total_vectors}", flush=True)
        print(f"Metadata records: {metadata_records}", flush=True)
        
        # Print sample document names (top 3)
        sample_docs = []
        if getattr(store, "docs", None) is not None:
            unique_sources = set()
            for doc_id, doc in store.docs.items():
                src = doc.metadata.get("source")
                if src:
                    unique_sources.add(src)
            sample_docs = sorted(list(unique_sources))[:3]
            
        print("Sample document names:", flush=True)
        for idx, name in enumerate(sample_docs, 1):
            print(f"{idx}. {name}", flush=True)
            
        # Validation warning check
        if num_expected_docs > 0 and total_vectors == 0:
            print("⚠️ WARNING: Expected documents exist in metadata but FAISS index size is 0!", flush=True)
            missing_parts = []
            if not index_path.exists():
                missing_parts.append("FAISS index file (.faiss)")
            if not (settings.INDEXES_DIR / settings.METADATA_PKL_FILE).exists():
                missing_parts.append("Metadata mapping file (.pkl)")
            if not (settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE).exists():
                missing_parts.append("Document metadata JSON")
            if missing_parts:
                print(f"  Missing files identified: {', '.join(missing_parts)}", flush=True)
            else:
                print("  All local index files exist on disk, but the loaded index contains 0 vectors. The index files may be corrupted or empty.", flush=True)
        else:
            print("✅ FAISS index startup validation passed successfully.", flush=True)
        print("", flush=True)
    except Exception as diag_err:
        print(f"⚠️ Warning: Failed to run FAISS startup diagnostics: {diag_err}\n", flush=True)
            
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
                st.session_state.selected_detail_doc = None
            st.rerun()

def toggle_select_all_callback(page_filenames, page_doc_ids):
    val = st.session_state.hdr_select_all
    if val:
        for f in page_filenames:
            st.session_state.selected_docs.add(f)
        for d_id in page_doc_ids:
            st.session_state[f"chk_{d_id}"] = True
    else:
        for f in page_filenames:
            st.session_state.selected_docs.discard(f)
        for d_id in page_doc_ids:
            st.session_state[f"chk_{d_id}"] = False

@st.dialog("🗄️ Document Management Portal", width="large")
def document_management_dialog():
    """
    Simplified Document Management Portal modal dialog overlay.
    """
    # Read metadata database
    metadata = get_document_metadata()
    indexed_docs = metadata.get("documents", {})
    
    # Inline Bulk Delete Confirmation (Requirement 8 - no nesting dialogs)
    if st.session_state.get("show_bulk_delete_confirm"):
        selected_ids = []
        selected_names = []
        for doc_id, doc in indexed_docs.items():
            if doc.get("filename") in st.session_state.selected_docs:
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
                delete_documents_bulk_workflow(selected_ids)
                st.session_state.selected_docs = set()
                # Clear checkbox keys
                for k in list(st.session_state.keys()):
                    if k.startswith("chk_"):
                        st.session_state[k] = False
                st.session_state.selected_detail_doc = None
                st.session_state.show_bulk_delete_confirm = False
                st.rerun()
        return

    # Header with subtitle only (auto-rendered dialog title has priority)
    st.markdown("<p style='color: #94a3b8 !important; font-size: 0.9em !important; margin-top: -12px; margin-bottom: 15px;'>Manage uploaded clinical documents.</p>", unsafe_allow_html=True)
    
    # Render simplified filter controls with File Type dropdown (Requirement 3 & 7)
    col_f1, col_f2, col_f3, col_f4 = st.columns([40, 20, 20, 20])
    with col_f1:
        st.markdown("<p style='font-size: 0.85em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px;'>Search filename</p>", unsafe_allow_html=True)
        doc_search = st.text_input("Search filename", placeholder="Search filename...", value=st.session_state.get("doc_search_filter", ""), key="doc_search_modal_input", label_visibility="collapsed")
    with col_f2:
        st.markdown("<p style='font-size: 0.85em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px;'>Status</p>", unsafe_allow_html=True)
        status_filter = st.selectbox("Status", ["All", "Indexed", "Processing", "Failed"], index=["All", "Indexed", "Processing", "Failed"].index(st.session_state.get("doc_status_filter", "All")), key="doc_status_modal_sel", label_visibility="collapsed")
    with col_f3:
        st.markdown("<p style='font-size: 0.85em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px;'>File Type</p>", unsafe_allow_html=True)
        type_filter = st.selectbox("File Type", ["All", "PDF", "DOCX", "TXT", "CSV"], index=["All", "PDF", "DOCX", "TXT", "CSV"].index(st.session_state.get("doc_type_filter", "All")), key="doc_type_modal_sel", label_visibility="collapsed")
    with col_f4:
        st.markdown("<p style='font-size: 0.85em; font-weight: 700; color: #f1f5f9; margin-bottom: 2px;'>Sort</p>", unsafe_allow_html=True)
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
            if doc_search.strip() and doc_search.lower() not in fname.lower():
                continue
            if status_filter != "All":
                status = doc.get("status", "Indexed")
                if status.lower() != status_filter.lower():
                    continue
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
    num_pages = max(1, (total_documents - 1) // 50 + 1)
    current_page = min(st.session_state.get("doc_page", 1), num_pages)
    st.session_state.doc_page = current_page
    start_index = (current_page - 1) * 50
    end_index = min(start_index + 50, total_documents)
    page_docs = filtered_docs_list[start_index:end_index]
    
    # Dynamic table and side details panel layout (Requirement 5)
    selected_detail_doc = st.session_state.get("selected_detail_doc")
    if selected_detail_doc is not None:
        col_table, col_details = st.columns([7, 5])
    else:
        col_table = st.container()
        col_details = None
        
    with col_table:
        # Compact Bulk Action Toolbar — always visible, enabled only when selection exists
        is_selected = len(st.session_state.selected_docs) > 0
        st.markdown("""
        <style>
        /* ── Dialog-level overrides ────────────────────────── */
        /* Strip ALL dialog secondary buttons to slate style */
        [data-testid="stDialog"] button {
            border-radius: 5px !important;
            font-size: 0.82em !important;
            font-weight: 500 !important;
            height: 30px !important;
            line-height: 30px !important;
            white-space: nowrap !important;
        }
        /* Filename row buttons — transparent, left-aligned link appearance */
        [data-testid="stDialog"] [data-testid="stButton"] button {
            background: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
            color: #e2e8f0 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 2px 4px !important;
            font-weight: 500 !important;
        }
        [data-testid="stDialog"] [data-testid="stButton"] button:hover {
            background: rgba(255,255,255,0.06) !important;
            color: #7dd3fc !important;
        }
        /* Toolbar action buttons — keep primary style but compact */
        [data-testid="stDialog"] [data-testid="stButton"]:has(button[data-basebuttonstyle="primary"]) button {
            background: #1d4ed8 !important;
            background-image: none !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 5px !important;
        }
        [data-testid="stDialog"] [data-testid="stButton"]:has(button[data-basebuttonstyle="primary"]) button:hover {
            background: #1e40af !important;
        }
        /* Download button inside dialog */
        [data-testid="stDialog"] [data-testid="stDownloadButton"] button {
            background: #0f766e !important;
            background-image: none !important;
            color: white !important;
            border: none !important;
        }
        [data-testid="stDialog"] [data-testid="stDownloadButton"] button:hover {
            background: #0d9488 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🗑 Delete Selected", type="primary",
                         key="bulk_delete_action_btn",
                         use_container_width=True,
                         disabled=not is_selected):
                st.session_state.show_bulk_delete_confirm = True
        with col_b2:
            selected_names_list = list(st.session_state.selected_docs)
            zip_buffer = io.BytesIO()
            if is_selected:
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for doc_name in selected_names_list:
                        file_bytes = get_file_bytes(doc_name)
                        if file_bytes:
                            zip_file.writestr(doc_name, file_bytes)
            zip_data = zip_buffer.getvalue()
            st.download_button(
                label="⬇ Download Selected",
                data=zip_data,
                file_name="selected_clinical_documents.zip",
                key="bulk_download_action_btn",
                use_container_width=True,
                disabled=not is_selected
            )
        st.markdown("<hr style='margin: 8px 0; border:0; border-top:1px solid rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)

        # Table Headers with Select All callback inside header
        page_docs_filenames = [doc.get("filename") for doc_id, doc in page_docs]
        page_doc_ids = [doc_id for doc_id, doc in page_docs]
        
        all_selected = all(f in st.session_state.selected_docs for f in page_docs_filenames) if page_docs_filenames else False
        st.session_state["hdr_select_all"] = all_selected
        
        col_hdr_chk, col_hdr_name, col_hdr_status, col_hdr_size = st.columns([1, 6, 2.5, 2.5])
        with col_hdr_chk:
            st.checkbox(
                "",
                key="hdr_select_all",
                label_visibility="collapsed",
                on_change=toggle_select_all_callback,
                args=(page_docs_filenames, page_doc_ids)
            )
        with col_hdr_name:
            st.markdown("<span style='font-weight: bold; font-size: 0.9em; opacity: 0.9;'>Document</span>", unsafe_allow_html=True)
        with col_hdr_status:
            st.markdown("<div style='text-align: center; font-weight: bold; font-size: 0.9em; opacity: 0.9;'>Status</div>", unsafe_allow_html=True)
        with col_hdr_size:
            st.markdown("<div style='text-align: center; font-weight: bold; font-size: 0.9em; opacity: 0.9;'>Size</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 4px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.15);'/>", unsafe_allow_html=True)
        
        if page_docs:
            for doc_id, doc in page_docs:
                doc_name = doc.get("filename")
                size = doc.get("file_size_kb")
                size_str = f"{size/1024:.1f} MB" if size and size > 1024 else f"{size:.0f} KB" if size else "Unknown"
                status = doc.get("status", "Indexed")
                status_str = "🟢 Indexed" if status == "Indexed" else "🟡 Processing" if status == "Processing" else "🔴 Failed"
                    
                # Highlight active row in table via filename pointer (Requirement 9)
                is_active = False
                if selected_detail_doc and selected_detail_doc.get("filename") == doc_name:
                    is_active = True
                doc_display_name = f"👉 📄 {doc_name}" if is_active else f"📄 {doc_name}"
                
                col_row_chk, col_row_name, col_row_status, col_row_size = st.columns([1, 6, 2.5, 2.5])
                with col_row_chk:
                    doc_checked = st.checkbox("", value=(doc_name in st.session_state.selected_docs), key=f"chk_{doc_id}", label_visibility="collapsed")
                    if doc_checked != (doc_name in st.session_state.selected_docs):
                        if doc_checked:
                            st.session_state.selected_docs.add(doc_name)
                        else:
                            st.session_state.selected_docs.discard(doc_name)
                with col_row_name:
                    # Filename — only this cell is a button (opens details panel)
                    row_bg = "rgba(255,255,255,0.04)" if is_active else "transparent"
                    if st.button(doc_display_name, key=f"btn_detail_name_{doc_id}", use_container_width=True):
                        st.session_state.selected_detail_doc = doc
                with col_row_status:
                    # Status — plain markdown, no button needed
                    st.markdown(
                        f"<div style='text-align:center; padding-top:6px; font-size:0.88em; color:#94a3b8;'>{status_str}</div>",
                        unsafe_allow_html=True
                    )
                with col_row_size:
                    # Size — plain markdown, no button needed
                    st.markdown(
                        f"<div style='text-align:center; padding-top:6px; font-size:0.88em; color:#94a3b8;'>{size_str}</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("<hr style='margin: 2px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.04);'/>", unsafe_allow_html=True)
        else:
            st.info("No matching documents found.")
            
    if col_details is not None:
        with col_details:
            detail_doc = st.session_state.get("selected_detail_doc")
            if detail_doc:
                dfname = detail_doc.get("filename")
                st.markdown(f"### 📋 File Details")
                st.markdown(f"**Document Name**: `{dfname}`")
                dsize = detail_doc.get("file_size_kb")
                dsize_str = f"{dsize/1024:.1f} MB" if dsize and dsize > 1024 else f"{dsize:.0f} KB" if dsize else "Unknown"
                st.markdown(f"**Size**: `{dsize_str}`")
                dext = dfname.split(".")[-1].upper()
                st.markdown(f"**File Type**: `{dext}`")
                st.markdown(f"**Uploaded Date**: `{detail_doc.get('timestamp', 'Unknown')[:16].replace('T', ' ')}`")
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
                            ddoc_id = None
                            for d_id, d in indexed_docs.items():
                                if d.get("filename") == dfname:
                                    ddoc_id = d_id
                                    break
                            if ddoc_id:
                                delete_document_workflow(ddoc_id)
                                st.session_state.selected_docs.discard(dfname)
                                st.session_state[f"chk_{ddoc_id}"] = False
                                st.session_state.selected_detail_doc = None
                            st.session_state.show_single_delete_confirm = None
                else:
                    file_bytes = get_file_bytes(dfname)
                    st.download_button(
                        label="📥 Download Original",
                        data=file_bytes,
                        file_name=dfname,
                        key="detail_download_action_btn",
                        use_container_width=True
                    )
                    if st.button("🗑️ Delete File", key="detail_delete_action_btn", type="primary", use_container_width=True):
                        st.session_state.show_single_delete_confirm = dfname
                    if st.button("✖ Close Details", key="detail_close_panel_btn", use_container_width=True):
                        st.session_state.selected_detail_doc = None
                        st.rerun()
            
    # Dialog Footer Controls (Requirement 7 & 11)
    st.markdown("<hr style='margin: 10px 0 15px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.08);'/>", unsafe_allow_html=True)
    col_foot_left, col_foot_right = st.columns([6, 6])
    
    with col_foot_left:
        st.markdown(f"<p style='font-size: 0.85em; opacity: 0.85; margin-top: 8px;'>Showing <b>{start_index + 1}–{end_index}</b> of <b>{total_documents}</b> Documents</p>", unsafe_allow_html=True)
        
    with col_foot_right:
        col_p_prev, col_p_next, col_p_close = st.columns([3, 3, 4])
        with col_p_prev:
            if st.button("← Previous", disabled=(current_page == 1), key="btn_page_prev", use_container_width=True):
                st.session_state.doc_page = max(1, current_page - 1)
        with col_p_next:
            if st.button("Next →", disabled=(current_page >= num_pages), key="btn_page_next", use_container_width=True):
                st.session_state.doc_page = min(num_pages, current_page + 1)
        with col_p_close:
            if st.button("Close", key="btn_portal_close", type="primary", use_container_width=True):
                st.session_state.show_doc_manager_dialog = False
                st.rerun()

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
    if curr_time - last_check < 300.0:  # Check at most every 5 minutes
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

def delete_documents_bulk_workflow(doc_ids: list):
    if not doc_ids:
        return
        
    owner_id = f"bulk_delete_session_{int(time.time())}"
    with st.spinner(f"🗑️ Acquiring R2 Lock & deleting {len(doc_ids)} documents..."):
        lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=45)
        if not lock_acquired:
            st.error("❌ Locking Conflict: another operation is writing. Please try again.")
            return
            
        try:
            # 1. Download current indices from R2
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
            
            deleted_filenames = []
            chunk_ids_to_remove = []
            
            # 2. Gather vectors and files to delete
            for doc_id in doc_ids:
                if doc_id in metadata.get("documents", {}):
                    doc_info = metadata["documents"][doc_id]
                    filename = doc_info["filename"]
                    chunk_ids = doc_info.get("chunk_ids", [])
                    chunk_ids_to_remove.extend(chunk_ids)
                    deleted_filenames.append(filename)
                    
                    # Delete the actual file from R2
                    r2_key = doc_info.get("r2_path", f"{settings.R2_DOCUMENTS_PREFIX}{filename}")
                    try:
                        r2_storage.delete_file(r2_key)
                    except Exception as e:
                        logger.warning(f"Could not delete R2 object {r2_key}: {e}")
                        
                    # Remove from processed files cache
                    if filename in st.session_state.processed_files:
                        st.session_state.processed_files.remove(filename)
                        
                    # Delete from metadata dictionary
                    del metadata["documents"][doc_id]
            
            # 3. Batch remove vectors from FAISS
            if chunk_ids_to_remove:
                import numpy as np
                ids_to_remove = np.array(chunk_ids_to_remove, dtype=np.int64)
                removed_count = st.session_state.vector_store.index.remove_ids(ids_to_remove)
                print(f"🗑️ Removed {removed_count} vectors from FAISS index in bulk.", flush=True)
                
            # 4. Save and Upload
            metadata["version"] = metadata.get("version", 0) + 1
            metadata["last_updated"] = datetime.now().isoformat()
            
            st.session_state.vector_store.save(str(settings.INDEXES_DIR))
            save_document_metadata(metadata)
            
            r2_storage.backup_indexes()
            r2_storage.upload_file(settings.INDEXES_DIR / settings.FAISS_INDEX_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.FAISS_INDEX_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.METADATA_PKL_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.METADATA_PKL_FILE}")
            r2_storage.upload_file(settings.INDEXES_DIR / settings.DOCUMENT_METADATA_JSON_FILE, f"{settings.R2_INDEXES_PREFIX}{settings.DOCUMENT_METADATA_JSON_FILE}")
            
            st.session_state.health_status["version"] = metadata["version"]
            st.session_state.health_status["last_updated"] = metadata["last_updated"]
            
            st.toast(f"🗑️ Successfully deleted {len(deleted_filenames)} documents!")
        except Exception as e:
            st.error(f"❌ Failed to delete documents: {e}")
        finally:
            r2_storage.release_lock(owner_id)

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
    .stApp textarea, .stApp input, .stApp select,
    textarea, input, select,
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    .stTextArea textarea,
    .stTextInput input {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
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
        height: 28px !important;
        line-height: 28px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
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
    [data-testid="stDialog"] [data-testid="stHeading"] h2, 
    [data-testid="stDialog"] h2 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 1.6em !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        background: none !important;
        margin-top: 5px !important;
        margin-bottom: 2px !important;
    }
    
    /* Pagination controls (Previous and Next secondary buttons inside dialog) */
    [data-testid="stDialog"] .stButton button[data-basebuttonstyle="secondary"] {
        background: #334155 !important;
        background-color: #334155 !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        text-align: center !important;
        justify-content: center !important;
        height: 28px !important;
        line-height: 28px !important;
        padding: 0 12px !important;
        font-size: 0.85em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stDialog"] .stButton button[data-basebuttonstyle="secondary"]:hover {
        background: #475569 !important;
        background-color: #475569 !important;
        color: #ffffff !important;
    }

    /* Primary buttons (Delete, Close) inside dialog modal */
    [data-testid="stDialog"] .stButton button[data-basebuttonstyle="primary"],
    [data-testid="stDialog"] .stDownloadButton button,
    [data-testid="stDialog"] div[data-testid="stDownloadButton"] button {
        background: #0284c7 !important;
        background-color: #0284c7 !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
        height: 28px !important;
        line-height: 28px !important;
        padding: 0 12px !important;
        font-size: 0.85em !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    [data-testid="stDialog"] .stButton button[data-basebuttonstyle="primary"]:hover,
    [data-testid="stDialog"] .stDownloadButton button:hover,
    [data-testid="stDialog"] div[data-testid="stDownloadButton"] button:hover {
        background: #0369a1 !important;
        background-color: #0369a1 !important;
    }

    /* Filename buttons inside the table rows: transparent, cyan/blue color */
    [data-testid="stDialog"] div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(2) button {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        color: #38bdf8 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 2px 6px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    [data-testid="stDialog"] div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(2) button:hover {
        color: #0284c7 !important;
        text-decoration: underline !important;
    }

    /* Status and Size buttons inside the table rows: transparent, centered text */
    [data-testid="stDialog"] div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(3) button,
    [data-testid="stDialog"] div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(4) button {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        color: #cbd5e1 !important;
        text-align: center !important;
        justify-content: center !important;
        font-weight: 400 !important;
        box-shadow: none !important;
    }
    [data-testid="stDialog"] div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(3) button:hover,
    [data-testid="stDialog"] div[data-testid="column"]:nth-of-type(1) div[data-testid="column"]:nth-of-type(4) button:hover {
        color: #38bdf8 !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        text-decoration: underline !important;
    }
    
    /* Style all dialog buttons to have uniform premium heights and widths */
    div[role="dialog"] button {
        height: 28px !important;
        line-height: 28px !important;
        padding: 0 12px !important;
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

# Check if vector_store needs reinitialization due to signature update (Requirement 9)
if "vector_store" in st.session_state:
    try:
        import inspect
        sig = inspect.signature(st.session_state.vector_store.search)
        if "filter_sources" not in sig.parameters:
            print("🔄 FAISSVectorStore.search signature mismatch. Force-clearing st.cache_resource...", flush=True)
            st.cache_resource.clear()
            for k in ["vector_store", "health_status"]:
                if k in st.session_state:
                    del st.session_state[k]
    except Exception as e:
        print(f"⚠️ Error checking signature: {e}", flush=True)

# Startup check trigger at top-level to prevent AttributeError
if "vector_store" not in st.session_state:
    with st.spinner("🔄 Loading RAG index and connecting to Cloudflare R2..."):
        try:
            store, health_status = initialize_rag()
            if store is None:
                raise RuntimeError("Embedding model initialization failed. Application cannot start.")
            st.session_state.vector_store = store
            st.session_state.health_status = health_status
        except Exception as e:
            st.error(f"❌ Embedding model initialization failed. Application cannot start.\n\nError: {e}")
            st.stop()
else:
    if st.session_state.vector_store is None:
        st.error("❌ Embedding model initialization failed. Application cannot start.")
        st.stop()
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

# Compute icon markers
r2_icon = "🟢" if status.get("r2_connected") else "🔴"
embed_icon = "🟢" if status.get("embeddings_loaded") else "🔴"

gemini_reach = status.get("gemini_reachable")
gemini_icon = "🟢" if gemini_reach else ("🔴" if gemini_reach is False else "⚪")
gemini_label = "Gemini API Ready" if gemini_reach else ("Gemini API Unreachable" if gemini_reach is False else "Gemini (Not Configured)")

groq_reach = status.get("groq_reachable")
groq_icon = "🟢" if groq_reach else ("🔴" if groq_reach is False else "⚪")
groq_label = "Groq API Ready" if groq_reach else ("Groq API Unreachable" if groq_reach is False else "Groq (Not Configured)")

# System ready check: need embedding loaded and at least one configured LLM online/reachable
sys_ready = status.get("embeddings_loaded") and (gemini_reach or groq_reach or (gemini_reach is None and groq_reach is None))
sys_icon = "🟢" if sys_ready else "🔴"

st.sidebar.markdown(f"""
- {r2_icon} **Cloud Storage Connected**
- {embed_icon} **Embedding Model Ready**
- {gemini_icon} **{gemini_label}**
- {groq_icon} **{groq_label}**
- {sys_icon} **System Ready**
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
        st.session_state.last_error = "❌ Index is empty. Please upload documents first."
        st.session_state.searching = False
        st.rerun()
    else:
        # ── Step-by-step progress display ────────────────────────────────
        _prog_placeholder = st.empty()
        def _show_step(icon, msg, sub=""):
            _prog_placeholder.markdown(
                f"""
                <div style='background:rgba(30,41,59,0.6);border:1px solid rgba(99,179,237,0.25);border-radius:10px;
                            padding:18px 22px;margin:8px 0;'>
                    <div style='font-size:1.05em;font-weight:600;color:#e2e8f0;'>{icon} {msg}</div>
                    {'<div style="font-size:0.85em;color:#94a3b8;margin-top:4px;">'+sub+'</div>' if sub else ''}
                </div>
                """,
                unsafe_allow_html=True
            )

        sel_docs = list(st.session_state.selected_docs) if st.session_state.get("selected_docs") else None
        scope_label = f"{len(sel_docs)} selected document(s)" if sel_docs else "all documents"

        _show_step("🔍", "Searching knowledge base...", f"Scope: {scope_label}")
        retriever = VectorStoreRetrieverAdapter(
            st.session_state.vector_store,
            k=settings.RETRIEVER_K,
            filter_sources=sel_docs
        )

        _show_step("⚙️", "Initialising AI model...", f"Model: {settings.LLM_MODEL}")
        try:
            qa = create_qa_from_retriever(retriever)
        except Exception as e:
            _prog_placeholder.empty()
            st.session_state.last_error = f"❌ Could not initialise AI model: {e}"
            st.session_state.searching = False
            st.rerun()

        try:
            search_q = q_text
            if sel_docs:
                search_q = f"In documents {sel_docs}: {q_text}"

            _show_step("🧠", "Generating answer...", "This usually takes 3–10 seconds")
            res_tuple = query_with_features(qa, search_q)
            _prog_placeholder.empty()

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
                    st.session_state.last_error = "No response returned. Please verify your API key."
            else:
                st.session_state.search_executed = False
                st.session_state.last_error = "No response returned. Please verify your API key."
        except Exception as e:
            _prog_placeholder.empty()
            st.session_state.search_executed = False
            err_msg = str(e)
            if "All configured AI providers" not in err_msg and "Primary AI provider" not in err_msg:
                if any(x in err_msg.lower() for x in ["429", "resource_exhausted", "quota"]):
                    err_msg = "Primary AI provider is temporarily unavailable. Switching to another provider."
                else:
                    err_msg = "All configured AI providers are currently unavailable. Please try again shortly."
            st.session_state.last_error = f"❌ {err_msg}"

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
        st.error("❌ Missing GOOGLE_API_KEY or GROQ_API_KEY. Please set at least one in Streamlit secrets.")
    else:
        st.session_state.searching = True
        st.rerun()

# Render Search Results if present
if st.session_state.search_executed and st.session_state.get("last_result"):
    result = st.session_state.last_result
    was_cached = st.session_state.get("last_was_cached", False)
    elapsed = st.session_state.get("last_elapsed", 0.0)
    
    # Group sources first so counts are available throughout
    from collections import defaultdict
    sources_group = defaultdict(list)
    for d in result.get("source_documents", []):
        source_name = d.metadata.get("source", "Unknown Document")
        page = d.metadata.get("page", None)
        chunk_id = d.metadata.get("chunk_id", None)
        sources_group[source_name].append((page, chunk_id))

    num_sources = len(sources_group)
    source_label = "Source" if num_sources == 1 else "Sources"
    doc_label   = "Document" if num_sources == 1 else "Documents"

    # ── 1. AI Answer ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("✨ AI Answer")
        st.markdown(result.get("result", "").strip() if result.get("result") else "")

        # ── 2. Sources ────────────────────────────────────────────────────
        if sources_group:
            st.markdown("---")
            st.markdown("#### 📄 Sources")
            for source_name, chunks_info in sources_group.items():
                pages = [p for p, c in chunks_info if p is not None]
                unique_pages = sorted(set(pages)) if pages else []

                st.markdown(f"📄 **{source_name}**")
                if unique_pages:
                    pages_display = ", ".join(str(p + 1) for p in unique_pages)
                    pg_label = "Page" if len(unique_pages) == 1 else "Pages"
                    st.caption(f"{pg_label}: {pages_display}")

        # ── 3. Summary Bar (no Chunks) ────────────────────────────────────
        st.markdown(f"""
        <div style='background-color: rgba(30, 41, 59, 0.35); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 10px 20px; margin-top: 15px; font-size: 0.9em; opacity: 0.9; display: flex; gap: 32px; align-items: center;'>
            <span>⏱️ <strong>{elapsed:.2f} sec</strong></span>
            <span>📄 <strong>{num_sources} {source_label}</strong></span>
            <span>📚 <strong>{num_sources} {doc_label}</strong></span>
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Collapsible Evidence — full technical details kept here ─────────
    with st.expander("🔍 View Context Evidence Snippets"):
        for i, d in enumerate(result.get("source_documents", [])[:6], 1):
            src      = d.metadata.get("source", "unknown")
            page     = d.metadata.get("page")
            chunk_id = d.metadata.get("chunk_id")
            score    = d.metadata.get("score")
            page_str = f" · Page {page + 1}" if page is not None else ""
            st.markdown(f"**{i}. {src}**{page_str}")
            if score is not None:
                st.caption(f"Similarity Score: `{score:.4f}` | Chunk ID: `{chunk_id}`")
            st.text(d.page_content[:400].replace("\n", " "))
            st.markdown("---")

    # ── 5. Feedback ───────────────────────────────────────────────────────
    st.markdown("### Was this helpful?")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 4])
    with col_f1:
        if st.button("👍 Yes", key="feedback_yes_btn"):
            st.success("Thanks for your feedback!")
    with col_f2:
        if st.button("👎 No"):
            feedback = st.text_input("What could be improved?")
            if feedback:
                st.info("Feedback recorded. Thank you!")

# Render the dialog modal if active (stay open across native reruns)
if st.session_state.get("show_doc_manager_dialog", False):
    document_management_dialog()
