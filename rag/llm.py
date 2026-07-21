# rag/llm.py
import os
import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

from config import settings

logger = logging.getLogger("RAGApp.LLM")
logger.setLevel(logging.INFO)

# --- Safe message schema imports ---
try:
    from langchain_core.messages import AIMessage, HumanMessage
except ImportError:
    try:
        from langchain.schema import AIMessage, HumanMessage
    except ImportError:
        class AIMessage:
            def __init__(self, content: str):
                self.content = content
        class HumanMessage:
            def __init__(self, content: str):
                self.content = content

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
    logger.info("Successfully monkey-patched google.genai BaseApiClient to disable SDK-level retries.")
except Exception as e:
    logger.warning(f"Failed to monkey-patch google.genai BaseApiClient: {e}")

def clean_llm_response(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    
    import re
    
    raw_len = len(text)
    
    # 1. First, check if there's a think tag
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if think_match:
        parts = re.split(r"</think>", text, flags=re.IGNORECASE)
        text = parts[1].strip() if len(parts) > 1 else ""
    
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    
    # 2. Check if any reasoning headers exist in the text (even without think tags)
    reasoning_headers = [
        r"Analyze\s+User\s+Input\s*:",
        r"Extract\s+Information\s+from\s+Context\s*:",
        r"Draft\s*-\s*Short\s+Definition\s*:",
        r"Draft\s+Response\s*:",
        r"Mental\s+Refinement\s*:",
        r"Check\s+Constraints\s*:",
        r"Final\s+Review\s*:",
        r"Output\s+Generation\s*:"
    ]
    
    has_reasoning = False
    for pattern in reasoning_headers:
        if re.search(pattern, text, flags=re.IGNORECASE):
            has_reasoning = True
            break
            
    if has_reasoning:
        # Extract only the final user-facing answer.
        output_gen_match = re.search(r"Output\s+Generation\s*:(.*)", text, flags=re.DOTALL | re.IGNORECASE)
        if output_gen_match:
            text = output_gen_match.group(1).strip()
        else:
            draft_resp_match = re.search(r"Draft\s+Response\s*:(.*)", text, flags=re.DOTALL | re.IGNORECASE)
            if draft_resp_match:
                text = draft_resp_match.group(1).strip()
            else:
                draft_def_match = re.search(r"(Draft\s*-\s*Short\s+Definition\s*:.*)", text, flags=re.DOTALL | re.IGNORECASE)
                if draft_def_match:
                    extracted = draft_def_match.group(1).strip()
                    stop_markers = [
                        r"Final\s+Review",
                        r"Self-Correction",
                        r"Output\s+Generation",
                        r"Check\s+formatting\s+rules"
                    ]
                    earliest_stop = -1
                    for sm in stop_markers:
                        m = re.search(sm, extracted, flags=re.IGNORECASE)
                        if m:
                            if earliest_stop == -1 or m.start() < earliest_stop:
                                earliest_stop = m.start()
                    if earliest_stop != -1:
                        text = extracted[:earliest_stop].strip()
                    else:
                        text = extracted

    # Clean up standard headings that might remain
    sections_to_remove = [
        r"Here's\s+a\s+thinking\s+process",
        r"Analyze\s+User\s+Input",
        r"Context\s+analysis",
        r"Format\s+Requirements",
        r"Scan\s+Context",
        r"Synthesize\s+Definition",
        r"Identify\s+Critical\s+Issue",
        r"Final\s+Review",
        r"Self-Correction",
        r"Output\s+Generation",
        r"Reasoning\s+Analysis",
        r"Reasoning:"
    ]
    for section in sections_to_remove:
        text = re.sub(rf"(?:^|\n)#*\s*\**{section}\**[^\n]*", "", text, flags=re.IGNORECASE)

    # Strip any remaining draft labels
    draft_labels = [
        r"Draft\s*-\s*Short\s+Definition\s*:",
        r"Draft\s*-\s*Key\s+Points\s*:",
        r"Draft\s*:",
        r"Answer\s*:",
        r"Response\s*:"
    ]
    for label in draft_labels:
        text = re.sub(rf"(?:^|\n)#*\s*\**{label}\**[^\n]*", "", text, flags=re.IGNORECASE)

    text = text.strip()
    cleaned_len = len(text)
    
    logger.info(f"Raw answer length: {raw_len}")
    logger.info(f"Clean answer length: {cleaned_len}")
    logger.info(f"Clean answer preview: {text[:200]}...")
    
    return text

def ensure_clinical_rag_format(text: str, docs: list) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
        
    import re
    import os
    
    # If the text is the empty context fallback, return it as is
    if "Information not available" in text or "uploaded documents do not provide" in text:
        return text
        
    # Re-check and strip any existing Sources section first to rebuild it fresh
    text = re.sub(r"\n*Sources\s*:.*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Separate definition and key points blocks by looking for "Key points:"
    parts = re.split(r"\bKey\s+points\s*:", text, maxsplit=1, flags=re.IGNORECASE)
    
    definition_text = parts[0].strip()
    # Strip optional "Short definition:" header from definition_text
    definition_text = re.sub(r"^(?:Short\s+)?definition\s*:\s*", "", definition_text, flags=re.IGNORECASE).strip()
    
    bullet_parts = []
    if len(parts) > 1:
        kp_text = parts[1].strip()
        raw_bullets = [line.strip() for line in kp_text.split("\n") if line.strip()]
        for b in raw_bullets:
            b_clean = re.sub(r"^[-•*]\s*", "", b).strip()
            if b_clean:
                bullet_parts.append(b_clean)
                
    if not bullet_parts:
        def_lines = [line.strip() for line in definition_text.split("\n") if line.strip()]
        if len(def_lines) > 1:
            definition_text = def_lines[0]
            bullet_parts = def_lines[1:]
        else:
            bullet_parts = ["No additional key points retrieved."]
            
    # Remove unsupported statements on raw definition_text first
    definition_text = re.sub(
        r"The\s+BDS\s+contains\s+variables\s+that\s+are\s+not\s+specific\s+to\s+ADSL\s+or\s+BDS\s+structures\.?",
        "BDS is a standard ADaM structure used for analysis datasets containing multiple records per subject.",
        definition_text,
        flags=re.IGNORECASE
    )
    
    # Terminology Correction on definition
    definition_text = re.sub(r"\bADSL\s*\(\s*Analysis\s+Dataset\s*\)", "ADSL (Subject-Level Analysis Dataset)", definition_text, flags=re.IGNORECASE)
    definition_text = re.sub(r"\bADSL\s*\(\s*Analysis\s+Subject-Level\s+Dataset\s*\)", "ADSL (Subject-Level Analysis Dataset)", definition_text, flags=re.IGNORECASE)
    definition_text = re.sub(r"\bSubject-Level\s+Analysis\s+Dataset\s*\(\s*ADSL\s*\)", "ADSL (Subject-Level Analysis Dataset)", definition_text, flags=re.IGNORECASE)
    definition_text = re.sub(r"\bAnalysis\s+Dataset\s*\(\s*ADSL\s*\)", "ADSL (Subject-Level Analysis Dataset)", definition_text, flags=re.IGNORECASE)
    
    # First occurrence normalization function
    def normalize_first_occurrence(t: str) -> str:
        adam_pat = r"\bADaM\b(?!\s*\(\s*CDISC\s+Analysis\s+Data\s+Model\s*\))"
        t = re.sub(adam_pat, "ADaM (CDISC Analysis Data Model)", t, count=1, flags=re.IGNORECASE)
        
        adsl_pat = r"\bADSL\b(?!\s*\(\s*Subject-Level\s+Analysis\s+Dataset\s*\))"
        t = re.sub(adsl_pat, "ADSL (Subject-Level Analysis Dataset)", t, count=1, flags=re.IGNORECASE)
        
        sdtm_pat = r"\bSDTM\b(?!\s*\(\s*Study\s+Data\s+Tabulation\s+Model\s*\))"
        t = re.sub(sdtm_pat, "SDTM (Study Data Tabulation Model)", t, count=1, flags=re.IGNORECASE)
        
        bds_pat = r"\bBDS\b(?!\s*\(\s*Basic\s+Data\s+Structure\s*\))"
        t = re.sub(bds_pat, "BDS (Basic Data Structure)", t, count=1, flags=re.IGNORECASE)
        
        occds_pat = r"\bOCCDS\b(?!\s*\(\s*Occurrence\s+Data\s+Structure\s*\))"
        t = re.sub(occds_pat, "OCCDS (Occurrence Data Structure)", t, count=1, flags=re.IGNORECASE)
        
        return t

    # Normalize first occurrence of terminology
    definition_text = normalize_first_occurrence(definition_text)
    
    # Ensure every bullet point ends with a citation and starts with "- "
    updated_bullets = []
    default_citation = ""
    def_cit_match = re.search(r"\(\s*Source:\s*[^,)]+?,\s*Page:?\s*\d+\s*\)", definition_text, flags=re.IGNORECASE)
    if def_cit_match:
        default_citation = def_cit_match.group(0)
    elif docs:
        src = docs[0].metadata.get("source")
        page = docs[0].metadata.get("page")
        if src:
            default_citation = f"(Source: {os.path.basename(src)}, Page: {page + 1})"
            
    for bullet in bullet_parts:
        bullet = re.sub(r"\bADSL\s*\(\s*Analysis\s+Dataset\s*\)", "ADSL (Subject-Level Analysis Dataset)", bullet, flags=re.IGNORECASE)
        bullet = re.sub(r"\bSubject-Level\s+Analysis\s+Dataset\s*\(\s*ADSL\s*\)", "ADSL (Subject-Level Analysis Dataset)", bullet, flags=re.IGNORECASE)
        bullet = re.sub(r"^[-•*]\s*", "", bullet).strip()
        
        if not re.search(r"\(\s*Source:\s*[^,)]+?,\s*Page:?\s*\d+\s*\)", bullet, flags=re.IGNORECASE):
            if default_citation:
                bullet = f"{bullet} {default_citation}"
                
        # Ensure Page: X formatting
        bullet = re.sub(r"\bPage:?\s*(\d+)", r"Page: \1", bullet, flags=re.IGNORECASE)
        updated_bullets.append(f"- {bullet}")
        
    bullets_text = "\n\n".join(updated_bullets)
    
    # Ensure Page: X formatting in definition
    definition_text = re.sub(r"\bPage:?\s*(\d+)", r"Page: \1", definition_text, flags=re.IGNORECASE)
    
    # Build unique sources list from the passed docs
    unique_sources = []
    if docs:
        for doc in docs:
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            if src:
                src_name = os.path.basename(src)
                unique_sources.append(f"- {src_name}, Page: {page + 1}")
        unique_sources = sorted(list(set(unique_sources)))
        
    if not unique_sources:
        combined_text = definition_text + "\n" + bullets_text
        citations = re.findall(r"\(\s*Source:\s*([^,)]+?),\s*Page:?\s*\d+\s*\)", combined_text, flags=re.IGNORECASE)
        for c_src, c_pg in citations:
            c_src_name = os.path.basename(c_src.strip())
            unique_sources.append(f"- {c_src_name}, Page: {c_pg}")
        unique_sources = sorted(list(set(unique_sources)))
        
    if not unique_sources:
        sources_text = "- Sources details not available."
    else:
        sources_text = "\n".join(unique_sources)
        
    restructured = (
        f"{definition_text}\n\n"
        f"Key points:\n\n{bullets_text}\n\n"
        f"Sources:\n\n{sources_text}"
    )
    
    # Clean duplicate parenthetical acronym expansion patterns
    restructured = re.sub(
        r"\bADaM\s*\(\s*CDISC\s+Analysis\s+Data\s+Model\s*\)\s+stands\s+for\s+(?:the\s+)?(?:CDISC\s+)?Analysis\s+Data\s+Model",
        "ADaM (CDISC Analysis Data Model) is a standard for clinical trial analysis datasets.",
        restructured,
        flags=re.IGNORECASE
    )
    restructured = re.sub(
        r"\bADaM\s*\(\s*CDISC\s+Analysis\s+Data\s+Model\s*\)\s+is\s+(?:the\s+)?(?:CDISC\s+)?Analysis\s+Data\s+Model",
        "ADaM (CDISC Analysis Data Model) is a standard for clinical trial analysis datasets.",
        restructured,
        flags=re.IGNORECASE
    )
    restructured = re.sub(
        r"\bADaM\s+stands\s+for\s+(?:the\s+)?(?:CDISC\s+)?Analysis\s+Data\s+Model",
        "ADaM (CDISC Analysis Data Model) is a standard for clinical trial analysis datasets.",
        restructured,
        flags=re.IGNORECASE
    )
    restructured = re.sub(
        r"\bADaM\s*\(\s*CDISC\s+Analysis\s+Data\s+Model\s*\)\s*\(\s*(?:CDISC\s+)?Analysis\s+Data\s+Model\s*\)",
        "ADaM (CDISC Analysis Data Model)",
        restructured,
        flags=re.IGNORECASE
    )
    
    # Clean duplicate parenthetical expansions for ADSL
    restructured = re.sub(
        r"\bADSL\s*\(\s*Subject-Level\s+Analysis\s+Dataset\s*\)\s*\(\s*(?:Subject-Level\s+)?Analysis\s+Dataset\s*\)",
        "ADSL (Subject-Level Analysis Dataset)",
        restructured,
        flags=re.IGNORECASE
    )
    
    # Clean duplicate parenthetical expansions for BDS
    restructured = re.sub(
        r"\bBDS\s*\(\s*Basic\s+Data\s+Structure\s*\)\s*\(\s*Basic\s+Data\s+Structure\s*\)",
        "BDS (Basic Data Structure)",
        restructured,
        flags=re.IGNORECASE
    )
    
    # Clean duplicate parenthetical expansions for OCCDS
    restructured = re.sub(
        r"\bOCCDS\s*\(\s*Occurrence\s+Data\s+Structure\s*\)\s*\(\s*Occurrence\s+Data\s+Structure\s*\)",
        "OCCDS (Occurrence Data Structure)",
        restructured,
        flags=re.IGNORECASE
    )
    
    return restructured

# Try to import providers safely
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

# Global state for circuit breakers (persists in process memory across reruns)
circuit_breakers = {}  # e.g., {"gemini": cooldown_until_timestamp}
circuit_breaker_lock = threading.Lock()

def activate_circuit_breaker(provider_name: str, duration_seconds: float = 900.0):
    """
    Disables the specified provider for duration_seconds (default 15 minutes).
    """
    with circuit_breaker_lock:
        cooldown_until = time.time() + duration_seconds
        circuit_breakers[provider_name] = cooldown_until
        logger.warning(
            f"🔌 Circuit breaker activated for provider '{provider_name.capitalize()}'. "
            f"Disabled until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cooldown_until))}."
        )

def is_provider_disabled(provider_name: str) -> bool:
    """
    Checks if a provider is currently disabled by the circuit breaker.
    """
    with circuit_breaker_lock:
        cooldown_until = circuit_breakers.get(provider_name, 0.0)
        if time.time() < cooldown_until:
            return True
        if provider_name in circuit_breakers:
            del circuit_breakers[provider_name]
            logger.info(f"🔄 Circuit breaker cooldown expired for provider '{provider_name.capitalize()}'. Re-enabling.")
        return False

def is_transient_error(exc: Exception) -> bool:
    """
    Retry policy check: returns True only for ConnectionError, Timeout, and HTTP 5xx.
    Returns False for auth (401/403), not found (404), and rate-limit/exhausted (429).
    """
    err_msg = str(exc).lower()

    # 1. Do NOT retry these (fail-fast or switch providers immediately)
    if any(q in err_msg for q in ["429", "resource_exhausted", "rate_limit", "quota exceeded", "quota_exceeded"]):
        return False
    if any(a in err_msg for a in ["401", "unauthorized", "invalid_api_key", "403", "forbidden"]):
        return False
    if any(n in err_msg for n in ["404", "not_found", "not supported", "unknown model", "unsupported model"]):
        return False

    # 2. Retry only transient errors
    import socket
    if isinstance(exc, (TimeoutError, socket.timeout, socket.gaierror)):
        return True
    if any(t in err_msg for t in ["timeout", "timed out", "connection", "connect", "httpexception"]):
        return True
    if any(s in err_msg for s in ["500", "502", "503", "504", "bad gateway", "service unavailable"]):
        return True

    return False


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass


class GeminiProvider(LLMProvider):
    """
    Gemini LLM Provider wrapper.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        if ChatGoogleGenerativeAI is None:
            raise ImportError("ChatGoogleGenerativeAI (langchain-google-genai) is not available.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.0,
            convert_system_message_to_human=True,
            google_api_key=api_key,
            max_retries=0  # Prevents internal retries on 429
        )
        logger.info(f"Initialized primary Gemini LLM with model: {self.model_name}")

    def invoke(self, prompt: str) -> AIMessage:
        response = self.client.invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        return AIMessage(content=text)

    def get_model_name(self) -> str:
        return self.model_name


class GroqProvider(LLMProvider):
    """
    Groq LLM Provider wrapper with automatic model selection/fallback.
    """
    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.current_model_idx = 0
        if ChatGroq is None:
            raise ImportError("ChatGroq (langchain-groq) is not available.")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        
        # Instantiate current client at startup
        self._init_current_client()

    def _init_current_client(self):
        model_name = self.model_names[self.current_model_idx]
        self.client = ChatGroq(
            model_name=model_name,
            temperature=0.0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_retries=0  # Prevents internal retries on Groq errors
        )
        logger.info(f"Initialized fallback Groq LLM with model: {model_name}")

    def invoke(self, prompt: str) -> AIMessage:
        last_error = None
        while self.current_model_idx < len(self.model_names):
            model_name = self.model_names[self.current_model_idx]
            try:
                if self.client is None:
                    self._init_current_client()
                response = self.client.invoke([HumanMessage(content=prompt)])
                text = response.content if hasattr(response, "content") else str(response)
                return AIMessage(content=text)
            except Exception as e:
                err_msg = str(e).lower()
                if any(x in err_msg for x in ["404", "not_found", "not supported", "unknown model", "unsupported model"]):
                    logger.warning(
                        f"⚠️ Groq model '{model_name}' is unavailable/deprecated. "
                        f"Error: {e}. Switching to secondary model..."
                    )
                    self.current_model_idx += 1
                    self.client = None
                    last_error = e
                else:
                    raise e
                    
        raise RuntimeError(f"All configured Groq models failed. Last error: {last_error}")

    def get_model_name(self) -> str:
        if self.current_model_idx < len(self.model_names):
            return self.model_names[self.current_model_idx]
        return "groq-fallback-exhausted"


class FallbackManager:
    """
    Handles primary/secondary LLM provider selection, routing, retries, and circuit breaking.
    """
    def __init__(self, primary_provider: str, gemini_model: str, groq_models: List[str]):
        self.primary_provider = primary_provider.strip().lower()
        self.gemini_model = gemini_model
        self.groq_models = groq_models
        self.providers = {}
        
        # Instantiate clients once during initialization to avoid repeated initialization logs
        if os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI is not None:
            self.providers["gemini"] = GeminiProvider(self.gemini_model)
        if os.getenv("GROQ_API_KEY") and ChatGroq is not None:
            self.providers["groq"] = GroqProvider(self.groq_models)

    def invoke(self, prompt: str) -> AIMessage:
        # Check available providers based on API keys and code imports
        available_providers = []
        
        # We consistently use GOOGLE_API_KEY as the single Gemini credential
        if os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI is not None:
            available_providers.append("gemini")
        if os.getenv("GROQ_API_KEY") and ChatGroq is not None:
            available_providers.append("groq")

        if not available_providers:
            friendly_err = "No LLM providers are configured. Please set GOOGLE_API_KEY or GROQ_API_KEY."
            logger.critical(friendly_err)
            raise RuntimeError(friendly_err)

        # Build order
        if self.primary_provider == "groq" and "groq" in available_providers:
            order = ["groq", "gemini"]
        else:
            order = ["gemini", "groq"]

        # Keep only configured/available providers in the order
        execution_order = [p for p in order if p in available_providers]

        last_error = None
        for provider_name in execution_order:
            if is_provider_disabled(provider_name):
                logger.warning(f"🔌 Bypassing provider '{provider_name.capitalize()}' (Circuit Breaker is active)")
                continue

            provider = self.providers.get(provider_name)
            if provider is None:
                # Instantiate on-demand fallback
                if provider_name == "gemini":
                    provider = GeminiProvider(self.gemini_model)
                else:
                    provider = GroqProvider(self.groq_models)
                self.providers[provider_name] = provider

            model_name = provider.get_model_name()
            logger.info("Using LLM provider:")
            logger.info(f"Model: {model_name}")

            if provider_name == "groq":
                logger.info("👉 [FALLBACK BRANCH REACHED] Fallback to Groq triggered.")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    if provider_name == "groq":
                        logger.info(f"👉 Calling Groq LLM (model: {model_name}) now...")
                    response = provider.invoke(prompt)
                    latency = time.time() - start_time
                    if provider_name == "groq":
                        logger.info("👉 Groq fallback returned response successfully.")
                    logger.info(f"✅ Completed | Latency: {latency:.2f} sec")
                    return response
                except Exception as e:
                    if provider_name == "groq":
                        logger.error(f"❌ Groq fallback invocation failed: {e}")
                    err_msg = str(e)
                    
                    # Detect quota exhaustion (429)
                    is_quota = any(q in err_msg.lower() for q in ["429", "resource_exhausted", "rate_limit", "quota exceeded"])
                    
                    if is_quota:
                        logger.warning(f"❌ Provider '{provider_name.capitalize()}' failed | Reason: RESOURCE_EXHAUSTED")
                        activate_circuit_breaker(provider_name)
                        break  # Fall back to next provider immediately
                    
                    # Detect non-transient errors (such as 401, 403, 404)
                    if not is_transient_error(e):
                        logger.error(f"❌ Non-transient error on '{provider_name.capitalize()}': {e}")
                        last_error = e
                        break  # Fall back to next provider immediately

                    # Handle transient errors (Connection, Timeout, 5xx)
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"⚠️ Transient error on '{provider_name.capitalize()}' (Attempt {attempt+1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
            else:
                logger.warning(f"⚠️ All retries exhausted for provider '{provider_name.capitalize()}'. Switching provider...")

        # If all options failed
        friendly_msg = "Primary AI provider is temporarily unavailable. Switching to another provider."
        if len(execution_order) == 1 or last_error:
            friendly_msg = "All configured AI providers are currently unavailable. Please try again shortly."
            
        logger.error(f"❌ LLM Fallback exhausted. Friendly error returned: {friendly_msg}")
        raise RuntimeError(friendly_msg)


class ClinicalRAGLLM:
    """
    Single public interface for LLM provider invocation.
    Uses configurable primary/fallback logic.
    """
    def __init__(self):
        primary_provider = os.getenv("PRIMARY_PROVIDER", "gemini").lower()
        gemini_model = settings.LLM_MODEL
        
        groq_primary = os.getenv("GROQ_PRIMARY_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        groq_secondary = os.getenv("GROQ_SECONDARY_MODEL", "llama-3.1-8b-instant")
        
        groq_models = [groq_primary, groq_secondary]
        
        logger.info("Initialized Gemini:")
        logger.info(f"{gemini_model}")
        logger.info("Initialized Groq fallback:")
        logger.info(f"{groq_primary}")
        logger.info("Secondary fallback:")
        logger.info(f"{groq_secondary}")
        
        self.manager = FallbackManager(
            primary_provider=primary_provider,
            gemini_model=gemini_model,
            groq_models=groq_models
        )

    def invoke(self, prompt: str) -> AIMessage:
        logger.info("ACTIVE QA PROMPT VERSION: CLEAN_V2")
        response = self.manager.invoke(prompt)
        if response is not None and hasattr(response, "content") and response.content:
            response.content = clean_llm_response(response.content)
            
        cleaned_text = response.content if response is not None else ""
        logger.info(f"Final cleaned response length: {len(cleaned_text)}")
        logger.info(f"Final response preview: {cleaned_text[:200]}...")
        
        return response
