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
    if not isinstance(text, str):
        return text
    
    import re
    
    raw_len = len(text)
    
    # Check if there is a <think> tag
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    
    if think_match:
        think_content = think_match.group(1).strip()
        # Find everything after </think>
        parts = text.split("</think>")
        after_think = parts[1].strip() if len(parts) > 1 else ""
        
        # 1. If content exists after </think>:
        if after_think:
            final_text = after_think
        else:
            # 2. Search inside think block for Draft - Short Definition, Draft, Answer:
            start_markers = [
                r"Draft\s*-\s*Short\s+Definition\s*:",
                r"Draft\s*-\s*Key\s+Points\s*:",
                r"Draft\s*:",
                r"Answer\s*:",
                r"Response\s*:"
            ]
            
            earliest_idx = -1
            matched_len = 0
            for marker in start_markers:
                m = re.search(marker, think_content, flags=re.IGNORECASE)
                if m:
                    idx = m.start()
                    if earliest_idx == -1 or idx < earliest_idx:
                        earliest_idx = idx
                        matched_len = m.end() - m.start()
            
            if earliest_idx != -1:
                extracted = think_content[earliest_idx + matched_len:].strip()
                
                # Stop extraction before the stop markers
                stop_markers = [
                    r"Final\s+Review",
                    r"Self-Correction",
                    r"Output\s+Generation",
                    r"Everything\s+looks\s+solid",
                    r"Output\s+matches\s+response",
                    r"Check\s+formatting\s+rules"
                ]
                
                earliest_stop_idx = -1
                for stop_marker in stop_markers:
                    sm = re.search(stop_marker, extracted, flags=re.IGNORECASE)
                    if sm:
                        s_idx = sm.start()
                        if earliest_stop_idx == -1 or s_idx < earliest_stop_idx:
                            earliest_stop_idx = s_idx
                
                if earliest_stop_idx != -1:
                    final_text = extracted[:earliest_stop_idx].strip()
                else:
                    final_text = extracted
            else:
                final_text = think_content
    else:
        # Remove any stray tags
        final_text = text.replace("<think>", "").replace("</think>", "").strip()
        
    # 3. Remove these sections (Run unconditionally for idempotency):
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
        final_text = re.sub(rf"(?:^|\n)#*\s*\**{section}\**[^\n]*", "", final_text, flags=re.IGNORECASE)
        
    # Also strip any remaining draft labels from the final text
    draft_labels = [
        r"Draft\s*-\s*Short\s+Definition\s*:",
        r"Draft\s*-\s*Key\s+Points\s*:",
        r"Draft\s*:",
        r"Answer\s*:",
        r"Response\s*:"
    ]
    for label in draft_labels:
        final_text = re.sub(rf"(?:^|\n)#*\s*\**{label}\**[^\n]*", "", final_text, flags=re.IGNORECASE)
        
    final_text = final_text.strip()
    cleaned_len = len(final_text)
    
    logger.info(f"Raw answer length: {raw_len}")
    logger.info(f"Clean answer length: {cleaned_len}")
    logger.info(f"Clean answer preview: {final_text[:200]}...")
    
    return final_text

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
            temperature=0.2,
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
            temperature=0.2,
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
            logger.info(f"🤖 Selected LLM Provider: {provider_name.capitalize()} | Model: {model_name}")

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
        
        groq_primary = os.getenv("GROQ_PRIMARY_MODEL", os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"))
        groq_secondary = os.getenv("GROQ_SECONDARY_MODEL", "openai/gpt-oss-20b")
        
        # Override deprecated model names defensively
        if "llama-3.3-70b" in groq_primary or "versatile" in groq_primary:
            groq_primary = "qwen/qwen3.6-27b"
        if "llama-3.3-70b" in groq_secondary or "versatile" in groq_secondary:
            groq_secondary = "openai/gpt-oss-20b"
            
        groq_models = [groq_primary, groq_secondary]
        
        self.manager = FallbackManager(
            primary_provider=primary_provider,
            gemini_model=gemini_model,
            groq_models=groq_models
        )

    def invoke(self, prompt: str) -> AIMessage:
        response = self.manager.invoke(prompt)
        if response is not None and hasattr(response, "content") and response.content:
            response.content = clean_llm_response(response.content)
        return response
