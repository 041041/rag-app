# tests/test_llm_architecture.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.llm import (
    ClinicalRAGLLM,
    FallbackManager,
    is_transient_error,
    is_provider_disabled,
    activate_circuit_breaker,
    circuit_breakers
)
from rag.faiss_store import get_embeddings_model
from app import initialize_rag, SimpleQAWrapper
from config import settings

class TestLLMArchitecture(unittest.TestCase):
    def setUp(self):
        # Clear circuit breakers before each test
        circuit_breakers.clear()
        
        # Setup environment variables for test
        os.environ["GOOGLE_API_KEY"] = "mock-gemini-key"
        os.environ["GROQ_API_KEY"] = "mock-groq-key"

    def test_transient_error_detection(self):
        """Verify that only transient errors are retried."""
        self.assertTrue(is_transient_error(RuntimeError("502 Bad Gateway")))
        self.assertTrue(is_transient_error(TimeoutError("Connection timed out")))
        
        self.assertFalse(is_transient_error(ValueError("429 Too Many Requests")))
        self.assertFalse(is_transient_error(RuntimeError("RESOURCE_EXHAUSTED")))
        self.assertFalse(is_transient_error(ValueError("404 Not Found")))
        self.assertFalse(is_transient_error(ValueError("401 Unauthorized")))

    @patch("rag.llm.GeminiProvider.invoke")
    @patch("rag.llm.GroqProvider.invoke")
    def test_gemini_429_immediate_fallback(self, mock_groq_invoke, mock_gemini_invoke):
        """Test that Gemini 429 triggers immediate fallback to Groq without retries."""
        # Gemini throws a 429
        mock_gemini_invoke.side_effect = Exception("429 Resource Exhausted")
        mock_groq_invoke.return_value = "Mock Groq Response"

        manager = FallbackManager(
            primary_provider="gemini",
            gemini_model="gemini-2.0-flash",
            groq_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        )

        response = manager.invoke("test prompt")

        # Verify response matches Groq output
        self.assertEqual(response, "Mock Groq Response")
        # Verify Gemini was invoked exactly once (no retries for 429)
        mock_gemini_invoke.assert_called_once()
        # Verify Groq was invoked
        mock_groq_invoke.assert_called_once()
        # Verify Gemini circuit breaker was activated
        self.assertTrue(is_provider_disabled("gemini"))

    @patch("rag.llm.ChatGroq")
    def test_groq_primary_unavailable_switch_to_secondary(self, mock_chat_groq):
        """Test that if Groq primary returns 404, it immediately switches to secondary model."""
        mock_primary_client = MagicMock()
        mock_secondary_client = MagicMock()
        
        # Primary client throws 404 on invoke
        mock_primary_client.invoke.side_effect = Exception("404 Model Not Found")
        mock_secondary_client.invoke.return_value = MagicMock(content="Mock Secondary Response")
        
        def mock_init(model_name, **kwargs):
            if model_name == "llama-3.3-70b-versatile":
                return mock_primary_client
            return mock_secondary_client
            
        mock_chat_groq.side_effect = mock_init

        manager = FallbackManager(
            primary_provider="groq",
            gemini_model="gemini-2.0-flash",
            groq_models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        )

        response = manager.invoke("test prompt")
        # Handle if response is AIMessage/Mock with content attribute
        self.assertEqual(response.content, "Mock Secondary Response")

    @patch("rag.faiss_store.HuggingFaceEmbeddings")
    def test_offline_embeddings_fallback(self, mock_hf_embeddings):
        """Verify embeddings loading flow preferred order fallback."""
        # Clear Streamlit cache to force execution of the function body
        get_embeddings_model.clear()
        
        # Mock load failure for BAAI/bge-small-en-v1.5 to trigger all-MiniLM fallback
        mock_hf_embeddings.side_effect = [Exception("Download blocked"), MagicMock()]
        
        model = get_embeddings_model()
        self.assertIsNotNone(model)
        # Should call HuggingFaceEmbeddings twice (once for BAAI, then fallback to MiniLM)
        self.assertEqual(mock_hf_embeddings.call_count, 2)

    @patch("app.FAISSVectorStore")
    @patch("storage.r2_storage.verify_connection")
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    @patch("langchain_groq.ChatGroq")
    def test_startup_health_checks(self, mock_chat_groq, mock_chat_gemini, mock_verify_r2, mock_faiss_store):
        """Test that startup health checks successfully diagnostic verify system state."""
        # Clear Streamlit cache to force execution of the function body
        initialize_rag.clear()
        
        mock_verify_r2.return_value = True
        
        store, health_status = initialize_rag()
        
        self.assertTrue(health_status["r2_connected"])
        self.assertTrue(health_status["embeddings_loaded"])
        self.assertIn("gemini_reachable", health_status)
        self.assertIn("groq_reachable", health_status)

    @patch("rag.indexing.r2_storage.list_files")
    @patch("rag.indexing.r2_storage.download_file")
    @patch("rag.indexing.r2_storage.upload_file")
    @patch("rag.indexing.r2_storage.backup_indexes")
    @patch("rag.indexing._get_loader_for_path")
    @patch("rag.indexing.compute_file_hash")
    @patch("rag.faiss_store.get_embeddings_model")
    def test_rebuild_index_from_r2_docs(self, mock_get_embeddings, mock_hash, mock_get_loader, mock_backup, mock_upload, mock_download, mock_list):
        """Verify that index rebuilding flow correctly lists, downloads, and processes R2 files."""
        mock_list.return_value = ["documents/ADaM_IG.pdf", "documents/sdtm_ig.pdf"]
        mock_download.return_value = True
        mock_upload.return_value = True
        mock_backup.return_value = True
        mock_hash.return_value = "mock_hash_val"
        
        # Mock embeddings model
        mock_embed = MagicMock()
        mock_embed.embed_documents.return_value = [[0.1] * 384]
        mock_get_embeddings.return_value = mock_embed
        
        # Mock document loader
        mock_loader_cls = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = [MagicMock(page_content="Mock page content", metadata={})]
        mock_loader_cls.return_value = mock_loader
        mock_get_loader.return_value = mock_loader_cls
        
        from rag.indexing import rebuild_index_from_r2_docs
        mock_store = MagicMock()
        mock_store.add_documents.return_value = [0, 1]
        
        # Patch local write_bytes
        with patch("pathlib.Path.read_bytes", return_value=b"mock pdf bytes"), \
             patch("pathlib.Path.write_bytes"), \
             patch("pathlib.Path.mkdir"), \
             patch("builtins.open", unittest.mock.mock_open()):
             
            res = rebuild_index_from_r2_docs(mock_store)
            
            self.assertEqual(res["status"], "success")
            self.assertIn("ADaM_IG.pdf", res["processed_files"])
            self.assertIn("sdtm_ig.pdf", res["processed_files"])
            self.assertEqual(mock_list.call_count, 1)
            self.assertEqual(mock_download.call_count, 2)
            self.assertEqual(mock_store.load.call_count, 1)

if __name__ == "__main__":
    unittest.main()
