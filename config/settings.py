# config/settings.py
import os
from pathlib import Path

# Base Paths (resolve absolute path relative to workspace)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEXES_DIR = BASE_DIR / "indexes"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

# Local cache file paths
FAISS_INDEX_FILE = "index.faiss"
METADATA_PKL_FILE = "metadata.pkl"
DOCUMENT_METADATA_JSON_FILE = "document_metadata.json"

# Helper to load from Streamlit Secrets defensively, falling back to Environment variables
def get_secret(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# Credentials and endpoints
R2_ACCOUNT_ID = get_secret("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = get_secret("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = get_secret("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = get_secret("R2_BUCKET_NAME", "")
R2_ENDPOINT = get_secret("R2_ENDPOINT", "")

# Dynamically construct endpoint if not provided but account ID exists
if not R2_ENDPOINT and R2_ACCOUNT_ID:
    R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# R2 Bucket Prefixes (Folders)
R2_DOCUMENTS_PREFIX = "documents/"
R2_INDEXES_PREFIX = "indexes/"
R2_BACKUPS_PREFIX = "backups/"

# RAG specific configurations
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
RETRIEVER_K = 8
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
