# scratch/rebuild_index.py
import os
import sys
from pathlib import Path

# Add parent path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load dotenv if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RebuildIndex")

from config import settings
from storage import r2_storage
from rag.faiss_store import FAISSVectorStore
from rag.indexing import rebuild_index_from_r2_docs

def main():
    print("🚀 Starting R2 Index Rebuild CLI Tool...")
    print(f"R2 Endpoint: {settings.R2_ENDPOINT}")
    print(f"R2 Bucket: {settings.R2_BUCKET_NAME}")
    print(f"R2 Documents Prefix: {settings.R2_DOCUMENTS_PREFIX}")
    
    # 1. Verify R2 connection
    print("🔌 Verifying R2 connection...")
    r2_connected = r2_storage.verify_connection()
    if not r2_connected:
        print("❌ Error: Could not connect to Cloudflare R2 bucket. Verify credentials in .env.")
        sys.exit(1)
        
    print("✅ R2 Connected successfully.")
    
    # 2. Rebuild Lock Owner ID
    owner_id = f"rebuild_cli_{int(os.getpid())}"
    print(f"🔒 Acquiring distributed lock as owner: {owner_id}...")
    
    lock_acquired = r2_storage.acquire_lock(owner_id, timeout_seconds=60)
    if not lock_acquired:
        print("❌ Error: Locking Conflict. Another process is currently editing the index.")
        sys.exit(1)
        
    try:
        # Create temporary store
        print("🛠️ Initializing empty local store...")
        store = FAISSVectorStore()
        
        # Rebuild!
        print("🔄 Listing R2 files and downloading/indexing all of them...")
        res = rebuild_index_from_r2_docs(store)
        
        if res["status"] == "success":
            print(f"✅ Success! Rebuilt index with {len(res['processed_files'])} files:")
            for f in res["processed_files"]:
                print(f"  - {f}")
        elif res["status"] == "empty":
            print("⚠️ Warning: No files found under documents/ prefix in R2 to index.")
        else:
            print(f"❌ Error: Rebuild failed: {res.get('errors')}")
            sys.exit(1)
            
    finally:
        print("🔓 Releasing distributed lock...")
        r2_storage.release_lock(owner_id)
        print("✅ Finished.")

if __name__ == "__main__":
    main()
