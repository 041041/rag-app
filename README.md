# Clinical Trial RAG Application with Cloudflare R2 & FAISS

A Streamlit RAG (Retrieval-Augmented Generation) application configured to ingest clinical trial documents, index them using HuggingFace embeddings (`all-MiniLM-L6-v2`) and FAISS, and query them with Google Gemini (`gemini-2.5-flash`). Persistent storage is managed through Cloudflare R2 with automatic backups and multi-user optimistic concurrency lock protection.

## Architecture Flow

```
Streamlit Application
      │
      ▼
Cloudflare R2 Bucket
 ├── documents/ (original uploaded files)
 ├── indexes/ (FAISS index, metadata mappings)
 └── backups/ (timestamped index history)
      │
      ▼
Local Cache (caches indexes locally at runtime)
 ├── index.faiss
 ├── metadata.pkl
 └── document_metadata.json
      │
      ▼
FAISS IndexIDMap (performs fast vector similarity searches)
      │
      ▼
Google Gemini (LLM response generation)
```

## Features

* **Cloudflare R2 Persistent Storage**: Stores original documents, vector index files, and metadata maps securely on R2 using S3-compatible APIs.
* **FAISS IndexIDMap**: Wraps flat inner product vector index to map document chunks to unique 64-bit integer IDs.
* **Optimistic Concurrency Control**: Uses S3-compatible file locking (`indexes/lock.json`) coupled with local threading locks to serialize multi-user index modifications.
* **Automatic Backups**: Creates server-side index backups inside the `backups/` prefix before any update.
* **SHA-256 Duplicate Prevention**: Compares document hashes before encoding to skip duplicate processing and lower costs.
* **Execution Profiling**: UI timing metrics showing execution duration of chunking, embedding, storage, and search.
* **Decoupled Architecture**: Abstract `VectorStore` base class makes it easy to migrate to other vector databases (Qdrant, Chroma, Supabase, Pinecone, etc.) in the future.

## Prerequisites

* Python 3.9 or higher
* A Cloudflare R2 bucket
* A Google Gemini API key

## Getting Started

### 1. Clone the Repository
Clone this repository to your target server:
```bash
git clone <git-repo-url> clinical_rag_app
cd clinical_rag_app
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
Open `.env` and edit:
```ini
R2_ACCOUNT_ID=your_cloudflare_account_id
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_r2_bucket_name
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 3. Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit application:
```bash
streamlit run app.py
```

## Operations & Maintenance

### Startup Synchronization
The application checks Cloudflare R2 on startup. If index files exist under the `indexes/` folder prefix, they are downloaded and loaded into memory as local cache. If no index files exist, a new empty FAISS index is initialized.

### Ingestion Flow
When you upload documents via the sidebar, the application:
1. Acquires the distributed lock.
2. Downloads the latest index version from R2.
3. Performs a SHA-256 duplicate hash check.
4. Uploads the original file to R2 under `documents/`.
5. Extracts and splits the text into chunks.
6. Generates embeddings and appends them to FAISS.
7. Saves index files locally.
8. Copies current R2 index files to `backups/` with a timestamp.
9. Uploads the new index files to R2.
10. Releases the lock.
