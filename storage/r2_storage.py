# storage/r2_storage.py
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from config import settings

logger = logging.getLogger("RAGApp.Storage")
logger.setLevel(logging.INFO)

# Threading lock for local in-process concurrency safety
import threading
_local_lock = threading.Lock()

def get_r2_client():
    """
    Initialize and return a boto3 client connected to Cloudflare R2.
    """
    if not settings.R2_ACCESS_KEY_ID or not settings.R2_SECRET_ACCESS_KEY:
        raise ValueError("Missing Cloudflare R2 credentials (access key or secret key).")
    
    # Configure custom timeout and retry behavior for robustness
    config = Config(
        retries={'max_attempts': 3, 'mode': 'standard'},
        connect_timeout=5,
        read_timeout=10
    )
    
    return boto3.client(
        's3',
        endpoint_url=settings.R2_ENDPOINT,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=config
    )

def verify_connection() -> bool:
    """
    Startup Health Check: Verifies connection and checks if the bucket exists.
    """
    try:
        logger.info("Connecting to R2...")
        client = get_r2_client()
        # head_bucket returns metadata about the bucket if it exists and we have permissions
        client.head_bucket(Bucket=settings.R2_BUCKET_NAME)
        logger.info("R2 Connection verified successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to verify R2 connection: {e}")
        return False

def upload_file(local_path: Path, r2_key: str) -> bool:
    """
    Uploads a local file to a specific key in the Cloudflare R2 bucket.
    """
    t0 = time.time()
    logger.info(f"Uploading file: {local_path} to R2 key: {r2_key}...")
    try:
        client = get_r2_client()
        client.upload_file(str(local_path), settings.R2_BUCKET_NAME, r2_key)
        elapsed = time.time() - t0
        logger.info(f"Uploading document completed in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to R2 ({r2_key}): {e}")
        return False

def download_file(r2_key: str, local_path: Path) -> bool:
    """
    Downloads a file from Cloudflare R2 bucket to a local path.
    """
    t0 = time.time()
    logger.info(f"Downloading R2 key: {r2_key} to {local_path}...")
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client = get_r2_client()
        client.download_file(settings.R2_BUCKET_NAME, r2_key, str(local_path))
        elapsed = time.time() - t0
        logger.info(f"Downloading completed in {elapsed:.2f}s")
        return True
    except ClientError as e:
        # If object is not found, return False instead of raising exception (helps startup logic)
        if e.response.get('Error', {}).get('Code') == '404':
            logger.warning(f"R2 key: {r2_key} not found (404).")
            return False
        logger.error(f"ClientError downloading {r2_key}: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to download {r2_key} from R2: {e}")
        return False

def check_file_exists(r2_key: str) -> bool:
    """
    Checks if a file key exists in the R2 bucket.
    """
    try:
        client = get_r2_client()
        client.head_object(Bucket=settings.R2_BUCKET_NAME, Key=r2_key)
        return True
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in ['404', 'NoSuchKey']:
            return False
        logger.error(f"Error checking if {r2_key} exists: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking if {r2_key} exists: {e}")
        return False

def backup_indexes() -> bool:
    """
    Backup the existing index files in the indexes/ folder on R2 to the backups/ folder.
    Filenames are formatted with a timestamp.
    """
    t0 = time.time()
    logger.info("Starting index backup process on R2...")
    try:
        client = get_r2_client()
        bucket = settings.R2_BUCKET_NAME
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        index_files = [
            settings.FAISS_INDEX_FILE,
            settings.METADATA_PKL_FILE,
            settings.DOCUMENT_METADATA_JSON_FILE
        ]
        
        backed_up_count = 0
        for f in index_files:
            source_key = f"{settings.R2_INDEXES_PREFIX}{f}"
            
            # Extract filename components for backup formatting
            path_obj = Path(f)
            stem, suffix = path_obj.stem, path_obj.suffix
            dest_key = f"{settings.R2_BACKUPS_PREFIX}{stem}_{timestamp}{suffix}"
            
            if check_file_exists(source_key):
                logger.info(f"Copying R2 source {source_key} to backup {dest_key}...")
                client.copy_object(
                    Bucket=bucket,
                    CopySource={'Bucket': bucket, 'Key': source_key},
                    Key=dest_key
                )
                backed_up_count += 1
            else:
                logger.warning(f"Skip backing up {source_key} - file does not exist on R2.")
                
        elapsed = time.time() - t0
        logger.info(f"Backup completed. Backed up {backed_up_count} files in {elapsed:.2f}s.")
        return True
    except Exception as e:
        logger.error(f"Failed to backup existing index files: {e}")
        return False

# Lock variables
LOCK_KEY = f"{settings.R2_INDEXES_PREFIX}lock.json"
LOCK_TTL_SECONDS = 300  # 5 minutes lock expiry

def acquire_lock(owner_id: str, timeout_seconds: int = 30) -> bool:
    """
    Acquire a distributed lock on Cloudflare R2 (optimistic concurrency control).
    Also utilizes local threading lock for local safety.
    """
    # 1. Acquire local lock first
    local_acquired = _local_lock.acquire(timeout=timeout_seconds)
    if not local_acquired:
        logger.warning(f"Local thread lock acquisition timed out for owner: {owner_id}")
        return False
        
    client = get_r2_client()
    bucket = settings.R2_BUCKET_NAME
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        try:
            # Check if lock file exists
            lock_exists = False
            existing_lock = None
            try:
                response = client.get_object(Bucket=bucket, Key=LOCK_KEY)
                existing_lock = json.loads(response['Body'].read().decode('utf-8'))
                lock_exists = True
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') not in ['404', 'NoSuchKey']:
                    raise e
                    
            if lock_exists and existing_lock:
                lock_time_str = existing_lock.get("timestamp", "")
                try:
                    lock_time = datetime.fromisoformat(lock_time_str)
                    age = (datetime.now() - lock_time).total_seconds()
                except Exception:
                    age = LOCK_TTL_SECONDS + 1 # Force invalid/corrupted locks to expire
                    
                if age < LOCK_TTL_SECONDS:
                    # Lock is active, wait and poll
                    logger.info(f"R2 index lock is currently held by {existing_lock.get('owner')}. Waiting...")
                    time.sleep(2)
                    continue
                else:
                    logger.warning(f"Lock held by {existing_lock.get('owner')} is expired ({age:.1f}s old). Overwriting.")
            
            # Write our lock
            lock_data = {
                "owner": owner_id,
                "timestamp": datetime.now().isoformat()
            }
            client.put_object(
                Bucket=bucket,
                Key=LOCK_KEY,
                Body=json.dumps(lock_data).encode('utf-8'),
                ContentType='application/json'
            )
            
            # Read back to verify (optimistic check)
            verify_response = client.get_object(Bucket=bucket, Key=LOCK_KEY)
            verified_lock = json.loads(verify_response['Body'].read().decode('utf-8'))
            if verified_lock.get("owner") == owner_id:
                logger.info(f"Successfully acquired R2 index lock for owner: {owner_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error during R2 lock acquisition: {e}")
            time.sleep(2)
            
    # Clean up local lock if we failed to acquire R2 lock
    _local_lock.release()
    logger.error(f"Failed to acquire distributed R2 lock within {timeout_seconds}s for owner: {owner_id}")
    return False

def release_lock(owner_id: str) -> bool:
    """
    Release the distributed lock on R2 and local threading lock.
    """
    released_r2 = False
    try:
        client = get_r2_client()
        bucket = settings.R2_BUCKET_NAME
        
        # Verify ownership before releasing
        try:
            response = client.get_object(Bucket=bucket, Key=LOCK_KEY)
            current_lock = json.loads(response['Body'].read().decode('utf-8'))
            if current_lock.get("owner") == owner_id:
                client.delete_object(Bucket=bucket, Key=LOCK_KEY)
                logger.info(f"Released R2 index lock for owner: {owner_id}")
                released_r2 = True
            else:
                logger.warning(f"Cannot release lock: owner is {current_lock.get('owner')}, but requested by {owner_id}")
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') in ['404', 'NoSuchKey']:
                logger.info("Lock was already deleted or not present on R2.")
                released_r2 = True
            else:
                raise e
    except Exception as e:
        logger.error(f"Error releasing R2 lock: {e}")
    finally:
        # Always release local thread lock
        try:
            _local_lock.release()
        except RuntimeError:
            # In case the lock was not acquired by the calling thread
            pass
            
    return released_r2

def delete_file(r2_key: str) -> bool:
    """
    Deletes a file key from the Cloudflare R2 bucket.
    """
    logger.info(f"Deleting R2 key: {r2_key}...")
    try:
        client = get_r2_client()
        client.delete_object(Bucket=settings.R2_BUCKET_NAME, Key=r2_key)
        logger.info(f"Successfully deleted {r2_key} from R2.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {r2_key} from R2: {e}")
        return False

def list_files(prefix: str) -> list:
    """
    List all file keys in the Cloudflare R2 bucket under a specific prefix.
    """
    try:
        client = get_r2_client()
        response = client.list_objects_v2(Bucket=settings.R2_BUCKET_NAME, Prefix=prefix)
        keys = []
        if 'Contents' in response:
            for obj in response['Contents']:
                keys.append(obj['Key'])
        return keys
    except Exception as e:
        logger.error(f"Failed to list files in R2 under prefix {prefix}: {e}")
        return []

