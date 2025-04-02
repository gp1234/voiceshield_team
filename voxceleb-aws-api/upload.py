#!/usr/bin/env python3
"""
Upload script for VoxCeleb2 dataset to AWS S3.

This script:
1. Uploads audio files from the VoxCeleb2 dataset to an existing S3 bucket in parallel.
2. Checks if a file already exists in S3 before uploading and skips if it does.
3. Uploads metadata CSV file to the same S3 bucket.

Assumes the S3 bucket already exists and the AWS credentials have necessary permissions (ListBucket, PutObject, HeadObject).

Usage:
    python upload.py --audio_dir /path/to/voxceleb_audio --metadata_file /path/to/metadata.csv [--max_workers 10] [--max_files 1000]

Requirements:
    - boto3
    - python-dotenv
    - pandas
    - tqdm (for progress bars)
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import boto3
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upload.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('upload')

# Load environment variables
load_dotenv()

# Get AWS credentials from environment
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')  # Required now

if not S3_BUCKET_NAME:
    logger.error(
        "S3_BUCKET_NAME not found in .env file. Please specify the target bucket name.")
    sys.exit(1)

# Thread-local storage for S3 client to avoid potential issues with sharing clients across threads
thread_local = threading.local()


def get_s3_client():
    """
    Create and return a boto3 S3 client for the current thread.
    Caches the client in thread-local storage.
    """
    if not hasattr(thread_local, 's3_client'):
        logger.debug(
            f"Creating S3 client for thread {threading.current_thread().name}")
        try:
            thread_local.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                aws_session_token=AWS_SESSION_TOKEN,
                region_name=AWS_REGION
            )
            # Basic check during initial creation (optional, adds latency)
            # thread_local.s3_client.list_buckets()
        except Exception as e:
            logger.error(
                f"Error creating S3 client in thread {threading.current_thread().name}: {e}")
            raise
    return thread_local.s3_client


def transform_csv_path(csv_path):
    """
    Transforms a single path string from the CSV format 'real/idXXXXX_vidYYYYY_ZZZZZ.wav'
    (where vidYYYYY might contain underscores) to the desired S3-like format 
    'idXXXXX/vidYYYYY/ZZZZZ.wav'.
    Returns the original path if transformation fails.
    """
    if pd.isna(csv_path):
        return csv_path

    # Regex:
    # ^real/     - Matches 'real/' at the beginning
    # (id\d{5})  - Captures speaker ID (group 1)
    # _          - Matches the first underscore
    # (.+)       - Captures video ID (group 2) - everything up to the last underscore
    # _          - Matches the last underscore
    # (\d{5})    - Captures segment ID (group 3)
    # \.wav$     - Matches .wav at the end
    match = re.match(r"^real/(id\d{5})_(.+)_(\d{5})\.wav$", str(csv_path))

    if match:
        speaker_id, video_id, segment_id = match.groups()
        # Reconstruct using forward slashes and add .wav back to segment
        transformed_path = f"{speaker_id}/{video_id}/{segment_id}.wav"
        # logger.debug(f"Transformed path: {csv_path} -> {transformed_path}")
        return transformed_path
    else:
        # logger.warning(f"Path '{csv_path}' did not match expected format for transformation.")
        return csv_path  # Return original if no match


def upload_metadata(metadata_file):
    """
    Reads the metadata CSV, transforms the 'file_path' column, 
    and uploads the modified version to S3 bucket.

    Args:
        metadata_file (str): Path to the original metadata CSV file
    """
    temp_modified_csv = None
    try:
        s3_client = get_s3_client()  # Get client for this thread/context
        original_path = Path(metadata_file)
        if not original_path.exists():
            logger.error(f"Original metadata file not found: {metadata_file}")
            return False

        # Read the original CSV
        logger.info(f"Reading original metadata file: {metadata_file}")
        df = pd.read_csv(original_path)

        if 'file_path' not in df.columns:
            logger.error(
                "'file_path' column not found in metadata CSV. Cannot transform.")
            return False

        # Apply the transformation
        logger.info("Transforming 'file_path' column...")
        original_paths_sample = df['file_path'].head().tolist()
        df['file_path'] = df['file_path'].apply(transform_csv_path)
        transformed_paths_sample = df['file_path'].head().tolist()
        logger.info(
            f"Transformation example: {original_paths_sample} -> {transformed_paths_sample}")

        # Save modified DataFrame to a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_modified.csv', newline='') as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_modified_csv = temp_file.name
            logger.info(
                f"Saved transformed metadata to temporary file: {temp_modified_csv}")

        # Upload the modified temporary file to S3
        # Use original base name for S3 key
        s3_key = f"metadata/{original_path.name}"
        logger.info(
            f"Uploading transformed metadata from {temp_modified_csv} to S3 bucket {S3_BUCKET_NAME} as {s3_key}")

        # Check if metadata file already exists (optional, can overwrite)
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            logger.warning(
                f"Metadata file {s3_key} already exists in bucket. Overwriting with transformed data.")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(
                    f"Metadata file {s3_key} not found. Proceeding with upload of transformed data.")
            else:
                # Log other errors but proceed with upload attempt
                logger.error(
                    f"Error checking metadata existence, attempting upload anyway: {e}")

        s3_client.upload_file(
            temp_modified_csv,  # Upload the modified file
            S3_BUCKET_NAME,
            s3_key
        )

        logger.info(
            f"Transformed metadata file uploaded successfully: s3://{S3_BUCKET_NAME}/{s3_key}")
        return True

    except Exception as e:
        # Log full traceback
        logger.exception(f"Error processing and uploading metadata: {e}")
        return False
    finally:
        # Clean up the temporary file
        if temp_modified_csv and os.path.exists(temp_modified_csv):
            try:
                os.unlink(temp_modified_csv)
                logger.info(
                    f"Removed temporary metadata file: {temp_modified_csv}")
            except Exception as e_unlink:
                logger.error(
                    f"Error removing temporary metadata file {temp_modified_csv}: {e_unlink}")


def upload_single_audio_file(args):
    """
    Worker function to upload a single audio file. Checks existence first.
    Takes a tuple (wav_file_path, audio_dir_path) as argument.
    Returns the s3_key if successful or skipped, None otherwise.
    """
    wav_file, audio_dir_str = args
    audio_path = Path(audio_dir_str)
    s3_client = get_s3_client()  # Get thread-local client

    try:
        # Create S3 key preserving relative path structure
        rel_path = os.path.relpath(wav_file, audio_path)
        s3_key = f"audio/{rel_path}"  # e.g., audio/id00001/video_id/00001.wav

        # 1. Check if file already exists in S3
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            # logger.debug(f"Skipping existing file: {s3_key}")
            return s3_key  # Return key even if skipped for progress tracking
        except s3_client.exceptions.ClientError as e:
            # If a 404 error, the object does not exist, proceed to upload
            if e.response['Error']['Code'] != '404':
                logger.error(
                    f"Error checking {s3_key} existence (Code: {e.response['Error']['Code']}): {e}")
                return None  # Indicate failure

        # 2. Upload if it doesn't exist
        # logger.debug(f"Uploading {wav_file} to {s3_key}")
        s3_client.upload_file(
            wav_file,
            S3_BUCKET_NAME,
            s3_key
        )
        return s3_key  # Indicate success

    except Exception as e:
        logger.error(f"Error uploading file {wav_file} to {s3_key}: {e}")
        return None  # Indicate failure


def upload_audio_files(audio_dir, max_workers=10, max_files=None):
    """
    Upload audio files to S3 bucket in parallel, searching recursively and skipping existing files.

    Args:
        audio_dir (str): Directory containing audio files (expects subdirs like idXXXXX/video_id/segment.wav)
        max_workers (int): Number of parallel upload threads.
        max_files (int, optional): Maximum number of files to attempt to upload.
    """
    start_time_total = time.time()
    try:
        audio_path = Path(audio_dir)
        if not audio_path.exists() or not audio_path.is_dir():
            logger.error(f"Audio directory not found: {audio_dir}")
            return False

        # Find all WAV files recursively
        wav_files_to_process = []
        logger.info(f"Searching recursively for .wav files in {audio_path}...")
        # Limit search if max_files is set early
        file_iterator = audio_path.rglob('*.wav')

        count = 0
        for file in file_iterator:
            if file.is_file():
                wav_files_to_process.append(str(file))
                count += 1
                if max_files and count >= max_files:
                    logger.info(
                        f"Found {max_files} files, stopping search due to --max_files limit.")
                    break

        total_files_found = len(wav_files_to_process)
        if total_files_found == 0:
            logger.warning(
                f"No WAV files found recursively in directory: {audio_dir}")
            return False

        logger.info(
            f"Found {total_files_found} WAV files. Will attempt to upload to bucket {S3_BUCKET_NAME} using {max_workers} workers.")

        # Prepare arguments for worker function
        tasks = [(wav_file, str(audio_path))
                 for wav_file in wav_files_to_process]

        uploaded_count = 0
        skipped_count = 0
        failed_count = 0

        # Use ThreadPoolExecutor for parallel uploads
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='S3UploadWorker') as executor:
            # Wrap executor.map or as_completed with tqdm for progress bar
            futures = [executor.submit(
                upload_single_audio_file, task) for task in tasks]

            for future in tqdm(as_completed(futures), total=total_files_found, desc="Uploading audio files"):
                result = future.result()
                if result is not None:
                    # Simple check: Did the worker return the key? If yes, it either uploaded or skipped successfully.
                    # A more robust check might involve head_object again, but adds latency.
                    # For now, we assume returning the key means success/skipped.
                    uploaded_count += 1  # Count includes skipped files for progress
                else:
                    failed_count += 1

        # Note: A more precise 'skipped' count would require returning a status from upload_single_audio_file
        # This implementation counts skipped as part of the 'successful' progress.

        end_time_total = time.time()
        elapsed_total = end_time_total - start_time_total
        rate = total_files_found / elapsed_total if elapsed_total > 0 else 0

        logger.info(
            f"Finished processing {total_files_found} files in {elapsed_total:.2f} seconds. Average rate: ({rate:.2f} files/sec)")
        logger.info(
            f"Successfully processed (uploaded or skipped): {uploaded_count}, Failed: {failed_count}")

        if uploaded_count > 0:  # Check if at least some processing happened
            logger.info(
                f"Audio files upload process completed for s3://{S3_BUCKET_NAME}/audio/")
            return True
        elif failed_count > 0:
            logger.error("Upload process completed but with failures.")
            return False
        else:  # No files found or processed
            logger.warning("No new files were uploaded or processed.")
            return True  # Consider this success if no files needed uploading

    except Exception as e:
        logger.error(f"Error during parallel audio file upload process: {e}")
        return False


def main():
    """Main function to handle the upload process."""
    parser = argparse.ArgumentParser(
        description='Upload VoxCeleb2 dataset to an existing AWS S3 bucket in parallel')
    parser.add_argument('--audio_dir', required=True,
                        help='Directory containing audio WAV files')
    parser.add_argument('--metadata_file', required=True,
                        help='Path to metadata CSV file')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of audio files to process (for testing)')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Number of parallel upload threads')

    args = parser.parse_args()

    try:
        # Upload metadata file (sequentially first)
        metadata_success = upload_metadata(args.metadata_file)

        if not metadata_success:
            logger.warning(
                "Metadata upload failed. Continuing with audio upload...")
            # Decide if you want to stop here if metadata fails

        # Upload audio files in parallel
        audio_success = upload_audio_files(
            args.audio_dir, args.max_workers, args.max_files)

        # Audio process completed (might include skips/failures reported within)
        if audio_success:
            logger.info("Upload process completed.")
            # Check metadata_success again if it's critical
            return 0 if metadata_success else 1
        else:
            logger.error("Audio upload process failed.")
            return 1

    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
