#!/usr/bin/env python3
"""
Flask API for VoxCeleb2 Dataset.

This API allows users to download a subset of the VoxCeleb2 dataset that is stored in AWS S3.
The API has a single endpoint that returns a ZIP file containing:
1. Audio files randomly selected from the dataset
2. A CSV file with metadata for only the selected audio files

Usage:
    GET /download?count=50

Requirements:
    - flask
    - boto3
    - pandas
    - python-dotenv
"""

import os
import io
import random
import zipfile
import tempfile
import logging
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import boto3
import pandas as pd
from flask import Flask, request, send_file, jsonify, g, after_this_request
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('voxceleb-api')

# Load environment variables
load_dotenv()

# Get AWS credentials from environment
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'voxceleb2-dataset')

# Create Flask app
app = Flask(__name__)


def get_s3_client():
    """
    Create and return a boto3 S3 client using credentials from environment.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
            region_name=AWS_REGION
        )
        return s3_client
    except Exception as e:
        logger.error(f"Error creating S3 client: {e}")
        raise


def list_audio_files(s3_client, prefix="audio/") -> List[str]:
    """
    List all audio files in the S3 bucket.

    Args:
        s3_client: Boto3 S3 client
        prefix: S3 key prefix to filter by

    Returns:
        List of S3 keys for audio files
    """
    try:
        # Use paginator for large datasets
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=prefix)

        audio_files = []
        for page in pages:
            if 'Contents' in page:
                for item in page['Contents']:
                    # Ensure the key is not a pseudo-directory and ends with .wav
                    if not item['Key'].endswith('/') and item['Key'].endswith('.wav'):
                        audio_files.append(item['Key'])

        logger.info(f"Found {len(audio_files)} audio files in S3 bucket")
        return audio_files
    except Exception as e:
        logger.error(f"Error listing audio files: {e}")
        return []


def get_metadata_file(s3_client) -> str:
    """
    Download the metadata CSV file from S3 bucket to a temporary file.

    Args:
        s3_client: Boto3 S3 client

    Returns:
        Path to downloaded temporary metadata file, or None on error.
    """
    temp_file_path = None
    try:
        # List metadata directory to find the CSV file
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix="metadata/"
        )

        metadata_key = None
        if 'Contents' in response:
            for item in response['Contents']:
                if item['Key'].endswith('.csv'):
                    metadata_key = item['Key']
                    break

        if not metadata_key:
            logger.error("No CSV file found in metadata directory")
            return None

        logger.info(f"Found metadata file: {metadata_key}")

        # Download metadata file to temp file
        # Use delete=False so the file persists until we explicitly delete it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as temp_file:
            s3_client.download_fileobj(S3_BUCKET_NAME, metadata_key, temp_file)
            temp_file_path = temp_file.name
            logger.info(
                f"Downloaded metadata to temporary file: {temp_file_path}")

        return temp_file_path

    except Exception as e:
        logger.error(f"Error getting metadata file: {e}")
        # Clean up temp file if created but download failed
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return None


def filter_metadata_for_files(metadata_file_path: str, selected_s3_keys: List[str]) -> str:
    """
    Filter metadata CSV to only include rows for selected files.
    Assumes the CSV 'file_path' column already matches the S3 key format after 'audio/'.

    Args:
        metadata_file_path: Path to the downloaded metadata CSV file.
        selected_s3_keys: List of selected S3 keys (e.g., "audio/id/vid/seg.wav").

    Returns:
        Path to filtered temporary metadata CSV file, or None on error.
    """
    filtered_file_path = None
    try:
        # Extract relative file paths expected in the CSV from S3 keys
        # Assumes CSV file_path format is: id/vid/seg.wav
        expected_file_paths = set()
        for s3_key in selected_s3_keys:
            if s3_key.startswith("audio/"):
                expected_file_paths.add(s3_key.split('audio/', 1)[1])
            else:
                logger.warning(
                    f"Skipping S3 key without expected 'audio/' prefix: {s3_key}")

        if not expected_file_paths:
            logger.error(
                "Could not extract any valid file paths from selected S3 keys.")
            return None

        logger.debug(
            f"Filtering metadata for paths like: {list(expected_file_paths)[:5]}")

        # Read metadata CSV
        df = pd.read_csv(metadata_file_path)
        logger.debug(f"Metadata columns: {df.columns.tolist()}")
        if 'file_path' not in df.columns:
            logger.error("'file_path' column not found in metadata CSV.")
            return None

        # Filter to only include rows matching the expected file paths
        df['file_path'] = df['file_path'].astype(str).str.strip()
        filtered_df = df[df['file_path'].isin(expected_file_paths)]

        if filtered_df.empty:
            logger.warning(
                f"No matching metadata entries found for the {len(expected_file_paths)} selected files.")
            # Create an empty filtered file

        # Create new temp file for filtered metadata
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='') as filtered_file:
            filtered_df.to_csv(filtered_file, index=False)
            filtered_file_path = filtered_file.name

        logger.info(
            f"Created filtered metadata CSV ({filtered_file_path}) with {len(filtered_df)} entries")
        return filtered_file_path

    except Exception as e:
        # Log full traceback
        logger.exception(f"Error filtering metadata: {e}")
        if filtered_file_path and os.path.exists(filtered_file_path):
            os.unlink(filtered_file_path)
        return None


def create_zip_file(s3_client, selected_s3_keys: List[str], filtered_metadata_path: str) -> str:
    """
    Create a ZIP file containing selected audio files and filtered metadata.

    Args:
        s3_client: Boto3 S3 client
        selected_s3_keys: List of selected S3 keys for audio files.
        filtered_metadata_path: Path to filtered temporary metadata CSV.

    Returns:
        Path to created temporary ZIP file, or None on error.
    """
    zip_file_path = None
    try:
        # Create temp file for ZIP
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as zip_file:
            zip_file_path = zip_file.name

        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add filtered metadata CSV
            if filtered_metadata_path and os.path.exists(filtered_metadata_path):
                # Use a generic name inside the zip for consistency
                zipf.write(filtered_metadata_path, "filtered_metadata.csv")
                logger.info(f"Added filtered_metadata.csv to ZIP.")
            else:
                logger.warning(
                    "Filtered metadata file path is invalid or missing. Not added to ZIP.")

            # Download and add audio files
            files_added_count = 0
            for s3_key in selected_s3_keys:
                try:
                    # Use BytesIO to avoid writing audio to another temp file
                    audio_fileobj = io.BytesIO()
                    s3_client.download_fileobj(
                        S3_BUCKET_NAME, s3_key, audio_fileobj)
                    audio_fileobj.seek(0)  # Reset stream position
                    # Add to ZIP, preserving the S3 key structure
                    zipf.writestr(s3_key, audio_fileobj.read())
                    files_added_count += 1
                except Exception as e:
                    logger.error(f"Error adding file {s3_key} to ZIP: {e}")

        logger.info(
            f"Created ZIP file ({zip_file_path}) with {files_added_count} audio files and metadata")
        return zip_file_path

    except Exception as e:
        logger.error(f"Error creating ZIP file: {e}")
        if zip_file_path and os.path.exists(zip_file_path):
            os.unlink(zip_file_path)
        return None


def cleanup_temp_files(*file_paths):
    """Utility function to remove temporary files."""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing temp file {file_path}: {e}")


@app.route('/download', methods=['GET'])
def download_dataset():
    """
    API endpoint to download a subset of the VoxCeleb2 dataset.

    Query Parameters:
        count (int): Number of audio files to download (default: 10)

    Returns:
        ZIP file containing audio files and metadata, or JSON error.
    """
    metadata_file = None
    filtered_metadata = None
    zip_file = None

    try:
        count = request.args.get('count', default=10, type=int)
        if count <= 0:
            return jsonify({"error": "Count must be greater than 0"}), 400

        s3_client = get_s3_client()
        audio_files = list_audio_files(s3_client)
        if not audio_files:
            return jsonify({"error": "No audio files found in S3 bucket"}), 404

        selected_count = min(count, len(audio_files))
        if selected_count == 0:
            return jsonify({"error": "Cannot select 0 files."}), 400
        selected_files = random.sample(audio_files, selected_count)
        logger.info(f"Selected {selected_count} random audio files.")

        metadata_file = get_metadata_file(s3_client)
        if not metadata_file:
            # Decide if you want to proceed without metadata
            logger.warning(
                "Metadata file not found in S3 bucket. Proceeding without metadata.")
            # return jsonify({"error": "Metadata file not found in S3 bucket"}), 404

        # Only filter if metadata was successfully downloaded
        if metadata_file:
            filtered_metadata = filter_metadata_for_files(
                metadata_file, selected_files)
            if not filtered_metadata:
                # Decide if you want to proceed without metadata
                logger.warning(
                    "Error creating filtered metadata. Proceeding without metadata.")
                # return jsonify({"error": "Error creating filtered metadata"}), 500

        # Create ZIP file (pass None if filtered_metadata failed)
        zip_file = create_zip_file(
            s3_client, selected_files, filtered_metadata)
        if not zip_file:
            # Ensure partial temp files are cleaned up even if zip creation fails
            cleanup_temp_files(metadata_file, filtered_metadata)
            return jsonify({"error": "Error creating ZIP file"}), 500

        # Use Flask's after_this_request to schedule cleanup
        @after_this_request
        def cleanup(response):
            cleanup_temp_files(metadata_file, filtered_metadata, zip_file)
            logger.debug("Scheduled cleanup for temporary files.")
            return response

        # Send ZIP file
        return send_file(
            zip_file,
            mimetype='application/zip',
            as_attachment=True,
            # Use the requested format: voxceleb<N>.zip
            download_name=f'voxceleb{selected_count}.zip'
        )

    except Exception as e:
        # Log full traceback
        logger.exception(f"Error processing download request: {e}")
        # Ensure cleanup happens even on unexpected errors before returning
        cleanup_temp_files(metadata_file, filtered_metadata, zip_file)
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    # Use waitress or gunicorn in production instead of Flask dev server
    # Disable debug mode for safety
    app.run(host='0.0.0.0', port=5000, debug=False)
