# VoxCeleb2 AWS S3 API

This project provides a simple Flask API to download subsets of the VoxCeleb2 dataset, which is assumed to be hosted in an AWS S3 bucket. It allows users to retrieve a specified number of random audio samples along with their corresponding metadata in a single ZIP file.

The project also includes a utility script (`upload.py`) for uploading the dataset (audio files and metadata) to the S3 bucket.

## Features

*   **Download Subset:** Get a ZIP file containing a random subset of audio files and a filtered metadata CSV.
*   **S3 Integration:** Reads audio files and metadata directly from an AWS S3 bucket.
*   **Easy Setup:** Uses Poetry for dependency management and environment setup.
*   **Configurable:** Uses environment variables for AWS credentials and S3 bucket configuration.

## Requirements

*   Python 3.9+
*   [Poetry](https://python-poetry.org/) for dependency management.
*   The **exact name** of the publicly accessible AWS S3 bucket containing the dataset: `voxceleb2-dataset-bts-final-project`.
*   Your **own** valid AWS Credentials (Access Key ID, Secret Access Key). While the bucket allows public read access, the API client still requires valid credentials for initialization.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # If you haven't cloned the main project yet
    git clone <repository_url>
    cd <repository_directory>/voxceleb-aws-api
    ```

2.  **Install Dependencies using Poetry:**
    This command will create a virtual environment (if one doesn't exist) and install the dependencies listed in `pyproject.toml`.
    ```bash
    poetry install
    ```

3.  **Configure Environment Variables:**
    Each user should create their own `.env` file in the `voxceleb-aws-api` directory. Populate it with your **personal**, valid AWS credentials and the **exact** team bucket name:
    ```dotenv
    # Your personal AWS Credentials (Needed for API client initialization)
    AWS_ACCESS_KEY_ID="YOUR_PERSONAL_AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY="YOUR_PERSONAL_AWS_SECRET_ACCESS_KEY"
    # AWS_SESSION_TOKEN="YOUR_AWS_SESSION_TOKEN" # Optional: Include if using temporary credentials

    # AWS Region 
    AWS_REGION="us-east-1"

    # S3 Bucket Name - MUST match the team's shared, public bucket
    S3_BUCKET_NAME="voxceleb2-dataset-bts-final-project"
    ```
    **Important:**
    *   The `S3_BUCKET_NAME` **must** be exactly `voxceleb2-dataset-bts-final-project`.
    *   You need to provide **valid** AWS credentials, even though the bucket is public, as the underlying AWS SDK client used by the API requires them for initialization.
    *   The bucket `voxceleb2-dataset-bts-final-project` has been configured for public read access via its Bucket Policy, simplifying permissions.

## Running the API Server

Once the setup is complete, you can run the Flask API server using Poetry:

```bash
poetry run flask run --host=0.0.0.0 --port=5000
```

*   `--host=0.0.0.0`: Makes the server accessible from other machines on your network (use `127.0.0.1` or omit for local access only).
*   `--port=5000`: Specifies the port the server will listen on.

The server will start, and you should see output indicating it's running, similar to:
`* Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)`

## API Endpoints

### 1. Download Dataset Subset

*   **Endpoint:** `/download`
*   **Method:** `GET`
*   **Description:** Downloads a ZIP archive containing a randomly selected subset of audio files and a corresponding filtered metadata CSV file.
*   **Query Parameters:**
    *   `count` (integer, optional, default: 10): The number of audio files to include in the subset. If the requested count exceeds the total number of available files, all available files will be returned.
*   **Success Response:**
    *   **Code:** 200 OK
    *   **Content-Type:** `application/zip`
    *   **Body:** A ZIP file named `voxceleb<N>.zip` (where `<N>` is the number of audio files actually included) containing:
        *   `filtered_metadata.csv`: A CSV file with metadata rows only for the included audio files.
        *   `audio/idXXXXX/vidYYYYY/ZZZZZ.wav`: Audio files preserving their original path structure relative to the `audio/` prefix in S3.
*   **Error Responses:**
    *   **Code:** 400 Bad Request (e.g., if `count` is not a positive integer).
    *   **Code:** 404 Not Found (e.g., if no audio files are found in the bucket).
    *   **Code:** 500 Internal Server Error (e.g., issues connecting to S3, creating the ZIP file).

*   **Example Usage (using `curl`):**
    *   Download a subset of 5 audio files:
        ```bash
        curl -o voxceleb5.zip "http://localhost:5000/download?count=5"
        ```
    *   Download a subset using the default count (10 files):
        ```bash
        curl -o voxceleb10.zip "http://localhost:5000/download"
        ```

### 2. Health Check

*   **Endpoint:** `/health`
*   **Method:** `GET`
*   **Description:** A simple endpoint to check if the API server is running and responding.
*   **Success Response:**
    *   **Code:** 200 OK
    *   **Content-Type:** `application/json`
    *   **Body:** `{"status": "ok"}`

*   **Example Usage (using `curl`):**
    ```bash
    curl "http://localhost:5000/health"
    ```

## Upload Script (`upload.py`)

This project also includes `upload.py`, a script designed to upload the VoxCeleb2 audio files and metadata CSV to the specified S3 bucket.

*   **Purpose:** Populates the S3 bucket with the dataset files in the structure expected by the API (`audio/...` and `metadata/...`).
*   **Features:** Parallel uploads, checks for existing files to avoid re-uploading, transforms metadata paths to the required format.
*   **Usage:**
    ```bash
    # Ensure you are in the voxceleb-aws-api directory
    poetry run python upload.py --audio_dir /path/to/local/voxceleb_audio --metadata_file /path/to/local/metadata.csv
    ```
    *   `--audio_dir`: Path to the local directory containing the raw VoxCeleb audio files (e.g., `.../voxceleb_audio/idXXXXX/vidYYYYY/ZZZZZ.wav`).
    *   `--metadata_file`: Path to the original metadata CSV file.
    *   See `poetry run python upload.py --help` for more options like `--max_workers` and `--max_files`.

    **Note:** This script requires write permissions (`s3:PutObject`, `s3:HeadObject`) for the target S3 bucket, in addition to the read permissions needed by the API. Configure AWS credentials accordingly (e.g., via the `.env` file). 