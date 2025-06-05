import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import joblib
import numpy as np
import soundfile as sf
import openl3
import librosa
from pydantic import BaseModel
from pydub import AudioSegment
import io
import time  # For basic timing and more informative logs
import traceback

# --- Configuration ---
# Assuming main.py is in the 'app/' directory relative to the project root
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for deployed models within the 'app' directory
# User indicates model is directly in 'app/saved_models/'
SAVED_MODEL_DIR = os.path.join(APP_DIR, "saved_models")
# The actual filename of the bundle
MODEL_BUNDLE_FILENAME = "model_and_scaler.joblib"
MODEL_PATH = os.path.join(SAVED_MODEL_DIR, MODEL_BUNDLE_FILENAME)

# For logging purposes, indicate what type of model this bundle is expected to be
CHOSEN_MODEL_TYPE_FOR_LOGGING = "svm"

MIN_AUDIO_DURATION_SEC = 3.0
TARGET_SR = 16000
# Assuming static_frontend is also in app/
STATIC_DIR = os.path.join(APP_DIR, "static_frontend")

# --- Model Loading ---
model = None
scaler = None


async def load_model_and_scaler():
    global model, scaler

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Server startup: Attempting to load model and scaler.")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Expected model path: {MODEL_PATH}")

    if not os.path.exists(SAVED_MODEL_DIR):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Saved models directory not found: {SAVED_MODEL_DIR}. Attempting to create it.")
        try:
            os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        except Exception as e:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Could not create saved models directory {SAVED_MODEL_DIR}: {e}")
            return

    if not os.path.exists(MODEL_PATH):
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Model file not found at {MODEL_PATH}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Please ensure your trained model bundle ('{MODEL_BUNDLE_FILENAME}')")
        print(f"      is placed in the directory: '{SAVED_MODEL_DIR}'")
        print(
            f"      This structure should be part of your Docker image if deploying with Docker (e.g., app/saved_models/{MODEL_BUNDLE_FILENAME}).")
        return
    try:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Loading joblib bundle from {MODEL_PATH}...")
        model_bundle = joblib.load(MODEL_PATH)
        model = model_bundle.get("model")
        scaler = model_bundle.get("scaler")
        if model is None or scaler is None:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: 'model' or 'scaler' key not found in the loaded joblib bundle from {MODEL_PATH}.")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model (expected type: '{CHOSEN_MODEL_TYPE_FOR_LOGGING}') and scaler loaded successfully from {MODEL_PATH}.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Exception during model loading from {MODEL_PATH}: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model_and_scaler()
    yield
    # Shutdown (if needed)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Server shutdown.")

# --- Application Setup ---
app = FastAPI(title="VoiceShield - AI Voice Analyzer", lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    filename: str

# --- Helper Functions ---


def pad_if_short(audio_data: np.ndarray, sample_rate: int, min_duration: float, filename_for_log: str) -> np.ndarray:
    target_len = int(sample_rate * min_duration)
    current_duration = len(audio_data) / sample_rate
    if len(audio_data) < target_len:
        pad_width = target_len - len(audio_data)
        print(f"[{filename_for_log}] INFO: Padding audio from {current_duration:.2f}s to {min_duration:.2f}s (target samples: {target_len}, current: {len(audio_data)}).")
        return np.pad(audio_data, (0, pad_width), mode='constant')
    print(f"[{filename_for_log}] INFO: Audio duration {current_duration:.2f}s is >= min_duration {min_duration:.2f}s. No padding needed.")
    return audio_data


def extract_embedding(audio_data: np.ndarray, sample_rate: int, filename_for_log: str) -> np.ndarray | None:
    print(f"[{filename_for_log}] INFO: Starting embedding extraction. Initial audio shape: {audio_data.shape}, sample rate: {sample_rate}Hz.")
    try:
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            print(f"[{filename_for_log}] INFO: Audio is stereo, converting to mono.")
            audio_data = np.mean(audio_data, axis=1)
            print(
                f"[{filename_for_log}] INFO: Mono audio shape: {audio_data.shape}.")

        if sample_rate != TARGET_SR:
            print(
                f"[{filename_for_log}] INFO: Resampling from {sample_rate}Hz to {TARGET_SR}Hz.")
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR
            print(
                f"[{filename_for_log}] INFO: Resampled audio shape: {audio_data.shape}.")

        audio_data = pad_if_short(
            audio_data, sample_rate, MIN_AUDIO_DURATION_SEC, filename_for_log)

        print(f"[{filename_for_log}] INFO: Calling OpenL3 get_audio_embedding (content_type='music'). Final audio length for OpenL3: {len(audio_data)/sample_rate:.2f}s.")
        emb, ts = openl3.get_audio_embedding(
            audio_data, sample_rate, content_type="music", embedding_size=512, hop_size=0.5)

        if emb.shape[0] == 0:
            print(f"[{filename_for_log}] WARNING: OpenL3 returned empty embedding. This can happen if audio is too short or silent after processing.")
            return None

        mean_embedding = np.mean(emb, axis=0)
        print(
            f"[{filename_for_log}] INFO: Embedding extracted successfully. Shape: {mean_embedding.shape}.")
        return mean_embedding
    except Exception as e:
        print(
            f"[filename_for_log] ERROR: Exception during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- API Endpoints ---

@app.post("/analyze_audio/", response_model=PredictionResponse)
async def analyze_audio(file: UploadFile = File(...)):
    request_received_time = time.time()
    log_prefix = f"[{file.filename} - {time.strftime('%Y-%m-%d %H:%M:%S')}]"
    print(f"\n{log_prefix} INFO: === New Request: Received file: {file.filename}, Content-Type: {file.content_type} ===")

    # This endpoint always uses the SVM model from saved_models directory
    if model is None or scaler is None:
        print(f"{log_prefix} ERROR: SVM model or scaler not loaded at request time.")
        raise HTTPException(
            status_code=503, detail="SVM model or scaler not loaded. Please check server startup logs.")

    print(f"{log_prefix} INFO: Reading file content...")
    file_content = await file.read()
    print(f"{log_prefix} INFO: File content read, length: {len(file_content)} bytes.")
    audio_buffer = io.BytesIO(file_content)

    temp_wav_path = f"temp_converted_{time.time_ns()}_{file.filename}.wav"

    try:
        print(f"{log_prefix} INFO: Attempting to load audio with pydub...")
        try:
            audio_segment = AudioSegment.from_file(audio_buffer)
            print(f"{log_prefix} INFO: Audio loaded with pydub. Duration: {len(audio_segment)/1000.0:.2f}s, Channels: {audio_segment.channels}, Frame Rate: {audio_segment.frame_rate}Hz.")
        except Exception as pydub_err:
            print(f"{log_prefix} ERROR: Pydub error loading audio: {pydub_err}")
            raise HTTPException(
                status_code=400, detail=f"Could not process audio format with pydub: {pydub_err}")

        print(
            f"{log_prefix} INFO: Exporting audio to temporary WAV file: {temp_wav_path}")
        audio_segment.export(temp_wav_path, format="wav")
        print(f"{log_prefix} INFO: Audio exported to WAV successfully.")

        print(f"{log_prefix} INFO: Reading standardized WAV file with soundfile...")
        audio_data, sample_rate = sf.read(temp_wav_path)
        print(f"{log_prefix} INFO: Audio read by soundfile. Sample rate: {sample_rate}, Shape: {audio_data.shape}, Duration: {len(audio_data)/sample_rate:.2f}s.")

        # Always use OpenL3 embeddings for SVM model from saved_models directory
        embedding = extract_embedding(
            audio_data, sample_rate, f"{file.filename} (API)")
        if embedding is None:
            print(
                f"{log_prefix} ERROR: Failed to extract features (embedding was None).")
            raise HTTPException(
                status_code=500, detail="Could not extract features from audio.")

        normalized_embedding = embedding
        print(
            f"{log_prefix} INFO: Embedding ready for scaling (no explicit L2 norm applied).")

        print(f"{log_prefix} INFO: Scaling features...")
        scaled_embedding = scaler.transform(
            normalized_embedding.reshape(1, -1))
        print(f"{log_prefix} INFO: Features scaled successfully.")

        print(
            f"{log_prefix} INFO: Making prediction with SVM model from saved_models...")
        prediction_value = model.predict(scaled_embedding)[0]
        print(f"{log_prefix} INFO: Raw prediction value from model: {prediction_value}")

        prediction_label = "REAL" if prediction_value == 1 else "FAKE"  # Real=1, Fake=0
        print(f"{log_prefix} INFO: Predicted label: {prediction_label}")

        print(f"{log_prefix} INFO: Getting prediction probabilities...")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_embedding)[0]
            print(
                f"{log_prefix} INFO: Probabilities: Class 0 (Fake)={probabilities[0]:.4f}, Class 1 (Real)={probabilities[1]:.4f}")

            if prediction_label == "REAL":
                confidence = probabilities[1] * 100
            else:
                confidence = probabilities[0] * 100
            print(
                f"{log_prefix} INFO: Calculated confidence for '{prediction_label}': {confidence:.2f}%")
        else:
            print(
                f"{log_prefix} WARNING: Model does not have predict_proba method. Setting confidence to -1 (unavailable).")
            confidence = -1.0

    except HTTPException:
        print(f"{log_prefix} ERROR: HTTPException occurred.")
        raise
    except Exception as e:
        print(f"{log_prefix} ERROR: UNEXPECTED ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing audio: {str(e)}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"{log_prefix} INFO: Cleaned up temporary file: {temp_wav_path}")
        await file.close()
        print(f"{log_prefix} INFO: File stream closed.")

    processing_end_time = time.time()
    processing_time = processing_end_time - request_received_time
    print(f"{log_prefix} INFO: Request processed successfully in {processing_time:.2f} seconds.")
    print(f"{log_prefix} INFO: === Finished request for file: {file.filename} ===")

    return PredictionResponse(
        prediction=prediction_label,
        confidence=round(confidence, 2),
        filename=file.filename
    )



@app.post("/wav_inference/", response_model=PredictionResponse)
async def wav_inference(file: UploadFile = File(...)):
    """Analyze WAV audio file for deepfake detection"""
    request_received_time = time.time()
    log_prefix = f"[WAV_INFERENCE - {file.filename} - {time.strftime('%Y-%m-%d %H:%M:%S')}]"
    print(f"\n{log_prefix} INFO: === New WAV Inference Request: Received file: {file.filename}, Content-Type: {file.content_type} ===")

    if model is None or scaler is None:
        print(f"{log_prefix} ERROR: Model or scaler not loaded at request time. This shouldn't happen if startup was successful.")
        raise HTTPException(
            status_code=503, detail="Model or scaler not loaded. Please check server startup logs.")

    print(f"{log_prefix} INFO: Reading file content...")
    file_content = await file.read()
    print(f"{log_prefix} INFO: File content read, length: {len(file_content)} bytes.")
    audio_buffer = io.BytesIO(file_content)

    temp_wav_path = f"temp_converted_{time.time_ns()}_{file.filename}.wav"

    try:
        print(f"{log_prefix} INFO: Attempting to load audio with pydub...")
        try:
            audio_segment = AudioSegment.from_file(audio_buffer)
            print(f"{log_prefix} INFO: Audio loaded with pydub. Duration: {len(audio_segment)/1000.0:.2f}s, Channels: {audio_segment.channels}, Frame Rate: {audio_segment.frame_rate}Hz.")
        except Exception as pydub_err:
            print(f"{log_prefix} ERROR: Pydub error loading audio: {pydub_err}")
            raise HTTPException(
                status_code=400, detail=f"Could not process audio format with pydub: {pydub_err}")

        print(f"{log_prefix} INFO: Exporting audio to temporary WAV file: {temp_wav_path}")
        audio_segment.export(temp_wav_path, format="wav")
        print(f"{log_prefix} INFO: Audio exported to WAV successfully.")

        print(f"{log_prefix} INFO: Reading standardized WAV file with soundfile...")
        audio_data, sample_rate = sf.read(temp_wav_path)
        print(f"{log_prefix} INFO: Audio read by soundfile. Sample rate: {sample_rate}, Shape: {audio_data.shape}, Duration: {len(audio_data)/sample_rate:.2f}s.")

        embedding = extract_embedding(
            audio_data, sample_rate, f"{file.filename} (WAV_INFERENCE)")
        if embedding is None:
            print(f"{log_prefix} ERROR: Failed to extract features (embedding was None).")
            raise HTTPException(
                status_code=500, detail="Could not extract features from audio.")

        normalized_embedding = embedding
        print(f"{log_prefix} INFO: Embedding ready for scaling (no explicit L2 norm applied).")

        print(f"{log_prefix} INFO: Scaling features...")
        scaled_embedding = scaler.transform(normalized_embedding.reshape(1, -1))
        print(f"{log_prefix} INFO: Features scaled successfully.")

        print(f"{log_prefix} INFO: Making prediction with '{CHOSEN_MODEL_TYPE_FOR_LOGGING}' model...")
        prediction_value = model.predict(scaled_embedding)[0]
        print(f"{log_prefix} INFO: Raw prediction value from model: {prediction_value}")

        prediction_label = "REAL" if prediction_value == 1 else "FAKE"  # Real=1, Fake=0
        print(f"{log_prefix} INFO: Predicted label: {prediction_label}")

        print(f"{log_prefix} INFO: Getting prediction probabilities...")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_embedding)[0]
            print(f"{log_prefix} INFO: Probabilities: Class 0 (Fake)={probabilities[0]:.4f}, Class 1 (Real)={probabilities[1]:.4f}")

            if prediction_label == "REAL":
                confidence = probabilities[1] * 100
            else:
                confidence = probabilities[0] * 100
            print(f"{log_prefix} INFO: Calculated confidence for '{prediction_label}': {confidence:.2f}%")
        else:
            print(f"{log_prefix} WARNING: Model does not have predict_proba method. Setting confidence to -1 (unavailable).")
            confidence = -1.0

    except HTTPException:
        print(f"{log_prefix} ERROR: HTTPException occurred.")
        raise
    except Exception as e:
        print(f"{log_prefix} ERROR: UNEXPECTED ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing audio: {str(e)}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
            print(f"{log_prefix} INFO: Cleaned up temporary file: {temp_wav_path}")
        await file.close()
        print(f"{log_prefix} INFO: File stream closed.")

    processing_end_time = time.time()
    processing_time = processing_end_time - request_received_time
    print(f"{log_prefix} INFO: WAV inference processed successfully in {processing_time:.2f} seconds.")
    print(f"{log_prefix} INFO: === Finished WAV inference request for file: {file.filename} ===")

    return PredictionResponse(
        prediction=prediction_label,
        confidence=round(confidence, 2),
        filename=file.filename
    )

# --- Route to serve HTML frontend ---


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_file_path = os.path.join(STATIC_DIR, "index.html")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Attempting to serve frontend from: {html_file_path}")
    if not os.path.exists(html_file_path):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Frontend HTML file not found at {html_file_path}")
        raise HTTPException(
            status_code=404, detail=f"Frontend HTML not found. Expected at {html_file_path}")
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# To run: uvicorn main:app --host 0.0.0.0 --port 8000
# Ensure your model bundle is correctly placed:
# saved_models/model_and_scaler.joblib
# And your frontend:
# static_frontend/index.html