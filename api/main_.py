import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import joblib
import numpy as np
import soundfile as sf
import openl3
import librosa
from pydantic import BaseModel
from typing import List
from pydub import AudioSegment
import io
import time  # For basic timing and more informative logs
import traceback
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import logging
import io
# --- Configuration ---
# Assuming main.py is in the 'app/' directory relative to the project root
APP_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger("uvicorn.error")
# Choose which model to use: 'svm' or 'wav2vec2'
MODEL_TYPE = os.getenv("MODEL_TYPE", "wav2vec2")  # Default to wav2vec2, can be overridden with env var

# Directory for deployed models within the 'app' directory
SAVED_MODEL_DIR = os.path.join(APP_DIR, "saved_models")
SAVED_NEW_MODEL_DIR = os.path.join(APP_DIR, "saved_new_model")

# SVM Model configuration
SVM_MODEL_BUNDLE_FILENAME = "model_and_scaler.joblib"
SVM_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, SVM_MODEL_BUNDLE_FILENAME)

# Wav2Vec2 Model configuration
WAV2VEC2_MODEL_PATH = os.path.join(SAVED_NEW_MODEL_DIR, "checkpoint-2121")

# For logging purposes
CHOSEN_MODEL_TYPE_FOR_LOGGING = MODEL_TYPE

MIN_AUDIO_DURATION_SEC = 3.0
TARGET_SR = 16000
# Assuming static_frontend is also in app/
STATIC_DIR = os.path.join(APP_DIR, "static_frontend")

# --- Model Loading ---
model = None
scaler = None
wav2vec2_model = None
wav2vec2_feature_extractor = None


async def load_model_and_scaler():
    global model, scaler, wav2vec2_model, wav2vec2_feature_extractor

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Server startup: Loading both SVM and Wav2Vec2 models...")
    
    # Load both models
    await load_svm_model()
    await load_wav2vec2_model()
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model loading completed. Default MODEL_TYPE: {MODEL_TYPE}")


async def load_svm_model():
    global model, scaler
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Loading SVM model from {SVM_MODEL_PATH}")

    if not os.path.exists(SAVED_MODEL_DIR):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Saved models directory not found: {SAVED_MODEL_DIR}. Attempting to create it.")
        try:
            os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Could not create saved models directory {SAVED_MODEL_DIR}: {e}")
            return

    if not os.path.exists(SVM_MODEL_PATH):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Model file not found at {SVM_MODEL_PATH}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Please ensure your trained model bundle ('{SVM_MODEL_BUNDLE_FILENAME}')")
        print(f"      is placed in the directory: '{SAVED_MODEL_DIR}'")
        print(f"      This structure should be part of your Docker image if deploying with Docker (e.g., app/saved_models/{SVM_MODEL_BUNDLE_FILENAME}).")
        return
    
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Loading joblib bundle from {SVM_MODEL_PATH}...")
        model_bundle = joblib.load(SVM_MODEL_PATH)
        model = model_bundle.get("model")
        scaler = model_bundle.get("scaler")
        if model is None or scaler is None:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: 'model' or 'scaler' key not found in the loaded joblib bundle from {SVM_MODEL_PATH}.")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: SVM model and scaler loaded successfully from {SVM_MODEL_PATH}.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Exception during SVM model loading from {SVM_MODEL_PATH}: {e}")
        import traceback
        traceback.print_exc()


async def load_wav2vec2_model():
    global wav2vec2_model, wav2vec2_feature_extractor
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Loading Wav2Vec2 model from {WAV2VEC2_MODEL_PATH}")

    if not os.path.exists(WAV2VEC2_MODEL_PATH):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Wav2Vec2 model directory not found at {WAV2VEC2_MODEL_PATH}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Please ensure your fine-tuned Wav2Vec2 model is placed in: '{WAV2VEC2_MODEL_PATH}'")
        return
    
    try:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Loading Wav2Vec2 model and feature extractor...")
        wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC2_MODEL_PATH)
        wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_PATH)
        
        # Set model to evaluation mode
        wav2vec2_model.eval()
        
        # Check if CUDA is available and move model to GPU if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav2vec2_model.to(device)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Wav2Vec2 model loaded successfully. Device: {device}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: Model config: {wav2vec2_model.config}")
        
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Exception during Wav2Vec2 model loading: {e}")
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
    
    # Log audio characteristics for debugging
    print(f"[{filename_for_log}] DEBUG: Audio stats - Min: {np.min(audio_data):.4f}, Max: {np.max(audio_data):.4f}, Mean: {np.mean(audio_data):.4f}, Std: {np.std(audio_data):.4f}")
    print(f"[{filename_for_log}] DEBUG: Audio RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
    
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

        # Log resampled audio characteristics
        print(f"[{filename_for_log}] DEBUG: After resampling - Min: {np.min(audio_data):.4f}, Max: {np.max(audio_data):.4f}, RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")

        # Normalize audio to prevent volume differences from affecting the model
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            print(f"[{filename_for_log}] INFO: Audio normalized. New range: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")
        
        # Apply a simple high-pass filter to remove low-frequency noise common in microphone recordings
        from scipy import signal
        try:
            # High-pass filter at 80Hz to remove low-frequency noise
            sos = signal.butter(4, 80, btype='highpass', fs=sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
            print(f"[{filename_for_log}] INFO: Applied high-pass filter (80Hz cutoff)")
        except Exception as filter_err:
            print(f"[{filename_for_log}] WARNING: Could not apply high-pass filter: {filter_err}")

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
        print(f"[{filename_for_log}] DEBUG: Embedding stats - Min: {np.min(mean_embedding):.4f}, Max: {np.max(mean_embedding):.4f}, Mean: {np.mean(mean_embedding):.4f}")
        return mean_embedding
    except Exception as e:
        print(
            f"[{filename_for_log}] ERROR: Exception during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_audio_for_wav2vec2(audio_data: np.ndarray, sample_rate: int, filename_for_log: str) -> np.ndarray | None:
    """Process audio specifically for Wav2Vec2 model"""
    print(f"[{filename_for_log}] INFO: Starting Wav2Vec2 audio processing. Initial audio shape: {audio_data.shape}, sample rate: {sample_rate}Hz.")
    
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            print(f"[{filename_for_log}] INFO: Audio is stereo, converting to mono.")
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
        if sample_rate != TARGET_SR:
            print(f"[{filename_for_log}] INFO: Resampling from {sample_rate}Hz to {TARGET_SR}Hz.")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            print(f"[{filename_for_log}] INFO: Audio normalized for Wav2Vec2.")

        # Pad if too short
        audio_data = pad_if_short(audio_data, sample_rate, MIN_AUDIO_DURATION_SEC, filename_for_log)
        
        print(f"[{filename_for_log}] INFO: Wav2Vec2 audio processing completed. Final shape: {audio_data.shape}")
        return audio_data
        
    except Exception as e:
        print(f"[{filename_for_log}] ERROR: Exception during Wav2Vec2 audio processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- API Endpoint ---


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


# --- Specific Model Endpoints ---

@app.post("/analyze_audio_wav2")
async def analyze_audio_wav2(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_data, _ = librosa.load(io.BytesIO(contents), sr=16000)
        
        inputs = feature_extractor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        label = "REAL" if predicted_class == 1 else "FAKE"
        confidence = torch.softmax(logits, dim=-1).squeeze().tolist()

        return {
            "prediction": label,
            "confidence": {
                "real": float(confidence[1]),
                "fake": float(confidence[0])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing audio: {str(e)}")

@app.get("/model_status/")
async def get_model_status():
    """Get the status of all loaded models"""
    return {
        "current_model_type": MODEL_TYPE,
        "svm_model_loaded": model is not None and scaler is not None,
        "wav2vec2_model_loaded": wav2vec2_model is not None and wav2vec2_feature_extractor is not None,
        "svm_model_path": SVM_MODEL_PATH,
        "wav2vec2_model_path": WAV2VEC2_MODEL_PATH,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

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

# To run: uvicorn app.main:app --host 0.0.0.0 --port 8000
# Ensure your model bundle is correctly placed, e.g.:   
# app/saved_models/model_and_scaler.joblib
# And your frontend:
# app/static_frontend/index.html
## cd /Users/giovannipoveda/Documents/deepfake_voice_clonning && uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile app/localhost-key.pem --ssl-certfile app/localhost.pem