import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles 
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import numpy as np
import soundfile as sf
import openl3
import librosa
from pydantic import BaseModel
from typing import List
from pydub import AudioSegment # Import pydub
import io # To handle byte streams with pydub

# --- Configuration ---
MODEL_DIR = "saved_models" # This should be relative to where main.py is, or an absolute path
# Assuming your training script (train_from_openl3_v2.py) saves models like:
# <project_root>/results_openl3_real_is_1/<model_name>/model_and_scaler.joblib
# And your FastAPI app is in <project_root>/app/main.py
# You might need to adjust this path based on your actual structure.
# For simplicity, let's assume 'saved_models' is next to 'main.py' for now.
# If your 'model_and_scaler.joblib' is for a specific model (e.g., svm), reflect that.
# Let's assume you've chosen one model, e.g., svm, and placed its bundle here:
CHOSEN_MODEL_NAME = "svm" # Example: choose which model to use
MODEL_PATH = os.path.join(MODEL_DIR, f"{CHOSEN_MODEL_NAME}_model_and_scaler.joblib") 

MIN_AUDIO_DURATION_SEC = 3.0
TARGET_SR = 16000 
STATIC_DIR = "static_frontend" 

# --- Application Setup ---
app = FastAPI(title="Deepfake Voice Analyzer")

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

# --- Model Loading ---
model_bundle = None
model = None
scaler = None

@app.on_event("startup")
async def load_model_and_scaler():
    global model_bundle, model, scaler
    
    # Adjust MODEL_PATH if your structure is different or if you want to load a specific model
    # For example, if your training script saves into 'results_openl3_real_is_1/svm/model_and_scaler.joblib'
    # and 'main.py' is in 'app/', then MODEL_PATH might be:
    # training_results_path = os.path.join("..", "code", "models_testing", "step_3_embeddings", "results", CHOSEN_MODEL_NAME, "model_and_scaler.joblib")
    # actual_model_path = os.path.abspath(training_results_path)
    
    # For this example, let's assume you've copied your chosen model bundle to 'saved_models'
    # relative to where main.py is.
    
    # Create saved_models directory if it doesn't exist (for clarity, though model should be there)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Example: Manually specify the path to the model you want to use from your training output
    # This assumes your training output is in ../code/models_testing/step_3_embeddings/results/
    # And main.py is in an 'app' folder at the root.
    # Path relative from current file (main.py) to the model file
    # Example: if main.py is in project_root/app/ and models are in project_root/code/models_testing/step_3_embeddings/results/
    # This needs to be robust. For now, let's assume a simpler 'saved_models' dir.
    
    # Corrected path assuming 'saved_models' is in the same directory as main.py
    # and you've copied the specific model bundle there.
    # Example: saved_models/svm_model_and_scaler.joblib
    
    # Let's try to make the path more robust based on typical project structure
    # If main.py is in 'app' folder, and 'results_openl3_real_is_1' is at project root:
    project_root = os.path.dirname(os.path.abspath(__file__)) # if main.py is at root
    # If main.py is in an 'app' subfolder:
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    
    # This is a common source of error. For now, let's use a simplified path.
    # Ensure MODEL_PATH points to the correct .joblib file.
    # If your training script created 'results_openl3_real_is_1/svm/model_and_scaler.joblib',
    # you need to make MODEL_PATH point there, or copy that file to 'saved_models/svm_model_and_scaler.joblib'
    
    # For the sake of this example, let's assume you have copied the desired model to:
    # saved_models/model_and_scaler.joblib (generic name)
    # OR saved_models/svm_model_and_scaler.joblib (if you set CHOSEN_MODEL_NAME)

    # Let's assume a fixed name for the bundle in saved_models for simplicity here
    fixed_bundle_name = "model_and_scaler.joblib" # You'd copy your chosen model (e.g. svm's) here
    actual_model_path = os.path.join(MODEL_DIR, fixed_bundle_name)


    if not os.path.exists(actual_model_path):
        print(f"ERROR: Model file not found at {actual_model_path}")
        print(f"Please ensure your trained model (e.g., from 'results_openl3_real_is_1/{CHOSEN_MODEL_NAME}/model_and_scaler.joblib')")
        print(f"is copied to '{MODEL_DIR}/{fixed_bundle_name}' relative to where main.py is run.")
        return
    try:
        model_bundle = joblib.load(actual_model_path)
        model = model_bundle.get("model")
        scaler = model_bundle.get("scaler")
        if model is None or scaler is None:
            print(f"ERROR: 'model' or 'scaler' not found in the loaded joblib bundle from {actual_model_path}.")
        else:
            print(f"Model and scaler loaded successfully from {actual_model_path}.")
    except Exception as e:
        print(f"Error loading model from {actual_model_path}: {e}")


# --- Pydantic Models ---
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    filename: str

# --- Helper Functions ---
def pad_if_short(audio_data: np.ndarray, sample_rate: int, min_duration: float) -> np.ndarray:
    target_len = int(sample_rate * min_duration)
    if len(audio_data) < target_len:
        pad_width = target_len - len(audio_data)
        return np.pad(audio_data, (0, pad_width), mode='constant')
    return audio_data

def extract_embedding(audio_data: np.ndarray, sample_rate: int, content_type="music", embedding_size=512) -> np.ndarray | None:
    try:
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sample_rate != TARGET_SR:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
            sample_rate = TARGET_SR
        audio_data = pad_if_short(audio_data, sample_rate, MIN_AUDIO_DURATION_SEC)
        # Reverted to "music" as "speech" caused issues with default OpenL3 loader
        emb, _ = openl3.get_audio_embedding(audio_data, sample_rate, content_type="music", embedding_size=embedding_size, hop_size=0.5)
        if emb.shape[0] == 0: return None
        return np.mean(emb, axis=0)
    except Exception as e:
        print(f"Error during embedding extraction: {e}")
        return None

# --- API Endpoint ---
@app.post("/analyze_audio/", response_model=PredictionResponse)
async def analyze_audio(file: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded. Check server logs.")
    
    # Read uploaded file content
    file_content = await file.read()
    
    # Use an in-memory buffer for pydub
    audio_buffer = io.BytesIO(file_content)
    
    temp_wav_path = f"temp_converted_{file.filename}.wav" # For soundfile to read from after pydub

    try:
        # Load audio with pydub (handles various formats)
        # pydub might need the file extension to guess format, or you can specify it
        # For uploaded files, it's safer to let it infer or catch errors.
        try:
            audio_segment = AudioSegment.from_file(audio_buffer)
        except Exception as pydub_err: # Catch pydub specific errors
            print(f"Pydub error loading file {file.filename}: {pydub_err}")
            raise HTTPException(status_code=400, detail=f"Could not process audio format: {pydub_err}")

        # Export to WAV format in memory (or to a temp file if sf.read needs path)
        # Forcing WAV ensures soundfile can read it.
        # Soundfile typically needs a file path or a file-like object that supports seek.
        # Exporting to a temp file is more robust here.
        audio_segment.export(temp_wav_path, format="wav")
        
        # Now read the standardized WAV file with soundfile
        audio_data, sample_rate = sf.read(temp_wav_path)

        embedding = extract_embedding(audio_data, sample_rate)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Could not extract features from audio.")
        
        # L2 Normalization (if done during training, apply here too)
        # This was not in your training script, so commenting out.
        # If you add it to training, uncomment here.
        # embedding_norm = np.linalg.norm(embedding)
        # normalized_embedding = embedding / embedding_norm if embedding_norm > 0 else embedding
        normalized_embedding = embedding # Assuming no explicit L2 norm in training

        scaled_embedding = scaler.transform(normalized_embedding.reshape(1, -1))
        prediction_value = model.predict(scaled_embedding)[0]
        # Your convention: Real=1, Fake=0
        prediction_label = "REAL" if prediction_value == 1 else "FAKE"
        
        probabilities = model.predict_proba(scaled_embedding)[0]
        
        # Confidence for the predicted class
        if prediction_label == "REAL": # Model predicted 1
            confidence = probabilities[1] * 100 # Probability of class 1 (Real)
        else: # Model predicted 0 (Fake)
            confidence = probabilities[0] * 100 # Probability of class 0 (Fake)

    except Exception as e:
        print(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        await file.close()


    return PredictionResponse(
        prediction=prediction_label,
        confidence=round(confidence, 2),
        filename=file.filename
    )

# --- Route to serve HTML ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_file_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="Frontend not found.")
    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# To run: uvicorn main:app --reload
# Ensure 'saved_models/your_chosen_model_and_scaler.joblib' exists
# and 'static_frontend/index.html' exists.
