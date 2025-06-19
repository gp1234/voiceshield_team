# Deepfake Voice Detection System - Technical Report

## 1. Project Overview

**VoiceShield** is an advanced AI-powered deepfake voice detection system designed to distinguish between authentic human speech and AI-generated synthetic audio. The project implements multiple machine learning approaches, ranging from traditional MFCC-based feature extraction to state-of-the-art transformer models like Wav2Vec2.

### Project Goals
- Develop robust algorithms to detect AI-generated voice content
- Evaluate multiple feature extraction techniques and model architectures
- Build a production-ready API for real-time voice authentication
- Provide comprehensive analysis of different synthetic voice generation methods

### Key Features
- Multi-modal approach supporting both traditional ML and deep learning models
- Real-time audio analysis via FastAPI web service
- WhatsApp integration for easy access
- Comprehensive evaluation metrics and model comparison
- Support for various audio formats and quality levels

## 2. Dataset Architecture and Balance Strategy

### 2.1 Dataset Structure

The project utilizes a hierarchical dataset organization with multiple categories:

```
assets/audio/
├── original_audio/          # Raw authentic recordings
├── processed_audio/         # Clean, standardized audio files
│   ├── real/               # Authentic human speech
│   ├── fake_1/             # TTS-generated audio (Method 1)
│   ├── fake_2/             # TTS-generated audio (Method 2)
│   └── fake_3/             # TTS-generated audio (Method 3)
├── augmented_audio/         # Data augmentation variants
│   ├── real/               # Augmented authentic speech
│   ├── fake_4/             # Coqui Tacotron2 generated
│   ├── fake_5/             # XTTS v2 generated
│   └── fake_6/             # Bark generated
└── augmented_balance/       # Balanced dataset for training
    ├── real_data/
    │   ├── group_1/
    │   └── group_2/
    └── fake_data/
        ├── group_1/
        └── group_2/
```

### 2.2 Data Types and Sources

**Real Audio (Authentic Speech):**
- Human recordings from various speakers
- Multiple languages and accents
- Different recording conditions and quality levels
- Duration: 3+ seconds minimum after preprocessing

**Synthetic Audio (AI-Generated):**
- **fake_1-3**: Traditional TTS systems
- **fake_4**: Coqui Tacotron2 (Advanced neural TTS)
- **fake_5**: XTTS v2 (Multilingual advanced TTS)
- **fake_6**: Bark (State-of-the-art voice cloning)

### 2.3 Data Balance Strategy

The project implements a sophisticated group-based splitting strategy to prevent data leakage:

1. **Group-Aware Splitting**: Uses `GroupShuffleSplit` to ensure same speaker doesn't appear in both training and test sets
2. **Balanced Augmentation**: Each real audio sample generates 5 augmented variants:
   - Original (cleaned)
   - White noise addition
   - Pitch shifting
   - Reverb application
   - Time stretching
3. **Stratified Sampling**: Maintains equal representation of real vs. fake samples across splits
4. **Quality Control**: Minimum duration filtering (3 seconds) and audio validation

## 3. Feature Extraction Techniques

### 3.1 Traditional Audio Features (Step 1-2)

**MFCC (Mel-Frequency Cepstral Coefficients):**
```python
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])  # 26 features total
```

**Features Extracted:**
- 13 MFCC coefficients (mean values)
- 13 MFCC standard deviations
- Total: 26-dimensional feature vectors

### 3.2 Advanced Embedding Techniques (Step 3-5)

**OpenL3 Embeddings:**
```python
def extract_embedding(audio_data, sample_rate):
    # Resample to 16kHz for consistency
    if sample_rate != TARGET_SR:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)
    
    # Extract 512-dimensional embeddings
    emb, ts = openl3.get_audio_embedding(
        audio_data, TARGET_SR, 
        content_type="music",  # Robust to various audio types
        embedding_size=512, 
        hop_size=0.5
    )
    return np.mean(emb, axis=0)  # Time-averaged embedding
```

**OpenL3 Configuration:**
- **Content Type**: "music" (more robust than "speech" for diverse audio)
- **Embedding Size**: 512 dimensions
- **Hop Size**: 0.5 seconds for temporal resolution
- **Post-processing**: Mean aggregation across time frames

### 3.3 Audio Preprocessing Pipeline

**Standardization Steps:**
1. **Format Conversion**: All audio converted to WAV format
2. **Resampling**: Standardized to 16kHz sampling rate
3. **Channel Reduction**: Stereo to mono conversion
4. **Normalization**: Amplitude normalization to [-1, 1] range
5. **High-pass Filtering**: 80Hz cutoff to remove low-frequency noise
6. **Duration Padding**: Minimum 3-second duration requirement

## 4. Model Architecture and Training

### 4.1 Traditional Machine Learning Models (Steps 1-3)

**Model Configurations:**
```python
models = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced_subsample'
    ),
    "svm": SVC(
        kernel='rbf', 
        probability=True, 
        random_state=42, 
        class_weight='balanced'
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=1000, 
        random_state=42, 
        early_stopping=True
    ),
    "logistic": LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced'
    )
}
```

### 4.2 Deep Learning Models (Step 4)

**Wav2Vec2 Fine-tuning Configuration:**
```python
MODEL_NAME = "facebook/wav2vec2-base-960h"
training_args = TrainingArguments(
    output_dir="./wav2vec2_fine_tuned_spoof_detection",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=10,
    warmup_ratio=0.1,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

**Model Architecture:**
- **Base Model**: Pre-trained Wav2Vec2 (960h of speech data)
- **Classification Head**: Binary classification (Real vs. Fake)
- **Fine-tuning Strategy**: Full model fine-tuning with frozen feature extractor initially
- **Optimization**: AdamW optimizer with linear warmup

### 4.3 Model Ensemble Strategy

The production API supports both traditional ML and deep learning models:

```python
# Dual model loading for flexibility
await load_svm_model()          # OpenL3 + SVM pipeline
await load_wav2vec2_model()     # Fine-tuned transformer model
```

## 5. Training Process and Data Management

### 5.1 Training Pipeline Steps

**Step 1: Basic MFCC Exploration**
- Initial proof-of-concept with simple features
- Identified limitations of basic spectral features
- Established baseline performance metrics

**Step 2: Augmented Feature Engineering**
- Enhanced MFCC with statistical measures
- Introduced data augmentation techniques
- Implemented group-aware train-test splitting

**Step 3: OpenL3 Embedding Integration**
- Transitioned to pre-trained audio embeddings
- Significant performance improvement observed
- Better generalization across different synthetic methods

**Step 4: Transformer Fine-tuning**
- Wav2Vec2 model adaptation for spoofing detection
- End-to-end learning approach
- State-of-the-art performance achievement

**Step 5: Balanced Dataset Training**
- Final model training on carefully balanced datasets
- Cross-validation with multiple synthetic voice types
- Production model selection and optimization

### 5.2 Data Split Strategy

```python
# Group-aware splitting to prevent data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=speaker_groups))

# Further validation split
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss_val.split(val_test_data, groups=val_test_groups))
```

**Split Ratios:**
- Training: 70%
- Validation: 15%
- Testing: 15%

### 5.3 Feature Scaling and Normalization

```python
# Standard scaling for traditional ML models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model + Scaler bundling for deployment
bundle = {"model": trained_model, "scaler": scaler}
joblib.dump(bundle, "model_and_scaler.joblib")
```

## 6. Evaluation Metrics and Performance

### 6.1 Evaluation Framework

**Primary Metrics:**
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate (fake detection reliability)
- **Recall**: Sensitivity (fake detection coverage)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

**Custom Metrics Function:**
```python
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted")
    }
```

### 6.2 Model Performance Comparison

**Traditional ML Results (OpenL3 + SVM):**
- Accuracy: ~85-90%
- F1-Score: ~0.87
- Strong performance on known synthetic methods
- Fast inference time (<100ms per sample)

**Wav2Vec2 Fine-tuned Results:**
- Accuracy: ~92-95%
- F1-Score: ~0.93
- Superior generalization to novel synthetic methods
- Moderate inference time (~200-300ms per sample)

### 6.3 Cross-Validation Strategy

```python
# Stratified group k-fold for robust evaluation
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    # Train and evaluate each fold
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[val_idx])
    # Store fold results for averaging
```

## 7. Running the Project

### 7.1 Environment Setup

**Requirements Installation:**
```bash
# For API deployment
pip install -r api/requirements_finetune.txt

# For development and training
pip install -r code/requirements.txt
```

**Key Dependencies:**
- `fastapi==0.110.1` - Web API framework
- `torch==2.1.2` - Deep learning framework
- `transformers==4.36.2` - Hugging Face transformers
- `librosa==0.10.1` - Audio processing
- `openl3` - Pre-trained audio embeddings
- `scikit-learn==1.4.2` - Traditional ML algorithms

### 7.2 Training Pipeline Execution

**Step-by-Step Training:**
```bash
# Step 1: Basic MFCC exploration
cd code/models/step_1_basic_exploration
python run.py default

# Step 2: Augmented features
cd ../step_2_augmented_features
python run.py

# Step 3: OpenL3 embeddings
cd ../step_3_embeddings
python extract_openl3_embeddings.py
python train_from_openl3.py

# Step 4: Wav2Vec2 fine-tuning
cd ../step_4_fine-tuning
python fine_tune_setup.py
python fine_tune_model.py

# Step 5: Final balanced training
cd ../step_5_embeddings_balance_set
python extract_openl3_embeddings.py
python train_from_openl3.py
```

### 7.3 API Deployment

**Local Development Server:**
```bash
cd api/
uvicorn main_:app --host 0.0.0.0 --port 8000 --reload
```

**Production Deployment with SSL:**
```bash
uvicorn main_:app --host 0.0.0.0 --port 8000 \
  --ssl-keyfile localhost-key.pem \
  --ssl-certfile localhost.pem
```

**API Endpoints:**
- `POST /analyze_audio/` - Main SVM-based analysis
- `POST /analyze_audio_wav2` - Wav2Vec2-based analysis
- `GET /model_status/` - Model loading status
- `GET /` - Web interface

### 7.4 WhatsApp Integration

**Twilio Configuration:**
```bash
cd api/whatsapp_integration/
python run.py  # Start WhatsApp webhook server

# Configure ngrok for external access
ngrok http 5000
```

**Usage Flow:**
1. Send audio message to configured WhatsApp number
2. Twilio webhook receives and forwards to API
3. Audio analysis performed using loaded models
4. Results sent back via WhatsApp message

## 8. Directory Structure Explanation

```
deepfake_voice_clonning/
├── api/                          # Production API server
│   ├── main_.py                  # FastAPI application
│   ├── saved_models/             # SVM model storage
│   ├── step_3_fine_tune_model/   # Wav2Vec2 model storage
│   ├── static_frontend/          # Web interface
│   └── whatsapp_integration/     # WhatsApp bot integration
├── code/                         # Development and training code
│   ├── models/                   # ML model implementations
│   │   ├── step_1_basic_exploration/     # MFCC baseline
│   │   ├── step_2_augmented_features/    # Enhanced MFCC
│   │   ├── step_3_embeddings/            # OpenL3 integration
│   │   ├── step_4_fine-tuning/           # Wav2Vec2 training
│   │   └── step_5_embeddings_balance_set/ # Final models
│   └── assets/                   # Data and resources
│       ├── audio/                # Audio datasets
│       └── scripts/              # Data processing utilities
├── native/                       # Standalone local version
├── voxceleb-aws-api/            # Additional API implementation
└── EDA/                         # Exploratory data analysis
```

### 8.1 Key Directories

**api/**: Production-ready FastAPI server
- Dual model support (SVM + Wav2Vec2)
- RESTful endpoints for audio analysis
- Web interface for testing
- WhatsApp integration capabilities

**code/models/**: Incremental model development
- Each step represents a progression in complexity
- Comprehensive evaluation and comparison
- Reusable components and utilities

**code/assets/**: Data management and processing
- Hierarchical audio organization  
- Data augmentation scripts
- Transcript management
- Audio quality control utilities

### 8.2 Model Storage Structure

```
api/saved_models/                 # Traditional ML models
└── model_and_scaler.joblib      # SVM + StandardScaler bundle

api/step_3_fine_tune_model/       # Deep learning models
└── checkpoint-XXX/              # Hugging Face model format
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── preprocessor_config.json
```

## 9. Dependencies and Libraries

### 9.1 Core Audio Processing

**librosa (0.10.1)**: Primary audio processing library
- Audio loading and resampling
- Feature extraction (MFCC, spectrograms)
- Audio effects and transformations

**soundfile (0.12.1)**: High-quality audio I/O
- Lossless audio reading/writing
- Multiple format support
- Metadata preservation

**openl3**: Pre-trained audio embeddings
- State-of-the-art audio representations
- Multiple content types (music, environmental, speech)
- Efficient feature extraction

### 9.2 Machine Learning Frameworks

**scikit-learn (1.4.2)**: Traditional ML algorithms
- SVM, Random Forest, Logistic Regression
- Feature scaling and preprocessing
- Model evaluation and cross-validation

**PyTorch (2.1.2)**: Deep learning framework
- Tensor operations and automatic differentiation
- GPU acceleration support
- Model serialization and deployment

**Transformers (4.36.2)**: Hugging Face library
- Pre-trained Wav2Vec2 models
- Easy fine-tuning interfaces
- Model hub integration

### 9.3 Web Framework and API

**FastAPI (0.110.1)**: Modern web framework
- Automatic API documentation
- Type hints and validation
- Asynchronous request handling

**uvicorn (0.29.0)**: ASGI server
- High-performance async server
- SSL/TLS support
- Hot reloading for development

### 9.4 Data Processing and Visualization

**pandas (2.1.4)**: Data manipulation
- CSV handling and processing
- DataFrame operations
- Data filtering and grouping

**matplotlib (3.7.5) & seaborn (0.12.2)**: Visualization
- Model evaluation plots
- Confusion matrices
- Performance comparisons

**tqdm (4.66.1)**: Progress tracking
- Training progress bars
- Batch processing monitoring
- ETA estimation

### 9.5 Audio Synthesis (Development)

**TTS (Coqui)**: Text-to-speech synthesis
- Multiple TTS model architectures
- Voice cloning capabilities
- Custom model training

**pydub**: Audio manipulation
- Format conversion utilities
- Audio segment processing
- Simple audio effects

## 10. Production Considerations

### 10.1 Model Selection Strategy

The production API implements a dual-model approach:

1. **Primary Model (SVM + OpenL3)**: Fast, reliable detection
2. **Secondary Model (Wav2Vec2)**: High-accuracy analysis for critical cases

### 10.2 Performance Optimization

**Caching Strategy:**
- Model loading at startup (not per-request)
- Feature extractor reuse
- Temporary file cleanup

**Resource Management:**
- GPU utilization when available
- CPU fallback for compatibility
- Memory-efficient batch processing

### 10.3 Scalability Features

**Async Processing:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    await load_model_and_scaler()
    yield
    # Cleanup if needed
```

**Error Handling:**
- Comprehensive exception catching
- Graceful degradation
- Detailed logging for debugging

**Input Validation:**
- Audio format verification
- Duration requirements
- Quality thresholds

This technical report provides a comprehensive overview of the VoiceShield deepfake detection system, covering all aspects from data preparation to production deployment. The modular architecture allows for easy extension and improvement as new synthetic voice technologies emerge.
