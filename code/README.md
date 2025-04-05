# Deepfake Voice Detection System

This project implements a system for detecting synthetic voices using various audio processing and machine learning techniques. The system processes audio files, extracts features, and uses machine learning models to classify between real and synthetic voices.


## Project Structure

```
code/
├── assets/                 # Data storage
│   ├── audio/             # Audio files (original and processed)
│   └── transcripts/       # Text transcripts for synthetic voice generation
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── models_testing/        # Model evaluation and selection
└── scripts/              # Utility scripts for audio processing and model training
```

## Setup

1. Install Poetry (Python package manager):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install project dependencies:

```bash
cd code
poetry install
```

3. Activate the virtual environment:

```bash
poetry shell
```

## Workflow

The project follows a structured approach to voice detection, with each step building upon the previous ones. Each script on the top has the way to run the scripts.

### 1. Basic MFCC Analysis

- Purpose: Initial exploration of MFCC (Mel-frequency cepstral coefficients) features. We add reports for each of the models tested.
- Key Findings:
  - MFCC features alone show limited generalization capabilities
  - Basic feature extraction provides a foundation for more complex analysis
  - Single dataset testing reveals the need for more diverse data sources
  - We are getting data leakage using tes_train_split
  - We try to change different parameters to look how well the models perform. MLP seems a good candidate.

## Running the Pipeline

1. Generate synthetic audio:

```bash
python scripts/generate_synthetized_audio.py
```

2. Go to each of the steps and run the files to see the results

## Key Conclusions

1. **Basic MFCC Analysis**:

   - Simple MFCC features alone are insufficient for robust voice detection
   - Need for more sophisticated feature extraction methods
   - Single dataset testing reveals limitations in generalization
   - train_test_split generate data leakeage
   - Trying to change the parameters to underfit and overfit give us

2. **Data Augmentation**:

   - Multiple data sources significantly improve model performance
   - Group-based train-test splitting prevents data leakage
   - Augmented data helps capture various voice characteristics
   - Results show clear improvement over basic MFCC analysis

3. **Embeddings Analysis**:

   - Voice embeddings provide better feature representation
   - Improved separation between real and synthetic voices
   - More robust to variations in voice characteristics
   - Better generalization capabilities

4. **Fine-tuning**:
   - Fine-tuned models show superior performance
   - Better handling of edge cases
   - Improved robustness against various synthetic voices
   - More reliable in production environments
