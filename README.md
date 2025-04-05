# Deepfake Voice Detection System 📣

This project implements a comprehensive system for detecting synthetic voices using various audio processing and machine learning techniques. The system processes audio files, extracts features, and uses machine learning models to classify between real and synthetic voices.

## Project Overview

The Deepfake Voice Detection System is designed to identify synthetic voices with high accuracy. It employs a multi-stage approach to feature extraction and model training, with each stage building upon the previous ones to improve detection capabilities.

## Project Structure

```
├── code/                  # Main codebase
│   ├── assets/            # Data storage
│   │   ├── audio/         # Audio files (original and processed)
│   │   └── transcripts/   # Text transcripts for synthetic voice generation
│   ├── models_testing/    # Model evaluation and selection
│   │   ├── step_1_basic_mggc/      # Basic MFCC analysis
│   │   ├── step_2_augmented_mgcc/  # Augmented MFCC analysis
│   │   ├── step_3_embeddings/      # Voice embeddings analysis
│   │   └── step_4_fine_tune/       # Fine-tuned models
│   ├── notebooks/         # Jupyter notebooks for exploration and analysis
│   └── scripts/           # Utility scripts for audio processing and model training
├── EDA/                   # Exploratory Data Analysis
│   └── EDA_commonVoice_Musa.ipynb  # Analysis of Common Voice dataset
└── voxceleb-aws-api/      # API for VoxCeleb dataset integration
```

## Components

### 1. Code Module

The main codebase contains the core functionality for voice detection. It includes:

- **Feature Extraction**: Processing audio files to extract MFCC features and voice embeddings
- **Model Training**: Training various machine learning models for voice classification
- **Model Evaluation**: Testing and comparing different models for optimal performance
- **Data Augmentation**: Techniques to enhance the training dataset

For detailed instructions on running the code module, see [code/README.md](code/README.md).

### 2. Exploratory Data Analysis (EDA)

The EDA directory contains notebooks for analyzing voice datasets, particularly the Common Voice dataset. These analyses help understand the characteristics of real and synthetic voices.

### 3. VoxCeleb API

The voxceleb-aws-api directory contains an API for integrating with the VoxCeleb dataset, which provides a large collection of voice samples for training and testing.

## Key Findings

1. **Basic MFCC Analysis**:

   - Simple MFCC features alone are insufficient for robust voice detection
   - Need for more sophisticated feature extraction methods
   - Single dataset testing reveals limitations in generalization
   - Train-test splitting can lead to data leakage if not properly implemented

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

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/deepfake_voice_clonning.git
   cd deepfake_voice_clonning
   ```

2. Follow the setup instructions in [code/README.md](code/README.md) to set up the main codebase.

3. For the VoxCeleb API, follow the instructions in [voxceleb-aws-api/README.md](voxceleb-aws-api/README.md).

4. Explore the EDA notebooks to understand the dataset characteristics.

## Future Work

- Integration of more advanced voice embedding techniques
- Development of a real-time detection system
- Expansion to more voice datasets for improved generalization
- Implementation of adversarial training for better robustness
