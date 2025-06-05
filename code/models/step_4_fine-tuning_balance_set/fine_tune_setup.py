import os
import re
import pandas as pd
import numpy as np
import librosa
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Feature Extractor Initialization ---
feature_extractor = None

def get_feature_extractor():
    """Get or initialize the feature extractor."""
    global feature_extractor
    if feature_extractor is None:
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
            logger.info("Feature extractor 'facebook/wav2vec2-base-960h' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load AutoFeatureExtractor: {e}")
            raise
    return feature_extractor

def create_dataframe(dataset_root: str) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from the structured dataset.
    """
    data = []
    
    real_data_path = os.path.join(dataset_root, 'real_data')
    fake_data_path = os.path.join(dataset_root, 'fake_data')

    if not os.path.isdir(real_data_path) or not os.path.isdir(fake_data_path):
        raise FileNotFoundError(
            f"Expected 'real_data' and 'fake_data' folders in '{dataset_root}'. "
            "Please check your dataset root path and structure."
        )

    # Process real data
    logger.info(f"Scanning real data from: {real_data_path}")
    for group_folder in os.listdir(real_data_path):
        group_path = os.path.join(real_data_path, group_folder)
        if os.path.isdir(group_path):
            match = re.search(r'group_(\d+)', group_folder)
            if not match:
                logger.warning(f"Skipping malformed group folder: {group_folder} in real_data.")
                continue
            group_id = int(match.group(1))
            for filename in os.listdir(group_path):
                if filename.endswith('.wav'):
                    data.append({
                        'path': os.path.join(group_path, filename),
                        'label': 0,
                        'group_id': group_id
                    })
    
    # Process fake data
    logger.info(f"Scanning fake data from: {fake_data_path}")
    for group_folder in os.listdir(fake_data_path):
        group_path = os.path.join(fake_data_path, group_folder)
        if os.path.isdir(group_path):
            match = re.search(r'group_(\d+)', group_folder)
            if not match:
                logger.warning(f"Skipping malformed group folder: {group_folder} in fake_data.")
                continue
            group_id = int(match.group(1))
            for filename in os.listdir(group_path):
                if filename.endswith('.wav'):
                    data.append({
                        'path': os.path.join(group_path, filename),
                        'label': 1,
                        'group_id': group_id
                    })
    
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No audio files found in the specified dataset root. Please check paths and file types.")
    
    logger.info(f"DataFrame created with {len(df)} samples.")
    logger.info(f"Real samples: {len(df[df['label']==0])}, Fake samples: {len(df[df['label']==1])}")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def prepare_dataset(example):
    """
    Prepares audio samples for the model.
    Loads audio, resamples, and extracts features.
    """
    feature_extractor = get_feature_extractor()
    
    # Handle LazyRow objects from datasets
    try:
        # Convert to dict if it's a LazyRow
        if hasattr(example, '__getitem__') and hasattr(example, 'keys'):
            if "path" in example:
                audio_path = example["path"]
                label = example["label"]
            else:
                logger.error(f"'path' key not found in example: {list(example.keys())}")
                return None
        else:
            logger.error(f"prepare_dataset received unexpected format: {type(example)}")
            return None
    except Exception as e:
        logger.error(f"Error accessing example data: {e}")
        return None
    
    try:
        # Load audio using librosa for more control
        audio_array, sample_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
        
        # Check if audio is valid
        if audio_array is None or len(audio_array) == 0:
            logger.warning(f"Audio file '{audio_path}' could not be loaded or is empty.")
            return None

        # Extract features
        inputs = feature_extractor(
            audio_array, 
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100000  # Adjust based on your audio length
        )
        
        return {
            "input_values": inputs.input_values[0].numpy(),
            "labels": label
        }
        
    except Exception as e:
        logger.error(f"Error processing audio file {audio_path}: {e}")
        return None

def get_data_collator():
    """Initializes and returns the DataCollator for audio classification."""
    feature_extractor = get_feature_extractor()
    return DataCollatorWithPadding(tokenizer=feature_extractor, padding="longest")

def compute_metrics(pred) -> Dict[str, float]:
    """
    Computes metrics for binary classification.
    """
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)

    return {
        "accuracy": accuracy, 
        "f1": f1, 
        "precision": precision, 
        "recall": recall
    }