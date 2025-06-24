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
    This version processes .mp3 files with _real.mp3 or _fake.mp3 suffixes in a single directory.
    It does NOT expect 'real_data' or 'fake_data' subfolders.
    """
    data = []
    
    # This is the only directory existence check needed now
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root directory '{dataset_root}' not found. Please check this path.")

    logger.info(f"Scanning audio files in: {dataset_root}")
    for filename in os.listdir(dataset_root):
        # Check for .mp3 extension and the _fake.mp3 or _real.mp3 suffix
        if filename.endswith(".mp3"):
            file_path = os.path.join(dataset_root, filename)
            label = None
            group_id = 0 # Default group_id if not extracted from filename.

            if "_fake.mp3" in filename:
                label = 1  # Fake audio (assuming 1 for fake, 0 for real)
            elif "_real.mp3" in filename:
                label = 0  # Real audio
            
            if label is not None:
                # You can add logic here to extract a more specific group_id from the filename
                # if your filenames contain a pattern (e.g., 'speakerX_common_voice_..._real.mp3')
                # For example:
                # match = re.search(r'common_voice_en_(\d+)', filename)
                # if match:
                #     group_id = int(match.group(1))
                data.append({
                    'path': file_path,
                    'label': label,
                    'group_id': group_id # Use extracted or default group_id
                })
            else:
                logger.warning(f"Skipping file '{filename}': Does not have '_fake.mp3' or '_real.mp3' suffix.")
    
    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("No audio files with '_fake.mp3' or '_real.mp3' suffix found in the specified dataset root. "
                         "Please check paths and file naming conventions.")
    
    logger.info(f"DataFrame created with {len(df)} samples.")
    logger.info(f"Real samples: {len(df[df['label']==0])}, Fake samples: {len(df[df['label']==1])}")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def prepare_dataset(example):
    """
    Prepares audio samples for the model.
    Loads audio, resamples, and extracts features.
    """
    feature_extractor = get_feature_extractor()
    
    try:
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
        # librosa.load can handle .mp3 files
        audio_array, sample_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
        
        # Check if audio is valid
        if audio_array is None or len(audio_array) == 0:
            logger.warning(f"Audio file '{audio_path}' could not be loaded or is empty. Skipping.")
            return None
        
        # Check for non-finite values in the loaded audio array
        if not np.isfinite(audio_array).all():
            logger.warning(f"Audio file '{audio_path}' contains non-finite values (NaN/Inf). Skipping.")
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