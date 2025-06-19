import os
import re
import numpy as np
import pandas as pd
import librosa
import torch
from datasets import Dataset
from transformers import (
    Wav2Vec2ForSequenceClassification, # <--- CHANGED: Import specific model class
    AutoFeatureExtractor,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Union
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Feature Extractor Initialization ---
feature_extractor_instance = None

def get_feature_extractor():
    """Get or initialize the feature extractor. """
    global feature_extractor_instance
    if feature_extractor_instance is None:
        try:
            feature_extractor_instance = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
            logger.info("Feature extractor 'facebook/wav2vec2-base-960h' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load AutoFeatureExtractor: {e}")
            raise
    return feature_extractor_instance

# --- create_dataframe ---
def create_dataframe(dataset_root: str) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from the structured dataset.
    This version processes .mp3 files with _real.mp3 or _fake.mp3 suffixes in a single directory.
    It does NOT expect 'real_data' or 'fake_data' subfolders.
    """
    data = []

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root directory '{dataset_root}' not found. Please check this path.")

    logger.info(f"Scanning audio files in: {dataset_root}")
    for filename in os.listdir(dataset_root):
        if filename.endswith(".mp3"):
            file_path = os.path.join(dataset_root, filename)
            label = None
            group_id = 0

            if "_fake.mp3" in filename:
                label = 1
            elif "_real.mp3" in filename:
                label = 0

            if label is not None:
                data.append({
                    'path': file_path,
                    'label': label,
                    'group_id': group_id
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

# --- prepare_dataset ---
def prepare_dataset(example):
    """
    Prepares audio samples for the model.
    Loads audio, resamples, and extracts features.
    """
    feature_extractor_local = get_feature_extractor()

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
        audio_array, sample_rate = librosa.load(audio_path, sr=feature_extractor_local.sampling_rate)

        if audio_array is None or len(audio_array) == 0:
            logger.warning(f"Audio file '{audio_path}' could not be loaded or is empty. Skipping.")
            return None

        if not np.isfinite(audio_array).all():
            logger.warning(f"Audio file '{audio_path}' contains non-finite values (NaN/Inf). Skipping.")
            return None

        inputs = feature_extractor_local(
            audio_array,
            sampling_rate=feature_extractor_local.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100000
        )

        return {
            "input_values": inputs.input_values[0].numpy(),
            "labels": label
        }

    except Exception as e:
        logger.error(f"Error processing audio file {audio_path}: {e}")
        return None

# --- get_data_collator ---
def get_data_collator():
    """Initializes and returns the DataCollator for audio classification."""
    feature_extractor_local = get_feature_extractor()
    return DataCollatorWithPadding(tokenizer=feature_extractor_local, padding="longest")

# --- compute_metrics ---
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

# --- Main evaluation logic ---
def main():
    print("Script started.")

    # --- Configuration ---
    DATASET_ROOT = "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/samples/audios"

    # THIS IS THE CORRECTED ABSOLUTE PATH FOR YOUR FINE-TUNED MODEL
    FINETUNED_MODEL_PATH = "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/models/step_4_fine-tuning_2/results/checkpoint-2121"

    EVAL_BATCH_SIZE = 8

    # --- 1. Load and Prepare Data ---
    logger.info("Loading and preparing dataset for evaluation...")
    print("Before create_dataframe.")

    df = create_dataframe(DATASET_ROOT)

    print(f"DataFrame loaded with {len(df)} entries.")
    logger.info(f"DataFrame has {len(df)} entries.")

    _ = get_feature_extractor()

    test_df = df

    logger.info(f"Total samples for evaluation: {len(test_df)}")

    test_dataset = Dataset.from_pandas(test_df)
    print("Before dataset mapping and filtering.")
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=["path", "group_id", "label"], num_proc=os.cpu_count()).filter(lambda x: x is not None)
    print(f"Test dataset has {len(test_dataset)} entries after mapping and filtering.")
    logger.info(f"Test dataset has {len(test_dataset)} entries after mapping and filtering.")

    data_collator = get_data_collator()

    # --- 2. Load Fine-tuned Model ---
    logger.info(f"Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
    print("Before model loading.")
    try:
        # <--- CHANGED: Use Wav2Vec2ForSequenceClassification directly
        model = Wav2Vec2ForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
        feature_extractor_for_trainer = AutoFeatureExtractor.from_pretrained(FINETUNED_MODEL_PATH)
        print("Model and feature extractor loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model or feature extractor: {e}")
        logger.error(f"Error loading model or feature extractor from {FINETUNED_MODEL_PATH}: {e}")
        logger.info("Please ensure the model path is correct and the model was saved properly.")
        return

    # --- 3. Configure Trainer for Evaluation ---
    logger.info("Setting up Trainer for evaluation...")
    eval_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        dataloader_num_workers=os.cpu_count(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor_for_trainer
    )

    # --- 4. Evaluate Model ---
    logger.info("Evaluating model on the test set...")
    print("Before evaluation.")
    test_results = trainer.evaluate(test_dataset)
    print(f"Evaluation complete. Results: {test_results}")
    logger.info(f"Final Test Set Evaluation Results: {test_results}")
    print("Script finished.")


if __name__ == "__main__":
    main()