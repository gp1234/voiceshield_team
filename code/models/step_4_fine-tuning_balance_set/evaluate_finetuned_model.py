import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoFeatureExtractor, Trainer, TrainingArguments
# Removed GroupShuffleSplit as it might be causing issues with flat directory structure
# from sklearn.model_selection import GroupShuffleSplit 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

# Import necessary components from fine_tune_setup.py
from fine_tune_setup import create_dataframe, prepare_dataset, get_data_collator, compute_metrics, get_feature_extractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("Script started.") # Added for debugging

    # --- Configuration ---
    # !!! IMPORTANT: SET THIS TO THE ROOT OF YOUR DATASET !!!
    # This path points directly to your 'audios' folder containing MP3s.
    DATASET_ROOT = "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/samples/audios" 
    
    # Path to your saved fine-tuned model (e.g., from fine_tune_model.py's OUTPUT_DIR/final_model)
    FINETUNED_MODEL_PATH = "./wav2vec2_fine_tuned_spoof_detection/final_model" 
    
    EVAL_BATCH_SIZE = 8
    # VAL_TEST_SPLIT_SIZE is no longer directly used for splitting if not using GSS
    # You will simply evaluate on all data found in DATASET_ROOT that matches the pattern.

    # --- 1. Load and Prepare Data ---
    logger.info("Loading and preparing dataset for evaluation...")
    print("Before create_dataframe.") # Added for debugging
    
    df = create_dataframe(DATASET_ROOT)
    
    print(f"DataFrame loaded with {len(df)} entries.") # Added for debugging
    logger.info(f"DataFrame has {len(df)} entries.") # Added for debugging

    # --- Ensure the feature extractor is initialized before mapping ---
    _ = get_feature_extractor() # Call once to ensure it's loaded

    # Simplified: Use the entire DataFrame as the test set for evaluation
    # This assumes DATASET_ROOT only contains the data you want to evaluate on.
    test_df = df 

    logger.info(f"Total samples for evaluation: {len(test_df)}")

    test_dataset = Dataset.from_pandas(test_df)
    print("Before dataset mapping and filtering.") # Added for debugging
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=["path", "group_id", "label"], num_proc=os.cpu_count()).filter(lambda x: x is not None)
    print(f"Test dataset has {len(test_dataset)} entries after mapping and filtering.") # Added for debugging
    logger.info(f"Test dataset has {len(test_dataset)} entries after mapping and filtering.") # Added for debugging


    # Initialize Data Collator
    data_collator = get_data_collator()

    # --- 2. Load Fine-tuned Model ---
    logger.info(f"Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
    print("Before model loading.") # Added for debugging
    try:
        model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
        # Ensure feature extractor is loaded correctly for the model
        feature_extractor = AutoFeatureExtractor.from_pretrained(FINETUNED_MODEL_PATH)
        print("Model and feature extractor loaded successfully.") # Added for debugging
    except Exception as e:
        print(f"ERROR: Failed to load model or feature extractor: {e}") # Added for debugging
        logger.error(f"Error loading model or feature extractor from {FINETUNED_MODEL_PATH}: {e}")
        logger.info("Please ensure the model path is correct and the model was saved properly.")
        return

    # --- 3. Configure Trainer for Evaluation ---
    logger.info("Setting up Trainer for evaluation...")
    eval_args = TrainingArguments(
        output_dir="./eval_results", # Temporary directory for evaluation logs
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        dataloader_num_workers=os.cpu_count(),
        report_to="none" # No need to report to TensorBoard for simple eval
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        eval_dataset=test_dataset, # Use test_dataset for evaluation
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor # Feature extractor is used as tokenizer for audio
    )

    # --- 4. Evaluate Model ---
    logger.info("Evaluating model on the test set...")
    print("Before evaluation.") # Added for debugging
    test_results = trainer.evaluate(test_dataset)
    print(f"Evaluation complete. Results: {test_results}") # Added for debugging
    logger.info(f"Final Test Set Evaluation Results: {test_results}")
    print("Script finished.") # Added for debugging


if __name__ == "__main__":
    main()