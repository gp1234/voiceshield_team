import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import GroupShuffleSplit
import logging

# Import necessary components from fine_tune_setup.py
from fine_tune_setup import create_dataframe, prepare_dataset, get_data_collator, compute_metrics, get_feature_extractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # --- Configuration ---
    MODEL_NAME = "facebook/wav2vec2-base-960h"
    OUTPUT_DIR = "./wav2vec2_fine_tuned_spoof_detection"
    
    # !!! IMPORTANT: SET THIS TO THE ROOT OF YOUR DATASET !!!
    # This path should contain 'real_data' and 'fake_data' folders.
    DATASET_ROOT = "./your_dataset_root" # Example: "/Users/youruser/my_project/my_audio_dataset"

    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    NUM_TRAIN_EPOCHS = 10
    WARMUP_RATIO = 0.1
    LOGGING_STEPS = 100
    SAVE_STEPS = 500
    VAL_TEST_SPLIT_SIZE = 0.15 # 15% for validation, 15% for test from total data.

    # --- 1. Load and Prepare Data ---
    logger.info("Loading and preparing dataset...")
    df = create_dataframe(DATASET_ROOT)

    # --- Ensure the feature extractor is initialized before mapping ---
    _ = get_feature_extractor() # Call once to ensure it's loaded

    # Use GroupShuffleSplit for splitting, ensuring groups stay together
    group_ids = df['group_id'].values
    unique_groups = np.unique(group_ids)

    if len(unique_groups) < 1 / VAL_TEST_SPLIT_SIZE: # Check if enough groups for splitting
         logger.warning(f"Not enough unique groups ({len(unique_groups)}) for specified split ratio ({VAL_TEST_SPLIT_SIZE*2}). "
                        "Consider reducing split size or acquiring more groups.")

    # First split: train and (val + test)
    # n_splits=1 as we only need one split iteration for train/val_test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_TEST_SPLIT_SIZE * 2, random_state=42)
    # The split method returns indices, so use .iloc for DataFrame slicing
    train_idx, val_test_idx = next(gss1.split(df, groups=group_ids))

    train_df = df.iloc[train_idx]
    val_test_df = df.iloc[val_test_idx]
    
    # Second split: val and test from val_test_df
    val_test_group_ids = val_test_df['group_id'].values
    # Test size is 0.5 to split the combined val_test set 50/50
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 
    val_idx, test_idx = next(gss2.split(val_test_df, groups=val_test_group_ids))

    val_df = val_test_df.iloc[val_idx]
    test_df = val_test_df.iloc[test_idx]

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Train samples: {len(train_df)} (Real: {len(train_df[train_df['label']==0])}, Fake: {len(train_df[train_df['label']==1])})")
    logger.info(f"Validation samples: {len(val_df)} (Real: {len(val_df[val_df['label']==0])}, Fake: {len(val_df[val_df['label']==1])})")
    logger.info(f"Test samples: {len(test_df)} (Real: {len(test_df[test_df['label']==0])}, Fake: {len(test_df[test_df['label']==1])})")

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Map the preprocessing function
    # Use num_proc for parallel processing if you have multiple CPU cores
    logger.info("Preprocessing datasets...")
    # .filter(lambda x: x is not None) removes samples that failed processing (e.g., corrupted audio)
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=["path", "group_id", "label"], num_proc=os.cpu_count()).filter(lambda x: x is not None)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=["path", "group_id", "label"], num_proc=os.cpu_count()).filter(lambda x: x is not None)
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=["path", "group_id", "label"], num_proc=os.cpu_count()).filter(lambda x: x is not None)

    # Initialize Data Collator
    data_collator = get_data_collator()

    # --- 2. Load Model ---
    logger.info(f"Loading model: {MODEL_NAME} for sequence classification...")
    # num_labels=2 for binary classification (real vs. fake)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        ignore_mismatched_sizes=True # Allow loading even if head changes size
    )

    # --- 3. Configure Training Arguments ---
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1, 
        evaluation_strategy="steps", 
        eval_steps=SAVE_STEPS, 
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        save_steps=SAVE_STEPS,
        save_total_limit=3, 
        dataloader_num_workers=os.cpu_count(), 
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        greater_is_better=True,
        # For M1 Mac (MPS support if available) or CPU
        report_to="tensorboard" # Or "none" if not using TensorBoard
    )

    # --- 4. Initialize Trainer ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=get_feature_extractor() # Feature extractor is used as tokenizer for audio
    )

    # --- 5. Train Model ---
    logger.info("Starting model training...")
    trainer.train()

    # --- 6. Evaluate on Test Set ---
    logger.info("Evaluating model on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test Set Evaluation Results: {test_results}")

    # --- 7. Save Final Model ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    get_feature_extractor().save_pretrained(final_model_path) # Save feature extractor with model
    logger.info(f"Final fine-tuned model saved to: {final_model_path}")

    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()