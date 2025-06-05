import os
import random
import logging
import pandas as pd
import numpy as np
import torch
import transformers
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorCTCWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [np.asarray(feature["input_values"], dtype=np.float32).squeeze() for feature in features]
        labels = [feature["labels"] for feature in features]
        batch = self.feature_extractor(raw_speech=input_values, sampling_rate=16000, padding=self.padding, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def preprocess_function(example, feature_extractor):
    speech_array, label = example["speech"], example["label"]
    # Check for non-finite values in the speech_array *before* passing to feature_extractor
    if not np.isfinite(speech_array).all():
        logger.warning(f"Speech array contains non-finite values. Skipping example.")
        return None # Return None to filter out this example
        
    inputs = feature_extractor(speech_array, sampling_rate=16000)
    inputs["labels"] = label
    return inputs

def main():
    logger.info(f"Transformers version: {transformers.__version__}")

    audio_dir = "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/audio/augmented_balance"
    real_dir = os.path.join(audio_dir, "real_data")
    fake_dir = os.path.join(audio_dir, "fake_data")

    def get_wav_files(directory):
        all_files = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".wav"):
                    all_files.append(os.path.join(root, f))
        return all_files

    real_files = get_wav_files(real_dir)
    fake_files = get_wav_files(fake_dir)

    if not real_files and not fake_files:
        raise RuntimeError("No audio files found in real_data or fake_data directories.")

    data = pd.DataFrame({
        "path": real_files + fake_files,
        "label": [1] * len(real_files) + [0] * len(fake_files)
    })
    data = data.sample(frac=1).reset_index(drop=True)

    logger.info(f"Real samples: {len(real_files)}, Fake samples: {len(fake_files)}")

    def load_audio(path):
        import librosa
        audio, _ = librosa.load(path, sr=16000)
        return audio

    data["speech"] = data["path"].map(load_audio)

    train_df, temp_df = train_test_split(data, test_size=0.45, stratify=data["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    logger.info("Feature extractor loaded.")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Filter out samples where preprocessing returns None (e.g., due to non-finite audio or errors)
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, feature_extractor), remove_columns=["speech", "path"]).filter(lambda x: x is not None)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, feature_extractor), remove_columns=["speech", "path"]).filter(lambda x: x is not None)


    model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=2)
    logger.info("Model loaded.")

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=10,
        save_total_limit=1,
        num_train_epochs=3,
        logging_dir="./logs",
        learning_rate=1e-4,
        max_grad_norm=1.0 # <--- ADDED: Prevents exploding gradients
    )

    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__":
    main()