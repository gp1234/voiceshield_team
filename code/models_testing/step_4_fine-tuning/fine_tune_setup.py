# Step 1: Fine-tuning Configuration & Dataset Preparation

import os
import re
import pandas as pd
from datasets import Dataset, Audio
from sklearn.model_selection import GroupShuffleSplit

# Path setup
current_file_path = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(current_file_path)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
AUDIO_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')
AUGMENTED_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'augmented_audio')
FINE_TUNE_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'fine_tune_set')
os.makedirs(FINE_TUNE_DIR, exist_ok=True)

# 1. Collect audio files: real + augmented as real, fakes (from fake_1, fake_2, fake_3)
all_paths = []
labels = []
speakers = []

for label_dir, is_fake in [("real", 1), ("fake_1", 0), ("fake_2", 0), ("fake_3", 0)]:
    base_dir = AUGMENTED_DIR if label_dir == "real" else AUDIO_DIR
    folder = os.path.join(base_dir, label_dir)
    for f in os.listdir(folder):
        if f.endswith(".wav"):
            all_paths.append(os.path.join(folder, f))
            labels.append(is_fake)
            match = re.search(r"common_voice_en_(\d+)", f)
            speaker_id = match.group(1) if match else f.split(".")[0]
            speakers.append(speaker_id)

# 2. Build DataFrame for HuggingFace Dataset
meta_df = pd.DataFrame({
    "audio": all_paths,
    "label": labels,
    "group": speakers
})

# Debug: check speaker distribution
print("\n[Debug] Unique groups:", meta_df['group'].nunique())
print(meta_df['group'].value_counts())

# 3. Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(meta_df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 4. Group-aware train/test split
if meta_df['group'].nunique() < 2:
    raise ValueError("Need at least two unique speaker groups for GroupShuffleSplit.")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(meta_df, groups=meta_df["group"]))
dataset_train = dataset.select(train_idx)
dataset_test = dataset.select(test_idx)

# Save locally for reference
meta_df.iloc[train_idx].to_csv(os.path.join(FINE_TUNE_DIR, "train_metadata.csv"), index=False)
meta_df.iloc[test_idx].to_csv(os.path.join(FINE_TUNE_DIR, "test_metadata.csv"), index=False)

print(f"✅ Fine-tuning data ready: {len(dataset_train)} train / {len(dataset_test)} test")
