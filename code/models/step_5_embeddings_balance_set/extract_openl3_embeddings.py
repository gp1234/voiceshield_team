import os
import openl3
import soundfile as sf
import numpy as np
from pathlib import Path

INPUT_DIR = Path("/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/audio/augmented_balance")
OUTPUT_DIR = Path("openl3_embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_embedding(audio_path):
    audio, sr = sf.read(audio_path)
    audio = audio[:, 0] if len(audio.shape) > 1 else audio
    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="env", embedding_size=512)
    return emb.mean(axis=0)

def process_folder(input_path, output_path):
    for root, _, files in os.walk(input_path):
        for f in files:
            if f.endswith(".wav"):
                full_path = Path(root) / f
                emb = extract_embedding(full_path)
                out_file = output_path / (full_path.stem + ".npy")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_file, emb)

for group_folder in (INPUT_DIR / "fake_data").iterdir():
    if group_folder.is_dir():
        process_folder(group_folder, OUTPUT_DIR / "fake_data" / group_folder.name)

process_folder(INPUT_DIR / "real_data", OUTPUT_DIR / "real_data")
