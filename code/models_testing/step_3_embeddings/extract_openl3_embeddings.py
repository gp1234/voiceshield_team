import os
import openl3
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

AUDIO_ROOT = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')
output_path = os.path.join(os.getcwd(), 'openl3_features')
os.makedirs(output_path, exist_ok=True)
OUTPUT_CSV = os.path.join(output_path, 'openl3_features.csv')

def extract_embedding(audio_path):
    try:
        audio, sr = sf.read(audio_path)
        emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
        return np.mean(emb, axis=0)
    except Exception as e:
        print(f"Failed on {audio_path}: {e}")
        return None

def label_and_group(file_path):
    parts = file_path.split(os.sep)
    label = 0 if parts[-2] == "real" else 1
    source = parts[-2]
    group = os.path.splitext(parts[-1])[0].replace(".mp3", "").replace(".wav", "")
    return label, group, source

def main():
    rows = []

    for root, _, files in os.walk(AUDIO_ROOT):
        for file in tqdm(files):
            if file.endswith(".wav") or file.endswith(".mp3"):
                full_path = os.path.join(root, file)
                embedding = extract_embedding(full_path)
                if embedding is not None:
                    label, group, source = label_and_group(full_path)
                    rows.append({
                        "file": full_path,
                        "label": label,
                        "features": ",".join(map(str, embedding)),
                        "group": group,
                        "source": source
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(df)} samples to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()