import os
import openl3
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_embedding(file_path, sr=16000):
    try:
        audio, orig_sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr, sr)
        embedding, _ = openl3.get_audio_embedding(
            audio, sr, content_type="speech", embedding_size=512
        )
        return np.mean(embedding, axis=0)  
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def build_dataset(root_dir):
    data = []
    for label_dir, label in [('real', 0), ('fake', 1)]:
        path = os.path.join(root_dir, label_dir)
        if not os.path.exists(path):
            continue

        for file in tqdm(os.listdir(path), desc=label_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(path, file)
                emb = extract_embedding(file_path)
                if emb is not None:
                    data.append({
                        'file': file_path,
                        'label': label,
                        'features': emb
                    })

    return pd.DataFrame(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract OpenL3 embeddings for real and fake voice samples")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with 'real/' and 'fake/' subdirs")
    parser.add_argument("--output", type=str, default="openl3_dataset.csv", help="Output CSV path")

    args = parser.parse_args()

    print(f"📂 Loading audio from {args.input_dir}")
    df = build_dataset(args.input_dir)

    print(f"💾 Saving to {args.output}")
    df['features'] = df['features'].apply(lambda x: ','.join(map(str, x)))
    df.to_csv(args.output, index=False)

    print("✅ Done.")