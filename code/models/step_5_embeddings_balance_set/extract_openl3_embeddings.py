import os
import openl3
import soundfile as sf
import numpy as np
from pathlib import Path

# ─────────── CONFIGURATION ───────────

# Base folder where your augmented WAV files live:
#   AUDIO_ROOT/
#     real_data/
#       group_1/
#       group_2/
#     fake_data/
#       group_1/
#       group_2/
AUDIO_ROOT = Path("/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/audio/augmented_balance")

# Where embeddings will be saved (mirroring real_data/ and fake_data/ folders):
EMBEDDING_ROOT = Path("openl3_embeddings")

# Choose content_type="env" (robust to speech/noise) or "music" if you prefer.
CONTENT_TYPE = "env"
EMBEDDING_SIZE = 512  # 512‐dimensional OpenL3 embeddings

# ─────────── FUNCTIONS ───────────

def extract_embedding(wav_path: Path):
    """
    Load a WAV, run openl3.get_audio_embedding, then return the time‐averaged 1×512 vector.
    """
    audio, sr = sf.read(str(wav_path))
    # If stereo, pick first channel:
    if audio.ndim > 1:
        audio = audio[:, 0]
    emb, _ = openl3.get_audio_embedding(
        audio, sr, content_type=CONTENT_TYPE, embedding_size=EMBEDDING_SIZE
    )
    # Average over time frames → (512,)
    return np.mean(emb, axis=0)


def process_folder(input_dir: Path, output_dir: Path):
    """
    Walk input_dir recursively, find all .wav files, extract embeddings, and save .npy in output_dir.
    """
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(".wav"):
                continue

            wav_path = Path(root) / filename
            # Build a parallel path under output_dir:
            relative_subpath = wav_path.relative_to(input_dir)  # e.g. group_1/foo_orig.wav
            np_out_path = output_dir / relative_subpath.with_suffix(".npy")
            np_out_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract and save:
            emb_vec = extract_embedding(wav_path)
            np.save(str(np_out_path), emb_vec)


def mirror_structure_and_extract():
    """
    For each of 'real_data' and 'fake_data', for each group_X folder inside,
    call process_folder() to extract embeddings into EMBEDDING_ROOT.
    """
    for label in ["real_data", "fake_data"]:
        input_base = AUDIO_ROOT / label
        output_base = EMBEDDING_ROOT / label

        if not input_base.exists():
            print(f"⚠️  Skipping '{label}' — folder not found at {input_base}")
            continue

        # Ensure top‐level output folder exists:
        (output_base).mkdir(parents=True, exist_ok=True)

        # Iterate over each group subfolder:
        for group_folder in input_base.iterdir():
            if not group_folder.is_dir():
                continue  # skip any stray files
            out_group_folder = output_base / group_folder.name
            out_group_folder.mkdir(parents=True, exist_ok=True)

            print(f"🔄 Processing {label}/{group_folder.name} …")
            process_folder(group_folder, out_group_folder)
            print(f"✅ Saved embeddings to {out_group_folder}\n")


if __name__ == "__main__":
    mirror_structure_and_extract()
    print("🏁 All done. Embeddings saved under 'openl3_embeddings/'.")
