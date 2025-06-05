import pandas as pd
import numpy as np
from TTS.api import TTS
import os
import scipy.io.wavfile

# Initialize XTTS-v2 model (CPU only)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Load transcripts
df = pd.read_csv("transcripts.csv", encoding="utf-8")

# Create output directory
os.makedirs("xtts_outputs", exist_ok=True)

# Process each transcript
for idx, row in df.iterrows():
    filename = os.path.basename(row["id"]).replace(".mp3", ".wav")
    text = row["text"]

    print(f"🔊 Synthesizing: {filename} -> {text}")
    try:
        # Generate audio (default speaker, English)
        output_path = os.path.join("xtts_outputs", filename)
        tts.tts_to_file(text=text, language="en", file_path=output_path)

        # Ensure compatibility with scipy (normalize audio)
        sampling_rate, audio_array = scipy.io.wavfile.read(output_path)
        audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_array)
    except Exception as e:
        print(f"Error generating audio for {filename}: {e}")
        continue

print("✅ All audios generated.")