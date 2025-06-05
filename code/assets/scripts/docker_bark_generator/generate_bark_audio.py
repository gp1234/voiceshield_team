import pandas as pd
import numpy as np
from transformers import AutoProcessor, BarkModel
import torch
import scipy.io.wavfile
import os

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

df = pd.read_csv("transcripts.csv", encoding="utf-8")

os.makedirs("bark_outputs", exist_ok=True)

for idx, row in df.iterrows():
    filename = os.path.basename(row["id"]).replace(".mp3", ".wav")
    text = row["text"]

    print(f"🔊 Synthesizing: {filename} -> {text}")
    try:
        inputs = processor(text, return_tensors="pt")  # Remove padding=True
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        
        audio_array = audio_array[0].cpu().numpy()
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
        sampling_rate = model.config.sample_rate

        output_path = os.path.join("bark_outputs", filename)
        scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_array)
    except Exception as e:
        print(f"Error generating audio for {filename}: {e}")
        continue

print("✅ All audios generated.")