import os
import pandas as pd
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

import torchaudio

waveform, sr = torchaudio.load(file_path)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)


# Set paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "checkpoint-280")

SAMPLES_DIR = "./samples"
OUTPUT_DIR = "./test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model & processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Predict
results = []
for file in os.listdir(SAMPLES_DIR):
    if not file.lower().endswith((".wav", ".mp3")):
        continue

    file_path = os.path.join(SAMPLES_DIR, file)
    try:
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        label = "REAL" if pred == 1 else "FAKE"
        results.append({"file": file, "prediction": pred, "label": label, "confidence": f"{confidence:.2f}"})
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

# Save individual CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

# Update summary
summary_path = "./test_run_results/all_models_test_predictions_summary.csv"
summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame({"file": df["file"]})
summary_df["fine_tuned_prediction"] = df["prediction"]
summary_df.to_csv(summary_path, index=False)

print("✅ Done. Predictions saved.")
