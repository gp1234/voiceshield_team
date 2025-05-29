import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from pydub import AudioSegment

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results",'checkpoint-280')
SAMPLES_DIR = os.path.join(os.path.dirname(__file__),  "samples")
SUMMARY_PATH = os.path.join(SAMPLES_DIR, "summary_predictions.csv")

model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model.eval()

predictions = []
files = []
true_labels = []
corrects = []

def convert_to_wav(mp3_path):
    audio = AudioSegment.from_file(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".temp.wav")
    audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")
    return wav_path

for fname in os.listdir(SAMPLES_DIR):
    if not (fname.endswith(".wav") or fname.endswith(".mp3")):
        continue

    file_path = os.path.join(SAMPLES_DIR, fname)
    is_temp = False

    if fname.endswith(".mp3"):
        file_path = convert_to_wav(file_path)
        is_temp = True

    waveform, sr = torchaudio.load(file_path)

    if sr != 16000:
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding="longest"
    ).input_values

    with torch.no_grad():
        logits = model(inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()

    true_label = 1 if "real" in fname.lower() else 0
    correct = int(prediction == true_label)

    predictions.append(prediction)
    true_labels.append(true_label)
    corrects.append(correct)
    files.append(fname)

    if is_temp:
        os.remove(file_path)

summary_df = pd.DataFrame({
    "file": files,
    "prediction": predictions,
    "true_label": true_labels,
    "correct": corrects
})

accuracy = summary_df["correct"].mean()
summary_df.to_csv(SUMMARY_PATH, index=False)
print("✅ Predictions saved to", SUMMARY_PATH)
print(f"🎯 Accuracy: {accuracy:.2%}")
