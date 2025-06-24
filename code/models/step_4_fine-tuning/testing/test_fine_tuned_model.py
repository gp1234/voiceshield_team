import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from pydub import AudioSegment
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)
import json
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results",'checkpoint-280')
SAMPLES_DIR = "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/samples/audios"


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary_predictions.csv")
METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")

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

# Calculate comprehensive metrics
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
roc_auc = roc_auc_score(true_labels, predictions)
mcc = matthews_corrcoef(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)
class_report = classification_report(true_labels, predictions)

# Save metrics to disk
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "matthews_corrcoef": mcc
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to", METRICS_PATH)

# Save confusion matrix and classification report as well
conf_matrix_df = pd.DataFrame(conf_matrix, index=["fake", "real"], columns=["fake", "real"])
conf_matrix_df.to_csv(CONFUSION_MATRIX_PATH)

with open(CLASSIFICATION_REPORT_PATH, "w") as f:
    f.write(class_report)

print("✅ Confusion matrix saved to", CONFUSION_MATRIX_PATH)
print("✅ Classification report saved to", CLASSIFICATION_REPORT_PATH)

# Print all metrics to console for quick review
print("\n📊 Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_df)
