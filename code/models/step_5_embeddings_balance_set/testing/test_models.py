import os
import re
import csv
import joblib
import openl3
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Paths and settings
# ──────────────────────────────────────────────────────────────────────────────

# Folder containing your saved models (each model has "<ModelName>_bundle.joblib")
MODELS_DIR = Path("/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/models/step_5_embeddings_balance_set/training_results")

# Folder containing new audio samples for testing (files end with "_real.mp3" or "_fake.mp3")
SAMPLE_DIR = Path(
    "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/audio/samples/audios"
)

# Where to store test results
OUTPUT_DIR = Path("test_run_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio and embedding settings
EMBEDDING_SAMPLE_RATE = 16000
EMBEDDING_CONTENT_TYPE = "env"   # must match whatever was used during training
EMBEDDING_EMBEDDING_SIZE = 512      # training used size=512

# ──────────────────────────────────────────────────────────────────────────────
# 2) Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_embedding_mean(audio_path: Path):
    """
    Load an audio file (mp3, wav, etc.), resample to EMBEDDING_SAMPLE_RATE,
    compute the OpenL3 embedding (content_type="speech", embedding_size=512),
    then return the mean embedding vector over time (shape=(512,)).
    """
    # Load with librosa
    audio, sr = librosa.load(str(audio_path), sr=EMBEDDING_SAMPLE_RATE, mono=True)
    # Get embedding for each frame: shape (n_frames, embedding_dim), and timestamps
    emb, ts = openl3.get_audio_embedding(
        audio,
        sr,
        content_type=EMBEDDING_CONTENT_TYPE,
        embedding_size=EMBEDDING_EMBEDDING_SIZE,
    )
    # Take mean over frames → single vector
    return np.mean(emb, axis=0)


def load_model_bundle(bundle_path: Path):
    """
    Load a joblib “bundle” dictionary containing {"model": ..., "scaler": ...}.
    Returns (model, scaler).
    """
    bundle = joblib.load(str(bundle_path))
    model = bundle.get("model", None)
    scaler = bundle.get("scaler", None)
    return model, scaler


def infer_true_label_from_filename(filename: str):
    """
    Given a filename ending in "_real.mp3" or "_fake.mp3",
    return 1 for real, 0 for fake. If neither is matched, return None.
    """
    if filename.lower().endswith("_real.mp3") or filename.lower().endswith("_real.wav"):
        return 1
    if filename.lower().endswith("_fake.mp3") or filename.lower().endswith("_fake.wav"):
        return 0
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 3) Gather all sample files and their true labels
# ──────────────────────────────────────────────────────────────────────────────

sample_files = []
for f in sorted(SAMPLE_DIR.iterdir()):
    if not f.is_file():
        continue
    true_label = infer_true_label_from_filename(f.name)
    if true_label is None:
        continue
    sample_files.append((f, true_label))

if not sample_files:
    print("‼ No valid sample files found under:", SAMPLE_DIR)
    exit(1)

print(f"▶ Found {len(sample_files)} sample files for testing.")


# ──────────────────────────────────────────────────────────────────────────────
# 4) Precompute embeddings for all samples (store in-memory)
# ──────────────────────────────────────────────────────────────────────────────

print("▶ Extracting embeddings for all samples…")
sample_embeddings = {}  # maps Path → np.ndarray of shape (512,)
for audio_path, true_label in sample_files:
    try:
        emb_vec = extract_embedding_mean(audio_path)
    except Exception as e:
        print(f"‼ Failed to extract embedding from {audio_path.name}: {e}")
        continue
    sample_embeddings[audio_path] = (emb_vec, true_label)

if not sample_embeddings:
    print("‼ No embeddings were successfully extracted.")
    exit(1)

print(f"▶ Extracted embeddings for {len(sample_embeddings)} files.")


# ──────────────────────────────────────────────────────────────────────────────
# 5) Load each model and run inference on the sample embeddings
# ──────────────────────────────────────────────────────────────────────────────

# Prepare a summary CSV with header: filename,true_label,<Model1>,<Model2>,...
model_names = []
bundles = {}  # model_name -> (model, scaler)
for model_folder in sorted(MODELS_DIR.iterdir()):
    if not model_folder.is_dir():
        continue
    # Expect a file named "<ModelName>_bundle.joblib" inside model_folder
    bundle_file = next(model_folder.glob("*_bundle.joblib"), None)
    if bundle_file is None:
        continue
    model_name = model_folder.name  # e.g. "LogisticRegression"
    model, scaler = load_model_bundle(bundle_file)
    if model is None:
        continue
    model_names.append(model_name)
    bundles[model_name] = (model, scaler)

if not bundles:
    print("‼ No model bundles found under:", MODELS_DIR)
    exit(1)

print(f"▶ Loaded {len(bundles)} models: {model_names}")

# Create output subfolders under OUTPUT_DIR for each model
for m in model_names:
    (OUTPUT_DIR / m).mkdir(parents=True, exist_ok=True)

# Prepare summary CSV file
summary_csv_path = OUTPUT_DIR / "all_models_test_predictions_summary.csv"
with open(summary_csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["filename", "true_label"] + model_names
    writer.writerow(header)

    # We will store per-model predictions in a dict of lists
    per_model_preds = {m: [] for m in model_names}
    filenames = []
    true_labels = []

    # For each sample, run through all models
    for audio_path, true_label in sample_files:
        if audio_path not in sample_embeddings:
            continue
        filenames.append(audio_path.name)
        true_labels.append(true_label)

        emb_vec, _ = sample_embeddings[audio_path]
        row = [audio_path.name, true_label]

        for m in model_names:
            model, scaler = bundles[m]
            # Scale if needed
            if scaler is not None:
                emb_scaled = scaler.transform(emb_vec.reshape(1, -1))
            else:
                emb_scaled = emb_vec.reshape(1, -1)

            # Predict probability or label
            try:
                pred_prob = model.predict_proba(emb_scaled)[0, 1]
                pred_label = int(pred_prob >= 0.5)
            except AttributeError:
                # Some models (e.g. if user replaced SVM without probability) might only have predict()
                pred_label = int(model.predict(emb_scaled)[0])

            row.append(pred_label)
            per_model_preds[m].append(pred_label)

        writer.writerow(row)

print(f"▶ Wrote summary CSV: {summary_csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6) For each model: compute metrics and save report + confusion matrix + ROC
# ──────────────────────────────────────────────────────────────────────────────

for m in model_names:
    preds = per_model_preds[m]
    y_true = true_labels
    y_pred = preds

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Create model-specific folder under OUTPUT_DIR
    model_out_dir = OUTPUT_DIR / m

    # 1) Save report.txt
    report_path = model_out_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # 2) Save confusion_matrix.png
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{m} - Confusion Matrix")
    plt.tight_layout()
    cm_path = model_out_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 3) Save ROC curve (if probabilities available)
    # Recompute probabilities if possible
    model, scaler = bundles[m]
    prob_list = []
    for audio_path, _ in sample_files:
        if audio_path not in sample_embeddings:
            continue
        emb_vec, _ = sample_embeddings[audio_path]
        if scaler is not None:
            emb_scaled = scaler.transform(emb_vec.reshape(1, -1))
        else:
            emb_scaled = emb_vec.reshape(1, -1)
        try:
            prob = model.predict_proba(emb_scaled)[0, 1]
        except AttributeError:
            prob = None
        prob_list.append(prob)

    # Check if any probability is not None
    if any([p is not None for p in prob_list]):
        # Filter out None (some rare models might not have predict_proba)
        y_prob = np.array([p if p is not None else 0.0 for p in prob_list])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{m} - ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = model_out_dir / "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

    print(f"▶ Completed results for model: {m}")

print("✅ Testing complete. All results in:", OUTPUT_DIR)
