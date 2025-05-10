import os
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
import openl3
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# --- Path Definitions ---
current_file_path = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(current_file_path)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)

MODELS_DIR = os.path.join(ROOT_DIR,  "results") 
TEST_AUDIO_DIR = os.path.join(SCRIPTS_DIR, "samples") 
OUTPUT_DIR = os.path.join(SCRIPTS_DIR, "test_run_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration for Audio Processing ---
TARGET_SR = 16000 # OpenL3 models generally expect a standard sample rate, 48kHz for music/env, 16kHz for speech if specific models used
MIN_AUDIO_DURATION_SEC = 3.0

def pad_if_short(audio_data: np.ndarray, sample_rate: int, min_duration: float) -> np.ndarray:
    target_len = int(sample_rate * min_duration)
    if len(audio_data) < target_len:
        pad_width = target_len - len(audio_data)
        return np.pad(audio_data, (0, pad_width), mode='constant')
    return audio_data

def extract_embedding(audio_path):
    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1 and audio.shape[1] > 1: 
            audio = np.mean(audio, axis=1)
        
        # Resample if necessary. OpenL3's default models (music/env) often work with 48kHz.
        # If using a specific speech model variant, it might prefer 16kHz.
        # For now, let's stick to TARGET_SR for consistency, but be mindful of OpenL3's model expectations.
        if sr != TARGET_SR: 
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
            
        audio = pad_if_short(audio, sr, MIN_AUDIO_DURATION_SEC)
        
        # Reverted content_type to "music" to resolve the "Invalid content type" error.
        # OpenL3's get_audio_embedding might not directly support "speech" as a general content_type.
        # It might require a specific speech model to be loaded if that's the intention.
        # For now, "music" is a safer default that was previously working.
        emb, ts = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512, hop_size=0.5)
        
        if emb.shape[0] == 0:
            print(f"Warning: Empty embedding for {os.path.basename(audio_path)} after processing.")
            return None
        return np.mean(emb, axis=0)
    except Exception as e:
        print(f"Failed to extract embedding for {os.path.basename(audio_path)}: {e}")
        return None

def test_model(model_name, model, scaler, test_files, true_labels_dict=None):
    print(f"\n▶️ Testing {model_name}")
    
    model_results_dir = os.path.join(OUTPUT_DIR, model_name) 
    os.makedirs(model_results_dir, exist_ok=True)
    
    embeddings = []
    file_names_processed = []
    actual_labels_for_report = []
    
    for file_path in tqdm(test_files, desc=f"Processing files for {model_name}"):
        base_name = os.path.basename(file_path)
        embedding = extract_embedding(file_path)
        if embedding is not None:
            embeddings.append(embedding)
            file_names_processed.append(base_name)
            if true_labels_dict and base_name in true_labels_dict:
                actual_labels_for_report.append(true_labels_dict[base_name])
    
    if not embeddings:
        print(f"No valid embeddings extracted for {model_name}. Skipping.")
        return None
    
    X_test_np = np.array(embeddings)
    X_test_scaled = scaler.transform(X_test_np) 
    y_pred = model.predict(X_test_scaled)
    
    y_proba = []
    if hasattr(model, "predict_proba"):
        proba_all_classes = model.predict_proba(X_test_scaled)
        for i, p_val in enumerate(y_pred):
            if p_val == 1: 
                y_proba.append(proba_all_classes[i][1] * 100) 
            else: 
                y_proba.append(proba_all_classes[i][0] * 100) 
    else:
        y_proba = [np.nan] * len(y_pred)

    results_data = []
    for i in range(len(file_names_processed)):
        pred_label_text = "REAL" if y_pred[i] == 1 else "FAKE" 
        results_data.append({
            "file": file_names_processed[i],
            "prediction_numeric": y_pred[i],
            "prediction_label": pred_label_text,
            "confidence_percent": round(y_proba[i], 2) if not np.isnan(y_proba[i]) else "N/A"
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(model_results_dir, "predictions.csv"), index=False)
    
    print("\nPredictions:")
    for _, row in results_df.iterrows():
        print(f"{row['file']}: {row['prediction_label']} (Confidence: {row['confidence_percent']}%)")

    if true_labels_dict and actual_labels_for_report:
        if len(actual_labels_for_report) == len(y_pred):
            target_names_report = ['Fake (0)', 'Real (1)'] 
            report_str = classification_report(actual_labels_for_report, y_pred, target_names=target_names_report, zero_division=0)
            print(f"\nClassification Report for {model_name} (on provided labels):\n", report_str)
            with open(os.path.join(model_results_dir, "classification_report.txt"), "w") as f:
                f.write(report_str)
            
            cm = confusion_matrix(actual_labels_for_report, y_pred, labels=[0, 1]) 
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r', 
                        xticklabels=target_names_report, yticklabels=target_names_report)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(os.path.join(model_results_dir, "confusion_matrix.png"))
            plt.close()
        else:
            print("Warning: Mismatch between #true_labels and #predictions. Skipping report generation.")
    
    return results_df

def main():
    models_and_scalers = {}
    if not os.path.exists(MODELS_DIR): 
        print(f"ERROR: Trained models directory not found: {MODELS_DIR}")
        return

    print(f"Loading trained models from subdirectories in: {MODELS_DIR}")
    for model_folder_name in os.listdir(MODELS_DIR):
        potential_model_dir_path = os.path.join(MODELS_DIR, model_folder_name)
        if os.path.isdir(potential_model_dir_path): 
            bundle_path = os.path.join(potential_model_dir_path, "model_and_scaler.joblib")
            if os.path.isfile(bundle_path):
                try:
                    bundle = joblib.load(bundle_path)
                    if "model" in bundle and "scaler" in bundle:
                        models_and_scalers[model_folder_name] = bundle
                        print(f"Loaded model and scaler for: {model_folder_name}")
                    else:
                        print(f"Warning: Bundle incomplete (missing 'model' or 'scaler'): {bundle_path}")
                except Exception as e:
                    print(f"Error loading bundle {bundle_path}: {e}")

    if not models_and_scalers:
        print(f"No valid models (with scalers) found in subdirectories of {MODELS_DIR}")
        return
    
    test_files = []
    if not os.path.exists(TEST_AUDIO_DIR): 
        print(f"Test audio directory not found: {TEST_AUDIO_DIR}")
        return
    
    print(f"\nLoading test audio files from: {TEST_AUDIO_DIR}")
    for item_name in os.listdir(TEST_AUDIO_DIR):
        item_path = os.path.join(TEST_AUDIO_DIR, item_name)
        if os.path.isfile(item_path) and item_name.lower().endswith((".wav", ".mp3")):
            test_files.append(item_path)
            
    if not test_files:
        print(f"No test audio files (.wav, .mp3) found in {TEST_AUDIO_DIR}")
        return
    
    print(f"Found {len(test_files)} test audio files.")

    true_labels_for_test_files = {}
    for f_path in test_files:
        f_name = os.path.basename(f_path)
        if "real" in f_name.lower(): 
            true_labels_for_test_files[f_name] = 1 
        elif "fake" in f_name.lower():
            true_labels_for_test_files[f_name] = 0 

    all_model_predictions = {}
    for model_name, bundle in models_and_scalers.items():
        model_instance = bundle["model"]
        scaler_instance = bundle["scaler"]
        predictions_df = test_model(model_name, model_instance, scaler_instance, test_files, true_labels_dict=true_labels_for_test_files)
        if predictions_df is not None:
            all_model_predictions[model_name] = predictions_df
    
    if all_model_predictions:
        first_model_name = list(all_model_predictions.keys())[0]
        summary_df = all_model_predictions[first_model_name][['file']].copy()
        for model_name, predictions_df in all_model_predictions.items():
            temp_df = predictions_df[['file', 'prediction_label', 'confidence_percent']].rename(
                columns={
                    'prediction_label': f'{model_name}_pred_label',
                    'confidence_percent': f'{model_name}_confidence'
                }
            )
            summary_df = pd.merge(summary_df, temp_df, on='file', how='left')
        
        summary_df_path = os.path.join(OUTPUT_DIR, "all_models_test_predictions_summary.csv")
        summary_df.to_csv(summary_df_path, index=False)
        print(f"\nOverall test predictions summary saved to: {summary_df_path}")

    print(f"\n✅ Testing complete. All results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
