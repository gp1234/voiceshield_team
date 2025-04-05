import os
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
import openl3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(os.getcwd(), "results")
TEST_AUDIO_DIR = os.path.join(ROOT_DIR, "assets", "audio", "test_audio")
OUTPUT_DIR = os.path.join(os.getcwd(), "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_embedding(audio_path):
    """Extract OpenL3 embedding from audio file."""
    try:
        audio, sr = sf.read(audio_path)
        emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
        return np.mean(emb, axis=0)
    except Exception as e:
        print(f"Failed on {audio_path}: {e}")
        return None

def test_model(model_name, model, test_files, true_labels=None):
    """Test a single model on the test files."""
    print(f"\n▶️ Testing {model_name}")
    
    # Create model-specific output directory
    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Extract embeddings and make predictions
    predictions = []
    embeddings = []
    file_names = []
    
    for file_path in tqdm(test_files, desc=f"Processing files for {model_name}"):
        embedding = extract_embedding(file_path)
        if embedding is not None:
            embeddings.append(embedding)
            file_names.append(os.path.basename(file_path))
    
    if not embeddings:
        print(f"No valid embeddings extracted for {model_name}")
        return
    
    # Convert to numpy array
    X_test = np.array(embeddings)
    
    # Make predictions
    y_pred = model.predict(X_test)
    predictions = y_pred.tolist()
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "file": file_names,
        "prediction": predictions
    })
    
    # Save results
    results_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)
    
    # If true labels are provided, calculate metrics
    if true_labels is not None:
        y_test = true_labels
        
        # Calculate and save metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save report
        with open(os.path.join(model_dir, "report.txt"), "w") as f:
            for k, v in report.items():
                f.write(f"{k}: {v}\n")
        
        # Create and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
        plt.close()
        
        print(f"Accuracy: {report['accuracy']:.4f}")
    
    return results_df

def main():
    # Load all trained models
    models = {}
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name, "model.joblib")
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    
    if not models:
        print("No trained models found!")
        return
    
    # Get test files
    test_files = []
    true_labels = []
    
    # Check if test directory exists
    if not os.path.exists(TEST_AUDIO_DIR):
        print(f"Test directory not found: {TEST_AUDIO_DIR}")
        print("Please create the directory and add test audio files.")
        return
    
    # Process test files
    for root, _, files in os.walk(TEST_AUDIO_DIR):
        for file in files:
            if file.endswith(".wav") or file.endswith(".mp3"):
                full_path = os.path.join(root, file)
                test_files.append(full_path)
                
                # Extract label from path (assuming same structure as training data)
                parts = full_path.split(os.sep)
                label = 0 if parts[-2] == "real" else 1
                true_labels.append(label)
    
    if not test_files:
        print("No test audio files found!")
        return
    
    print(f"Found {len(test_files)} test files")
    
    # Test each model
    all_results = {}
    for model_name, model in models.items():
        results = test_model(model_name, model, test_files, true_labels)
        if results is not None:
            all_results[model_name] = results
    
    # Create a summary of all models' predictions
    if all_results:
        summary_df = pd.DataFrame()
        for model_name, results in all_results.items():
            summary_df[f"{model_name}_prediction"] = results["prediction"]
        
        summary_df["file"] = all_results[list(all_results.keys())[0]]["file"]
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "all_predictions_summary.csv"), index=False)
        
        print(f"\n✅ Test results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 