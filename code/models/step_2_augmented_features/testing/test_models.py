import os
import numpy as np
import pandas as pd
import joblib
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "results")
TEST_AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
OUTPUT_DIR = os.path.join(os.getcwd(), "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ModelWithScaler:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def load_models():
    models = {}
    model_dirs = ['logistic', 'mlp', 'random_forest', 'svm']
    
    for model_dir in model_dirs:
        model_path = os.path.join(MODELS_DIR, model_dir, f"{model_dir}.joblib")
        if os.path.exists(model_path):
            try:
                # Configure joblib to use our local ModelWithScaler class
                with open(model_path, 'rb') as f:
                    model = joblib.load(f)
                models[model_dir] = model
                print(f"Loaded model: {model_dir}")
            except Exception as e:
                print(f"Error loading model {model_dir}: {e}")
    
    return models

def test_samples(models):
    sample_files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.endswith(('.wav', '.mp3'))]
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTesting model: {model_name}")
        model_results = []
        
        for sample_file in tqdm(sample_files, desc=f"Processing samples with {model_name}"):
            sample_path = os.path.join(TEST_AUDIO_DIR, sample_file)
            
            try:
                features = extract_features(sample_path)
                features = features.reshape(1, -1)  # Reshape for single prediction
                prediction = model.predict(features)[0]
                prediction_label = "fake" if prediction == 1 else "real"
                
                model_results.append({
                    "sample": sample_file,
                    "prediction": prediction,
                    "prediction_label": prediction_label
                })
                
            except Exception as e:
                print(f"Error processing {sample_file} with {model_name}: {e}")
        
        results[model_name] = pd.DataFrame(model_results)
        
        # Save results to CSV
        results[model_name].to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_results.csv"), index=False)
        
        # Generate summary report
        with open(os.path.join(OUTPUT_DIR, f"{model_name}_summary.txt"), 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Sample Predictions:\n")
            f.write("-" * 20 + "\n")
            for _, row in results[model_name].iterrows():
                f.write(f"Sample: {row['sample']}\n")
                f.write(f"Prediction: {row['prediction_label']}\n")
                f.write("-" * 10 + "\n")
            
            # Count predictions
            prediction_counts = results[model_name]['prediction_label'].value_counts()
            f.write("\nPrediction Summary:\n")
            f.write("-" * 20 + "\n")
            for label, count in prediction_counts.items():
                f.write(f"{label}: {count} samples\n")
    
    return results

def main():
    # Load models
    models = load_models()
    
    if not models:
        print("No models found to test.")
        return
    
    # Test samples with each model
    results = test_samples(models)
    
    print(f"\nTesting complete! Results saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
