import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

class ModelEvaluator:
    def __init__(self, models, output_dir='evaluation_results'):
        self.models = models
        self.base_output_dir = output_dir

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)

    def get_duration(self, audio_path):
        y, sr = librosa.load(audio_path)
        return librosa.get_duration(y=y, sr=sr)

    def prepare_data(self, data_dir):
        features = []
        labels = []
        durations = []
        
        # Check if the directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist.")
            return np.array([]), np.array([]), np.array([])
        
        # Get all WAV files in the directory
        audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        
        if not audio_files:
            print(f"Error: No WAV files found in {data_dir}")
            return np.array([]), np.array([]), np.array([])
        
        print(f"Found {len(audio_files)} WAV files in {data_dir}")
        
        # Process all files in the directory
        for audio_file in audio_files:
            audio_path = os.path.join(data_dir, audio_file)
            try:
                features.append(self.extract_features(audio_path))
                # For now, we'll label all files as "real" since we're only testing with real files
                labels.append(0)  # 0 for real
                durations.append(self.get_duration(audio_path))
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
        
        if not features:
            print("Error: No features could be extracted from the audio files.")
            return np.array([]), np.array([]), np.array([])
        
        return np.array(features), np.array(labels), np.array(durations)

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return metrics, cm

    def plot_confusion_matrix(self, cm, model_name, output_dir):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()

    def generate_report(self, metrics_dict, model_name, output_dir):
        report = f"Model Evaluation Report - {model_name}\n"
        report += "=" * 50 + "\n\n"
        
        report += "Classification Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in metrics_dict.items():
            report += f"{metric}: {value}\n"
        
        with open(os.path.join(output_dir, f'report_{model_name}.txt'), 'w') as f:
            f.write(report)

    def evaluate(self, data_dir):
        X, y, durations = self.prepare_data(data_dir)
        
        # Check if we have data to work with
        if len(X) == 0:
            print("No data available for evaluation. Exiting.")
            return {}
        
        # Since we only have one class (real), we need to create a dummy class for testing
        # This is just for demonstration purposes
        if len(np.unique(y)) == 1:
            print("Warning: Only one class found in the data. Creating a dummy class for testing.")
            # Create a dummy class by adding some noise to the features
            X_dummy = X + np.random.normal(0, 0.1, X.shape)
            y_dummy = np.ones(len(y))  # Label as fake
            
            # Combine the original and dummy data
            X = np.vstack((X, X_dummy))
            y = np.concatenate((y, y_dummy))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        for model_name, model in self.models.items():
            output_dir = f"{self.base_output_dir}"
            os.makedirs(output_dir, exist_ok=True)
            
            metrics, cm = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            results[model_name] = metrics
            
            self.plot_confusion_matrix(cm, model_name, output_dir)
            self.generate_report(metrics, model_name, output_dir)
        
        
        return results 