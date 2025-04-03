import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ModelEvaluator:
    def __init__(self, models, output_dir='evaluation_results', models_dir=None):
        self.models = models
        self.output_dir = output_dir
        self.models_dir = models_dir if models_dir else os.path.join(output_dir, 'trained_models')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        return np.concatenate([mfccs_mean, mfccs_std])

    def get_duration(self, audio_path):
        y, sr = librosa.load(audio_path)
        return librosa.get_duration(y=y, sr=sr)

    def prepare_data(self, data_dir):
        features = []
        labels = []
        durations = []
        
        for label in ['real', 'fake']:
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                continue
                
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(label_dir, audio_file)
                    features.append(self.extract_features(audio_path))
                    labels.append(1 if label == 'fake' else 0)
                    durations.append(self.get_duration(audio_path))
        
        return np.array(features), np.array(labels), np.array(durations)

    def save_model(self, model, model_name):
        """Save a trained model using joblib"""
        model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved trained model to: {model_path}")
        return model_path

    def load_model(self, model_name):
        """Load a trained model using joblib"""
        model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from: {model_path}")
            return joblib.load(model_path)
        return None

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        # Try to load pre-trained model first
        loaded_model = self.load_model(model_name)
        if loaded_model is not None:
            model = loaded_model
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Save the trained model
        self.save_model(model, model_name)
        
        # Make predictions and get metrics
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return metrics, cm

    def plot_confusion_matrix(self, cm, model_name, model_dir):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
        plt.close()

    def generate_report(self, metrics_dict, model_name, model_dir):
        report = f"Model Evaluation Report - {model_name}\n"
        report += "=" * 50 + "\n\n"
        
        report += "Classification Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in metrics_dict.items():
            if isinstance(value, dict):
                report += f"\n{metric}:\n"
                for k, v in value.items():
                    report += f"  {k}: {v}\n"
            else:
                report += f"{metric}: {value}\n"
        
        with open(os.path.join(model_dir, 'report.txt'), 'w') as f:
            f.write(report)

    def evaluate(self, data_dir):
        X, y, durations = self.prepare_data(data_dir)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            model_dir = os.path.join(self.output_dir, model_name.lower().replace(' ', '_'))
            os.makedirs(model_dir, exist_ok=True)
            
            metrics, cm = self.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
            results[model_name] = metrics
            
            self.plot_confusion_matrix(cm, model_name, model_dir)
            self.generate_report(metrics, model_name, model_dir)
        
        return results 