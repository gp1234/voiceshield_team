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

    def plot_pca(self, X, y, output_dir):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Real', alpha=0.6)
        plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Fake', alpha=0.6)
        plt.title('PCA Visualization of Features')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
        plt.close()

    def plot_duration_distribution(self, durations, labels, output_dir):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=labels, y=durations)
        plt.title('Duration Distribution by Class')
        plt.xlabel('Class (0: Real, 1: Fake)')
        plt.ylabel('Duration (seconds)')
        plt.savefig(os.path.join(output_dir, 'duration_distribution.png'))
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        for model_name, model in self.models.items():
            output_dir = f"{self.base_output_dir}_{model_name.lower().replace(' ', '_')}"
            os.makedirs(output_dir, exist_ok=True)
            
            metrics, cm = self.evaluate_model(model, X_train, X_test, y_train, y_test)
            results[model_name] = metrics
            
            self.plot_confusion_matrix(cm, model_name, output_dir)
            self.generate_report(metrics, model_name, output_dir)
        
        self.plot_pca(X, y, self.base_output_dir)
        self.plot_duration_distribution(durations, y, self.base_output_dir)
        
        return results 