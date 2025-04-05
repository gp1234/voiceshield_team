import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ModelEvaluator:
    def __init__(self, models, labeled_data, output_dir='evaluation_results'):
        self.models = models
        self.labeled_data = labeled_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name, scaler=None):
        # Scale the data if scaler is provided
        if scaler is not None:
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
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

    def evaluate(self):
        df = self.labeled_data.copy()
        df['label_num'] = df['label'] if df['label'].dtype == int else df['label'].map({'real': 0, 'fake': 1})
        X = np.vstack(df['features'].values)
        y = df['label_num'].values
        groups = df['group_id'].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\nTrain labels:\n{np.unique(y_train, return_counts=True)}")
        print(f"Test labels:\n{np.unique(y_test, return_counts=True)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            model_dir = os.path.join(self.output_dir, model_name.lower().replace(' ', '_'))
            os.makedirs(model_dir, exist_ok=True)

            metrics, cm = self.evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, model_name, scaler)
            results[model_name] = metrics

            self.plot_confusion_matrix(cm, model_name, model_dir)
            self.generate_report(metrics, model_name, model_dir)

        return results
