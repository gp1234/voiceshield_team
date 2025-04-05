import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib



EMBEDDING_CSV = "openl3_features/openl3_features.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(EMBEDDING_CSV)
X = np.vstack(df['features'].str.split(',').apply(lambda x: list(map(float, x))))
y = df['label'].values
groups = df['group'].values

splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel='rbf', probability=True, random_state=42),
    "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    "logistic": LogisticRegression(max_iter=1000, random_state=42)
}

for name, model in models.items():
    print(f"\n▶️ Training {name}")
    

    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    # Save report
    with open(os.path.join(model_dir, "report.txt"), "w") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")
    
    # Save model
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

print(f"\n✅ Results saved to: {OUTPUT_DIR}")