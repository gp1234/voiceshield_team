import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

EMBEDDING_DIR = Path("openl3_embeddings")
RESULTS_DIR = Path("training_results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_embeddings(label_dir, label):
    data = []
    labels = []
    for root, _, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".npy"):
                emb = np.load(Path(root) / f)
                data.append(emb)
                labels.append(label)
    return data, labels

X_real, y_real = load_embeddings(EMBEDDING_DIR / "real_data", 1)
X_fake = [], []
for group in (EMBEDDING_DIR / "fake_data").iterdir():
    if group.is_dir():
        x, y = load_embeddings(group, 0)
        X_fake[0].extend(x)
        X_fake[1].extend(y)

X = np.array(X_real + X_fake[0])
y = np.array(y_real + X_fake[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    report_path = RESULTS_DIR / f"{name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}_confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}_roc_curve.png")
    plt.close()
