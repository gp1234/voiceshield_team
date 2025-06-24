import os
import re
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────────────────────────────────────
# 1) Change this to the absolute path where your "openl3_embeddings" folder lives:
EMBEDDING_DIR = Path(
    "/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/models/"
    "step_5_embeddings_balance_set/openl3_embeddings"
)
RESULTS_DIR = Path("training_results")
RESULTS_DIR.mkdir(exist_ok=True)
# ──────────────────────────────────────────────────────────────────────────────

# 2) Sanity‐check: ensure the script sees "real_data" and "fake_data"
print("▶ Checking EMBEDDING_DIR:", EMBEDDING_DIR)
print("▶ real_data exists:", (EMBEDDING_DIR / "real_data").exists())
print(
    "▶ real_data subfolders:",
    sorted([p.name for p in (EMBEDDING_DIR / "real_data").iterdir() if p.is_dir()]),
)
print("▶ fake_data exists:", (EMBEDDING_DIR / "fake_data").exists())
print(
    "▶ fake_data subfolders:",
    sorted([p.name for p in (EMBEDDING_DIR / "fake_data").iterdir() if p.is_dir()]),
)
print("▶ Any .npy under real_data?", any((EMBEDDING_DIR / "real_data").rglob("*.npy")))
print("▶ Any .npy under fake_data?", any((EMBEDDING_DIR / "fake_data").rglob("*.npy")))
print("──────────────────────────────────────────────────────────────────────────────")


def load_grouped_embeddings(label_dir: Path, label: int):
    """
    Walk through label_dir (e.g. openl3_embeddings/real_data/group_1),
    find all “*.npy” files named like
      "common_voice_en_<digits>_orig.npy" or "common_voice_en_<digits>_aug1.npy",
    extract the <digits> as base-ID, and group all embeddings of that base-ID together.

    label = 1 for real_data, label = 0 for fake_data.
    Returns: { base_id_str: [ (embedding_array, label), … ] }
    """
    grouped = defaultdict(list)
    for root, _, files in os.walk(label_dir):
        for f in files:
            if not f.endswith(".npy"):
                continue

            # Match "common_voice_en_<digits>_" → capture digits as base-ID
            m = re.search(r"common_voice_en_(\d+)_", f)
            if not m:
                continue

            base_id = m.group(1)  # e.g. "40865211"
            emb_arr = np.load(Path(root) / f)
            grouped[base_id].append((emb_arr, label))

    return grouped


def unpack_groups(group_list):
    """
    Given a list of groups (each group is a list of (embedding, label) tuples),
    flatten into:
      X → np.ndarray of shape [n_samples, embedding_dim]
      y → np.ndarray of shape [n_samples], containing labels 0 or 1.
    """
    X, y = [], []
    for group in group_list:
        for emb, lbl in group:
            X.append(emb)
            y.append(lbl)
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Load “real” embeddings (label = 1)
real_groups = {}
raw_real_dir = EMBEDDING_DIR / "real_data"
if raw_real_dir.exists():
    for sub in sorted(raw_real_dir.iterdir()):
        if sub.is_dir():
            loaded = load_grouped_embeddings(sub, 1)
            real_groups.update(loaded)
else:
    print(f"‼ real_data folder not found at {raw_real_dir}")

print(
    f"▶ Found real base-IDs: {len(real_groups)}  "
    f"→ {list(real_groups.keys())[:10]}{'…' if len(real_groups) > 10 else ''}"
)

# 4) Load “fake” embeddings (label = 0)
fake_groups = {}
raw_fake_dir = EMBEDDING_DIR / "fake_data"
if raw_fake_dir.exists():
    for sub in sorted(raw_fake_dir.iterdir()):
        if sub.is_dir():
            loaded = load_grouped_embeddings(sub, 0)
            fake_groups.update(loaded)
else:
    print(f"‼ fake_data folder not found at {raw_fake_dir}")

print(
    f"▶ Found fake base-IDs: {len(fake_groups)}  "
    f"→ {list(fake_groups.keys())[:10]}{'…' if len(fake_groups) > 10 else ''}"
)

# 5) Combine all “real” and “fake” groups into one list, then shuffle
all_groups = list(real_groups.values()) + list(fake_groups.values())
print(f"▶ Total groups (real + fake): {len(all_groups)}")
np.random.seed(42)
np.random.shuffle(all_groups)

# 6) Split 80% train / 20% test at the group level (prevents any base-ID leakage)
split_idx = int(len(all_groups) * 0.8)
train_groups = all_groups[:split_idx]
test_groups = all_groups[split_idx:]

print(f"▶ Train groups count: {len(train_groups)}")
print(f"▶ Test groups count:  {len(test_groups)}")

# 7) Build reverse‐map so we can check which base-IDs went where
reverse_map = {}
for bid, grp in real_groups.items():
    reverse_map[id(grp)] = bid
for bid, grp in fake_groups.items():
    reverse_map[id(grp)] = bid

train_base_ids = [reverse_map.get(id(grp)) for grp in train_groups]
test_base_ids = [reverse_map.get(id(grp)) for grp in test_groups]

print(f"▶ Train base-IDs  (first 10): {train_base_ids[:10]}{'…' if len(train_base_ids) > 10 else ''}")
print(f"▶ Test base-IDs   (first 10): {test_base_ids[:10]}{'…' if len(test_base_ids) > 10 else ''}")
overlap = set(train_base_ids) & set(test_base_ids)
print(f"▶ Overlap in base-IDs between train/test: {overlap}")
print("──────────────────────────────────────────────────────────────────────────────")

# 8) Flatten groups into X_train, y_train, X_test, y_test
X_train, y_train = unpack_groups(train_groups)
X_test, y_test = unpack_groups(test_groups)

print("▶ After flattening:")
print(f"    • X_train shape: {X_train.shape}")
print(f"    • y_train distribution: {Counter(y_train)}")
print(f"    • X_test shape:  {X_test.shape}")
print(f"    • y_test distribution:  {Counter(y_test)}")

# 9) Check how many unique embeddings are in train
unique_train = len(np.unique([emb.tobytes() for emb in X_train]))
print(f"▶ Unique train embeddings: {unique_train} / {len(X_train)}")

# 10) Guard against empty data
if X_train.size == 0 or X_test.size == 0:
    print("\n‼ No embeddings were loaded. Please check EMBEDDING_DIR and filename patterns.")
    exit(1)

# 11) PCA visualization (2D)
pca = PCA(n_components=2)
X_vis = pca.fit_transform(np.vstack((X_train, X_test)))
y_vis = np.hstack((y_train, y_test))

plt.figure(figsize=(6, 6))
plt.scatter(X_vis[y_vis == 0, 0], X_vis[y_vis == 0, 1], alpha=0.5, label="Fake")
plt.scatter(X_vis[y_vis == 1, 0], X_vis[y_vis == 1, 1], alpha=0.5, label="Real")
plt.title("PCA Projection of Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "pca_projection_debug.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 12) Define classifiers
models = {
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MLP":                MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42),
    "SVM":                SVC(kernel="rbf", probability=True),
    "LightGBM":           LGBMClassifier(force_col_wise=True),
}

# 13) Train, evaluate, and save outputs for each model (model+scaler in one .joblib)
for name, model in models.items():
    print(f"\n🔧 Training {name}…")

    model_dir = RESULTS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Decide if we need a scaler
    if name in ["LogisticRegression", "MLP", "SVM", "LightGBM"]:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        scaler    = None
        X_train_s = X_train
        X_test_s  = X_test

    # Fit the model
    model.fit(X_train_s, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # ─► Save both model and scaler in one bundled file
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, model_dir / f"{name}_bundle.joblib")

    # Write classification report
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    with open(model_dir / "report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
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
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix.png")
    plt.close()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(model_dir / "roc_curve.png")
    plt.close()
