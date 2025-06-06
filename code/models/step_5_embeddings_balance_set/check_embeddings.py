import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

EMBEDDING_DIR = Path("/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/models/step_5_embeddings_balance_set/openl3_embeddings")

def load_grouped_paths(label_dir):
    grouped = defaultdict(list)
    for root, _, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".npy"):
                match = re.search(r"(\\d+)_", f)
                if match:
                    base_id = match.group(1)
                    grouped[base_id].append(str(Path(root) / f))
    return grouped

def collect_all_base_ids():
    real_ids = load_grouped_paths(EMBEDDING_DIR / "real_data")
    fake_ids = {}
    for group in (EMBEDDING_DIR / "fake_data").iterdir():
        if group.is_dir():
            fake_ids.update(load_grouped_paths(group))
    return real_ids, fake_ids

def simulate_split(real_ids, fake_ids, test_ratio=0.2):
    all_groups = list(real_ids.items()) + list(fake_ids.items())
    np.random.seed(42)
    np.random.shuffle(all_groups)
    split = int(len(all_groups) * (1 - test_ratio))
    train_ids = set([base for base, _ in all_groups[:split]])
    test_ids = set([base for base, _ in all_groups[split:]])
    return train_ids, test_ids

real_ids, fake_ids = collect_all_base_ids()
train_ids, test_ids = simulate_split(real_ids, fake_ids)

overlap = train_ids & test_ids

print(f"✅ Train IDs: {len(train_ids)}")
print(f"✅ Test IDs: {len(test_ids)}")
print(f"❌ Overlapping base IDs: {len(overlap)}")

if overlap:
    print("⚠️ Potential leakage from these base IDs:")
    print(sorted(list(overlap)))
else:
    print("✅ No base ID leakage detected.")
