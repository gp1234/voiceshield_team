
from huggingface_hub import snapshot_download

model_dir = "xtts_model"

print(f"📥 Downloading XTTS v1 model to '{model_dir}'...")

snapshot_download(
    repo_id="coqui/XTTS-v1",
    local_dir=model_dir,
    ignore_patterns=["*.onnx"]  # Skips ONNX files unless you need them
)

print("✅ Download complete!")
x