import os
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Trainer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Paths
current_file_path = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(current_file_path)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))

AUGMENTED_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'fine_tune_set')

MODEL_DIR = os.path.join(os.path.dirname(__file__), "results", "checkpoint-280")  # adjust if you want a different checkpoint
DATASET_CSV = os.path.join(AUGMENTED_DIR,  'test_metadata.csv')

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load dataset
dataset = load_dataset("csv", data_files={"validation": DATASET_CSV})
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Preprocessing
def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=16000,
        return_tensors="pt",
        max_length=16000,
        padding="max_length",
        truncation=True,
        no_cuda=True
    )
    batch["input_values"] = inputs["input_values"].squeeze(0)
    
    if "attention_mask" in inputs:
        batch["attention_mask"] = inputs["attention_mask"].squeeze(0)
    
    batch["label"] = batch["label"]
    return batch



dataset = dataset.map(preprocess, remove_columns=["audio", "group"])


# Prepare trainer
trainer = Trainer(model=model, tokenizer=processor)

# Evaluate
metrics = trainer.evaluate(dataset["validation"])
print("✅ Evaluation Metrics:")
print(metrics)
