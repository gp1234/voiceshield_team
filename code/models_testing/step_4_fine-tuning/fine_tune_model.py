from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from datasets import load_dataset, Audio
import torch
import os
current_file_path = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(current_file_path)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))

AUGMENTED_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'fine_tune_set')
TRAIN_CSV = os.path.join(AUGMENTED_DIR, 'train_metadata.csv')
VAL_CSV = os.path.join(AUGMENTED_DIR,  'test_metadata.csv')

MODEL_NAME = "facebook/wav2vec2-base-960h"
# Load dataset
data_files = {
    "train": TRAIN_CSV,
    "validation": VAL_CSV
}

# Define the processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification",
)

def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=16000, max_length=16000, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k: v.squeeze() for k, v in inputs.items()}  # ensure correct shape [seq_len]
    inputs["label"] = batch["label"]
    return inputs

dataset = load_dataset("csv", data_files=data_files)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.map(preprocess, remove_columns=["audio", "group"])

data_collator = DataCollatorWithPadding(processor, padding=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=10,
    no_cuda=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator
)

trainer.train()
