import pandas as pd
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import scipy.io.wavfile
import os
import logging
import sys
import timeout_decorator
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sys.stdout.flush()

# Initialize SpeechT5 model, processor, and vocoder (CPU only)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to("cpu")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cpu")

# Load speaker embeddings (for consistent voice)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0).to("cpu")

# Load transcripts
df = pd.read_csv("transcripts.csv", encoding="utf-8")

# Create output directory
os.makedirs("speecht5_outputs", exist_ok=True)

# Process each transcript with timeout
@timeout_decorator.timeout(30, timeout_exception=TimeoutError)
def generate_audio(inputs, filename):
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    speech = speech.cpu().numpy()
    speech = np.clip(speech, -1.0, 1.0)
    speech = (speech * 32767).astype(np.int16)
    scipy.io.wavfile.write(os.path.join("speecht5_outputs", filename), rate=16000, data=speech)

for idx, row in df.iterrows():
    filename = os.path.basename(row["id"]).replace(".mp3", ".wav")
    text = row["text"]

    logger.info(f"Synthesizing: {filename} -> {text}")
    try:
        inputs = processor(text=text, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        generate_audio(inputs, filename)
    except TimeoutError:
        logger.error(f"Timeout generating audio for {filename}")
        continue
    except Exception as e:
        logger.error(f"Error generating audio for {filename}: {e}")
        continue

logger.info("All audios generated.")