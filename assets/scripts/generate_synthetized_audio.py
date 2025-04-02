import os
import pandas as pd
from TTS.api import TTS

current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

transcript_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'transcripts', 'transcripts.csv')
processed_audio_path_1 = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio', 'fake_1')
processed_audio_path_2 = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio', 'fake_2')

def generate_coqui(text, output_path, model_name):
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    tts.tts_to_file(text=text, file_path=output_path)

def ensure_clean_dir(path):
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        if f.endswith(".wav"):
            os.remove(os.path.join(path, f))

def main(transcript_path, processed_audio_path_1, processed_audio_path_2):
    df = pd.read_csv(transcript_path)  

    # Use two different Coqui TTS models
    coqui_dir_1 = os.path.join(processed_audio_path_1, "fake_coqui_1")
    coqui_dir_2 = os.path.join(processed_audio_path_2, "fake_coqui_2")

    ensure_clean_dir(coqui_dir_1)
    ensure_clean_dir(coqui_dir_2)

    # Define two different TTS models that don't require espeak
    model_1 = "tts_models/en/ljspeech/tacotron2-DDC"
    model_2 = "tts_models/en/ljspeech/glow-tts"

    for _, row in df.iterrows():
        text, uid = row['text'], row['id']
        print(f"Processing: {uid}")

        # Generate with first model
        generate_coqui(text, os.path.join(coqui_dir_1, f"{uid}_coqui_1.wav"), model_1)
        # Generate with second model
        generate_coqui(text, os.path.join(coqui_dir_2, f"{uid}_coqui_2.wav"), model_2)

    print(f"Generated {len(df)} files for each TTS model.")

if __name__ == "__main__":
    main(transcript_path, processed_audio_path_1, processed_audio_path_2)