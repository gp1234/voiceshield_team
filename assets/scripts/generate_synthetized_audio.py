import os
import pandas as pd
from TTS.api import TTS
from gtts import gTTS
import time
import re

current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

transcript_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'transcripts', 'transcripts.csv')
processed_audio_path_1 = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio', 'fake_1')
processed_audio_path_2 = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio', 'fake_2')
processed_audio_path_3 = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio', 'fake_3')

def clean_filename(filename):
    return re.sub(r'\.mp3', '', filename)

def generate_coqui(text, output_dir, model_name, filename, speaker=None):
    """Generate audio using Coqui TTS"""
    try:
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
        output_path = os.path.join(output_dir, filename)
        # Add speaker parameter for multi-speaker models
        if speaker:
            tts.tts_to_file(text=text, file_path=output_path, speaker=speaker)
        else:
            tts.tts_to_file(text=text, file_path=output_path)
        return True
    except Exception as e:
        print(f"Error with Coqui TTS: {str(e)}")
        return False

def generate_gtts(text, output_dir, filename):
    """Generate audio using gTTS"""
    try:
        tts = gTTS(text=text, lang='en')
        output_path = os.path.join(output_dir, filename)
        tts.save(output_path)
        time.sleep(0.1)  # Small delay to prevent rate limiting
        return True
    except Exception as e:
        print(f"Error with gTTS: {str(e)}")
        return False

def ensure_clean_dir(path):
    """Ensure directory exists and is empty of wav files"""
    os.makedirs(path, exist_ok=True)
    for f in os.listdir(path):
        if f.endswith(".wav"):
            os.remove(os.path.join(path, f))

def main(transcript_path, processed_audio_path_1, processed_audio_path_2, processed_audio_path_3):
    """Main function to generate audio using different TTS models"""
    df = pd.read_csv(transcript_path)  

    # Clean output directories
    ensure_clean_dir(processed_audio_path_1)
    ensure_clean_dir(processed_audio_path_2)
    ensure_clean_dir(processed_audio_path_3)

    # Using different models/speakers for variety
    model_1 = "tts_models/en/vctk/vits"  # First VCTK model with male speaker
    model_2 = "tts_models/en/vctk/vits"  # Same model but female speaker

    for _, row in df.iterrows():
        text = row['text']
        # Clean filename by removing .mp3
        filename = clean_filename(f"{row['id']}.wav")
        print(f"Processing: {filename}")

        # Generate audio using different speakers
        generate_coqui(text, processed_audio_path_1, model_1, filename, speaker="p226")  # Male voice
        generate_coqui(text, processed_audio_path_2, model_2, filename, speaker="p229")  # Female voice
        generate_gtts(text, processed_audio_path_3, filename)

    print(f"\nGenerated files for each TTS model.")
    print(f"Files saved in:")
    print(f"- {processed_audio_path_1} (VITS speaker p226 - Male)")
    print(f"- {processed_audio_path_2} (VITS speaker p229 - Female)")
    print(f"- {processed_audio_path_3} (gTTS)")

if __name__ == "__main__":
    main(transcript_path, processed_audio_path_1, processed_audio_path_2, processed_audio_path_3)