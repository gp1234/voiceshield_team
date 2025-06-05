import os
import openl3
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa # Added for potential resampling if needed by OpenL3's speech model

# Determine ROOT_DIR based on the script's location
# Assuming this script is in a subdirectory of your main project (e.g., a 'scripts' folder)
# Adjust the number of os.path.dirname calls if your structure is different.
# If this script is at the root of your project, ROOT_DIR = os.getcwd() or os.path.dirname(os.path.abspath(__file__))
current_file_path = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(current_file_path) # e.g., /path/to/project/scripts
ROOT_DIR = os.path.dirname(SCRIPTS_DIR) # e.g., /path/to/project/

AUDIO_ROOT = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')
# Ensure the output path is relative to the project root or a defined output area
# For example, saving it into an 'embeddings_data' directory at the project root
FEATURES_DATA_DIR = os.path.join(ROOT_DIR, 'step_3_embeddings', 'openl3_features')
EMBEDDING_CSV = os.path.join(FEATURES_DATA_DIR, 'openl3_features.csv')

output_features_dir = os.path.join(ROOT_DIR, 'step_3_embeddings', 'summary') # Changed output directory
os.makedirs(output_features_dir, exist_ok=True)
OUTPUT_CSV = os.path.join(output_features_dir, 'openl3_speech_features_real.csv')

TARGET_SR = 16000 # OpenL3 speech models often expect 16kHz

def extract_embedding(audio_path):
    """
    Extracts OpenL3 embedding for speech content.
    Audio is resampled to TARGET_SR if necessary.
    """
    try:
        audio, sr = sf.read(audio_path)

        # Ensure audio is mono for OpenL3
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample audio if its sample rate is different from TARGET_SR
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR # Update sample rate to target

        # Using content_type="speech"
        emb, ts = openl3.get_audio_embedding(audio, sr, content_type="speech", embedding_size=512, hop_size=0.5) # Using hop_size for potentially more frames
        if emb.shape[0] == 0: # Check if embedding is empty
            print(f"Warning: Empty embedding for {audio_path}. Audio might be too short even after processing.")
            return None
        return np.mean(emb, axis=0) # Average over time frames
    except Exception as e:
        print(f"Failed to extract embedding for {audio_path}: {e}")
        return None

def label_and_group(file_path, audio_root_path):
    """
    Determines label, group_id, and source from the file path.
    Labeling: Real = 1, Fake = 0.
    """
    # Get the relative path from the audio_root_path to determine the source folder
    relative_path = os.path.relpath(file_path, audio_root_path)
    parts = relative_path.split(os.sep)
    
    # The source folder (e.g., "real", "fake_gtts") is the first part of the relative path
    source_folder_name = parts[0] 
    
    # UPDATED LABELING: Real = 1, Fake = 0
    label = 1 if source_folder_name == "real" else 0 
    
    source_type = source_folder_name # e.g., "real", "fake_gtts", "fake_coqui_1"
    
    # Group ID is the filename without extension
    # This assumes filenames are unique identifiers for content across different source_types
    # e.g., "audio_clip_001.wav" in "real" and "audio_clip_001.wav" in "fake_gtts" belong to the same group.
    group_id = os.path.splitext(parts[-1])[0] 
    
    return label, group_id, source_type

def main():
    rows = []
    print(f"Looking for audio files in: {AUDIO_ROOT}")
    if not os.path.exists(AUDIO_ROOT):
        print(f"ERROR: Audio root directory not found: {AUDIO_ROOT}")
        return

    for source_folder in os.listdir(AUDIO_ROOT):
        current_source_path = os.path.join(AUDIO_ROOT, source_folder)
        if os.path.isdir(current_source_path):
            print(f"Processing folder: {source_folder}")
            for file_name in tqdm(os.listdir(current_source_path), desc=f"Extracting from {source_folder}"):
                if file_name.lower().endswith((".wav", ".mp3")): # Process both wav and mp3
                    full_file_path = os.path.join(current_source_path, file_name)
                    
                    embedding = extract_embedding(full_file_path)
                    
                    if embedding is not None:
                        label, group_id, source_type = label_and_group(full_file_path, AUDIO_ROOT)
                        rows.append({
                            "file": full_file_path, # Storing full path for clarity
                            "label": label,
                            "features": ",".join(map(str, embedding)), # Store embedding as comma-separated string
                            "group": group_id,
                            "source": source_type
                        })
    
    if not rows:
        print("No embeddings were extracted. Please check audio paths and file formats.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Successfully extracted {len(df)} embeddings.")
    print(f"Dataset saved to: {OUTPUT_CSV}")
    print("\nLabel distribution in the generated CSV:")
    print(df['label'].value_counts())
    print("\nSource distribution in the generated CSV:")
    print(df['source'].value_counts())

if __name__ == "__main__":
    main()
