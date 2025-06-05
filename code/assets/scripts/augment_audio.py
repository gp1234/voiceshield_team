import os
import re
import argparse
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Audio processing constants
MIN_DURATION = 3.0 # Minimum duration for padding
TARGET_SR = 16000 # Target sample rate for processing

def clean_filename(filename):
    """Removes file extension from filename."""
    return Path(filename).stem

def convert_mp3_to_wav(input_path, output_path, target_sr=TARGET_SR):
    """Convert MP3 file to WAV format and resample if needed."""
    try:
        audio = AudioSegment.from_mp3(input_path)
        audio = audio.set_frame_rate(target_sr) # Resample during conversion
        audio = audio.set_channels(1) # Ensure mono
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_path} to WAV: {str(e)}")
        return False

def pad_if_short(y, sr, min_duration=MIN_DURATION):
    """Pad audio if shorter than minimum duration."""
    target_len = int(sr * min_duration)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode='constant')
    return y

def add_white_noise(y, snr_db=20):
    """Add white noise to audio."""
    rms = np.sqrt(np.mean(y**2))
    # Avoid division by zero if audio is silent
    if rms == 0:
        return y
    noise = np.random.normal(0, rms / (10**(snr_db / 20)), size=y.shape)
    return y + noise

def apply_reverb(y, sr):
    """Apply reverb effect."""
    # Simple impulse response for a small room effect
    ir = np.random.normal(0, 0.1, int(0.2 * sr))
    return np.convolve(y, ir, mode='full')[:len(y)]

def apply_pitch_shift(y, sr, n_steps=2):
    """Apply pitch shift."""
    # Ensure float32 for librosa
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=n_steps)

def apply_time_stretch(y, rate=0.9):
    """Apply time stretching."""
    # Ensure float32 for librosa
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)

def check_if_all_augmentations_exist(output_dir, base_name):
    """Check if all 5 augmented versions already exist for this base_name."""
    required_suffixes = ['_orig.wav', '_noise.wav', '_reverb.wav', '_pitch.wav', '_stretch.wav']
    for suffix in required_suffixes:
        file_path = os.path.join(output_dir, f"{base_name}{suffix}.wav")
        if not os.path.exists(file_path):
            return False
    return True

def apply_all_augmentations_to_file(input_filepath, output_dir, base_name):
    """Applies all defined augmentations to a single audio file and saves them."""
    try:
        y, sr = sf.read(input_filepath)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Resample to TARGET_SR if necessary
        if sr != TARGET_SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # Convert to float64 for intermediate processing if ufunc errors occur
        y = y.astype(np.float64) 
        
        # Pad if shorter than minimum duration
        y = pad_if_short(y, sr)

        augmentations = {
            '_orig': y,
            '_noise': add_white_noise(y),
            '_reverb': apply_reverb(y, sr),
            '_pitch': apply_pitch_shift(y, sr),
            '_stretch': pad_if_short(apply_time_stretch(y), sr)
        }

        successful_augmentations = 0
        for suffix, audio_data in augmentations.items():
            output_path = os.path.join(output_dir, f"{base_name}{suffix}.wav")
            # Convert back to float32 for saving for common compatibility
            sf.write(output_path, audio_data.astype(np.float32), sr)
            successful_augmentations += 1
        
        return successful_augmentations == len(augmentations)
    except Exception as e:
        print(f"❌ Error augmenting {input_filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Augment audio files with noise, reverb, pitch shift, and time stretch.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing input audio files (.wav or .mp3).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to the directory to save augmented audio files. If not provided, saves to input_dir.")
    parser.add_argument("--process_mp3", action="store_true",
                        help="Set this flag to convert .mp3 files to .wav before augmentation. "
                             "Otherwise, only .wav files will be processed for augmentation.")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if output_dir is None:
        output_dir = input_dir
        print(f"Output directory not specified. Augmented files will be saved in: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Starting audio augmentation process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Process MP3s: {args.process_mp3}")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # List all audio files in the input directory
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]
    
    if not audio_files:
        print(f"No .wav or .mp3 files found in '{input_dir}'. Exiting.")
        return

    for filename in tqdm(audio_files, desc="Augmenting audio files"):
        file_path = os.path.join(input_dir, filename)
        base_name = clean_filename(filename) # Name without extension

        target_filepath_for_augmentation = file_path
        needs_temp_wav_conversion = False
        temp_wav_path = None

        if filename.lower().endswith('.mp3'):
            if args.process_mp3:
                temp_wav_filename = base_name + '_temp_converted.wav'
                temp_wav_path = os.path.join(output_dir, temp_wav_filename) # Save temp WAV in output_dir
                print(f"Converting '{filename}' to WAV for augmentation...")
                if not convert_mp3_to_wav(file_path, temp_wav_path):
                    error_count += 1
                    continue
                target_filepath_for_augmentation = temp_wav_path
                needs_temp_wav_conversion = True
            else:
                print(f"Skipping '{filename}' (MP3) as --process_mp3 flag not set.")
                skipped_count += 1
                continue
        elif not filename.lower().endswith('.wav'): # Skip non-audio files
            print(f"Skipping '{filename}' (not a .wav or .mp3 file).")
            skipped_count += 1
            continue

        # Check if all augmentations for this file already exist in the output directory
        if check_if_all_augmentations_exist(output_dir, base_name):
            print(f"  ⏭️ All augmentations for '{filename}' already exist. Skipping.")
            skipped_count += 1
            if needs_temp_wav_conversion and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path) # Clean up temp if skipped
            continue

        print(f"  Processing '{filename}'...")
        if apply_all_augmentations_to_file(target_filepath_for_augmentation, output_dir, base_name):
            processed_count += 1
        else:
            error_count += 1
        
        # Clean up temporary WAV file if created during MP3 conversion
        if needs_temp_wav_conversion and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    print(f"\n🎉 Augmentation process complete!")
    print(f"📊 Summary:")
    print(f"   ✅ Files augmented: {processed_count}")
    print(f"   ⏭️ Files skipped (already augmented or not processed MP3): {skipped_count}")
    print(f"   ❌ Files with errors: {error_count}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main()