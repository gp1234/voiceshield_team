import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(current_file_path)
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
AUDIO_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio')
AUGMENTED_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'augmented_audio')

MIN_DURATION = 3.0  
TARGET_SR = 16000

os.makedirs(AUGMENTED_DIR, exist_ok=True)

def pad_if_short(y, sr, min_duration=MIN_DURATION):
    target_len = int(sr * min_duration)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode='constant')
    return y

def add_white_noise(y, snr_db=20):
    rms = np.sqrt(np.mean(y**2))
    noise = np.random.normal(0, rms / (10**(snr_db / 20)), size=y.shape)
    return y + noise

def apply_reverb(y, sr):
    ir = np.random.normal(0, 0.1, int(0.2 * sr))  # impulse response (fake small room)
    return np.convolve(y, ir, mode='full')[:len(y)]

def apply_pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=n_steps)

def apply_time_stretch(y, rate=0.9):
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)

def check_augmented_files_exist(base_name, output_dir):
    """Check if all augmented versions of a file already exist."""
    suffixes = ['_orig.wav', '_noise.wav', '_reverb.wav', '_pitch.wav', '_stretch.wav']
    for suffix in suffixes:
        file_path = os.path.join(output_dir, f"{base_name}{suffix}")
        if not os.path.exists(file_path):
            return False
    return True

def count_existing_files(output_dir):
    """Count how many .wav files already exist in the output directory."""
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith('.wav')])

def augment_file(file_path, output_dir):
    """Augment a single audio file with various effects."""
    base_name = Path(file_path).stem
    
    # Check if augmented files already exist
    if check_augmented_files_exist(base_name, output_dir):
        print(f"⏭️  Skipping {base_name} - augmented files already exist")
        return
    
    try:
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)  # convert to mono
        if sr != TARGET_SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        y = pad_if_short(y, sr)

        # Original
        sf.write(os.path.join(output_dir, f"{base_name}_orig.wav"), y, sr)

        # White noise
        sf.write(os.path.join(output_dir, f"{base_name}_noise.wav"), add_white_noise(y), sr)

        # Reverb
        sf.write(os.path.join(output_dir, f"{base_name}_reverb.wav"), apply_reverb(y, sr), sr)

        # Pitch shift
        sf.write(os.path.join(output_dir, f"{base_name}_pitch.wav"), apply_pitch_shift(y, sr), sr)

        # Time stretch
        stretched = apply_time_stretch(y)
        stretched = pad_if_short(stretched, sr)
        sf.write(os.path.join(output_dir, f"{base_name}_stretch.wav"), stretched, sr)

        print(f"✅ Processed {base_name}")

    except Exception as e:
        import traceback
        print(f"\n❌ Error processing {file_path}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🎵 Starting audio augmentation process...")
    print(f"📁 Source directory: {AUDIO_DIR}")
    print(f"📁 Output directory: {AUGMENTED_DIR}")
    
    total_processed = 0
    total_skipped = 0
    
    for subdir in os.listdir(AUDIO_DIR):
        full_path = os.path.join(AUDIO_DIR, subdir)
        if os.path.isdir(full_path):
            out_path = os.path.join(AUGMENTED_DIR, subdir)
            os.makedirs(out_path, exist_ok=True)

            files = [f for f in os.listdir(full_path) if f.endswith('.wav')]
            existing_count = count_existing_files(out_path)
            
            print(f"\n📂 Processing directory: {subdir}")
            print(f"   📊 Input files: {len(files)} | Existing augmented files: {existing_count}")
            
            # If we already have the expected number of augmented files, skip the entire directory
            expected_augmented_files = len(files) * 5  # 5 augmented versions per file
            if existing_count >= expected_augmented_files:
                print(f"   ⏭️  Skipping {subdir} - all files already augmented ({existing_count} files)")
                total_skipped += len(files)
                continue
            
            files_processed_in_dir = 0
            for file in tqdm(files, desc=f"Augmenting {subdir}"):
                base_name = Path(file).stem
                if not check_augmented_files_exist(base_name, out_path):
                    augment_file(os.path.join(full_path, file), out_path)
                    files_processed_in_dir += 1
                    total_processed += 1
                else:
                    total_skipped += 1
            
            print(f"   ✅ Processed {files_processed_in_dir} new files in {subdir}")
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Files processed: {total_processed}")
    print(f"   ⏭️  Files skipped: {total_skipped}")
    print("🏁 Audio augmentation complete!")
