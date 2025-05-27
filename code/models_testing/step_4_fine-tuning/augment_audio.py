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

MIN_DURATION = 3.0  # seconds
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

def augment_file(file_path, output_dir):
    try:
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)  # convert to mono
        if sr != TARGET_SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        y = pad_if_short(y, sr)
        base_name = Path(file_path).stem

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

    except Exception as e:
        import traceback
        print(f"\n❌ Error processing {file_path}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    for subdir in os.listdir(AUDIO_DIR):
        full_path = os.path.join(AUDIO_DIR, subdir)
        if os.path.isdir(full_path):
            out_path = os.path.join(AUGMENTED_DIR, subdir)
            os.makedirs(out_path, exist_ok=True)

            files = [f for f in os.listdir(full_path) if f.endswith('.wav')]
            for file in tqdm(files, desc=f"Augmenting {subdir}"):
                augment_file(os.path.join(full_path, file), out_path)
