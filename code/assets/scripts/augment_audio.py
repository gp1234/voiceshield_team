import os
import random
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_level * noise
    return np.clip(augmented, -1.0, 1.0)

def time_stretch():
    def stretch(audio):
        rate = random.uniform(0.8, 1.2)
        audio_2d = np.expand_dims(audio, axis=0)
        stretched_audio_2d = librosa.effects.time_stretch(y=audio_2d, rate=rate)
        return stretched_audio_2d.flatten()
    return stretch

def pitch_shift(sr):
    def shift(audio):
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=random.uniform(-3, 3))
    return shift

def apply_random_augmentation(audio, sr):
    augmentations = [
        lambda x: add_noise(x),
        time_stretch(),
        pitch_shift(sr)
    ]
    selected = random.sample(augmentations, k=random.randint(1, 2))
    for aug in selected:
        audio = aug(audio)
    return audio

def process_directory(input_dir, output_dir, sample_rate=16000, augmentations_per_file=3):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old augmented wavs directly in the output directory
    # This will remove any previously generated _augX.wav files, but keep _orig.wav files.
    for f in output_dir.rglob("*.wav"):
        if not f.name.endswith("_orig.wav"):
            f.unlink()

    for file in input_dir.rglob("*_orig.wav"):
        base_id = file.stem.replace("_orig", "")
        audio, sr = librosa.load(file, sr=sample_rate)

        # Augmentations
        for i in range(augmentations_per_file):
            augmented = apply_random_augmentation(audio, sr)
            # Save augmented file directly in the output_dir
            out_path = output_dir / f"{base_id}_aug{i+1}.wav"
            sf.write(out_path, augmented, sr)

if __name__ == "__main__":
    process_directory(
        input_dir="/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/audio/augmented_balance/fake_data/group_2",
        output_dir="/Users/giovannipoveda/Documents/deepfake_voice_clonning/code/assets/audio/augmented_balance/fake_data/group_2",
        sample_rate=16000,
        augmentations_per_file=3
    )