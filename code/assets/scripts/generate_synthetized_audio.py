import os
import pandas as pd
from TTS.api import TTS
from gtts import gTTS
import time
import re
from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

current_file = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

transcript_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'transcripts', 'transcripts.csv')
original_audio_path = os.path.join(ROOT_DIR, 'assets', 'audio', 'original_audio')
processed_audio_path_real = os.path.join(ROOT_DIR, 'assets', 'audio', 'processed_audio', 'real')

# Base directory for augmented audio (where existing fake_1, fake_2, fake_3 already exist)
AUGMENTED_AUDIO_DIR = os.path.join(ROOT_DIR, 'assets', 'audio', 'augmented_audio')

# NEW MODELS TO GENERATE (starting from fake_4)
# Note: fake_1, fake_2, fake_3 already exist in augmented_audio folder
TTS_MODELS_CONFIG = [
    {
        'name': 'coqui_tacotron2',
        'path': os.path.join(AUGMENTED_AUDIO_DIR, 'fake_4'),
        'type': 'coqui',
        'model': 'tts_models/en/ljspeech/tacotron2-DDC',
        'speaker': None,
        'description': 'Coqui Tacotron2',
        'enabled': True
    },
    {
        'name': 'xtts_v2',
        'path': os.path.join(AUGMENTED_AUDIO_DIR, 'fake_5'),
        'type': 'xtts',
        'model': 'tts_models/multilingual/multi-dataset/xtts_v2',
        'speaker': 'female', # XTTS v2 requires a speaker. This is a placeholder.
        'description': 'XTTS v2 (Advanced)',
        'enabled': False # Disabled - focus on Bark
    },
    {
        'name': 'bark',
        'path': os.path.join(AUGMENTED_AUDIO_DIR, 'fake_6'),
        'type': 'bark',
        'model': None,
        'speaker': None,
        'description': 'Bark (Very Advanced)',
        'enabled': True # Enabled for testing
    }
]

# Audio processing constants
MIN_DURATION = 3.0
TARGET_SR = 16000

def clean_filename(filename):
    return re.sub(r'\.mp3', '', filename)

# =================== AUDIO AUGMENTATION FUNCTIONS ===================
def pad_if_short(y, sr, min_duration=MIN_DURATION):
    """Pad audio if shorter than minimum duration"""
    target_len = int(sr * min_duration)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode='constant')
    return y

def add_white_noise(y, snr_db=20):
    """Add white noise to audio"""
    rms = np.sqrt(np.mean(y**2))
    noise = np.random.normal(0, rms / (10**(snr_db / 20)), size=y.shape)
    return y + noise

def apply_reverb(y, sr):
    """Apply reverb effect"""
    ir = np.random.normal(0, 0.1, int(0.2 * sr))  # impulse response (fake small room)
    return np.convolve(y, ir, mode='full')[:len(y)]

def apply_pitch_shift(y, sr, n_steps=2):
    """Apply pitch shift"""
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=n_steps)

def apply_time_stretch(y, rate=0.9):
    """Apply time stretching"""
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)

def check_if_already_generated(output_dir, base_name):
    """Check if all 5 augmented versions already exist for this base_name"""
    required_suffixes = ['_orig.wav', '_noise.wav', '_pitch.wav', '_reverb.wav', '_stretch.wav']
    for suffix in required_suffixes:
        file_path = os.path.join(output_dir, f"{base_name}{suffix}")
        if not os.path.exists(file_path):
            return False
    return True

def apply_augmentations(audio_path, output_dir, base_name):
    """Apply all augmentations to an audio file"""
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)  # convert to mono
        if sr != TARGET_SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # CRITICAL ADDITION: Convert to float64 before augmentations if problems persist
        # This can sometimes resolve ufunc type mismatches.
        y = y.astype(np.float64) 
        
        y = pad_if_short(y, sr)

        # Save augmented versions
        augmentations = {
            '_orig': y,
            '_noise': add_white_noise(y),
            '_reverb': apply_reverb(y, sr),
            '_pitch': apply_pitch_shift(y, sr),
            '_stretch': pad_if_short(apply_time_stretch(y), sr)
        }

        for suffix, audio_data in augmentations.items():
            output_path = os.path.join(output_dir, f"{base_name}{suffix}.wav")
            # Convert back to float32 for saving to reduce file size and common compatibility
            sf.write(output_path, audio_data.astype(np.float32), sr) 
        
        return True
    except Exception as e:
        print(f"❌ Error augmenting {audio_path}: {e}")
        return False

# =================== TTS GENERATION FUNCTIONS ===================

def generate_coqui(text, output_dir, model_name, filename, speaker=None):
    """Generate audio using Coqui TTS"""
    try:
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
        output_path = os.path.join(output_dir, filename)
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
        time.sleep(0.1) 
        return True
    except Exception as e:
        print(f"Error with gTTS: {str(e)}")
        return False

def generate_xtts(text, output_dir, filename, speaker_wav_path=None):
    """Generate audio using XTTS v2 (Advanced)"""
    try:
        import torch
        import warnings
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Removed the add_safe_globals line for XTTS as it's now disabled.
        
        # Initialize XTTS with error handling
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        output_path = os.path.join(output_dir, filename)
        
        # Use a shorter text if too long (XTTS has length limits)
        if len(text) > 200:
            text = text[:200] + "..."
        
        # XTTS v2 requires a speaker_wav for voice cloning
        if speaker_wav_path is None:
            # Try to find a reference speaker audio in the 'real' directory
            real_audio_dir = os.path.join(AUGMENTED_AUDIO_DIR, 'real')
            if os.path.exists(real_audio_dir):
                speaker_files = [f for f in os.listdir(real_audio_dir) if f.endswith('_orig.wav')]
                if speaker_files:
                    speaker_wav_path = os.path.join(real_audio_dir, speaker_files[0])
                    print(f"    🎤 Using speaker reference: {speaker_files[0]}")
                else:
                    raise Exception("No speaker reference available in 'real' directory.")
            else:
                raise Exception("Real audio directory not found for speaker reference.")
        
        print(f"    🎤 Generating with XTTS v2 using speaker reference: {speaker_wav_path}")
        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav_path, language="en")
        
        return True
        
    except Exception as e:
        print(f"Error with XTTS: {str(e)}")
        return False

def generate_bark(text, output_dir, filename):
    """Generate audio using Bark (Very Advanced) with Transformers"""
    try:
        import torch
        import warnings
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Try to import transformers components
        try:
            from transformers import AutoProcessor, BarkModel
        except ImportError:
            print("❌ Transformers not installed. Run 'pip install transformers torch'")
            return False
        
        # Initialize processor and model with error handling
        try:
            processor = AutoProcessor.from_pretrained("suno/bark")
            model = BarkModel.from_pretrained("suno/bark")
        except Exception as e:
            print(f"❌ Failed to load Bark model: {str(e)}")
            return False
        
        # Use a consistent voice preset (female English speaker)
        voice_preset = "v2/en_speaker_6"
        
        # Limit text length for Bark (it has limits)
        if len(text) > 100:
            text = text[:100] + "..."
        
        # Process input text with voice preset
        inputs = processor(text, voice_preset=voice_preset)
        
        # Generate audio
        with torch.no_grad():
            audio_array = model.generate(**inputs, do_sample=True, top_k=50, top_p=0.95, temperature=0.7) # Added generation parameters for better control
            audio_array = audio_array.cpu().numpy().squeeze()
        
        # Save audio at standard sample rate (Bark uses 24kHz)
        sample_rate = 24000  # Bark's native sample rate
        temp_path = os.path.join(output_dir, f"temp_{filename}")
        sf.write(temp_path, audio_array, sample_rate)
        
        # Convert to standard format (16kHz)
        audio = AudioSegment.from_wav(temp_path)
        audio = audio.set_frame_rate(TARGET_SR)  # Convert to 16kHz
        final_path = os.path.join(output_dir, filename)
        audio.export(final_path, format="wav")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return True
        
    except Exception as e:
        print(f"Error with Bark: {str(e)}")
        return False

def ensure_clean_dir(path):
    """Ensure directory exists (but preserve existing wav files for resume functionality)"""
    os.makedirs(path, exist_ok=True)
    # Note: No longer deleting existing .wav files to support resume functionality

def convert_mp3_to_wav(input_path, output_path):
    """Convert MP3 file to WAV format"""
    try:
        audio = AudioSegment.from_mp3(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_path} to WAV: {str(e)}")
        return False

def process_original_audio(original_dir, output_dir):
    """Process all MP3 files from original directory and convert to WAV"""
    ensure_clean_dir(output_dir)
    
    for filename in os.listdir(original_dir):
        if filename.endswith('.mp3'):
            input_path = os.path.join(original_dir, filename)
            output_filename = clean_filename(filename) + '.wav'
            output_path = os.path.join(output_dir, output_filename)
            print(f"Converting {filename} to WAV...")
            convert_mp3_to_wav(input_path, output_path)

def main_comprehensive():
    """Main function to generate NEW TTS models with augmentation in augmented_audio folder"""
    
    # Load transcripts
    df = pd.read_csv(transcript_path)
    
    # Filter only enabled NEW models
    new_models = [config for config in TTS_MODELS_CONFIG if config.get('enabled', False)]
    
    print(f"\n🎵 Starting NEW TTS model generation...")
    print(f"📋 Plan:")
    print(f"   ✅ Existing models: fake_1, fake_2, fake_3, real (skip - already processed)")
    print(f"   🆕 Generate {len(new_models)} NEW models with augmentation")
    print(f"   📁 Save to: {AUGMENTED_AUDIO_DIR}")
    
    if not new_models:
        print("\n⚠️ No new models enabled. Enable models in TTS_MODELS_CONFIG to generate.")
        return
    
    print(f"\n🆕 NEW models to generate:")
    for config in new_models:
        folder_name = os.path.basename(config['path'])
        print(f"  - {config['description']} → {folder_name}/")
    
    # Create directories for new models
    for model_config in new_models:
        ensure_clean_dir(model_config['path'])
        folder_name = os.path.basename(model_config['path'])
        print(f"📁 Created directory: {folder_name}/ for {model_config['description']}")
    
    # Process each transcript for NEW models
    print(f"\n🔄 Processing {len(df)} transcripts...")
    skipped_count = 0
    generated_count = 0
    
    # Determine a speaker_wav_path for XTTS v2 if it's enabled (now disabled)
    xtts_speaker_wav_global = None # This will remain None as XTTS is disabled.


    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating new models"):
        text = row['text']
        base_filename = clean_filename(f"{row['id']}")
        print(f"\n📝 Processing: {base_filename}")
        
        # Generate with each NEW TTS model
        for model_config in new_models:
            model_name = model_config['name']
            output_dir = model_config['path']
            
            # Check if already generated (all 5 augmented versions exist)
            if check_if_already_generated(output_dir, base_filename):
                print(f"  ⏭️  {model_config['description']} - Already generated, skipping")
                skipped_count += 1
                continue
            
            temp_filename = f"{base_filename}_temp.wav"
            temp_path = os.path.join(output_dir, temp_filename)
            
            print(f"  🔊 Generating with {model_config['description']}...")
            
            # Generate base audio with appropriate TTS
            success = False
            if model_config['type'] == 'coqui':
                success = generate_coqui(
                    text, output_dir, model_config['model'], 
                    temp_filename, model_config['speaker']
                )
            elif model_config['type'] == 'gtts':
                success = generate_gtts(text, output_dir, temp_filename)
            elif model_config['type'] == 'xtts': # This branch won't be hit if XTTS is disabled
                # Pass the global speaker_wav_path to XTTS
                success = generate_xtts(text, output_dir, temp_filename, speaker_wav_path=xtts_speaker_wav_global)
            elif model_config['type'] == 'bark':
                success = generate_bark(text, output_dir, temp_filename)
            
            if success and os.path.exists(temp_path):
                # Apply augmentations to the generated audio
                print(f"    🎛️  Applying augmentations...")
                augment_success = apply_augmentations(temp_path, output_dir, base_filename)
                
                if augment_success:
                    print(f"    ✅ Generated 5 augmented versions")
                    generated_count += 1
                    # Remove temporary file
                    os.remove(temp_path)
                else:
                    print(f"    ⚠️  Augmentation failed, keeping original")
                    # Rename temp file to original
                    final_path = os.path.join(output_dir, f"{base_filename}_orig.wav")
                    os.rename(temp_path, final_path)
            else:
                print(f"    ❌ Failed to generate with {model_config['description']}")
            
            time.sleep(0.1)  # Small delay between models
    
    # Summary
    print(f"\n🎉 NEW TTS model generation complete!")
    print(f"📊 Summary:")
    print(f"   📝 Processed {len(df)} texts")
    print(f"   🆕 Generated {len(new_models)} new TTS models")
    print(f"   ✅ Successfully generated: {generated_count} model-text combinations")
    print(f"   ⏭️  Skipped (already existed): {skipped_count} model-text combinations")
    print(f"   🎛️  Applied 5 augmentations per model per text")
    
    print(f"\n📂 Complete dataset structure in {AUGMENTED_AUDIO_DIR}:")
    
    # Check existing folders (skip real audio processing)
    existing_folders = ['fake_1', 'fake_2', 'fake_3', 'real']
    for folder in existing_folders:
        folder_path = os.path.join(AUGMENTED_AUDIO_DIR, folder)
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.wav')])
            print(f"   ✅ {folder}/ - {file_count} files (existing - not modified)")
    
    # Check NEW generated folders
    for config in new_models:
        folder_name = os.path.basename(config['path'])
        if os.path.exists(config['path']):
            file_count = len([f for f in os.listdir(config['path']) if f.endswith('.wav')])
            print(f"   🆕 {folder_name}/ - {file_count} files ({config['description']})")
    
    print(f"\n🎯 Ready for wav2vec2 fine-tuning!")
    print(f"   ✅ All models in: {AUGMENTED_AUDIO_DIR}")
    print(f"   ✅ Each NEW model has 5 augmented versions per transcript")
    print(f"   ✅ Existing models (fake_1, fake_2, fake_3, real) untouched")

def main(transcript_path, processed_audio_path_1, processed_audio_path_2, processed_audio_path_3):
    """Legacy main function for compatibility"""
    print("⚠️  Using legacy main function. Consider using main_comprehensive() for all models.")
    
    # First process original audio files
    print("Processing original audio files...")
    process_original_audio(original_audio_path, processed_audio_path_real)
    
    df = pd.read_csv(transcript_path)  

    ensure_clean_dir(processed_audio_path_1)
    ensure_clean_dir(processed_audio_path_2)
    ensure_clean_dir(processed_audio_path_3)

    model_1 = "tts_models/en/vctk/vits"  
    model_2 = "tts_models/en/vctk/vits" 

    for _, row in df.iterrows():
        text = row['text']
        filename = clean_filename(f"{row['id']}.wav")
        print(f"Processing: {filename}")

        # Generate audio using different speakers
        generate_coqui(text, processed_audio_path_1, model_1, filename, speaker="p226")  # Male voice
        generate_coqui(text, processed_audio_path_2, model_2, filename, speaker="p229")  # Female voice
        generate_gtts(text, processed_audio_path_3, filename)

    print(f"\nGenerated files for each TTS model.")
    print(f"Files saved in:")
    print(f"- {processed_audio_path_real} (Original audio converted to WAV)")
    print(f"- {processed_audio_path_1} (VITS speaker p226 - Male)")
    print(f"- {processed_audio_path_2} (VITS speaker p229 - Female)")
    print(f"- {processed_audio_path_3} (gTTS)")

if __name__ == "__main__":
    # Use the comprehensive function for all models with augmentation
    print("🚀 Starting comprehensive TTS generation with augmentation...")
    main_comprehensive()
    
    # Uncomment below to use legacy function instead
    # main(transcript_path, processed_audio_path_1, processed_audio_path_2, processed_audio_path_3)