import os
import random
from TTS.api import TTS
from string import punctuation

# List of sample sentences to generate fake voices
SAMPLE_SENTENCES = [
    "The weather is beautiful today.",
    "I love going for long walks in the park.",
    "Technology is advancing at an incredible pace.",
    "Music has the power to change our mood.",
    "The coffee shop on the corner makes amazing pastries.",
    "Reading books is my favorite way to relax.",
    "The sunset painted the sky in brilliant colors.",
    "Learning new skills is always exciting.",
    "Time flies when you're having fun.",
    "The garden is blooming with colorful flowers.",
    "Cooking is both an art and a science.",
    "The mountains are covered in fresh snow.",
    "Friendship is one of life's greatest treasures.",
    "The ocean waves are so calming to watch.",
    "Innovation drives progress forward."
]

def create_output_dirs():
    """Create necessary directories for output"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(script_dir, 'files')
    os.makedirs(files_dir, exist_ok=True)
    return files_dir

def initialize_tts_models():
    """Initialize TTS models"""
    # Initialize both models for variety
    models = [
        TTS("tts_models/multilingual/multi-dataset/xtts_v2"),
        TTS("tts_models/en/ljspeech/tacotron2-DDC")
    ]
    return models

def generate_fake_voices(num_samples=5):
    """Generate fake voice samples using different TTS models"""
    output_dir = create_output_dirs()
    models = initialize_tts_models()
    
    print("\nGenerating fake voice samples...")
    print("=" * 50)
    
    for i in range(num_samples):
        try:
            # Randomly select a model and a sentence
            model = random.choice(models)
            text = random.choice(SAMPLE_SENTENCES)
            
            # Create unique filename
            clean_text = ''.join(c for c in text[:30] if c not in punctuation).lower()
            clean_text = clean_text.replace(' ', '_')
            filename = f"fake_voice_{i+1}_{clean_text}.wav"
            output_path = os.path.join(output_dir, filename)
            
            print(f"\nGenerating sample {i+1}/{num_samples}")
            print(f"Text: {text}")
            print(f"Output: {filename}")
            
            # Generate audio
            model.tts_to_file(text=text, file_path=output_path)
            print("✓ Generated successfully")
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {str(e)}")
    
    print("\nDone! Generated files are in the 'files' directory.")
    print("You can now use model_inference.py to test these samples.")

if __name__ == '__main__':
    generate_fake_voices() 