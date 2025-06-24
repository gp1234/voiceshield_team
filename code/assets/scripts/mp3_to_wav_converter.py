"""
Usage: python convert_mp3_to_wav.py assets/audio/augmented_audio
"""
import os
from pydub import AudioSegment
import argparse

def convert_mp3_to_wav(folder_path):
    """
    Convert all MP3 files in the specified folder to WAV format and delete the original MP3 files.
    
    Args:
        folder_path (str): Path to the folder containing MP3 files.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    # Get all files in the folder
    files = os.listdir(folder_path)
    mp3_files = [f for f in files if f.lower().endswith('.mp3')]

    if not mp3_files:
        print(f"No MP3 files found in '{folder_path}'.")
        return

    # Process each MP3 file
    for mp3_file in mp3_files:
        mp3_path = os.path.join(folder_path, mp3_file)
        wav_file = os.path.splitext(mp3_file)[0] + '.wav'
        wav_path = os.path.join(folder_path, wav_file)

        try:
            # Load MP3 file
            audio = AudioSegment.from_mp3(mp3_path)
            # Export as WAV
            audio.export(wav_path, format="wav")
            print(f"Converted '{mp3_file}' to '{wav_file}'.")

            # Delete the original MP3 file
            os.remove(mp3_path)
            print(f"Deleted original MP3 file: '{mp3_file}'.")
        except Exception as e:
            print(f"Error processing '{mp3_file}': {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert MP3 files to WAV and delete the original MP3 files.")
    parser.add_argument('folder_path', type=str, help="Path to the folder containing MP3 files")
    args = parser.parse_args()

    # Run the conversion
    convert_mp3_to_wav(args.folder_path)

if __name__ == "__main__":
    main()