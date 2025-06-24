import argparse
import librosa
from pathlib import Path
import sys
import math

def format_hhmmss(seconds: float) -> str:
    """Convert a duration in seconds to H:MM:SS format."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan a folder for .wav and .mp3 files and sum their durations."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the directory containing your audio files."
    )
    args = parser.parse_args()

    root_dir = args.folder
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    wav_total = 0.0
    mp3_total = 0.0
    # (Optionally) list any files that fail
    failed = []

    # Recursively find all .wav and .mp3
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in (".wav", ".mp3"):
            continue

        try:
            # librosa.get_duration can take a filename directly
            dur = librosa.get_duration(filename=str(path))
        except Exception as e:
            failed.append((path, str(e)))
            continue

        if suffix == ".wav":
            wav_total += dur
        else:  # suffix == ".mp3"
            mp3_total += dur

    combined = wav_total + mp3_total

    print("\n====== Audio Duration Summary ======\n")
    print(f"Folder scanned: {root_dir.resolve()}\n")
    print(f" • Total .wav duration: {wav_total:.2f} seconds   ({format_hhmmss(wav_total)})")
    print(f" • Total .mp3 duration: {mp3_total:.2f} seconds   ({format_hhmmss(mp3_total)})")
    print(f" • Combined total   : {combined:.2f} seconds   ({format_hhmmss(combined)})\n")

    if failed:
        print("⚠️  Some files failed to load. See below:")
        for p, err in failed:
            print(f"   – {p.name}: {err}")
    print()

if __name__ == "__main__":
    main()
