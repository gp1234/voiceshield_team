"""
Video Processing Module for VoiceShield WhatsApp Integration
Extracts audio from video files for voice analysis
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
from pydub import AudioSegment


def detect_media_type(content_type: str, file_path: str = None) -> str:
    """
    Detect if the media file is audio or video based on content type and file extension

    Args:
        content_type: MIME content type from Twilio
        file_path: Optional file path for extension checking

    Returns:
        'audio', 'video', or 'unknown'
    """
    print(
        f"[VIDEO_PROCESSOR] INFO: Detecting media type for content_type: {content_type}")

    # Video MIME types commonly sent via WhatsApp
    video_types = [
        'video/mp4',
        'video/3gpp',
        'video/quicktime',
        'video/x-msvideo',  # AVI
        'video/webm',
        'video/ogg'
    ]

    # Audio MIME types
    audio_types = [
        'audio/ogg',
        'audio/mpeg',
        'audio/mp3',
        'audio/wav',
        'audio/webm',
        'audio/aac',
        'audio/x-wav'
    ]

    if content_type in video_types:
        print(f"[VIDEO_PROCESSOR] INFO: Detected as VIDEO based on content type")
        return 'video'
    elif content_type in audio_types:
        print(f"[VIDEO_PROCESSOR] INFO: Detected as AUDIO based on content type")
        return 'audio'

    # Fallback: check file extension if available
    if file_path:
        extension = Path(file_path).suffix.lower()
        video_extensions = ['.mp4', '.3gp', '.mov', '.avi', '.webm', '.ogv']
        audio_extensions = ['.ogg', '.mp3', '.wav', '.aac', '.m4a']

        if extension in video_extensions:
            print(
                f"[VIDEO_PROCESSOR] INFO: Detected as VIDEO based on file extension: {extension}")
            return 'video'
        elif extension in audio_extensions:
            print(
                f"[VIDEO_PROCESSOR] INFO: Detected as AUDIO based on file extension: {extension}")
            return 'audio'

    print(f"[VIDEO_PROCESSOR] WARNING: Could not determine media type, defaulting to 'unknown'")
    return 'unknown'


def extract_audio_from_video(video_file_path: str) -> Optional[str]:
    """
    Extract audio from video file and save as temporary audio file

    Args:
        video_file_path: Path to the video file

    Returns:
        Path to extracted audio file or None if failed
    """
    print(f"[VIDEO_PROCESSOR] INFO: === Starting audio extraction from video ===")
    print(f"[VIDEO_PROCESSOR] INFO: Input video file: {video_file_path}")

    try:
        # Verify input file exists
        if not os.path.exists(video_file_path):
            print(
                f"[VIDEO_PROCESSOR] ERROR: Video file not found: {video_file_path}")
            return None

        # Get file size for logging
        file_size = os.path.getsize(video_file_path)
        print(f"[VIDEO_PROCESSOR] INFO: Video file size: {file_size} bytes")

        # Load video file with pydub (relies on FFmpeg)
        print(f"[VIDEO_PROCESSOR] INFO: Loading video file with pydub...")
        video_segment = AudioSegment.from_file(video_file_path)

        print(f"[VIDEO_PROCESSOR] INFO: Video loaded successfully!")
        print(
            f"[VIDEO_PROCESSOR] INFO: Duration: {video_segment.duration_seconds:.2f}s")
        print(f"[VIDEO_PROCESSOR] INFO: Channels: {video_segment.channels}")
        print(
            f"[VIDEO_PROCESSOR] INFO: Frame rate: {video_segment.frame_rate}Hz")

        # Create temporary file for extracted audio
        temp_audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.wav',  # Use WAV for maximum compatibility
            prefix=f'extracted_audio_{int(time.time())}_'
        )
        temp_audio_file.close()

        print(
            f"[VIDEO_PROCESSOR] INFO: Exporting audio to: {temp_audio_file.name}")

        # Export audio as WAV (uncompressed for best quality)
        video_segment.export(
            temp_audio_file.name,
            format="wav",
            parameters=["-ac", "1"]  # Force mono audio
        )

        # Verify exported file
        if not os.path.exists(temp_audio_file.name):
            print(f"[VIDEO_PROCESSOR] ERROR: Exported audio file not found")
            return None

        exported_size = os.path.getsize(temp_audio_file.name)
        print(f"[VIDEO_PROCESSOR] INFO: Audio extraction successful!")
        print(
            f"[VIDEO_PROCESSOR] INFO: Exported audio size: {exported_size} bytes")
        print(
            f"[VIDEO_PROCESSOR] INFO: Exported audio path: {temp_audio_file.name}")

        return temp_audio_file.name

    except Exception as e:
        print(
            f"[VIDEO_PROCESSOR] ERROR: Failed to extract audio from video: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_media_info(file_path: str) -> dict:
    """
    Get basic information about a media file

    Args:
        file_path: Path to the media file

    Returns:
        Dictionary with media information
    """
    try:
        audio_segment = AudioSegment.from_file(file_path)
        return {
            "duration_seconds": audio_segment.duration_seconds,
            "channels": audio_segment.channels,
            "frame_rate": audio_segment.frame_rate,
            "sample_width": audio_segment.sample_width,
            "file_size_bytes": os.path.getsize(file_path)
        }
    except Exception as e:
        print(f"[VIDEO_PROCESSOR] ERROR: Could not get media info: {e}")
        return {}


def cleanup_temp_file(file_path: str) -> bool:
    """
    Clean up temporary file

    Args:
        file_path: Path to file to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(
                f"[VIDEO_PROCESSOR] INFO: Cleaned up temporary file: {file_path}")
            return True
        return False
    except Exception as e:
        print(
            f"[VIDEO_PROCESSOR] ERROR: Failed to cleanup file {file_path}: {e}")
        return False
