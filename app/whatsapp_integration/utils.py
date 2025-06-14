"""
Utilities for WhatsApp integration with Twilio
Handles audio download, API communication, and response formatting
"""
import os
import requests
import tempfile
import time
from typing import Optional, Tuple
from .config import config


def download_audio_from_twilio(media_url: str, auth: Tuple[str, str]) -> Optional[str]:
    """
    Download audio file from Twilio media URL

    Args:
        media_url: Twilio media URL
        auth: Tuple of (account_sid, auth_token) for authentication

    Returns:
        Path to downloaded temporary file or None if failed
    """
    try:
        print(f"[UTILS] INFO: Downloading audio from: {media_url}")

        # Make authenticated request to Twilio
        response = requests.get(media_url, auth=auth, timeout=30)
        response.raise_for_status()

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.ogg',  # WhatsApp typically sends OGG files
            prefix=f'whatsapp_audio_{int(time.time())}_'
        )

        # Write audio content
        temp_file.write(response.content)
        temp_file.close()

        file_size = len(response.content)
        print(
            f"[UTILS] INFO: Audio downloaded successfully. Size: {file_size} bytes, Path: {temp_file.name}")

        return temp_file.name

    except requests.exceptions.RequestException as e:
        print(f"[UTILS] ERROR: Failed to download audio from Twilio: {e}")
        return None
    except Exception as e:
        print(f"[UTILS] ERROR: Unexpected error downloading audio: {e}")
        return None


def send_audio_to_api(audio_file_path: str, api_url: str = "http://localhost:8000/analyze_audio/") -> Optional[dict]:
    """
    Send audio file to VoiceShield analysis API

    Args:
        audio_file_path: Path to audio file
        api_url: URL of VoiceShield analysis API endpoint

    Returns:
        API response dict or None if failed
    """
    try:
        print(f"[UTILS] INFO: Sending audio to VoiceShield API: {api_url}")

        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (os.path.basename(
                audio_file_path), audio_file, 'audio/ogg')}

            response = requests.post(api_url, files=files, timeout=60)
            response.raise_for_status()

            result = response.json()
            print(f"[UTILS] INFO: VoiceShield API response received: {result}")

            return result

    except requests.exceptions.RequestException as e:
        print(f"[UTILS] ERROR: Failed to send audio to VoiceShield API: {e}")
        return None
    except Exception as e:
        print(f"[UTILS] ERROR: Unexpected error sending audio to API: {e}")
        return None


def format_analysis_response(api_response: dict) -> str:
    """
    Format VoiceShield API response into user-friendly WhatsApp message

    Args:
        api_response: Response from VoiceShield analysis API

    Returns:
        Formatted message string in English
    """
    try:
        prediction = api_response.get('prediction', 'UNKNOWN')
        confidence = api_response.get('confidence', 0)
        filename = api_response.get('filename', 'audio')

        # Choose emoji based on prediction
        if prediction == 'REAL':
            emoji = "✅"
            status_msg = "REAL"
        elif prediction == 'FAKE':
            emoji = "⚠️"
            status_msg = "FAKE"
        else:
            emoji = "❓"
            status_msg = "UNKNOWN"

        # Format confidence
        if confidence >= 0:
            confidence_text = f"Confidence: {confidence:.1f}%"
        else:
            confidence_text = "Confidence: N/A"

        # Create formatted message in English
        message = f"""🎤 *Audio Analysis Complete*

{emoji} *Result: {status_msg}*
📊 {confidence_text}

_Analysis powered by VoiceShield AI_"""

        print(f"[UTILS] INFO: Formatted response: {message}")
        return message

    except Exception as e:
        print(f"[UTILS] ERROR: Failed to format response: {e}")
        return "❌ Error processing analysis result."


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file

    Args:
        file_path: Path to temporary file to delete
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            print(f"[UTILS] INFO: Cleaned up temporary file: {file_path}")
    except Exception as e:
        print(f"[UTILS] WARNING: Failed to cleanup temp file {file_path}: {e}")


def get_error_message(error_type: str = "general") -> str:
    """
    Get user-friendly error messages in English

    Args:
        error_type: Type of error (download, api, processing, general)

    Returns:
        User-friendly error message in English
    """
    error_messages = {
        "download": "❌ Error downloading audio. Please try sending again.",
        "api": "❌ Error analyzing audio. Our system may be temporarily unavailable.",
        "processing": "❌ Error processing audio. Please check if the file is a valid audio.",
        "general": "❌ Unexpected error. Please try again in a few moments."
    }

    return error_messages.get(error_type, error_messages["general"])


def get_help_message() -> str:
    """
    Get help message for users in English

    Returns:
        Help message string in English
    """
    return """🤖 *VoiceShield - AI Voice Detector*

📝 *How to use:*
• Send an audio message (voice note)
• Wait a few seconds
• Receive analysis: REAL or FAKE

⚡ *Features:*
• AI-generated voice detection
• Machine Learning based analysis
• Fast and automatic response

❓ *Questions?* Send "help" to see this message again."""
