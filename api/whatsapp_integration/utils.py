"""
Utilities for WhatsApp integration with Twilio
Handles audio download, API communication, and response formatting
"""
import os
import requests
import tempfile
import time
from typing import Optional, Tuple
from twilio.rest import Client
from .config import config

# Configuration
ANALYSIS_API_URL = "http://localhost:8001/analyze_audio/"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox Number


def send_whatsapp_message(to_number: str, message_body: str):
    """
    Sends a WhatsApp message using the Twilio API.

    Args:
        to_number: The recipient's WhatsApp number (e.g., 'whatsapp:+15551234567').
        message_body: The text of the message to send.
    """
    if not config.is_configured():
        print("[UTILS] ERROR: Twilio is not configured. Cannot send message.")
        return

    try:
        client = Client(config.account_sid, config.auth_token)
        print(
            f"[UTILS] INFO: Sending message to {to_number}: '{message_body[:30]}...'")

        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=to_number
        )
        print(f"[UTILS] INFO: Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"[UTILS] ERROR: Failed to send WhatsApp message: {e}")


def download_audio_from_twilio(media_url: str, auth: Tuple[str, str]) -> Optional[str]:
    """
    Download media file (audio or video) from Twilio media URL

    Args:
        media_url: Twilio media URL
        auth: Tuple of (account_sid, auth_token) for authentication

    Returns:
        Path to downloaded temporary file or None if failed
    """
    try:
        print(f"[UTILS] INFO: Downloading media from: {media_url}")

        # Make authenticated request to Twilio
        response = requests.get(media_url, auth=auth, timeout=30)
        response.raise_for_status()

        # Determine file extension based on content type
        content_type = response.headers.get('content-type', '').lower()
        if 'video' in content_type:
            if 'mp4' in content_type:
                suffix = '.mp4'
            elif '3gpp' in content_type:
                suffix = '.3gp'
            else:
                suffix = '.mp4'  # Default for videos
            prefix = f'whatsapp_video_{int(time.time())}_'
        else:
            # Default to audio handling
            suffix = '.ogg'  # WhatsApp typically sends OGG files for audio
            prefix = f'whatsapp_audio_{int(time.time())}_'

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix=prefix
        )

        # Write media content
        temp_file.write(response.content)
        temp_file.close()

        file_size = len(response.content)
        print(
            f"[UTILS] INFO: Media downloaded successfully. Size: {file_size} bytes, Path: {temp_file.name}")
        print(
            f"[UTILS] INFO: Content type: {content_type}, File extension: {suffix}")

        return temp_file.name

    except requests.exceptions.RequestException as e:
        print(f"[UTILS] ERROR: Failed to download media from Twilio: {e}")
        return None
    except Exception as e:
        print(f"[UTILS] ERROR: Unexpected error downloading media: {e}")
        return None


def send_audio_to_analysis_api(audio_file_path: str, user_number: str, media_type: str):
    """
    Send the audio file to the analysis API with user and media info.
    """
    print(f"[UTILS] INFO: === Sending audio to analysis API ===")
    print(f"[UTILS] INFO: Audio file path: {audio_file_path}")
    print(
        f"[UTILS] INFO: User number: {user_number}, Media type: {media_type}")

    try:
        # Check if file exists and get size
        if not os.path.exists(audio_file_path):
            print(
                f"[UTILS] ERROR: Audio file does not exist: {audio_file_path}")
            return {"error": "Audio file not found"}

        file_size = os.path.getsize(audio_file_path)
        print(f"[UTILS] INFO: Audio file size: {file_size} bytes")

        with open(audio_file_path, 'rb') as audio_file:
            print(f"[UTILS] INFO: Opening file for reading...")

            # Prepare the files and data dictionary for requests
            files = {'file': (os.path.basename(
                audio_file_path), audio_file, 'application/octet-stream')}
            data = {'user_number': user_number, 'media_type': media_type}

            print(f"[UTILS] INFO: Prepared files and data payload")
            print(f"[UTILS] INFO: Target URL: {ANALYSIS_API_URL}")

            print(f"[UTILS] INFO: Making POST request...")
            response = requests.post(
                ANALYSIS_API_URL, files=files, data=data, timeout=60)

            print(f"[UTILS] INFO: Response received!")
            print(f"[UTILS] INFO: Status code: {response.status_code}")
            print(
                f"[UTILS] INFO: Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                print(f"[UTILS] INFO: Response JSON parsed successfully")
                print(
                    f"[UTILS] INFO: Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                return result
            else:
                print(f"[UTILS] ERROR: API returned non-200 status")
                print(f"[UTILS] ERROR: Response text: {response.text}")
                return {"error": f"API error: {response.status_code} - {response.text}"}

    except requests.exceptions.Timeout:
        print(f"[UTILS] ERROR: Request timeout")
        return {"error": "Request timeout"}
    except requests.exceptions.RequestException as e:
        print(f"[UTILS] ERROR: Request exception: {e}")
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        print(f"[UTILS] ERROR: Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Unexpected error: {str(e)}"}


def format_analysis_response(api_response: dict, media_type: str = 'audio') -> str:
    """
    Format VoiceShield API response into user-friendly WhatsApp message.
    Handles both simple (old) and orchestrated (new) API responses.

    Args:
        api_response: Response from the VoiceShield API
        media_type: Type of media processed ('audio', 'video', 'unknown')
    """
    try:
        # Check for new orchestrated response format
        if 'final_prediction' in api_response:
            final_prediction = api_response.get('final_prediction')
            summary = api_response.get('analysis_summary', {})
            total_chunks = summary.get('total_chunks', 0)

            if final_prediction == 'REAL':
                emoji = "✅"
                status_msg = "Result: REAL"
                details = f"Analyzed {total_chunks} segments of the audio, all appear to be authentic."
            elif final_prediction == 'FAKE':
                emoji = "⚠️"
                status_msg = "Result: FAKE"
                details = f"Analyzed {total_chunks} segments of the audio, all appear to be AI-generated."
            elif final_prediction == 'MIXED':
                emoji = "🚨"
                status_msg = "Result: MIXED"
                fake_segments_list = api_response.get(
                    'suspicious_segments', [])
                segments_str = ", ".join(
                    [f"{s['start']:.1f}s-{s['end']:.1f}s" for s in fake_segments_list])
                details = f"Suspicious segments detected at: {segments_str}"
            else:
                emoji = "❓"
                status_msg = "Result: UNKNOWN"
                details = "Could not determine the nature of the audio."

            # Customize header based on media type
            if media_type == 'video':
                header = "🎥 *Video Audio Analysis Complete*"
            else:
                header = "🎤 *Audio Analysis Complete*"

            message = f"""{header}

{emoji} *{status_msg}*
{details}

_Analysis powered by VoiceShield AI_"""

        # Fallback to old, simple response format
        else:
            prediction = api_response.get('prediction', 'UNKNOWN')
            confidence = api_response.get('confidence', 0)

            if prediction == 'REAL':
                emoji = "✅"
                status_msg = "Result: REAL"
            elif prediction == 'FAKE':
                emoji = "⚠️"
                status_msg = "Result: FAKE"
            else:
                emoji = "❓"
                status_msg = "Result: UNKNOWN"

            if confidence >= 0:
                confidence_text = f"Confidence: {confidence:.1f}%"
            else:
                confidence_text = "Confidence: N/A"

            # Customize header based on media type
            if media_type == 'video':
                header = "🎥 *Video Audio Analysis Complete*"
            else:
                header = "🎤 *Audio Analysis Complete*"

            message = f"""{header}

{emoji} *{status_msg}*
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
        "download": "❌ Error downloading media file. Please try sending again.",
        "api": "❌ Error analyzing media. Our system may be temporarily unavailable.",
        "processing": "❌ Error processing media file. Please check if the file is valid audio or video.",
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
• Send an audio message (voice note) 🎤
• Send a video message 🎥
• Wait a few seconds
• Receive analysis: REAL or FAKE

⚡ *Features:*
• AI-generated voice detection
• Video audio extraction
• Machine Learning based analysis
• Fast and automatic response

❓ *Questions?* Send "help" to see this message again."""
