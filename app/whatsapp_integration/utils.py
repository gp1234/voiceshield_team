"""
Utilities for WhatsApp integration with Twilio
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
    Send audio file to our analysis API

    Args:
        audio_file_path: Path to audio file
        api_url: URL of our analysis API endpoint

    Returns:
        API response dict or None if failed
    """
    try:
        print(f"[UTILS] INFO: Sending audio to API: {api_url}")

        with open(audio_file_path, 'rb') as audio_file:
            files = {'file': (os.path.basename(
                audio_file_path), audio_file, 'audio/ogg')}

            response = requests.post(api_url, files=files, timeout=60)
            response.raise_for_status()

            result = response.json()
            print(f"[UTILS] INFO: API response received: {result}")

            return result

    except requests.exceptions.RequestException as e:
        print(f"[UTILS] ERROR: Failed to send audio to API: {e}")
        return None
    except Exception as e:
        print(f"[UTILS] ERROR: Unexpected error sending audio to API: {e}")
        return None


def format_analysis_response(api_response: dict) -> str:
    """
    Format API response into user-friendly WhatsApp message

    Args:
        api_response: Response from our analysis API

    Returns:
        Formatted message string
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
            confidence_text = f"Confiança: {confidence:.1f}%"
        else:
            confidence_text = "Confiança: N/A"

        # Create formatted message
        message = f"""🎤 *Análise de Áudio Concluída*

{emoji} *Resultado: {status_msg}*
📊 {confidence_text}

_Análise realizada pelo VoiceShield AI_"""

        print(f"[UTILS] INFO: Formatted response: {message}")
        return message

    except Exception as e:
        print(f"[UTILS] ERROR: Failed to format response: {e}")
        return "❌ Erro ao processar resultado da análise."


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
    Get user-friendly error messages

    Args:
        error_type: Type of error (download, api, processing, general)

    Returns:
        User-friendly error message
    """
    error_messages = {
        "download": "❌ Erro ao baixar o áudio. Tente enviar novamente.",
        "api": "❌ Erro na análise do áudio. Nosso sistema pode estar temporariamente indisponível.",
        "processing": "❌ Erro ao processar o áudio. Verifique se o arquivo é um áudio válido.",
        "general": "❌ Erro inesperado. Tente novamente em alguns instantes."
    }

    return error_messages.get(error_type, error_messages["general"])


def get_help_message() -> str:
    """
    Get help message for users

    Returns:
        Help message string
    """
    return """🤖 *VoiceShield - Detector de Áudio IA*

📝 *Como usar:*
• Envie um áudio (nota de voz)
• Aguarde alguns segundos
• Receba a análise: REAL ou FAKE

⚡ *Funcionalidades:*
• Detecção de vozes geradas por IA
• Análise baseada em Machine Learning
• Resposta rápida e automática

❓ *Dúvidas?* Envie "ajuda" para ver esta mensagem novamente."""
