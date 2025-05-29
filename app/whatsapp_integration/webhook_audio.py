"""
Audio WhatsApp webhook for Phase 2 testing
Implements audio detection and fixed response functionality
"""
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import time
from typing import Optional
from .config import config


def create_audio_webhook_app() -> FastAPI:
    """Create FastAPI app with audio detection webhook for Phase 2 testing"""

    app = FastAPI(title="WhatsApp Audio Webhook - Phase 2")

    @app.get("/")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "ok",
            "phase": "2 - Audio Detection Test",
            "twilio_configured": config.is_configured(),
            "missing_vars": config.get_missing_vars() if not config.is_configured() else [],
            "features": [
                "Text message echo",
                "Audio message detection",
                "Fixed audio response"
            ]
        }

    @app.post("/whatsapp")
    async def whatsapp_webhook(
        request: Request,
        From: str = Form(...),
        To: str = Form(...),
        Body: str = Form(default=""),
        MediaUrl0: Optional[str] = Form(default=None),
        MediaContentType0: Optional[str] = Form(default=None),
        NumMedia: str = Form(default="0")
    ):
        """
        WhatsApp webhook that handles both text and audio messages
        Phase 2: Audio detection with fixed response
        """

        # Log incoming message
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] === WhatsApp Message Received ===")
        print(f"From: {From}")
        print(f"To: {To}")
        print(f"Body: {Body}")
        print(f"NumMedia: {NumMedia}")
        print(f"MediaUrl0: {MediaUrl0}")
        print(f"MediaContentType0: {MediaContentType0}")

        # Create Twilio response
        response = MessagingResponse()

        # Check if message contains media (audio)
        num_media = int(NumMedia) if NumMedia.isdigit() else 0

        if num_media > 0 and MediaUrl0:
            # Audio message detected
            print(f"[{timestamp}] 🎵 AUDIO MESSAGE DETECTED!")
            print(f"[{timestamp}] Media URL: {MediaUrl0}")
            print(f"[{timestamp}] Content Type: {MediaContentType0}")

            # Check if it's an audio file
            if MediaContentType0 and ('audio' in MediaContentType0.lower() or 'ogg' in MediaContentType0.lower()):
                reply_text = (
                    "🎵 *Áudio Recebido!* 🎵\n\n"
                    "✅ *VoiceShield Bot - Fase 2*\n\n"
                    "🔍 *Análise (Demo):*\n"
                    "📊 Resultado: **FAKE**\n"
                    "🎯 Confiança: 78.5%\n"
                    "⚠️ Áudio gerado por IA detectado\n\n"
                    "📝 *Nota:* Esta é uma resposta fixa para teste.\n"
                    f"🕐 Processado em: {timestamp}\n\n"
                    "🚀 Fase 2: Detecção de áudio funcionando!"
                )
                print(f"[{timestamp}] 🤖 Sending AUDIO response (fixed demo)")
            else:
                # Media but not audio
                reply_text = (
                    "📎 *Mídia Recebida*\n\n"
                    f"Tipo: {MediaContentType0}\n\n"
                    "⚠️ Por favor, envie apenas *áudios* para análise.\n\n"
                    "🎵 Formatos aceitos: áudio de voz do WhatsApp"
                )
                print(
                    f"[{timestamp}] 📎 Non-audio media received: {MediaContentType0}")

        elif Body.strip():
            # Text message - keep echo functionality from Phase 1
            if Body.lower().strip() in ['oi', 'olá', 'hello', 'hi']:
                reply_text = (
                    "🤖 *Olá! Sou o VoiceShield Bot*\n\n"
                    "✅ *Fase 2: Detecção de Áudio*\n\n"
                    "📝 *Como usar:*\n"
                    "🎵 Envie um *áudio* para análise\n"
                    "💬 Ou envie texto para teste de eco\n\n"
                    f"Você disse: '{Body}'\n\n"
                    "🚀 Sistema funcionando!"
                )
            elif Body.lower().strip() in ['help', 'ajuda', '?']:
                reply_text = (
                    "🆘 *VoiceShield - Ajuda*\n\n"
                    "🎵 *Para análise de áudio:*\n"
                    "• Grave um áudio no WhatsApp\n"
                    "• Envie para este número\n"
                    "• Receba o resultado da análise\n\n"
                    "💬 *Para teste:*\n"
                    "• Envie qualquer texto\n"
                    "• Receba um eco da mensagem\n\n"
                    "🔧 *Status:* Fase 2 - Teste de Áudio"
                )
            else:
                reply_text = (
                    "📢 *Echo Test - VoiceShield*\n\n"
                    f"Você enviou: '{Body}'\n\n"
                    "✅ Conexão funcionando!\n"
                    "🎵 Envie um *áudio* para testar a detecção\n\n"
                    f"🔄 Timestamp: {timestamp}"
                )
            print(f"[{timestamp}] 💬 Sending TEXT response")

        else:
            # Empty message
            reply_text = (
                "❓ *Mensagem vazia recebida*\n\n"
                "🎵 Envie um *áudio* para análise\n"
                "💬 Ou envie *texto* para teste\n\n"
                "Digite 'ajuda' para instruções"
            )
            print(f"[{timestamp}] ❓ Empty message received")

        response.message(reply_text)

        # Convert to string and log
        twiml_response = str(response)
        print(f"[{timestamp}] TwiML Response: {twiml_response}")
        print(f"[{timestamp}] Reply sent: {reply_text[:50]}...")
        print(f"[{timestamp}] === End of WhatsApp Request ===\n")

        # Return with correct content type for TwiML
        return Response(
            content=twiml_response,
            media_type="application/xml",
            headers={"Content-Type": "text/xml"}
        )

    return app


# Create app instance for this module
audio_app = create_audio_webhook_app()
