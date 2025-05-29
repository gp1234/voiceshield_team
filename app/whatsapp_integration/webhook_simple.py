"""
Simple WhatsApp webhook for Phase 1 testing
Implements basic echo functionality to validate Twilio <-> API communication
"""
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
import time
from .config import config


def create_simple_webhook_app() -> FastAPI:
    """Create FastAPI app with simple WhatsApp webhook for testing"""

    app = FastAPI(title="WhatsApp Simple Webhook - Phase 1")

    @app.get("/")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "ok",
            "phase": "1 - Simple Echo Test",
            "twilio_configured": config.is_configured(),
            "missing_vars": config.get_missing_vars() if not config.is_configured() else []
        }

    @app.post("/whatsapp")
    async def whatsapp_webhook(
        request: Request,
        From: str = Form(...),
        To: str = Form(...),
        Body: str = Form(...)
    ):
        """
        Simple WhatsApp webhook that echoes received messages
        Phase 1: Basic text message echo functionality
        """

        # Log incoming message
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{timestamp}] === WhatsApp Message Received ===")
        print(f"From: {From}")
        print(f"To: {To}")
        print(f"Body: {Body}")

        # Create Twilio response
        response = MessagingResponse()

        # Simple echo logic
        if Body.lower().strip() in ['oi', 'olá', 'hello', 'hi']:
            reply_text = f"🤖 Olá! Sou o VoiceShield Bot.\n\n✅ Teste de conexão funcionando!\n\nVocê disse: '{Body}'\n\n📝 Fase 1: Echo Test ativo"
        else:
            reply_text = f"📢 Echo Test - VoiceShield\n\nVocê enviou: '{Body}'\n\n✅ Conexão Twilio ↔ API funcionando!\n\n🔄 Timestamp: {timestamp}"

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
simple_app = create_simple_webhook_app()
