"""
WhatsApp Integration Webhook for VoiceShield
Complete integration: WhatsApp -> Twilio -> VoiceShield API -> ML Analysis -> Response
"""
import time
from fastapi import FastAPI, Form
from fastapi.responses import Response as FastAPIResponse
from twilio.twiml.messaging_response import MessagingResponse
from .config import config
from .utils import (
    download_audio_from_twilio,
    send_audio_to_analysis_api,
    format_analysis_response,
    cleanup_temp_file,
    get_error_message,
    get_help_message,
    send_whatsapp_message
)

app = FastAPI(title="VoiceShield WhatsApp Integration")


@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(default=""),
    NumMedia: str = Form(default="0"),
    MediaUrl0: str = Form(default=""),
    MediaContentType0: str = Form(default="")
):
    """
    Main WhatsApp webhook that processes both text and audio messages.
    The orchestrator now handles the initial confirmation message.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_prefix = f"[WHATSAPP_WEBHOOK - {timestamp}]"

    print(f"\n{log_prefix} INFO: === New WhatsApp Message ===")
    print(f"{log_prefix} INFO: From: {From}")
    print(f"{log_prefix} INFO: Body: '{Body}'")
    print(f"{log_prefix} INFO: NumMedia: {NumMedia}")
    print(f"{log_prefix} INFO: MediaUrl0: {MediaUrl0}")
    print(f"{log_prefix} INFO: MediaContentType0: {MediaContentType0}")

    # Create TwiML response object, which will be returned empty to Twilio
    resp = MessagingResponse()

    # Check if Twilio is configured
    if not config.is_configured():
        missing_vars = config.get_missing_vars()
        error_msg = f"❌ Configuration incomplete. Missing variables: {', '.join(missing_vars)}"
        print(f"{log_prefix} ERROR: {error_msg}")
        resp.message(error_msg)
        return create_twiml_response(str(resp))

    # Handle text messages (help, commands, echo)
    if NumMedia == "0" or not MediaUrl0:
        print(f"{log_prefix} INFO: Text message received")

        body_lower = Body.lower().strip()

        if body_lower in ["help", "ajuda", "?"]:
            help_msg = get_help_message()
            resp.message(help_msg)
            print(f"{log_prefix} INFO: Help message sent")
        else:
            # Echo message with instructions
            response_text = f"""📱 Hello! I'm VoiceShield, your AI voice detector.

{get_help_message()}"""
            resp.message(response_text)
            print(f"{log_prefix} INFO: Echo with instructions sent")

        return create_twiml_response(str(resp))

    # Handle media messages (audio/video) - Real AI Analysis
    from .video_processor import detect_media_type

    media_type = detect_media_type(MediaContentType0, MediaUrl0)
    print(f"{log_prefix} INFO: Media message detected - Type: {media_type}")

    # The initial confirmation message is now sent by the orchestrator.
    # This webhook will only send the final result.
    temp_audio_path = None

    try:
        # Step 1: Download media from Twilio
        print(f"{log_prefix} INFO: Step 1 - Downloading media from Twilio")
        auth = (config.account_sid, config.auth_token)
        temp_audio_path = download_audio_from_twilio(MediaUrl0, auth)

        if not temp_audio_path:
            # If download fails, we send an error message back to the user
            error_msg = get_error_message("download")
            print(f"{log_prefix} ERROR: Failed to download media")
            send_whatsapp_message(to_number=From, message_body=error_msg)
            return create_twiml_response(str(resp))

        print(f"{log_prefix} INFO: Media downloaded successfully: {temp_audio_path}")

        # Step 2: Send media to VoiceShield analysis API (Orchestrator handles video/audio)
        print(f"{log_prefix} INFO: Step 2 - Sending media to VoiceShield API")
        api_response = send_audio_to_analysis_api(
            audio_file_path=temp_audio_path,
            user_number=From,
            media_type=media_type
        )

        if not api_response or 'error' in api_response:
            error_msg = get_error_message("api")
            print(
                f"{log_prefix} ERROR: Failed to get API response. Details: {api_response.get('error', 'N/A')}")
            send_whatsapp_message(to_number=From, message_body=error_msg)
            return create_twiml_response(str(resp))

        print(f"{log_prefix} INFO: VoiceShield AI analysis completed successfully")

        # Step 3: Format and send response with enhanced message for video
        print(f"{log_prefix} INFO: Step 3 - Formatting response")
        formatted_response = format_analysis_response(api_response, media_type)

        # Send result via Twilio API
        send_whatsapp_message(to_number=From, message_body=formatted_response)

        print(f"{log_prefix} INFO: Analysis complete - result sent via Twilio API")

    except Exception as e:
        print(f"{log_prefix} ERROR: Unexpected error during audio processing: {e}")
        import traceback
        traceback.print_exc()

        error_msg = get_error_message("processing")
        send_whatsapp_message(to_number=From, message_body=error_msg)

    finally:
        # Always cleanup temporary files
        if temp_audio_path:
            cleanup_temp_file(temp_audio_path)

        print(f"{log_prefix} INFO: === Request completed ===\n")

    # Return an empty TwiML response to prevent sending a duplicate message.
    # All user-facing messages are now sent via the Twilio REST API.
    return create_twiml_response(str(resp))


def create_twiml_response(twiml_content: str) -> FastAPIResponse:
    """
    Create properly formatted TwiML response for Twilio

    Args:
        twiml_content: TwiML XML content as string

    Returns:
        FastAPI Response with correct headers for Twilio
    """
    return FastAPIResponse(
        content=twiml_content,
        media_type="application/xml",
        headers={
            "Content-Type": "application/xml; charset=utf-8"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "voiceshield_whatsapp_webhook",
        "twilio_configured": config.is_configured(),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "VoiceShield WhatsApp Integration",
        "status": "running",
        "webhook_endpoint": "/whatsapp",
        "health_check": "/health",
        "twilio_configured": config.is_configured()
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting VoiceShield WhatsApp Integration Webhook...")
    print(f"📱 Twilio configured: {config.is_configured()}")
    if not config.is_configured():
        print(f"⚠️  Missing variables: {config.get_missing_vars()}")
    print("🔗 Webhook will be available at: http://localhost:8001/whatsapp")
    print("💡 Use ngrok to expose: ngrok http 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
