"""
Full WhatsApp webhook integration with audio analysis
Connects WhatsApp -> Twilio -> Our API -> ML Analysis -> Response
"""
import time
from fastapi import FastAPI, Form
from fastapi.responses import Response as FastAPIResponse
from twilio.twiml.messaging_response import MessagingResponse
from .config import config
from .utils import (
    download_audio_from_twilio,
    send_audio_to_api,
    format_analysis_response,
    cleanup_temp_file,
    get_error_message,
    get_help_message
)

app = FastAPI()


@app.post("/whatsapp")
async def whatsapp_webhook_full(
    From: str = Form(...),
    Body: str = Form(default=""),
    NumMedia: str = Form(default="0"),
    MediaUrl0: str = Form(default=""),
    MediaContentType0: str = Form(default="")
):
    """
    Full WhatsApp webhook that processes audio and returns real analysis
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_prefix = f"[WEBHOOK_FULL - {timestamp}]"

    print(f"\n{log_prefix} INFO: === New WhatsApp Message ===")
    print(f"{log_prefix} INFO: From: {From}")
    print(f"{log_prefix} INFO: Body: '{Body}'")
    print(f"{log_prefix} INFO: NumMedia: {NumMedia}")
    print(f"{log_prefix} INFO: MediaUrl0: {MediaUrl0}")
    print(f"{log_prefix} INFO: MediaContentType0: {MediaContentType0}")

    # Create TwiML response
    resp = MessagingResponse()

    # Check if Twilio is configured
    if not config.is_configured():
        missing_vars = config.get_missing_vars()
        error_msg = f"❌ Configuração incompleta. Variáveis faltando: {', '.join(missing_vars)}"
        print(f"{log_prefix} ERROR: {error_msg}")
        resp.message(error_msg)
        return create_twiml_response(str(resp))

    # Handle text messages (help, commands)
    if NumMedia == "0" or not MediaUrl0:
        print(f"{log_prefix} INFO: Text message received")

        body_lower = Body.lower().strip()

        if body_lower in ["ajuda", "help", "?"]:
            help_msg = get_help_message()
            resp.message(help_msg)
            print(f"{log_prefix} INFO: Help message sent")
        else:
            # Echo message with instructions
            response_text = f"""📱 Mensagem recebida: "{Body}"

{get_help_message()}"""
            resp.message(response_text)
            print(f"{log_prefix} INFO: Echo with instructions sent")

        return create_twiml_response(str(resp))

    # Handle audio messages
    print(f"{log_prefix} INFO: Audio message detected - starting analysis")

    # Send immediate acknowledgment
    resp.message("🎤 Áudio recebido! Analisando... ⏳")

    temp_audio_path = None

    try:
        # Step 1: Download audio from Twilio
        print(f"{log_prefix} INFO: Step 1 - Downloading audio from Twilio")
        auth = (config.account_sid, config.auth_token)
        temp_audio_path = download_audio_from_twilio(MediaUrl0, auth)

        if not temp_audio_path:
            error_msg = get_error_message("download")
            print(f"{log_prefix} ERROR: Failed to download audio")
            resp.message(error_msg)
            return create_twiml_response(str(resp))

        print(f"{log_prefix} INFO: Audio downloaded successfully: {temp_audio_path}")

        # Step 2: Send audio to our analysis API
        print(f"{log_prefix} INFO: Step 2 - Sending audio to analysis API")
        api_response = send_audio_to_api(temp_audio_path)

        if not api_response:
            error_msg = get_error_message("api")
            print(f"{log_prefix} ERROR: Failed to get API response")
            resp.message(error_msg)
            return create_twiml_response(str(resp))

        print(f"{log_prefix} INFO: API analysis completed successfully")

        # Step 3: Format and send response
        print(f"{log_prefix} INFO: Step 3 - Formatting response")
        formatted_response = format_analysis_response(api_response)
        resp.message(formatted_response)

        print(f"{log_prefix} INFO: Analysis complete - response sent")

    except Exception as e:
        print(f"{log_prefix} ERROR: Unexpected error during audio processing: {e}")
        import traceback
        traceback.print_exc()

        error_msg = get_error_message("processing")
        resp.message(error_msg)

    finally:
        # Always cleanup temporary files
        if temp_audio_path:
            cleanup_temp_file(temp_audio_path)

        print(f"{log_prefix} INFO: === Request completed ===\n")

    return create_twiml_response(str(resp))


def create_twiml_response(twiml_content: str) -> FastAPIResponse:
    """
    Create properly formatted TwiML response for Twilio

    Args:
        twiml_content: TwiML XML content as string

    Returns:
        FastAPI Response with correct headers
    """
    return FastAPIResponse(
        content=twiml_content,
        media_type="application/xml",
        headers={
            "Content-Type": "application/xml; charset=utf-8"
        }
    )

# Health check endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "whatsapp_webhook_full",
        "twilio_configured": config.is_configured(),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting WhatsApp Full Integration Webhook...")
    print(f"📱 Twilio configured: {config.is_configured()}")
    if not config.is_configured():
        print(f"⚠️  Missing variables: {config.get_missing_vars()}")
    print("🔗 Webhook will be available at: http://localhost:8001/whatsapp")
    print("💡 Use ngrok to expose: ngrok http 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
