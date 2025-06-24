"""
VoiceShield WhatsApp Integration Runner
runner for the WhatsApp webhook service
"""
import uvicorn
import sys


def main():
    """
    Main entry point for VoiceShield WhatsApp Integration
    Runs the production webhook service
    """
    print("\n" + "=" * 60)
    print("🤖 VoiceShield WhatsApp Integration")
    print("=" * 60)
    print("\n🎯 Starting production webhook service...")

    # Import the main webhook app
    from .webhook import app
    from .config import config

    # Check Twilio configuration
    if not config.is_configured():
        print("\n❌ ERROR: Twilio credentials are required!")
        print("Missing environment variables:", config.get_missing_vars())
        print("\n📋 To configure:")
        print("1. Create .env file in project root")
        print("2. Add: TWILIO_ACCOUNT_SID=your_sid_here")
        print("3. Add: TWILIO_AUTH_TOKEN=your_token_here")
        print("4. Add: WEBHOOK_URL=http://localhost:8002")
        print("\n🔧 See SETUP_WHATSAPP.md for detailed instructions")
        print("\n❌ Cannot start without proper configuration.")
        return False

    print("✅ Twilio credentials configured!")

    print("\n🌐 Server will start on: http://localhost:8002")
    print("🔗 Webhook endpoint: http://localhost:8002/whatsapp")
    print("📊 Health check: http://localhost:8002/health")
    print("\n🎯 Features:")
    print("  • Text message echo with help")
    print("  • Real audio download from Twilio")
    print("  • Integration with VoiceShield ML API")
    print("  • Real FAKE/REAL detection with confidence")
    print("  • User-friendly formatted responses")
    print("\n⚠️  IMPORTANT:")
    print("  • Make sure VoiceShield API is running on http://localhost:8000")
    print("  • Start the main API first: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("  • Then start this webhook service")
    print("  • Use ngrok to expose webhook: ngrok http 8002")
    print("\n🚀 Starting server...")

    try:
        # Run the webhook server
        uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
        return True
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
