"""
Run script for WhatsApp integration phases
Phase 1: Simple echo webhook
Phase 2: Audio detection webhook
Phase 3: Full integration with real audio analysis
"""
import uvicorn
import os
import sys


def run_phase_1():
    """Run Phase 1 - Simple echo webhook"""
    print("🚀 Starting VoiceShield WhatsApp Integration - Phase 1")
    print("📝 Phase 1: Simple Echo Test")
    print("=" * 50)

    # Import the simple webhook app
    from .webhook_simple import simple_app
    from .config import config

    # Check configuration
    if not config.is_configured():
        print("⚠️  WARNING: Twilio credentials not configured!")
        print("Missing environment variables:", config.get_missing_vars())
        print("\n📋 To configure:")
        print("1. Create .env file in project root")
        print("2. Add: TWILIO_ACCOUNT_SID=your_sid_here")
        print("3. Add: TWILIO_AUTH_TOKEN=your_token_here")
        print("4. Add: WEBHOOK_URL=http://localhost:8001")
        print("\n🔧 See SETUP_WHATSAPP.md for detailed instructions")
        print("\n⚡ Starting anyway for testing...")
    else:
        print("✅ Twilio credentials configured!")

    print("\n🌐 Server will start on: http://localhost:8001")
    print("🔗 Webhook endpoint: http://localhost:8001/whatsapp")
    print("📊 Health check: http://localhost:8001/")
    print("\n🚀 Starting server...")

    # Run the server
    uvicorn.run(simple_app, host="0.0.0.0", port=8001, log_level="info")


def run_phase_2():
    """Run Phase 2 - Audio detection webhook"""
    print("🚀 Starting VoiceShield WhatsApp Integration - Phase 2")
    print("🎵 Phase 2: Audio Detection Test")
    print("=" * 50)

    # Import the audio webhook app
    from .webhook_audio import audio_app
    from .config import config

    # Check configuration
    if not config.is_configured():
        print("⚠️  WARNING: Twilio credentials not configured!")
        print("Missing environment variables:", config.get_missing_vars())
        print("\n📋 To configure:")
        print("1. Create .env file in project root")
        print("2. Add: TWILIO_ACCOUNT_SID=your_sid_here")
        print("3. Add: TWILIO_AUTH_TOKEN=your_token_here")
        print("4. Add: WEBHOOK_URL=http://localhost:8001")
        print("\n🔧 See SETUP_WHATSAPP.md for detailed instructions")
        print("\n⚡ Starting anyway for testing...")
    else:
        print("✅ Twilio credentials configured!")

    print("\n🌐 Server will start on: http://localhost:8001")
    print("🔗 Webhook endpoint: http://localhost:8001/whatsapp")
    print("📊 Health check: http://localhost:8001/")
    print("\n🎵 Features:")
    print("  • Text message echo (from Phase 1)")
    print("  • Audio message detection")
    print("  • Fixed demo response for audio")
    print("\n🚀 Starting server...")

    # Run the server
    uvicorn.run(audio_app, host="0.0.0.0", port=8001, log_level="info")


def run_phase_3():
    """Run Phase 3 - Full integration with real audio analysis"""
    print("🚀 Starting VoiceShield WhatsApp Integration - Phase 3")
    print("🎯 Phase 3: Full Integration with Real Audio Analysis")
    print("=" * 60)

    # Import the full webhook app
    from .webhook_full import app
    from .config import config

    # Check configuration
    if not config.is_configured():
        print("❌ ERROR: Twilio credentials REQUIRED for Phase 3!")
        print("Missing environment variables:", config.get_missing_vars())
        print("\n📋 To configure:")
        print("1. Create .env file in project root")
        print("2. Add: TWILIO_ACCOUNT_SID=your_sid_here")
        print("3. Add: TWILIO_AUTH_TOKEN=your_token_here")
        print("4. Add: WEBHOOK_URL=http://localhost:8001")
        print("\n🔧 See SETUP_WHATSAPP.md for detailed instructions")
        print("\n❌ Cannot start Phase 3 without proper configuration.")
        return
    else:
        print("✅ Twilio credentials configured!")

    print("\n🌐 Server will start on: http://localhost:8001")
    print("🔗 Webhook endpoint: http://localhost:8001/whatsapp")
    print("📊 Health check: http://localhost:8001/health")
    print("\n🎯 Features:")
    print("  • Text message echo with help")
    print("  • Real audio download from Twilio")
    print("  • Integration with VoiceShield ML API")
    print("  • Real FAKE/REAL detection with confidence")
    print("  • User-friendly formatted responses")
    print("\n⚠️  IMPORTANT:")
    print("  • Make sure VoiceShield API is running on http://localhost:8000")
    print("  • Start the main API first: python app/main.py")
    print("  • Then start this webhook service")
    print("\n🚀 Starting server...")

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


def show_menu():
    """Show phase selection menu"""
    print("\n" + "=" * 60)
    print("🤖 VoiceShield WhatsApp Integration")
    print("=" * 60)
    print("\n📋 Available Phases:")
    print("1️⃣  Phase 1: Simple Echo Test")
    print("    • Text message echo functionality")
    print("    • Basic Twilio ↔ API communication test")
    print("    • ✅ Status: Completed and tested")
    print()
    print("2️⃣  Phase 2: Audio Detection Test")
    print("    • Audio message detection")
    print("    • Fixed demo response for audio")
    print("    • Text echo still available")
    print("    • ✅ Status: Completed and tested")
    print()
    print("3️⃣  Phase 3: Full Integration")
    print("    • Real audio analysis with ML model")
    print("    • Complete WhatsApp → API → ML → Response flow")
    print("    • 🆕 Status: Ready for testing!")
    print()

    while True:
        try:
            choice = input("🚀 Select phase to run (1, 2, or 3): ").strip()
            if choice == "1":
                return run_phase_1
            elif choice == "2":
                return run_phase_2
            elif choice == "3":
                return run_phase_3
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                continue
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)


def main():
    """Main entry point"""
    try:
        # Check if running as module
        if __name__ == "__main__":
            # Direct execution - show menu
            phase_runner = show_menu()
            phase_runner()
        else:
            # Default to Phase 1 for backward compatibility
            print("🔄 Running default Phase 1...")
            run_phase_1()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
