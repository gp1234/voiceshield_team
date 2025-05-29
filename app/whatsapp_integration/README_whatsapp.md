# VoiceShield WhatsApp Integration

## Overview

VoiceShield WhatsApp Integration allows users to send voice messages through WhatsApp and receive real-time AI analysis to determine if the audio is **REAL** or **FAKE** (AI-generated).

## Features

- 🎤 **Real-time Audio Analysis**: Send voice messages and get instant AI-powered analysis
- 🤖 **AI Voice Detection**: Uses advanced Machine Learning to detect AI-generated voices
- 📱 **WhatsApp Integration**: Works directly through WhatsApp using Twilio
- ⚡ **Fast Response**: Analysis typically completes in seconds
- 🔒 **Secure**: Audio files are processed temporarily and automatically deleted

## How It Works

```
[User] → [WhatsApp Voice Message] → [Twilio] → [VoiceShield API] → [ML Analysis] → [WhatsApp Response]
```

1. User sends a voice message to the configured WhatsApp number
2. Twilio receives the message and forwards it to our webhook
3. The webhook downloads the audio and sends it to the VoiceShield API
4. The API analyzes the audio using OpenL3 embeddings and SVM model
5. Results are formatted and sent back to the user via WhatsApp

## Setup Instructions

### Prerequisites

1. **Python Environment**: Conda environment with all dependencies installed
2. **Twilio Account**: Free account at [console.twilio.com](https://console.twilio.com)
3. **ngrok**: For exposing local webhook to the internet

### Configuration

1. **Environment Variables**: Create `.env` file in project root:
```bash
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
WEBHOOK_URL=http://localhost:8001
```

2. **Twilio WhatsApp Sandbox**: 
   - Go to Twilio Console → Messaging → Try it out → Send a WhatsApp message
   - Follow instructions to join the sandbox
   - Configure webhook URL: `https://your-ngrok-url.ngrok.io/whatsapp`

### Running the System

1. **Start the VoiceShield API** (Terminal 1):
```bash
conda activate bts_final_project
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. **Start the WhatsApp Webhook** (Terminal 2):
```bash
conda activate bts_final_project
python -m app.whatsapp_integration.run
```

3. **Expose with ngrok** (Terminal 3):
```bash
ngrok http 8001
```

4. **Update Twilio Webhook URL** with the ngrok URL

## Usage

### Text Messages
- Send any text message to get help and instructions
- Send "help" to see the help message

### Voice Messages
- Send a voice message (voice note) through WhatsApp
- Wait a few seconds for analysis
- Receive result: **REAL** or **FAKE** with confidence percentage

### Example Response
```
🎤 Audio Analysis Complete

✅ Result: REAL
📊 Confidence: 87.3%

Analysis powered by VoiceShield AI
```

## API Endpoints

### WhatsApp Webhook
- **POST** `/whatsapp` - Main webhook for Twilio
- **GET** `/health` - Health check endpoint
- **GET** `/` - Service information

### VoiceShield API
- **POST** `/analyze_audio/` - Audio analysis endpoint
- **GET** `/` - Web interface

## Technical Details

### Audio Processing
- **Supported Formats**: WhatsApp voice messages (typically OGG)
- **Minimum Duration**: 3 seconds (automatically padded if shorter)
- **Sample Rate**: Resampled to 16kHz for analysis
- **Features**: OpenL3 embeddings (512-dimensional)

### Model
- **Algorithm**: Support Vector Machine (SVM)
- **Features**: OpenL3 audio embeddings
- **Training**: Trained on real vs AI-generated voice samples
- **Performance**: Provides confidence scores with predictions

### Security
- Audio files are temporarily stored during processing
- Files are automatically deleted after analysis
- No permanent storage of user audio data

## Troubleshooting

### Common Issues

1. **"Configuration incomplete"**
   - Check `.env` file exists and has correct Twilio credentials
   - Verify environment variables are loaded

2. **"Error downloading audio"**
   - Check Twilio credentials are correct
   - Verify webhook URL is accessible from internet

3. **"Error analyzing audio"**
   - Ensure VoiceShield API is running on port 8000
   - Check API logs for detailed error information

4. **No response from WhatsApp**
   - Verify ngrok is running and webhook URL is updated in Twilio
   - Check webhook logs for incoming requests

### Logs
- **API Logs**: Detailed processing information in terminal running the API
- **Webhook Logs**: Request/response information in webhook terminal
- **Health Check**: Visit `http://localhost:8001/health` for status

## Development

### File Structure
```
app/whatsapp_integration/
├── webhook.py          # Main webhook application
├── utils.py           # Utility functions
├── config.py          # Configuration management
├── run.py             # Production runner
└── __init__.py        # Package initialization
```

### Adding Features
- Modify `webhook.py` for new webhook functionality
- Update `utils.py` for new utility functions
- Extend `config.py` for additional configuration options

## Support

For issues or questions:
1. Check the logs for detailed error information
2. Verify all prerequisites are installed and configured
3. Ensure all services are running and accessible
4. Review this documentation for troubleshooting steps

---

**VoiceShield Team** - AI Voice Detection System 