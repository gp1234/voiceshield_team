"""
Configuration module for WhatsApp integration
Handles environment variables and Twilio settings
"""
import os
from typing import Optional


class TwilioConfig:
    """Configuration class for Twilio WhatsApp integration"""

    def __init__(self):
        self.account_sid: Optional[str] = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token: Optional[str] = os.getenv('TWILIO_AUTH_TOKEN')
        self.webhook_url: Optional[str] = os.getenv(
            'WEBHOOK_URL', 'http://localhost:8000')

    def is_configured(self) -> bool:
        """Check if all required Twilio credentials are configured"""
        return bool(self.account_sid and self.auth_token)

    def get_missing_vars(self) -> list:
        """Return list of missing environment variables"""
        missing = []
        if not self.account_sid:
            missing.append('TWILIO_ACCOUNT_SID')
        if not self.auth_token:
            missing.append('TWILIO_AUTH_TOKEN')
        return missing


# Global config instance
config = TwilioConfig()
