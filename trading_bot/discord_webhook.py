import requests
import json
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiscordWebhook:
    """Handles sending notifications to Discord via webhook."""
    
    def __init__(self, webhook_url: str):
        """Initialize with Discord webhook URL."""
        self.webhook_url = webhook_url
        self.headers = {'Content-Type': 'application/json'}
    
    def _send_payload(self, payload: Dict[str, Any]) -> bool:
        """Send payload to Discord webhook."""
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers=self.headers
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    def send_trade_opened(self, pair: str, direction: str, entry_price: float, 
                         stop_loss: float, take_profit: float, units: int) -> bool:
        """Send notification for a new trade."""
        color = 0x00ff00 if direction.lower() == 'buy' else 0xff0000
        direction_emoji = "🟢" if direction.lower() == 'buy' else "🔴"
        
        embed = {
            "title": f"{direction_emoji} {direction.upper()} {pair} {direction_emoji}",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"{entry_price:.5f}", "inline": True},
                {"name": "Stop Loss", "value": f"{stop_loss:.5f}", "inline": True},
                {"name": "Take Profit", "value": f"{take_profit:.5f}", "inline": True},
                {"name": "Units", "value": f"{units:,}", "inline": True}
            ],
            "footer": {
                "text": "OANDA Trading Bot"
            },
            "timestamp": None  # Will be set to current time by Discord
        }
        
        payload = {
            "embeds": [embed],
            "username": "AthenaMyst Trading Bot",
            "avatar_url": "https://i.imgur.com/4M34hi2.png"
        }
        
        return self._send_payload(payload)
    
    def send_trade_closed(self, pair: str, pnl: float, pnl_percent: float, 
                         reason: str = "Take Profit") -> bool:
        """Send notification for a closed trade."""
        color = 0x00ff00 if pnl >= 0 else 0xff0000
        result = "PROFIT" if pnl >= 0 else "LOSS"
        
        embed = {
            "title": f"🏁 {result} - {pair} Closed",
            "color": color,
            "fields": [
                {"name": "P&L", "value": f"${abs(pnl):.2f}", "inline": True},
                {"name": "P&L %", "value": f"{abs(pnl_percent):.2f}%", "inline": True},
                {"name": "Reason", "value": reason, "inline": True}
            ],
            "footer": {"text": "OANDA Trading Bot"},
            "timestamp": None
        }
        
        payload = {
            "embeds": [embed],
            "username": "AthenaMyst Trading Bot",
            "avatar_url": "https://i.imgur.com/4M34hi2.png"
        }
        
        return self._send_payload(payload)
    
    def send_error(self, error_message: str) -> bool:
        """Send error notification."""
        embed = {
            "title": "❌ Trading Bot Error",
            "description": f"```{error_message[:2000]}```",
            "color": 0xff0000,
            "footer": {"text": "OANDA Trading Bot"},
            "timestamp": None
        }
        
        payload = {
            "embeds": [embed],
            "username": "AthenaMyst Trading Bot",
            "avatar_url": "https://i.imgur.com/4M34hi2.png"
        }
        
        return self._send_payload(payload)
    
    def send_status_update(self, message: str, color: int = 0x3498db) -> bool:
        """Send a status update message."""
        embed = {
            "title": "ℹ️ Status Update",
            "description": message,
            "color": color,
            "footer": {"text": "OANDA Trading Bot"},
            "timestamp": None
        }
        
        payload = {
            "embeds": [embed],
            "username": "AthenaMyst Trading Bot",
            "avatar_url": "https://i.imgur.com/4M34hi2.png"
        }
        
        return self._send_payload(payload)
