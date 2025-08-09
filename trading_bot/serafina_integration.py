#!/usr/bin/env python3
"""
Serafina Integration for Kraken Trading Bot

This module provides integration with the Serafina bot to report
crypto trading status and P&L information.
"""
import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SerafinaIntegration')

class SerafinaClient:
    """Client for interacting with the Serafina bot API."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the Serafina client.
        
        Args:
            base_url: Base URL of the Serafina API
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv('SERAFINA_API_URL', 'http://localhost:8000')
        self.api_key = api_key or os.getenv('SERAFINA_API_KEY', '')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        })
    
    def send_trade_update(self, trade_data: Dict[str, Any]) -> bool:
        """Send a trade update to Serafina.
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            url = f"{self.base_url}/api/v1/trades"
            response = self.session.post(url, json=trade_data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error sending trade update to Serafina: {str(e)}")
            return False
    
    def get_portfolio_status(self) -> Optional[Dict[str, Any]]:
        """Get the current portfolio status from Serafina.
        
        Returns:
            Optional[Dict]: Portfolio status if successful, None otherwise
        """
        try:
            url = f"{self.base_url}/api/v1/portfolio"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting portfolio status from Serafina: {str(e)}")
            return None

def format_trade_data(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    order_id: str,
    exchange: str = 'kraken',
    notes: str = ''
) -> Dict[str, Any]:
    """Format trade data for Serafina.
    
    Args:
        symbol: Trading pair (e.g., 'SHIB/USD')
        side: 'buy' or 'sell'
        amount: Amount in base currency
        price: Price in quote currency
        order_id: Exchange order ID
        exchange: Exchange name (default: 'kraken')
        notes: Additional notes about the trade
        
    Returns:
        Dict: Formatted trade data for Serafina
    """
    base, quote = symbol.split('/') if '/' in symbol else (symbol, 'USD')
    
    return {
        'exchange': exchange,
        'symbol': symbol,
        'base_currency': base,
        'quote_currency': quote,
        'side': side.lower(),
        'amount': float(amount),
        'price': float(price),
        'cost': float(amount) * float(price),
        'order_id': str(order_id),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'notes': notes
    }

# Singleton instance
_serafina_client = None

def get_serafina_client() -> SerafinaClient:
    """Get or create a singleton instance of SerafinaClient."""
    global _serafina_client
    if _serafina_client is None:
        _serafina_client = SerafinaClient()
    return _serafina_client

def report_trade_to_serafina(
    symbol: str,
    side: str,
    amount: float,
    price: float,
    order_id: str,
    exchange: str = 'kraken',
    notes: str = ''
) -> bool:
    """Report a trade to Serafina.
    
    Args:
        symbol: Trading pair (e.g., 'SHIB/USD')
        side: 'buy' or 'sell'
        amount: Amount in base currency
        price: Price in quote currency
        order_id: Exchange order ID
        exchange: Exchange name (default: 'kraken')
        notes: Additional notes about the trade
        
    Returns:
        bool: True if the report was successful, False otherwise
    """
    client = get_serafina_client()
    trade_data = format_trade_data(
        symbol=symbol,
        side=side,
        amount=amount,
        price=price,
        order_id=order_id,
        exchange=exchange,
        notes=notes
    )
    return client.send_trade_update(trade_data)
