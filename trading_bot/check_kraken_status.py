#!/usr/bin/env python3
"""
Kraken Account Status Checker

This script checks the current status of the Kraken account,
including balances, open positions, and recent trades,
then reports this information to Serafina.
"""
import os
import ccxt
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to allow importing serafina_integration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.serafina_integration import (
    SerafinaClient,
    report_trade_to_serafina,
    get_serafina_client
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kraken_status_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KrakenStatusChecker')

def get_kraken_exchange():
    """Initialize and return a configured Kraken exchange instance."""
    load_dotenv('.env.live')
    return ccxt.kraken({
        'apiKey': os.getenv('KRAKEN_API_KEY'),
        'secret': os.getenv('KRAKEN_API_SECRET'),
        'enableRateLimit': True,
        'options': {
            'createMarketBuyOrderRequiresPrice': False,
        },
    })

def get_account_balance(kraken) -> Dict[str, float]:
    """Get the current account balance.
    
    Returns:
        Dict[str, float]: Dictionary of currency balances
    """
    try:
        balance = kraken.fetch_balance()
        return {
            currency: {
                'free': float(data.get('free', 0)),
                'used': float(data.get('used', 0)),
                'total': float(data.get('total', 0))
            }
            for currency, data in balance['total'].items()
            if float(data.get('total', 0)) > 0  # Only include non-zero balances
        }
    except Exception as e:
        logger.error(f"Error fetching balance: {str(e)}")
        return {}

def get_open_orders(kraken, symbol: str = None) -> List[Dict[str, Any]]:
    """Get open orders.
    
    Args:
        kraken: Kraken exchange instance
        symbol: Optional trading pair symbol to filter by
        
    Returns:
        List[Dict]: List of open orders
    """
    try:
        return kraken.fetch_open_orders(symbol)
    except Exception as e:
        logger.error(f"Error fetching open orders: {str(e)}")
        return []

def get_recent_trades(kraken, since: int = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent trades.
    
    Args:
        kraken: Kraken exchange instance
        since: Timestamp in milliseconds to fetch trades since
        limit: Maximum number of trades to return
        
    Returns:
        List[Dict]: List of recent trades
    """
    try:
        if since is None:
            # Default to last 24 hours
            since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        
        # Get all markets and iterate through them to find trades
        all_trades = []
        for symbol in kraken.markets:
            try:
                trades = kraken.fetch_my_trades(symbol=symbol, since=since, limit=limit)
                all_trades.extend(trades)
            except Exception as e:
                # Skip symbols that don't support fetching trades
                continue
        
        # Sort by timestamp (newest first)
        return sorted(all_trades, key=lambda x: x['timestamp'], reverse=True)[:limit]
    except Exception as e:
        logger.error(f"Error fetching recent trades: {str(e)}")
        return []

def report_status_to_serafina(
    kraken,
    serafina: SerafinaClient,
    report_trades: bool = True
) -> Dict[str, Any]:
    """Report account status to Serafina.
    
    Args:
        kraken: Kraken exchange instance
        serafina: Serafina client instance
        report_trades: Whether to report recent trades to Serafina
        
    Returns:
        Dict: Status report
    """
    status = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'balance': {},
        'open_orders': [],
        'recent_trades': []
    }
    
    try:
        # Get account balance
        balance = get_account_balance(kraken)
        status['balance'] = balance
        
        # Get open orders
        open_orders = get_open_orders(kraken)
        status['open_orders'] = [{
            'id': o['id'],
            'symbol': o['symbol'],
            'type': o['type'],
            'side': o['side'],
            'price': float(o['price']),
            'amount': float(o['amount']),
            'filled': float(o['filled']),
            'status': o['status']
        } for o in open_orders]
        
        # Get recent trades
        if report_trades:
            recent_trades = get_recent_trades(kraken, limit=20)
            status['recent_trades'] = [{
                'id': t['id'],
                'order': t['order'],
                'symbol': t['symbol'],
                'side': t['side'],
                'price': float(t['price']),
                'amount': float(t['amount']),
                'cost': float(t['cost']),
                'fee': float(t['fee']['cost']) if t.get('fee') else 0.0,
                'fee_currency': t['fee']['currency'] if t.get('fee') else '',
                'timestamp': t['datetime']
            } for t in recent_trades]
            
            # Report trades to Serafina
            for trade in recent_trades:
                try:
                    report_trade_to_serafina(
                        symbol=trade['symbol'],
                        side=trade['side'],
                        amount=float(trade['amount']),
                        price=float(trade['price']),
                        order_id=trade['order'],
                        exchange='kraken',
                        notes=f"Trade ID: {trade['id']}"
                    )
                except Exception as e:
                    logger.error(f"Error reporting trade {trade['id']} to Serafina: {str(e)}")
        
        # Calculate P&L (simplified - would need more sophisticated tracking)
        total_balance = sum(data['total'] for data in balance.values())
        status['total_balance'] = total_balance
        
        # Here you would typically compare against a baseline or previous balance
        # For now, we'll just return the current status
        
        return status
        
    except Exception as e:
        logger.error(f"Error generating status report: {str(e)}")
        return status

def main():
    """Main function to check Kraken status and report to Serafina."""
    logger.info("Starting Kraken status check...")
    
    try:
        # Initialize clients
        kraken = get_kraken_exchange()
        serafina = get_serafina_client()
        
        # Generate and log status report
        status = report_status_to_serafina(kraken, serafina)
        
        # Log summary
        logger.info("\n=== Kraken Account Status ===")
        logger.info(f"Timestamp: {status['timestamp']}")
        logger.info("\nBalances:")
        for currency, data in status['balance'].items():
            logger.info(f"  {currency}: {data['total']} (Free: {data['free']}, Used: {data['used']})")
        
        if status['open_orders']:
            logger.info("\nOpen Orders:")
            for order in status['open_orders']:
                logger.info(f"  {order['side'].upper()} {order['amount']} {order['symbol']} @ {order['price']} (Filled: {order['filled']})")
        else:
            logger.info("\nNo open orders.")
        
        if status['recent_trades']:
            logger.info("\nRecent Trades:")
            for trade in status['recent_trades']:
                logger.info(f"  {trade['timestamp']} - {trade['side'].upper()} {trade['amount']} {trade['symbol']} @ {trade['price']} = {trade['cost']} {trade['fee_currency']}")
        
        logger.info("\nStatus check completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Kraken status check completed successfully")
    else:
        logger.error("❌ Kraken status check failed")
