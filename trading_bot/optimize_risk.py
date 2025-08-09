#!/usr/bin/env python3
"""
Optimize risk parameters based on account balance and performance.
"""
import os
import ccxt
import json
import logging
from dotenv import load_dotenv
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_optimizer.log')
    ]
)
logger = logging.getLogger('RiskOptimizer')

def get_account_balance(exchange: ccxt.Exchange) -> Tuple[float, Dict]:
    """Fetch and return the account balance in USD."""
    try:
        # Fetch all balances
        balance = exchange.fetch_balance()
        
        # Get total balance in USD
        usd_balance = balance.get('USD', {}).get('free', 0.0)
        
        # If we have crypto, we need to convert to USD
        if usd_balance == 0:
            # Try to find a USD stablecoin
            for currency in ['USDT', 'USDC', 'DAI', 'BUSD']:
                if currency in balance and balance[currency]['free'] > 0:
                    usd_balance = balance[currency]['free']
                    break
        
        return float(usd_balance), balance
    
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        return 0.0, {}

def optimize_risk_parameters(balance: float) -> Dict:
    """Calculate optimal risk parameters based on account balance."""
    # Base parameters (can be adjusted based on backtesting)
    base_risk = 0.02  # 2% base risk per trade
    max_risk = 0.05   # 5% max risk per trade
    
    # Adjust risk based on account size (larger accounts can take more risk)
    if balance < 1000:
        risk_per_trade = min(base_risk * 1.5, max_risk)  # Higher risk for smaller accounts
        position_size_multiplier = 1.5
    elif balance < 5000:
        risk_per_trade = min(base_risk * 1.25, max_risk)
        position_size_multiplier = 1.25
    else:
        risk_per_trade = base_risk
        position_size_multiplier = 1.0
    
    # Adjust position sizing based on performance (simplified)
    # In a real implementation, you'd want to look at recent trade history
    
    return {
        'risk_per_trade': risk_per_trade,
        'position_size_multiplier': position_size_multiplier,
        'stop_loss_pct': 0.002,  # Tighter stop loss for higher risk
        'take_profit_pct': 0.008,  # 1:4 risk-reward ratio
        'max_open_trades': min(25, max(5, int(balance / 1000))),  # Scale with account size
    }

def update_env_file(params: Dict):
    """Update the .env file with new parameters."""
    env_file = '.env'
    
    # Read current .env file
    try:
        with open(env_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error("Could not find .env file")
        return
    
    # Update parameters
    updated_lines = []
    for line in lines:
        if line.startswith('RISK_PER_TRADE'):
            updated_lines.append(f'RISK_PER_TRADE={params["risk_per_trade"]:.4f}  # Auto-adjusted based on balance\n')
        elif line.startswith('STOP_LOSS_PERCENT'):
            updated_lines.append(f'STOP_LOSS_PERCENT={params["stop_loss_pct"]:.4f}  # Auto-adjusted for risk\n')
        elif line.startswith('TAKE_PROFIT_PERCENT'):
            updated_lines.append(f'TAKE_PROFIT_PERCENT={params["take_profit_pct"]:.4f}  # Auto-adjusted for risk\n')
        elif line.startswith('MAX_OPEN_TRADES'):
            updated_lines.append(f'MAX_OPEN_TRADES={params["max_open_trades"]}  # Auto-adjusted based on balance\n')
        else:
            updated_lines.append(line)
    
    # Write back to .env file
    try:
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        logger.info("Updated .env file with optimized parameters")
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")

def main():
    """Main function to optimize risk parameters."""
    # Load environment variables
    load_dotenv()
    
    # Initialize exchange
    exchange = ccxt.kraken({
        'apiKey': os.getenv('KRAKEN_API_KEY'),
        'secret': os.getenv('KRAKEN_API_SECRET'),
        'enableRateLimit': True,
    })
    
    try:
        # Get account balance
        usd_balance, full_balance = get_account_balance(exchange)
        logger.info(f"Current USD Balance: ${usd_balance:.2f}")
        
        # Optimize risk parameters
        params = optimize_risk_parameters(usd_balance)
        logger.info("Optimized Parameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Update .env file
        update_env_file(params)
        
        return {
            'status': 'success',
            'balance': usd_balance,
            'parameters': params
        }
        
    except Exception as e:
        logger.error(f"Error in risk optimization: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == "__main__":
    result = main()
    print("\nRisk Optimization Complete:")
    print(json.dumps(result, indent=2))
