"""
Test script for position sizing logic.
"""
import asyncio
import os
from oanda_engine import OandaTradingEngine
from dotenv import load_dotenv

async def test_position_sizing(account_balance, entry_price, stop_loss_price, pair='EUR_USD'):
    """Test the position sizing with different account balances."""
    # Create a test config
    test_config = {
        'OANDA_ACCOUNT_ID': 'test',
        'OANDA_API_KEY': 'test',
        'OANDA_ACCOUNT_TYPE': 'practice',
        'TRADING_PAIR': pair,
        'ACCOUNT_CURRENCY': 'USD',
        'RISK_PERCENT': '10.0',
        'STOP_LOSS_PIPS': '20',
        'TAKE_PROFIT_PIPS': '40'
    }
    
    # Save to a test config file
    with open('.env.test', 'w') as f:
        for key, value in test_config.items():
            f.write(f"{key}={value}\n")
    
    # Initialize the trading engine
    engine = OandaTradingEngine(config_file='.env.test')
    
    # Override the account balance and margin for testing
    engine.account_balance = account_balance
    engine.margin_available = account_balance * 50  # Assuming 50:1 leverage
    
    # Test the position size calculation
    print(f"\n{'='*50}")
    print(f"TEST CASE - {pair}")
    print(f"{'='*50}")
    print(f"Account Balance:    ${account_balance:,.2f}")
    print(f"Entry Price:        {entry_price:.5f}")
    print(f"Stop Loss Price:    {stop_loss_price:.5f}")
    print(f"Stop Loss Pips:     {abs(entry_price - stop_loss_price) * 10000:.1f}")
    
    # Calculate position value
    units = await engine.calculate_position_size(entry_price, stop_loss_price)
    position_value = (units * entry_price) / 100000  # Position value in lots
    
    print(f"\nCalculated Position:")
    print(f"- Units:            {int(units):,}")
    print(f"- Position Value:   {position_value:,.2f} lots")
    print(f"- Margin Required:  ${(units * entry_price) / 50:,.2f} (at 50:1)")
    
    # Calculate position size as a percentage of account
    if account_balance > 0:
        position_pct = (position_value * 1000) / account_balance * 100
        print(f"- Position Size:    {position_pct:.1f}% of account")
    
    # Clean up
    if os.path.exists('.env.test'):
        os.remove('.env.test')
    
    return units

async def main():
    # Test with different account balances
    test_cases = [
        # (account_balance, entry_price, stop_loss_price, pair)
        (1000, 1.0800, 1.0780, 'EUR_USD'),
        (5000, 1.0800, 1.0780, 'EUR_USD'),
        (10000, 1.0800, 1.0780, 'EUR_USD'),
        (50000, 1.0800, 1.0780, 'EUR_USD'),
        (1000, 150.50, 150.30, 'USD_JPY'),
        (10000, 150.50, 150.30, 'USD_JPY'),
    ]
    
    print("Testing Position Sizing Logic")
    print("============================")
    
    for case in test_cases:
        await test_position_sizing(*case)

if __name__ == "__main__":
    asyncio.run(main())
