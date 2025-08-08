import asyncio
import os
from dotenv import load_dotenv
from oanda_engine import OandaTradingEngine

async def test_trading():
    # Load environment variables
    load_dotenv('.env.oanda')
    
    # Initialize the trading engine
    engine = OandaTradingEngine()
    
    try:
        # Test account connection
        print("Testing OANDA connection...")
        balance = await engine.get_account_balance()
        print(f"Account balance: {balance} {engine.account_currency}")
        
        # Get current price
        price = await engine.get_current_price()
        print(f"Current {engine.trading_pair} price: {price}")
        
        # Test position sizing
        stop_loss = price * 0.999  # 0.1% below current price
        position_size = await engine.calculate_position_size(price, stop_loss)
        print(f"Calculated position size: {position_size} units")
        
        # Test placing a small trade (will not execute in live mode)
        print("\nTesting trade execution (will not execute in live mode)...")
        trade = await engine.place_trade('buy', reason='Test trade')
        if trade:
            print(f"Trade executed successfully! Trade ID: {trade.get('id')}")
        else:
            print("Trade execution failed. Check logs for details.")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    await engine.close()

if __name__ == "__main__":
    asyncio.run(test_trading())
