import asyncio
import websockets
import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import pandas as pd
from trading_engine import TradingEngine
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('divina_scalper.log')
    ]
)
logger = logging.getLogger('DivinaScalper')

class DivinaScalper:
    def __init__(self):
        load_dotenv()
        self.trading_engine = TradingEngine()
        self.active_trades = {}
        self.websocket = None
        self.app = FastAPI(title="Divina Scalping Bot")
        self.setup_routes()
        
        # Trading parameters
        self.trading_pair = os.getenv('TRADING_PAIR', 'BTC/USDT')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        self.amount = float(os.getenv('TRADE_AMOUNT', 100))
        self.max_open_trades = int(os.getenv('MAX_OPEN_TRADES', 3))
        self.websocket_url = os.getenv('WEBSOCKET_URL', 'wss://stream.binance.com:9443/ws')
        
        # Initialize price data
        self.price_history = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.last_price = None
        
        logger.info(f"Divina Scalper initialized for {self.trading_pair} on {self.timeframe} timeframe")
    
    def setup_routes(self):
        @self.app.post("/message")
        async def receive_message(request: Request):
            try:
                data = await request.json()
                msg = data.get("message")
                logger.info(f"Received message from Serafina: {msg}")
                
                # Process message
                if msg == "status":
                    return {
                        "status": "online",
                        "trading_pair": self.trading_pair,
                        "active_trades": len(self.active_trades),
                        "last_price": self.last_price
                    }
                elif msg == "close_all":
                    await self.close_all_positions()
                    return {"status": "closing_all_positions"}
                else:
                    return {"status": "message_received", "content": msg}
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                raise HTTPException(status_code=400, detail=str(e))
    
    async def connect_websocket(self):
        """Connect to the WebSocket feed."""
        symbol = self.trading_pair.replace("/", "").lower()
        stream_name = f"{symbol}@kline_{self.timeframe}"
        ws_url = f"{self.websocket_url}/{stream_name}"
        
        logger.info(f"Connecting to WebSocket: {ws_url}")
        
        while True:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.websocket = websocket
                    logger.info("WebSocket connected")
                    
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            await self.process_websocket_message(data)
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed. Reconnecting...")
                            await asyncio.sleep(5)
                            break
                            
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {e}")
                            await asyncio.sleep(1)
                            
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
    
    async def process_websocket_message(self, data: Dict):
        """Process incoming WebSocket messages."""
        try:
            if 'k' in data:  # Kline data
                kline = data['k']
                is_closed = kline['x']  # Is this kline closed?
                
                # Update last price
                self.last_price = float(kline['c'])
                
                # Update price history
                new_row = {
                    'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v'])
                }
                
                # Convert to DataFrame and append
                new_df = pd.DataFrame([new_row])
                self.price_history = pd.concat([self.price_history, new_df], ignore_index=True)
                
                # Keep only the last 1000 candles to save memory
                if len(self.price_history) > 1000:
                    self.price_history = self.price_history.iloc[-1000:]
                
                # Only process at candle close
                if is_closed:
                    await self.analyze_and_trade()
        
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def analyze_and_trade(self):
        """Analyze market conditions and execute trades."""
        try:
            if len(self.price_history) < 50:  # Need enough data for analysis
                return
                
            # Get the latest data
            df = self.price_history.copy()
            
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Get the latest values
            current = df.iloc[-1]
            
            # Trading logic
            if len(self.active_trades) < self.max_open_trades:
                # Buy signal: Fast SMA crosses above Slow SMA and RSI > 50
                if (current['sma_20'] > current['sma_50'] and 
                    df['sma_20'].iloc[-2] <= df['sma_50'].iloc[-2] and 
                    current['rsi'] > 50):
                    
                    await self.enter_trade('buy')
                
                # Sell signal: Fast SMA crosses below Slow SMA and RSI < 50
                elif (current['sma_20'] < current['sma_50'] and 
                      df['sma_20'].iloc[-2] >= df['sma_50'].iloc[-2] and 
                      current['rsi'] < 50):
                    
                    await self.enter_trade('sell')
            
            # Check for exit conditions on open trades
            await self.manage_open_trades()
            
        except Exception as e:
            logger.error(f"Error in analyze_and_trade: {e}")
    
    async def enter_trade(self, side: str):
        """Enter a new trade."""
        try:
            # Get current price
            price = self.last_price
            if price is None:
                logger.warning("No price data available")
                return
            
            # Calculate position size (in quote currency)
            amount = self.amount / price
            
            # Place order
            order = await self.trading_engine.exchange.create_order(
                symbol=self.trading_pair,
                type='market',
                side=side,
                amount=amount,
                params={
                    'leverage': self.trading_engine.leverage,
                    'stopLoss': price * (0.99 if side == 'buy' else 1.01),  # 1% stop loss
                    'takeProfit': price * (1.02 if side == 'buy' else 0.98)  # 2% take profit
                }
            )
            
            # Track the trade
            trade_id = order['id']
            self.active_trades[trade_id] = {
                'side': side,
                'entry_price': price,
                'amount': amount,
                'timestamp': datetime.now(),
                'status': 'open'
            }
            
            logger.info(f"Entered {side} trade at {price}")
            
        except Exception as e:
            logger.error(f"Error entering trade: {e}")
    
    async def manage_open_trades(self):
        """Check and manage open trades."""
        try:
            if not self.active_trades:
                return
                
            # Get current positions
            positions = await self.trading_engine.exchange.fetch_positions([self.trading_pair])
            open_positions = {}
            
            for pos in positions:
                if float(pos['contracts']) != 0:
                    open_positions[pos['id']] = pos
            
            # Check each active trade
            for trade_id in list(self.active_trades.keys()):
                trade = self.active_trades[trade_id]
                
                # If trade is no longer in open positions, mark as closed
                if trade_id not in open_positions and trade['status'] == 'open':
                    trade['status'] = 'closed'
                    trade['exit_price'] = self.last_price
                    trade['exit_time'] = datetime.now()
                    
                    # Calculate P&L
                    if trade['side'] == 'buy':
                        pnl = (trade['exit_price'] - trade['entry_price']) * trade['amount']
                    else:  # sell
                        pnl = (trade['entry_price'] - trade['exit_price']) * trade['amount']
                    
                    trade['pnl'] = pnl
                    logger.info(f"Trade {trade_id} closed. P&L: {pnl:.2f} {self.trading_pair.split('/')[1]}")
            
            # Clean up closed trades
            self.active_trades = {k: v for k, v in self.active_trades.items() if v['status'] == 'open'}
            
        except Exception as e:
            logger.error(f"Error managing open trades: {e}")
    
    async def close_all_positions(self):
        """Close all open positions."""
        try:
            logger.info("Closing all open positions...")
            
            # Get current positions
            positions = await self.trading_engine.exchange.fetch_positions([self.trading_pair])
            
            for pos in positions:
                if float(pos['contracts']) != 0:
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    await self.trading_engine.exchange.create_order(
                        symbol=self.trading_pair,
                        type='market',
                        side=side,
                        amount=abs(float(pos['contracts'])),
                        params={'reduceOnly': True}
                    )
                    logger.info(f"Closed position: {pos['id']}")
            
            # Clear active trades
            self.active_trades = {}
            logger.info("All positions closed")
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            raise

async def main():
    # Initialize the scalper
    scalper = DivinaScalper()
    
    # Start the WebSocket connection in the background
    asyncio.create_task(scalper.connect_websocket())
    
    # Start the FastAPI server
    config = uvicorn.Config(
        app=scalper.app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("Starting Divina Scalper...")
    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Divina Scalper...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
