import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import websockets
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_monitor.log')
    ]
)
logger = logging.getLogger('TradingMonitor')

class TradingMonitor:
    """Monitor trading bot performance and activity."""
    
    def __init__(self):
        self.bot_url = "http://localhost:8002"
        self.websocket_url = "ws://localhost:8002/ws"
        self.trades: List[Dict] = []
        self.performance_metrics: Dict = {}
        self.start_time = datetime.now()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        logger.info("Trading Monitor initialized")
    
    async def check_bot_status(self):
        """Check if the trading bot is online and get its status."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.bot_url}/health") as response:
                    if response.status == 200:
                        status = await response.json()
                        logger.info(f"Bot status: {status}")
                        return True
        except Exception as e:
            logger.error(f"Error checking bot status: {e}")
        return False
    
    async def get_bot_metrics(self):
        """Get performance metrics from the trading bot."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.bot_url}/metrics") as response:
                    if response.status == 200:
                        metrics = await response.json()
                        self.performance_metrics = metrics
                        logger.info(f"Updated performance metrics: {metrics}")
                        return metrics
        except Exception as e:
            logger.error(f"Error getting bot metrics: {e}")
        return {}
    
    async def connect_websocket(self):
        """Connect to the bot's WebSocket for real-time updates."""
        while True:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    logger.info("Connected to bot's WebSocket")
                    
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
        """Process incoming WebSocket messages from the bot."""
        try:
            event_type = data.get('event')
            
            if event_type == 'trade_executed':
                trade = data.get('data', {})
                self.trades.append(trade)
                logger.info(f"New trade executed: {trade}")
                
                # Save trades to CSV
                self.save_trades_to_csv()
                
                # Generate performance report
                await self.generate_performance_report()
                
            elif event_type == 'price_update':
                # Update price chart
                price = data.get('data', {}).get('price')
                if price:
                    await self.update_price_chart(price)
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def save_trades_to_csv(self):
        """Save trades to a CSV file."""
        if not self.trades:
            return
            
        df = pd.DataFrame(self.trades)
        filepath = f"data/trades_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} trades to {filepath}")
    
    async def generate_performance_report(self):
        """Generate a performance report with key metrics."""
        if not self.trades:
            return
            
        df = pd.DataFrame(self.trades)
        
        # Calculate performance metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 else float('inf')
        
        # Save metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'last_updated': datetime.now().isoformat()
        }
        
        # Save metrics to file
        with open('data/performance_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Performance report generated: {metrics}")
        
        # Generate and save charts
        self.generate_performance_charts(df)
    
    def generate_performance_charts(self, df: pd.DataFrame):
        """Generate performance charts."""
        try:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Cumulative P&L chart
            plt.figure(figsize=(12, 6))
            df['cumulative_pnl'] = df['pnl'].cumsum()
            plt.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative P&L')
            plt.title('Cumulative P&L Over Time')
            plt.xlabel('Date')
            plt.ylabel('P&L (USDT)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('data/cumulative_pnl.png')
            plt.close()
            
            # Win/Loss distribution
            plt.figure(figsize=(8, 6))
            df['pnl'].plot(kind='hist', bins=20, alpha=0.7, color='green' if df['pnl'].mean() >= 0 else 'red')
            plt.axvline(x=0, color='black', linestyle='--')
            plt.title('P&L Distribution')
            plt.xlabel('P&L (USDT)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('data/pnl_distribution.png')
            plt.close()
            
            # Daily P&L
            df['date'] = df['timestamp'].dt.date
            daily_pnl = df.groupby('date')['pnl'].sum().reset_index()
            
            plt.figure(figsize=(12, 6))
            plt.bar(daily_pnl['date'], daily_pnl['pnl'], 
                   color=['green' if x >= 0 else 'red' for x in daily_pnl['pnl']],
                   alpha=0.7)
            plt.title('Daily P&L')
            plt.xlabel('Date')
            plt.ylabel('P&L (USDT)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('data/daily_pnl.png')
            plt.close()
            
            logger.info("Generated performance charts")
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
    
    async def update_price_chart(self, price_data: Dict):
        """Update price chart with latest price data."""
        try:
            # This would be implemented to update a real-time price chart
            # For now, we'll just log the price update
            logger.debug(f"Price update: {price_data}")
            
        except Exception as e:
            logger.error(f"Error updating price chart: {e}")
    
    async def run(self):
        """Run the monitoring service."""
        logger.info("Starting Trading Monitor...")
        
        # Check if bot is online
        if not await self.check_bot_status():
            logger.error("Trading bot is not responding. Please start the bot first.")
            return
        
        # Start WebSocket connection in the background
        asyncio.create_task(self.connect_websocket())
        
        # Main monitoring loop
        while True:
            try:
                # Update metrics every 5 minutes
                await self.get_bot_metrics()
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                logger.info("Monitoring stopped by user")
                break
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

async def main():
    monitor = TradingMonitor()
    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down Trading Monitor...")
    except Exception as e:
        logger.error(f"Fatal error in Trading Monitor: {e}")

if __name__ == "__main__":
    asyncio.run(main())
