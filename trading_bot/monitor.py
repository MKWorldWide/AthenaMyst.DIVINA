import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import ccxt
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitoring.log')
    ]
)
logger = logging.getLogger('Monitor')

class TradingMonitor:
    def __init__(self, exchange_id: str = 'kraken'):
        """Initialize the trading monitor."""
        load_dotenv()
        self.exchange_id = exchange_id
        self.exchange = self._init_exchange()
        self.pnl_file = 'trading_pnl.csv'
        self.refresh_interval = 10  # seconds
        
    def _init_exchange(self):
        """Initialize the exchange with API credentials."""
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange = exchange_class({
            'apiKey': os.getenv(f'{self.exchange_id.upper()}_API_KEY'),
            'secret': os.getenv(f'{self.exchange_id.upper()}_API_SECRET'),
            'enableRateLimit': True,
        })
        return exchange
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            if hasattr(self.exchange, 'fetch_positions'):
                return self.exchange.fetch_posions()
            return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_balance(self) -> Dict:
        """Get current account balance."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    def get_pnl_history(self) -> pd.DataFrame:
        """Load PnL history from CSV."""
        if not os.path.exists(self.pnl_file):
            return pd.DataFrame()
        return pd.read_csv(self.pnl_file)
    
    def calculate_stats(self, pnl_df: pd.DataFrame) -> Dict:
        """Calculate trading statistics."""
        if pnl_df.empty:
            return {}
            
        stats = {
            'total_trades': len(pnl_df),
            'win_rate': (pnl_df['pnl'] > 0).mean() * 100,
            'avg_win': pnl_df[pnl_df['pnl'] > 0]['pnl'].mean(),
            'avg_loss': pnl_df[pnl_df['pnl'] <= 0]['pnl'].mean(),
            'profit_factor': -pnl_df[pnl_df['pnl'] > 0]['pnl'].sum() / max(1, pnl_df[pnl_df['pnl'] < 0]['pnl'].sum()),
            'max_drawdown': self.calculate_max_drawdown(pnl_df['cumulative_pnl'])
        }
        return stats
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calculate maximum drawdown."""
        peak = cumulative_returns.max()
        trough = cumulative_returns[cumulative_returns.argmax():].min()
        return (trough - peak) / peak * 100
    
    def display_dashboard(self):
        """Display real-time monitoring dashboard."""
        try:
            while True:
                # Clear console
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Get current status
                balance = self.get_balance()
                positions = self.get_open_positions()
                pnl_df = self.get_pnl_history()
                
                # Display header
                print(f"""
╔══════════════════════════════════════════════════╗
║         CRYPTO SCALPING BOT - LIVE MONITOR       ║
╚══════════════════════════════════════════════════╝
Exchange: {self.exchange_id.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
                
                # Display balance
                if balance:
                    print("\n╔══════════════════════════════════════════════════╗")
                    print("║                     BALANCE                        ║")
                    print("╚══════════════════════════════════════════════════╝")
                    for currency, amount in balance['total'].items():
                        if amount > 0:
                            print(f"{currency}: {amount:.8f}")
                
                # Display open positions
                if positions:
                    print("\n╔══════════════════════════════════════════════════╗")
                    print("║                 OPEN POSITIONS                     ║")
                    print("╚══════════════════════════════════════════════════╝")
                    for pos in positions:
                        print(f"{pos['symbol']}: {pos['amount']} @ {pos['entryPrice']} "
                              f"(P&L: {pos['percentage']:.2f}%)")
                
                # Display performance metrics
                if not pnl_df.empty:
                    stats = self.calculate_stats(pnl_df)
                    print("\n╔══════════════════════════════════════════════════╗")
                    print("║                  PERFORMANCE                        ║")
                    print("╚══════════════════════════════════════════════════╝")
                    print(f"Total Trades: {stats.get('total_trades', 0)}")
                    print(f"Win Rate: {stats.get('win_rate', 0):.2f}%")
                    print(f"Avg Win: {stats.get('avg_win', 0):.2f}%")
                    print(f"Avg Loss: {stats.get('avg_loss', 0):.2f}%")
                    print(f"Profit Factor: {stats.get('profit_factor', 0):.2f}")
                    print(f"Max Drawdown: {stats.get('max_drawdown', 0):.2f}%")
                
                print("\nPress Ctrl+C to exit...")
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")

def main():
    monitor = TradingMonitor()
    monitor.display_dashboard()

if __name__ == "__main__":
    main()
