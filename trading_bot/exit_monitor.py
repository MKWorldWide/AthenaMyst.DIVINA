import time
import ccxt
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TradeState:
    symbol: str
    side: str           # 'buy' or 'sell'
    entry_price: float  # Entry price
    amount: float       # Position size in base currency
    take_profit: float  # TP price
    stop_loss: float    # SL price
    timestamp: float    # When the trade was opened

class ExitMonitor:
    """
    Monitors open positions and manages exits based on TP/SL levels.
    Runs in a separate thread for each position.
    """
    
    def __init__(self, exchange: ccxt.Exchange, trade: TradeState):
        """
        Initialize the exit monitor for a single position.
        
        Args:
            exchange: ccxt exchange instance
            trade: TradeState with position details
        """
        self.exchange = exchange
        self.trade = trade
        self.active = True
        self.exit_reason = None
        self.exit_price = None
        
    def run(self):
        """Monitor the position and exit when TP/SL is hit."""
        print(f"[ExitMonitor] Started monitoring {self.trade.symbol} {self.trade.side.upper()} "
              f"@{self.trade.entry_price:.8f} (TP: {self.trade.take_profit:.8f}, "
              f"SL: {self.trade.stop_loss:.8f})")
        
        while self.active:
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(self.trade.symbol)
                current_price = ticker['last']
                
                # Check exit conditions
                if self.trade.side == 'buy':
                    if current_price >= self.trade.take_profit:
                        self._exit_position('take_profit', current_price)
                        break
                    elif current_price <= self.trade.stop_loss:
                        self._exit_position('stop_loss', current_price)
                        break
                else:  # sell (short)
                    if current_price <= self.trade.take_profit:
                        self._exit_position('take_profit', current_price)
                        break
                    elif current_price >= self.trade.stop_loss:
                        self._exit_position('stop_loss', current_price)
                        break
                
                # Small delay to avoid rate limiting
                time.sleep(5)
                
            except ccxt.NetworkError as e:
                print(f"[ExitMonitor] Network error monitoring {self.trade.symbol}: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"[ExitMonitor] Error monitoring {self.trade.symbol}: {e}")
                time.sleep(10)
    
    def _exit_position(self, reason: str, exit_price: float):
        """Execute the exit trade and clean up."""
        try:
            # Determine exit side (opposite of entry)
            exit_side = 'sell' if self.trade.side == 'buy' else 'buy'
            
            # Place the exit order
            order = self.exchange.create_order(
                symbol=self.trade.symbol,
                type='market',
                side=exit_side,
                amount=self.trade.amount,
                params={"type": "spot"}
            )
            
            self.exit_reason = reason
            self.exit_price = exit_price
            
            # Calculate P&L
            if self.trade.side == 'buy':
                pnl_pct = ((exit_price - self.trade.entry_price) / self.trade.entry_price) * 100
            else:  # short
                pnl_pct = ((self.trade.entry_price - exit_price) / self.trade.entry_price) * 100
            
            print(f"[ExitMonitor] Exited {self.trade.symbol} {self.trade.side.upper()} "
                  f"@ {exit_price:.8f} ({reason.replace('_', ' ').title()}) - "
                  f"P&L: {pnl_pct:+.2f}%")
            
        except Exception as e:
            print(f"[ExitMonitor] Error exiting {self.trade.symbol}: {e}")
            self.exit_reason = f'error: {str(e)}'
        finally:
            self.active = False
    
    def stop(self):
        """Stop the monitoring thread."""
        self.active = False
