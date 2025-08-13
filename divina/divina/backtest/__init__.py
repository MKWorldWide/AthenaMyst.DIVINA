"""
Backtesting module for AthenaMyst:Divina.

This module provides functionality for backtesting trading strategies
on historical market data with various performance metrics and reporting.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field

from divina.strategy.rules import Signal, SignalDirection
from divina.data.manager import DataManager


class TradeResult(BaseModel):
    """Represents the result of a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol: str
    direction: SignalDirection
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    holding_period: pd.Timedelta
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    max_drawdown: float
    max_runup: float
    mfe: float  # Maximum Favorable Excursion
    mae: float  # Maximum Adverse Excursion
    exit_reason: str
    metadata: Dict = Field(default_factory=dict)


class BacktestResult(BaseModel):
    """Aggregated results of a backtest."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_balance: float
    final_balance: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_holding_period: pd.Timedelta
    trades: List[TradeResult]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.Timestamp: lambda x: x.isoformat(),
            pd.Timedelta: lambda x: x.isoformat(),
        }


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""
    
    def __init__(
        self,
        data_manager: DataManager,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        commission: float = 0.0005,  # 0.05% per trade
        slippage: float = 0.0001,    # 0.01% slippage
        position_sizing: str = 'fixed',  # 'fixed' or 'percent_risk'
        max_position_size: Optional[float] = None,
    ):
        """Initialize the backtest engine.
        
        Args:
            data_manager: DataManager instance with market data
            initial_balance: Starting account balance in quote currency
            risk_per_trade: Percentage of account to risk per trade (0.01 = 1%)
            commission: Commission per trade as a percentage of trade value
            slippage: Slippage per trade as a percentage of trade value
            position_sizing: Position sizing method ('fixed' or 'percent_risk')
            max_position_size: Maximum position size in base currency
        """
        self.data_manager = data_manager
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        
        # State
        self.open_trades: List[TradeResult] = []
        self.closed_trades: List[TradeResult] = []
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        self.current_prices: Dict[str, float] = {}
        self.current_time: Optional[pd.Timestamp] = None
    
    def run_backtest(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h',
    ) -> BacktestResult:
        """Run the backtest.
        
        Args:
            start_date: Start date of the backtest period
            end_date: End date of the backtest period
            symbols: List of symbols to backtest
            timeframe: Timeframe for the backtest
            
        Returns:
            BacktestResult with performance metrics and trade history
        """
        # Initialize backtest
        self._initialize_backtest(start_date, end_date, symbols, timeframe)
        
        # Main backtest loop
        for timestamp, data in self._iter_market_data():
            self.current_time = timestamp
            self._update_prices(data)
            
            # Generate signals for the current bar
            signals = self._generate_signals(timestamp, data)
            
            # Process signals and update positions
            self._process_signals(signals)
            
            # Update open positions
            self._update_positions()
            
            # Update equity curve
            self._update_equity_curve()
        
        # Close any remaining open positions
        self._close_all_positions()
        
        # Generate final results
        return self._generate_results()
    
    def _initialize_backtest(
        self,
        start_date: Optional[Union[str, pd.Timestamp]],
        end_date: Optional[Union[str, pd.Timestamp]],
        symbols: Optional[List[str]],
        timeframe: str,
    ) -> None:
        """Initialize backtest parameters and state."""
        # Reset state
        self.balance = self.initial_balance
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = pd.Series(dtype=float)
        self.drawdown_curve = pd.Series(dtype=float)
        self.current_prices = {}
        self.current_time = None
        
        # Set default values if not provided
        if symbols is None:
            symbols = list(self.data_manager.symbols)
        
        # Convert string dates to timestamps
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, utc=True)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date, utc=True)
        
        # Set default date range if not provided
        if start_date is None or end_date is None:
            # Get the earliest start date and latest end date across all symbols
            all_start_dates = []
            all_end_dates = []
            
            for symbol in symbols:
                df = self.data_manager.get_market_data(symbol, timeframe)
                if df is not None and not df.empty:
                    all_start_dates.append(df.index.min())
                    all_end_dates.append(df.index.max())
            
            if not all_start_dates:
                raise ValueError("No market data available for the specified symbols and timeframe.")
            
            if start_date is None:
                start_date = max(all_start_dates)
            if end_date is None:
                end_date = min(all_end_dates)
        
        # Store backtest parameters
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.timeframe = timeframe
    
    def _iter_market_data(self):
        """Iterate through market data in chronological order."""
        # Get all data points across all symbols
        all_data = []
        
        for symbol in self.symbols:
            df = self.data_manager.get_market_data(
                symbol=symbol,
                timeframe=self.timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if df is not None and not df.empty:
                df['symbol'] = symbol
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No market data available for the specified parameters.")
        
        # Combine all data and sort by timestamp
        combined = pd.concat(all_data).sort_index()
        
        # Group by timestamp to process all symbols at each time step
        for timestamp, group in combined.groupby(level=0):
            yield timestamp, group
    
    def _update_prices(self, data: pd.DataFrame) -> None:
        """Update current prices from the latest market data."""
        for _, row in data.iterrows():
            symbol = row.get('symbol')
            if symbol:
                self.current_prices[symbol] = row['close']
    
    def _generate_signals(
        self,
        timestamp: pd.Timestamp,
        data: pd.DataFrame
    ) -> List[Signal]:
        """Generate trading signals for the current bar.
        
        This method should be implemented by subclasses to define specific
        trading strategies.
        
        Args:
            timestamp: Current timestamp
            data: Market data for the current bar
            
        Returns:
            List of trading signals
        """
        # Default implementation returns no signals
        # Subclasses should override this method
        return []
    
    def _process_signals(self, signals: List[Signal]) -> None:
        """Process trading signals and update positions."""
        for signal in signals:
            if signal.direction == SignalDirection.LONG:
                self._enter_long(signal)
            elif signal.direction == SignalDirection.SHORT:
                self._enter_short(signal)
    
    def _enter_long(self, signal: Signal) -> None:
        """Enter a long position based on a signal."""
        if not self._can_enter_trade(signal):
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        if position_size <= 0:
            return
        
        # Calculate entry price with slippage
        entry_price = signal.entry_price * (1 + self.slippage)
        
        # Calculate stop loss and take profit
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        
        # Calculate risk/reward ratio
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Create trade
        trade = TradeResult(
            entry_time=self.current_time,
            exit_time=None,
            symbol=signal.symbol,
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            exit_price=None,
            size=position_size,
            pnl=0.0,
            pnl_pct=0.0,
            holding_period=pd.Timedelta(0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            max_drawdown=0.0,
            max_runup=0.0,
            mfe=0.0,
            mae=0.0,
            exit_reason="",
            metadata=signal.metadata.copy() if hasattr(signal, 'metadata') else {},
        )
        
        # Deduct position cost from balance
        position_cost = position_size * entry_price
        commission_cost = position_cost * self.commission
        self.balance -= (position_cost + commission_cost)
        
        # Add to open trades
        self.open_trades.append(trade)
    
    def _enter_short(self, signal: Signal) -> None:
        """Enter a short position based on a signal."""
        if not self._can_enter_trade(signal):
            return
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        if position_size <= 0:
            return
        
        # Calculate entry price with slippage
        entry_price = signal.entry_price * (1 - self.slippage)
        
        # Calculate stop loss and take profit
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        
        # Calculate risk/reward ratio
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Create trade
        trade = TradeResult(
            entry_time=self.current_time,
            exit_time=None,
            symbol=signal.symbol,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            exit_price=None,
            size=position_size,
            pnl=0.0,
            pnl_pct=0.0,
            holding_period=pd.Timedelta(0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            max_drawdown=0.0,
            max_runup=0.0,
            mfe=0.0,
            mae=0.0,
            exit_reason="",
            metadata=signal.metadata.copy() if hasattr(signal, 'metadata') else {},
        )
        
        # For short positions, we receive money when entering
        position_value = position_size * entry_price
        commission_cost = position_value * self.commission
        self.balance += (position_value - commission_cost)
        
        # Add to open trades
        self.open_trades.append(trade)
    
    def _can_enter_trade(self, signal: Signal) -> bool:
        """Check if we can enter a new trade based on the signal."""
        # Check if we already have an open position for this symbol
        for trade in self.open_trades:
            if trade.symbol == signal.symbol and trade.exit_time is None:
                return False
        
        # Check if we have enough balance
        position_size = self._calculate_position_size(signal)
        if position_size <= 0:
            return False
            
        return True
    
    def _calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size based on the signal and risk parameters."""
        if self.position_sizing == 'fixed':
            # Fixed position size (number of units)
            position_size = self.initial_balance * self.risk_per_trade / signal.entry_price
        elif self.position_sizing == 'percent_risk':
            # Position size based on percentage of account at risk
            risk_amount = self.balance * self.risk_per_trade
            
            if signal.direction == SignalDirection.LONG:
                risk_per_unit = signal.entry_price - signal.stop_loss
            else:  # SHORT
                risk_per_unit = signal.stop_loss - signal.entry_price
            
            if risk_per_unit <= 0:
                return 0.0
                
            position_size = risk_amount / risk_per_unit
        else:
            raise ValueError(f"Unknown position sizing method: {self.position_sizing}")
        
        # Apply maximum position size if specified
        if self.max_position_size is not None:
            position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def _update_positions(self) -> None:
        """Update open positions and check for exits."""
        if not self.open_trades:
            return
        
        for trade in list(self.open_trades):
            if trade.exit_time is not None:
                continue  # Already closed
            
            current_price = self.current_prices.get(trade.symbol)
            if current_price is None:
                continue  # No price data for this symbol
            
            # Update trade metrics
            if trade.direction == SignalDirection.LONG:
                # For long positions
                trade.pnl = (current_price - trade.entry_price) * trade.size
                trade.pnl_pct = (current_price / trade.entry_price - 1) * 100
                
                # Check for exit conditions
                if current_price >= trade.take_profit:
                    self._exit_trade(trade, current_price, "take_profit")
                elif current_price <= trade.stop_loss:
                    self._exit_trade(trade, current_price, "stop_loss")
                else:
                    # Update max drawdown/runup
                    trade.max_runup = max(trade.max_runup, trade.pnl_pct)
                    trade.mfe = max(trade.mfe, trade.pnl_pct)
                    trade.mae = min(trade.mae, trade.pnl_pct)
                    
                    # Update holding period
                    trade.holding_period = self.current_time - trade.entry_time
            
            else:  # SHORT
                # For short positions
                trade.pnl = (trade.entry_price - current_price) * trade.size
                trade.pnl_pct = (1 - current_price / trade.entry_price) * 100
                
                # Check for exit conditions
                if current_price <= trade.take_profit:
                    self._exit_trade(trade, current_price, "take_profit")
                elif current_price >= trade.stop_loss:
                    self._exit_trade(trade, current_price, "stop_loss")
                else:
                    # Update max drawdown/runup
                    trade.max_runup = max(trade.max_runup, trade.pnl_pct)
                    trade.mfe = max(trade.mfe, trade.pnl_pct)
                    trade.mae = min(trade.mae, trade.pnl_pct)
                    
                    # Update holding period
                    trade.holding_period = self.current_time - trade.entry_time
    
    def _exit_trade(self, trade: TradeResult, exit_price: float, reason: str) -> None:
        """Close a trade and update account balance."""
        # Update trade with exit information
        trade.exit_time = self.current_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate final P&L with commission
        if trade.direction == SignalDirection.LONG:
            # For long positions: (exit - entry) * size - commission
            trade_value = trade.exit_price * trade.size
            commission = trade_value * self.commission
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size - commission
            self.balance += trade_value - commission
        else:  # SHORT
            # For short positions: (entry - exit) * size - commission
            trade_value = trade.exit_price * trade.size
            commission = trade_value * self.commission
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.size - commission
            self.balance += trade.pnl
        
        # Calculate P&L percentage
        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.size)) * 100
        
        # Update holding period
        trade.holding_period = trade.exit_time - trade.entry_time
        
        # Move from open to closed trades
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
    
    def _close_all_positions(self) -> None:
        """Close all open positions at the current market price."""
        for trade in list(self.open_trades):
            if trade.exit_time is None:
                current_price = self.current_prices.get(trade.symbol)
                if current_price is not None:
                    self._exit_trade(trade, current_price, "end_of_backtest")
    
    def _update_equity_curve(self) -> None:
        """Update the equity curve with the current account balance."""
        if self.current_time is not None:
            self.equity_curve[self.current_time] = self.balance
    
    def _generate_results(self) -> BacktestResult:
        """Generate backtest results and performance metrics."""
        if not self.closed_trades:
            raise ValueError("No trades were executed during the backtest.")
        
        # Calculate performance metrics
        total_return = (self.balance / self.initial_balance - 1) * 100
        
        # Calculate annualized return
        days = (self.end_date - self.start_date).days
        years = max(days / 365.25, 1/365.25)  # At least 1 day to avoid division by zero
        cagr = (self.balance / self.initial_balance) ** (1 / years) - 1
        
        # Calculate win rate
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
        
        # Calculate average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        # Calculate profit factor
        gross_profit = sum(max(0, t.pnl) for t in self.closed_trades)
        gross_loss = abs(sum(min(0, t.pnl) for t in self.closed_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown
        peak = self.equity_curve.cummax()
        drawdowns = (self.equity_curve - peak) / peak
        max_drawdown_pct = abs(drawdowns.min()) * 100 if not drawdowns.empty else 0
        max_drawdown = abs((self.equity_curve - peak).min()) if not self.equity_curve.empty else 0
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        returns = self.equity_curve.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Calculate Sortino ratio (assuming risk-free rate = 0)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calculate average holding period
        holding_periods = [t.holding_period for t in self.closed_trades]
        avg_holding_period = pd.to_timedelta(np.mean(holding_periods)) if holding_periods else pd.Timedelta(0)
        
        # Create and return results
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_return=total_return,
            annualized_return=cagr * 100,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0,
            sortino_ratio=float(sortino_ratio) if not np.isnan(sortino_ratio) else 0,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            total_trades=len(self.closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            avg_holding_period=avg_holding_period,
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            drawdown_curve=drawdowns * 100,  # As percentage
        )


def save_backtest_results(
    results: BacktestResult,
    output_dir: Union[str, Path],
    prefix: str = "backtest"
) -> Dict[str, Path]:
    """Save backtest results to CSV files.
    
    Args:
        results: Backtest results to save
        output_dir: Directory to save the results
        prefix: Prefix for output filenames
        
    Returns:
        Dictionary mapping output types to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Save trades to CSV
    trades_df = pd.DataFrame([t.dict() for t in results.trades])
    trades_file = output_dir / f"{prefix}_trades.csv"
    
    # Convert datetime/timedelta columns to strings for CSV
    for col in ['entry_time', 'exit_time', 'holding_period']:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].astype(str)
    
    trades_df.to_csv(trades_file, index=False)
    output_files['trades'] = trades_file
    
    # Save equity curve to CSV
    equity_file = output_dir / f"{prefix}_equity_curve.csv"
    equity_df = results.equity_curve.reset_index()
    equity_df.columns = ['timestamp', 'equity']
    equity_df.to_csv(equity_file, index=False)
    output_files['equity'] = equity_file
    
    # Save drawdown curve to CSV
    drawdown_file = output_dir / f"{prefix}_drawdown.csv"
    drawdown_df = results.drawdown_curve.reset_index()
    drawdown_df.columns = ['timestamp', 'drawdown_pct']
    drawdown_df.to_csv(drawdown_file, index=False)
    output_files['drawdown'] = drawdown_file
    
    # Save summary to a text file
    summary_file = output_dir / f"{prefix}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Backtest Results - {prefix}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Summary:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Period: {results.start_date} to {results.end_date}\n")
        f.write(f"Initial Balance: ${results.initial_balance:,.2f}\n")
        f.write(f"Final Balance: ${results.final_balance:,.2f}\n")
        f.write(f"Total Return: {results.total_return:,.2f}%\n")
        f.write(f"Annualized Return: {results.annualized_return:,.2f}%\n")
        f.write(f"Max Drawdown: {results.max_drawdown_pct:,.2f}% (${results.max_drawdown:,.2f})\n")
        f.write(f"Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
        f.write(f"Sortino Ratio: {results.sortino_ratio:.2f}\n\n")
        
        f.write("Trade Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Trades: {results.total_trades}\n")
        f.write(f"Winning Trades: {results.winning_trades} ({results.win_rate:.1f}%)\n")
        f.write(f"Losing Trades: {results.losing_trades} ({100 - results.win_rate:.1f}%)\n")
        f.write(f"Profit Factor: {results.profit_factor:.2f}\n")
        f.write(f"Average Win: ${results.avg_win:,.2f} ({results.avg_win_pct:,.2f}%)\n")
        f.write(f"Average Loss: ${results.avg_loss:,.2f} ({results.avg_loss_pct:,.2f}%)\n")
        f.write(f"Average Holding Period: {results.avg_holding_period}\n")
    
    output_files['summary'] = summary_file
    
    return output_files
