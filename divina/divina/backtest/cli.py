""
Command-line interface for backtesting trading strategies.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from divina.backtest import BacktestEngine, BacktestResult, save_backtest_results
from divina.data.manager import DataManager
from divina.strategy.rules import Signal, SignalDirection, SignalType

# Initialize Typer app
app = typer.Typer(
    name="backtest",
    help="Backtest trading strategies on historical data",
    add_completion=False,
    no_args_is_help=True,
)

# Initialize console for rich output
console = Console()


def display_results(results: BacktestResult, show_trades: bool = False) -> None:
    """Display backtest results in a formatted table."""
    # Create a table for the summary
    summary_table = Table(
        title=f"Backtest Results ({results.start_date.date()} to {results.end_date.date()})",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    
    # Add columns
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right")
    
    # Add rows
    summary_table.add_row("Initial Balance", f"${results.initial_balance:,.2f}")
    summary_table.add_row("Final Balance", f"${results.final_balance:,.2f}")
    summary_table.add_row("Total Return", f"{results.total_return:,.2f}%")
    summary_table.add_row("Annualized Return", f"{results.annualized_return:,.2f}%")
    summary_table.add_row("Max Drawdown", f"{results.max_drawdown_pct:,.2f}% (${results.max_drawdown:,.2f})")
    summary_table.add_row("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
    summary_table.add_row("Sortino Ratio", f"{results.sortino_ratio:.2f}")
    summary_table.add_row("", "")  # Empty row for spacing
    
    # Add trade statistics
    summary_table.add_row("Total Trades", str(results.total_trades))
    summary_table.add_row("Winning Trades", f"{results.winning_trades} ({results.win_rate:.1f}%)")
    summary_table.add_row("Losing Trades", f"{results.losing_trades} ({(100 - results.win_rate):.1f}%)")
    summary_table.add_row("Profit Factor", f"{results.profit_factor:.2f}")
    summary_table.add_row("Avg Win", f"${results.avg_win:,.2f} ({results.avg_win_pct:,.2f}%)")
    summary_table.add_row("Avg Loss", f"${results.avg_loss:,.2f} ({results.avg_loss_pct:,.2f}%)")
    summary_table.add_row("Avg Holding Period", str(results.avg_holding_period))
    
    # Display the summary table
    console.print(Panel(summary_table, title="Backtest Summary"))
    
    # Display trades if requested
    if show_trades and results.trades:
        trades_table = Table(
            title="Trades",
            show_header=True,
            header_style="bold blue",
            expand=True,
        )
        
        # Add columns
        trades_table.add_column("Entry Time")
        trades_table.add_column("Exit Time")
        trades_table.add_column("Symbol")
        trades_table.add_column("Direction")
        trades_table.add_column("Entry")
        trades_table.add_column("Exit")
        trades_table.add_column("P&L %", justify="right")
        trades_table.add_column("P&L $", justify="right")
        trades_table.add_column("Holding")
        trades_table.add_column("Exit Reason")
        
        # Add rows
        for trade in results.trades:
            direction = "LONG" if trade.direction == SignalDirection.LONG else "SHORT"
            pnl_color = "green" if trade.pnl_pct > 0 else "red"
            
            trades_table.add_row(
                str(trade.entry_time),
                str(trade.exit_time),
                trade.symbol,
                f"[green]{direction}" if direction == "LONG" else f"[red]{direction}",
                f"${trade.entry_price:.5f}",
                f"${trade.exit_price:.5f}",
                f"[{pnl_color}]{trade.pnl_pct:,.2f}%[/{pnl_color}]",
                f"[{pnl_color}]{trade.pnl:,.2f}[/{pnl_color}]",
                str(trade.holding_period),
                trade.exit_reason,
            )
        
        console.print(Panel(trades_table, title="Trade History"))


def load_data(
    data_dir: Path,
    symbols: List[str],
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> DataManager:
    """Load market data from CSV files in the specified directory."""
    data_manager = DataManager()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading market data...", total=len(symbols))
        
        for symbol in symbols:
            file_path = data_dir / f"{symbol}_{timeframe}.csv"
            
            if not file_path.exists():
                console.print(f"[yellow]Warning: Data file not found for {symbol} {timeframe}")
                progress.update(task, advance=1)
                continue
            
            try:
                # Read CSV file
                df = pd.read_csv(
                    file_path,
                    parse_dates=['date'],
                    index_col='date',
                    date_parser=pd.to_datetime,
                )
                
                # Ensure timezone awareness
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Filter by date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df[mask]
                
                if not df.empty:
                    data_manager.update_market_data(symbol, timeframe, df)
                
                progress.update(task, advance=1, description=f"Loaded {symbol} {timeframe}")
                
            except Exception as e:
                console.print(f"[red]Error loading {file_path}: {e}")
                progress.update(task, advance=1, description=f"Error loading {symbol} {timeframe}")
    
    return data_manager


class SimpleStrategy(BacktestEngine):
    """A simple moving average crossover strategy for demonstration."""
    
    def __init__(
        self,
        data_manager: DataManager,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        commission: float = 0.0005,
        slippage: float = 0.0001,
        fast_ma: int = 10,
        slow_ma: int = 30,
    ):
        super().__init__(
            data_manager=data_manager,
            initial_balance=initial_balance,
            risk_per_trade=risk_per_trade,
            commission=commission,
            slippage=slippage,
        )
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
    
    def _generate_signals(
        self,
        timestamp: pd.Timestamp,
        data: pd.DataFrame,
    ) -> List[Signal]:
        """Generate signals based on moving average crossover."""
        signals = []
        
        for _, row in data.iterrows():
            symbol = row.get('symbol')
            if not symbol:
                continue
            
            # Get historical data for this symbol
            df = self.data_manager.get_market_data(symbol, self.timeframe)
            if df is None or len(df) < self.slow_ma + 1:
                continue
            
            # Calculate moving averages
            close = df['close']
            fast_ma = close.rolling(window=self.fast_ma).mean()
            slow_ma = close.rolling(window=self.slow_ma).mean()
            
            # Get current and previous values
            curr_fast = fast_ma.iloc[-1]
            curr_slow = slow_ma.iloc[-1]
            prev_fast = fast_ma.iloc[-2]
            prev_slow = slow_ma.iloc[-2]
            
            # Check for crossover
            current_price = row['close']
            atr = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
            atr = atr.iloc[-1] if not atr.empty else current_price * 0.02  # Default 2% ATR
            
            # Generate signals
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                # Bullish crossover - buy signal
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.MOVING_AVERAGE,
                    direction=SignalDirection.LONG,
                    entry_price=current_price,
                    stop_loss=current_price - 2 * atr,  # 2 ATR stop loss
                    take_profit=current_price + 4 * atr,  # 2:1 reward:risk
                    timeframe=self.timeframe,
                    timestamp=timestamp,
                    confidence=0.7,
                    metadata={
                        'fast_ma': self.fast_ma,
                        'slow_ma': self.slow_ma,
                        'atr': atr,
                    },
                ))
            
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                # Bearish crossover - sell signal
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.MOVING_AVERAGE,
                    direction=SignalDirection.SHORT,
                    entry_price=current_price,
                    stop_loss=current_price + 2 * atr,  # 2 ATR stop loss
                    take_profit=current_price - 4 * atr,  # 2:1 reward:risk
                    timeframe=self.timeframe,
                    timestamp=timestamp,
                    confidence=0.7,
                    metadata={
                        'fast_ma': self.fast_ma,
                        'slow_ma': self.slow_ma,
                        'atr': atr,
                    },
                ))
        
        return signals


@app.command()
def run(
    data_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing CSV files with market data",
    ),
    symbols: List[str] = typer.Argument(
        ...,
        help="List of symbols to backtest (e.g., EURUSD GBPUSD)",
    ),
    timeframe: str = typer.Option(
        "1d",
        "--timeframe", "-t",
        help="Timeframe for the backtest (e.g., 1h, 4h, 1d)",
    ),
    start_date: str = typer.Option(
        (datetime.now(timezone.utc) - pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
        "--start-date", "-s",
        help="Start date (YYYY-MM-DD)",
    ),
    end_date: str = typer.Option(
        datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        "--end-date", "-e",
        help="End date (YYYY-MM-DD)",
    ),
    initial_balance: float = typer.Option(
        10000.0,
        "--initial-balance", "-b",
        help="Initial account balance",
    ),
    risk_per_trade: float = typer.Option(
        0.01,
        "--risk", "-r",
        help="Risk per trade as a fraction of account balance",
    ),
    fast_ma: int = typer.Option(
        10,
        "--fast-ma",
        help="Period for fast moving average",
    ),
    slow_ma: int = typer.Option(
        30,
        "--slow-ma",
        help="Period for slow moving average",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Directory to save backtest results",
    ),
    show_trades: bool = typer.Option(
        False,
        "--show-trades",
        help="Show detailed trade history",
    ),
):
    """Run a backtest with a simple moving average crossover strategy."""
    # Parse dates
    try:
        start_date_dt = pd.to_datetime(start_date).tz_localize('UTC')
        end_date_dt = pd.to_datetime(end_date).tz_localize('UTC')
    except Exception as e:
        console.print(f"[red]Error parsing dates: {e}")
        raise typer.Exit(1)
    
    if start_date_dt >= end_date_dt:
        console.print("[red]Error: Start date must be before end date")
        raise typer.Exit(1)
    
    # Load market data
    with console.status("Loading market data...", spinner="dots"):
        data_manager = load_data(data_dir, symbols, timeframe, start_date_dt, end_date_dt)
    
    if not data_manager.symbols:
        console.print("[red]Error: No market data loaded")
        raise typer.Exit(1)
    
    # Initialize backtest engine
    engine = SimpleStrategy(
        data_manager=data_manager,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
    )
    
    # Run backtest
    with console.status("Running backtest...", spinner="dots"):
        try:
            results = engine.run_backtest(
                start_date=start_date_dt,
                end_date=end_date_dt,
                symbols=symbols,
                timeframe=timeframe,
            )
        except Exception as e:
            console.print(f"[red]Error during backtest: {e}")
            if "DEBUG" in sys.argv:
                raise
            raise typer.Exit(1)
    
    # Display results
    console.print("\n" + "=" * 80)
    display_results(results, show_trades=show_trades)
    
    # Save results if output directory is specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"backtest_{timestamp}"
        
        try:
            saved_files = save_backtest_results(results, output_dir, prefix)
            console.print(f"\n[green]Backtest results saved to:[/green]")
            for file_type, file_path in saved_files.items():
                console.print(f"- {file_type}: {file_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save results: {e}")
    
    # Return non-zero exit code if final balance is less than initial balance
    if results.final_balance < results.initial_balance:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
