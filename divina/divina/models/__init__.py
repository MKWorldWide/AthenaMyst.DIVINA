""
Data models for AthenaMyst:Divina.

This module contains all the Pydantic models used throughout the application.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class Timeframe(str, Enum):
    """Supported timeframes for market data and analysis."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN = "MN"


class SignalDirection(str, Enum):
    """Trading signal directions."""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SignalStrength(str, Enum):
    """Signal strength indicators."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class Candle(BaseModel):
    """OHLCV candle data model."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            from datetime import datetime
            return datetime.fromisoformat(v)
        return v


class IndicatorValues(BaseModel):
    """Container for indicator values at a specific timestamp."""
    timestamp: datetime
    values: Dict[str, float]
    
    def __getitem__(self, key: str) -> float:
        return self.values[key]


class Signal(BaseModel):
    """Trading signal model."""
    pair: str
    timeframe: Timeframe
    direction: SignalDirection
    strength: SignalStrength
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Trade(BaseModel):
    """Trade execution model."""
    id: str
    pair: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float  # Position size in base currency
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED_OUT, TAKE_PROFIT
    signals: List[Signal] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarketData(BaseModel):
    """Market data response model."""
    pair: str
    timeframe: Timeframe
    candles: List[Candle]
    indicators: Dict[str, List[float]] = Field(default_factory=dict)
    
    @property
    def last_candle(self) -> Optional[Candle]:
        """Get the most recent candle."""
        if not self.candles:
            return None
        return self.candles[-1]
    
    def add_indicator(self, name: str, values: List[float]) -> None:
        """Add indicator values to the market data."""
        if len(values) != len(self.candles):
            raise ValueError(
                f"Indicator length {len(values)} does not match candle length {len(self.candles)}"
            )
        self.indicators[name] = values


class BacktestResult(BaseModel):
    """Results of a backtest run."""
    pair: str
    timeframe: Timeframe
    start_time: datetime
    end_time: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    total_return: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    trade_history: List[Trade] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
