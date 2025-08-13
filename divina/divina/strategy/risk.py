""
Risk management module for AthenaMyst:Divina.

Handles position sizing, stop-loss, take-profit, and risk calculations.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import math

from loguru import logger

from ..models import Signal, SignalDirection, Timeframe, Candle
from ..config import settings


@dataclass
class PositionSizing:
    """Position sizing calculation result."""
    size: float  # Position size in units of the base currency
    stop_loss: float  # Stop loss price
    take_profit: float  # Take profit price
    risk_amount: float  # Amount to risk in account currency
    reward_risk_ratio: float  # Reward to risk ratio
    leverage: int  # Leverage to use (if any)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'size': self.size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_amount': self.risk_amount,
            'reward_risk_ratio': self.reward_risk_ratio,
            'leverage': self.leverage
        }


class RiskManager:
    """Manages risk calculations and position sizing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the risk manager."""
        self.config = config or {}
        
        # Risk parameters with defaults
        self.risk_per_trade = float(self.config.get('risk_per_trade', 0.01))  # 1% risk per trade
        self.max_leverage = int(self.config.get('max_leverage', 10))
        self.min_reward_risk = float(self.config.get('min_reward_risk', 1.5))
        self.default_sl_pct = float(self.config.get('default_sl_pct', 0.01))  # 1% default SL
        self.default_tp_pct = float(self.config.get('default_tp_pct', 0.02))  # 2% default TP
        
        # Account information (can be updated)
        self.account_balance = float(self.config.get('account_balance', 10000.0))
        self.account_currency = self.config.get('account_currency', 'USD')
        self.leverage = int(self.config.get('leverage', 1))
        
        # Instrument information (pip values, lot sizes, etc.)
        self.instrument_info: Dict[str, Dict[str, float]] = {}
    
    def update_account_balance(self, balance: float) -> None:
        """Update the account balance."""
        self.account_balance = float(balance)
    
    def update_leverage(self, leverage: int) -> None:
        """Update the leverage."""
        self.leverage = min(max(1, int(leverage)), self.max_leverage)
    
    def calculate_pip_value(
        self,
        pair: str,
        price: float,
        account_currency: Optional[str] = None
    ) -> float:
        """Calculate the value of one pip in the account currency."""
        account_currency = account_currency or self.account_currency
        
        # For pairs where the quote currency is the account currency
        if pair.endswith(account_currency):
            return 0.0001  # 1 pip = 0.0001 for most pairs
        
        # For pairs where the base currency is the account currency
        if pair.startswith(account_currency):
            return 0.0001 * price
        
        # For cross pairs, we'd need the conversion rate to the account currency
        # This is a simplified version - in practice, you'd need to fetch the rate
        logger.warning(f"Cross-currency pip calculation not fully implemented for {pair}")
        return 0.0001  # Fallback
    
    def calculate_position_size(
        self,
        pair: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        direction: SignalDirection = SignalDirection.BUY,
        risk_percentage: Optional[float] = None,
        leverage: Optional[int] = None
    ) -> PositionSizing:
        """
        Calculate position size based on risk parameters.
        
        Args:
            pair: Trading pair (e.g., 'EUR_USD')
            entry_price: Entry price
            stop_loss: Stop loss price (if None, uses default percentage)
            take_profit: Take profit price (if None, uses default percentage)
            direction: Long or short position
            risk_percentage: Risk percentage (0-1), defaults to self.risk_per_trade
            leverage: Leverage to use, defaults to self.leverage
            
        Returns:
            PositionSizing object with calculated values
        """
        # Use instance defaults if not provided
        risk_percentage = risk_percentage or self.risk_per_trade
        leverage = leverage or self.leverage
        
        # Ensure risk is within bounds
        risk_percentage = max(0.0001, min(risk_percentage, 0.5))  # Cap at 50% risk
        
        # Calculate stop loss and take profit if not provided
        if stop_loss is None:
            if direction == SignalDirection.BUY:
                stop_loss = entry_price * (1 - self.default_sl_pct)
            else:
                stop_loss = entry_price * (1 + self.default_sl_pct)
        
        if take_profit is None:
            if direction == SignalDirection.BUY:
                take_profit = entry_price * (1 + self.default_tp_pct)
            else:
                take_profit = entry_price * (1 - self.default_tp_pct)
        
        # Calculate risk amount in account currency
        risk_amount = self.account_balance * risk_percentage
        
        # Calculate price difference (risk per unit)
        if direction == SignalDirection.BUY:
            price_diff = abs(entry_price - stop_loss)
            reward_diff = abs(take_profit - entry_price)
        else:
            price_diff = abs(stop_loss - entry_price)
            reward_diff = abs(entry_price - take_profit)
        
        # Avoid division by zero
        if price_diff <= 0:
            raise ValueError("Stop loss is too close to entry price")
        
        # Calculate position size
        pip_value = self.calculate_pip_value(pair, entry_price)
        pip_risk = price_diff / 0.0001  # Convert to pips
        
        # Position size calculation with leverage
        position_size = (risk_amount / pip_risk) * leverage
        
        # Calculate reward/risk ratio
        reward_risk_ratio = reward_diff / price_diff if price_diff > 0 else 0
        
        # Round position size to appropriate lot size
        # Standard lot = 100,000 units, mini lot = 10,000, micro lot = 1,000
        lot_size = self._get_lot_size(pair)
        position_size = self._round_to_lot(position_size, lot_size)
        
        return PositionSizing(
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reward_risk_ratio=reward_risk_ratio,
            leverage=leverage
        )
    
    def _get_lot_size(self, pair: str) -> float:
        """Get the standard lot size for a trading pair."""
        # In a real implementation, this would come from the broker/instrument info
        # This is a simplified version
        return 1000.0  # Default to micro lots (0.01 standard lot)
    
    def _round_to_lot(self, size: float, lot_size: float) -> float:
        """Round position size to the nearest lot size."""
        if lot_size <= 0:
            return size
        return math.floor(size / lot_size) * lot_size
    
    def calculate_volatility_stop_loss(
        self,
        candles: List[Candle],
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        use_pct: bool = False
    ) -> float:
        """
        Calculate a volatility-based stop loss using ATR.
        
        Args:
            candles: List of recent candles
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR to set stop distance
            use_pct: If True, return as percentage of price
            
        Returns:
            Stop loss price or percentage
        """
        if not candles or len(candles) < atr_period + 1:
            return self.default_sl_pct if use_pct else 0.0
        
        # Calculate ATR
        high_prices = [c.high for c in candles[-atr_period:]]
        low_prices = [c.low for c in candles[-atr_period:]]
        close_prices = [c.close for c in candles[-atr_period-1:-1]]  # Previous closes
        
        tr_sum = 0.0
        for i in range(len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i])
            tr3 = abs(low_prices[i] - close_prices[i])
            tr_sum += max(tr1, tr2, tr3)
        
        atr = tr_sum / atr_period
        
        if use_pct:
            last_close = candles[-1].close
            return (atr_multiplier * atr) / last_close
        
        return atr_multiplier * atr
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: SignalDirection,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0,
        min_pct: float = 0.005,
        max_pct: float = 0.05
    ) -> float:
        """
        Calculate a trailing stop price.
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            direction: Long or short position
            atr: ATR value for dynamic stop (optional)
            atr_multiplier: Multiplier for ATR (if using ATR)
            min_pct: Minimum stop distance as percentage of price
            max_pct: Maximum stop distance as percentage of price
            
        Returns:
            Trailing stop price
        """
        if atr is not None:
            # Use ATR for dynamic stop
            stop_distance = atr * atr_multiplier
        else:
            # Use percentage of price
            stop_distance = current_price * min_pct
        
        # Apply min/max constraints
        min_distance = current_price * min_pct
        max_distance = current_price * max_pct
        stop_distance = max(min(stop_distance, max_distance), min_distance)
        
        if direction == SignalDirection.BUY:
            return current_price - stop_distance
        else:
            return current_price + stop_distance
    
    def get_position_metrics(
        self,
        entry_price: float,
        current_price: float,
        position_size: float,
        direction: SignalDirection,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate various position metrics.
        
        Returns:
            Dictionary with metrics like P&L, P&L %, drawdown, etc.
        """
        # Calculate P&L
        if direction == SignalDirection.BUY:
            pnl = (current_price - entry_price) * position_size
        else:
            pnl = (entry_price - current_price) * position_size
        
        pnl_pct = (pnl / (entry_price * position_size)) * 100 if position_size > 0 else 0.0
        
        # Calculate drawdown from peak
        if direction == SignalDirection.BUY:
            drawdown = ((current_price - entry_price) / entry_price) * 100
        else:
            drawdown = ((entry_price - current_price) / entry_price) * 100
        
        # Risk metrics
        metrics = {
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'drawdown_pct': abs(drawdown),
            'current_price': current_price,
            'position_value': current_price * position_size
        }
        
        # Add stop loss metrics if provided
        if stop_loss is not None:
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_reward = ((take_profit - entry_price) / (entry_price - stop_loss)) if take_profit else 0.0
            
            metrics.update({
                'risk_amount': risk_amount,
                'risk_reward_ratio': risk_reward,
                'stop_loss': stop_loss,
                'stop_loss_pct': abs((stop_loss - entry_price) / entry_price * 100)
            })
        
        # Add take profit metrics if provided
        if take_profit is not None:
            metrics.update({
                'take_profit': take_profit,
                'take_profit_pct': abs((take_profit - entry_price) / entry_price * 100)
            })
        
        return metrics
