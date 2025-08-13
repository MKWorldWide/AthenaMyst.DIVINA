""
Trading strategy implementation for AthenaMyst:Divina.

This module implements the core trading strategy that combines technical indicators,
signal generation, and risk management.
"""
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json

from loguru import logger

from ..models import (
    Signal, SignalDirection, SignalStrength, Timeframe, Candle,
    OrderType, OrderStatus, Order, Position, Trade
)
from ..indicators import Indicators
from .signals import SignalManager
from .risk import RiskManager, PositionSizing
from ..data import DataManager
from ..config import settings


@dataclass
class StrategyConfig:
    """Configuration for the trading strategy."""
    # Signal parameters
    signal_timeframe: Timeframe = Timeframe(settings.trading.signal_tf)
    confirm_timeframe: Timeframe = Timeframe(settings.trading.confirm_tf)
    
    # Indicator parameters
    ichimoku_params: Dict[str, int] = field(default_factory=lambda: {
        'tenkan': 9,
        'kijun': 26,
        'senkou_b': 52,
        'displacement': 26
    })
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Volume parameters
    volume_lookback: int = 20
    volume_multiplier: float = 2.0
    
    # Risk parameters
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_leverage: int = 10
    min_reward_risk: float = 1.5
    
    # Position management
    trailing_stop_enabled: bool = True
    trailing_stop_atr_multiplier: float = 2.0
    
    # Cooldown between signals (minutes)
    cooldown_minutes: int = 15
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal_timeframe': str(self.signal_timeframe),
            'confirm_timeframe': str(self.confirm_timeframe),
            'ichimoku_params': self.ichimoku_params,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'volume_lookback': self.volume_lookback,
            'volume_multiplier': self.volume_multiplier,
            'risk_per_trade': self.risk_per_trade,
            'max_leverage': self.max_leverage,
            'min_reward_risk': self.min_reward_risk,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_atr_multiplier': self.trailing_stop_atr_multiplier,
            'cooldown_minutes': self.cooldown_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        config = cls()
        
        # Update with provided values
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle timeframes
        if 'signal_timeframe' in data:
            config.signal_timeframe = Timeframe(data['signal_timeframe'])
        if 'confirm_timeframe' in data:
            config.confirm_timeframe = Timeframe(data['confirm_timeframe'])
        
        return config


class TradingStrategy:
    """Main trading strategy implementation."""
    
    def __init__(
        self,
        data_manager: DataManager,
        config: Optional[StrategyConfig] = None,
        signal_manager: Optional[SignalManager] = None,
        risk_manager: Optional[RiskManager] = None
    ):
        """Initialize the trading strategy."""
        self.data_manager = data_manager
        self.config = config or StrategyConfig()
        self.signal_manager = signal_manager or SignalManager({
            'cooldown_minutes': self.config.cooldown_minutes
        })
        self.risk_manager = risk_manager or RiskManager({
            'risk_per_trade': self.config.risk_per_trade,
            'max_leverage': self.config.max_leverage,
            'min_reward_risk': self.config.min_reward_risk
        })
        
        # Active positions and orders
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        
        # State
        self.is_running = False
        self.last_update = datetime.utcnow()
    
    async def initialize(self) -> None:
        """Initialize the strategy and load any saved state."""
        # Ensure data manager is initialized
        if not hasattr(self.data_manager, 'timeframe_data'):
            await self.data_manager.initialize()
        
        # Load any saved state (positions, orders, etc.)
        await self._load_state()
        
        logger.info("Trading strategy initialized")
    
    async def start(self) -> None:
        """Start the strategy."""
        if self.is_running:
            logger.warning("Strategy is already running")
            return
        
        self.is_running = True
        logger.info("Starting trading strategy")
        
        # Main trading loop
        while self.is_running:
            try:
                await self._run_iteration()
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self.is_running = False
        await self._save_state()
        logger.info("Trading strategy stopped")
    
    async def _run_iteration(self) -> None:
        """Run one iteration of the trading strategy."""
        try:
            # Update market data
            await self.data_manager.update_all()
            
            # Check for new signals on all pairs
            for pair in settings.trading.pairs:
                # Get data for both timeframes
                signal_data = self.data_manager.get_timeframe_data(
                    pair, self.config.signal_timeframe
                )
                confirm_data = self.data_manager.get_timeframe_data(
                    pair, self.config.confirm_timeframe
                )
                
                if not signal_data or not confirm_data:
                    continue
                
                # Generate signals
                signals = await self._generate_signals(pair, signal_data, confirm_data)
                
                # Process signals
                for signal in signals:
                    await self._process_signal(signal)
            
            # Update positions (trailing stops, etc.)
            await self._update_positions()
            
            # Save state periodically
            if (datetime.utcnow() - self.last_update).total_seconds() > 300:  # Every 5 minutes
                await self._save_state()
                self.last_update = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}")
            raise
    
    async def _generate_signals(
        self,
        pair: str,
        signal_data: 'TimeframeData',
        confirm_data: 'TimeframeData'
    ) -> List[Signal]:
        """Generate trading signals based on indicators."""
        signals: List[Signal] = []
        
        # Check if we have enough data
        if not signal_data.candles or not confirm_data.candles:
            return signals
        
        last_candle = signal_data.candles[-1]
        indicators = signal_data.indicators or {}
        
        # Check for valid indicators
        if not indicators:
            return signals
        
        # Check for Ichimoku signals
        ichimoku = indicators.get('ichimoku', {})
        tenkan = ichimoku.get('tenkan_sen', [])
        kijun = ichimoku.get('kijun_sen', [])
        senkou_a = ichimoku.get('senkou_span_a', [])
        senkou_b = ichimoku.get('senkou_span_b', [])
        
        # Check for RSI signals
        rsi = indicators.get('rsi', [])
        
        # Check for volume surge
        volume_profile = indicators.get('volume_profile', {})
        volume_surge = volume_profile.get('volume_surge', False)
        
        # Generate signals based on strategy
        signals.extend(self._check_ichimoku_signals(pair, signal_data, ichimoku))
        signals.extend(self._check_rsi_signals(pair, signal_data, rsi))
        signals.extend(self._check_volume_signals(pair, signal_data, volume_surge))
        
        # Add confirmation from higher timeframe
        confirmed_signals = []
        for signal in signals:
            if await self._confirm_signal(signal, confirm_data):
                confirmed_signals.append(signal)
        
        return confirmed_signals
    
    def _check_ichimoku_signals(
        self,
        pair: str,
        data: 'TimeframeData',
        ichimoku: Dict[str, List[float]]
    ) -> List[Signal]:
        """Generate signals based on Ichimoku Cloud."""
        signals = []
        
        if not all(k in ichimoku for k in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']):
            return signals
        
        tenkan = ichimoku['tenkan_sen']
        kijun = ichimoku['kijun_sen']
        senkou_a = ichimoku['senkou_span_a']
        senkou_b = ichimoku['senkou_span_b']
        
        if not all(len(x) >= 2 for x in [tenkan, kijun, senkou_a, senkou_b]):
            return signals
        
        # Get current and previous values
        tenkan_curr, tenkan_prev = tenkan[-1], tenkan[-2]
        kijun_curr, kijun_prev = kijun[-1], kijun[-2]
        senkou_a_curr = senkou_a[-1]
        senkou_b_curr = senkou_b[-1]
        
        # Price is above/below cloud
        price = data.candles[-1].close
        above_cloud = price > max(senkou_a_curr, senkou_b_curr)
        below_cloud = price < min(senkou_a_curr, senkou_b_curr)
        
        # Tenkan/Kijun cross
        tenkan_above_kijun = tenkan_curr > kijun_curr and tenkan_prev <= kijun_prev
        tenkan_below_kijun = tenkan_curr < kijun_curr and tenkan_prev >= kijun_prev
        
        # Generate signals
        if tenkan_above_kijun and above_cloud:
            signals.append(Signal(
                pair=pair,
                timeframe=data.timeframe,
                direction=SignalDirection.BUY,
                price=price,
                strength=SignalStrength.STRONG,
                indicators={
                    'ichimoku': {
                        'tenkan': tenkan_curr,
                        'kijun': kijun_curr,
                        'senkou_a': senkou_a_curr,
                        'senkou_b': senkou_b_curr,
                        'price_relative_to_cloud': 'above'
                    }
                },
                metadata={
                    'signal_type': 'ichimoku_cloud_breakout',
                    'tenkan_kijun_cross': 'bullish'
                }
            ))
        
        elif tenkan_below_kijun and below_cloud:
            signals.append(Signal(
                pair=pair,
                timeframe=data.timeframe,
                direction=SignalDirection.SELL,
                price=price,
                strength=SignalStrength.STRONG,
                indicators={
                    'ichimoku': {
                        'tenkan': tenkan_curr,
                        'kijun': kijun_curr,
                        'senkou_a': senkou_a_curr,
                        'senkou_b': senkou_b_curr,
                        'price_relative_to_cloud': 'below'
                    }
                },
                metadata={
                    'signal_type': 'ichimoku_cloud_breakdown',
                    'tenkan_kijun_cross': 'bearish'
                }
            ))
        
        return signals
    
    def _check_rsi_signals(
        self,
        pair: str,
        data: 'TimeframeData',
        rsi: List[float]
    ) -> List[Signal]:
        """Generate signals based on RSI."""
        signals = []
        
        if len(rsi) < 2:
            return signals
        
        rsi_curr, rsi_prev = rsi[-1], rsi[-2]
        price = data.candles[-1].close
        
        # RSI crosses above oversold
        if rsi_curr > self.config.rsi_oversold and rsi_prev <= self.config.rsi_oversold:
            signals.append(Signal(
                pair=pair,
                timeframe=data.timeframe,
                direction=SignalDirection.BUY,
                price=price,
                strength=SignalStrength.MODERATE,
                indicators={'rsi': rsi_curr},
                metadata={'signal_type': 'rsi_oversold_bounce'}
            ))
        
        # RSI crosses below overbought
        elif rsi_curr < self.config.rsi_overbought and rsi_prev >= self.config.rsi_overbought:
            signals.append(Signal(
                pair=pair,
                timeframe=data.timeframe,
                direction=SignalDirection.SELL,
                price=price,
                strength=SignalStrength.MODERATE,
                indicators={'rsi': rsi_curr},
                metadata={'signal_type': 'rsi_overbought_rejection'}
            ))
        
        return signals
    
    def _check_volume_signals(
        self,
        pair: str,
        data: 'TimeframeData',
        volume_surge: bool
    ) -> List[Signal]:
        """Generate signals based on volume profile."""
        if not volume_surge or not data.candles:
            return []
        
        last_candle = data.candles[-1]
        
        # Volume surge with price movement
        if last_candle.close > last_candle.open:  # Bullish candle
            return [Signal(
                pair=pair,
                timeframe=data.timeframe,
                direction=SignalDirection.BUY,
                price=last_candle.close,
                strength=SignalStrength.WEAK,
                indicators={'volume_surge': True},
                metadata={'signal_type': 'volume_surge_bullish'}
            )]
        elif last_candle.close < last_candle.open:  # Bearish candle
            return [Signal(
                pair=pair,
                timeframe=data.timeframe,
                direction=SignalDirection.SELL,
                price=last_candle.close,
                strength=SignalStrength.WEAK,
                indicators={'volume_surge': True},
                metadata={'signal_type': 'volume_surge_bearish'}
            )]
        
        return []
    
    async def _confirm_signal(
        self,
        signal: Signal,
        confirm_data: 'TimeframeData'
    ) -> bool:
        """Confirm a signal with higher timeframe analysis."""
        if not confirm_data.indicators or not confirm_data.candles:
            return False
        
        # Simple confirmation: price should be above/below key levels on higher TF
        price = signal.price
        indicators = confirm_data.indicators
        
        # Check Ichimoku cloud on higher timeframe
        ichimoku = indicators.get('ichimoku', {})
        senkou_a = ichimoku.get('senkou_span_a', [])
        senkou_b = ichimoku.get('senkou_span_b', [])
        
        if senkou_a and senkou_b:
            cloud_top = max(senkou_a[-1], senkou_b[-1])
            cloud_bottom = min(senkou_a[-1], senkou_b[-1])
            
            if signal.direction == SignalDirection.BUY:
                return price > cloud_top
            else:  # SELL
                return price < cloud_bottom
        
        return True  # If no confirmation available, proceed with signal
    
    async def _process_signal(self, signal: Signal) -> None:
        """Process a trading signal."""
        # Process through signal manager (handles deduplication, cooldown, etc.)
        processed_signal = self.signal_manager.process_signal(signal)
        if not processed_signal:
            return
        
        # Calculate position sizing and risk parameters
        try:
            position_sizing = await self._calculate_position_sizing(processed_signal)
            
            # Create order
            order = await self._create_order(processed_signal, position_sizing)
            if order:
                logger.info(f"Created order: {order}")
                self.orders[order.id] = order
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def _calculate_position_sizing(
        self,
        signal: Signal
    ) -> PositionSizing:
        """Calculate position sizing and risk parameters."""
        # Get recent candles for volatility calculation
        candles = self.data_manager.get_timeframe_data(
            signal.pair, signal.timeframe
        ).candles[-100:]  # Last 100 candles
        
        # Calculate ATR for stop loss
        atr = Indicators.calculate_atr(
            [c.high for c in candles],
            [c.low for c in candles],
            [c.close for c in candles],
            period=14
        )
        
        # Calculate stop loss and take profit
        if signal.direction == SignalDirection.BUY:
            stop_loss = signal.price - (atr[-1] * 2)  # 2x ATR stop
            take_profit = signal.price + (atr[-1] * 4)  # 4x ATR target (1:2 RR)
        else:  # SELL
            stop_loss = signal.price + (atr[-1] * 2)
            take_profit = signal.price - (atr[-1] * 4)
        
        # Calculate position size
        return self.risk_manager.calculate_position_size(
            pair=signal.pair,
            entry_price=signal.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=signal.direction,
            risk_percentage=self.config.risk_per_trade,
            leverage=self.risk_manager.leverage
        )
    
    async def _create_order(
        self,
        signal: Signal,
        position_sizing: PositionSizing
    ) -> Optional[Order]:
        """Create an order based on the signal and position sizing."""
        # Check if we already have an open position for this pair
        if signal.pair in self.positions:
            position = self.positions[signal.pair]
            
            # Check if we should close the position first
            if position.direction != signal.direction:
                # Close existing position before opening a new one
                await self._close_position(position, "Opposite signal")
            else:
                # Same direction - skip or average in (implement as needed)
                logger.info(f"Position already open for {signal.pair} in same direction")
                return None
        
        # Create a new order
        order = Order(
            id=f"order_{int(datetime.utcnow().timestamp())}",
            pair=signal.pair,
            direction=signal.direction,
            order_type=OrderType.MARKET,  # or LIMIT with price
            size=position_sizing.size,
            price=signal.price,
            stop_loss=position_sizing.stop_loss,
            take_profit=position_sizing.take_profit,
            status=OrderStatus.NEW,
            created_at=datetime.utcnow(),
            metadata={
                'signal_id': signal.id,
                'risk_percentage': self.risk_manager.risk_per_trade,
                'leverage': position_sizing.leverage,
                'reward_risk_ratio': position_sizing.reward_risk_ratio
            }
        )
        
        # Here you would typically send the order to your broker/execution system
        # For now, we'll just log it
        logger.info(f"Order created: {order}")
        
        # Simulate order execution
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        
        # Create a position
        position = Position(
            id=f"pos_{order.id}",
            pair=order.pair,
            direction=order.direction,
            size=order.size,
            entry_price=order.price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=order.filled_at,
            metadata=order.metadata
        )
        
        self.positions[order.pair] = position
        
        # Record the trade
        trade = Trade(
            id=f"trade_{order.id}",
            order_id=order.id,
            position_id=position.id,
            pair=order.pair,
            direction=order.direction,
            size=order.size,
            entry_price=order.price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_at=order.filled_at,
            status='OPEN',
            metadata=order.metadata
        )
        
        self.trades.append(trade)
        
        return order
    
    async def _update_positions(self) -> None:
        """Update open positions (trailing stops, take profit, etc.)."""
        if not self.positions:
            return
        
        for pair, position in list(self.positions.items()):
            try:
                # Get current market data
                data = self.data_manager.get_timeframe_data(
                    pair, self.config.signal_timeframe
                )
                
                if not data or not data.candles:
                    continue
                
                current_price = data.candles[-1].close
                
                # Check for take profit or stop loss
                if (position.direction == SignalDirection.BUY and 
                    (current_price >= position.take_profit or 
                     current_price <= position.stop_loss)):
                    # Close position
                    await self._close_position(
                        position,
                        "Take profit/stop loss hit"
                    )
                elif (position.direction == SignalDirection.SELL and 
                      (current_price <= position.take_profit or 
                       current_price >= position.stop_loss)):
                    # Close position
                    await self._close_position(
                        position,
                        "Take profit/stop loss hit"
                    )
                
                # Update trailing stop if enabled
                elif self.config.trailing_stop_enabled:
                    await self._update_trailing_stop(position, data.candles)
            
            except Exception as e:
                logger.error(f"Error updating position {position.id}: {e}")
    
    async def _update_trailing_stop(
        self,
        position: Position,
        candles: List[Candle]
    ) -> None:
        """Update trailing stop for a position."""
        if len(candles) < 20:  # Need enough data for ATR
            return
        
        current_price = candles[-1].close
        
        # Calculate ATR for dynamic stop
        highs = [c.high for c in candles[-20:]]
        lows = [c.low for c in candles[-20:]]
        closes = [c.close for c in candles[-21:-1]]  # Previous closes
        
        atr = Indicators.calculate_atr(highs, lows, closes, period=14)
        if not atr:
            return
        
        # Calculate new trailing stop
        new_stop = self.risk_manager.calculate_trailing_stop(
            entry_price=position.entry_price,
            current_price=current_price,
            direction=position.direction,
            atr=atr[-1],
            atr_multiplier=self.config.trailing_stop_atr_multiplier
        )
        
        # Only move stop in the favorable direction
        if (position.direction == SignalDirection.BUY and 
            new_stop > position.stop_loss):
            position.stop_loss = new_stop
            logger.info(f"Updated trailing stop for {position.id} to {new_stop}")
        
        elif (position.direction == SignalDirection.SELL and 
              new_stop < position.stop_loss):
            position.stop_loss = new_stop
            logger.info(f"Updated trailing stop for {position.id} to {new_stop}")
    
    async def _close_position(
        self,
        position: Position,
        reason: str = ""
    ) -> None:
        """Close an open position."""
        try:
            # Get current price for exit
            data = self.data_manager.get_timeframe_data(
                position.pair, self.config.signal_timeframe
            )
            
            if not data or not data.candles:
                logger.warning(f"No data to close position {position.id}")
                return
            
            exit_price = data.candles[-1].close
            exit_time = datetime.utcnow()
            
            # Calculate P&L
            if position.direction == SignalDirection.BUY:
                pnl = (exit_price - position.entry_price) * position.size
            else:  # SELL
                pnl = (position.entry_price - exit_price) * position.size
            
            pnl_pct = (pnl / (position.entry_price * position.size)) * 100
            
            # Update position
            position.exit_price = exit_price
            position.closed_at = exit_time
            position.status = 'CLOSED'
            position.pnl = pnl
            position.pnl_pct = pnl_pct
            
            # Update trade
            for trade in reversed(self.trades):
                if trade.position_id == position.id and trade.status == 'OPEN':
                    trade.exit_price = exit_price
                    trade.closed_at = exit_time
                    trade.status = 'CLOSED'
                    trade.pnl = pnl
                    trade.pnl_pct = pnl_pct
                    trade.metadata['exit_reason'] = reason
                    break
            
            # Remove from active positions
            if position.pair in self.positions:
                del self.positions[position.pair]
            
            logger.info(
                f"Closed position {position.id} at {exit_price} "
                f"(P&L: {pnl:.2f} {position.pair.split('_')[-1]}, {pnl_pct:.2f}%)"
            )
        
        except Exception as e:
            logger.error(f"Error closing position {position.id}: {e}")
    
    async def _save_state(self) -> None:
        """Save strategy state to persistent storage."""
        try:
            state = {
                'positions': {k: v.to_dict() for k, v in self.positions.items()},
                'orders': {k: v.to_dict() for k, v in self.orders.items()},
                'trades': [t.to_dict() for t in self.trades],
                'last_updated': datetime.utcnow().isoformat(),
                'config': self.config.to_dict()
            }
            
            # Save to file (in a real app, use a database)
            state_file = Path(settings.data_dir) / 'strategy_state.json'
            with open(state_file, 'w') as f:
                json.dump(state, f, default=str)
            
            logger.debug("Strategy state saved")
        
        except Exception as e:
            logger.error(f"Error saving strategy state: {e}")
    
    async def _load_state(self) -> None:
        """Load strategy state from persistent storage."""
        try:
            state_file = Path(settings.data_dir) / 'strategy_state.json'
            
            if not state_file.exists():
                logger.info("No saved state found, starting fresh")
                return
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Load config
            if 'config' in state:
                self.config = StrategyConfig.from_dict(state['config'])
            
            # Load positions
            self.positions = {
                k: Position.from_dict(v) for k, v in state.get('positions', {}).items()
            }
            
            # Load orders
            self.orders = {
                k: Order.from_dict(v) for k, v in state.get('orders', {}).items()
            }
            
            # Load trades
            self.trades = [
                Trade.from_dict(t) for t in state.get('trades', [])
            ]
            
            logger.info(
                f"Loaded strategy state: {len(self.positions)} positions, "
                f"{len(self.orders)} orders, {len(self.trades)} trades"
            )
        
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")
            # Start fresh if there's an error
            self.positions = {}
            self.orders = {}
            self.trades = []


# Factory function to create a strategy
def create_strategy(
    data_manager: DataManager,
    config: Optional[Dict[str, Any]] = None
) -> TradingStrategy:
    """Create and initialize a trading strategy."""
    strategy_config = StrategyConfig()
    
    # Update with provided config
    if config:
        strategy_config = StrategyConfig.from_dict({
            **strategy_config.to_dict(),
            **config
        })
    
    return TradingStrategy(
        data_manager=data_manager,
        config=strategy_config
    )
