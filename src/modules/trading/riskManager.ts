import TechnicalIndicators from './indicators';

type Trade = {
  id: string;
  pair: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  stopLoss?: number;
  takeProfit?: number;
  size: number;
  riskAmount: number;
  status: 'OPEN' | 'CLOSED' | 'PENDING';
  openTime: string;
  closeTime?: string;
  closePrice?: number;
  pnl?: number;
};

type RiskParameters = {
  maxAccountRiskPerTrade: number;  // Max % of account to risk per trade (e.g., 0.01 for 1%)
  maxDailyDrawdown: number;        // Max daily drawdown % before stopping trading
  maxPositionSize: number;         // Max position size as % of account
  volatilityPeriod: number;        // Lookback period for volatility calculation
  atrMultiplier: number;           // Multiplier for ATR-based position sizing
};

export class RiskManager {
  private accountBalance: number;
  private dailyPnL: number = 0;
  private dailyHighBalance: number;
  private riskParams: RiskParameters;
  private trades: Trade[] = [];
  
  constructor(initialBalance: number, params: Partial<RiskParameters> = {}) {
    this.accountBalance = initialBalance;
    this.dailyHighBalance = initialBalance;
    
    // Default risk parameters
    this.riskParams = {
      maxAccountRiskPerTrade: 0.01,  // 1% risk per trade
      maxDailyDrawdown: 0.05,        // 5% max daily drawdown
      maxPositionSize: 0.1,          // 10% of account per position
      volatilityPeriod: 14,          // 14-period lookback
      atrMultiplier: 2,              // 2x ATR for position sizing
      ...params
    };
  }
  
  /**
   * Update account balance and track daily P&L
   */
  updateAccountBalance(newBalance: number): void {
    this.accountBalance = newBalance;
    this.dailyHighBalance = Math.max(this.dailyHighBalance, newBalance);
    this.dailyPnL = this.accountBalance - this.dailyHighBalance;
  }
  
  /**
   * Calculate position size based on risk parameters
   */
  calculatePositionSize(
    entryPrice: number,
    stopLoss: number,
    atr: number,
    volatility: number = 1
  ): { size: number; riskAmount: number } {
    // Check if we've hit daily drawdown limit
    const drawdownPct = Math.abs(this.dailyPnL) / this.dailyHighBalance;
    if (drawdownPct >= this.riskParams.maxDailyDrawdown) {
      throw new Error(`Daily drawdown limit (${this.riskParams.maxDailyDrawdown * 100}%) reached`);
    }
    
    // Calculate risk amount based on account balance and risk per trade
    const riskAmount = this.accountBalance * this.riskParams.maxAccountRiskPerTrade * (1 / Math.max(1, volatility));
    
    // Calculate position size based on stop loss
    const riskPerUnit = Math.abs(entryPrice - stopLoss);
    let size = riskAmount / riskPerUnit;
    
    // Adjust position size based on volatility using ATR
    const atrBasedSize = (this.accountBalance * this.riskParams.maxPositionSize) / (atr * this.riskParams.atrMultiplier);
    size = Math.min(size, atrBasedSize);
    
    // Ensure position size doesn't exceed maximum position size
    const maxPositionSize = this.accountBalance * this.riskParams.maxPositionSize;
    const positionValue = size * entryPrice;
    
    if (positionValue > maxPositionSize) {
      size = maxPositionSize / entryPrice;
    }
    
    return {
      size: parseFloat(size.toFixed(8)), // Round to 8 decimal places
      riskAmount: parseFloat(riskAmount.toFixed(2))
    };
  }
  
  /**
   * Check if a new trade can be opened based on risk parameters
   */
  canOpenNewTrade(currentOpenTrades: number): { canTrade: boolean; reason?: string } {
    // Check daily drawdown limit
    const drawdownPct = Math.abs(this.dailyPnL) / this.dailyHighBalance;
    if (drawdownPct >= this.riskParams.maxDailyDrawdown) {
      return {
        canTrade: false,
        reason: `Daily drawdown limit (${this.riskParams.maxDailyDrawdown * 100}%) reached`
      };
    }
    
    // Check maximum number of open trades (optional)
    const maxOpenTrades = Math.floor(1 / this.riskParams.maxPositionSize);
    if (currentOpenTrades >= maxOpenTrades) {
      return {
        canTrade: false,
        reason: `Maximum number of open trades (${maxOpenTrades}) reached`
      };
    }
    
    return { canTrade: true };
  }
  
  /**
   * Calculate stop loss and take profit levels based on volatility
   */
  calculateStopLossAndTakeProfit(
    entryPrice: number,
    direction: 'LONG' | 'SHORT',
    atr: number,
    riskRewardRatio: number = 2
  ): { stopLoss: number; takeProfit: number } {
    // Use ATR for dynamic stop loss
    const stopLoss = direction === 'LONG' 
      ? entryPrice - (atr * this.riskParams.atrMultiplier)
      : entryPrice + (atr * this.riskParams.atrMultiplier);
    
    // Calculate take profit based on risk-reward ratio
    const riskAmount = Math.abs(entryPrice - stopLoss);
    const takeProfit = direction === 'LONG'
      ? entryPrice + (riskAmount * riskRewardRatio)
      : entryPrice - (riskAmount * riskRewardRatio);
    
    return {
      stopLoss: parseFloat(stopLoss.toFixed(5)),
      takeProfit: parseFloat(takeProfit.toFixed(5))
    };
  }
  
  /**
   * Calculate position value as a percentage of account
   */
  getPositionValueAsPercentage(positionValue: number): number {
    return (positionValue / this.accountBalance) * 100;
  }
  
  /**
   * Get current risk metrics
   */
  getRiskMetrics() {
    const openTrades = this.trades.filter(t => t.status === 'OPEN');
    const totalRisk = openTrades.reduce((sum, trade) => sum + trade.riskAmount, 0);
    const totalPositionValue = openTrades.reduce((sum, trade) => {
      return sum + (trade.size * trade.entryPrice);
    }, 0);
    
    return {
      accountBalance: this.accountBalance,
      dailyHighBalance: this.dailyHighBalance,
      dailyPnL: this.dailyPnL,
      dailyPnLPct: (this.dailyPnL / this.dailyHighBalance) * 100,
      totalRisk,
      totalRiskPct: (totalRisk / this.accountBalance) * 100,
      totalPositionValue,
      totalPositionValuePct: this.getPositionValueAsPercentage(totalPositionValue),
      maxDailyDrawdown: this.riskParams.maxDailyDrawdown * 100,
      currentDrawdownPct: Math.abs(this.dailyPnL) / this.dailyHighBalance * 100,
      openTradesCount: openTrades.length
    };
  }
  
  /**
   * Reset daily metrics (to be called at the start of each trading day)
   */
  resetDailyMetrics(): void {
    this.dailyHighBalance = this.accountBalance;
    this.dailyPnL = 0;
  }
  
  /**
   * Add a new trade to the risk manager
   */
  addTrade(trade: Omit<Trade, 'riskAmount'>): void {
    this.trades.push({
      ...trade,
      riskAmount: Math.abs(trade.entryPrice - (trade.stopLoss || 0)) * trade.size
    });
  }
  
  /**
   * Update a trade (e.g., when closed)
   */
  updateTrade(tradeId: string, updates: Partial<Trade>): void {
    const tradeIndex = this.trades.findIndex(t => t.id === tradeId);
    if (tradeIndex !== -1) {
      this.trades[tradeIndex] = { ...this.trades[tradeIndex], ...updates };
      
      // Update risk amount if stop loss changed
      if (updates.stopLoss !== undefined) {
        this.trades[tradeIndex].riskAmount = Math.abs(
          this.trades[tradeIndex].entryPrice - updates.stopLoss
        ) * this.trades[tradeIndex].size;
      }
    }
  }
  
  /**
   * Get all trades
   */
  getTrades(): Trade[] {
    return [...this.trades];
  }
  
  /**
   * Get open trades
   */
  getOpenTrades(): Trade[] {
    return this.trades.filter(t => t.status === 'OPEN');
  }
  
  /**
   * Get trade history
   */
  getTradeHistory(): Trade[] {
    return this.trades.filter(t => t.status === 'CLOSED');
  }
}

export default RiskManager;
