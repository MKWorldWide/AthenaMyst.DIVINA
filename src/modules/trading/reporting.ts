import { RiskManager } from './riskManager';
import { TechnicalIndicators } from './indicators';
import { format, subDays, startOfDay, endOfDay } from 'date-fns';

type TimeFrame = 'daily' | 'weekly' | 'monthly' | 'yearly' | 'all';
type Metric = 'pnl' | 'winRate' | 'sharpeRatio' | 'maxDrawdown' | 'profitFactor';

interface TradeMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnl: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  expectancy: number;
  avgTradeDuration: number; // in hours
}

interface PerformanceReport {
  period: {
    start: Date;
    end: Date;
    timeFrame: TimeFrame;
  };
  metrics: TradeMetrics;
  performanceByTime: Array<{
    date: Date;
    pnl: number;
    cumulativePnl: number;
  }>;
  performanceByAsset: Record<string, {
    pnl: number;
    winRate: number;
    trades: number;
  }>;
  recentTrades: Array<{
    id: string;
    pair: string;
    direction: 'LONG' | 'SHORT';
    entryPrice: number;
    exitPrice: number;
    pnl: number;
    pnlPct: number;
    duration: string;
    openTime: Date;
    closeTime: Date;
  }>;
  riskMetrics: {
    maxDailyDrawdown: number;
    currentDrawdown: number;
    sharpeRatio: number;
    sortinoRatio: number;
    profitFactor: number;
    avgRiskPerTrade: number;
    avgRewardToRisk: number;
  };
  charts: {
    equityCurve: Array<{ date: Date; value: number }>;
    drawdown: Array<{ date: Date; value: number }>;
    monthlyReturns: Array<{ month: string; pnl: number }>;
  };
}

export class TradingReporter {
  private riskManager: RiskManager;
  private timeZone: string;

  constructor(riskManager: RiskManager, timeZone: string = 'UTC') {
    this.riskManager = riskManager;
    this.timeZone = timeZone;
  }

  /**
   * Generate a performance report for the specified time frame
   */
  generatePerformanceReport(timeFrame: TimeFrame = 'all'): PerformanceReport {
    const { startDate, endDate } = this.getDateRange(timeFrame);
    const trades = this.getTradesInDateRange(startDate, endDate);
    const metrics = this.calculateMetrics(trades);
    const performanceByTime = this.calculatePerformanceByTime(trades, timeFrame);
    const performanceByAsset = this.calculatePerformanceByAsset(trades);
    const recentTrades = this.getRecentTrades(trades);
    const riskMetrics = this.calculateRiskMetrics(trades);
    const charts = this.generateCharts(performanceByTime, trades);

    return {
      period: { start: startDate, end: endDate, timeFrame },
      metrics,
      performanceByTime,
      performanceByAsset,
      recentTrades,
      riskMetrics,
      charts
    };
  }

  /**
   * Generate an HTML report from the performance data
   */
  generateHtmlReport(performanceReport: PerformanceReport): string {
    const { period, metrics, riskMetrics, recentTrades } = performanceReport;
    
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading Performance Report</title>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
          .header { text-align: center; margin-bottom: 30px; }
          .summary-cards { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
          .card { background: #f9f9f9; border-radius: 8px; padding: 20px; flex: 1; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
          .card h3 { margin-top: 0; color: #2c3e50; }
          .metric { font-size: 24px; font-weight: bold; margin: 10px 0; }
          .positive { color: #27ae60; }
          .negative { color: #e74c3c; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
          th { background-color: #f2f2f2; font-weight: bold; }
          tr:hover { background-color: #f5f5f5; }
          .chart { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
          .chart-placeholder { background: #f9f9f9; height: 300px; display: flex; align-items: center; justify-content: center; color: #777; }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>🌌 DivineTrader Performance Report</h1>
          <p>${format(period.start, 'MMM d, yyyy')} to ${format(period.end, 'MMM d, yyyy')} (${period.timeFrame})</p>
        </div>

        <div class="summary-cards">
          <div class="card">
            <h3>Total P&L</h3>
            <div class="metric ${metrics.totalPnl >= 0 ? 'positive' : 'negative'}">
              ${metrics.totalPnl >= 0 ? '+' : ''}${metrics.totalPnl.toFixed(2)} (${((metrics.totalPnl / Math.abs(metrics.totalPnl - metrics.avgWin * metrics.winningTrades)) * 100).toFixed(2)}%)
            </div>
            <p>${metrics.winningTrades} wins / ${metrics.losingTrades} losses</p>
          </div>
          
          <div class="card">
            <h3>Win Rate</h3>
            <div class="metric">${(metrics.winRate * 100).toFixed(1)}%</div>
            <p>${metrics.winningTrades} of ${metrics.totalTrades} trades</p>
          </div>
          
          <div class="card">
            <h3>Risk/Reward</h3>
            <div class="metric">${metrics.profitFactor.toFixed(2)}</div>
            <p>Profit Factor</p>
          </div>
          
          <div class="card">
            <h3>Risk Metrics</h3>
            <div class="metric">${(riskMetrics.maxDailyDrawdown * 100).toFixed(1)}%</div>
            <p>Max Drawdown</p>
          </div>
        </div>

        <div class="chart">
          <h2>Equity Curve</h2>
          <div class="chart-placeholder">[Equity Curve Chart]</div>
        </div>

        <div class="chart">
          <h2>Performance by Asset</h2>
          <table>
            <thead>
              <tr>
                <th>Asset</th>
                <th>P&L</th>
                <th>Win Rate</th>
                <th>Trades</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(performanceReport.performanceByAsset)
                .map(([asset, data]) => `
                  <tr>
                    <td>${asset}</td>
                    <td class="${data.pnl >= 0 ? 'positive' : 'negative'}">${data.pnl >= 0 ? '+' : ''}${data.pnl.toFixed(2)}</td>
                    <td>${(data.winRate * 100).toFixed(1)}%</td>
                    <td>${data.trades}</td>
                  </tr>
                `).join('')}
            </tbody>
          </table>
        </div>

        <div class="chart">
          <h2>Recent Trades</h2>
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Pair</th>
                <th>Direction</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              ${recentTrades.map(trade => `
                <tr>
                  <td>${format(trade.openTime, 'MMM d, HH:mm')}</td>
                  <td>${trade.pair}</td>
                  <td>${trade.direction}</td>
                  <td>${trade.entryPrice.toFixed(5)}</td>
                  <td>${trade.exitPrice?.toFixed(5) || '—'}</td>
                  <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">
                    ${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)} (${trade.pnlPct.toFixed(2)}%)
                  </td>
                  <td>${trade.duration}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>

        <div class="chart">
          <h2>Risk Metrics</h2>
          <table>
            <tr>
              <td>Sharpe Ratio</td>
              <td>${riskMetrics.sharpeRatio.toFixed(2)}</td>
            </tr>
            <tr>
              <td>Sortino Ratio</td>
              <td>${riskMetrics.sortinoRatio.toFixed(2)}</td>
            </tr>
            <tr>
              <td>Max Daily Drawdown</td>
              <td>${(riskMetrics.maxDailyDrawdown * 100).toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Average Risk per Trade</td>
              <td>${riskMetrics.avgRiskPerTrade.toFixed(2)}</td>
            </tr>
            <tr>
              <td>Average Reward to Risk</td>
              <td>${riskMetrics.avgRewardToRisk.toFixed(2)}</td>
            </tr>
          </table>
        </div>

        <div class="footer" style="margin-top: 40px; text-align: center; color: #777; font-size: 0.9em;">
          <p>Generated on ${new Date().toISOString()} by AthenaMyst DivineTrader</p>
          <p>This report is for informational purposes only and does not constitute financial advice.</p>
        </div>
      </body>
      </html>
    `;
  }

  /**
   * Generate a text summary of the performance report
   */
  generateTextSummary(performanceReport: PerformanceReport): string {
    const { metrics, period } = performanceReport;
    
    return `
      📊 *DivineTrader Performance Report* 📊
      *Period:* ${format(period.start, 'MMM d, yyyy')} to ${format(period.end, 'MMM d, yyyy')} (${period.timeFrame})
      
      💰 *Performance*
      • P&L: ${metrics.totalPnl >= 0 ? '✅' : '❌'} ${metrics.totalPnl.toFixed(2)} (${((metrics.totalPnl / Math.abs(metrics.totalPnl - metrics.avgWin * metrics.winningTrades)) * 100).toFixed(2)}%)
      • Win Rate: ${(metrics.winRate * 100).toFixed(1)}% (${metrics.winningTrades}/${metrics.totalTrades} trades)
      • Avg Win: ${metrics.avgWin.toFixed(2)} | Avg Loss: ${metrics.avgLoss.toFixed(2)}
      • Profit Factor: ${metrics.profitFactor.toFixed(2)}
      • Max Drawdown: ${(performanceReport.riskMetrics.maxDailyDrawdown * 100).toFixed(1)}%
      
      📈 *Risk Metrics*
      • Sharpe Ratio: ${performanceReport.riskMetrics.sharpeRatio.toFixed(2)}
      • Sortino Ratio: ${performanceReport.riskMetrics.sortinoRatio.toFixed(2)}
      • Avg Risk per Trade: ${performanceReport.riskMetrics.avgRiskPerTrade.toFixed(2)}
      • Avg Reward/Risk: ${performanceReport.riskMetrics.avgRewardToRisk.toFixed(2)}
      
      🔄 *Recent Trades*
      ${performanceReport.recentTrades.slice(0, 5).map(trade => 
        `${trade.pair} ${trade.direction} | ${trade.pnl >= 0 ? '✅' : '❌'} ${trade.pnl.toFixed(2)} (${trade.pnlPct.toFixed(2)}%)`
      ).join('\n      ')}
      
      _Generated on ${new Date().toISOString()}_
    `;
  }

  /**
   * Calculate performance metrics from trades
   */
  private calculateMetrics(trades: any[]): TradeMetrics {
    if (trades.length === 0) {
      return {
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0,
        winRate: 0,
        totalPnl: 0,
        avgWin: 0,
        avgLoss: 0,
        profitFactor: 0,
        maxDrawdown: 0,
        sharpeRatio: 0,
        sortinoRatio: 0,
        expectancy: 0,
        avgTradeDuration: 0
      };
    }

    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl <= 0);
    const totalPnl = trades.reduce((sum, t) => sum + t.pnl, 0);
    const winRate = winningTrades.length / trades.length;
    
    const avgWin = winningTrades.length > 0 
      ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length 
      : 0;
      
    const avgLoss = losingTrades.length > 0 
      ? losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length 
      : 0;
      
    const profitFactor = Math.abs(avgWin * winningTrades.length) / 
      Math.abs(avgLoss * losingTrades.length) || 0;
    
    // Calculate max drawdown
    let peak = trades[0].pnl;
    let maxDrawdown = 0;
    let runningSum = 0;
    
    const drawdowns: number[] = [];
    const returns: number[] = [];
    
    trades.forEach(trade => {
      runningSum += trade.pnl;
      peak = Math.max(peak, runningSum);
      const drawdown = (peak - runningSum) / (peak || 1);
      maxDrawdown = Math.max(maxDrawdown, drawdown);
      drawdowns.push(drawdown);
      returns.push(trade.pnl);
    });
    
    // Calculate Sharpe and Sortino ratios (simplified)
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - avgReturn, 2), 0) / returns.length);
    const downsideReturns = returns.filter(r => r < 0);
    const downsideStdDev = downsideReturns.length > 0 
      ? Math.sqrt(downsideReturns.reduce((a, b) => a + Math.pow(b, 2), 0) / downsideReturns.length)
      : 0;
      
    const sharpeRatio = stdDev !== 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0; // Annualized
    const sortinoRatio = downsideStdDev !== 0 ? (avgReturn / downsideStdDev) * Math.sqrt(252) : 0; // Annualized
    
    // Calculate average trade duration in hours
    const totalDuration = trades.reduce((sum, trade) => {
      if (!trade.closeTime) return sum;
      const duration = (new Date(trade.closeTime).getTime() - new Date(trade.openTime).getTime()) / (1000 * 60 * 60);
      return sum + duration;
    }, 0);
    
    const avgTradeDuration = trades.length > 0 ? totalDuration / trades.length : 0;
    
    return {
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate,
      totalPnl,
      avgWin,
      avgLoss,
      profitFactor,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      expectancy: (winRate * avgWin + (1 - winRate) * avgLoss) / Math.abs(avgLoss || 1),
      avgTradeDuration
    };
  }

  /**
   * Calculate performance over time
   */
  private calculatePerformanceByTime(trades: any[], timeFrame: TimeFrame): Array<{ date: Date; pnl: number; cumulativePnl: number }> {
    // Group trades by time period
    const grouped: Record<string, number> = {};
    
    trades.forEach(trade => {
      let key: string;
      const date = new Date(trade.closeTime || trade.openTime);
      
      switch (timeFrame) {
        case 'daily':
          key = format(date, 'yyyy-MM-dd');
          break;
        case 'weekly':
          key = format(date, "yyyy-'W'ww");
          break;
        case 'monthly':
          key = format(date, 'yyyy-MM');
          break;
        case 'yearly':
          key = format(date, 'yyyy');
          break;
        default:
          key = 'all';
      }
      
      if (!grouped[key]) {
        grouped[key] = 0;
      }
      
      grouped[key] += trade.pnl || 0;
    });
    
    // Convert to array and sort by date
    const result = Object.entries(grouped)
      .map(([date, pnl]) => ({
        date: new Date(date.includes('W') ? date + '1' : date), // Handle week format
        pnl,
        cumulativePnl: 0 // Will be calculated next
      }))
      .sort((a, b) => a.date.getTime() - b.date.getTime());
    
    // Calculate cumulative P&L
    let cumulativePnl = 0;
    return result.map(item => {
      cumulativePnl += item.pnl;
      return { ...item, cumulativePnl };
    });
  }

  /**
   * Calculate performance by asset
   */
  private calculatePerformanceByAsset(trades: any[]): Record<string, { pnl: number; winRate: number; trades: number }> {
    const result: Record<string, { pnl: number; wins: number; trades: number }> = {};
    
    trades.forEach(trade => {
      if (!result[trade.pair]) {
        result[trade.pair] = { pnl: 0, wins: 0, trades: 0 };
      }
      
      result[trade.pair].pnl += trade.pnl || 0;
      result[trade.pair].trades += 1;
      
      if (trade.pnl > 0) {
        result[trade.pair].wins += 1;
      }
    });
    
    // Convert to final format with win rate
    const finalResult: Record<string, { pnl: number; winRate: number; trades: number }> = {};
    
    Object.entries(result).forEach(([pair, data]) => {
      finalResult[pair] = {
        pnl: data.pnl,
        winRate: data.wins / data.trades,
        trades: data.trades
      };
    });
    
    return finalResult;
  }

  /**
   * Get recent trades
   */
  private getRecentTrades(trades: any[], limit: number = 10): any[] {
    return trades
      .sort((a, b) => new Date(b.closeTime || b.openTime).getTime() - new Date(a.closeTime || a.openTime).getTime())
      .slice(0, limit)
      .map(trade => ({
        ...trade,
        pnl: trade.pnl || 0,
        pnlPct: trade.pnlPct || 0,
        duration: this.formatDuration(
          new Date(trade.openTime),
          trade.closeTime ? new Date(trade.closeTime) : new Date()
        )
      }));
  }

  /**
   * Calculate risk metrics
   */
  private calculateRiskMetrics(trades: any[]): {
    maxDailyDrawdown: number;
    currentDrawdown: number;
    sharpeRatio: number;
    sortinoRatio: number;
    profitFactor: number;
    avgRiskPerTrade: number;
    avgRewardToRisk: number;
  } {
    if (trades.length === 0) {
      return {
        maxDailyDrawdown: 0,
        currentDrawdown: 0,
        sharpeRatio: 0,
        sortinoRatio: 0,
        profitFactor: 0,
        avgRiskPerTrade: 0,
        avgRewardToRisk: 0
      };
    }

    // Calculate max drawdown
    let peak = trades[0].pnl;
    let maxDrawdown = 0;
    let runningSum = 0;
    
    trades.forEach(trade => {
      runningSum += trade.pnl;
      peak = Math.max(peak, runningSum);
      const drawdown = (peak - runningSum) / (peak || 1);
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    });
    
    // Calculate current drawdown
    const currentPeak = Math.max(...trades.map(t => t.pnl));
    const currentValue = trades.reduce((sum, t) => sum + t.pnl, 0);
    const currentDrawdown = (currentPeak - currentValue) / (currentPeak || 1);
    
    // Calculate returns for risk metrics
    const returns = trades.map(t => t.pnl);
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev = Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - avgReturn, 2), 0) / returns.length);
    
    // Calculate downside deviation for Sortino ratio
    const downsideReturns = returns.filter(r => r < 0);
    const downsideStdDev = downsideReturns.length > 0 
      ? Math.sqrt(downsideReturns.reduce((a, b) => a + Math.pow(b, 2), 0) / downsideReturns.length)
      : 0;
    
    // Calculate Sharpe and Sortino ratios (annualized)
    const sharpeRatio = stdDev !== 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;
    const sortinoRatio = downsideStdDev !== 0 ? (avgReturn / downsideStdDev) * Math.sqrt(252) : 0;
    
    // Calculate profit factor
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl <= 0);
    
    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
    
    const profitFactor = grossLoss !== 0 ? grossProfit / grossLoss : 0;
    
    // Calculate average risk per trade and reward/risk ratio
    let totalRisk = 0;
    let totalRewardToRisk = 0;
    let validTrades = 0;
    
    trades.forEach(trade => {
      if (trade.riskAmount && trade.riskAmount > 0) {
        totalRisk += trade.riskAmount;
        totalRewardToRisk += (trade.pnl || 0) / trade.riskAmount;
        validTrades++;
      }
    });
    
    const avgRiskPerTrade = validTrades > 0 ? totalRisk / validTrades : 0;
    const avgRewardToRisk = validTrades > 0 ? totalRewardToRisk / validTrades : 0;
    
    return {
      maxDailyDrawdown: maxDrawdown,
      currentDrawdown,
      sharpeRatio,
      sortinoRatio,
      profitFactor,
      avgRiskPerTrade,
      avgRewardToRisk
    };
  }

  /**
   * Generate chart data
   */
  private generateCharts(
    performanceByTime: Array<{ date: Date; pnl: number; cumulativePnl: number }>,
    trades: any[]
  ) {
    // Equity curve
    const equityCurve = performanceByTime.map(item => ({
      date: item.date,
      value: item.cumulativePnl
    }));
    
    // Drawdown curve
    let peak = 0;
    const drawdown = performanceByTime.map(item => {
      peak = Math.max(peak, item.cumulativePnl);
      const dd = peak > 0 ? (peak - item.cumulativePnl) / peak : 0;
      return { date: item.date, value: dd };
    });
    
    // Monthly returns
    const monthlyReturns: Record<string, number> = {};
    
    trades.forEach(trade => {
      if (!trade.closeTime) return;
      
      const month = format(new Date(trade.closeTime), 'yyyy-MM');
      if (!monthlyReturns[month]) {
        monthlyReturns[month] = 0;
      }
      
      monthlyReturns[month] += trade.pnl || 0;
    });    
    
    const monthlyReturnsData = Object.entries(monthlyReturns)
      .map(([month, pnl]) => ({
        month,
        pnl
      }))
      .sort((a, b) => a.month.localeCompare(b.month));
    
    return {
      equityCurve,
      drawdown,
      monthlyReturns: monthlyReturnsData
    };
  }

  /**
   * Get trades within a date range
   */
  private getTradesInDateRange(startDate: Date, endDate: Date): any[] {
    const trades = this.riskManager.getTrades();
    
    return trades.filter(trade => {
      const tradeDate = trade.closeTime ? new Date(trade.closeTime) : new Date(trade.openTime);
      return tradeDate >= startOfDay(startDate) && tradeDate <= endOfDay(endDate);
    });
  }

  /**
   * Get date range based on time frame
   */
  private getDateRange(timeFrame: TimeFrame): { startDate: Date; endDate: Date } {
    const now = new Date();
    
    switch (timeFrame) {
      case 'daily':
        return {
          startDate: startOfDay(now),
          endDate: endOfDay(now)
        };
      case 'weekly':
        return {
          startDate: subDays(startOfDay(now), 7),
          endDate: endOfDay(now)
        };
      case 'monthly':
        return {
          startDate: subDays(startOfDay(now), 30),
          endDate: endOfDay(now)
        };
      case 'yearly':
        return {
          startDate: subDays(startOfDay(now), 365),
          endDate: endOfDay(now)
        };
      case 'all':
      default:
        return {
          startDate: new Date(0), // Beginning of time
          endDate: endOfDay(now)
        };
    }
  }

  /**
   * Format duration between two dates
   */
  private formatDuration(start: Date, end: Date): string {
    const diffMs = end.getTime() - start.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    const diffHrs = Math.floor((diffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    const diffMins = Math.round((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    
    const parts = [];
    if (diffDays > 0) parts.push(`${diffDays}d`);
    if (diffHrs > 0) parts.push(`${diffHrs}h`);
    if (diffMins > 0 && diffDays === 0) parts.push(`${diffMins}m`);
    
    return parts.join(' ') || '0m';
  }
}

export default TradingReporter;
