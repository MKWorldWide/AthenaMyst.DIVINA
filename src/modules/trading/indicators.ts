/**
 * Technical Indicators Module
 * 
 * This module provides various technical indicators for market analysis
 * and trading strategy development.
 */

type Candle = {
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  time: string | Date;
};

type IndicatorResult = {
  value: number;
  signal?: 'buy' | 'sell' | 'neutral';
  metadata?: Record<string, any>;
};

export class TechnicalIndicators {
  /**
   * Calculate Simple Moving Average (SMA)
   */
  static sma(prices: number[], period: number): number[] {
    if (prices.length < period) return [];
    
    const result: number[] = [];
    for (let i = period - 1; i < prices.length; i++) {
      const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
    return result;
  }

  /**
   * Calculate Exponential Moving Average (EMA)
   */
  static ema(prices: number[], period: number): number[] {
    if (prices.length < period) return [];
    
    const k = 2 / (period + 1);
    const result: number[] = [];
    
    // Use SMA for the first value
    let ema = this.sma(prices.slice(0, period), period)[0];
    result.push(ema);
    
    // Calculate EMA for subsequent values
    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] - ema) * k + ema;
      result.push(ema);
    }
    
    return result;
  }

  /**
   * Calculate Relative Strength Index (RSI)
   */
  static rsi(prices: number[], period: number = 14): IndicatorResult[] {
    if (prices.length < period + 1) return [];
    
    const result: IndicatorResult[] = [];
    const gains: number[] = [0];
    const losses: number[] = [0];
    
    // Calculate initial average gains and losses
    for (let i = 1; i < prices.length; i++) {
      const diff = prices[i] - prices[i - 1];
      gains.push(diff > 0 ? diff : 0);
      losses.push(diff < 0 ? -diff : 0);
    }
    
    // Calculate first RSI value
    let avgGain = this.sma(gains.slice(1, period + 1), period)[0];
    let avgLoss = this.sma(losses.slice(1, period + 1), period)[0];
    
    if (avgLoss === 0) {
      result.push({ value: 100, signal: 'overbought' });
    } else {
      const rs = avgGain / avgLoss;
      const rsi = 100 - (100 / (1 + rs));
      result.push({
        value: rsi,
        signal: this.getRsiSignal(rsi)
      });
    }
    
    // Calculate subsequent RSI values
    for (let i = period + 1; i < prices.length; i++) {
      avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
      avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
      
      if (avgLoss === 0) {
        result.push({ value: 100, signal: 'overbought' });
      } else {
        const rs = avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));
        result.push({
          value: rsi,
          signal: this.getRsiSignal(rsi)
        });
      }
    }
    
    return result;
  }
  
  /**
   * Calculate Moving Average Convergence Divergence (MACD)
   */
  static macd(prices: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): {
    macd: number[];
    signal: number[];
    histogram: number[];
  } {
    const fastEMA = this.ema(prices, fastPeriod);
    const slowEMA = this.ema(prices, slowPeriod);
    
    // Calculate MACD line
    const macdLine: number[] = [];
    const minLength = Math.min(fastEMA.length, slowEMA.length);
    
    for (let i = 0; i < minLength; i++) {
      macdLine.push(fastEMA[i] - slowEMA[i]);
    }
    
    // Calculate signal line (EMA of MACD line)
    const signalLine = this.ema(macdLine, signalPeriod);
    
    // Calculate histogram
    const histogram: number[] = [];
    const signalStart = macdLine.length - signalLine.length;
    
    for (let i = 0; i < signalLine.length; i++) {
      histogram.push(macdLine[signalStart + i] - signalLine[i]);
    }
    
    return {
      macd: macdLine.slice(signalStart),
      signal: signalLine,
      histogram
    };
  }
  
  /**
   * Calculate Bollinger Bands
   */
  static bollingerBands(prices: number[], period: number = 20, stdDev: number = 2): {
    middle: number[];
    upper: number[];
    lower: number[];
    bandwidth: number[];
  } {
    const sma = this.sma(prices, period);
    const middle: number[] = [];
    const upper: number[] = [];
    const lower: number[] = [];
    const bandwidth: number[] = [];
    
    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      const avg = sma[i - period + 1];
      const variance = slice.reduce((sum, price) => sum + Math.pow(price - avg, 2), 0) / period;
      const stdDevValue = Math.sqrt(variance);
      
      const upperBand = avg + (stdDev * stdDevValue);
      const lowerBand = avg - (stdDev * stdDevValue);
      
      middle.push(avg);
      upper.push(upperBand);
      lower.push(lowerBand);
      bandwidth.push((upperBand - lowerBand) / avg * 100);
    }
    
    return { middle, upper, lower, bandwidth };
  }
  
  /**
   * Calculate Average True Range (ATR)
   */
  static atr(candles: Candle[], period: number = 14): number[] {
    if (candles.length < period) return [];
    
    const trueRanges: number[] = [];
    
    // Calculate True Range for each candle
    for (let i = 1; i < candles.length; i++) {
      const current = candles[i];
      const previous = candles[i - 1];
      
      const highLow = current.high - current.low;
      const highPrevClose = Math.abs(current.high - previous.close);
      const lowPrevClose = Math.abs(current.low - previous.close);
      
      trueRanges.push(Math.max(highLow, highPrevClose, lowPrevClose));
    }
    
    // Calculate ATR
    const atrValues: number[] = [];
    let atr = this.sma(trueRanges.slice(0, period), period)[0];
    atrValues.push(atr);
    
    for (let i = period; i < trueRanges.length; i++) {
      atr = ((atr * (period - 1)) + trueRanges[i]) / period;
      atrValues.push(atr);
    }
    
    return atrValues;
  }
  
  /**
   * Get RSI signal based on value
   */
  private static getRsiSignal(rsi: number): 'overbought' | 'oversold' | 'neutral' {
    if (rsi >= 70) return 'overbought';
    if (rsi <= 30) return 'oversold';
    return 'neutral';
  }
  
  /**
   * Get MACD signal based on MACD and signal line values
   */
  static getMacdSignal(macd: number, signal: number): 'buy' | 'sell' | 'neutral' {
    if (macd > signal) return 'buy';
    if (macd < signal) return 'sell';
    return 'neutral';
  }
  
  /**
   * Get Bollinger Bands signal
   */
  static getBollingerSignal(price: number, upper: number, lower: number): 'overbought' | 'oversold' | 'neutral' {
    if (price >= upper) return 'overbought';
    if (price <= lower) return 'oversold';
    return 'neutral';
  }
}

// Export as default for easier imports
export default TechnicalIndicators;
