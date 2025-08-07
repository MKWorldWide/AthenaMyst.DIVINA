import { oandaService } from '../services/oandaService';
import { tradingSettings } from '../config/oanda';
import { formatProphecy } from '../modules/divina';

export interface DivineSignal {
  pair: string;
  direction: 'buy' | 'sell';
  confidence: number;
  entry: number;
  stopLoss: number;
  takeProfit: number;
  message: string;
  celestialInfluence: string;
  elementalFlow: string;
}

export class DivineTradingStrategy {
  private riskPerTrade: number;
  private stopLossPips: number;
  private takeProfitPips: number;

  constructor() {
    this.riskPerTrade = tradingSettings.riskPerTrade;
    this.stopLossPips = tradingSettings.defaultStopLossPips;
    this.takeProfitPips = tradingSettings.defaultTakeProfitPips;
  }

  private getCelestialInfluence() {
    const influences = [
      'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 
      'Uranus', 'Neptune', 'Pluto', 'The Moon', 'The Sun'
    ];
    return influences[Math.floor(Math.random() * influences.length)];
  }

  private getElementalFlow() {
    const elements = ['fire', 'water', 'earth', 'air', 'ether'];
    return elements[Math.floor(Math.random() * elements.length)];
  }

  private getMarketArchetype() {
    const archetypes = [
      'The Alchemist', 'The Oracle', 'The High Priestess',
      'The Magician', 'The Hermit', 'The Star'
    ];
    return archetypes[Math.floor(Math.random() * archetypes.length)];
  }

  private getMarketWisdom() {
    const wisdoms = [
      'In chaos, opportunity dances',
      'Still waters run deep with potential',
      'The market breathes in cycles',
      'Patience in the storm brings wisdom',
      'Even the mightiest wave returns to the sea'
    ];
    return wisdoms[Math.floor(Math.random() * wisdoms.length)];
  }

  async analyzeMarket(pair: string = tradingSettings.defaultPair): Promise<DivineSignal> {
    try {
      // Get market data
      const candles = await oandaService.getCandles(pair, 100, 'H1');
      const currentPrice = await oandaService.getCurrentPrice(pair);
      
      // Simple moving average strategy (replace with your divine algorithm)
      const prices = candles.map((c: any) => parseFloat(c.mid.c));
      const sma20 = prices.slice(0, 20).reduce((a: number, b: number) => a + b, 0) / 20;
      const sma50 = prices.slice(0, 50).reduce((a: number, b: number) => a + b, 0) / 50;
      
      const celestialInfluence = this.getCelestialInfluence();
      const elementalFlow = this.getElementalFlow();
      const marketArchetype = this.getMarketArchetype();
      const marketWisdom = this.getMarketWisdom();
      
      // Generate signal (simplified for example)
      const midPrice = (currentPrice.bid + currentPrice.ask) / 2;
      const pipValue = pair.includes('JPY') ? 0.01 : 0.0001;
      
      let signal: DivineSignal = {
        pair,
        direction: sma20 > sma50 ? 'buy' : 'sell',
        confidence: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
        entry: midPrice,
        stopLoss: 0,
        takeProfit: 0,
        message: '',
        celestialInfluence,
        elementalFlow
      };
      
      // Calculate stop loss and take profit
      if (signal.direction === 'buy') {
        signal.stopLoss = midPrice - (this.stopLossPips * pipValue);
        signal.takeProfit = midPrice + (this.takeProfitPips * pipValue);
        signal.message = `The celestial bodies align for a buying opportunity. ${marketArchetype} whispers of potential growth.`;
      } else {
        signal.stopLoss = midPrice + (this.stopLossPips * pipValue);
        signal.takeProfit = midPrice - (this.takeProfitPips * pipValue);
        signal.message = `The stars warn of a coming retreat. ${marketArchetype} advises caution.`;
      }
      
      return signal;
      
    } catch (error) {
      console.error('Error in divine market analysis:', error);
      throw new Error('Failed to divine the market signs');
    }
  }

  async executeTrade(signal: DivineSignal, units: number = tradingSettings.defaultUnits) {
    try {
      const order = {
        order: {
          units: signal.direction === 'buy' ? units.toString() : (-units).toString(),
          instrument: signal.pair,
          timeInForce: 'FOK',
          type: 'MARKET',
          positionFill: 'DEFAULT',
          stopLossOnFill: {
            price: signal.stopLoss.toFixed(5),
            timeInForce: 'GTC'
          },
          takeProfitOnFill: {
            price: signal.takeProfit.toFixed(5)
          }
        }
      };
      
      const result = await oandaService.placeOrder(order);
      
      // Format divine trade execution message
      const tradeMessage = `🔮 *${signal.direction.toUpperCase()} Order Executed* 🔮\n` +
        `  Pair: ${signal.pair}\n` +
        `  Entry: ${signal.entry.toFixed(5)}\n` +
        `  Stop Loss: ${signal.stopLoss.toFixed(5)}\n` +
        `  Take Profit: ${signal.takeProfit.toFixed(5)}\n` +
        `  ${signal.message}\n` +
        `  Celestial Influence: ${signal.celestialInfluence}\n` +
        `  Elemental Flow: ${signal.elementalFlow}\n`;
      
      return {
        success: true,
        tradeId: result.id,
        message: tradeMessage
      };
      
    } catch (error) {
      console.error('Error executing divine trade:', error);
      return {
        success: false,
        message: 'The fates have denied this trade. The market spirits are restless.'
      };
    }
  }
  
  async getMarketStatus() {
    try {
      const celestialInfluence = this.getCelestialInfluence();
      const elementalFlow = this.getElementalFlow();
      const marketArchetype = this.getMarketArchetype();
      const marketWisdom = this.getMarketWisdom();
      
      const openTrades = await oandaService.getOpenTrades();
      
      const statusMessage = `🔮 *Prophecy from the Castle* 🔮\n\n` +
        `📊 *Market Update* 📊\n` +
        `  Current Aspect: ${celestialInfluence}\n` +
        `  Elemental Flow: ${elementalFlow}\n` +
        `Active Trades: ${openTrades.length}\n` +
        `  ${marketWisdom}\n` +
        `  ${marketArchetype} guides our hand\n`;
      
      return {
        success: true,
        message: statusMessage,
        data: {
          celestialInfluence,
          elementalFlow,
          activeTrades: openTrades.length,
          marketWisdom,
          marketArchetype
        }
      };
      
    } catch (error) {
      console.error('Error getting market status:', error);
      return {
        success: false,
        message: 'The mists of time obscure the market from our sight.'
      };
    }
  }
}

// Export a singleton instance
export const divineStrategy = new DivineTradingStrategy();
