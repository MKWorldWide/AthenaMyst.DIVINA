import { v4 as uuidv4 } from 'uuid';
import { formatProphecy } from '../divina';
import { oandaService } from '../../services/oandaService';
import { divineStrategy } from '../../strategies/divineTradingStrategy';
import { tradingSettings } from '../../config/oanda';

// Types for our divine trading system
type TradeStatus = 'PENDING' | 'OPEN' | 'CLOSED' | 'STOPPED_OUT' | 'TAKE_PROFIT';
type TradeDirection = 'LONG' | 'SHORT';

interface DivineTrade {
  id: string;
  tradeId?: string;  // OANDA trade ID
  pair: string;
  direction: TradeDirection;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  status: TradeStatus;
  openTime: Date;
  closeTime?: Date;
  closePrice?: number;
  pips?: number;
  profitLoss?: number;
  units: number;
  spiritualNote: string;
  celestialInfluence: string;
  elementalFlow: string;
}

export class DivineTrader {
  private activeTrades: DivineTrade[] = [];
  private tradeHistory: DivineTrade[] = [];
  
  private spiritualAffirmations: string[] = [
    'The stars align in our favor',
    'Patience in the storm brings wisdom',
    'In chaos, opportunity dances',
    'Still waters run deep with potential',
    'The market breathes in cycles',
    'Even the mightiest wave returns to the sea'
  ];

  constructor() {
    this.syncWithOanda();
  }

  private getRandomAffirmation(): string {
    return this.spiritualAffirmations[Math.floor(Math.random() * this.spiritualAffirmations.length)];
  }

  private async syncWithOanda() {
    try {
      // Get open trades from OANDA
      const openTrades = await oandaService.getOpenTrades();
      
      // Update our active trades
      this.activeTrades = await Promise.all(openTrades.map(async (trade: any) => {
        return {
          id: uuidv4(),
          tradeId: trade.id,
          pair: trade.instrument,
          direction: parseFloat(trade.currentUnits) > 0 ? 'LONG' : 'SHORT',
          entryPrice: parseFloat(trade.price),
          stopLoss: trade.stopLossOrder ? parseFloat(trade.stopLossOrder.price) : 0,
          takeProfit: trade.takeProfitOrder ? parseFloat(trade.takeProfitOrder.price) : 0,
          status: 'OPEN' as TradeStatus,
          openTime: new Date(trade.openTime),
          units: Math.abs(parseInt(trade.currentUnits)),
          spiritualNote: this.getRandomAffirmation(),
          celestialInfluence: await this.getCelestialInfluence(),
          elementalFlow: this.getElementalFlow()
        };
      }));
      
      console.log(`Synced ${this.activeTrades.length} active trades with OANDA`);
    } catch (error) {
      console.error('Error syncing with OANDA:', error);
    }
  }

  private getCelestialInfluence(): Promise<string> {
    const influences = [
      'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 
      'Uranus', 'Neptune', 'Pluto', 'The Moon', 'The Sun'
    ];
    return Promise.resolve(influences[Math.floor(Math.random() * influences.length)]);
  }

  private getElementalFlow(): string {
    const elements = ['fire', 'water', 'earth', 'air', 'ether'];
    return elements[Math.floor(Math.random() * elements.length)];
  }

  public async executeScalpTrade(
    pair: string, 
    price: number
  ): Promise<{ success: boolean; message: string; tradeId?: string }> {
    try {
      // Generate divine signal
      const signal = await divineStrategy.analyzeMarket(pair);
      
      // Execute the trade through OANDA
      const result = await divineStrategy.executeTrade(signal);
      
      if (!result.success) {
        return {
          success: false,
          message: `❌ Failed to execute scalp trade: ${result.message}`
        };
      }
      
      // Create trade record
      const trade: DivineTrade = {
        id: uuidv4(),
        tradeId: result.tradeId,
        pair: signal.pair,
        direction: signal.direction.toUpperCase() as 'LONG' | 'SHORT',
        entryPrice: signal.entry,
        stopLoss: signal.stopLoss,
        takeProfit: signal.takeProfit,
        status: 'OPEN',
        openTime: new Date(),
        units: tradingSettings.defaultUnits,
        spiritualNote: this.getRandomAffirmation(),
        celestialInfluence: signal.celestialInfluence,
        elementalFlow: signal.elementalFlow
      };
      
      this.activeTrades.push(trade);
      
      return {
        success: true,
        message: result.message,
        tradeId: trade.id
      };
      
    } catch (error) {
      console.error('Error executing scalp trade:', error);
      return {
        success: false,
        message: '❌ The market spirits resist our attempts to trade. The signs are unclear.'
      };
    }
  }

  public async closeTrade(
    tradeId: string, 
    closePrice: number
  ): Promise<{ success: boolean; message: string }> {
    try {
      const tradeIndex = this.activeTrades.findIndex(t => t.id === tradeId);
      if (tradeIndex === -1) {
        return {
          success: false,
          message: '❌ Trade not found in active trades.'
        };
      }

      const trade = this.activeTrades[tradeIndex];
      
      // Close the trade through OANDA if we have a tradeId
      if (trade.tradeId) {
        await oandaService.closeTrade(trade.tradeId);
      }
      
      // Calculate P&L
      const pips = this.calculatePips(trade, closePrice);
      const profitLoss = this.calculateProfitLoss(trade, closePrice);
      
      // Update trade status
      const closedTrade: DivineTrade = {
        ...trade,
        status: 'CLOSED',
        closeTime: new Date(),
        closePrice,
        pips,
        profitLoss
      };
      
      // Move to history
      this.activeTrades.splice(tradeIndex, 1);
      this.tradeHistory.push(closedTrade);
      
      return {
        success: true,
        message: `✅ Trade closed successfully. ${profitLoss >= 0 ? 'Profit' : 'Loss'}: ${Math.abs(profitLoss).toFixed(2)} (${pips} pips)`
      };
      
    } catch (error) {
      console.error('Error closing trade:', error);
      return {
        success: false,
        message: '❌ Failed to close trade. The market spirits resist.'
      };
    }
  }

  private calculatePips(trade: DivineTrade, currentPrice: number): number {
    const priceDifference = Math.abs(currentPrice - trade.entryPrice);
    const pipValue = trade.pair.includes('JPY') ? 0.01 : 0.0001;
    return Math.round(priceDifference / pipValue);
  }

  private calculateProfitLoss(trade: DivineTrade, currentPrice: number): number {
    const priceDifference = trade.direction === 'LONG' 
      ? currentPrice - trade.entryPrice 
      : trade.entryPrice - currentPrice;
    
    // Simplified P&L calculation (in account currency)
    // In a real implementation, you'd use the actual pip value from OANDA
    const pipValue = trade.pair.includes('JPY') ? 0.01 : 0.0001;
    const pips = priceDifference / pipValue;
    return pips * 10; // Assuming $10 per pip for this example
  }

  public async getMarketStatus() {
    try {
      // Get status from the divine strategy which includes OANDA data
      const status = await divineStrategy.getMarketStatus();
      
      if (!status.success) {
        throw new Error('Failed to get market status from OANDA');
      }
      
      // Add active trades information
      const activeTrades = this.activeTrades.length;
      const statusMessage = status.message.replace(
        'Active Trades: 0', 
        `Active Trades: ${activeTrades}${activeTrades > 0 ? ' (synced with OANDA)' : ''}`
      );
      
      return {
        success: true,
        message: statusMessage,
        data: {
          ...status.data,
          activeTrades
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

  public getActiveTrades(): DivineTrade[] {
    return [...this.activeTrades];
  }

  public getTradeHistory(): DivineTrade[] {
    return [...this.tradeHistory];
  }
}

// Export a singleton instance
export const divineTrader = new DivineTrader();
