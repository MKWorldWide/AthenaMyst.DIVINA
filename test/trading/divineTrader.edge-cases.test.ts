import { DivineTrader, Trade, TradeStatus } from '../../src/modules/trading/divineTrader';

// Utility function to create a test trade
function createTestTrade(overrides: Partial<Trade> = {}): Trade {
  return {
    id: 'test-trade-' + Math.random().toString(36).substring(7),
    pair: 'EUR_USD',
    direction: 'LONG',
    entryPrice: 1.1200,
    status: 'OPEN',
    openTime: new Date().toISOString(),
    ...overrides
  };
}

describe('DivineTrader - Edge Cases & Error Conditions', () => {
  let trader: DivineTrader;

  beforeEach(() => {
    trader = new DivineTrader();
  });

  describe('Trade Execution Edge Cases', () => {
    it('should handle invalid currency pairs', async () => {
      // Test with invalid pair format
      await expect(trader.executeScalpTrade('INVALID_PAIR', 1.1200))
        .rejects
        .toThrow('Invalid currency pair format');

      // Test with unsupported pair
      await expect(trader.executeScalpTrade('BTC_ETH', 0.05))
        .rejects
        .toThrow('Unsupported currency pair');
    });

    it('should handle invalid price values', async () => {
      // Test with zero price
      await expect(trader.executeScalpTrade('EUR_USD', 0))
        .rejects
        .toThrow('Invalid price');

      // Test with negative price
      await expect(trader.executeScalpTrade('EUR_USD', -1.1200))
        .rejects
        .toThrow('Invalid price');

      // Test with extremely high price
      await expect(trader.executeScalpTrade('EUR_USD', 1e10))
        .rejects
        .toThrow('Suspicious price');
    });
  });

  describe('Trade Management Edge Cases', () => {
    it('should handle closing non-existent trades', async () => {
      const result = await trader.closeTrade('non-existent-trade-id');
      expect(result.success).toBe(false);
      expect(result.message).toContain('not found');
    });

    it('should handle duplicate trade IDs', async () => {
      const trade = createTestTrade();
      
      // Manually add a trade
      trader.getActiveTrades = jest.fn().mockReturnValue([trade]);
      
      // Try to add a trade with the same ID
      await expect(trader.executeScalpTrade('EUR_USD', 1.1200, { forceId: trade.id }))
        .rejects
        .toThrow('Trade ID already exists');
    });
  });

  describe('Market Analysis Edge Cases', () => {
    it('should handle empty or invalid market data', async () => {
      // Test with empty pair
      await expect(trader.analyzeMarket(''))
        .rejects
        .toThrow('Invalid currency pair');

      // Test with invalid historical data
      trader['oandaService'].getCandles = jest.fn().mockResolvedValue([]);
      const analysis = await trader.analyzeMarket('EUR_USD');
      expect(analysis.status).toBe('warning');
      expect(analysis.message).toContain('insufficient data');
    });
  });

  describe('Risk Management Scenarios', () => {
    it('should enforce maximum open trades', async () => {
      // Set a low max open trades for testing
      const MAX_OPEN_TRADES = 3;
      trader['maxOpenTrades'] = MAX_OPEN_TRADES;

      // Fill up to max open trades
      for (let i = 0; i < MAX_OPEN_TRADES; i++) {
        await trader.executeScalpTrade('EUR_USD', 1.1200 + (i * 0.0001));
      }

      // Next trade should be rejected
      await expect(trader.executeScalpTrade('EUR_USD', 1.1300))
        .rejects
        .toThrow('Maximum open trades reached');
    });

    it('should enforce stop loss and take profit', async () => {
      const entryPrice = 1.1200;
      const stopLoss = 1.1150;
      const takeProfit = 1.1300;
      
      // Execute trade with stop loss and take profit
      const result = await trader.executeScalpTrade(
        'EUR_USD', 
        entryPrice,
        { stopLoss, takeProfit }
      );

      // Simulate price movement to trigger stop loss
      await trader.checkStopLossesAndTakeProfits([
        { pair: 'EUR_USD', bid: stopLoss - 0.0001, ask: stopLoss + 0.0001 }
      ]);

      // Verify trade was closed at stop loss
      const trade = trader.getTradeHistory().find(t => t.id === result.tradeId);
      expect(trade?.status).toBe('CLOSED');
      expect(trade?.exitPrice).toBeCloseTo(stopLoss);
    });
  });

  describe('Error Recovery', () => {
    it('should recover from OANDA API errors', async () => {
      // Simulate OANDA API error
      trader['oandaService'].placeOrder = jest.fn().mockRejectedOnce(new Error('OANDA API timeout'));
      
      // Should handle the error gracefully
      await expect(trader.executeScalpTrade('EUR_USD', 1.1200))
        .rejects
        .toThrow('Failed to execute trade');
      
      // Verify system is still operational
      const update = trader.getMarketUpdate();
      expect(update).toContain('Market Update');
    });
  });
});
