import { expect } from 'chai';
import { describe, it, beforeEach } from 'mocha';
import { DivineTrader } from '../../src/modules/trading/divineTrader';

describe('DivineTrader', () => {
  let trader: DivineTrader;

  beforeEach(() => {
    trader = new DivineTrader();
  });

  describe('Initialization', () => {
    it('should initialize with empty trade lists', () => {
      expect(trader.getActiveTrades()).to.be.an('array').that.is.empty;
      expect(trader.getTradeHistory()).to.be.an('array').that.is.empty;
    });
  });

  describe('Trade Execution', () => {
    it('should execute a scalp trade', async () => {
      const result = await trader.executeScalpTrade('EUR_USD', 1.1200);
      expect(result).to.have.property('success', true);
      expect(result).to.have.property('tradeId').that.is.a('string');
      expect(trader.getActiveTrades()).to.have.length(1);
    });

    it('should close a trade', async () => {
      const openResult = await trader.executeScalpTrade('EUR_USD', 1.1200);
      if (!openResult.tradeId) {
        throw new Error('Trade ID is missing from open result');
      }
      
      const closeResult = await trader.closeTrade(openResult.tradeId);
      
      expect(closeResult).to.have.property('success', true);
      expect(trader.getActiveTrades()).to.be.empty;
      expect(trader.getTradeHistory()).to.have.length(1);
    });
  });

  describe('Market Analysis', () => {
    it('should analyze market conditions', async () => {
      const analysis = await trader.analyzeMarket('EUR_USD');
      expect(analysis).to.have.property('status');
      expect(analysis).to.have.property('message');
      expect(analysis).to.have.property('data');
    });
  });

  describe('Reporting', () => {
    it('should generate a trade report', async () => {
      const openResult = await trader.executeScalpTrade('EUR_USD', 1.1200);
      if (!openResult.tradeId) {
        throw new Error('Trade ID is missing from open result');
      }
      
      const report = trader.getTradeReport(openResult.tradeId);
      if (!report) {
        throw new Error('Report is null or undefined');
      }
      
      expect(report).to.be.a('string');
      expect(report).to.include('Trade Report');
      expect(report).to.include('EUR_USD');
    });

    it('should generate a market update', () => {
      const update = trader.getMarketUpdate();
      expect(update).to.be.a('string');
      expect(update).to.include('Market Update');
    });
  });
});
