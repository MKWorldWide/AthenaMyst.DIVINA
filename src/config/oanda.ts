// OANDA API Configuration
export const oandaConfig = {
  apiKey: 'bcd66a3a1fd8ce666e9b3d38f30fdeb9-a870baae9ab623a698d8fdd542465a89',
  accountId: '001-001-13409355-002',
  baseUrl: 'https://api-fxtrade.oanda.com/v3',  // Use api-fxpractice.oanda.com for practice account
  streamUrl: 'https://stream-fxtrade.oanda.com/v3'
} as const;

export const tradingSettings = {
  defaultPair: 'EUR_USD',
  defaultUnits: 1000,
  riskPerTrade: 0.02, // 2% risk per trade
  defaultStopLossPips: 20,
  defaultTakeProfitPips: 40
};
