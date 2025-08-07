// Simple test runner for DivineTrader tests
const { DivineTrader } = require('../src/modules/trading/divineTrader');

async function runTests() {
  console.log('Starting DivineTrader tests...');
  const trader = new DivineTrader();
  let passed = 0;
  let failed = 0;

  // Test 1: Initialization
  try {
    console.log('\nTest 1: Initialization');
    if (trader.getActiveTrades().length !== 0 || trader.getTradeHistory().length !== 0) {
      throw new Error('Trader should be initialized with empty trade lists');
    }
    console.log('✅ Passed: Trader initialized correctly');
    passed++;
  } catch (error) {
    console.error('❌ Failed:', error.message);
    failed++;
  }

  // Test 2: Execute scalp trade
  try {
    console.log('\nTest 2: Execute scalp trade');
    const result = await trader.executeScalpTrade('EUR_USD', 1.1200);
    if (!result.success || !result.tradeId) {
      throw new Error('Failed to execute scalp trade');
    }
    console.log('✅ Passed: Scalp trade executed successfully');
    passed++;
  } catch (error) {
    console.error('❌ Failed:', error.message);
    failed++;
  }

  // Test 3: Get market update
  try {
    console.log('\nTest 3: Get market update');
    const update = trader.getMarketUpdate();
    if (typeof update !== 'string' || update.length === 0) {
      throw new Error('Invalid market update format');
    }
    console.log('✅ Passed: Market update generated successfully');
    passed++;
  } catch (error) {
    console.error('❌ Failed:', error.message);
    failed++;
  }

  // Summary
  console.log('\nTest Summary:');
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`🏁 Total:  ${passed + failed}`);
}

// Run the tests
runTests().catch(console.error);
