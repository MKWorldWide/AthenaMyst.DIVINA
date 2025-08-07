// Simple TypeScript test runner for DivineTrader
import { DivineTrader } from './src/modules/trading/divineTrader';

async function runTests() {
  console.log('🚀 Starting DivineTrader tests...');
  const trader = new DivineTrader();
  let passed = 0;
  let failed = 0;

  // Test 1: Initialization
  try {
    console.log('\n🔍 Test 1: Initialization');
    const activeTrades = trader.getActiveTrades();
    const tradeHistory = trader.getTradeHistory();
    
    if (activeTrades.length !== 0 || tradeHistory.length !== 0) {
      throw new Error('❌ Trader should be initialized with empty trade lists');
    }
    console.log('✅ Passed: Trader initialized correctly');
    passed++;
  } catch (error) {
    console.error('❌ Failed:', error instanceof Error ? error.message : String(error));
    failed++;
  }

  // Test 2: Execute scalp trade
  try {
    console.log('\n💱 Test 2: Execute scalp trade');
    const result = await trader.executeScalpTrade('EUR_USD', 1.1200);
    
    if (!result.success || !result.tradeId) {
      throw new Error('❌ Failed to execute scalp trade');
    }
    
    console.log(`✅ Passed: Scalp trade executed successfully (ID: ${result.tradeId})`);
    passed++;
    
    // Test 3: Get trade report
    try {
      console.log('\n📊 Test 3: Get trade report');
      const report = trader.getTradeReport(result.tradeId);
      
      if (!report || typeof report !== 'string') {
        throw new Error('❌ Invalid trade report format');
      }
      
      console.log('✅ Passed: Trade report generated successfully');
      console.log('   Report preview:', report.substring(0, 100) + '...');
      passed++;
    } catch (error) {
      console.error('❌ Failed to get trade report:', error instanceof Error ? error.message : String(error));
      failed++;
    }
    
  } catch (error) {
    console.error('❌ Failed to execute scalp trade:', error instanceof Error ? error.message : String(error));
    failed++;
  }

  // Test 4: Get market update
  try {
    console.log('\n📈 Test 4: Get market update');
    const update = trader.getMarketUpdate();
    
    if (typeof update !== 'string' || update.length === 0) {
      throw new Error('❌ Invalid market update format');
    }
    
    console.log('✅ Passed: Market update generated successfully');
    console.log('   Update preview:', update.substring(0, 100) + '...');
    passed++;
  } catch (error) {
    console.error('❌ Failed to get market update:', error instanceof Error ? error.message : String(error));
    failed++;
  }

  // Summary
  console.log('\n📊 Test Summary:');
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`🏁 Total:  ${passed + failed}`);
  
  return { passed, failed };
}

// Run the tests
runTests()
  .then(() => process.exit(0))
  .catch(error => {
    console.error('❌ Test runner error:', error);
    process.exit(1);
  });
