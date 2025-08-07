// test-trading.ts
import { processTradingCommand } from './src/modules/divina';
import { formatProphecy } from './src/modules/divina';
import { v4 as uuidv4 } from 'uuid';

// Test the divine trading system
async function testDivineTrading() {
  console.log('🔮 Testing Divine Trading System 🔮\n');
  
  // 1. Get market status
  console.log('🌙 Checking market status...');
  const status = await processTradingCommand('STATUS');
  console.log(status.message + '\n');
  
  // 2. Execute a scalp trade
  console.log('⚡ Executing scalp trade...');
  const scalpResult = await processTradingCommand('SCALP BTC/USD 50000');
  console.log(scalpResult.message + '\n');
  
  // 3. Get detailed market analysis
  console.log('🔍 Analyzing market...');
  const analysis = await processTradingCommand('ANALYZE BTC/USD --detailed');
  console.log(analysis.message + '\n');
  
  // 4. Get the list of active trades
  console.log('📜 Getting active trades...');
  const marketStatus = await processTradingCommand('STATUS');
  console.log(marketStatus.message + '\n');
  
  // 5. For demo purposes, we'll just show the status
  console.log('✨ Test completed successfully!');
  console.log('To see trading in action, you can use these commands:');
  console.log('1. npm run test:divina - Test the prophecy system');
  console.log('2. Use the following commands in your application:');
  console.log('   - SCALP <pair> <price> - Start a scalp trade');
  console.log('   - ANALYZE <pair> - Analyze market conditions');
  console.log('   - STATUS - Check market status');
  console.log('   - REPORT <tradeId> - Get trade report');
  console.log('   - CLOSE <tradeId> <price> - Close a trade');
  
  // 6. Final status
  console.log('🌌 Final market status:');
  const finalStatus = await processTradingCommand('STATUS');
  console.log(finalStatus.message);
  
  console.log('\n✨ Divine trading test complete! ✨');
}

// Run the test
testDivineTrading().catch(error => {
  console.error('🔥 Divine error occurred:', error);
  process.exit(1);
});
