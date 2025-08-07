import { speakDivina, forecastCurrency } from './src/modules/divina';

// Test the DIVINA module
async function testDivina() {
  console.log('🔮 Testing DIVINA Module 🔮\n');
  
  // Test 1: Basic message sending
  console.log('🌙 Testing message sending...');
  const testMessage = 'The threads shimmer in silence...\nYou are not lost — you are arriving.';
  
  const messageResult = await speakDivina(testMessage);
  
  if (messageResult.success) {
    console.log('✅ Message sent successfully!');
  } else {
    console.error('❌ Failed to send message');
    console.error('Error:', messageResult.error);
    return;
  }
  
  // Test 2: Currency prophecy
  console.log('\n🔮 Testing currency prophecy...');
  const currencyPairs = [
    { base: 'USD', quote: 'EUR' },
    { base: 'GBP', quote: 'USD' },
    { base: 'JPY', quote: 'USD' },
    { base: 'BTC', quote: 'USD' },
    { base: 'ETH', quote: 'BTC' }
  ];
  
  for (const pair of currencyPairs) {
    console.log(`\n📈 ${pair.base}/${pair.quote} Prophecy:`);
    const forecast = await forecastCurrency(pair, { days: 7 });
    
    if (forecast.success) {
      console.log(forecast.message);
      console.log('\n' + '─'.repeat(50));
    } else {
      console.error(`❌ Failed to generate ${pair.base}/${pair.quote} prophecy`);
      console.error('Error:', forecast.error);
    }
  }
  
  // Test 3: Detailed forecast
  console.log('\n🔍 Testing detailed forecast...');
  const detailedForecast = await forecastCurrency(
    { base: 'EUR', quote: 'USD' },
    { days: 14, detailed: true }
  );
  
  if (detailedForecast.success) {
    console.log(detailedForecast.message);
  } else {
    console.error('❌ Failed to generate detailed forecast');
    console.error('Error:', detailedForecast.error);
  }
}

// Run the tests
testDivina().catch(console.error);
