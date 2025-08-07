// src/modules/prophecy.ts

// Celestial bodies and their influence on markets
const CELESTIAL_BODIES = [
  { name: 'The Moon', influence: 0.7, aspects: ['emotions', 'intuition', 'cycles'] },
  { name: 'Mercury', influence: 0.9, aspects: ['communication', 'trade', 'volatility'] },
  { name: 'Venus', influence: 0.8, aspects: ['value', 'beauty', 'harmony'] },
  { name: 'Mars', influence: 1.2, aspects: ['action', 'conflict', 'momentum'] },
  { name: 'Jupiter', influence: 1.5, aspects: ['expansion', 'luck', 'optimism'] },
  { name: 'Saturn', influence: 0.5, aspects: ['restriction', 'discipline', 'structure'] },
];

// Mythical archetypes for market phases
const MARKET_ARCHETYPES = [
  { name: 'The Alchemist', description: 'Transforming base metals into gold', trend: 'bullish' },
  { name: 'The Siren', description: 'Luring traders with sweet promises', trend: 'volatile' },
  { name: 'The Phoenix', description: 'Rising from the ashes of the old', trend: 'recovering' },
  { name: 'The Oracle', description: 'Seeing beyond the veil of uncertainty', trend: 'sideways' },
  { name: 'The Titan', description: 'Powerful but bound to fall', trend: 'bearish' },
];

// Ancient elements and their market correlations
const ELEMENTS = {
  fire: { currencies: ['MXN', 'ZAR', 'INR'], description: 'Passionate and volatile' },
  earth: { currencies: ['CHF', 'SGD', 'NOK'], description: 'Stable and grounded' },
  air: { currencies: ['USD', 'EUR', 'GBP'], description: 'Ethereal and far-reaching' },
  water: { currencies: ['JPY', 'CAD', 'AUD'], description: 'Fluid and deep' },
};

// Generate a random number within a range
const randomInRange = (min: number, max: number): number => 
  Math.random() * (max - min) + min;

// Get a random item from an array
const getRandomItem = <T>(array: T[]): T => 
  array[Math.floor(Math.random() * array.length)];

// Calculate celestial influence on a currency
const calculateCelestialInfluence = (currency: string): number => {
  // Base influence from the currency's element
  const element = Object.entries(ELEMENTS).find(([_, data]) => 
    data.currencies.includes(currency)
  )?.[1];
  
  // Add random celestial influence
  const celestial = getRandomItem(CELESTIAL_BODIES);
  const influence = celestial.influence * randomInRange(0.8, 1.2);
  
  return influence;
};

// Generate a prophetic forecast for a currency pair
export function generateCurrencyProphecy(
  baseCurrency: string, 
  quoteCurrency: string,
  days: number = 7
): string {
  // Calculate the celestial influences
  const baseInfluence = calculateCelestialInfluence(baseCurrency);
  const quoteInfluence = calculateCelestialInfluence(quoteCurrency);
  const relativeStrength = baseInfluence / quoteInfluence;
  
  // Determine the market phase and archetype
  const archetype = getRandomItem(MARKET_ARCHETYPES);
  const phase = archetype.trend;
  
  // Generate the prophecy
  const celestial = getRandomItem(CELESTIAL_BODIES);
  const element = Object.entries(ELEMENTS).find(([_, data]) => 
    data.currencies.includes(baseCurrency)
  )?.[0] || 'ether';
  
  // Calculate price movement
  const priceChangePercent = (relativeStrength - 1) * 100;
  const priceDirection = priceChangePercent >= 0 ? 'rise' : 'fall';
  const absoluteChange = Math.abs(priceChangePercent).toFixed(2);
  
  // Generate the prophecy text
  const prophecyParts = [
    `🔮 *Celestial Alignment for ${baseCurrency}/${quoteCurrency}* 🔮`,
    `Under the gaze of ${celestial.name},`,
    `The ${element} element stirs the markets.`,
    `\n*${archetype.name} emerges...*`,
    `"${archetype.description}."`,
    `\nThe charts whisper of a ${phase} market.`,
    `I foresee a ${priceDirection} of ~${absoluteChange}%`,
    `in the coming ${days} days.`,
    `\n${getRandomElement(['🌙', '✨', '⚡', '🌊', '🔥'])} *Trust in the stars.*`
  ];
  
  return prophecyParts.join('\n');
}

// Get a random element from an array (helper function)
function getRandomElement<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

// Export the prophecy generator
export default {
  generateCurrencyProphecy,
};
