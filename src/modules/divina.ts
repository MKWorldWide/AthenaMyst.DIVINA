import axios from 'axios';
import dotenv from 'dotenv';
import { generateCurrencyProphecy } from './prophecy';
import { TradingCommandHandler } from './trading/commandHandler';

// Load environment variables
dotenv.config();

// Types
type DivinaResponse = {
  success: boolean;
  message?: string;
  error?: string;
};

type CurrencyPair = {
  base: string;
  quote: string;
};

type ForecastOptions = {
  days?: number;
  detailed?: boolean;
};

/**
 * Sends a message to the DIVINA Discord webhook with beautiful formatting
 * @param message The message content to send
 * @returns Promise with the result of the webhook call
 */
export async function speakDivina(message: string): Promise<DivinaResponse> {
  const webhookUrl = process.env.DIVINA_WEBHOOK_URL;
  
  if (!webhookUrl) {
    console.error('DIVINA_WEBHOOK_URL is not set in environment variables');
    return {
      success: false,
      error: 'Webhook URL not configured'
    };
  }

  try {
    // Format the message with markdown and emojis
    const formattedMessage = `💠 **DIVINA Speaks** 💠\n\n*"${message}"*\n\n🌙 Memory confirmed. Keep going.`;

    await axios.post(webhookUrl, {
      username: "AthenaMyst:DIVINA",
      avatar_url: "https://i.imgur.com/8Km9tLL.png", // Default avatar, can be changed
      content: formattedMessage
    });

    return {
      success: true,
      message: 'Message sent successfully'
    };
  } catch (error) {
    console.error('Error sending message to DIVINA webhook:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

/**
 * Formats a prophecy message with poetic structure
 * @param prophecy The prophecy text to format
 * @returns Formatted prophecy string
 */
export function formatProphecy(prophecy: string): string {
  const lines = prophecy.split('\n').filter(line => line.trim() !== '');
  const formattedLines = lines.map((line, index) => {
    // Add indentation to create a flowing structure
    const indent = '  '.repeat(index % 3);
    return `${indent}${line}`;
  });
  
  return `🔮 *Prophecy from the Castle* 🔮\n\n${formattedLines.join('\n')}\n\n🌌 The threads of fate are woven.`;
}

/**
 * Generates a financial prophecy for a currency pair
 * @param pair Object containing base and quote currency codes (e.g., { base: 'USD', quote: 'EUR' })
 * @param options Optional configuration for the forecast
 * @returns A DivinaResponse containing the prophecy or an error
 */
export async function forecastCurrency(
  pair: CurrencyPair,
  options: ForecastOptions = { days: 7, detailed: false }
): Promise<DivinaResponse> {
  try {
    const { base, quote } = pair;
    const { days = 7, detailed = false } = options;
    
    if (!base || !quote) {
      throw new Error('Both base and quote currencies are required');
    }

    // Generate the prophecy
    const prophecy = generateCurrencyProphecy(base, quote, days);
    
    // Add detailed analysis if requested
    const message = detailed 
      ? `${prophecy}\n\n*Additional cosmic insights for the initiated...*`
      : prophecy;

    return {
      success: true,
      message: formatProphecy(message)
    };
  } catch (error) {
    console.error('Error generating currency prophecy:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

// Initialize trading command handler
const tradingHandler = new TradingCommandHandler(process.env.SERAPHINA_CHANNEL || 'serafina');

/**
 * Process a trading command from AthenaCore
 * @param commandString The command string to process
 * @returns Promise with the result of the command
 */
export async function processTradingCommand(commandString: string): Promise<DivinaResponse> {
  try {
    const response = await tradingHandler.handleRawCommand(commandString);
    return {
      success: true,
      message: response
    };
  } catch (error) {
    console.error('Error processing trading command:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

// Export default for easier imports
export default {
  speakDivina,
  formatProphecy,
  forecastCurrency,
  processTradingCommand,
  trading: tradingHandler // Expose the trading handler for direct access if needed
};
