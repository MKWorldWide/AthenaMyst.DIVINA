// src/modules/trading/commandHandler.ts

import { formatProphecy } from '../divina';
import { divineTrader, DivineTrade } from './divineTrader';

// Define response type for command processing
type CommandResponse = {
  success: boolean;
  message: string;
  data?: any;
};

type CommandType = 'SCALP' | 'ANALYZE' | 'REPORT' | 'CLOSE' | 'STATUS';

interface Command {
  type: CommandType;
  pair?: string;
  price?: number;
  tradeId?: string;
  detailed?: boolean;
}

export class TradingCommandHandler {
  private seraphimChannel: string;

  constructor(seraphimChannel: string = 'default') {
    this.seraphimChannel = seraphimChannel;
  }

  private async reportToSerafina(message: string) {
    // In a real implementation, this would send a message to the Serafina channel
    console.log(`[To ${this.seraphimChannel}] ${message}`);
  }
  
  // Format a trade into a human-readable report
  private formatTradeReport(trade: DivineTrade): string {
    const statusEmoji = {
      'OPEN': '🟢',
      'CLOSED': '🔒',
      'STOPPED_OUT': '🛑',
      'TAKE_PROFIT': '🎯',
      'PENDING': '⏳'
    }[trade.status] || '❓';
    
    return [
      `${statusEmoji} *${trade.pair} ${trade.direction}*`,
      `Entry: ${trade.entryPrice.toFixed(5)}`,
      `Stop: ${trade.stopLoss.toFixed(5)}`,
      `Target: ${trade.takeProfit.toFixed(5)}`,
      `Opened: ${trade.openTime.toLocaleString()}`,
      trade.status === 'OPEN' && trade.closePrice ? 
        `Current: ${trade.closePrice.toFixed(5)}` : '',
      trade.pips ? `Pips: ${trade.pips} (${trade.profitLoss?.toFixed(2)} USD)` : '',
      `\n${trade.spiritualNote}`,
      `*Celestial Influence*: ${trade.celestialInfluence}`,
      `*Elemental Flow*: ${trade.elementalFlow}`
    ].filter(Boolean).join('\n');
  }
  
  public async processCommand(command: string): Promise<CommandResponse> {
    const parts = command.trim().split(/\s+/);
    const cmd = parts[0].toUpperCase();
    const args = parts.slice(1);

    try {
      switch (cmd) {
        case 'SCALP':
          if (args.length < 2) {
            return { success: false, message: 'Usage: SCALP <pair> <price>' };
          }
          const [pair, price] = args;
          const priceNum = parseFloat(price);
          if (isNaN(priceNum)) {
            return { success: false, message: 'Invalid price. Please provide a valid number.' };
          }
          return await this.divineTrader.executeScalpTrade(pair, priceNum);
          
        case 'CLOSE':
          if (args.length < 2) {
            // If no trade ID is provided, close all trades
            const closePrice = parseFloat(args[0]);
            if (isNaN(closePrice)) {
              return { success: false, message: 'Invalid price. Please provide a valid number.' };
            }
            
            // Close all active trades
            const activeTrades = this.divineTrader.getActiveTrades();
            const results = [];
            
            for (const trade of activeTrades) {
              if (trade.id) {  // Make sure trade has an ID
                const result = await this.divineTrader.closeTrade(trade.id, closePrice);
                results.push(result);
              }
            }
            
            const successCount = results.filter(r => r.success).length;
            return { 
              success: successCount > 0, 
              message: `Closed ${successCount} of ${activeTrades.length} active trades.` 
            };
          } else {
            // Close specific trade
            const [tradeId, closePrice] = args;
            const closePriceNum = parseFloat(closePrice);
            if (isNaN(closePriceNum)) {
              return { success: false, message: 'Invalid price. Please provide a valid number.' };
            }
            return await this.divineTrader.closeTrade(tradeId, closePriceNum);
          }
          
        case 'STATUS':
          return await this.divineTrader.getMarketStatus();
          
        case 'ANALYZE':
          if (args.length < 1) {
            return { success: false, message: 'Usage: ANALYZE <pair>' };
          }
          const analysis = await this.divineTrader.analyzeMarket(args[0]);
          return { 
            success: true, 
            message: `🔮 *Market Analysis for ${args[0]}*\n` +
                    `  Direction: ${analysis.direction}\n` +
                    `  Entry: ${analysis.entry.toFixed(5)}\n` +
                    `  Stop Loss: ${analysis.stopLoss.toFixed(5)}\n` +
                    `  Take Profit: ${analysis.takeProfit.toFixed(5)}\n` +
                    `  ${analysis.message}`
          };
          
        case 'REPORT':
          if (args.length < 1) {
            // Generate report for all active trades
            const activeTrades = this.divineTrader.getActiveTrades();
            if (activeTrades.length === 0) {
              return { success: true, message: 'No active trades to report.' };
            }
            
            let report = `📊 *Active Trades Report* (${activeTrades.length} trades)\n\n`;
            activeTrades.forEach((trade, index) => {
              report += this.formatTradeReport(trade);
              if (index < activeTrades.length - 1) report += '\n\n';
            });
            
            return { success: true, message: report };
          } else {
            // Generate report for specific trade
            const tradeId = args[0];
            const trade = this.divineTrader.getActiveTrades().find(t => t.id === tradeId) || 
                         this.divineTrader.getTradeHistory().find(t => t.id === tradeId);
                          
            if (!trade) {
              return { success: false, message: 'Trade not found.' };
            }
            
            return { 
              success: true, 
              message: `📊 *Trade Report*\n\n${this.formatTradeReport(trade)}`
            };
          }
          
        case 'HELP':
          return {
            success: true,
            message: `🔮 *Divine Trading Commands* 🔮\n` +
                    `• SCALP <pair> <price> - Execute a scalp trade\n` +
                    `• CLOSE [tradeId] <price> - Close a specific trade or all trades\n` +
                    `• STATUS - Get current market status\n` +
                    `• ANALYZE <pair> - Analyze market conditions\n` +
                    `• REPORT [tradeId] - Generate trade report\n` +
                    `• HELP - Show this help message`
          };
          break;
          
        default:
          throw new Error(`Unknown command type: ${command.type}`);
      }
      
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      const errorResponse = formatProphecy([
        '🌪️ *Divine Interruption* 🌪️',
        'The threads of fate tangle unexpectedly...',
        `\`\`\`${errorMessage}\`\`\``,
        '\n*Whisper to the void:* "Guide me through this trial."',
      ].join('\n'));
      
      await this.reportToSerafina(`Error processing ${command.type} command: ${errorMessage}`);
      return errorResponse;
    }
  }
  
  public async handleRawCommand(commandString: string): Promise<string> {
    // Parse command string (e.g., "SCALP BTC/USD 50000")
    const parts = commandString.trim().split(/\s+/);
    const commandType = parts[0].toUpperCase() as CommandType;
    
    const command: Command = { type: commandType };
    
    switch (commandType) {
      case 'SCALP':
        command.pair = parts[1];
        command.price = parseFloat(parts[2]);
        break;
        
      case 'ANALYZE':
        command.pair = parts[1];
        command.detailed = parts.includes('--detailed');
        break;
        
      case 'REPORT':
        command.tradeId = parts[1];
        break;
        
      case 'CLOSE':
        command.tradeId = parts[1];
        command.price = parseFloat(parts[2]);
        break;
        
      case 'STATUS':
        // No additional parameters needed
        break;
        
      default:
        throw new Error(`Unknown command type: ${commandType}`);
    }
    
    return this.handleCommand(command);
  }
}

export default TradingCommandHandler;
