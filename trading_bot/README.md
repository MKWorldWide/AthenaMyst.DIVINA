# Trading Bot with Discord Integration

A powerful trading bot that integrates with Discord for notifications and commands, featuring real-time market analysis and automated trading capabilities.

## Features

- 📊 Real-time market analysis with technical indicators (RSI, EMA, MACD, ATR)
- 🤖 Automated trading based on technical signals
- 💬 Discord integration for notifications and commands
- 🔔 Customizable alerts and notifications
- 📈 Support for multiple trading pairs
- 🛡️ Risk management with stop-loss and take-profit

## Prerequisites

- Python 3.8 or higher
- Discord account and server
- Binance account (or another supported exchange)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and update it with your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your Discord bot token, webhook URL, and exchange API keys.

## Configuration

1. **Discord Bot Setup**
   - Create a new Discord application at [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a bot user and copy the token
   - Invite the bot to your server with the following permissions:
     - Send Messages
     - Embed Links
     - Read Messages/View Channels
     - Use Slash Commands

2. **Discord Webhook Setup**
   - In your Discord server, go to Server Settings > Integrations > Webhooks
   - Create a new webhook and copy the webhook URL
   - Optionally, create a role for trade alerts and copy its ID

3. **Exchange API Setup**
   - Log in to your exchange account (e.g., Binance)
   - Generate API keys with trading permissions
   - For testing, use the testnet/sandbox environment

## Usage

1. Start the bot:
   ```bash
   python discord_bot.py
   ```

2. **Available Commands**
   - `!forecast [pair]` - Get market forecast for a trading pair (default: EUR/USDT)
   - `!balance` - Check your account balance
   - `!positions` - View open positions

3. The bot will automatically analyze the market and send trade signals to the configured Discord channel.

## Customization

### Trading Strategy
Edit the `generate_signal` method in `trading_engine.py` to implement your own trading strategy.

### Indicators
Modify the `calculate_indicators` method in `trading_engine.py` to add or remove technical indicators.

### Risk Management
Adjust the following environment variables in `.env`:
- `STOP_LOSS_PERCENT`: Stop loss percentage (default: 2.0%)
- `TAKE_PROFIT_PERCENT`: Take profit percentage (default: 4.0%)
- `TRADE_AMOUNT`: Amount to trade in quote currency (default: 100)
- `LEVERAGE`: Leverage for margin trading (default: 10)

## Security

- Never share your `.env` file or API keys
- Use environment variables for sensitive information
- Consider using a dedicated trading account with limited permissions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on the GitHub repository.
