# Crypto Trading Bot

A high-performance, multi-exchange cryptocurrency trading bot with a focus on reliability and risk management. The bot supports Binance.US and Kraken exchanges with dynamic symbol discovery and position sizing.

## Features

- 🚀 **Multi-Exchange Support**: Trade on Binance.US and Kraken simultaneously
- 🔍 **Dynamic Symbol Discovery**: Automatically finds top liquid assets under $5
- ⚡ **High Performance**: Asynchronous architecture for optimal performance
- 📊 **Risk Management**: Configurable risk per trade, stop-loss, and take-profit
- 📈 **Real-time Monitoring**: Built-in web interface for monitoring trades and PnL
- 🔒 **Secure**: Secure API key handling and request signing
- 📝 **Comprehensive Logging**: Detailed logs for debugging and performance analysis

## Prerequisites

- Python 3.9 or higher
- Binance.US and/or Kraken exchange account with API keys
- (Optional) Basic understanding of cryptocurrency trading and risk management

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment:
   - Copy `.env.example` to `.env`
   - Update with your API keys and trading parameters

## Configuration

Edit the `.env` file with your configuration:

```ini
# Trading Parameters
RISK_PER_TRADE=0.0125        # 1.25% of quote balance per trade
TP_PCT=0.004                 # 0.4% take profit
SL_PCT=0.003                 # 0.3% stop loss
MAX_OPEN_TRADES=6            # Max open trades per exchange
COOLDOWN_MIN=20              # Minutes between trades on same symbol
MIN_VOL_USD=500000           # Minimum 24h volume in USD
MAX_PRICE=5                  # Maximum asset price to trade

# Binance.US API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_RECVWINDOW=15000     # Request timeout in ms

# Kraken API
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log
PNL_LOG_FILE=trading_pnl.csv

# Status Server
STATUS_SERVER_HOST=0.0.0.0
STATUS_SERVER_PORT=8000
```
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
