# OANDA FX Scalping Bot

A high-frequency trading bot designed for OANDA's FX trading platform, implementing a 10% position sizing strategy with advanced technical analysis.

## Features

- **OANDA API Integration**: Seamless connection to OANDA's trading platform
- **10% Position Sizing**: Automatically calculates position sizes based on 10% of account equity
- **Advanced Technical Analysis**: Uses EMA, RSI, MACD, and ATR for signal generation
- **Risk Management**: Built-in stop-loss and take-profit orders
- **Web Interface**: Monitor and control the bot via a simple web interface
- **Real-time Logging**: Detailed logging for all trading activities

## Prerequisites

1. Python 3.8 or higher
2. OANDA account (Practice or Live)
3. OANDA API key with trading permissions
4. Required Python packages (install via `pip install -r requirements_oanda.txt`)

## Setup

1. **Clone the repository**
   ```bash
   git clone [your-repository-url]
   cd trading_bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_oanda.txt
   ```

3. **Configure your OANDA credentials**
   - Copy `.env.oanda.example` to `.env.oanda`
   - Update with your OANDA account details:
     ```
     OANDA_ACCOUNT_ID=your_account_id
     OANDA_API_KEY=your_api_key
     OANDA_ACCOUNT_TYPE=practice  # or 'live' for real trading
     
     # Trading parameters
     TRADING_PAIR=EUR_USD
     ACCOUNT_CURRENCY=USD
     RISK_PERCENT=10.0  # 10% of account per trade
     STOP_LOSS_PIPS=20
     TAKE_PROFIT_PIPS=40
     ```

## Running the Bot

### Start the bot
```bash
python oanda_scalper.py
```

The bot will start and be accessible at `http://localhost:8002`

### API Endpoints

- `GET /` - Basic status
- `GET /status` - Detailed trading status
- `POST /webhook` - For external signal integration
- `POST /close_all` - Close all open positions

## Strategy Details

The bot implements the following trading strategy:

1. **Entry Signals**
   - EMA crossover (9/21/50)
   - RSI confirmation (not overbought/oversold)
   - MACD crossover
   - Price above/below medium-term EMA

2. **Position Sizing**
   - 10% of account equity per trade
   - Stop loss at 20 pips
   - Take profit at 40 pips (2:1 reward:risk ratio)

3. **Risk Management**
   - Maximum 1 trade at a time
   - Stop-loss and take-profit orders placed immediately
   - No trading during high volatility or news events

## Monitoring

Check the following log files for monitoring:

- `oanda_scalper.log` - Main bot logs
- `oanda_engine.log` - Trading engine logs
- `data/` - Trade history and performance metrics

## Customization

You can adjust the strategy parameters in `oanda_scalper.py`:

```python
# Strategy parameters
self.ema_fast = 9
self.ema_medium = 21
self.ema_slow = 50
self.rsi_period = 14
self.rsi_overbought = 70
self.rsi_oversold = 30
```

## Risk Warning

- This is a high-frequency trading bot that can make or lose money rapidly
- Always start with the practice account
- Never risk more than you can afford to lose
- Monitor the bot's performance regularly

## Support

For issues and feature requests, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
