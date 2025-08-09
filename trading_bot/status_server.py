"""
Status server for the crypto trading bot.
Provides a web interface to monitor trading activity and account status.
"""
import os
import json
import time
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import uvicorn
import ccxt

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv("LOG_FILE", "status_server.log"), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Trading Bot Status",
    description="API for monitoring the crypto trading bot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# In-memory storage for trade data
TRADE_HISTORY: List[Dict[str, Any]] = []
LAST_UPDATE = 0
CACHE_TTL = 300  # 5 minutes

# Exchange instances
exchanges: Dict[str, ccxt.Exchange] = {}

def init_exchanges() -> None:
    """Initialize exchange connections."""
    global exchanges
    
    # Initialize Kraken
    if os.getenv("KRAKEN_API_KEY") and os.getenv("KRAKEN_API_SECRET"):
        try:
            exchanges["kraken"] = ccxt.kraken({
                "apiKey": os.getenv("KRAKEN_API_KEY"),
                "secret": os.getenv("KRAKEN_API_SECRET"),
                "enableRateLimit": True,
                "options": {"adjustForTimeDifference": True}
            })
            exchanges["kraken"].load_markets()
            logger.info("Initialized Kraken exchange")
        except Exception as e:
            logger.error(f"Failed to initialize Kraken: {e}")
    
    # Initialize Binance.US
    if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
        try:
            exchanges["binanceus"] = ccxt.binanceus({
                "apiKey": os.getenv("BINANCE_API_KEY"),
                "secret": os.getenv("BINANCE_API_SECRET"),
                "enableRateLimit": True,
                "options": {
                    "adjustForTimeDifference": True,
                    "recvWindow": int(os.getenv("BINANCE_RECVWINDOW", "15000")),
                }
            })
            exchanges["binanceus"].load_markets()
            logger.info("Initialized Binance.US exchange")
        except Exception as e:
            logger.error(f"Failed to initialize Binance.US: {e}")

def load_trade_history() -> None:
    """Load trade history from the PnL log file."""
    global TRADE_HISTORY, LAST_UPDATE
    
    pnl_file = os.getenv("PNL_LOG_FILE", "trading_pnl.csv")
    if not os.path.exists(pnl_file):
        logger.warning(f"PnL log file not found: {pnl_file}")
        TRADE_HISTORY = []
        return
    
    try:
        with open(pnl_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            TRADE_HISTORY = list(reader)
        LAST_UPDATE = time.time()
        logger.info(f"Loaded {len(TRADE_HISTORY)} trades from {pnl_file}")
    except Exception as e:
        logger.error(f"Error loading trade history: {e}")
        TRADE_HISTORY = []

def get_daily_pnl() -> Dict[str, float]:
    """Calculate daily PnL."""
    daily_pnl = defaultdict(float)
    
    for trade in TRADE_HISTORY:
        try:
            timestamp = datetime.fromisoformat(trade['timestamp'])
            date_str = timestamp.strftime('%Y-%m-%d')
            
            if trade['side'] == 'buy':
                daily_pnl[date_str] -= float(trade['amount']) * float(trade['price'])
            else:  # sell
                daily_pnl[date_str] += float(trade['amount']) * float(trade['price'])
        except (KeyError, ValueError) as e:
            logger.warning(f"Error processing trade: {e}")
    
    return dict(daily_pnl)

@app.get("/crypto/status", response_model=Dict[str, Any])
async def get_status() -> Dict[str, Any]:
    """
    Get the current status of the trading bot.
    
    Returns:
        Dict containing status information including:
        - balances: Account balances by exchange and currency
        - open_trades: Currently open positions
        - daily_pnl: Daily profit/loss
        - last_update: Timestamp of last update
    """
    global LAST_UPDATE
    
    # Reload trade history if cache is stale
    if time.time() - LAST_UPDATE > CACHE_TTL:
        load_trade_history()
    
    status = {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "exchanges": {},
        "open_trades": [],
        "daily_pnl": get_daily_pnl(),
        "last_update": datetime.fromtimestamp(LAST_UPDATE).isoformat()
    }
    
    # Get balances and open positions for each exchange
    for exchange_id, exchange in exchanges.items():
        try:
            # Get account balance
            balance = exchange.fetch_balance()
            
            # Filter out zero balances and format
            balances = {
                currency: {
                    "free": float(balance[currency].get('free', 0)),
                    "used": float(balance[currency].get('used', 0)),
                    "total": float(balance[currency].get('total', 0))
                }
                for currency in balance['total']
                if balance[currency].get('total', 0) > 0
            }
            
            # Get open positions
            positions = exchange.fetch_positions()
            open_positions = [
                {
                    "symbol": p['symbol'],
                    "side": p['side'],
                    "amount": float(p['contracts']) if 'contracts' in p else 0,
                    "entry_price": float(p['entryPrice']) if 'entryPrice' in p else 0,
                    "unrealized_pnl": float(p['unrealizedPnl']) if 'unrealizedPnl' in p else 0,
                    "leverage": float(p['leverage']) if 'leverage' in p else 1.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for p in positions
                if p and 'contracts' in p and float(p['contracts']) > 0
            ]
            
            status["exchanges"][exchange_id] = {
                "balances": balances,
                "open_positions": open_positions,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Add to open trades
            status["open_trades"].extend([
                {"exchange": exchange_id, **pos} for pos in open_positions
            ])
            
        except Exception as e:
            logger.error(f"Error fetching {exchange_id} status: {e}")
            status["exchanges"][exchange_id] = {
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat()
            }
    
    return status

@app.get("/crypto/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    raise Exception(f"No available ports in range {start_port}-{start_port + max_attempts - 1}")

def main() -> None:
    """Run the status server with error handling and port management."""
    try:
        # Initialize exchanges
        logger.info("Initializing exchanges...")
        init_exchanges()
        
        # Load initial trade history
        logger.info("Loading trade history...")
        load_trade_history()
        
        # Configure server
        host = os.getenv("STATUS_SERVER_HOST", "0.0.0.0")
        default_port = 8000
        port = int(os.getenv("STATUS_SERVER_PORT", str(default_port)))
        
        # Find an available port if default is in use
        if is_port_in_use(port):
            logger.warning(f"Port {port} is already in use, trying next available port...")
            port = find_available_port(port + 1)
        
        logger.info(f"Starting status server on http://{host}:{port}")
        logger.info(f"API docs available at http://{host}:{port}/docs")
        
        # Configure logging for uvicorn
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                },
            },
            "formatters": {
                "default": {
                    "fmt": "%(asctime)s - %(levelname)s - %(message)s",
                    "use_colors": None,
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
            },
        }
        
        # Start the server with explicit logging config
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            log_config=log_config,
            reload=False,
            workers=1
        )
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        logger.critical(f"Fatal error in status server: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
