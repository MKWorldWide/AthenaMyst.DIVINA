""
Configuration management for AthenaMyst:Divina.

Uses pydantic-settings to handle environment variables with validation.
"""
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OandaConfig(BaseModel):
    """OANDA API configuration."""
    api_key: str = Field(..., description="OANDA API key")
    account_id: str = Field(..., description="OANDA account ID")
    environment: str = Field("practice", description="OANDA environment (practice or live)")
    base_url: str = Field("https://api-fxtrade.oanda.com/v3", 
                         description="OANDA API base URL")
    stream_url: str = Field("https://stream-fxtrade.oanda.com/v3",
                           description="OANDA streaming API URL")


class KrakenConfig(BaseModel):
    """Kraken API configuration."""
    api_key: Optional[str] = Field(None, description="Kraken API key")
    api_secret: Optional[str] = Field(None, description="Kraken API secret")
    base_url: str = Field("https://api.kraken.com/0",
                         description="Kraken API base URL")


class DiscordConfig(BaseModel):
    """Discord webhook configuration."""
    webhook_url: Optional[str] = Field(None, description="Discord webhook URL for alerts")
    mention_role: Optional[str] = Field(None, description="Optional role ID to mention in alerts")


class TradingConfig(BaseModel):
    """Trading configuration."""
    pairs: List[str] = Field(
        default_factory=lambda: ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD"],
        description="List of currency pairs to trade"
    )
    signal_tf: str = Field("H1", description="Signal timeframe (e.g., H1, M15)")
    confirm_tf: str = Field("H4", description="Confirmation timeframe (e.g., H4, D1)")
    vol_mult: float = Field(1.5, description="Volume multiplier for volume surge detection")
    cooldown_min: int = Field(90, description="Cooldown period between signals (minutes)")
    risk_per_trade: float = Field(0.02, description="Risk per trade as fraction of account")


class Settings(BaseSettings):
    """Application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="DIVINA__",
        case_sensitive=False,
    )
    
    # Environment and logging
    env: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    log_json: bool = Field(False, description="Output logs in JSON format")
    
    # API configurations
    oanda: OandaConfig
    kraken: KrakenConfig = Field(default_factory=KrakenConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    
    # Trading settings
    trading: TradingConfig = Field(default_factory=TradingConfig)
    
    # Feature flags
    enable_backtest: bool = Field(False, description="Enable backtesting mode")
    dry_run: bool = Field(True, description="Run without executing real trades")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == Environment.PRODUCTION
    
    @property
    def is_test(self) -> bool:
        """Check if running in test environment."""
        return self.env == Environment.TEST
    
    @validator("trading")
    def validate_timeframes(cls, v: TradingConfig) -> TradingConfig:
        """Validate that signal TF is smaller than confirm TF."""
        tf_order = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN"]
        try:
            signal_idx = tf_order.index(v.signal_tf.upper())
            confirm_idx = tf_order.index(v.confirm_tf.upper())
            if signal_idx >= confirm_idx:
                raise ValueError("Signal TF must be smaller than confirm TF")
        except ValueError as e:
            raise ValueError(f"Invalid timeframe: {e}")
        return v


# Create settings instance
try:
    settings = Settings()
except Exception as e:
    import sys
    print(f"Error loading settings: {e}", file=sys.stderr)
    sys.exit(1)
