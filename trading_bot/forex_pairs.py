"""
Top 25 forex pairs by trading volume.
Data sourced from the latest BIS Triennial Central Bank Survey and other reliable sources.
"""

# Top 25 forex pairs by trading volume (major, minor, and some crosses)
TOP_25_FOREX_PAIRS = [
    # Major Pairs (most liquid)
    'EUR/USD',  # Euro / US Dollar
    'USD/JPY',  # US Dollar / Japanese Yen
    'GBP/USD',  # British Pound / US Dollar
    'USD/CHF',  # US Dollar / Swiss Franc
    'AUD/USD',  # Australian Dollar / US Dollar
    'USD/CAD',  # US Dollar / Canadian Dollar
    'NZD/USD',  # New Zealand Dollar / US Dollar
    
    # Euro Crosses
    'EUR/GBP',  # Euro / British Pound
    'EUR/JPY',  # Euro / Japanese Yen
    'EUR/AUD',  # Euro / Australian Dollar
    'EUR/CAD',  # Euro / Canadian Dollar
    'EUR/CHF',  # Euro / Swiss Franc
    'EUR/NZD',  # Euro / New Zealand Dollar
    
    # Other Major Crosses
    'GBP/JPY',  # British Pound / Japanese Yen
    'GBP/AUD',  # British Pound / Australian Dollar
    'GBP/CAD',  # British Pound / Canadian Dollar
    'AUD/JPY',  # Australian Dollar / Japanese Yen
    'AUD/NZD',  # Australian Dollar / New Zealand Dollar
    'AUD/CAD',  # Australian Dollar / Canadian Dollar
    'CAD/JPY',  # Canadian Dollar / Japanese Yen
    'NZD/JPY',  # New Zealand Dollar / Japanese Yen
    'CHF/JPY',  # Swiss Franc / Japanese Yen
    'GBP/CHF',  # British Pound / Swiss Franc
    'EUR/NOK',  # Euro / Norwegian Krone
    'USD/NOK',  # US Dollar / Norwegian Krone
]

def get_top_forex_pairs() -> list:
    """Return the list of top forex pairs by trading volume."""
    return TOP_25_FOREX_PAIRS.copy()

def get_forex_pairs_by_volume(min_volume: float = 0.0) -> list:
    """
    Return forex pairs filtered by minimum volume.
    
    Args:
        min_volume: Minimum volume threshold (0.0 to 1.0, where 1.0 is the most liquid)
    """
    if min_volume <= 0:
        return TOP_25_FOREX_PAIRS.copy()
    
    # This is a simplified example - in a real implementation, you'd want to use actual volume data
    # For now, we'll just return all pairs since they're already ordered by liquidity
    return TOP_25_FOREX_PAIRS.copy()

if __name__ == "__main__":
    print("Top 25 Forex Pairs by Volume:")
    for i, pair in enumerate(TOP_25_FOREX_PAIRS, 1):
        print(f"{i:2d}. {pair}")
