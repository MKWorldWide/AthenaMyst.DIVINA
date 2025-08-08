import os
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountDetails
from oandapyV20.endpoints.trades import OpenTrades
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.oanda')

# Get OANDA credentials
account_id = os.getenv('OANDA_ACCOUNT_ID')
api_key = os.getenv('OANDA_API_KEY')
account_type = os.getenv('OANDA_ACCOUNT_TYPE', 'practice')

print(f"Testing OANDA Connection...")
print(f"Account ID: {account_id}")
print(f"API Key: {'*' * len(api_key) if api_key else 'Not found'}")
print(f"Account Type: {account_type}")

# Initialize API client
try:
    client = API(access_token=api_key, environment=account_type)
    print("\nAPI client initialized successfully!")
    
    # Test account details
    print("\nFetching account details...")
    r = AccountDetails(accountID=account_id)
    response = client.request(r)
    
    print("\nAccount Details:")
    print(f"Account ID: {response['account']['id']}")
    print(f"Balance: {response['account']['balance']} {response['account']['currency']}")
    print(f"Unrealized P/L: {response['account']['unrealizedPL']}")
    print(f"Margin Available: {response['account']['marginAvailable']}")
    
    # List open trades
    print("\nOpen Trades:")
    try:
        r = OpenTrades(accountID=account_id)
        response = client.request(r)
        if 'trades' in response and response['trades']:
            for trade in response['trades']:
                print(f"- {trade['instrument']} {trade['currentUnits']} units @ {trade['price']}")
        else:
            print("No open trades found.")
    except Exception as e:
        print(f"Error fetching open trades: {e}")
        
except Exception as e:
    print(f"\nError: {e}")
    if hasattr(e, 'response'):
        print(f"Response: {e.response.text}")
