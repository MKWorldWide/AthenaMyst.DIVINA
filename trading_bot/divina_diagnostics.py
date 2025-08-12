"""
Divina Diagnostics & Live Monitor
-------------------------------
Comprehensive health check and monitoring for Oanda and Kraken trading operations.
"""

import os
import sys
import time
import logging
import json
import re
import requests
import hmac
import hashlib
import base64
import urllib.parse
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('divina_diagnostics.log')
    ]
)
logger = logging.getLogger('divina_diagnostics')

# Load environment variables
load_dotenv()

class ExchangeMonitor:
    """Base class for exchange monitoring with auto-remediation capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_check = None
        self.metrics = {
            'signals_per_min': 0,
            'orders_per_min': 0,
            'fills_per_min': 0,
            'api_errors': 0,
            'last_error': None,
            'last_successful_request': None,
            'consecutive_errors': 0,
            'auto_remediation_attempts': 0,
            'last_remediation': None
        }
        self.connected = False
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Divina/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self._last_reconnect_attempt = 0
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._reconnect_delay = 5  # seconds
        
    def check_connectivity(self) -> bool:
        """Check basic API connectivity."""
        raise NotImplementedError
        
    def fetch_recent_trades(self, hours: int = 24) -> List[dict]:
        """Fetch recent trades."""
        raise NotImplementedError
        
    def get_account_status(self) -> dict:
        """Get account status and balances."""
        raise NotImplementedError
        
    def _make_request(self, method: str, url: str, **kwargs) -> dict:
        """Make an HTTP request with error handling, metrics, and auto-remediation."""
        try:
            # Check if we need to attempt reconnection first
            current_time = time.time()
            if (self.metrics['consecutive_errors'] > 0 and 
                current_time - self._last_reconnect_attempt > 30):  # 30s cooldown
                self._attempt_reconnect()
                
            response = self._session.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Request succeeded, reset error counters
            self.metrics['consecutive_errors'] = 0
            self.metrics['last_successful_request'] = datetime.now(pytz.UTC).isoformat()
            return response.json() if response.content else {}
            
        except requests.exceptions.RequestException as e:
            self.metrics['api_errors'] += 1
            self.metrics['consecutive_errors'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"{self.name} API request failed: {str(e)}")
            
            # If we have multiple consecutive errors, try to auto-remediate
            if self.metrics['consecutive_errors'] >= 3:
                logger.warning(f"Multiple consecutive errors detected, attempting auto-remediation...")
                self._attempt_remediation()
                
            raise
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to the exchange."""
        current_time = time.time()
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.warning("Max reconnection attempts reached. Manual intervention may be required.")
            return False
            
        self._last_reconnect_attempt = current_time
        self._reconnect_attempts += 1
        
        logger.info(f"Attempting to reconnect to {self.name} (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
        
        try:
            # Close existing session
            self._session.close()
            
            # Create a new session
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'Divina/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            })
            
            # Reset connection state
            self.connected = False
            
            # Small delay before next attempt
            time.sleep(self._reconnect_delay)
            
            # Try to re-establish connection
            if self.check_connectivity():
                logger.info(f"Successfully reconnected to {self.name}")
                self._reconnect_attempts = 0
                return True
                
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {str(e)}")
            
        return False
    
    def _attempt_remediation(self):
        """Attempt to automatically fix common issues."""
        current_time = time.time()
        
        # Don't attempt remediation too frequently
        if (self.metrics.get('last_remediation') and 
            current_time - self.metrics['last_remediation'] < 300):  # 5 min cooldown
            return False
            
        self.metrics['auto_remediation_attempts'] += 1
        self.metrics['last_remediation'] = current_time
        
        logger.info(f"Attempting auto-remediation for {self.name}...")
        
        # 1. First try a simple reconnection
        if self._attempt_reconnect():
            return True
            
        # 2. Check for clock sync issues
        try:
            self._check_clock_sync()
        except Exception as e:
            logger.error(f"Clock sync check failed: {str(e)}")
            
        # 3. If we have a reset method, try it
        if hasattr(self, 'reset_connection'):
            try:
                self.reset_connection()
                logger.info("Connection reset completed")
                return True
            except Exception as e:
                logger.error(f"Connection reset failed: {str(e)}")
                
        logger.warning("Auto-remediation attempts were not successful")
        return False
    
    def _check_clock_sync(self):
        """Check if the system clock is in sync with the exchange."""
        try:
            # Get server time
            server_time = self._get_server_time()
            if not server_time:
                return
                
            # Calculate time difference
            local_time = datetime.now(pytz.UTC)
            time_diff = abs((server_time - local_time).total_seconds())
            
            if time_diff > 10:  # More than 10 seconds difference
                logger.warning(f"Clock out of sync with {self.name}: {time_diff:.2f} seconds")
                # Try to sync the clock (requires appropriate permissions)
                self._sync_clock(server_time)
                
        except Exception as e:
            logger.error(f"Clock sync check failed: {str(e)}")
    
    def _get_server_time(self):
        """Get the current server time from the exchange."""
        raise NotImplementedError("Server time check not implemented for this exchange")
    
    def _sync_clock(self, server_time):
        """Attempt to sync the local clock with the exchange."""
        # This is a placeholder - actual implementation would require admin privileges
        logger.warning(f"Clock out of sync with {self.name}. Please sync your system clock.")
        logger.warning(f"Server time: {server_time}, Local time: {datetime.now(pytz.UTC)}")


class KrakenMonitor(ExchangeMonitor):
    """Kraken exchange monitor with auto-remediation."""
    
    BASE_URL = 'https://api.kraken.com'
    API_VERSION = '0'
    
    def _get_server_time(self):
        """Get Kraken server time."""
        response = self._session.get(f"{self.BASE_URL}/{self.API_VERSION}/public/Time")
        response.raise_for_status()
        data = response.json()
        if 'result' in data and 'unixtime' in data['result']:
            return datetime.fromtimestamp(float(data['result']['unixtime']), tz=pytz.UTC)
        return None
    
    def reset_connection(self):
        """Reset Kraken connection, nonce, and session with improved error handling."""
        logger.info("Resetting Kraken connection...")
        
        try:
            # Close existing session if it exists
            if hasattr(self, '_session'):
                try:
                    self._session.close()
                except Exception as e:
                    logger.warning(f"Error closing existing session: {str(e)}")
            
            # Create new session with retry strategy
            retry_strategy = requests.adapters.Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "POST"]
            )
            
            adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
            self._session = requests.Session()
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
            
            # Set default headers
            self._session.headers.update({
                'User-Agent': f'DivinaKrakenMonitor/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate',
                'Content-Type': 'application/x-www-form-urlencoded'
            })
            
            # Reset nonce
            self._nonce = int(time.time() * 1000)
            
            # Reset connection state and metrics
            self.connected = False
            self.metrics['connection_resets'] = self.metrics.get('connection_resets', 0) + 1
            self.metrics['last_reset'] = datetime.now(pytz.UTC).isoformat()
            
            # Test the new connection
            if not self.check_connectivity():
                raise Exception("Failed to re-establish connection after reset")
                
            logger.info("Successfully reset Kraken connection")
            
        except Exception as e:
            logger.error(f"Error resetting Kraken connection: {str(e)}")
            self.connected = False
            raise
    
    def __init__(self):
        super().__init__('Kraken')
        self.api_key = os.getenv('KRAKEN_API_KEY')
        self.api_secret = os.getenv('KRAKEN_API_SECRET')
        self.symbols = os.getenv('KRAKEN_SYMBOLS', '').split(',')
        self._session = requests.Session()
        
    def _sign_message(self, url_path: str, data: dict) -> dict:
        """Sign message with API secret with enhanced error handling."""
        try:
            if not self.api_secret:
                raise ValueError("API secret is not set")
                
            nonce = str(int(time.time() * 1000))
            data['nonce'] = nonce
            postdata = urllib.parse.urlencode(data, safe=':,')
            
            # Ensure we're working with bytes for hashing
            encoded = (str(data['nonce']) + postdata).encode('utf-8')
            message = url_path.encode('utf-8') + hashlib.sha256(encoded).digest()
            
            # Ensure API secret is properly decoded
            api_secret = base64.b64decode(self.api_secret)
            signature = hmac.new(api_secret, message, hashlib.sha512)
            sigdigest = base64.b64encode(signature.digest())
            
            return {
                'API-Key': self.api_key,
                'API-Sign': sigdigest.decode('utf-8'),
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
        except Exception as e:
            logger.error(f"Error signing Kraken message: {str(e)}")
            raise
        
    def _api_request(self, method: str, endpoint: str, data: Optional[dict] = None, retry: bool = True) -> dict:
        """Make authenticated API request with retry logic and better error handling."""
        if data is None:
            data = {}
            
        url = f"{self.BASE_URL}/{self.API_VERSION}/{endpoint}"
        headers = {}
        
        try:
            # For private endpoints, sign the request
            if not endpoint.startswith('public/'):
                if not self.api_key or not self.api_secret:
                    raise ValueError("API key and secret are required for private endpoints")
                headers.update(self._sign_message(f'/{self.API_VERSION}/{endpoint}', data))
            
            # Make the request
            if method.upper() == 'POST':
                response = self._session.post(url, headers=headers, data=data, timeout=10)
            else:
                response = self._session.get(url, headers=headers, params=data, timeout=10)
                
            response.raise_for_status()
            result = response.json()
            
            # Check for API errors in response
            if 'error' in result and result['error']:
                error_msg = result['error']
                if isinstance(error_msg, list):
                    error_msg = ", ".join(error_msg)
                raise Exception(f"Kraken API error: {error_msg}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Kraken API request failed: {str(e)}")
            
            # If it's an auth error, don't retry
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 403:
                raise Exception("Authentication failed. Please check your API key and secret.")
                
            # For other errors, retry once if enabled
            if retry:
                logger.info("Retrying request...")
                time.sleep(1)  # Small delay before retry
                return self._api_request(method, endpoint, data, retry=False)
                
            raise
            
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing Kraken API response: {str(e)}")
            raise Exception(f"Failed to process API response: {str(e)}")
    
    def check_connectivity(self) -> bool:
        """Check Kraken API connectivity with detailed error reporting."""
        try:
            start_time = time.time()
            response = self._api_request('GET', 'public/Time')
            
            if 'result' not in response or 'unixtime' not in response['result']:
                logger.error("Unexpected response format from Kraken Time endpoint")
                self.connected = False
                return False
                
            # Calculate and log latency
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Kraken API connectivity check successful (latency: {latency_ms:.2f}ms)")
            
            # Update connection status and metrics
            self.connected = True
            self.metrics['last_successful_request'] = datetime.now(pytz.UTC).isoformat()
            self.metrics['consecutive_errors'] = 0
            
            # Check server time vs local time
            server_time = datetime.fromtimestamp(float(response['result']['unixtime']), tz=pytz.UTC)
            local_time = datetime.now(pytz.UTC)
            time_diff = (local_time - server_time).total_seconds()
            
            if abs(time_diff) > 10:  # More than 10 seconds difference
                logger.warning(f"Clock skew detected: Local time is {time_diff:.2f}s {'ahead' if time_diff > 0 else 'behind'} Kraken server time")
                self.metrics['clock_skew'] = time_diff
            
            return True
            
        except Exception as e:
            self.connected = False
            self.metrics['consecutive_errors'] += 1
            self.metrics['last_error'] = str(e)
            self.metrics['api_errors'] = self.metrics.get('api_errors', 0) + 1
            
            if self.metrics['consecutive_errors'] >= 3:
                logger.error(f"Multiple consecutive connection failures: {str(e)}")
                self._attempt_remediation()
                
            return False
            
    def fetch_recent_trades(self, hours: int = 24) -> List[dict]:
        """Fetch recent trades from Kraken."""
        if not self.connected and not self.check_connectivity():
            return []
            
        try:
            end_time = int(time.time())
            start_time = end_time - (hours * 3600)
            
            trades = []
            # This is a simplified example - actual implementation would paginate through results
            response = self._api_request('POST', 'private/TradesHistory', {
                'start': start_time,
                'end': end_time,
                'trades': 'true'
            })
            
            if 'result' in response and 'trades' in response['result']:
                for trade_id, trade in response['result']['trades'].items():
                    trades.append({
                        'id': trade_id,
                        'symbol': trade['pair'],
                        'side': 'buy' if trade['type'] == 'buy' else 'sell',
                        'price': float(trade['price']),
                        'amount': float(trade['vol']),
                        'cost': float(trade['cost']),
                        'fee': float(trade['fee']),
                        'time': datetime.fromtimestamp(float(trade['time']), tz=pytz.UTC)
                    })
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to fetch Kraken trades: {str(e)}")
            return []
            
    def get_account_status(self) -> dict:
        """Get Kraken account status and balances."""
        if not self.connected and not self.check_connectivity():
            return {}
            
        try:
            response = self._api_request('POST', 'private/Balance')
            if 'result' in response:
                return {
                    'balances': response['result'],
                    'timestamp': datetime.now(pytz.UTC).isoformat()
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to get Kraken account status: {str(e)}")
            return {}


class OandaMonitor(ExchangeMonitor):
    """Oanda exchange monitor with LIVE/PRACTICE environment support and auto-remediation."""
    
    def __init__(self):
        super().__init__('Oanda')
        self._last_environment_switch = 0
        self._environment_switched = False
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.environment = os.getenv('OANDA_ENV', 'PRACTICE').upper()
        self.pairs = [p.strip() for p in os.getenv('PAIRS_OANDA', 'EUR_USD,USD_JPY,GBP_USD').split(',') if p.strip()]
        
        # Initialize session with retry strategy
        self._init_session()
        
        # Set environment URLs
        self._set_environment_urls()
    
    def _init_session(self):
        """Initialize the HTTP session with retry strategy and headers."""
        # Close existing session if it exists
        if hasattr(self, '_session'):
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error closing existing Oanda session: {str(e)}")
        
        # Create a new session with retry strategy
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "POST"]
        )
        
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        
        # Set default headers
        self._session.headers.update({
            'User-Agent': f'DivinaOandaMonitor/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        })
    
    def _set_environment_urls(self):
        """Set the appropriate API URLs based on the environment."""
        if self.environment == 'LIVE':
            self.base_url = 'https://api-fxtrade.oanda.com/v3'
            self.stream_url = 'https://stream-fxtrade.oanda.com/v3'
        else:
            self.base_url = 'https://api-fxpractice.oanda.com/v3'
            self.stream_url = 'https://stream-fxpractice.oanda.com/v3'
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """
        Make an HTTP request with enhanced error handling, retries, and metrics.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (either full URL or path)
            **kwargs: Additional arguments to pass to requests.request()
            
        Returns:
            dict: Parsed JSON response
            
        Raises:
            Exception: If the request fails after all retries
        """
        # Ensure we have a valid URL
        url = endpoint if endpoint.startswith(('http://', 'https://')) else f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Set default timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10  # 10 second timeout by default
            
        # Add request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        
        try:
            # Log the request (without sensitive data)
            safe_headers = {k: v for k, v in self._session.headers.items() 
                          if k.lower() not in ['authorization', 'api-key']}
            logger.debug(f"[{request_id}] Oanda {method} {url} (headers: {safe_headers})")
            
            # Make the request
            start_time = time.time()
            response = self._session.request(method, url, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log response status
            logger.debug(f"[{request_id}] Response {response.status_code} in {latency_ms:.2f}ms")
            
            # Check for error responses
            if not response.ok:
                error_data = {}
                try:
                    error_data = response.json()
                except ValueError:
                    error_data = {'errorMessage': response.text or 'No error details'}
                
                # Log detailed error information
                error_msg = error_data.get('errorMessage', 
                                        error_data.get('error', 
                                                     str(response.status_code)))
                
                # Special handling for rate limits
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', '5'))
                    logger.warning(f"[{request_id}] Rate limited. Waiting {retry_after}s. Details: {error_msg}")
                    time.sleep(retry_after + 1)
                    return self._make_request(method, endpoint, **kwargs)
                
                # Handle authentication errors
                elif response.status_code in (401, 403):
                    logger.error(f"[{request_id}] Authentication failed: {error_msg}")
                    self.connected = False
                    raise Exception(f"Authentication failed: {error_msg}")
                
                # Handle not found errors
                elif response.status_code == 404:
                    logger.error(f"[{request_id}] Resource not found: {url}")
                    raise Exception(f"Resource not found: {endpoint}")
                
                # Handle other errors
                else:
                    logger.error(f"[{request_id}] API error {response.status_code}: {error_msg}")
                    raise Exception(f"API error: {error_msg}")
            
            # Update metrics for successful requests
            self.metrics['requests'] = self.metrics.get('requests', 0) + 1
            self.metrics['last_successful_request'] = datetime.now(pytz.UTC).isoformat()
            self.metrics['consecutive_errors'] = 0
            self.metrics['request_latency'] = latency_ms
            
            # Parse and return the response
            try:
                return response.json() if response.content else {}
            except ValueError as e:
                logger.error(f"[{request_id}] Failed to parse JSON response: {str(e)}")
                raise Exception("Invalid JSON response from server")
            
        except requests.exceptions.Timeout:
            logger.error(f"[{request_id}] Request timed out after {kwargs.get('timeout', 10)}s")
            raise Exception("Request timed out")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{request_id}] Connection error: {str(e)}")
            self.connected = False
            raise Exception("Connection error. Please check your internet connection.")
            
        except Exception as e:
            logger.error(f"[{request_id}] Request failed: {str(e)}")
            
            # Update error metrics
            self.metrics['errors'] = self.metrics.get('errors', 0) + 1
            self.metrics['last_error'] = str(e)
            self.metrics['consecutive_errors'] = self.metrics.get('consecutive_errors', 0) + 1
            
            # Attempt remediation after multiple failures
            if self.metrics['consecutive_errors'] >= 3:
                logger.warning("Multiple consecutive errors, attempting remediation...")
                self._attempt_remediation()
                
            raise
    
    def check_connectivity(self) -> bool:
        """Check Oanda API connectivity with detailed error reporting and environment fallback."""
        current_time = time.time()
        
        # Check if we have required credentials
        if not self.api_key or not self.account_id:
            logger.error("Missing Oanda API key or account ID")
            self.connected = False
            return False
            
        # If we recently switched environments, wait before checking again
        if current_time - self._last_environment_switch < 60:  # 1 minute cooldown
            return self.connected
            
        try:
            start_time = time.time()
            
            # First try the configured environment
            response = self._make_request('GET', f"{self.base_url}/accounts/{self.account_id}/summary")
            
            if 'account' in response:
                # Calculate and log latency
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"Oanda {self.environment} API connectivity check successful (latency: {latency_ms:.2f}ms)")
                
                # Update connection status and metrics
                self.connected = True
                self.metrics['last_successful_request'] = datetime.now(pytz.UTC).isoformat()
                self.metrics['consecutive_errors'] = 0
                self._environment_switched = False
                
                # Log account details
                account = response['account']
                logger.info(f"Account: {account['alias']} (ID: {account['id']}, Balance: {account['balance']} {account['currency']})")
                
                return True
                
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            
            # Log specific error details
            if status_code == 401:
                logger.error("Oanda authentication failed. Please check your API key and account ID.")
            elif status_code == 404:
                logger.error(f"Oanda account not found. Please verify the account ID: {self.account_id}")
            else:
                logger.error(f"Oanda API request failed: {error_msg}")
                
            # If we're in LIVE mode and it's a connection/authorization error, try falling back to PRACTICE
            if (self.environment == 'LIVE' and 
                not self._environment_switched and 
                current_time - self._last_environment_switch > 3600 and  # 1 hour cooldown
                (status_code in [401, 403, 404] or 'Connection' in error_msg)):
                
                logger.warning("LIVE environment failed, attempting fallback to PRACTICE environment...")
                self._switch_environment('PRACTICE')
                
                # Retry with PRACTICE environment
                try:
                    response = self._make_request('GET', f"{self.base_url}/accounts/{self.account_id}/summary")
                    if 'account' in response:
                        self.connected = True
                        self.metrics['environment_switch'] = 'LIVE→PRACTICE'
                        logger.warning("Successfully connected to Oanda PRACTICE environment")
                        return True
                except Exception as retry_e:
                    logger.error(f"PRACTICE environment also failed: {str(retry_e)}")
        
        # Update error metrics
        self.connected = False
        self.metrics['consecutive_errors'] = self.metrics.get('consecutive_errors', 0) + 1
        self.metrics['last_error'] = error_msg if 'error_msg' in locals() else 'Unknown error'
        self.metrics['api_errors'] = self.metrics.get('api_errors', 0) + 1
        
        # Attempt remediation after multiple failures
        if self.metrics['consecutive_errors'] >= 3:
            logger.error("Multiple consecutive connection failures, attempting remediation...")
            self._attempt_remediation()
            
        return False
    
    def _switch_environment(self, new_environment: str):
        """Switch between LIVE and PRACTICE environments."""
        if new_environment not in ['LIVE', 'PRACTICE']:
            raise ValueError("Environment must be either 'LIVE' or 'PRACTICE'")
            
        logger.info(f"Switching Oanda environment from {self.environment} to {new_environment}")
        
        self.environment = new_environment
        self._environment_switched = True
        self._last_environment_switch = time.time()
        
        # Update base URLs
        self._set_environment_urls()
        
        # Reinitialize session to clear any connection issues
        self._init_session()
            
    def _get_server_time(self):
        """Get Oanda server time."""
        response = self._session.get(f"{self.base_url}/v3/accounts/{self.account_id}/summary")
        response.raise_for_status()
        data = response.json()
        if 'time' in data:
            return datetime.strptime(data['time'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=pytz.UTC)
        return None
    
    def reset_connection(self):
        """Reset Oanda connection."""
        logger.info("Resetting Oanda connection...")
        
        # Close existing session
        self._session.close()
        
        # Create new session
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Reset connection state
        self.connected = False
        self.check_connectivity()
            
    def fetch_recent_trades(self, hours: int = 24) -> List[dict]:
        """Fetch recent trades from Oanda."""
        if not self.connected and not self.check_connectivity():
            return []
            
        try:
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(hours=hours)
            
            all_trades = []
            
            # Fetch transactions for the time period
            response = self._make_request(
                'GET',
                f"{self.base_url}/accounts/{self.account_id}/transactions/sinceid",
                params={
                    'from': int(start_time.timestamp()),
                    'to': int(end_time.timestamp()),
                    'type': 'ORDER_FILL',
                    'count': 500  # Max allowed by Oanda
                }
            )
            
            if 'transactions' in response:
                for tx in response['transactions']:
                    if tx.get('type') == 'ORDER_FILL':
                        all_trades.append({
                            'id': tx['id'],
                            'symbol': tx['instrument'],
                            'side': tx['side'].lower(),
                            'price': float(tx['price']),
                            'amount': float(tx['units']),
                            'cost': float(tx['price']) * abs(float(tx['units'])),
                            'fee': 0.0,  # Oanda includes fees in the price
                            'time': datetime.strptime(tx['time'].split('.')[0], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=pytz.UTC)
                        })
            
            return all_trades
            
        except Exception as e:
            logger.error(f"Failed to fetch Oanda trades: {str(e)}")
            return []
            
    def get_account_status(self) -> dict:
        """Get Oanda account status and balances."""
        if not self.connected and not self.check_connectivity():
            return {}
            
        try:
            # Get account summary
            summary = self._make_request(
                'GET',
                f"{self.base_url}/accounts/{self.account_id}/summary"
            )
            
            # Get current positions
            positions = self._make_request(
                'GET',
                f"{self.base_url}/accounts/{self.account_id}/openPositions"
            )
            
            return {
                'account': summary.get('account', {}),
                'positions': positions.get('positions', []),
                'timestamp': datetime.now(pytz.UTC).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get Oanda account status: {str(e)}")
            return {}


class LogAnalyzer:
    """Analyzes trading logs for signals, orders, and errors."""
    
    def __init__(self, log_dir: str = 'logs/divina'):
        self.log_dir = log_dir
        self.signal_patterns = {
            'entry': [r'entry signal', r'long signal', r'short signal', r'buy signal', r'sell signal', r'BUY signal', r'SELL signal'],
            'exit': [r'exit signal', r'close position', r'take profit', r'stop loss', r'exit_monitor', r'trailing stop'],
            'order': [r'order (?:submitted|created|executed)', r'placing order', r'sending order', r'created order', r'executed order'],
            'fill': [r'order filled', r'trade executed', r'position (?:opened|closed)', r'filled at'],
            'cancel': [r'order cancelled', r'cancelling order', r'canceled order'],
            'error': [r'error', r'exception', r'failed', r'timeout', r'reject', r'rate limit', r'timed out', r'connection error'],
            'debug': [r'debug', r'Signal check', r'No signal', r'Breakout levels']
        }
        self.compiled_patterns = {k: [re.compile(p, re.IGNORECASE) for p in v] 
                                for k, v in self.signal_patterns.items()}
        
    def analyze_logs(self, hours: int = 24) -> dict:
        """Analyze logs from the last N hours."""
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(hours=hours)
        
        results = {
            'signals': {'entry': 0, 'exit': 0},
            'orders': {'created': 0, 'filled': 0, 'cancelled': 0},
            'errors': [],
            'warnings': [],
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'log_files_analyzed': 0
        }
        
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory not found: {self.log_dir}")
            return results
            
        # Find all log files modified in the last 'hours' hours
        for root, _, files in os.walk(self.log_dir):
            for filename in files:
                if not filename.endswith('.log'):
                    continue
                    
                filepath = os.path.join(root, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath), tz=pytz.UTC)
                
                # Skip files older than our time window
                if file_mtime < start_time:
                    continue
                    
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            self._process_log_line(line, results)
                    results['log_files_analyzed'] += 1
                except Exception as e:
                    logger.error(f"Error processing log file {filepath}: {str(e)}")
                    
        return results
        
    def _process_log_line(self, line: str, results: dict):
        """Process a single log line and update results."""
        line_lower = line.lower()
        
        # Check for signal patterns
        for sig_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(line_lower):
                    if sig_type == 'entry':
                        results['signals']['entry'] += 1
                    elif sig_type == 'exit':
                        results['signals']['exit'] += 1
                    elif sig_type == 'order':
                        results['orders']['created'] += 1
                    elif sig_type == 'fill':
                        results['orders']['filled'] += 1
                    elif sig_type == 'cancel':
                        results['orders']['cancelled'] += 1
                    elif sig_type == 'error' and 'error' in line_lower:
                        results['errors'].append(line.strip())
                    elif 'warning' in line_lower:
                        results['warnings'].append(line.strip())
                    break


class HealthMonitor:
    """Main health monitoring class."""
    
    def __init__(self):
        self.exchanges = {}
        
        # Initialize Kraken if configured
        if os.getenv('KRAKEN_API_KEY') and os.getenv('KRAKEN_API_SECRET'):
            self.exchanges['kraken'] = KrakenMonitor()
            
        # Initialize Oanda if configured
        if os.getenv('OANDA_API_KEY') and os.getenv('OANDA_ACCOUNT_ID'):
            self.exchanges['oanda'] = OandaMonitor()
            
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.log_analyzer = LogAnalyzer()
        self.last_analysis = {}
        self.start_time = datetime.now(pytz.UTC)
        
        if not self.exchanges:
            logger.warning("No exchanges configured. Please check your environment variables.")
        
    def check_all_exchanges(self) -> dict:
        """Check connectivity for all configured exchanges."""
        results = {}
        for name, exchange in self.exchanges.items():
            try:
                connected = exchange.check_connectivity()
                results[name] = {
                    'connected': connected,
                    'status': 'OK' if connected else 'ERROR',
                    'timestamp': datetime.now(pytz.UTC).isoformat()
                }
                logger.info(f"{name.upper()} connectivity: {'OK' if connected else 'ERROR'}")
            except Exception as e:
                logger.error(f"Error checking {name}: {str(e)}")
                results[name] = {
                    'connected': False,
                    'status': f'ERROR: {str(e)}',
                    'timestamp': datetime.now(pytz.UTC).isoformat()
                }
        return results
    
    def analyze_trades(self, exchange_name: str, hours: int = 24):
        """
        Analyze recent trading activity with enhanced error handling and metrics.
        
        Args:
            exchange_name: Name of the exchange to analyze
            hours: Number of hours to look back for trades
            
        Returns:
            dict: Analysis results including trade metrics and statistics
        """
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not configured")
            return None
            
        exchange = self.exchanges[exchange_name]
        try:
            # Get recent trades with error handling
            try:
                trades = exchange.fetch_recent_trades(hours)
            except Exception as e:
                logger.error(f"Error fetching {exchange_name} trades: {e}")
                # Try to reset connection and retry once
                exchange.reset_connection()
                trades = exchange.fetch_recent_trades(hours)
            
            if not trades:
                logger.warning(f"No trades found for {exchange_name} in the last {hours} hours")
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'trades': []
                }
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate additional metrics
            profits = [t.get('profit', 0) for t in trades]
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            # Calculate max drawdown
            peak = -float('inf')
            max_dd = 0
            running_pnl = 0
            
            for p in profits:
                running_pnl += p
                if running_pnl > peak:
                    peak = running_pnl
                dd = peak - running_pnl
                if dd > max_dd:
                    max_dd = dd
            
            # Calculate Sharpe ratio (simplified)
            returns = [p for p in profits if p != 0]
            if len(returns) > 1:
                sharpe = (sum(returns) / len(returns)) / (np.std(returns) or 0.01)
            else:
                sharpe = 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {exchange_name} trades: {str(e)}")
            return {'error': str(e)}
    
    def analyze_signals_and_logs(self) -> dict:
        """Analyze trading signals and logs for the last 24 hours."""
        log_analysis = self.log_analyzer.analyze_logs(hours=24)
        
        # Calculate rates per minute
        analysis_duration_min = (datetime.now(pytz.UTC) - self.start_time).total_seconds() / 60
        analysis_duration_min = max(1, analysis_duration_min)  # Avoid division by zero
        
        signals_per_min = log_analysis['signals']['entry'] / analysis_duration_min
        orders_per_min = log_analysis['orders']['created'] / analysis_duration_min
        fills_per_min = log_analysis['orders']['filled'] / analysis_duration_min
        
        # Update metrics for each exchange
        for exchange in self.exchanges.values():
            exchange.metrics.update({
                'signals_per_min': signals_per_min,
                'orders_per_min': orders_per_min,
                'fills_per_min': fills_per_min,
                'last_analysis': datetime.now(pytz.UTC).isoformat()
            })
        
        # Store for later reference
        self.last_analysis = {
            'signals': log_analysis['signals'],
            'orders': log_analysis['orders'],
            'error_count': len(log_analysis['errors']),
            'warning_count': len(log_analysis['warnings']),
            'signals_per_min': signals_per_min,
            'orders_per_min': orders_per_min,
            'fills_per_min': fills_per_min,
            'analysis_timestamp': datetime.now(pytz.UTC).isoformat()
        }
        
        return self.last_analysis
    
    def get_health_summary(self) -> dict:
        """Generate a health summary for all exchanges."""
        # First update signal and log analysis
        signal_analysis = self.analyze_signals_and_logs()
        
        summary = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'uptime_minutes': round((datetime.now(pytz.UTC) - self.start_time).total_seconds() / 60, 2),
            'exchanges': {},
            'signals': signal_analysis['signals'],
            'orders': signal_analysis['orders'],
            'metrics': {
                'signals_per_min': round(signal_analysis['signals_per_min'], 2),
                'orders_per_min': round(signal_analysis['orders_per_min'], 2),
                'fills_per_min': round(signal_analysis['fills_per_min'], 2),
                'error_count': signal_analysis['error_count'],
                'warning_count': signal_analysis['warning_count']
            }
        }
        
        # Add exchange-specific status
        for name, exchange in self.exchanges.items():
            try:
                status = exchange.check_connectivity()
                account = exchange.get_account_status()
                trades = self.analyze_trades(name)
                
                summary['exchanges'][name] = {
                    'status': 'ONLINE' if status else 'OFFLINE',
                    'account': account,
                    'trading': trades,
                    'metrics': {
                        'signals_per_min': exchange.metrics['signals_per_min'],
                        'orders_per_min': exchange.metrics['orders_per_min'],
                        'fills_per_min': exchange.metrics['fills_per_min'],
                        'api_errors': exchange.metrics['api_errors'],
                        'last_error': exchange.metrics['last_error']
                    }
                }
                
                # Add critical issues if any
                if not status:
                    summary['exchanges'][name]['issues'] = ['API connectivity issue']
                
                if exchange.metrics['api_errors'] > 0:
                    if 'issues' not in summary['exchanges'][name]:
                        summary['exchanges'][name]['issues'] = []
                    summary['exchanges'][name]['issues'].append(f"{exchange.metrics['api_errors']} API errors")
                
            except Exception as e:
                summary['exchanges'][name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                
        return summary
    
    def generate_fix_report(self, summary: dict) -> dict:
        """Generate a detailed fix report with root causes and recommended actions."""
        report = {
            'timestamp': datetime.now(pytz.UTC).isoformat(),
            'status': 'HEALTHY',
            'issues': [],
            'recommendations': [],
            'exchange_status': {},
            'metrics': summary.get('metrics', {}),
            'next_steps': []
        }
        
        # Check each exchange for issues
        for name, exchange in summary.get('exchanges', {}).items():
            exchange_report = {
                'status': exchange.get('status', 'UNKNOWN'),
                'issues': [],
                'recommendations': []
            }
            
            # Check connectivity issues
            if exchange['status'] != 'ONLINE':
                exchange_report['issues'].append({
                    'code': 'CONNECTIVITY_ISSUE',
                    'severity': 'CRITICAL',
                    'message': f"{name.upper()} is not connected",
                    'details': exchange.get('error', 'No error details available')
                })
                exchange_report['recommendations'].extend([
                    "Check internet connection and VPN settings",
                    "Verify API keys and permissions",
                    "Check exchange status page for outages"
                ])
            
            # Check API errors
            if exchange.get('metrics', {}).get('api_errors', 0) > 0:
                exchange_report['issues'].append({
                    'code': 'API_ERRORS',
                    'severity': 'HIGH',
                    'message': f"{exchange['metrics']['api_errors']} API errors detected",
                    'details': exchange['metrics'].get('last_error', 'No error details')
                })
                exchange_report['recommendations'].append(
                    "Review error logs for patterns or rate limiting issues"
                )
            
            # Check order fill rate
            orders_created = exchange.get('trading', {}).get('orders', {}).get('created', 0)
            orders_filled = exchange.get('trading', {}).get('orders', {}).get('filled', 0)
            
            if orders_created > 0 and orders_filled == 0:
                exchange_report['issues'].append({
                    'code': 'LOW_FILL_RATE',
                    'severity': 'HIGH',
                    'message': "Orders created but none filled",
                    'details': f"{orders_created} orders created, 0 filled"
                })
                exchange_report['recommendations'].extend([
                    "Check if orders are being placed with correct parameters",
                    "Verify account has sufficient balance and margin",
                    "Check for market conditions that might prevent order execution"
                ])
            
            # Check for clock sync issues
            if exchange.get('metrics', {}).get('clock_skew', 0) > 10:  # More than 10 seconds
                exchange_report['issues'].append({
                    'code': 'CLOCK_SKEW',
                    'severity': 'MEDIUM',
                    'message': "System clock out of sync with exchange",
                    'details': f"Clock skew: {exchange['metrics']['clock_skew']:.2f} seconds"
                })
                exchange_report['recommendations'].append(
                    "Synchronize system clock with NTP server"
                )
            
            # Add exchange report to main report
            report['exchange_status'][name] = exchange_report
            
            # Update overall status
            if exchange_report['issues']:
                if 'CRITICAL' in [i['severity'] for i in exchange_report['issues']]:
                    report['status'] = 'CRITICAL'
                elif 'HIGH' in [i['severity'] for i in exchange_report['issues']] and report['status'] != 'CRITICAL':
                    report['status'] = 'WARNING'
        
        # Check for signal issues
        signals_per_min = summary.get('metrics', {}).get('signals_per_min', 0)
        if signals_per_min == 0:
            report['issues'].append({
                'code': 'NO_SIGNALS',
                'severity': 'HIGH',
                'message': "No trading signals detected",
                'details': "No trading signals generated in the monitoring period"
            })
            report['recommendations'].extend([
                "Check signal generation service status",
                "Verify market data feeds are active",
                "Review strategy parameters and conditions"
            ])
        
        # Generate overall recommendations
        if report['status'] == 'HEALTHY':
            report['recommendations'].append("No critical issues detected. System is operating normally.")
        else:
            # Add general recommendations based on severity
            if report['status'] == 'CRITICAL':
                report['recommendations'].append(
                    "Immediate attention required. Critical issues affecting trading operations."
                )
            elif report['status'] == 'WARNING':
                report['recommendations'].append(
                    "Warning conditions detected. Review and address issues to prevent service degradation."
                )
        
        # Generate next steps
        if report['status'] != 'HEALTHY':
            report['next_steps'] = [
                "1. Review the issues and recommendations above",
                "2. Implement recommended fixes for CRITICAL and HIGH severity issues first",
                "3. Monitor the system after making changes",
                "4. Check logs for additional details if needed"
            ]
        
        return report
    
    def send_alert(self, level: str, message: str, details: dict = None):
        """Send alert to Discord."""
        if not self.discord_webhook:
            logger.warning("No Discord webhook configured")
            return
            
        try:
            embed = {
                'title': f"[{level}] Divina Alert",
                'description': message,
                'color': 0x00ff00 if level == 'INFO' else 0xffa500 if level == 'WARN' else 0xff0000,
                'timestamp': datetime.utcnow().isoformat(),
                'fields': []
            }
            
            if details:
                for key, value in details.items():
                    embed['fields'].append({
                        'name': key,
                        'value': str(value)[:1000],  # Discord field value limit
                        'inline': False
                    })
            
            payload = {'embeds': [embed]}
            response = requests.post(
                self.discord_webhook,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {str(e)}")
    
    def send_fix_report(self, report: dict):
        """Send the fix report to Discord."""
        if not self.discord_webhook:
            logger.warning("No Discord webhook configured")
            return
            
        try:
            # Create main report embed
            status_color = {
                'HEALTHY': 0x00ff00,
                'WARNING': 0xffa500,
                'CRITICAL': 0xff0000
            }.get(report['status'], 0x0000ff)
            
            embed = {
                'title': f"🔧 DIVINA FIX REPORT - {report['status']}",
                'description': f"*Generated at {report['timestamp']}*\n" \
                             f"*Uptime: {report.get('metrics', {}).get('uptime_minutes', 0):.1f} minutes*\n\n" \
                             f"**Summary:** {self._get_status_summary(report)}",
                'color': status_color,
                'timestamp': datetime.utcnow().isoformat(),
                'fields': []
            }
            
            # Add exchange status
            for name, exchange in report.get('exchange_status', {}).items():
                status_text = "✅ ONLINE" if exchange['status'] == 'ONLINE' else "❌ OFFLINE"
                issues_text = "\n".join([f"• {i['message']} ({i['severity']})" for i in exchange.get('issues', [])])
                
                embed['fields'].append({
                    'name': f"{name.upper()} {status_text}",
                    'value': issues_text if issues_text else "• No issues detected",
                    'inline': False
                })
            
            # Add recommendations
            if report.get('recommendations'):
                embed['fields'].append({
                    'name': '🔍 Recommendations',
                    'value': '\n'.join([f"• {r}" for r in report['recommendations']]),
                    'inline': False
                })
            
            # Add next steps if there are issues
            if report.get('next_steps'):
                embed['fields'].append({
                    'name': '🚀 Next Steps',
                    'value': '\n'.join(report['next_steps']),
                    'inline': False
                })
            
            # Send the report
            payload = {'embeds': [embed]}
            response = requests.post(
                self.discord_webhook,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            logger.info("Fix report sent to Discord")
            
        except Exception as e:
            logger.error(f"Failed to send fix report to Discord: {str(e)}")
    
    def _get_status_summary(self, report: dict) -> str:
        """Generate a human-readable status summary."""
        if report['status'] == 'HEALTHY':
            return "All systems operational. No issues detected."
            
        issues_by_severity = {}
        for exchange in report.get('exchange_status', {}).values():
            for issue in exchange.get('issues', []):
                issues_by_severity[issue['severity']] = issues_by_severity.get(issue['severity'], 0) + 1
        
        # Add system-wide issues
        for issue in report.get('issues', []):
            issues_by_severity[issue['severity']] = issues_by_severity.get(issue['severity'], 0) + 1
        
        summary_parts = []
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in issues_by_severity:
                count = issues_by_severity[severity]
                summary_parts.append(f"{count} {severity.lower()}")
        
        return f"Detected {", ".join(summary_parts)} issues requiring attention."


def format_health_summary(summary: dict) -> str:
    """Format health summary as a human-readable string."""
    lines = [
        "\n" + "="*80,
        f"DIVINA TRADING MONITOR - {summary['timestamp']}",
        f"Uptime: {summary['uptime_minutes']:.1f} minutes",
        "-"*40,
        "EXCHANGE STATUS:"
    ]
    
    # Add exchange status
    for name, exchange in summary['exchanges'].items():
        status = exchange['status']
        issues = exchange.get('issues', [])
        
        status_line = f"  {name.upper():<8} {status}"
        if status != 'ONLINE':
            status_line += f" - {exchange.get('error', 'Unknown error')}"
        
        lines.append(status_line)
        
        # Add metrics
        if 'metrics' in exchange:
            m = exchange['metrics']
            lines.append(f"    Signals: {m.get('signals_per_min', 0):.1f}/min")
            lines.append(f"    Orders:  {m.get('orders_per_min', 0):.1f}/min")
            lines.append(f"    Fills:   {m.get('fills_per_min', 0):.1f}/min")
            
            if m.get('api_errors', 0) > 0:
                lines.append(f"    API Errors: {m['api_errors']} (Last: {m.get('last_error', 'N/A')})")
    
    # Add signals and orders summary
    lines.extend([
        "-"*40,
        f"SIGNALS (last 24h): {summary['signals']['entry']} entries, {summary['signals']['exit']} exits",
        f"ORDERS (last 24h):  {summary['orders']['created']} created, {summary['orders']['filled']} filled, "
        f"{summary['orders'].get('cancelled', 0)} cancelled",
        "-"*40,
        f"RATES: {summary['metrics']['signals_per_min']:.1f} signals/min, "
        f"{summary['metrics']['orders_per_min']:.1f} orders/min, "
        f"{summary['metrics']['fills_per_min']:.1f} fills/min",
        f"ISSUES: {summary['metrics']['error_count']} errors, {summary['metrics']['warning_count']} warnings",
        "="*80 + "\n"
    ])
    
    return "\n".join(lines)


def main():
    """Main monitoring loop with enhanced alerting and status reporting."""
    last_alert = {}
    alert_cooldown = 300  # 5 minutes cooldown between alerts for the same issue
    last_report_time = 0
    report_interval = 3600  # 1 hour between full reports
    
    # Initial health check
    logger.info("Starting Divina Health Monitor")
    logger.info("Press Ctrl+C to stop monitoring\n")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs/divina', exist_ok=True)
    
    # Initialize the monitor
    monitor = HealthMonitor()
    if not monitor.exchanges:
        logger.error("No exchanges configured. Please check your environment variables.")
        return
    
    # Main monitoring loop
    while True:
        try:
            current_time = time.time()
            
            # Generate health summary
            summary = monitor.get_health_summary()
            
            # Log summary to console
            print(format_health_summary(summary))
            
            # Generate and send fix report periodically
            if current_time - last_report_time > report_interval:
                fix_report = monitor.generate_fix_report(summary)
                monitor.send_fix_report(fix_report)
                last_report_time = current_time
            
            # Check for issues and alert if needed
            issues_found = []
            
            # Check exchange connectivity
            for name, exchange in summary['exchanges'].items():
                if exchange['status'] != 'ONLINE':
                    issue_key = f"{name}_connectivity"
                    if issue_key not in last_alert or (current_time - last_alert[issue_key]) > alert_cooldown:
                        monitor.send_alert(
                            'ERROR',
                            f"{name.upper()} connection lost!",
                            {'status': exchange.get('error', 'Unknown error')}
                        )
                        last_alert[issue_key] = current_time
                    issues_found.append(f"{name.upper()} is {exchange['status']}")
                
                # Check for API errors
                if exchange.get('metrics', {}).get('api_errors', 0) > 0:
                    issue_key = f"{name}_api_errors"
                    if issue_key not in last_alert or (current_time - last_alert[issue_key]) > alert_cooldown:
                        monitor.send_alert(
                            'WARN',
                            f"{name.upper()} API errors detected",
                            {
                                'error_count': exchange['metrics']['api_errors'],
                                'last_error': exchange['metrics'].get('last_error', 'Unknown')
                            }
                        )
                        last_alert[issue_key] = current_time
            
            # Check for signal issues
            signals_per_min = summary['metrics']['signals_per_min']
            if signals_per_min == 0:
                issue_key = 'no_signals'
                if issue_key not in last_alert or (current_time - last_alert[issue_key]) > alert_cooldown:
                    monitor.send_alert(
                        'WARN',
                        "No trading signals detected in the last 24 hours",
                        {'signals_per_min': signals_per_min}
                    )
                    last_alert[issue_key] = current_time
                issues_found.append("No trading signals detected")
            
            # Check for order fill rate issues
            if summary['orders']['created'] > 0 and summary['orders']['filled'] == 0:
                issue_key = 'no_fills'
                if issue_key not in last_alert or (current_time - last_alert[issue_key]) > alert_cooldown:
                    monitor.send_alert(
                        'WARN',
                        "Orders created but none filled in the last 24 hours",
                        {
                            'orders_created': summary['orders']['created'],
                            'orders_filled': summary['orders']['filled']
                        }
                    )
                    last_alert[issue_key] = current_time
                issues_found.append("Orders created but none filled")
            
            # Log to file with timestamp
            log_entry = {
                'timestamp': datetime.now(pytz.UTC).isoformat(),
                'status': 'WARNING' if issues_found else 'OK',
                'issues': issues_found,
                'metrics': summary['metrics'],
                'exchanges': {name: {
                    'status': ex['status'],
                    'metrics': ex.get('metrics', {})
                } for name, ex in summary['exchanges'].items()}
            }
            
            log_file = f"logs/divina/status_{datetime.utcnow().strftime('%Y%m%d')}.log"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Sleep until next check (15 seconds)
            time.sleep(15)
            
        except KeyboardInterrupt:
            logger.info("\nShutting down Divina Health Monitor")
            # Generate and send final report before exiting
            try:
                fix_report = monitor.generate_fix_report(monitor.get_health_summary())
                fix_report['shutdown'] = True
                monitor.send_fix_report(fix_report)
            except Exception as e:
                logger.error(f"Error generating final report: {str(e)}")
            break
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)
            time.sleep(60)  # Back off on error


if __name__ == "__main__":
    main()
