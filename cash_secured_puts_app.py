#!/usr/bin/env python3
"""
Cash-Secured Put Options Analysis App
A Streamlit application for analyzing and comparing cash-secured put options
Uses the Alpaca MCP server for data fetching
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import subprocess
import json
import sys
import asyncio
import math
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from scipy.stats import norm
from typing import List, Dict, Optional, Tuple

# Configure logging for validation errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Options - Cash-Secured Puts",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robinhood-inspired CSS styling
st.markdown("""
<style>
    /* Color palette */
    :root {
        --green: #00d09c;
        --green-dark: #00c389;
        --bg-primary: #0d0d0d;
        --bg-secondary: #1a1a1a;
        --bg-tertiary: #2a2a2a;
        --text-primary: #ffffff;
        --text-secondary: #9ca3af;
        --border: #3a3a3a;
        --red: #ff6b6b;
        --yellow: #ffd93d;
    }
    
    /* Reset and full-width layout */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        margin: 0;
        padding: 0;
    }
    
    /* Main container - full width */
    .main .block-container {
        background: var(--bg-primary);
        max-width: none !important;
        padding: 0 0.5rem !important;
        margin: 0 !important;
    }
    
    /* Remove all spacing */
    .stApp > div, .main > div, section.main > div, .stMarkdown {
        margin: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Headers */
    h1 {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 400;
        margin: 0 !important;
        padding: 0 !important;
        margin-bottom: 0.5rem;
    }
    
    .stMarkdown p {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--bg-tertiary);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--green);
        color: var(--text-primary);
        border: none;
        border-radius: 24px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        background: var(--green-dark);
    }
    
    /* Table - full width */
    .stDataFrame {
        background: var(--bg-secondary);
        border: 1px solid var(--bg-tertiary);
        border-radius: 8px;
        width: 100% !important;
        margin: 0 !important;
    }
    
    div[data-testid="stDataFrame"] {
        width: 100% !important;
    }
    
    .stDataFrame thead th {
        background: var(--bg-tertiary);
        color: var(--text-secondary);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        padding: 0.75rem 0.5rem;
    }
    
    .stDataFrame tbody td {
        background: var(--bg-secondary);
        color: var(--text-primary);
        padding: 0.75rem 0.5rem;
        font-size: 0.875rem;
    }
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 4px;
        color: var(--text-primary);
    }
    
    .stSelectbox > div > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        color: var(--text-primary);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: var(--green);
    }
    
    .stProgress > div > div > div {
        background: var(--bg-tertiary);
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, .stApp > header {
        display: none !important;
    }
    
    /* Status messages */
    .stSuccess {
        background: rgba(0, 208, 156, 0.1);
        border: 1px solid var(--green);
        color: var(--green);
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid var(--red);
        color: var(--red);
    }
    
    .stWarning {
        background: rgba(255, 217, 61, 0.1);
        border: 1px solid var(--yellow);
        color: var(--yellow);
    }
</style>
""", unsafe_allow_html=True)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class CacheManager:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default TTL
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[any]:
        """Get cached value if not expired"""
        if key not in self.cache:
            return None
            
        # Check if expired
        if time.time() - self.timestamps[key] > self.default_ttl:
            self._expire(key)
            return None
            
        return self.cache[key]
    
    def set(self, key: str, value: any, ttl: Optional[int] = None) -> None:
        """Set cache value with TTL"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
        logger.debug(f"Cached: {key}")
    
    def _expire(self, key: str) -> None:
        """Remove expired cache entry"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Cache cleared")

class MCPAlpacaClient:
    """Client to interact with the Alpaca MCP server"""
    
    def __init__(self):
        """Initialize the MCP client"""
        self.cache = CacheManager(default_ttl=300)  # 5 minute cache
        self.server_running = self._check_mcp_server()
        if not self.server_running:
            st.error("Alpaca MCP server is not running. Please start it first.")
            st.stop()
    
    def _check_mcp_server(self) -> bool:
        """Check if the MCP server is accessible"""
        try:
            # Try to call a simple function to test connectivity
            result = self._call_mcp_function("mcp_alpaca_get_account_info", {"random_string": "test"})
            return result is not None
        except Exception:
            return False
    
    def _call_mcp_function(self, function_name: str, params: Dict) -> Optional[Dict]:
        """Call an MCP function and return the result"""
        try:
            # Since we can't directly call MCP functions from Streamlit,
            # we'll import and call the functions from the alpaca_mcp_server module
            import sys
            import os
            
            # Add the current directory to the Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Import the MCP server functions
            try:
                import alpaca_mcp_server as mcp
            except ImportError:
                st.error("Could not import alpaca_mcp_server module. Please ensure it's in the same directory.")
                return None
            
            # Map function names to actual functions
            function_map = {
                "mcp_alpaca_get_stock_quote": mcp.get_stock_quote,
                "mcp_alpaca_get_option_contracts": mcp.get_option_contracts,
                "mcp_alpaca_get_option_snapshot": mcp.get_option_snapshot,
                "mcp_alpaca_get_account_info": mcp.get_account_info
            }
            
            if function_name not in function_map:
                st.error(f"Unknown MCP function: {function_name}")
                return None
            
            # Call the function
            if function_name == "mcp_alpaca_get_stock_quote":
                result = asyncio.run(function_map[function_name](params.get("symbol", "")))
            elif function_name == "mcp_alpaca_get_option_contracts":
                result = asyncio.run(function_map[function_name](
                    underlying_symbol=params.get("underlying_symbol", ""),
                    expiration_date=params.get("expiration_date"),
                    expiration_month=params.get("expiration_month"),
                    expiration_year=params.get("expiration_year"),
                    strike_price_gte=params.get("strike_price_gte"),
                    strike_price_lte=params.get("strike_price_lte"),
                    type=params.get("type"),
                    status=params.get("status"),
                    limit=params.get("limit")
                ))
            elif function_name == "mcp_alpaca_get_option_snapshot":
                result = asyncio.run(function_map[function_name](
                    symbol_or_symbols=params.get("symbol_or_symbols", "")
                ))
            elif function_name == "mcp_alpaca_get_account_info":
                result = asyncio.run(function_map[function_name]())
            else:
                result = asyncio.run(function_map[function_name](**params))
            
            return {"result": result}
            
        except Exception as e:
            st.error(f"Error calling MCP function {function_name}: {str(e)}")
            return None
    
    def get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """Get current stock quote using MCP server with caching"""
        if not symbol or not symbol.strip():
            raise DataValidationError("Empty symbol provided")
        
        symbol = symbol.upper().strip()
        cache_key = f"quote_{symbol}"
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for quote: {symbol}")
            return cached_result
            
        try:
            result = self._call_mcp_function("mcp_alpaca_get_stock_quote", {"symbol": symbol})
            
            if not result or "result" not in result:
                raise DataValidationError(f"No quote data returned for {symbol}")
            
            # Parse the result string to extract quote data
            quote_text = result["result"]
            if not quote_text or quote_text.strip() == "":
                raise DataValidationError(f"Empty quote response for {symbol}")
                
            lines = quote_text.strip().split('\n')
            quote_data = {}
            
            for line in lines:
                try:
                    if "Ask Price:" in line:
                        quote_data['ask'] = float(line.split('$')[1].strip())
                    elif "Bid Price:" in line:
                        quote_data['bid'] = float(line.split('$')[1].strip())
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse price from line: {line}")
                    continue
            
            # Ensure we have both bid and ask
            if 'ask' not in quote_data or 'bid' not in quote_data:
                raise DataValidationError(f"Incomplete quote data for {symbol} - missing bid or ask")
            
            if 'ask' in quote_data and 'bid' in quote_data:
                quote_data['mid_price'] = (quote_data['ask'] + quote_data['bid']) / 2
                quote_data['symbol'] = symbol
                
                # Cache the result (quotes expire quickly - 30 seconds)
                self.cache.set(cache_key, quote_data, ttl=30)
                return quote_data
                    
            return None
            
        except DataValidationError:
            raise  # Re-raise validation errors
        except ConnectionError:
            raise DataValidationError(f"Connection error while fetching quote for {symbol}")
        except Exception as e:
            raise DataValidationError(f"Unexpected error fetching quote for {symbol}: {str(e)}")
    
    def get_option_contracts(self, symbol: str, expiration_date: date, contract_type: str = "put") -> List[Dict]:
        """Get option contracts for a symbol and expiration date using MCP server"""
        try:
            # Format the expiration date as string
            exp_date_str = expiration_date.strftime("%Y-%m-%d")
            
            params = {
                "underlying_symbol": symbol,
                "expiration_date": exp_date_str,
                "type": contract_type.lower(),
                "status": "active",
                "limit": 100
            }
            
            result = self._call_mcp_function("mcp_alpaca_get_option_contracts", params)
            
            if result and "result" in result:
                # Parse the result string to extract option data
                contracts_text = result["result"]
                option_data = []
                
                # Split by contract separators
                contracts = contracts_text.split("-------------------------")
                
                for contract_block in contracts:
                    if "Symbol:" in contract_block and "Put" in contract_block:
                        lines = contract_block.strip().split('\n')
                        contract_info = {}
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith("Symbol:"):
                                contract_info['symbol'] = line.split(": ")[1]
                            elif line.startswith("Name:"):
                                contract_info['name'] = line.split(": ")[1]
                            elif line.startswith("Strike Price:"):
                                try:
                                    price_str = line.split("$")[1]
                                    contract_info['strike_price'] = float(price_str)
                                except:
                                    continue
                            elif line.startswith("Expiration Date:"):
                                contract_info['expiration_date'] = line.split(": ")[1]
                            elif line.startswith("Open Interest:"):
                                try:
                                    oi_str = line.split(": ")[1]
                                    if oi_str != "None":
                                        contract_info['open_interest'] = int(oi_str)
                                    else:
                                        contract_info['open_interest'] = 0
                                except:
                                    contract_info['open_interest'] = 0
                            elif line.startswith("Close Price:"):
                                try:
                                    price_str = line.split("$")[1]
                                    contract_info['close_price'] = float(price_str)
                                except:
                                    contract_info['close_price'] = 0.0
                        
                        if 'strike_price' in contract_info and contract_info['strike_price'] > 0:
                            option_data.append(contract_info)
                
                return sorted(option_data, key=lambda x: x['strike_price'])
            
            return []
            
        except Exception as e:
            st.error(f"Error fetching options for {symbol}: {str(e)}")
            return []

    def get_option_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get option snapshot with Greeks, pricing data, and volume from MCP server"""
        try:
            params = {"symbol_or_symbols": symbol}
            result = self._call_mcp_function("mcp_alpaca_get_option_snapshot", params)
            
            if result and "result" in result:
                # Parse the text response to extract Greeks, pricing data, and volume
                snapshot_text = result["result"]
                lines = snapshot_text.split('\n')
                data = {}
                
                for line in lines:
                    line = line.strip()
                    if 'Delta:' in line:
                        try:
                            delta_str = line.split('Delta:')[1].strip()
                            data['delta'] = float(delta_str)
                        except ValueError as e:
                            logger.warning(f"Could not parse Delta from line: {line}")
                            continue
                    elif 'Gamma:' in line:
                        try:
                            gamma_str = line.split('Gamma:')[1].strip()
                            data['gamma'] = float(gamma_str)
                        except ValueError:
                            continue
                    elif 'Theta:' in line:
                        try:
                            theta_str = line.split('Theta:')[1].strip()
                            data['theta'] = float(theta_str)
                        except ValueError:
                            continue
                    elif 'Vega:' in line:
                        try:
                            vega_str = line.split('Vega:')[1].strip()
                            data['vega'] = float(vega_str)
                        except ValueError:
                            continue
                    elif 'Rho:' in line:
                        try:
                            rho_str = line.split('Rho:')[1].strip()
                            data['rho'] = float(rho_str)
                        except ValueError:
                            continue
                    elif 'Implied Volatility:' in line:
                        try:
                            iv_str = line.split('Implied Volatility:')[1].strip().replace('%', '')
                            data['implied_volatility'] = float(iv_str)
                        except ValueError:
                            continue
                    elif 'Bid Price:' in line:
                        try:
                            bid_str = line.split('Bid Price:')[1].strip().replace('$', '')
                            data['bid'] = float(bid_str)
                        except ValueError:
                            continue
                    elif 'Ask Price:' in line:
                        try:
                            ask_str = line.split('Ask Price:')[1].strip().replace('$', '')
                            data['ask'] = float(ask_str)
                        except ValueError:
                            continue
                    elif line.startswith('Price:') and 'Bid Price:' not in line and 'Ask Price:' not in line:
                        try:
                            # This is likely the trade price
                            price_str = line.split('Price:')[1].strip().replace('$', '')
                            data['last_price'] = float(price_str)
                        except ValueError:
                            continue
                    elif 'Size:' in line and 'Bid Size:' not in line and 'Ask Size:' not in line:
                        try:
                            # This is likely the trade size (volume indicator)
                            size_str = line.split('Size:')[1].strip()
                            data['last_trade_size'] = int(size_str)
                        except ValueError:
                            continue
                    elif 'Bid Size:' in line:
                        try:
                            bid_size_str = line.split('Bid Size:')[1].strip()
                            data['bid_size'] = int(bid_size_str)
                        except ValueError:
                            continue
                    elif 'Ask Size:' in line:
                        try:
                            ask_size_str = line.split('Ask Size:')[1].strip()
                            data['ask_size'] = int(ask_size_str)
                        except ValueError:
                            continue
                
                return data
            
            return None
            
        except ConnectionError:
            logger.error(f"Connection error while fetching option snapshot for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching option snapshot for {symbol}: {str(e)}")
            return None

    def get_available_expirations(self, symbol: str, max_days: int) -> List[date]:
        """Get real expiration dates from Alpaca for the given symbol"""
        try:
            # Get option contracts for the symbol with a reasonable date range
            today = date.today()
            end_date = today + timedelta(days=max_days)
            
            # First, try to get contracts for different expiration months
            expirations = set()
            
            # Check current month and next month
            for month_offset in range(0, 3):  # Current + next 2 months
                check_date = today + timedelta(days=30 * month_offset)
                
                params = {
                    "underlying_symbol": symbol,
                    "expiration_month": check_date.month,
                    "expiration_year": check_date.year,
                    "limit": 50
                }
                
                result = self._call_mcp_function("mcp_alpaca_get_option_contracts", params)
                
                if result and "result" in result:
                    contracts_text = result["result"]
                    
                    # Extract expiration dates from contract symbols
                    for line in contracts_text.split('\n'):
                        if 'Symbol:' in line:
                            symbol_text = line.split('Symbol:')[1].strip()
                            # Parse expiration from option symbol (format: TICKER250822C00150000)
                            if len(symbol_text) >= 15:
                                try:
                                    date_part = symbol_text[-15:-9]  # Extract YYMMDD
                                    exp_year = 2000 + int(date_part[:2])
                                    exp_month = int(date_part[2:4]) 
                                    exp_day = int(date_part[4:6])
                                    exp_date = date(exp_year, exp_month, exp_day)
                                    
                                    # Only include dates within our range
                                    if today <= exp_date <= end_date:
                                        expirations.add(exp_date)
                                except (ValueError, IndexError):
                                    continue
            
            # Convert to sorted list
            expiration_list = sorted(list(expirations))
            
            # If no expirations found, fall back to Friday logic but with validation
            if not expiration_list:
                logger.warning(f"No option expirations found for {symbol}, using Friday estimates")
                return self._get_friday_estimates(max_days)
            
            return expiration_list[:6]  # Limit to first 6 expirations
            
        except Exception as e:
            logger.warning(f"Error getting expirations for {symbol}: {str(e)}, using Friday estimates")
            return self._get_friday_estimates(max_days)
    
    def _get_friday_estimates(self, max_days: int) -> List[date]:
        """Fallback method to estimate Friday expirations"""
        today = date.today()
        fridays = []
        
        # Find next Friday
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:
            days_ahead += 7
        
        current_friday = today + timedelta(days=days_ahead)
        
        # Get next few Fridays within max_days
        while len(fridays) < 6 and (current_friday - today).days <= max_days:
            fridays.append(current_friday)
            current_friday += timedelta(days=7)
        
        return fridays
    
    @staticmethod
    def validate_stock_quote(quote: Dict) -> Dict:
        """Validate and clean stock quote data"""
        if not quote:
            raise DataValidationError("Empty quote data received")
        
        required_fields = ['bid', 'ask']
        for field in required_fields:
            if field not in quote:
                raise DataValidationError(f"Missing required field in quote: {field}")
            
            if not isinstance(quote[field], (int, float)) or quote[field] <= 0:
                raise DataValidationError(f"Invalid {field} price: {quote[field]}")
        
        # Calculate mid price if not present
        if 'mid_price' not in quote:
            quote['mid_price'] = (quote['bid'] + quote['ask']) / 2
        
        # Validate spread isn't too wide (more than 10%)
        spread_pct = (quote['ask'] - quote['bid']) / quote['mid_price'] * 100
        if spread_pct > 10:
            logger.warning(f"Wide bid-ask spread detected for quote: {spread_pct:.1f}%")
        
        return quote
    
    @staticmethod 
    def validate_option_data(option: Dict) -> Dict:
        """Validate and clean option contract data"""
        if not option:
            raise DataValidationError("Empty option data received")
        
        # Required fields
        required_fields = ['symbol', 'strike_price', 'close_price', 'open_interest']
        for field in required_fields:
            if field not in option:
                raise DataValidationError(f"Missing required field in option: {field}")
        
        # Validate numeric fields
        if option['strike_price'] <= 0:
            raise DataValidationError(f"Invalid strike price: {option['strike_price']}")
            
        if option['close_price'] <= 0:
            raise DataValidationError(f"Invalid option price: {option['close_price']}")
            
        if option['open_interest'] < 0:
            raise DataValidationError(f"Invalid open interest: {option['open_interest']}")
        
        # Validate option symbol format
        if not option['symbol'] or len(option['symbol']) < 10:
            raise DataValidationError(f"Invalid option symbol: {option['symbol']}")
        
        return option
    
    @staticmethod
    def validate_metrics(metrics: Dict) -> Dict:
        """Validate calculated metrics"""
        if not metrics:
            raise DataValidationError("Empty metrics data")
        
        # Check for required fields
        required_fields = ['annualized_return', 'pitm', 'cash_required', 'premium_received']
        for field in required_fields:
            if field not in metrics:
                raise DataValidationError(f"Missing metric: {field}")
            
            if not isinstance(metrics[field], (int, float)):
                raise DataValidationError(f"Invalid metric type for {field}: {type(metrics[field])}")
        
        # Sanity checks
        if metrics['annualized_return'] < 0 or metrics['annualized_return'] > 1000:
            logger.warning(f"Unusual annualized return detected: {metrics['annualized_return']:.1f}%")
        
        if metrics['pitm'] < 0 or metrics['pitm'] > 100:
            raise DataValidationError(f"PITM out of range: {metrics['pitm']}")
        
        if metrics['cash_required'] <= 0:
            raise DataValidationError(f"Invalid cash required: {metrics['cash_required']}")
        
        return metrics
    
    def process_symbol_parallel(self, symbol: str, max_dte: int, max_pitm: float, 
                               min_open_interest: int, min_volume: int = 0) -> List[Dict]:
        """Process a single symbol and return all valid options"""
        try:
            # Get stock quote with validation
            quote = self.get_stock_quote(symbol)
            if not quote:
                logger.warning(f"Could not fetch quote for {symbol}")
                return []
            
            # Validate quote data
            quote = self.validate_stock_quote(quote)
            stock_price = quote['mid_price']
            
            # Get real available expiration dates from Alpaca
            today = date.today()
            expiration_dates = self.get_available_expirations(symbol, max_dte)
            if not expiration_dates:
                logger.warning(f"No valid expirations found for {symbol} within {max_dte} days")
                return []
            
            all_options = []
            
            # Analyze each expiration date
            for exp_date in expiration_dates:
                options = self.get_option_contracts(symbol, exp_date, "put")
                
                if not options:
                    continue
                
                days_to_exp = (exp_date - today).days
                
                for option in options:
                    try:
                        # Validate option data
                        option = self.validate_option_data(option)
                        
                        # Apply basic filters
                        if option['open_interest'] < min_open_interest:
                            continue
                        
                        strike_price = option['strike_price']
                        premium = option['close_price']
                        
                        # Calculate metrics with real Greeks
                        metrics = self.calculate_option_metrics(
                            stock_price, strike_price, premium, days_to_exp, 
                            option_symbol=option['symbol']
                        )
                        
                        # Validate calculated metrics
                        metrics = self.validate_metrics(metrics)
                        
                        # Apply filters
                        if metrics['pitm'] > max_pitm:
                            continue
                        
                        # Apply volume filter if specified
                        if min_volume > 0 and metrics['volume'] < min_volume:
                            continue
                        
                        # Create result object with shorter column names for better visibility
                        result = {
                            # Core Analysis (Most Important)
                            'Ticker': symbol,
                            'Annual Return %': f"{metrics['annualized_return']:.1f}%",
                            'PITM %': f"{metrics['pitm']:.1f}%",
                            'Premium $': f"${metrics['premium_received']:.0f}",
                            'Cash Req $': f"${metrics['cash_required']:,.0f}",
                            
                            # Option Details
                            'Strike $': f"${strike_price:.2f}",
                            'Current $': f"${stock_price:.2f}",
                            'Distance %': f"{metrics['distance_to_strike']:.1f}%",
                            'DTE': days_to_exp,
                            'Expiry': exp_date.strftime('%m/%d'),
                            
                            # Pricing & Greeks
                            'Bid $': f"${metrics['bid']:.2f}" if metrics['bid'] > 0 else f"${premium:.2f}",
                            'Ask $': f"${metrics['ask']:.2f}" if metrics['ask'] > 0 else f"${premium:.2f}",
                            'Delta': f"{metrics['delta']:.3f}" if metrics['delta'] != 0 else "N/A",
                            'Theta': f"{metrics['theta']:.3f}" if metrics['theta'] != 0 else "N/A", 
                            
                            # Additional Info
                            'IV %': f"{metrics['implied_volatility']:.1f}%",
                            'OI': option['open_interest'],
                            'Volume': metrics['volume'] if metrics['volume'] > 0 else "N/A",
                            'Period %': f"{metrics['period_return']:.2f}%",
                            'Source': metrics['data_source'],
                            'sort_annualized': metrics['annualized_return']
                        }
                        
                        all_options.append(result)
                        
                    except DataValidationError as e:
                        logger.warning(f"Skipping invalid option {option.get('symbol', 'unknown')}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing option {option.get('symbol', 'unknown')}: {str(e)}")
                        continue
            
            logger.info(f"Processed {symbol}: found {len(all_options)} valid options")
            return all_options
            
        except DataValidationError as e:
            logger.error(f"Invalid data for {symbol}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            return []
    
    def calculate_implied_volatility(self, stock_price: float, strike_price: float, 
                                   premium: float, days_to_expiration: int, 
                                   risk_free_rate: float = 0.05) -> float:
        """Estimate implied volatility using simplified approach"""
        # This is a rough approximation - in reality you'd use Newton-Raphson
        # For now, we'll estimate based on typical market volatilities
        
        # Time to expiration in years
        time_to_exp = days_to_expiration / 365.0
        
        # Moneyness (how far ITM/OTM the option is)
        moneyness = stock_price / strike_price
        
        # Rough IV estimation based on premium and moneyness
        # This is very simplified but gives reasonable estimates
        if moneyness > 1.1:  # Deep OTM put
            return 0.25  # 25% IV
        elif moneyness > 1.05:  # Moderately OTM
            return 0.35  # 35% IV
        elif moneyness > 0.95:  # Near the money
            return 0.45  # 45% IV
        else:  # ITM put
            return 0.55  # 55% IV

    def calculate_pitm_black_scholes(self, stock_price: float, strike_price: float,
                                   days_to_expiration: int, volatility: float,
                                   risk_free_rate: float = 0.05) -> float:
        """Calculate probability ITM using Black-Scholes framework"""
        
        if days_to_expiration <= 0:
            # Option has expired
            return 100.0 if stock_price < strike_price else 0.0
        
        # Time to expiration in years
        T = days_to_expiration / 365.0
        
        # Prevent division by zero
        if T <= 0 or volatility <= 0:
            return 50.0  # Default fallback
        
        try:
            # Black-Scholes d2 parameter
            d2 = (math.log(stock_price / strike_price) + 
                  (risk_free_rate - 0.5 * volatility**2) * T) / (volatility * math.sqrt(T))
            
            # For a put option, PITM = N(-d2) where N is cumulative normal distribution
            pitm = norm.cdf(-d2) * 100
            
            # Ensure reasonable bounds
            return max(0.1, min(99.9, pitm))
            
        except (ValueError, ZeroDivisionError, OverflowError):
            # Fallback to simple calculation if Black-Scholes fails
            distance_pct = ((stock_price - strike_price) / strike_price) * 100
            if distance_pct > 20:
                return 5.0
            elif distance_pct > 10:
                return 15.0
            elif distance_pct > 0:
                return 25.0
            else:
                return 50.0

    def calculate_option_metrics(self, stock_price: float, strike_price: float, 
                                premium: float, days_to_expiration: int, 
                                option_symbol: str = None) -> Dict:
        """Calculate option metrics for cash-secured puts"""
        
        # Cash required (strike price * 100 shares)
        cash_required = strike_price * 100
        
        # Premium received (premium * 100 shares)
        premium_received = premium * 100
        
        # Period return
        period_return = (premium_received / cash_required) * 100
        
        # Annualized return
        annualized_return = period_return * (365 / max(days_to_expiration, 1))
        
        # Distance to strike (for reference)
        distance_to_strike = ((stock_price - strike_price) / stock_price) * 100
        
        # Try to get real Greeks and IV from Alpaca if option symbol provided
        real_greeks = {}
        if option_symbol:
            snapshot = self.get_option_snapshot(option_symbol)
            if snapshot:
                real_greeks = snapshot
        
        # Calculate PITM using Delta (most accurate method)
        if 'delta' in real_greeks and 'implied_volatility' in real_greeks:
            # Use real data from Alpaca
            pitm = abs(real_greeks['delta']) * 100
            implied_vol = real_greeks.get('implied_volatility', 0) / 100  # Convert from percentage to decimal
            data_source = "Alpaca_Real"
        else:
            # Fallback to simplified calculation if no Greeks available
            # This should rarely happen if MCP server is working properly
            implied_vol = self.calculate_implied_volatility(
                stock_price, strike_price, premium, days_to_expiration
            )
            pitm = self.calculate_pitm_black_scholes(
                stock_price, strike_price, days_to_expiration, implied_vol
            )
            data_source = "Estimated"
        
        return {
            'cash_required': cash_required,
            'premium_received': premium_received,
            'period_return': period_return,
            'annualized_return': annualized_return,
            'pitm': pitm,
            'distance_to_strike': distance_to_strike,
            'implied_volatility': implied_vol * 100 if implied_vol else 0,  # Convert to percentage
            'delta': real_greeks.get('delta', 0),
            'gamma': real_greeks.get('gamma', 0),
            'theta': real_greeks.get('theta', 0),
            'vega': real_greeks.get('vega', 0),
            'rho': real_greeks.get('rho', 0),
            'bid': real_greeks.get('bid', 0),
            'ask': real_greeks.get('ask', 0),
            'last_price': real_greeks.get('last_price', premium),
            'volume': real_greeks.get('last_trade_size', 0),  # Use last trade size as volume indicator
            'bid_size': real_greeks.get('bid_size', 0),
            'ask_size': real_greeks.get('ask_size', 0),
            'data_source': data_source
        }

def main():
    """Main Streamlit application"""
    
    # Simple header
    st.markdown("# Options")
    st.markdown("Cash-secured puts - Find opportunities to buy the dip")
    
    # Initialize the MCP client
    if 'mcp_client' not in st.session_state:
        with st.spinner("Connecting to Alpaca MCP server..."):
            st.session_state.mcp_client = MCPAlpacaClient()
    
    mcp_client = st.session_state.mcp_client
    
    # Simple sidebar header
    st.sidebar.markdown("### Filters")
    
    # Stock symbols input
    default_symbols = "PLTR,UNH,LLY"
    symbols_input = st.sidebar.text_input(
        "Stock Symbols (comma-separated)", 
        value=default_symbols,
        help="Enter stock symbols separated by commas"
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Compact filters in columns
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        max_dte = st.number_input(
            "Max DTE",
            min_value=1,
            max_value=45,
            value=20,
            help="Days to expiration"
        )
        
        min_open_interest = st.number_input(
            "Min OI",
            min_value=0,
            max_value=1000,
            value=10,
            help="Open interest"
        )
    
    with col2:
        max_pitm = st.number_input(
            "Max PITM %",
            min_value=5,
            max_value=50,
            value=20,
            help="Probability ITM"
        )
        
        min_volume = st.number_input(
            "Min Volume",
            min_value=0,
            max_value=10000,
            value=0,
            help="Daily volume"
        )
    
    # Prominent analyze button
    st.sidebar.markdown("---")
    if st.sidebar.button("Analyze Options", type="primary", use_container_width=True):
        # Analysis logic starts here
        
        # Compact settings
        use_parallel = st.sidebar.checkbox("Fast Processing", value=True)
        
        # Input validation
        if not symbols:
            st.error("Please enter at least one stock symbol")
            return
        
        # Validate symbol format
        invalid_symbols = []
        for symbol in symbols:
            if not symbol.isalpha() or len(symbol) > 10:
                invalid_symbols.append(symbol)
        
        if invalid_symbols:
            st.error(f"Invalid stock symbols: {', '.join(invalid_symbols)}")
            return
        
        # Validate parameter ranges
        if max_dte <= 0 or max_dte > 365:
            st.error("Days to expiration must be between 1 and 365")
            return
            
        if max_pitm <= 0 or max_pitm > 100:
            st.error("Maximum PITM must be between 1 and 100")
            return
            
        if min_open_interest < 0:
            st.error("Minimum open interest cannot be negative")
            return
            
        if min_volume < 0:
            st.error("Minimum volume cannot be negative")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        start_time = time.time()
        
        if use_parallel and len(symbols) > 1:
            # Parallel processing for multiple symbols
            status_text.text("Processing symbols in parallel...")
            
            with ThreadPoolExecutor(max_workers=min(len(symbols), 4)) as executor:
                # Submit all symbol processing tasks
                future_to_symbol = {
                    executor.submit(mcp_client.process_symbol_parallel, symbol, max_dte, max_pitm, min_open_interest, min_volume): symbol 
                    for symbol in symbols
                }
                
                completed = 0
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    progress = completed / len(symbols)
                    progress_bar.progress(progress)
                    status_text.text(f"Completed {symbol} ({completed}/{len(symbols)})")
                    
                    try:
                        symbol_results = future.result()
                        all_results.extend(symbol_results)
                        logger.info(f"Parallel processing completed for {symbol}: {len(symbol_results)} options")
                    except Exception as e:
                        logger.error(f"Error in parallel processing for {symbol}: {str(e)}")
                        st.warning(f"Error processing {symbol}")
                        continue
        else:
            # Sequential processing (fallback or single symbol)
            for i, symbol in enumerate(symbols):
                status_text.text(f"Analyzing {symbol}...")
                progress = (i + 1) / len(symbols)
                progress_bar.progress(progress)
                
                try:
                    symbol_results = mcp_client.process_symbol_parallel(symbol, max_dte, max_pitm, min_open_interest, min_volume)
                    all_results.extend(symbol_results)
                    logger.info(f"Sequential processing completed for {symbol}: {len(symbol_results)} options")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    st.warning(f"Error processing {symbol}")
                    continue
        
        # Volume filtering is now implemented within option processing
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        status_text.text(f"Processing completed in {processing_time:.1f}s")
        
        progress_bar.empty()
        time.sleep(1)  # Brief pause to show completion message
        status_text.empty()
        
        if not all_results:
            st.warning("No options found matching your criteria. Try adjusting the filters.")
            return
        
        # Create DataFrame and get best result per ticker
        df = pd.DataFrame(all_results)
        
        # Get only the best result for each ticker based on annualized return
        df = df.loc[df.groupby('Ticker')['sort_annualized'].idxmax()]
        
        # Sort by annualized return (highest first)
        df = df.sort_values('sort_annualized', ascending=False)
        
        # Reset index to remove the original row numbers and create clean 0,1,2... indexing
        df = df.reset_index(drop=True)
        
        # Remove any empty rows more aggressively
        df = df.dropna(how='all')  # Drop rows where all values are NaN
        df = df.dropna(subset=['Ticker'])  # Drop rows where Ticker is NaN/empty
        
        # Remove rows with empty or None tickers
        df = df[df['Ticker'].notna()]
        df = df[df['Ticker'].astype(str).str.strip() != '']
        df = df[df['Ticker'].astype(str) != 'nan']
        
        # Final cleanup - remove any remaining empty rows
        # Check if any essential columns are empty
        essential_cols = ['Ticker', 'Strike Price', 'Annualized Return (%)']
        for col in essential_cols:
            if col in df.columns:
                df = df[df[col].notna()]
                df = df[df[col].astype(str).str.strip() != '']
        
        # Reset index again after all filtering
        df = df.reset_index(drop=True)
        
        df = df.drop('sort_annualized', axis=1)  # Remove sorting column
        
        # Display results
        

        # Results section - no header needed
        
        # Robinhood-style color coding
        def highlight_robinhood_style(val):
            """Robinhood-style highlighting with signature colors"""
            if 'Annual Return' in str(val):
                try:
                    return_val = float(str(val).rstrip('%'))
                    if return_val >= 15:
                        return 'color: #00d09c; font-weight: 500;'  # Robinhood green
                    elif return_val >= 10:
                        return 'color: #ffffff; font-weight: 500;'  # White
                    elif return_val >= 5:
                        return 'color: #9ca3af; font-weight: 400;'  # Gray
                    else:
                        return 'color: #ff6b6b; font-weight: 500;'  # Robinhood red
                except ValueError:
                    pass
            elif 'PITM' in str(val):
                try:
                    pitm_val = float(str(val).rstrip('%'))
                    if pitm_val <= 10:
                        return 'color: #00d09c; font-weight: 500;'  # Low risk = green
                    elif pitm_val <= 15:
                        return 'color: #ffffff; font-weight: 500;'  # Medium risk = white  
                    elif pitm_val <= 20:
                        return 'color: #ffd93d; font-weight: 500;'  # Higher risk = yellow
                    else:
                        return 'color: #ff6b6b; font-weight: 500;'  # High risk = red
                except ValueError:
                    pass
            return 'color: #ffffff;'  # Default white text
        
        # Display styled dataframe without index - full width
        styled_df = df.style.applymap(highlight_robinhood_style)
        st.dataframe(styled_df, use_container_width=True, height=500, hide_index=True)
        
    # Cache management at bottom (optional)
    with st.sidebar.expander("Advanced", expanded=False):
        cache_size = len(mcp_client.cache.cache)
        st.metric("Cache", f"{cache_size} items")
        if st.button("Clear Cache", help="Clear all cached data"):
            mcp_client.cache.clear()
            st.rerun()


if __name__ == "__main__":
    main()
