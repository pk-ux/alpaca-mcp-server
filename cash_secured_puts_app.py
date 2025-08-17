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
from scipy.stats import norm
from typing import List, Dict, Optional

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
    page_title="Cash-Secured Put Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class MCPAlpacaClient:
    """Client to interact with the Alpaca MCP server"""
    
    def __init__(self):
        """Initialize the MCP client"""
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
        """Get current stock quote using MCP server"""
        if not symbol or not symbol.strip():
            raise DataValidationError("Empty symbol provided")
            
        try:
            result = self._call_mcp_function("mcp_alpaca_get_stock_quote", {"symbol": symbol.upper().strip()})
            
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
        """Get option snapshot with Greeks and pricing data from MCP server"""
        try:
            params = {"symbol_or_symbols": symbol}
            result = self._call_mcp_function("mcp_alpaca_get_option_snapshot", params)
            
            if result and "result" in result:
                # Parse the text response to extract Greeks and pricing data
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
            'data_source': data_source
        }

def main():
    """Main Streamlit application"""
    
    # Title
    st.title("Cash-Secured Put Options Analyzer")
    
    # Initialize the MCP client
    if 'mcp_client' not in st.session_state:
        with st.spinner("Connecting to Alpaca MCP server..."):
            st.session_state.mcp_client = MCPAlpacaClient()
    
    mcp_client = st.session_state.mcp_client
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Parameters")
    
    # Stock symbols input
    default_symbols = "PLTR,UNH,LLY"
    symbols_input = st.sidebar.text_input(
        "Stock Symbols (comma-separated)", 
        value=default_symbols,
        help="Enter stock symbols separated by commas"
    )
    
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Days to expiration filter
    max_dte = st.sidebar.slider(
        "Maximum Days to Expiration",
        min_value=1,
        max_value=45,
        value=20,
        help="Filter options with DTE less than or equal to this value"
    )
    
    # Maximum PITM filter
    max_pitm = st.sidebar.slider(
        "Maximum Probability ITM (%)",
        min_value=5,
        max_value=50,
        value=20,
        help="Filter options with PITM less than or equal to this percentage"
    )
    
    # Minimum open interest filter
    min_open_interest = st.sidebar.number_input(
        "Minimum Open Interest",
        min_value=0,
        max_value=1000,
        value=10,
        help="Filter options with open interest greater than this value"
    )
    
    if st.sidebar.button("🔍 Analyze Options", type="primary"):
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
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol}...")
            progress = (i + 1) / len(symbols)
            progress_bar.progress(progress)
            
            # Get stock quote with validation
            try:
                quote = mcp_client.get_stock_quote(symbol)
                if not quote:
                    st.warning(f"Could not fetch quote for {symbol}")
                    continue
                
                # Validate quote data
                quote = mcp_client.validate_stock_quote(quote)
                stock_price = quote['mid_price']
                
            except DataValidationError as e:
                logger.error(f"Invalid quote data for {symbol}: {str(e)}")
                st.warning(f"Could not get valid quote for {symbol}")
                continue
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {str(e)}")
                st.warning(f"Could not fetch quote for {symbol}")
                continue
            
            # Get real available expiration dates from Alpaca
            today = date.today()
            try:
                expiration_dates = mcp_client.get_available_expirations(symbol, max_dte)
                if not expiration_dates:
                    st.warning(f"No valid expirations found for {symbol} within {max_dte} days")
                    continue
            except Exception as e:
                logger.error(f"Error getting expiration dates for {symbol}: {str(e)}")
                st.warning(f"Could not get expiration dates for {symbol}")
                continue
            
            # Analyze each expiration date
            for exp_date in expiration_dates:
                options = mcp_client.get_option_contracts(symbol, exp_date, "put")
                
                if not options:
                    continue
                
                days_to_exp = (exp_date - today).days
                
                for option in options:
                    try:
                        # Validate option data
                        option = mcp_client.validate_option_data(option)
                        
                        # Apply basic filters
                        if option['open_interest'] < min_open_interest:
                            continue
                        
                        strike_price = option['strike_price']
                        premium = option['close_price']
                        
                        # Calculate metrics with real Greeks
                        metrics = mcp_client.calculate_option_metrics(
                            stock_price, strike_price, premium, days_to_exp, 
                            option_symbol=option['symbol']
                        )
                        
                        # Validate calculated metrics
                        metrics = mcp_client.validate_metrics(metrics)
                        
                        # Apply filters
                        if metrics['pitm'] > max_pitm:
                            continue
                            
                    except DataValidationError as e:
                        logger.warning(f"Skipping invalid option {option.get('symbol', 'unknown')}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing option {option.get('symbol', 'unknown')}: {str(e)}")
                        continue
                    
                    # Add to results - organized by relevance for cash-secured puts
                    result = {
                        # Core Analysis (Most Important)
                        'Ticker': symbol,
                        'Annualized Return (%)': f"{metrics['annualized_return']:.1f}%",
                        'PITM (%)': f"{metrics['pitm']:.1f}%",
                        'Premium Income': f"${metrics['premium_received']:.0f}",
                        'Cash Required': f"${metrics['cash_required']:,.0f}",
                        
                        # Option Details
                        'Strike Price': f"${strike_price:.2f}",
                        'Current Price': f"${stock_price:.2f}",
                        'Distance to Strike (%)': f"{metrics['distance_to_strike']:.1f}%",
                        'DTE': days_to_exp,
                        'Expiration Date': exp_date.strftime('%b %d, %Y'),
                        
                        # Pricing & Greeks
                        'Bid': f"${metrics['bid']:.2f}" if metrics['bid'] > 0 else f"${premium:.2f}",
                        'Ask': f"${metrics['ask']:.2f}" if metrics['ask'] > 0 else f"${premium:.2f}",
                        'Delta': f"{metrics['delta']:.3f}" if metrics['delta'] != 0 else "N/A",
                        'Theta': f"{metrics['theta']:.3f}" if metrics['theta'] != 0 else "N/A", 
                        
                        # Additional Info
                        'Implied Vol (%)': f"{metrics['implied_volatility']:.1f}%",
                        'Data Source': metrics['data_source'],
                        'Open Interest': option['open_interest'],
                        'Period Return (%)': f"{metrics['period_return']:.2f}%",
                        'sort_annualized': metrics['annualized_return']
                    }
                    
                    all_results.append(result)
        
        progress_bar.empty()
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
        

        # Display the results table
        st.subheader("Analysis Results")
        
        # Color coding for better visualization
        def highlight_best_returns(val):
            """Highlight cells based on annualized return"""
            if 'Annualized Return' in str(val):
                return_val = float(str(val).rstrip('%'))
                if return_val >= 15:
                    return 'background-color: #d4edda'  # Light green
                elif return_val >= 10:
                    return 'background-color: #fff3cd'  # Light yellow
            return ''
        
        # Display styled dataframe without index
        styled_df = df.style.applymap(highlight_best_returns)
        st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)
        
        # Top 3 recommendations
        st.subheader("Top 3 Recommendations")
        
        top_3 = df.head(3)
        
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            with st.expander(f"#{i}: {row['Ticker']} - {row['Annualized Return (%)']} Annual Return"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Symbol:** {row['Ticker']}")
                    st.write(f"**Current Price:** {row['Current Price']}")
                    st.write(f"**Strike Price:** {row['Strike Price']}")
                    st.write(f"**Expiration:** {row['Expiration Date']}")
                    st.write(f"**Days to Expiration:** {row['DTE']}")
                
                with col2:
                    st.write(f"**Bid:** {row['Bid']}")
                    st.write(f"**PITM:** {row['PITM (%)']}")
                    st.write(f"**Cash Required:** {row['Cash Required']}")
                    st.write(f"**Premium Income:** {row['Premium Income']}")
                    st.write(f"**Annualized Return:** {row['Annualized Return (%)']}**")
        
        # Export functionality
        st.subheader("Export Results")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"cash_secured_puts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
