# Cash-Secured Put Options Analyzer

A Streamlit web application for analyzing and comparing cash-secured put options to find the best opportunities for generating income while potentially acquiring stocks at a discount.

## Features

- **MCP Integration**: Uses the Alpaca MCP server for data fetching (no direct API calls needed)
- **Real-time Data**: Fetches live stock quotes and options data through the MCP server
- **Multiple Symbol Analysis**: Compare options across multiple stocks simultaneously
- **Advanced Filtering**: Filter by days to expiration, probability ITM, and open interest
- **Comprehensive Metrics**: Calculate annualized returns, cash requirements, and risk metrics
- **Interactive Interface**: User-friendly web interface with real-time updates
- **Greeks Integration**: Uses real option Greeks (Delta, Theta, etc.) from Alpaca when available

## What the Application Provides
- Interactive parameter selection in the sidebar
- Real-time progress tracking during analysis
- Comprehensive results table with color-coded returns
- Parallel processing for faster multi-symbol analysis
- Data source transparency (shows real vs estimated data)

## Installation

1. **Clone or download** this repository

2. **Install dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Ensure the Alpaca MCP server is running**:
   Make sure you have the `alpaca_mcp_server.py` in the same directory and that it's properly configured with your Alpaca API credentials in the `.env` file.
   
   The Streamlit app will automatically import and use the MCP server functions directly.

## Usage

1. **Start the application**:
   ```bash
   streamlit run cash_secured_puts_app.py
   ```

2. **Configure analysis parameters** in the sidebar:
   - **Stock Symbols**: Enter comma-separated ticker symbols (e.g., PLTR,UNH,LLY)
   - **Maximum Days to Expiration**: Filter options by time to expiration
   - **Maximum Probability ITM**: Set risk tolerance (lower = safer)
   - **Minimum Open Interest**: Ensure adequate liquidity
   - **Fast Processing**: Enable parallel processing for multiple symbols

3. **Click "Analyze Options"** to run the analysis

4. **Review results**:
   - Complete results table with color-coded returns
   - Best opportunity per ticker (highest annualized return)
   - Real-time data source indicators
   - Comprehensive options metrics and Greeks

## Key Metrics Explained

- **Annual Return %**: The premium income annualized based on days to expiration
- **PITM (Probability ITM)**: Estimated probability the option will be in-the-money at expiration (calculated using real Delta when available)
- **Cash Required**: Total cash needed to secure the put (strike price × 100)
- **Premium Income**: Total premium received for selling the put
- **Distance to Strike**: Safety margin - how far the current stock price is above the strike price
- **Greeks**: Real option sensitivities (Delta, Theta) from Alpaca when available
- **Data Source**: Shows whether using "Alpaca_Real" or "Estimated" calculations

## Cash-Secured Put Strategy

A cash-secured put involves:
1. **Selling a put option** on a stock you'd like to own
2. **Setting aside cash** equal to 100 shares at the strike price
3. **Collecting premium** immediately
4. **Potential outcomes**:
   - Stock stays above strike: Keep premium as profit
   - Stock falls below strike: Acquire shares at strike price (discount to original price)

## Risk Considerations

- **Assignment Risk**: You may be required to purchase shares if the stock falls below the strike
- **Opportunity Cost**: Cash is tied up during the option period
- **Market Risk**: Stock prices can be volatile
- **Liquidity Risk**: Some options may have low trading volume

## Current Limitations

The following features are not yet implemented:
- **CSV Export**: Download functionality for results
- **Volume Filtering**: Volume filter UI exists but not functional
- **Summary Dashboard**: Portfolio-level metrics overview
- **Watchlist Integration**: Import symbols from Alpaca watchlists

## Requirements

This application requires:
- **Alpaca MCP Server**: The `alpaca_mcp_server.py` file in the same directory
- **Alpaca Account**: Available at [alpaca.markets](https://alpaca.markets) with API credentials configured
- **MCP Server Setup**: Ensure the MCP server has proper `.env` configuration

## Troubleshooting

**Common Issues:**

1. **"Alpaca MCP server is not running"**: Ensure `alpaca_mcp_server.py` is in the same directory and properly configured
2. **"Could not import alpaca_mcp_server module"**: Verify the MCP server file is in the same directory as the Streamlit app
3. **"No options found"**: Try adjusting filters (increase max DTE or max PITM)
4. **"Import error"**: Ensure all required packages are installed

**Getting Help:**

- Check the Alpaca Markets documentation for API issues
- Verify your account has options trading enabled
- Ensure you're using paper trading credentials for testing

## Disclaimer

This application is for educational and analysis purposes only. It does not constitute financial advice. Options trading involves significant risk and may not be suitable for all investors. Always consult with a qualified financial advisor before making investment decisions.

## License

This project is provided as-is for educational purposes. Please ensure compliance with your local financial regulations when using this software.
