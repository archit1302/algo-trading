"""
Sample configuration for downloading SBIN 15-minute historical data.
"""

# Symbol configuration
SYMBOL = "SBIN"  # Trading symbol to download data for

# Timeframe configuration
UNIT = "minutes"  # Possible values: minutes, hours, days, weeks, months
INTERVAL = "15"   # Interval value (depends on unit)

# Date range
START_DATE = "2025-01-01"  # Format: YYYY-MM-DD
END_DATE = "2025-01-31"    # Format: YYYY-MM-DD

# API configuration
API_BASE_URL = "https://api.upstox.com/v3/historical-candle"

# Output configuration
OUTPUT_DIRECTORY = "historical_data"
OUTPUT_FORMAT = "csv"  # Options: csv, json

# API authorization (replace with your actual API key)
API_KEY = "your_api_key_here"
