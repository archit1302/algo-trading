"""
Module 2: Technical Analysis and Data Processing
Setup Script for Dependencies and Configuration

This script sets up the complete environment for Module 2 learning materials.
It installs required packages, downloads sample data, and configures the workspace.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_package(package):
    """Check if a package is already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("  MODULE 2: TECHNICAL ANALYSIS SETUP")
    print("=" * 60)
    
    # Core dependencies for Module 2
    core_packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "pandas-datareader>=0.10.0"
    ]
    
    # Optional packages for advanced features
    optional_packages = [
        "jupyter>=1.0.0",
        "plotly>=5.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "openpyxl>=3.0.0",  # For Excel file support
        "requests>=2.25.0"   # For API calls
    ]
    
    print("\n🔧 Installing Core Dependencies...")
    print("-" * 40)
    
    failed_packages = []
    
    for package in core_packages:
        package_name = package.split(">=")[0]
        if not check_package(package_name):
            if not install_package(package):
                failed_packages.append(package)
        else:
            print(f"✅ {package_name} already installed")
    
    print("\n🎯 Installing Optional Packages...")
    print("-" * 40)
    
    for package in optional_packages:
        package_name = package.split(">=")[0]
        if not check_package(package_name):
            install_package(package)
        else:
            print(f"✅ {package_name} already installed")
    
    # Create necessary directories
    print("\n📁 Creating Directory Structure...")
    print("-" * 40)
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/indicators",
        "notebooks",
        "charts",
        "reports",
        "backtest_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create configuration file
    config_content = '''# Module 2 Configuration File
# Technical Analysis and Data Processing Settings

# Data Settings
DATA_DIRECTORY = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
INDICATORS_DIR = "data/indicators"

# Chart Settings
CHART_DIRECTORY = "charts"
CHART_STYLE = "seaborn-v0_8"
FIGURE_SIZE = (12, 8)
DPI = 300

# Analysis Settings
DEFAULT_TIMEFRAME = "1d"
LOOKBACK_PERIODS = {
    "short": 10,
    "medium": 20, 
    "long": 50,
    "very_long": 200
}

# Technical Indicator Parameters
INDICATOR_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2
}

# Performance Analysis
BENCHMARK_SYMBOL = "^NSEI"  # Nifty 50 Index
RISK_FREE_RATE = 0.06       # 6% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252

# Backtesting Settings
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001    # 0.1% per trade
SLIPPAGE = 0.0005          # 0.05% slippage
'''
    
    with open("config.py", "w") as f:
        f.write(config_content)
    print("✅ Created configuration file: config.py")
    
    # Create sample data downloader
    downloader_content = '''"""
Sample Data Downloader for Module 2
Downloads historical market data for learning and practice
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(symbols, period="1y", interval="1d"):
    """Download stock data for given symbols"""
    print(f"📥 Downloading data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                # Save to CSV
                filename = f"data/raw/{symbol}_{period}_{interval}.csv"
                data.to_csv(filename)
                print(f"✅ {symbol}: {len(data)} records saved to {filename}")
            else:
                print(f"❌ {symbol}: No data available")
                
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

def main():
    """Download sample data for Module 2"""
    # Indian stock symbols for practice
    indian_stocks = [
        "SBIN.NS",      # State Bank of India
        "RELIANCE.NS",  # Reliance Industries  
        "TCS.NS",       # Tata Consultancy Services
        "INFY.NS",      # Infosys
        "HDFCBANK.NS",  # HDFC Bank
        "ICICIBANK.NS", # ICICI Bank
        "ITC.NS",       # ITC Limited
        "HINDUNILVR.NS", # Hindustan Unilever
        "LT.NS",        # Larsen & Toubro
        "KOTAKBANK.NS"  # Kotak Mahindra Bank
    ]
    
    # Index data
    indices = [
        "^NSEI",    # Nifty 50
        "^BSESN"    # BSE Sensex
    ]
    
    print("=" * 50)
    print("  DOWNLOADING SAMPLE MARKET DATA")
    print("=" * 50)
    
    # Download different timeframes
    print("\\n📊 Downloading daily data (1 year)...")
    download_stock_data(indian_stocks + indices, period="1y", interval="1d")
    
    print("\\n📊 Downloading hourly data (1 month)...")  
    download_stock_data(indian_stocks[:5], period="1mo", interval="1h")
    
    print("\\n✅ Data download completed!")
    print("\\nFiles saved to data/raw/ directory")

if __name__ == "__main__":
    main()
'''
    
    with open("data/download_market_data.py", "w") as f:
        f.write(downloader_content)
    print("✅ Created data downloader: data/download_market_data.py")
    
    # Create requirements.txt
    requirements_content = '''# Module 2 Requirements
# Technical Analysis and Data Processing

# Core Data Processing
pandas>=1.5.0
numpy>=1.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Financial Data
yfinance>=0.1.70
pandas-datareader>=0.10.0

# Scientific Computing
scipy>=1.7.0
scikit-learn>=1.0.0

# Development Tools
jupyter>=1.0.0
ipython>=7.0.0

# File Handling
openpyxl>=3.0.0
xlrd>=2.0.0

# Web and APIs
requests>=2.25.0
beautifulsoup4>=4.9.0

# Optional Technical Analysis
# ta-lib>=0.4.0  # Uncomment if you want to install TA-Lib
'''
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    print("✅ Created requirements file: requirements.txt")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    
    if failed_packages:
        print(f"⚠️  Some packages failed to install: {failed_packages}")
        print("   Please install them manually using: pip install <package_name>")
    
    print(f"""
✅ WHAT'S READY:
   • All core dependencies installed
   • Directory structure created  
   • Configuration file setup
   • Sample data downloader ready
   • Requirements file generated

🚀 NEXT STEPS:
   1. Download sample data: python data/download_market_data.py
   2. Start with theory notes: notes/01_pandas_fundamentals.md
   3. Try Assignment 1: assignments/assignment1.md
   4. Launch Jupyter: jupyter notebook (optional)

📚 LEARNING PATH:
   Module 2 → Technical Analysis → Data Processing → Strategy Development

Happy learning! 📈🎯
""")

if __name__ == "__main__":
    main()
