#!/usr/bin/env python3
"""
Assignment 1 Solution: Pandas Fundamentals for Financial Data
Module 2: Technical Analysis and Data Processing

This solution demonstrates comprehensive pandas operations for financial data analysis,
including data cleaning, manipulation, and basic financial calculations.

Author: Financial Analytics Course
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set pandas display options for better output
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

def load_and_clean_data(file_path):
    """
    Load stock data from CSV and perform initial cleaning.
    
    Args:
        file_path (str): Path to the CSV file containing stock data
        
    Returns:
        pd.DataFrame: Cleaned stock data with proper datetime index
    """
    try:
        # Load data with proper date parsing
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        
        print(f"✓ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Basic data info
        print(f"✓ Date range: {data.index.min()} to {data.index.max()}")
        print(f"✓ Columns: {list(data.columns)}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print(f"⚠️  Missing values found:\n{missing_values[missing_values > 0]}")
            
            # Forward fill missing values for financial data
            data = data.fillna(method='ffill')
            print("✓ Missing values handled using forward fill")
        else:
            print("✓ No missing values found")
        
        # Ensure proper data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Basic data validation
        if 'Close' in data.columns:
            if (data['Close'] <= 0).any():
                print("⚠️  Warning: Found non-positive closing prices")
        
        # Sort by date to ensure proper chronological order
        data = data.sort_index()
        
        return data
        
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def calculate_basic_metrics(data):
    """
    Calculate basic financial metrics from OHLCV data.
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        
    Returns:
        pd.DataFrame: Data with additional calculated metrics
    """
    if data is None or data.empty:
        print("❌ Error: No data provided for metric calculation")
        return None
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Daily Returns
    if 'Close' in df.columns:
        df['Daily_Return'] = df['Close'].pct_change()
        print("✓ Daily returns calculated")
    
    # Price Range (High - Low)
    if 'High' in df.columns and 'Low' in df.columns:
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = (df['Price_Range'] / df['Close']) * 100
        print("✓ Price range metrics calculated")
    
    # True Range (for volatility calculation)
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df['Prev_Close'] = df['Close'].shift(1)
        df['True_Range'] = np.maximum.reduce([
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Prev_Close']),
            np.abs(df['Low'] - df['Prev_Close'])
        ])
        print("✓ True Range calculated")
    
    # Price Position within the day's range
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        df['Price_Position'] = ((df['Close'] - df['Low']) / 
                               (df['High'] - df['Low']) * 100)
        print("✓ Price position calculated")
    
    # Volume-based metrics
    if 'Volume' in df.columns:
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        print("✓ Volume metrics calculated")
    
    # Price gaps
    if 'Open' in df.columns and 'Close' in df.columns:
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
        print("✓ Gap analysis completed")
    
    return df

def calculate_moving_averages(data, windows=[5, 10, 20, 50, 200]):
    """
    Calculate multiple moving averages for the closing price.
    
    Args:
        data (pd.DataFrame): Stock data with Close column
        windows (list): List of window sizes for moving averages
        
    Returns:
        pd.DataFrame: Data with moving average columns added
    """
    if data is None or 'Close' not in data.columns:
        print("❌ Error: No Close price data available")
        return data
    
    df = data.copy()
    
    # Calculate Simple Moving Averages
    for window in windows:
        ma_col = f'SMA_{window}'
        df[ma_col] = df['Close'].rolling(window=window).mean()
        
        # Calculate distance from moving average
        distance_col = f'Distance_SMA_{window}'
        df[distance_col] = ((df['Close'] - df[ma_col]) / df[ma_col]) * 100
    
    print(f"✓ Moving averages calculated for windows: {windows}")
    
    # Calculate Exponential Moving Averages for common periods
    ema_windows = [12, 26, 50]
    for window in ema_windows:
        ema_col = f'EMA_{window}'
        df[ema_col] = df['Close'].ewm(span=window).mean()
    
    print(f"✓ Exponential moving averages calculated for windows: {ema_windows}")
    
    return df

def analyze_volatility(data, window=20):
    """
    Calculate various volatility measures.
    
    Args:
        data (pd.DataFrame): Stock data with returns
        window (int): Window size for rolling calculations
        
    Returns:
        pd.DataFrame: Data with volatility metrics added
    """
    if data is None or 'Daily_Return' not in data.columns:
        print("❌ Error: Daily returns not available for volatility analysis")
        return data
    
    df = data.copy()
    
    # Rolling volatility (standard deviation of returns)
    df['Volatility_Rolling'] = df['Daily_Return'].rolling(window=window).std()
    
    # Annualized volatility (assuming 252 trading days)
    df['Volatility_Annualized'] = df['Volatility_Rolling'] * np.sqrt(252)
    
    # Parkinson volatility (using High-Low range)
    if all(col in df.columns for col in ['High', 'Low']):
        df['Parkinson_Vol'] = np.sqrt((1/(4*np.log(2))) * 
                                     np.log(df['High']/df['Low'])**2)
        df['Parkinson_Vol_Rolling'] = df['Parkinson_Vol'].rolling(window=window).mean()
    
    # Average True Range (ATR) based volatility
    if 'True_Range' in df.columns:
        df['ATR'] = df['True_Range'].rolling(window=window).mean()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    
    print(f"✓ Volatility metrics calculated with {window}-day window")
    
    return df

def performance_analysis(data):
    """
    Perform comprehensive performance analysis of the stock.
    
    Args:
        data (pd.DataFrame): Stock data with calculated metrics
        
    Returns:
        dict: Dictionary containing performance statistics
    """
    if data is None or 'Daily_Return' not in data.columns:
        print("❌ Error: Cannot perform analysis without return data")
        return {}
    
    # Remove any infinite or NaN values for calculations
    returns = data['Daily_Return'].dropna()
    
    if returns.empty:
        print("❌ Error: No valid return data available")
        return {}
    
    # Basic statistics
    stats = {
        'total_return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100,
        'annualized_return': returns.mean() * 252 * 100,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'max_daily_gain': returns.max() * 100,
        'max_daily_loss': returns.min() * 100,
        'positive_days': (returns > 0).sum(),
        'negative_days': (returns < 0).sum(),
        'win_rate': (returns > 0).mean() * 100
    }
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = ((cumulative_returns - peak) / peak) * 100
    stats['max_drawdown'] = drawdown.min()
    
    # Risk metrics
    stats['var_95'] = np.percentile(returns, 5) * 100  # 5% VaR
    stats['skewness'] = returns.skew()
    stats['kurtosis'] = returns.kurtosis()
    
    # Trading activity metrics
    if 'Volume' in data.columns:
        stats['avg_daily_volume'] = data['Volume'].mean()
        stats['total_volume'] = data['Volume'].sum()
    
    return stats

def create_summary_report(data, stats):
    """
    Create a comprehensive summary report of the analysis.
    
    Args:
        data (pd.DataFrame): Processed stock data
        stats (dict): Performance statistics
        
    Returns:
        str: Formatted report string
    """
    if not stats:
        return "❌ Cannot generate report: No statistics available"
    
    report = f"""
{'='*60}
STOCK ANALYSIS SUMMARY REPORT
{'='*60}

DATA OVERVIEW:
• Analysis Period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}
• Total Trading Days: {len(data)}
• Data Quality: {len(data.dropna())} complete records

PRICE PERFORMANCE:
• Starting Price: ₹{data['Close'].iloc[0]:.2f}
• Ending Price: ₹{data['Close'].iloc[-1]:.2f}
• Total Return: {stats['total_return']:.2f}%
• Annualized Return: {stats['annualized_return']:.2f}%

RISK METRICS:
• Volatility (Annual): {stats['volatility']:.2f}%
• Sharpe Ratio: {stats['sharpe_ratio']:.3f}
• Maximum Drawdown: {stats['max_drawdown']:.2f}%
• Value at Risk (95%): {stats['var_95']:.2f}%

DAILY PERFORMANCE:
• Best Day: +{stats['max_daily_gain']:.2f}%
• Worst Day: {stats['max_daily_loss']:.2f}%
• Positive Days: {stats['positive_days']} ({stats['win_rate']:.1f}%)
• Negative Days: {stats['negative_days']}

STATISTICAL PROPERTIES:
• Skewness: {stats['skewness']:.3f}
• Kurtosis: {stats['kurtosis']:.3f}
• Distribution: {'Normal' if -0.5 < stats['skewness'] < 0.5 else 'Skewed'}

CURRENT TECHNICAL LEVELS:
"""
    
    # Add current technical levels if moving averages exist
    if 'SMA_20' in data.columns:
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else None
        
        report += f"• Current Price: ₹{current_price:.2f}\n"
        report += f"• 20-Day SMA: ₹{sma_20:.2f} ({((current_price/sma_20-1)*100):+.1f}%)\n"
        
        if sma_50:
            report += f"• 50-Day SMA: ₹{sma_50:.2f} ({((current_price/sma_50-1)*100):+.1f}%)\n"
    
    if 'Volume' in data.columns:
        report += f"\nVOLUME ANALYSIS:\n"
        report += f"• Average Daily Volume: {stats['avg_daily_volume']:,.0f}\n"
        report += f"• Latest Volume: {data['Volume'].iloc[-1]:,.0f}\n"
    
    report += f"\n{'='*60}\n"
    
    return report

def create_visualizations(data, symbol="Stock"):
    """
    Create comprehensive visualizations of the stock analysis.
    
    Args:
        data (pd.DataFrame): Processed stock data
        symbol (str): Stock symbol for chart titles
    """
    if data is None or data.empty:
        print("❌ Error: No data available for visualization")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{symbol} - Comprehensive Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Price Chart with Moving Averages
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='blue')
    
    if 'SMA_20' in data.columns:
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7, color='orange')
    if 'SMA_50' in data.columns:
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='red')
    
    ax1.set_title('Price Chart with Moving Averages')
    ax1.set_ylabel('Price (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Daily Returns Distribution
    ax2 = axes[0, 1]
    if 'Daily_Return' in data.columns:
        returns = data['Daily_Return'].dropna() * 100
        ax2.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.2f}%')
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Volatility Analysis
    ax3 = axes[1, 0]
    if 'Volatility_Annualized' in data.columns:
        vol_data = data['Volatility_Annualized'].dropna() * 100
        ax3.plot(vol_data.index, vol_data, color='purple', linewidth=1.5)
        ax3.set_title('Rolling Volatility (Annualized)')
        ax3.set_ylabel('Volatility (%)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Volume Analysis
    ax4 = axes[1, 1]
    if 'Volume' in data.columns:
        # Plot volume bars
        ax4.bar(data.index, data['Volume'], alpha=0.6, color='gray', width=0.8)
        
        # Add volume moving average if available
        if 'Volume_MA_20' in data.columns:
            ax4.plot(data.index, data['Volume_MA_20'], color='red', 
                    linewidth=2, label='20-Day MA')
            ax4.legend()
        
        ax4.set_title('Trading Volume Analysis')
        ax4.set_ylabel('Volume')
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualizations created successfully")

def main():
    """
    Main function demonstrating comprehensive pandas analysis workflow.
    """
    print("🚀 Starting Comprehensive Financial Data Analysis")
    print("="*60)
    
    # Sample data creation (in real scenario, this would be loaded from file)
    print("📊 Creating sample data for demonstration...")
    
    # Generate sample stock data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Simulate realistic stock price movement
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
    prices = [100]  # Starting price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create comprehensive OHLCV data
    sample_data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.lognormal(14, 0.5, len(dates)).astype(int)
    }, index=dates)
    
    # Ensure OHLC relationships are correct
    sample_data['High'] = np.maximum.reduce([
        sample_data['Open'], sample_data['High'], 
        sample_data['Close']
    ])
    sample_data['Low'] = np.minimum.reduce([
        sample_data['Open'], sample_data['Low'], 
        sample_data['Close']
    ])
    
    print(f"✓ Sample data created: {len(sample_data)} trading days")
    
    # Step 1: Load and clean data
    print("\n📋 Step 1: Data Loading and Cleaning")
    print("-" * 40)
    data = sample_data.copy()
    print(f"✓ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"✓ Date range: {data.index.min()} to {data.index.max()}")
    print(f"✓ No missing values found")
    
    # Step 2: Calculate basic metrics
    print("\n📈 Step 2: Basic Financial Metrics")
    print("-" * 40)
    data = calculate_basic_metrics(data)
    
    # Step 3: Calculate moving averages
    print("\n📊 Step 3: Moving Average Analysis")
    print("-" * 40)
    data = calculate_moving_averages(data)
    
    # Step 4: Volatility analysis
    print("\n📉 Step 4: Volatility Analysis")
    print("-" * 40)
    data = analyze_volatility(data)
    
    # Step 5: Performance analysis
    print("\n📋 Step 5: Performance Analysis")
    print("-" * 40)
    stats = performance_analysis(data)
    
    # Step 6: Generate summary report
    print("\n📄 Step 6: Summary Report Generation")
    print("-" * 40)
    report = create_summary_report(data, stats)
    print(report)
    
    # Step 7: Create visualizations
    print("\n📊 Step 7: Data Visualization")
    print("-" * 40)
    create_visualizations(data, "SAMPLE_STOCK")
    
    # Additional pandas operations demonstration
    print("\n🔧 Advanced Pandas Operations Demonstration")
    print("-" * 50)
    
    # Resampling to different timeframes
    monthly_data = data.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'Daily_Return': 'sum'
    })
    print(f"✓ Monthly resampling: {len(monthly_data)} months")
    
    # Group by year for annual analysis
    yearly_stats = data.groupby(data.index.year).agg({
        'Daily_Return': ['mean', 'std', 'min', 'max'],
        'Volume': 'mean',
        'Close': ['first', 'last']
    })
    yearly_stats['Annual_Return'] = ((yearly_stats[('Close', 'last')] / 
                                    yearly_stats[('Close', 'first')] - 1) * 100)
    print(f"✓ Yearly analysis completed for {len(yearly_stats)} years")
    
    # Rolling correlation analysis
    if len(data) > 50:
        # Create a benchmark (simulated market index)
        benchmark_returns = np.random.normal(0.0006, 0.015, len(data))
        data['Benchmark_Return'] = benchmark_returns
        data['Rolling_Correlation'] = data['Daily_Return'].rolling(30).corr(data['Benchmark_Return'])
        print("✓ Rolling correlation with benchmark calculated")
    
    # Quantile analysis
    price_quantiles = data['Close'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"✓ Price quantiles calculated:")
    for q, value in price_quantiles.items():
        print(f"   {int(q*100)}th percentile: ₹{value:.2f}")
    
    print(f"\n🎉 Analysis Complete! Processed {len(data)} trading days")
    print(f"📊 Generated {len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} derived metrics")
    print("\n" + "="*60)

# Additional utility functions for assignment completion

def export_results(data, filename='financial_analysis_results.csv'):
    """Export analysis results to CSV file."""
    try:
        # Select most important columns for export
        export_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                         'Daily_Return', 'SMA_20', 'SMA_50', 
                         'Volatility_Annualized', 'ATR']
        
        export_data = data[[col for col in export_columns if col in data.columns]]
        export_data.to_csv(filename)
        print(f"✓ Results exported to {filename}")
        return True
    except Exception as e:
        print(f"❌ Export failed: {str(e)}")
        return False

def load_multiple_stocks(file_paths, symbols):
    """Load and analyze multiple stocks simultaneously."""
    results = {}
    
    for file_path, symbol in zip(file_paths, symbols):
        print(f"\n📈 Analyzing {symbol}...")
        data = load_and_clean_data(file_path)
        
        if data is not None:
            data = calculate_basic_metrics(data)
            data = calculate_moving_averages(data)
            data = analyze_volatility(data)
            stats = performance_analysis(data)
            
            results[symbol] = {
                'data': data,
                'stats': stats
            }
            print(f"✅ {symbol} analysis complete")
        else:
            print(f"❌ Failed to analyze {symbol}")
    
    return results

if __name__ == "__main__":
    main()
