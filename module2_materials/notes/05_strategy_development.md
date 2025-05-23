# Module 2.5: Strategy Development and Backtesting

## Introduction

Strategy development is the process of creating systematic trading rules based on technical analysis, fundamental analysis, or quantitative models. This module covers building trading strategies, implementing entry/exit rules, and basic backtesting concepts.

## Learning Objectives

By the end of this lesson, you will be able to:
- Develop systematic trading strategies
- Implement entry and exit rules
- Create signal generation systems
- Build basic backtesting frameworks
- Calculate strategy performance metrics
- Understand risk management principles

## 1. Strategy Fundamentals

### 1.1 Types of Trading Strategies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate sample market data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='B')
initial_price = 100
returns = np.random.normal(0.0008, 0.015, len(dates))

prices = [initial_price]
for ret in returns[1:]:
    prices.append(prices[-1] * (1 + ret))

market_data = pd.DataFrame({
    'Date': dates,
    'Close': prices,
    'Volume': np.random.randint(1000000, 5000000, len(dates))
}, index=dates)

market_data['Returns'] = market_data['Close'].pct_change()

print("Sample market data:")
print(market_data.head())
```

### 1.2 Strategy Categories

```python
# Define different strategy types
strategy_types = {
    'Trend Following': {
        'Description': 'Follow the direction of price trends',
        'Examples': ['Moving Average Crossover', 'Breakout', 'Momentum'],
        'Market Condition': 'Trending markets'
    },
    'Mean Reversion': {
        'Description': 'Trade against extreme price movements',
        'Examples': ['Bollinger Band Reversals', 'RSI Oversold/Overbought'],
        'Market Condition': 'Sideways/Range-bound markets'
    },
    'Momentum': {
        'Description': 'Trade in direction of strong price movements',
        'Examples': ['Relative Strength', 'Price Acceleration'],
        'Market Condition': 'Strong trending markets'
    },
    'Arbitrage': {
        'Description': 'Exploit price differences between instruments',
        'Examples': ['Statistical Arbitrage', 'Pairs Trading'],
        'Market Condition': 'Any market condition'
    }
}

for strategy, details in strategy_types.items():
    print(f"\n{strategy}:")
    print(f"  Description: {details['Description']}")
    print(f"  Examples: {', '.join(details['Examples'])}")
    print(f"  Best for: {details['Market Condition']}")
```

## 2. Technical Indicators for Strategies

### 2.1 Moving Average Indicators

```python
def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

# Apply indicators to market data
market_data['SMA_20'] = calculate_sma(market_data['Close'], 20)
market_data['SMA_50'] = calculate_sma(market_data['Close'], 50)
market_data['EMA_12'] = calculate_ema(market_data['Close'], 12)
market_data['EMA_26'] = calculate_ema(market_data['Close'], 26)

macd, macd_signal, macd_hist = calculate_macd(market_data['Close'])
market_data['MACD'] = macd
market_data['MACD_Signal'] = macd_signal
market_data['MACD_Histogram'] = macd_hist

print("Technical indicators added:")
print(market_data[['Close', 'SMA_20', 'SMA_50', 'MACD']].tail())
```

### 2.2 Oscillators and Momentum Indicators

```python
def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band, sma

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

# Generate high and low prices for demonstrations
market_data['High'] = market_data['Close'] * (1 + np.random.uniform(0, 0.02, len(market_data)))
market_data['Low'] = market_data['Close'] * (1 - np.random.uniform(0, 0.02, len(market_data)))

# Calculate additional indicators
market_data['RSI'] = calculate_rsi(market_data['Close'])
bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(market_data['Close'])
market_data['BB_Upper'] = bb_upper
market_data['BB_Lower'] = bb_lower
market_data['BB_Middle'] = bb_middle

stoch_k, stoch_d = calculate_stochastic(
    market_data['High'], market_data['Low'], market_data['Close']
)
market_data['Stoch_K'] = stoch_k
market_data['Stoch_D'] = stoch_d

print("Additional indicators:")
print(market_data[['Close', 'RSI', 'BB_Upper', 'BB_Lower', 'Stoch_K']].tail())
```

## 3. Strategy Implementation

### 3.1 Simple Moving Average Crossover Strategy

```python
class MovingAverageCrossoverStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        self.positions = []
        
    def generate_signals(self, data):
        """Generate buy/sell signals based on MA crossover"""
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate moving averages
        signals['SMA_Short'] = data['Close'].rolling(window=self.short_window).mean()
        signals['SMA_Long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals['Signal'] = 0
        signals['Signal'][self.short_window:] = np.where(
            signals['SMA_Short'][self.short_window:] > signals['SMA_Long'][self.short_window:], 1, 0
        )
        
        # Calculate position changes
        signals['Position'] = signals['Signal'].diff()
        
        return signals

# Implement the strategy
ma_strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
signals = ma_strategy.generate_signals(market_data)

print("Moving Average Crossover Signals:")
print(signals[signals['Position'] != 0][['Price', 'SMA_Short', 'SMA_Long', 'Position']].head(10))

# Plot the strategy
plt.figure(figsize=(12, 8))
plt.plot(signals.index, signals['Price'], label='Price', linewidth=1)
plt.plot(signals.index, signals['SMA_Short'], label='SMA 20', linewidth=2)
plt.plot(signals.index, signals['SMA_Long'], label='SMA 50', linewidth=2)

# Mark buy/sell signals
buy_signals = signals[signals['Position'] == 1]
sell_signals = signals[signals['Position'] == -1]

plt.scatter(buy_signals.index, buy_signals['Price'], 
           color='green', marker='^', s=100, label='Buy')
plt.scatter(sell_signals.index, sell_signals['Price'], 
           color='red', marker='v', s=100, label='Sell')

plt.title('Moving Average Crossover Strategy')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3.2 RSI Mean Reversion Strategy

```python
class RSIMeanReversionStrategy:
    def __init__(self, rsi_window=14, oversold=30, overbought=70):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
        
    def generate_signals(self, data):
        """Generate signals based on RSI levels"""
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        signals['RSI'] = calculate_rsi(data['Close'], self.rsi_window)
        
        # Generate signals
        signals['Signal'] = 0
        
        # Buy when RSI is oversold (below 30)
        signals.loc[signals['RSI'] < self.oversold, 'Signal'] = 1
        
        # Sell when RSI is overbought (above 70)
        signals.loc[signals['RSI'] > self.overbought, 'Signal'] = -1
        
        # Calculate position changes
        signals['Position'] = signals['Signal'].diff()
        
        return signals

# Implement RSI strategy
rsi_strategy = RSIMeanReversionStrategy(oversold=30, overbought=70)
rsi_signals = rsi_strategy.generate_signals(market_data)

print("RSI Mean Reversion Signals:")
print(rsi_signals[rsi_signals['Position'] != 0][['Price', 'RSI', 'Position']].head(10))
```

### 3.3 Bollinger Band Strategy

```python
class BollingerBandStrategy:
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, data):
        """Generate signals based on Bollinger Band touches"""
        signals = pd.DataFrame(index=data.index)
        signals['Price'] = data['Close']
        
        # Calculate Bollinger Bands
        bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(
            data['Close'], self.window, self.num_std
        )
        signals['BB_Upper'] = bb_upper
        signals['BB_Lower'] = bb_lower
        signals['BB_Middle'] = bb_middle
        
        # Generate signals
        signals['Signal'] = 0
        
        # Buy when price touches lower band (oversold)
        signals.loc[signals['Price'] <= signals['BB_Lower'], 'Signal'] = 1
        
        # Sell when price touches upper band (overbought)
        signals.loc[signals['Price'] >= signals['BB_Upper'], 'Signal'] = -1
        
        # Exit when price returns to middle band
        signals.loc[
            (signals['Signal'].shift(1) == 1) & (signals['Price'] >= signals['BB_Middle']), 
            'Signal'
        ] = 0
        signals.loc[
            (signals['Signal'].shift(1) == -1) & (signals['Price'] <= signals['BB_Middle']), 
            'Signal'
        ] = 0
        
        # Calculate position changes
        signals['Position'] = signals['Signal'].diff()
        
        return signals

# Implement Bollinger Band strategy
bb_strategy = BollingerBandStrategy()
bb_signals = bb_strategy.generate_signals(market_data)

print("Bollinger Band Strategy Signals:")
print(bb_signals[bb_signals['Position'] != 0][['Price', 'BB_Upper', 'BB_Lower', 'Position']].head(10))
```

## 4. Basic Backtesting Framework

### 4.1 Simple Backtesting Engine

```python
class SimpleBacktester:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def backtest_strategy(self, signals, data):
        """Backtest a trading strategy"""
        # Initialize portfolio
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['Price'] = data['Close']
        portfolio['Signal'] = signals['Signal']
        portfolio['Position'] = signals['Position'].fillna(0)
        
        # Calculate positions and holdings
        portfolio['Holdings'] = portfolio['Position'].cumsum()
        portfolio['Cash'] = self.initial_capital
        portfolio['Total'] = self.initial_capital
        
        # Track cash and total portfolio value
        cash = self.initial_capital
        holdings = 0
        
        for i in range(len(portfolio)):
            if portfolio['Position'].iloc[i] != 0:  # Trade occurred
                trade_value = abs(portfolio['Position'].iloc[i]) * portfolio['Price'].iloc[i]
                transaction_costs = trade_value * self.transaction_cost
                
                if portfolio['Position'].iloc[i] > 0:  # Buy
                    cash -= trade_value + transaction_costs
                    holdings += portfolio['Position'].iloc[i]
                else:  # Sell
                    cash += trade_value - transaction_costs
                    holdings += portfolio['Position'].iloc[i]
            
            portfolio['Cash'].iloc[i] = cash
            portfolio['Holdings'].iloc[i] = holdings
            portfolio['Total'].iloc[i] = cash + (holdings * portfolio['Price'].iloc[i])
        
        # Calculate returns
        portfolio['Returns'] = portfolio['Total'].pct_change()
        portfolio['Cumulative_Returns'] = (portfolio['Total'] / self.initial_capital) - 1
        
        return portfolio

# Backtest the moving average strategy
backtester = SimpleBacktester(initial_capital=100000, transaction_cost=0.001)
ma_portfolio = backtester.backtest_strategy(signals, market_data)

print("Backtest Results (Moving Average Strategy):")
print(ma_portfolio[['Price', 'Position', 'Holdings', 'Total', 'Cumulative_Returns']].tail())
```

### 4.2 Performance Metrics

```python
def calculate_performance_metrics(portfolio):
    """Calculate comprehensive performance metrics"""
    returns = portfolio['Returns'].dropna()
    
    # Basic metrics
    total_return = portfolio['Cumulative_Returns'].iloc[-1]
    annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate analysis
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    win_rate = len(winning_trades) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
    
    # Average win/loss
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else np.inf
    
    metrics = {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Profit Factor': profit_factor,
        'Total Trades': len(returns[returns != 0])
    }
    
    return metrics

# Calculate performance for moving average strategy
ma_metrics = calculate_performance_metrics(ma_portfolio)

print("Moving Average Strategy Performance:")
print("-" * 40)
for metric, value in ma_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")
```

## 5. Risk Management

### 5.1 Position Sizing

```python
def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price):
    """Calculate position size based on risk management rules"""
    risk_per_share = abs(entry_price - stop_loss_price)
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / risk_per_share
    return int(position_size)

def apply_risk_management(signals, data, capital=100000, risk_per_trade=0.02, stop_loss_pct=0.05):
    """Apply risk management rules to trading signals"""
    risk_managed_signals = signals.copy()
    risk_managed_signals['Position_Size'] = 0
    risk_managed_signals['Stop_Loss'] = 0
    risk_managed_signals['Take_Profit'] = 0
    
    for i in range(len(signals)):
        if signals['Position'].iloc[i] == 1:  # Buy signal
            entry_price = data['Close'].iloc[i]
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + stop_loss_pct * 2)  # 2:1 risk-reward
            
            position_size = calculate_position_size(
                capital, risk_per_trade, entry_price, stop_loss_price
            )
            
            risk_managed_signals['Position_Size'].iloc[i] = position_size
            risk_managed_signals['Stop_Loss'].iloc[i] = stop_loss_price
            risk_managed_signals['Take_Profit'].iloc[i] = take_profit_price
    
    return risk_managed_signals

# Apply risk management to moving average strategy
risk_managed_ma = apply_risk_management(signals, market_data)

print("Risk-Managed Trading Signals:")
print(risk_managed_ma[risk_managed_ma['Position'] == 1][
    ['Price', 'Position_Size', 'Stop_Loss', 'Take_Profit']
].head())
```

### 5.2 Portfolio-Level Risk Controls

```python
def apply_portfolio_risk_controls(signals, max_positions=3, max_exposure=0.5):
    """Apply portfolio-level risk controls"""
    controlled_signals = signals.copy()
    controlled_signals['Approved'] = False
    
    current_positions = 0
    current_exposure = 0
    
    for i in range(len(signals)):
        if signals['Position'].iloc[i] == 1:  # New buy signal
            position_value = signals['Position_Size'].iloc[i] * signals['Price'].iloc[i]
            new_exposure = current_exposure + (position_value / 100000)  # Assuming 100k capital
            
            # Check risk limits
            if current_positions < max_positions and new_exposure <= max_exposure:
                controlled_signals['Approved'].iloc[i] = True
                current_positions += 1
                current_exposure = new_exposure
            
        elif signals['Position'].iloc[i] == -1:  # Sell signal
            controlled_signals['Approved'].iloc[i] = True
            current_positions = max(0, current_positions - 1)
            # Recalculate exposure after position close
    
    return controlled_signals

# Apply portfolio risk controls
portfolio_controlled = apply_portfolio_risk_controls(risk_managed_ma)

print("Portfolio Risk-Controlled Signals:")
approved_trades = portfolio_controlled[
    (portfolio_controlled['Position'] != 0) & (portfolio_controlled['Approved'] == True)
]
print(approved_trades[['Price', 'Position', 'Position_Size', 'Approved']].head())
```

## 6. Strategy Comparison and Selection

### 6.1 Multiple Strategy Backtest

```python
def compare_strategies():
    """Compare performance of multiple strategies"""
    strategies = {
        'Moving Average': ma_portfolio,
        'RSI Mean Reversion': backtester.backtest_strategy(rsi_signals, market_data),
        'Bollinger Bands': backtester.backtest_strategy(bb_signals, market_data)
    }
    
    comparison = {}
    
    for name, portfolio in strategies.items():
        metrics = calculate_performance_metrics(portfolio)
        comparison[name] = {
            'Total Return': metrics['Total Return'],
            'Sharpe Ratio': metrics['Sharpe Ratio'],
            'Max Drawdown': metrics['Max Drawdown'],
            'Win Rate': metrics['Win Rate']
        }
    
    return pd.DataFrame(comparison).T

# Compare strategies
strategy_comparison = compare_strategies()
print("Strategy Comparison:")
print(strategy_comparison.round(4))

# Plot strategy performance
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(ma_portfolio.index, ma_portfolio['Cumulative_Returns'], 
         label='Moving Average', linewidth=2)
plt.plot(backtester.backtest_strategy(rsi_signals, market_data).index, 
         backtester.backtest_strategy(rsi_signals, market_data)['Cumulative_Returns'], 
         label='RSI Mean Reversion', linewidth=2)
plt.plot(backtester.backtest_strategy(bb_signals, market_data).index, 
         backtester.backtest_strategy(bb_signals, market_data)['Cumulative_Returns'], 
         label='Bollinger Bands', linewidth=2)
plt.title('Strategy Performance Comparison')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(market_data.index, market_data['Close'], 
         label='Buy & Hold', linewidth=2, color='black')
plt.title('Underlying Asset Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 7. Advanced Strategy Concepts

### 7.1 Multi-Timeframe Analysis

```python
def multi_timeframe_strategy(data):
    """Example of multi-timeframe analysis"""
    # Daily signals (primary timeframe)
    daily_data = data.copy()
    daily_data['SMA_20'] = daily_data['Close'].rolling(20).mean()
    daily_data['SMA_50'] = daily_data['Close'].rolling(50).mean()
    
    # Weekly signals (higher timeframe for trend confirmation)
    weekly_data = data.resample('W').agg({
        'Close': 'last',
        'Volume': 'sum'
    })
    weekly_data['SMA_10'] = weekly_data['Close'].rolling(10).mean()
    weekly_data['Weekly_Trend'] = np.where(
        weekly_data['Close'] > weekly_data['SMA_10'], 1, -1
    )
    
    # Combine timeframes
    daily_data = daily_data.join(weekly_data[['Weekly_Trend']], how='left')
    daily_data['Weekly_Trend'] = daily_data['Weekly_Trend'].fillna(method='ffill')
    
    # Generate signals only when daily and weekly align
    daily_data['Signal'] = 0
    daily_data.loc[
        (daily_data['SMA_20'] > daily_data['SMA_50']) & 
        (daily_data['Weekly_Trend'] == 1), 'Signal'
    ] = 1
    
    return daily_data

# Implement multi-timeframe strategy
mtf_data = multi_timeframe_strategy(market_data)
print("Multi-timeframe Strategy:")
print(mtf_data[['Close', 'SMA_20', 'SMA_50', 'Weekly_Trend', 'Signal']].tail())
```

### 7.2 Dynamic Strategy Parameters

```python
def adaptive_moving_average_strategy(data, lookback=50):
    """Example of adaptive strategy parameters based on market conditions"""
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    
    # Calculate market volatility
    signals['Volatility'] = data['Returns'].rolling(lookback).std()
    signals['Vol_Percentile'] = signals['Volatility'].rolling(lookback).rank(pct=True)
    
    # Adapt MA periods based on volatility
    signals['Short_Period'] = np.where(
        signals['Vol_Percentile'] > 0.7, 10, 20  # Shorter in high vol
    )
    signals['Long_Period'] = np.where(
        signals['Vol_Percentile'] > 0.7, 30, 50  # Shorter in high vol
    )
    
    # Calculate adaptive moving averages (simplified)
    signals['SMA_Short'] = data['Close'].rolling(20).mean()  # Simplified for demo
    signals['SMA_Long'] = data['Close'].rolling(50).mean()   # Simplified for demo
    
    # Generate signals
    signals['Signal'] = np.where(
        signals['SMA_Short'] > signals['SMA_Long'], 1, 0
    )
    
    return signals

# Example of adaptive strategy
adaptive_signals = adaptive_moving_average_strategy(market_data)
print("Adaptive Strategy Example:")
print(adaptive_signals[['Price', 'Volatility', 'Vol_Percentile', 'Short_Period', 'Signal']].tail())
```

## Practice Exercises

1. **Custom Strategy**: Develop a strategy combining RSI and MACD signals
2. **Parameter Optimization**: Test different MA periods and find optimal parameters
3. **Risk-Adjusted Returns**: Implement a strategy with dynamic position sizing
4. **Market Regime Detection**: Create different strategies for different market conditions
5. **Walk-Forward Analysis**: Implement out-of-sample testing methodology

## Key Takeaways

- Strategy development requires clear entry and exit rules
- Backtesting helps evaluate strategy performance before live trading
- Risk management is crucial for long-term success
- Multiple strategies can be combined for diversification
- Market conditions affect strategy performance
- Parameter optimization should be done carefully to avoid overfitting

## Next Steps

In the next module, we'll learn about performance analysis and how to evaluate and improve trading strategies systematically.
