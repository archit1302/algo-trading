# Module 2.3: Data Visualization for Financial Analysis

## Introduction

Data visualization is crucial for financial analysis as it helps identify patterns, trends, and anomalies in market data. This module covers creating professional charts and graphs for financial data using matplotlib and plotly.

## Learning Objectives

By the end of this lesson, you will be able to:
- Create various types of financial charts (line charts, candlestick charts, volume charts)
- Customize chart appearance for professional presentations
- Build interactive charts using plotly
- Create multi-panel charts for comprehensive analysis
- Export charts in various formats

## 1. Basic Chart Types

### 1.1 Line Charts for Price Data

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample stock price data
dates = pd.date_range('2023-01-01', periods=252, freq='D')
prices = 100 + np.cumsum(np.random.randn(252) * 0.02)
stock_data = pd.DataFrame({'Date': dates, 'Close': prices})

# Basic line chart
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'], stock_data['Close'], linewidth=2, color='blue')
plt.title('Stock Price Movement', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 1.2 Multiple Line Charts for Comparison

```python
# Compare multiple stocks
stocks = ['AAPL', 'GOOGL', 'MSFT']
colors = ['blue', 'red', 'green']

plt.figure(figsize=(12, 6))
for i, stock in enumerate(stocks):
    # Generate sample data for each stock
    prices = 100 + np.cumsum(np.random.randn(252) * 0.02)
    plt.plot(dates, prices, label=stock, linewidth=2, color=colors[i])

plt.title('Stock Price Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 2. Candlestick Charts

### 2.1 Basic Candlestick Chart

```python
import mplfinance as mpf

# Generate OHLC data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=50, freq='D')
open_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
high_prices = open_prices + np.random.uniform(0, 2, 50)
low_prices = open_prices - np.random.uniform(0, 2, 50)
close_prices = open_prices + np.random.randn(50) * 0.8

ohlc_data = pd.DataFrame({
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': np.random.randint(1000000, 5000000, 50)
}, index=dates)

# Create candlestick chart
mpf.plot(ohlc_data, type='candle', style='charles',
         title='AAPL Stock Price - Candlestick Chart',
         ylabel='Price ($)',
         volume=True,
         figsize=(12, 8))
```

### 2.2 Advanced Candlestick with Technical Indicators

```python
# Add moving averages to candlestick chart
ohlc_data['SMA_20'] = ohlc_data['Close'].rolling(window=20).mean()
ohlc_data['SMA_50'] = ohlc_data['Close'].rolling(window=50).mean()

# Create addplot for moving averages
add_plots = [
    mpf.make_addplot(ohlc_data['SMA_20'], color='blue', width=1),
    mpf.make_addplot(ohlc_data['SMA_50'], color='red', width=1)
]

mpf.plot(ohlc_data, type='candle', style='charles',
         addplot=add_plots,
         title='Stock Price with Moving Averages',
         ylabel='Price ($)',
         volume=True,
         figsize=(12, 8))
```

## 3. Volume Analysis Charts

### 3.1 Volume Bar Chart

```python
plt.figure(figsize=(12, 8))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                               gridspec_kw={'height_ratios': [3, 1]})

# Price chart
ax1.plot(dates, ohlc_data['Close'], linewidth=2, color='blue')
ax1.set_title('Stock Price and Volume Analysis', fontsize=16, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.grid(True, alpha=0.3)

# Volume chart
colors = ['red' if close < open else 'green' 
          for open, close in zip(ohlc_data['Open'], ohlc_data['Close'])]
ax2.bar(dates, ohlc_data['Volume'], color=colors, alpha=0.7)
ax2.set_ylabel('Volume', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 4. Interactive Charts with Plotly

### 4.1 Interactive Line Chart

```python
import plotly.graph_objects as go
import plotly.express as px

# Create interactive line chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_data['Date'],
    y=stock_data['Close'],
    mode='lines',
    name='Stock Price',
    line=dict(width=2, color='blue'),
    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title='Interactive Stock Price Chart',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    hovermode='x unified',
    showlegend=True,
    width=800,
    height=500
)

fig.show()
```

### 4.2 Interactive Candlestick Chart

```python
fig = go.Figure(data=go.Candlestick(
    x=ohlc_data.index,
    open=ohlc_data['Open'],
    high=ohlc_data['High'],
    low=ohlc_data['Low'],
    close=ohlc_data['Close'],
    name='OHLC'
))

# Add moving averages
fig.add_trace(go.Scatter(
    x=ohlc_data.index,
    y=ohlc_data['SMA_20'],
    mode='lines',
    name='SMA 20',
    line=dict(width=1, color='blue')
))

fig.add_trace(go.Scatter(
    x=ohlc_data.index,
    y=ohlc_data['SMA_50'],
    mode='lines',
    name='SMA 50',
    line=dict(width=1, color='red')
))

fig.update_layout(
    title='Interactive Candlestick Chart with Moving Averages',
    yaxis_title='Price ($)',
    xaxis_rangeslider_visible=False,
    showlegend=True,
    width=1000,
    height=600
)

fig.show()
```

## 5. Multi-Panel Charts

### 5.1 Price, Volume, and RSI Chart

```python
from plotly.subplots import make_subplots

# Calculate RSI (simplified version)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

ohlc_data['RSI'] = calculate_rsi(ohlc_data['Close'])

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Price', 'Volume', 'RSI'),
    row_heights=[0.5, 0.25, 0.25]
)

# Price chart (candlestick)
fig.add_trace(go.Candlestick(
    x=ohlc_data.index,
    open=ohlc_data['Open'],
    high=ohlc_data['High'],
    low=ohlc_data['Low'],
    close=ohlc_data['Close'],
    name='OHLC'
), row=1, col=1)

# Volume chart
colors = ['red' if row['Close'] < row['Open'] else 'green' 
          for _, row in ohlc_data.iterrows()]
fig.add_trace(go.Bar(
    x=ohlc_data.index,
    y=ohlc_data['Volume'],
    name='Volume',
    marker_color=colors
), row=2, col=1)

# RSI chart
fig.add_trace(go.Scatter(
    x=ohlc_data.index,
    y=ohlc_data['RSI'],
    mode='lines',
    name='RSI',
    line=dict(width=2, color='purple')
), row=3, col=1)

# Add RSI reference lines
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

fig.update_layout(
    title='Comprehensive Stock Analysis',
    xaxis_rangeslider_visible=False,
    showlegend=False,
    height=800
)

fig.show()
```

## 6. Chart Customization

### 6.1 Professional Styling

```python
# Set style parameters
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Financial Dashboard', fontsize=20, fontweight='bold')

# Plot 1: Price trend
axes[0, 0].plot(stock_data['Date'], stock_data['Close'], 
                color=colors[0], linewidth=2)
axes[0, 0].set_title('Price Trend', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Returns distribution
returns = stock_data['Close'].pct_change().dropna()
axes[0, 1].hist(returns, bins=30, color=colors[1], alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Returns Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Daily Returns')
axes[0, 1].set_ylabel('Frequency')

# Plot 3: Moving averages
ma_short = stock_data['Close'].rolling(window=10).mean()
ma_long = stock_data['Close'].rolling(window=30).mean()
axes[1, 0].plot(stock_data['Date'], stock_data['Close'], 
                color=colors[2], linewidth=1, label='Price')
axes[1, 0].plot(stock_data['Date'], ma_short, 
                color=colors[0], linewidth=2, label='MA 10')
axes[1, 0].plot(stock_data['Date'], ma_long, 
                color=colors[3], linewidth=2, label='MA 30')
axes[1, 0].set_title('Moving Averages', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Volatility
volatility = returns.rolling(window=20).std() * np.sqrt(252)
axes[1, 1].plot(stock_data['Date'][20:], volatility[20:], 
                color=colors[3], linewidth=2)
axes[1, 1].set_title('Annualized Volatility', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Volatility')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 7. Chart Export and Saving

### 7.1 Saving Static Charts

```python
# High-quality chart export
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'], stock_data['Close'], linewidth=2, color='blue')
plt.title('Stock Price Analysis', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save in multiple formats
plt.savefig('stock_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('stock_analysis.pdf', bbox_inches='tight')
plt.savefig('stock_analysis.svg', bbox_inches='tight')
plt.show()
```

### 7.2 Saving Interactive Charts

```python
# Save plotly chart as HTML
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_data['Date'],
    y=stock_data['Close'],
    mode='lines',
    name='Stock Price'
))

fig.update_layout(title='Interactive Stock Chart')

# Save as HTML file
fig.write_html("interactive_chart.html")

# Save as static image
fig.write_image("chart.png", width=1200, height=600)
```

## 8. Best Practices

### 8.1 Chart Design Principles
- Use clear, descriptive titles and axis labels
- Choose appropriate color schemes for your audience
- Include legends when plotting multiple series
- Maintain consistent styling across charts
- Use appropriate chart types for your data

### 8.2 Performance Considerations
- Limit data points for interactive charts (use sampling for large datasets)
- Use appropriate figure sizes for your output medium
- Consider chart loading time for web applications
- Cache processed data to avoid recalculation

### 8.3 Financial Chart Standards
- Use green for positive movements, red for negative
- Include volume data when analyzing price movements
- Add reference lines for key levels (support/resistance)
- Use log scale for long-term price charts
- Include proper time axis formatting

## Practice Exercises

1. **Basic Charting**: Create a line chart showing the price movement of a stock over one year
2. **Candlestick Analysis**: Build a candlestick chart with volume and moving averages
3. **Comparison Charts**: Create a chart comparing the performance of 3 different stocks
4. **Technical Analysis Dashboard**: Build a multi-panel chart showing price, volume, RSI, and MACD
5. **Interactive Dashboard**: Create an interactive chart with zoom, pan, and hover features

## Key Takeaways

- Choose the right chart type for your analysis purpose
- Candlestick charts are essential for technical analysis
- Interactive charts enhance user engagement and exploration
- Volume analysis provides crucial market insights
- Professional styling improves chart readability and impact
- Export capabilities allow sharing and presentation of analysis

## Next Steps

In the next module, we'll learn about time series analysis and how to handle temporal aspects of financial data effectively.
