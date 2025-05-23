# Module 2 Assignment 3: Data Visualization for Financial Analysis

## Objective
Create professional financial charts and interactive visualizations that effectively communicate market insights and trading opportunities.

## Instructions
Build a comprehensive visualization system for financial data that includes static charts, interactive plots, and dashboard-style layouts.

## Prerequisites
- Complete Module 2.3 notes (Data Visualization)
- Familiarity with matplotlib, plotly, and seaborn
- Understanding of financial chart types

## Tasks

### Task 1: Basic Financial Charts (25 points)

**Requirements:**
1. Create a comprehensive price chart system:
   - Line charts for closing prices
   - OHLC (Open, High, Low, Close) bar charts
   - Candlestick charts with proper coloring
   - Volume bars with color coding

2. Implement professional styling:
   - Financial color scheme (green/red for bull/bear)
   - Grid lines and proper axis formatting
   - Date formatting on x-axis
   - Price formatting on y-axis

3. Create multi-timeframe charts:
   - Daily, weekly, and monthly views
   - Automatic data aggregation for each timeframe
   - Consistent styling across timeframes

**Implementation Example:**
```python
def create_candlestick_chart(df, title="Stock Price Analysis"):
    """
    Create professional candlestick chart
    Args:
        df: DataFrame with OHLC data
        title: Chart title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Candlestick chart logic here
    # Volume chart logic here
    
    return fig

def style_financial_chart(ax):
    """Apply consistent styling to financial charts"""
    # Styling logic here
    pass
```

### Task 2: Technical Analysis Visualization (25 points)

**Requirements:**
1. Create indicator overlay charts:
   - Moving averages on price charts
   - Bollinger Bands with fill areas
   - Volume indicators overlay

2. Create oscillator subplots:
   - RSI with overbought/oversold levels
   - MACD with signal line and histogram
   - Stochastic oscillator with %K and %D lines

3. Implement signal visualization:
   - Buy/sell arrows on price charts
   - Color-coded background for trend periods
   - Alert markers for indicator crossovers

**Multi-panel Layout Example:**
```python
def create_technical_analysis_chart(df):
    """
    Create comprehensive technical analysis dashboard
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), 
                            gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Panel 1: Price with Moving Averages and Bollinger Bands
    # Panel 2: Volume
    # Panel 3: RSI
    # Panel 4: MACD
    
    return fig
```

### Task 3: Interactive Charts with Plotly (25 points)

**Requirements:**
1. Create interactive candlestick charts:
   - Hover information showing OHLC values
   - Zoom and pan functionality
   - Range selector buttons (1M, 3M, 6M, 1Y)
   - Crossfilter functionality

2. Build interactive indicator charts:
   - Toggle indicators on/off
   - Adjustable parameters (e.g., MA periods)
   - Linked charts that zoom together

3. Create comparative analysis:
   - Multiple stock overlay charts
   - Normalized price comparison
   - Correlation heatmaps

**Interactive Features:**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dashboard(df):
    """
    Create interactive financial dashboard
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Volume', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Add candlestick chart
    # Add volume bars
    # Add RSI line
    # Add MACD components
    
    # Add interactivity
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        showlegend=True,
        title="Interactive Financial Analysis Dashboard"
    )
    
    return fig
```

### Task 4: Performance and Portfolio Visualization (25 points)

**Requirements:**
1. Create performance charts:
   - Cumulative return comparison
   - Drawdown analysis charts
   - Rolling metrics visualization
   - Risk-return scatter plots

2. Build portfolio analysis charts:
   - Asset allocation pie charts
   - Portfolio composition over time
   - Correlation matrix heatmaps
   - Performance attribution charts

3. Create risk visualization:
   - Value at Risk (VaR) histograms
   - Return distribution analysis
   - Volatility clustering charts
   - Risk metrics dashboard

**Performance Visualization Example:**
```python
def create_performance_dashboard(returns_df):
    """
    Create comprehensive performance analysis dashboard
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Cumulative returns
    axes[0, 0].plot(returns_df.cumsum())
    axes[0, 0].set_title('Cumulative Returns')
    
    # Drawdown analysis
    # Return distribution
    # Rolling Sharpe ratio
    # Risk-return scatter
    # Correlation heatmap
    
    return fig
```

## Bonus Tasks (Additional 10 points each)

### Bonus 1: Real-time Simulation
Create a real-time chart simulation:
- Animate price updates
- Live indicator calculations
- Dynamic alert system
- Streaming data visualization

### Bonus 2: Export and Reporting
Implement chart export functionality:
- High-resolution image export
- PDF report generation
- HTML dashboard export
- Email-ready chart formatting

### Bonus 3: Mobile-Responsive Dashboard
Create responsive layouts:
- Mobile-friendly chart sizing
- Touch-optimized interactions
- Adaptive layout for different screen sizes
- Progressive web app features

## Deliverables

Submit a Python script (`assignment3_solution.py`) containing:

1. **Chart Library**: Complete set of chart creation functions
2. **Dashboard Framework**: Modular dashboard components
3. **Interactive Examples**: Working plotly implementations
4. **Styling System**: Consistent visual theme across all charts
5. **Demo Script**: Examples using sample data

## Required Chart Types

Your solution must include:

### Static Charts (Matplotlib/Seaborn):
1. **Candlestick Chart** with volume
2. **Technical Indicator Panel** (RSI, MACD, Bollinger Bands)
3. **Performance Comparison Chart** (multiple assets)
4. **Risk Analysis Charts** (drawdown, distribution)
5. **Correlation Heatmap**

### Interactive Charts (Plotly):
1. **Interactive Candlestick** with crossfilter
2. **Multi-asset Comparison Dashboard**
3. **Performance Attribution Analysis**
4. **Risk Dashboard** with parameter controls

## Sample Output Structure

```
=== FINANCIAL VISUALIZATION SYSTEM ===

1. Basic Charts Created:
   ✓ Candlestick chart with volume
   ✓ OHLC bar chart
   ✓ Line chart with moving averages
   ✓ Professional styling applied

2. Technical Analysis Dashboard:
   ✓ 4-panel layout (Price, Volume, RSI, MACD)
   ✓ Signal markers added
   ✓ Indicator overlays implemented

3. Interactive Features:
   ✓ Plotly dashboard with zoom/pan
   ✓ Range selectors added
   ✓ Hover information configured
   ✓ Parameter controls implemented

4. Performance Analysis:
   ✓ Cumulative return charts
   ✓ Drawdown analysis
   ✓ Risk-return visualization
   ✓ Portfolio allocation charts

Charts saved to: ./charts/
Dashboard available at: ./dashboard.html
```

## Visualization Best Practices

### Color Schemes:
```python
# Financial color palette
COLORS = {
    'bullish': '#26a69a',     # Green for upward movement
    'bearish': '#ef5350',     # Red for downward movement
    'neutral': '#666666',     # Gray for neutral
    'volume': '#90a4ae',      # Light gray for volume
    'ma_short': '#1976d2',    # Blue for short MA
    'ma_long': '#d32f2f',     # Dark red for long MA
}
```

### Chart Guidelines:
1. **Consistent axis formatting** across all charts
2. **Clear legends** and labels
3. **Appropriate chart types** for data
4. **Professional color schemes**
5. **Responsive layouts**

## Testing Your Visualizations

Test with different scenarios:

```python
# Test data scenarios
scenarios = {
    'trending_up': generate_trending_data(trend='up'),
    'trending_down': generate_trending_data(trend='down'),
    'sideways': generate_sideways_data(),
    'volatile': generate_volatile_data(),
    'crisis': generate_crisis_data()
}

for scenario, data in scenarios.items():
    print(f"Testing {scenario} scenario...")
    create_all_charts(data)
```

## Performance Requirements

Your visualizations should:
- **Load quickly** (< 3 seconds for standard datasets)
- **Scale well** (handle 1000+ data points)
- **Memory efficient** (< 100MB for typical charts)
- **Cross-platform compatible**

## Evaluation Criteria

- **Visual Quality (30%)**: Professional appearance and design
- **Functionality (25%)**: All required features working correctly
- **Interactivity (20%)**: Smooth and intuitive user experience
- **Code Quality (15%)**: Clean, modular, well-documented code
- **Innovation (10%)**: Creative features and insights

## Common Chart Types Reference

### 1. Price Charts:
- Line charts for trends
- Candlestick for OHLC analysis
- Bar charts for volume
- Area charts for cumulative data

### 2. Indicator Charts:
- Line charts for moving averages
- Histogram for MACD
- Horizontal lines for RSI levels
- Band charts for Bollinger Bands

### 3. Analysis Charts:
- Scatter plots for correlation
- Heatmaps for matrices
- Bar charts for performance
- Area charts for allocation

## Resources

- Matplotlib gallery: https://matplotlib.org/stable/gallery/
- Plotly documentation: https://plotly.com/python/
- Financial charting examples: Trading platforms and financial websites
- Color theory for data visualization
- Module 2.3 notes for implementation guidance

Create charts that tell the story of the data and help traders make informed decisions!
