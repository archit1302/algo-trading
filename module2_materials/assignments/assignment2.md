# Module 2 Assignment 2: Technical Indicators Implementation

## Objective
Implement and analyze various technical indicators used in financial trading, understanding their mathematical foundations and practical applications.

## Instructions
Create a comprehensive technical analysis system that calculates multiple indicators and generates trading insights.

## Prerequisites
- Complete Module 2.2 notes (Technical Indicators)
- Basic understanding of financial markets
- Python, pandas, numpy, matplotlib

## Tasks

### Task 1: Moving Averages and Trend Analysis (25 points)

**Requirements:**
1. Implement from scratch (without using built-in functions):
   - Simple Moving Average (SMA) for 10, 20, and 50 periods
   - Exponential Moving Average (EMA) for 12 and 26 periods
   - Weighted Moving Average (WMA) for 20 periods

2. Create moving average crossover signals:
   - Golden Cross: 50-day SMA crosses above 200-day SMA
   - Death Cross: 50-day SMA crosses below 200-day SMA
   - Short-term signals: 10-day SMA vs 20-day SMA

3. Analyze trend strength:
   - Calculate the slope of moving averages
   - Determine trend direction and momentum
   - Identify trend reversals

**Implementation Example:**
```python
def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    # Implement without using pandas.rolling()
    pass

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average"""
    # Use alpha = 2/(window+1)
    pass

def detect_crossover(short_ma, long_ma):
    """Detect bullish/bearish crossovers"""
    pass
```

### Task 2: Momentum Oscillators (25 points)

**Requirements:**
1. Implement RSI (Relative Strength Index):
   - Calculate with 14-period default
   - Identify overbought (>70) and oversold (<30) conditions
   - Generate buy/sell signals based on RSI levels

2. Implement Stochastic Oscillator:
   - Calculate %K and %D lines
   - Identify oversold/overbought conditions
   - Generate crossover signals

3. Implement MACD (Moving Average Convergence Divergence):
   - Calculate MACD line (12-day EMA - 26-day EMA)
   - Calculate Signal line (9-day EMA of MACD)
   - Calculate MACD Histogram
   - Generate bullish/bearish divergence signals

**Key Formulas:**
```python
# RSI Calculation
def calculate_rsi(prices, window=14):
    """
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    pass

# Stochastic Calculation
def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = 3-period SMA of %K
    """
    pass
```

### Task 3: Volatility and Bands (25 points)

**Requirements:**
1. Implement Bollinger Bands:
   - 20-period SMA as middle band
   - Upper band: SMA + (2 * standard deviation)
   - Lower band: SMA - (2 * standard deviation)
   - Calculate %B and Band Width

2. Implement Average True Range (ATR):
   - Calculate True Range for each period
   - Calculate 14-period ATR
   - Use ATR for volatility-based position sizing

3. Create volatility analysis:
   - Identify periods of high/low volatility
   - Bollinger Band squeeze detection
   - Volatility breakout signals

**Bollinger Band Signals:**
```python
def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands and related indicators
    """
    pass

def bollinger_signals(prices, upper_band, lower_band):
    """
    Generate trading signals:
    - Buy when price touches lower band
    - Sell when price touches upper band
    """
    pass
```

### Task 4: Volume-Based Indicators (25 points)

**Requirements:**
1. Implement On-Balance Volume (OBV):
   - Calculate cumulative volume flow
   - Identify bullish/bearish divergences
   - Generate trend confirmation signals

2. Implement Volume Weighted Average Price (VWAP):
   - Calculate intraday VWAP
   - Identify when price is above/below VWAP
   - Use as support/resistance level

3. Create volume analysis:
   - Volume spike detection
   - Volume trend analysis
   - Price-volume relationship analysis

**Volume Indicators:**
```python
def calculate_obv(prices, volume):
    """
    OBV = Previous OBV + Volume (if close > previous close)
    OBV = Previous OBV - Volume (if close < previous close)
    """
    pass

def calculate_vwap(prices, volume):
    """
    VWAP = Cumulative(Price × Volume) / Cumulative Volume
    """
    pass
```

## Bonus Tasks (Additional 10 points each)

### Bonus 1: Custom Composite Indicator
Create your own indicator combining multiple signals:
- Weight different indicators based on market conditions
- Create a composite score (0-100)
- Generate buy/sell signals based on composite score

### Bonus 2: Indicator Performance Analysis
Backtest each indicator's performance:
- Calculate hit rate for each signal type
- Measure average profit/loss per signal
- Compare indicator effectiveness across different market conditions

### Bonus 3: Advanced Patterns
Implement pattern recognition:
- Double top/bottom patterns
- Head and shoulders patterns
- Support and resistance level detection

## Deliverables

Submit a Python script (`assignment2_solution.py`) that includes:

1. **Indicator Library**: All technical indicator implementations
2. **Signal Generation**: Buy/sell signal logic for each indicator
3. **Analysis Framework**: Tools to analyze indicator performance
4. **Visualization**: Charts showing indicators and signals
5. **Documentation**: Comprehensive comments and explanations

## Expected Output

Your script should generate analysis similar to:

```
=== TECHNICAL INDICATORS ANALYSIS ===

1. Moving Average Analysis:
   Current Trend: Bullish (10 > 20 > 50 SMA)
   Golden Cross: Detected on 2023-08-15
   Trend Strength: Strong (slope = 0.25)

2. Momentum Analysis:
   RSI: 65.2 (Neutral)
   Stochastic: 78.5 (Overbought)
   MACD: Bullish crossover on 2023-10-03

3. Volatility Analysis:
   Bollinger %B: 0.75 (Upper half)
   ATR: 2.15 (Moderate volatility)
   Band Squeeze: No

4. Volume Analysis:
   OBV Trend: Bullish divergence
   Price vs VWAP: Above (Bullish)
   Volume Spike: Detected on 2023-11-02

5. Trading Signals:
   Current Recommendation: BUY
   Signal Strength: 7/10
   Risk Level: Medium
```

## Visualization Requirements

Create the following charts:

1. **Price with Moving Averages**: Show price action with all MAs
2. **Oscillator Panel**: RSI, Stochastic, and MACD in subplots
3. **Bollinger Bands Chart**: Price with bands and %B
4. **Volume Analysis**: Price with volume bars and OBV

Example visualization structure:
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# ax1: Price with Moving Averages
# ax2: RSI and Stochastic
# ax3: Bollinger Bands
# ax4: Volume and OBV
```

## Evaluation Criteria

- **Mathematical Accuracy (35%)**: Correct implementation of formulas
- **Signal Quality (25%)**: Logical and well-timed signals
- **Code Quality (20%)**: Clean, efficient, well-documented code
- **Analysis Depth (20%)**: Insightful interpretation of results

## Testing Your Implementation

Use the following test cases to verify your indicators:

```python
# Test data: Known values for verification
test_prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89]
test_volume = [1000, 1200, 800, 1500, 2000, 1800, 1300, 2200, 1600, 1400]

# Expected RSI (14-period) for longer dataset should be reasonable (0-100)
# Expected SMA should be smooth version of price data
# Expected Bollinger Bands should contain ~95% of price action
```

## Common Pitfalls to Avoid

1. **Look-ahead bias**: Don't use future data in calculations
2. **Division by zero**: Handle edge cases in calculations
3. **Insufficient data**: Ensure enough periods for indicator calculation
4. **Parameter sensitivity**: Test with different parameter values
5. **Signal lag**: Understand that indicators are lagging by nature

## Resources

- Technical Analysis Theory: "Technical Analysis of the Financial Markets" by John Murphy
- Implementation Reference: TA-Lib documentation
- Module 2.2 notes for detailed formulas and explanations
- Financial data patterns and market behavior studies

Good luck building your technical analysis toolkit!
