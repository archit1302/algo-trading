# Assignment 5: Strategy Development and Backtesting

## Objective
Develop systematic trading strategies and implement comprehensive backtesting frameworks to evaluate their performance using historical market data.

## Prerequisites
- Complete Assignments 1-4
- Complete reading: `05_strategy_development.md`
- Understanding of pandas, technical indicators, and data visualization

## Tasks

### Task 1: Simple Moving Average Crossover Strategy (30 points)
Implement a classic SMA crossover strategy:

1. **Strategy Logic**
   - Use 20-day and 50-day simple moving averages
   - Generate BUY signal when SMA20 crosses above SMA50
   - Generate SELL signal when SMA20 crosses below SMA50
   - Implement position sizing (fixed amount per trade)

2. **Signal Generation**
   - Create functions to detect crossovers
   - Generate entry and exit signals with timestamps
   - Handle overlapping signals and position management

3. **Trade Execution Simulation**
   - Simulate buying/selling at next day's open price
   - Include transaction costs (0.1% per trade)
   - Track position sizes and cash balance

### Task 2: RSI Mean Reversion Strategy (25 points)
Develop an RSI-based contrarian strategy:

1. **RSI Strategy Rules**
   - BUY when RSI(14) < 30 (oversold condition)
   - SELL when RSI(14) > 70 (overbought condition)
   - Use stop-loss at 5% below entry price
   - Take profit at 10% above entry price

2. **Advanced Signal Filtering**
   - Only trade during high volume days (> 20-day average)
   - Avoid trading near earnings announcements
   - Implement cooling-off period between trades

3. **Risk Management**
   - Maximum 3 concurrent positions
   - Position sizing based on volatility (ATR)
   - Portfolio-level stop-loss at 15%

### Task 3: Multi-Timeframe Momentum Strategy (25 points)
Create a sophisticated momentum strategy:

1. **Timeframe Analysis**
   - Daily trend using 50-day EMA slope
   - Weekly momentum using price change over 5 days
   - Monthly strength using 20-day RSI

2. **Composite Scoring System**
   - Assign scores to each timeframe signal
   - Weight: Daily (40%), Weekly (35%), Monthly (25%)
   - Trade only when composite score > threshold

3. **Dynamic Position Sizing**
   - Larger positions for higher conviction trades
   - Reduce size during high market volatility
   - Scale positions based on available capital

### Task 4: Strategy Backtesting Framework (20 points)
Build a comprehensive backtesting system:

1. **Performance Metrics Calculation**
   - Total return, annualized return, volatility
   - Sharpe ratio, maximum drawdown, win rate
   - Average trade duration, profit factor

2. **Equity Curve Generation**
   - Daily portfolio value tracking
   - Drawdown analysis with visualization
   - Rolling performance metrics

3. **Strategy Comparison**
   - Compare multiple strategies side-by-side
   - Statistical significance testing
   - Risk-adjusted performance ranking

## Implementation Requirements

### Strategy Base Class
```python
class TradingStrategy:
    def __init__(self, name, initial_capital=100000):
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.signals = []
    
    def generate_signals(self, data):
        """Override this method to implement strategy logic"""
        pass
    
    def execute_trade(self, symbol, signal, price, quantity, date):
        """Execute buy/sell orders"""
        pass
    
    def calculate_performance(self):
        """Calculate strategy performance metrics"""
        pass
```

### Required Functions
```python
def sma_crossover_strategy(data, short_window=20, long_window=50):
    """Implement SMA crossover strategy"""
    
def rsi_mean_reversion_strategy(data, rsi_period=14, oversold=30, overbought=70):
    """Implement RSI mean reversion strategy"""
    
def momentum_scoring_system(data, daily_weight=0.4, weekly_weight=0.35, monthly_weight=0.25):
    """Calculate composite momentum score"""
    
def backtest_strategy(strategy, data, transaction_cost=0.001):
    """Run comprehensive backtest"""
    
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate risk-adjusted returns"""
    
def plot_equity_curve(portfolio_values, strategy_name):
    """Visualize strategy performance"""
    
def generate_trade_report(trades):
    """Create detailed trade analysis report"""
```

## Sample Data Requirements

Use the provided historical data files:
- `SBIN_historical.csv` (2 years of daily data)
- `RELIANCE_historical.csv`
- `TCS_historical.csv`
- `INFY_historical.csv`

**Expected Data Format:**
```csv
Date,Open,High,Low,Close,Volume,Adj_Close
2023-01-01,665.20,678.50,662.10,675.25,2500000,675.25
2023-01-02,675.00,682.30,673.80,678.90,1800000,678.90
...
```

## Expected Output

### Console Output
```
=== STRATEGY BACKTESTING RESULTS ===

Strategy: SMA Crossover (20/50)
Period: 2023-01-01 to 2024-12-31
Initial Capital: ₹1,00,000

PERFORMANCE METRICS:
✓ Total Return: 23.45%
✓ Annualized Return: 11.23%
✓ Volatility: 18.67%
✓ Sharpe Ratio: 0.52
✓ Maximum Drawdown: -12.34%
✓ Win Rate: 58.33%
✓ Total Trades: 24
✓ Average Trade Duration: 18 days

TRADE SUMMARY:
- Winning Trades: 14 (Avg: +4.2%)
- Losing Trades: 10 (Avg: -2.8%)
- Best Trade: +12.5% (SBIN, Mar 2023)
- Worst Trade: -8.2% (RELIANCE, Aug 2023)

=== RSI MEAN REVERSION STRATEGY ===
✓ Total Return: 31.78%
✓ Sharpe Ratio: 0.67
✓ Maximum Drawdown: -8.91%
✓ Win Rate: 64.29%

=== MOMENTUM STRATEGY ===
✓ Total Return: 28.92%
✓ Sharpe Ratio: 0.61
✓ Maximum Drawdown: -10.45%
✓ Win Rate: 62.50%

STRATEGY RANKING (by Sharpe Ratio):
1. RSI Mean Reversion: 0.67
2. Momentum Strategy: 0.61
3. SMA Crossover: 0.52
```

### Generated Files
```
results/
├── sma_crossover_backtest.csv
├── rsi_strategy_backtest.csv
├── momentum_strategy_backtest.csv
├── strategy_comparison.html
├── equity_curves.png
├── drawdown_analysis.png
└── trade_reports/
    ├── sma_crossover_trades.csv
    ├── rsi_strategy_trades.csv
    └── momentum_strategy_trades.csv
```

## Advanced Features Implementation

### Portfolio-Level Analytics
```python
def calculate_portfolio_metrics(strategies_returns):
    """Calculate portfolio-level risk metrics"""
    
def correlation_analysis(strategy_returns):
    """Analyze strategy correlation matrix"""
    
def monte_carlo_simulation(strategy, num_simulations=1000):
    """Run Monte Carlo analysis for strategy robustness"""
```

### Risk Management
```python
def implement_stop_loss(position, current_price, stop_percentage=0.05):
    """Implement trailing stop-loss"""
    
def position_sizing_kelly(win_rate, avg_win, avg_loss):
    """Calculate optimal position size using Kelly criterion"""
    
def portfolio_heat_check(positions, max_risk_per_trade=0.02):
    """Monitor portfolio-level risk exposure"""
```

## Visualization Requirements

Create the following charts:
1. **Equity Curve Comparison** - All strategies on same plot
2. **Drawdown Analysis** - Underwater equity curves
3. **Monthly Returns Heatmap** - Strategy performance by month
4. **Rolling Sharpe Ratio** - 6-month rolling window
5. **Trade Analysis** - Win/loss distribution and duration

## Submission Guidelines
1. Create `assignment5_solution.py` with all strategy implementations
2. Include a separate `backtesting_framework.py` module
3. Generate comprehensive HTML report with all metrics
4. Create sample portfolio allocation recommendations
5. Document strategy logic and assumptions clearly

## Evaluation Criteria
- Strategy implementation correctness (25%)
- Backtesting framework robustness (25%)
- Performance metrics accuracy (20%)
- Risk management implementation (15%)
- Code quality and documentation (15%)

## Bonus Challenges (30 extra points)
1. Implement walk-forward optimization for parameter tuning
2. Add machine learning features for signal enhancement
3. Create real-time strategy monitoring dashboard
4. Implement regime detection for strategy switching
5. Add options strategies (covered calls, protective puts)
6. Develop multi-asset portfolio optimization

## Real-World Considerations
- Include realistic transaction costs and slippage
- Handle stock splits and dividend adjustments
- Account for market holidays and trading halts
- Implement proper order types (market, limit, stop)
- Consider regulatory constraints and margin requirements
