# Assignment 6: Performance Analysis and Risk Management

## Objective
Master advanced performance analysis techniques and implement comprehensive risk management frameworks for trading strategies and investment portfolios.

## Prerequisites
- Complete Assignments 1-5
- Complete reading: `06_performance_analysis.md`
- Understanding of strategy development and backtesting

## Tasks

### Task 1: Advanced Performance Metrics (25 points)
Implement sophisticated performance measurement techniques:

1. **Risk-Adjusted Returns**
   - Sharpe Ratio with rolling windows
   - Sortino Ratio (downside deviation)
   - Calmar Ratio (return/max drawdown)
   - Information Ratio vs benchmark

2. **Tail Risk Metrics**
   - Value at Risk (VaR) at 95% and 99% confidence
   - Conditional Value at Risk (CVaR)
   - Maximum Drawdown duration and recovery time
   - Ulcer Index for pain measurement

3. **Statistical Analysis**
   - Skewness and kurtosis of returns
   - Jarque-Bera normality test
   - Autocorrelation analysis
   - Volatility clustering detection

### Task 2: Benchmark Comparison and Attribution (25 points)
Analyze strategy performance relative to market benchmarks:

1. **Benchmark Analysis**
   - Alpha and Beta calculation vs NIFTY 50
   - Tracking error measurement
   - Up/down capture ratios
   - R-squared correlation analysis

2. **Performance Attribution**
   - Security selection effect
   - Market timing effect
   - Sector allocation impact
   - Style factor exposure analysis

3. **Relative Performance Metrics**
   - Active return decomposition
   - Information coefficient analysis
   - Batting average (% of periods beating benchmark)
   - Excess return consistency

### Task 3: Portfolio Risk Management (25 points)
Develop comprehensive risk management systems:

1. **Position Risk Controls**
   - Individual position size limits (max 5% per stock)
   - Sector concentration limits (max 20% per sector)
   - Correlation-based position sizing
   - Liquidity risk assessment

2. **Portfolio-Level Risk Metrics**
   - Portfolio volatility decomposition
   - Risk contribution by position
   - Marginal VaR for each holding
   - Stress testing scenarios

3. **Dynamic Risk Management**
   - Volatility-adjusted position sizing
   - Risk parity implementation
   - Market regime detection
   - Adaptive stop-loss mechanisms

### Task 4: Stress Testing and Scenario Analysis (25 points)
Implement robust scenario analysis frameworks:

1. **Historical Stress Tests**
   - 2008 Financial Crisis simulation
   - COVID-19 market crash scenario
   - Sector-specific stress events
   - Interest rate shock analysis

2. **Monte Carlo Simulations**
   - 10,000 path simulation
   - Portfolio value distributions
   - Probability of various outcomes
   - Worst-case scenario analysis

3. **Custom Scenario Building**
   - User-defined market scenarios
   - Factor shock modeling
   - Correlation breakdown scenarios
   - Liquidity crisis simulations

## Implementation Requirements

### Performance Analytics Class
```python
class PerformanceAnalyzer:
    def __init__(self, returns, benchmark_returns=None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.metrics = {}
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.06):
        """Calculate Sharpe ratio"""
        pass
    
    def calculate_sortino_ratio(self, target_return=0):
        """Calculate Sortino ratio using downside deviation"""
        pass
    
    def calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk"""
        pass
    
    def calculate_cvar(self, confidence_level=0.95):
        """Calculate Conditional Value at Risk"""
        pass
    
    def benchmark_analysis(self):
        """Comprehensive benchmark comparison"""
        pass
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        pass
```

### Risk Management Framework
```python
class RiskManager:
    def __init__(self, portfolio, risk_limits):
        self.portfolio = portfolio
        self.risk_limits = risk_limits
        self.alerts = []
    
    def check_position_limits(self):
        """Monitor individual position sizes"""
        pass
    
    def calculate_portfolio_var(self):
        """Calculate portfolio-level VaR"""
        pass
    
    def stress_test(self, scenario):
        """Run stress test scenarios"""
        pass
    
    def generate_risk_report(self):
        """Create detailed risk assessment"""
        pass
```

### Required Functions
```python
def calculate_maximum_drawdown(returns):
    """Calculate max drawdown and duration"""
    
def rolling_sharpe_ratio(returns, window=252, risk_free_rate=0.06):
    """Calculate rolling Sharpe ratios"""
    
def calculate_beta(asset_returns, market_returns):
    """Calculate beta coefficient"""
    
def performance_attribution(portfolio_returns, benchmark_returns, weights):
    """Decompose performance into components"""
    
def monte_carlo_portfolio_simulation(returns, num_simulations=10000, time_horizon=252):
    """Run Monte Carlo simulation"""
    
def stress_test_portfolio(portfolio, scenario_shocks):
    """Apply stress scenarios to portfolio"""
    
def liquidity_risk_assessment(volumes, avg_daily_volume):
    """Assess position liquidity risk"""
    
def correlation_risk_analysis(returns_matrix):
    """Analyze correlation risk in portfolio"""
```

## Sample Data Requirements

Use comprehensive datasets:
- Portfolio holdings and weights
- Historical returns for all positions
- NIFTY 50 benchmark data
- Sector classification data
- Volume and liquidity metrics

**Portfolio Data Format:**
```csv
Date,Symbol,Weight,Returns,Volume,Market_Cap
2023-01-01,RELIANCE,0.15,0.025,2500000,1200000
2023-01-01,TCS,0.12,0.018,1800000,980000
2023-01-01,SBIN,0.10,0.032,3200000,450000
...
```

**Risk Scenario Data:**
```csv
Scenario,Asset_Class,Shock_Magnitude,Probability
Market_Crash,Equity,-0.30,0.05
Interest_Rate_Rise,Bonds,-0.15,0.15
Sector_Rotation,Technology,-0.20,0.10
...
```

## Expected Output

### Comprehensive Performance Report
```
=== PORTFOLIO PERFORMANCE ANALYSIS ===
Analysis Period: 2023-01-01 to 2024-12-31
Portfolio Value: ₹15,75,000 (Initial: ₹10,00,000)

RETURN METRICS:
✓ Total Return: 57.50%
✓ Annualized Return: 24.85%
✓ Volatility (Annualized): 16.23%
✓ Sharpe Ratio: 1.16
✓ Sortino Ratio: 1.68
✓ Calmar Ratio: 2.01

RISK METRICS:
✓ Maximum Drawdown: -12.36%
✓ Drawdown Duration: 45 days
✓ VaR (95%): -2.89%
✓ CVaR (95%): -4.23%
✓ Ulcer Index: 4.56

BENCHMARK COMPARISON (vs NIFTY 50):
✓ Alpha: 8.45% (annualized)
✓ Beta: 0.92
✓ R-squared: 0.78
✓ Tracking Error: 6.78%
✓ Information Ratio: 1.25
✓ Up Capture: 98.5%
✓ Down Capture: 87.2%

STATISTICAL PROPERTIES:
✓ Skewness: 0.23 (slight positive skew)
✓ Kurtosis: 2.89 (normal distribution)
✓ Jarque-Bera p-value: 0.12 (normal at 5%)
✓ Autocorrelation (lag-1): 0.08

=== PERFORMANCE ATTRIBUTION ===
✓ Security Selection: +4.2%
✓ Market Timing: +1.8%
✓ Sector Allocation: +2.1%
✓ Interaction Effect: +0.3%
✓ Total Active Return: +8.4%

=== RISK ANALYSIS ===
POSITION RISK CHECKS:
✓ All positions within 5% limit
✓ Technology sector: 18% (within 20% limit)
✓ Financial sector: 22% (BREACH - Limit: 20%)
⚠️  ALERT: Financial sector overweight

PORTFOLIO RISK DECOMPOSITION:
✓ Systematic Risk: 68%
✓ Idiosyncratic Risk: 32%
✓ Top 3 Risk Contributors:
   1. RELIANCE: 22% of portfolio risk
   2. TCS: 18% of portfolio risk
   3. SBIN: 15% of portfolio risk

=== STRESS TEST RESULTS ===
SCENARIO: 2008-Style Market Crash (-40%)
✓ Portfolio Impact: -32.5%
✓ Expected Recovery Time: 18 months
✓ Worst Position: SBIN (-45%)
✓ Best Position: TCS (-22%)

SCENARIO: Interest Rate Shock (+3%)
✓ Portfolio Impact: -8.2%
✓ Banking stocks most affected
✓ Technology stocks resilient

MONTE CARLO SIMULATION (10,000 paths):
✓ 95% Confidence Interval: [-15.2%, +78.9%]
✓ Probability of Loss: 18.3%
✓ Probability of >50% Gain: 42.7%
✓ Expected Value: +28.4%
```

### Risk Management Dashboard Output
```
=== DAILY RISK MONITORING ===
Date: 2024-12-31

POSITION LIMITS STATUS:
✅ Individual positions: All within limits
⚠️  Sector concentration: Financial sector at 22% (limit: 20%)
✅ Correlation risk: Within acceptable range
✅ Liquidity risk: All positions liquid

PORTFOLIO METRICS:
✓ Daily VaR (95%): ₹28,900
✓ Portfolio volatility: 16.2% (annual)
✓ Beta vs NIFTY: 0.92
✓ Current drawdown: -3.2%

ALERTS:
🔴 URGENT: Financial sector overweight by 2%
🟡 WATCH: Technology sector correlation increasing
🟢 NORMAL: All other metrics within range

RECOMMENDED ACTIONS:
1. Reduce financial sector exposure by ₹3,00,000
2. Consider adding defensive stocks
3. Monitor technology sector correlation
```

## Advanced Analytics Implementation

### Factor Risk Analysis
```python
def factor_risk_decomposition(returns, factor_loadings):
    """Decompose portfolio risk by factors"""
    
def style_factor_analysis(returns, market_cap, book_to_market, momentum):
    """Analyze exposure to style factors"""
    
def sector_risk_contribution(weights, sector_classification, covariance_matrix):
    """Calculate sector risk contributions"""
```

### Dynamic Risk Models
```python
def garch_volatility_forecast(returns, horizon=22):
    """Forecast volatility using GARCH model"""
    
def regime_switching_model(returns):
    """Detect market regime changes"""
    
def adaptive_var_model(returns, confidence_level=0.95):
    """Calculate time-varying VaR"""
```

## Visualization Requirements

Create professional risk dashboards:
1. **Risk Metrics Dashboard** - Key metrics in traffic light format
2. **Drawdown Analysis** - Underwater equity curve with recovery periods
3. **Rolling Risk Metrics** - Sharpe, volatility, and VaR over time
4. **Correlation Heatmap** - Portfolio correlation matrix
5. **Scenario Analysis Charts** - Stress test impact visualization
6. **Monte Carlo Results** - Distribution of potential outcomes
7. **Risk Attribution** - Risk contribution by position/sector

## Real-Time Monitoring Features

### Alert System
```python
class RiskAlertSystem:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.alerts = []
    
    def check_var_breach(self, current_var):
        """Monitor VaR threshold breaches"""
        
    def position_limit_alert(self, current_weights):
        """Alert on position limit breaches"""
        
    def correlation_spike_alert(self, correlation_matrix):
        """Detect unusual correlation spikes"""
        
    def send_notifications(self):
        """Send risk alerts via email/SMS"""
```

## Submission Guidelines
1. Create `assignment6_solution.py` with all analytics implementations
2. Include separate `risk_management.py` module
3. Generate interactive HTML dashboard
4. Provide detailed methodology documentation
5. Include recommendations for portfolio improvements

## Evaluation Criteria
- Performance metrics accuracy (25%)
- Risk framework completeness (25%)
- Stress testing robustness (20%)
- Dashboard quality and usability (15%)
- Code efficiency and documentation (15%)

## Bonus Challenges (40 extra points)
1. Implement machine learning for risk prediction
2. Create real-time risk monitoring system
3. Develop ESG risk integration framework
4. Build liquidity risk stress testing
5. Implement dynamic hedging strategies
6. Create regulatory capital calculation (Basel III)
7. Develop custom risk factors identification
8. Build portfolio optimization with risk constraints

## Professional Standards
- Follow institutional risk management practices
- Implement proper error handling and validation
- Use industry-standard risk metrics and benchmarks
- Provide clear documentation for all methodologies
- Create audit trail for all risk calculations

## Integration with Previous Modules
This assignment integrates:
- Module 1: Python programming foundations
- Assignment 2: Technical indicator calculations
- Assignment 3: Data visualization techniques
- Assignment 4: Time series analysis methods
- Assignment 5: Strategy backtesting frameworks

The complete framework should provide institutional-grade risk management capabilities suitable for professional trading and investment management.
